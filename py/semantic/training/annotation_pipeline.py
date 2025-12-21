"""
Annotation pipeline for NFL NER and sentiment fine-tuning.

Converts structured database records (injuries, games, players) into
annotated training data for LLM fine-tuning.

Formats supported:
- JSON Lines (for Ollama/Mistral fine-tuning)
- spaCy DocBin (for transformer fine-tuning)
- CSV (for human review)

Usage:
    pipeline = AnnotationPipeline()
    dataset = pipeline.create_ner_dataset(seasons=[2022, 2023, 2024])
    pipeline.export_jsonl(dataset, "data/ner_training.jsonl")
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


@dataclass
class NERAnnotation:
    """A single NER annotation."""

    text: str
    entities: list[tuple[int, int, str]]  # (start, end, label)
    metadata: dict = field(default_factory=dict)

    def to_spacy_format(self) -> tuple[str, dict]:
        """Convert to spaCy training format."""
        return (self.text, {"entities": self.entities})

    def to_jsonl(self) -> dict:
        """Convert to JSON Lines format for LLM fine-tuning."""
        # Format entities as structured output
        entity_list = []
        for start, end, label in self.entities:
            entity_list.append({
                "text": self.text[start:end],
                "type": label,
                "start": start,
                "end": end,
            })

        return {
            "text": self.text,
            "entities": entity_list,
            "metadata": self.metadata,
        }


@dataclass
class SentimentAnnotation:
    """A single sentiment annotation."""

    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)

    def to_jsonl(self) -> dict:
        """Convert to JSON Lines format."""
        return {
            "text": self.text,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


# NER entity types
NER_LABELS = {
    "PLAYER": "NFL player name",
    "TEAM": "NFL team name or abbreviation",
    "INJURY": "Type of injury (ankle, knee, concussion, etc.)",
    "STATUS": "Injury status (questionable, doubtful, out, probable)",
    "POSITION": "Player position (QB, WR, RB, etc.)",
    "DATE": "Date or time reference",
    "GAME": "Game reference (Week X, playoffs, etc.)",
    "COACH": "Coach name",
}

# Templates for generating natural language from structured data
INJURY_TEMPLATES = [
    "{player} ({position}, {team}) is listed as {status} with a {injury} injury.",
    "{team}'s {player} is {status} for Week {week} due to a {injury} issue.",
    "{player} ({team}) has been ruled {status} after missing practice with a {injury}.",
    "Injury update: {player} ({position}) is {status} with {injury}. {practice_note}",
    "{team} {position} {player} is dealing with a {injury} injury and is listed as {status}.",
    "The {team} have listed {player} as {status} on the injury report ({injury}).",
    "{player} practiced {practice_status} on {day} and is {status} for Sunday.",
    "Per the injury report, {player} ({team}) - {injury} - {status}.",
]

PRACTICE_NOTES = {
    "Did Not Participate In Practice": "He did not practice this week.",
    "Limited Participation in Practice": "He was limited in practice.",
    "Full Participation in Practice": "He was a full participant in practice.",
}

DAYS = ["Wednesday", "Thursday", "Friday"]


class AnnotationPipeline:
    """
    Pipeline for creating NER and sentiment training datasets.

    Uses historical NFL data to generate annotated examples for fine-tuning.
    """

    def __init__(self, db_config: dict | None = None):
        """
        Initialize annotation pipeline.

        Args:
            db_config: Database configuration
        """
        self.db_config = db_config or {
            "host": "localhost",
            "port": 5544,
            "dbname": "devdb01",
            "user": "dro",
            "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
        }

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.db_config)

    def create_ner_dataset(
        self,
        seasons: list[int] | None = None,
        max_samples: int = 5000,
        include_validation: bool = True,
    ) -> dict[str, list[NERAnnotation]]:
        """
        Create NER training dataset from injury reports.

        Args:
            seasons: Seasons to include (default: 2022-2024)
            max_samples: Maximum samples to generate
            include_validation: Whether to create train/val split

        Returns:
            Dict with 'train' and optionally 'validation' lists
        """
        if seasons is None:
            # Include 2025 for NER training - no leakage concern
            # (NER task is orthogonal to game outcome prediction)
            seasons = [2022, 2023, 2024, 2025]

        logger.info(f"Creating NER dataset from seasons {seasons}")

        # Fetch injury data
        injuries = self._fetch_injuries(seasons, max_samples * 2)
        logger.info(f"Fetched {len(injuries)} injury records")

        # Generate annotated examples
        annotations = []
        for injury in injuries:
            try:
                ann = self._create_ner_annotation(injury)
                if ann:
                    annotations.append(ann)
            except Exception as e:
                logger.debug(f"Failed to annotate record: {e}")
                continue

        logger.info(f"Generated {len(annotations)} NER annotations")

        # Shuffle and limit
        random.shuffle(annotations)
        annotations = annotations[:max_samples]

        # Split train/validation
        if include_validation:
            split_idx = int(len(annotations) * 0.9)
            return {
                "train": annotations[:split_idx],
                "validation": annotations[split_idx:],
            }
        else:
            return {"train": annotations}

    def _fetch_injuries(self, seasons: list[int], limit: int) -> list[dict]:
        """Fetch injury records from database."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("""
                    SELECT full_name, position, team,
                           report_primary_injury, report_status,
                           practice_status, season, week
                    FROM injuries
                    WHERE season = ANY(%s)
                      AND report_status IS NOT NULL
                      AND full_name IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT %s
                """, (seasons, limit))
                return [dict(row) for row in cur.fetchall()]

    def _create_ner_annotation(self, injury: dict) -> NERAnnotation | None:
        """Create NER annotation from injury record."""
        player = injury.get("full_name", "")
        position = injury.get("position", "")
        team = injury.get("team", "")
        injury_type = injury.get("report_primary_injury", "")
        status = injury.get("report_status", "")
        practice = injury.get("practice_status", "")
        week = injury.get("week", "")

        if not all([player, team, injury_type, status]):
            return None

        # Choose random template
        template = random.choice(INJURY_TEMPLATES)

        # Build practice note
        practice_note = PRACTICE_NOTES.get(practice, "")
        day = random.choice(DAYS)

        # Generate text
        text = template.format(
            player=player,
            position=position,
            team=team,
            injury=injury_type.lower(),
            status=status.lower(),
            week=week,
            practice_note=practice_note,
            practice_status="limited" if "Limited" in (practice or "") else "fully",
            day=day,
        )

        # Find entity spans
        entities = []

        # Player name
        player_start = text.find(player)
        if player_start >= 0:
            entities.append((player_start, player_start + len(player), "PLAYER"))

        # Team
        team_start = text.find(team)
        if team_start >= 0:
            entities.append((team_start, team_start + len(team), "TEAM"))

        # Injury type
        injury_lower = injury_type.lower()
        injury_start = text.lower().find(injury_lower)
        if injury_start >= 0:
            entities.append((injury_start, injury_start + len(injury_lower), "INJURY"))

        # Status
        status_lower = status.lower()
        status_start = text.lower().find(status_lower)
        if status_start >= 0:
            entities.append((status_start, status_start + len(status_lower), "STATUS"))

        # Position
        if position and position in text:
            pos_start = text.find(position)
            if pos_start >= 0:
                entities.append((pos_start, pos_start + len(position), "POSITION"))

        # Sort by start position
        entities.sort(key=lambda x: x[0])

        return NERAnnotation(
            text=text,
            entities=entities,
            metadata={
                "season": injury.get("season"),
                "week": injury.get("week"),
                "source": "injury_report",
            },
        )

    def create_sentiment_dataset(
        self,
        seasons: list[int] | None = None,
        max_samples: int = 2000,
    ) -> dict[str, list[SentimentAnnotation]]:
        """
        Create sentiment training dataset from game outcomes.

        Uses game results to generate positive/negative sentiment examples.

        Args:
            seasons: Seasons to include
            max_samples: Maximum samples

        Returns:
            Dict with train/validation splits
        """
        if seasons is None:
            # Include 2025 for sentiment training - no leakage concern
            seasons = [2022, 2023, 2024, 2025]

        logger.info(f"Creating sentiment dataset from seasons {seasons}")

        # Fetch game data
        games = self._fetch_games(seasons, max_samples * 2)

        annotations = []
        for game in games:
            try:
                anns = self._create_sentiment_annotations(game)
                annotations.extend(anns)
            except Exception as e:
                logger.debug(f"Failed to create sentiment annotation: {e}")

        random.shuffle(annotations)
        annotations = annotations[:max_samples]

        split_idx = int(len(annotations) * 0.9)
        return {
            "train": annotations[:split_idx],
            "validation": annotations[split_idx:],
        }

    def _fetch_games(self, seasons: list[int], limit: int) -> list[dict]:
        """Fetch game records from database."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("""
                    SELECT game_id, season, week,
                           home_team, away_team,
                           home_score, away_score,
                           spread_close
                    FROM games
                    WHERE season = ANY(%s)
                      AND home_score IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT %s
                """, (seasons, limit))
                return [dict(row) for row in cur.fetchall()]

    def _create_sentiment_annotations(self, game: dict) -> list[SentimentAnnotation]:
        """Create sentiment annotations from game outcome."""
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        home_score = game.get("home_score", 0)
        away_score = game.get("away_score", 0)
        week = game.get("week", 0)

        annotations = []
        margin = home_score - away_score

        # Positive template for winner
        if margin > 0:
            winner, loser = home, away
            win_margin = margin
        else:
            winner, loser = away, home
            win_margin = -margin

        # Generate positive sentiment
        positive_templates = [
            f"The {winner} dominated with a {win_margin}-point victory over the {loser}.",
            f"{winner} looking strong after Week {week} win against {loser}.",
            f"Impressive performance by {winner}, defense held {loser} to {min(home_score, away_score)} points.",
        ]
        annotations.append(SentimentAnnotation(
            text=random.choice(positive_templates),
            sentiment="positive",
            confidence=0.9 if win_margin > 14 else 0.7,
            reasoning=f"Decisive {win_margin}-point victory indicates strong performance",
            metadata={"game_id": game.get("game_id"), "team": winner},
        ))

        # Generate negative sentiment for loser
        negative_templates = [
            f"Disappointing loss for {loser}, falling to {winner} by {win_margin} points.",
            f"{loser} struggles continue after Week {week} defeat.",
            f"Concerning performance by {loser}, offense managed only {min(home_score, away_score)} points.",
        ]
        annotations.append(SentimentAnnotation(
            text=random.choice(negative_templates),
            sentiment="negative",
            confidence=0.9 if win_margin > 14 else 0.7,
            reasoning=f"Loss by {win_margin} points suggests team struggles",
            metadata={"game_id": game.get("game_id"), "team": loser},
        ))

        return annotations

    def export_jsonl(
        self,
        dataset: dict[str, list],
        output_path: str,
        format_type: str = "ner",
    ) -> None:
        """
        Export dataset to JSON Lines format for fine-tuning.

        Args:
            dataset: Dict with train/validation splits
            output_path: Output file path
            format_type: 'ner' or 'sentiment'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        for split_name, annotations in dataset.items():
            split_path = output_path.parent / f"{output_path.stem}_{split_name}.jsonl"

            with open(split_path, "w") as f:
                for ann in annotations:
                    if format_type == "ner":
                        # Format for instruction fine-tuning
                        prompt = f"Extract NFL entities from the following text:\n\n{ann.text}"
                        response = json.dumps({"entities": ann.to_jsonl()["entities"]})
                        record = {
                            "prompt": prompt,
                            "response": response,
                            "metadata": ann.metadata,
                        }
                    else:  # sentiment
                        prompt = f"Analyze the sentiment of this NFL-related text:\n\n{ann.text}"
                        response = json.dumps({
                            "sentiment": ann.sentiment,
                            "confidence": ann.confidence,
                            "reasoning": ann.reasoning,
                        })
                        record = {
                            "prompt": prompt,
                            "response": response,
                            "metadata": ann.metadata,
                        }

                    f.write(json.dumps(record) + "\n")

            logger.info(f"Exported {len(annotations)} samples to {split_path}")

    def export_ollama_format(
        self,
        dataset: dict[str, list],
        output_path: str,
        model_name: str = "nfl-ner",
    ) -> None:
        """
        Export dataset in Ollama fine-tuning format.

        Creates a Modelfile and training data for Ollama create.

        Args:
            dataset: Dataset to export
            output_path: Output directory
            model_name: Name for the fine-tuned model
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export training data
        train_path = output_dir / "train.jsonl"
        with open(train_path, "w") as f:
            for ann in dataset.get("train", []):
                # Ollama format: system, user, assistant
                record = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an NFL entity extraction expert. Extract entities as JSON."
                        },
                        {
                            "role": "user",
                            "content": f"Extract entities from: {ann.text}"
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps({"entities": ann.to_jsonl()["entities"]})
                        },
                    ]
                }
                f.write(json.dumps(record) + "\n")

        # Create Modelfile
        modelfile_content = f"""FROM qwen3:8b
PARAMETER temperature 0.1
PARAMETER num_predict 1024

SYSTEM You are an NFL entity extraction expert. Extract entities from text in JSON format.
Categories: PLAYER, TEAM, INJURY, STATUS, POSITION, DATE, GAME, COACH
Output format: {{"entities": [{{"text": "...", "type": "...", "start": N, "end": N}}]}}
"""

        modelfile_path = output_dir / "Modelfile"
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        logger.info(f"Exported Ollama format to {output_dir}")
        logger.info(f"To fine-tune: ollama create {model_name} -f {modelfile_path}")

    def get_dataset_stats(self, dataset: dict[str, list]) -> dict[str, Any]:
        """Get statistics about the dataset."""
        stats = {}

        for split, annotations in dataset.items():
            if not annotations:
                continue

            # Entity counts
            entity_counts = {}
            for ann in annotations:
                if hasattr(ann, "entities"):
                    for _, _, label in ann.entities:
                        entity_counts[label] = entity_counts.get(label, 0) + 1

            # Sentiment distribution
            sentiment_counts = {}
            for ann in annotations:
                if hasattr(ann, "sentiment"):
                    sentiment_counts[ann.sentiment] = sentiment_counts.get(ann.sentiment, 0) + 1

            stats[split] = {
                "count": len(annotations),
                "entity_counts": entity_counts,
                "sentiment_counts": sentiment_counts,
            }

        return stats


def test_annotation_pipeline():
    """Test the annotation pipeline."""
    print("=== Testing Annotation Pipeline ===")

    pipeline = AnnotationPipeline()

    # Create NER dataset
    print("\nCreating NER dataset (100 samples)...")
    ner_dataset = pipeline.create_ner_dataset(
        seasons=[2023, 2024],
        max_samples=100,
    )

    stats = pipeline.get_dataset_stats(ner_dataset)
    print(f"Train samples: {stats['train']['count']}")
    print(f"Validation samples: {stats['validation']['count']}")
    print(f"Entity distribution: {stats['train']['entity_counts']}")

    # Show sample
    print("\nSample NER annotation:")
    sample = ner_dataset["train"][0]
    print(f"  Text: {sample.text}")
    print(f"  Entities: {sample.to_jsonl()['entities']}")

    # Create sentiment dataset
    print("\nCreating sentiment dataset (100 samples)...")
    sent_dataset = pipeline.create_sentiment_dataset(
        seasons=[2023, 2024],
        max_samples=100,
    )

    stats = pipeline.get_dataset_stats(sent_dataset)
    print(f"Sentiment distribution: {stats['train']['sentiment_counts']}")

    print("\nSample sentiment annotation:")
    sample = sent_dataset["train"][0]
    print(f"  Text: {sample.text}")
    print(f"  Sentiment: {sample.sentiment} ({sample.confidence:.1%})")


if __name__ == "__main__":
    test_annotation_pipeline()
