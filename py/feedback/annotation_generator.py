"""
Annotation Generator for LLM Fine-tuning.

Converts game outcomes into training data for semantic LLM refinement.
Generates sentiment and NER annotations based on prediction accuracy.

Usage:
    generator = AnnotationGenerator()
    annotations = generator.generate_sentiment_annotations(season=2025, week=16)
    generator.export_training_batch(Path("data/semantic_training"))
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class AnnotationStats:
    """Statistics from annotation generation."""
    season: int
    week: int
    sentiment_annotations: int = 0
    ner_annotations: int = 0
    games_processed: int = 0
    news_matched: int = 0
    injuries_matched: int = 0
    errors: list[str] = field(default_factory=list)


class AnnotationGenerator:
    """
    Generate training annotations from game outcomes.

    Creates two types of annotations:
    1. Sentiment: News text + outcome (win/loss, cover/not)
    2. NER: Injury news with player/injury entities

    Output format: JSONL for Ollama/MLX fine-tuning
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5544,
        database: str = "devdb01",
        user: str = "dro",
        password: str = "sicillionbillions",
    ):
        """Initialize with database connection parameters."""
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.conn_params)

    def get_games_with_outcomes(
        self,
        season: int,
        week: int
    ) -> list[dict]:
        """
        Get completed games for annotation generation.

        Returns games with predictions and actual outcomes.
        """
        query = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            g.home_score - g.away_score AS actual_spread,
            g.home_score + g.away_score AS actual_total,
            g.kickoff,
            CASE WHEN g.home_score > g.away_score THEN TRUE
                 WHEN g.home_score < g.away_score THEN FALSE
                 ELSE NULL END AS home_won
        FROM games g
        WHERE g.season = %s AND g.week = %s
          AND g.home_score IS NOT NULL
        ORDER BY g.game_id
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (season, week))
                return [dict(row) for row in cur.fetchall()]

    def get_pregame_news(
        self,
        team: str,
        before: datetime,
        hours: int = 48,
        limit: int = 10
    ) -> list[dict]:
        """
        Get news articles published before game time.

        For temporal safety, only uses news BEFORE kickoff.
        """
        query = """
        SELECT
            id,
            source,
            title,
            description,
            published_at,
            category,
            teams,
            players
        FROM semantic.news_snapshots
        WHERE %s = ANY(teams)
          AND published_at < %s
          AND published_at >= %s
        ORDER BY published_at DESC
        LIMIT %s
        """

        since = before - timedelta(hours=hours)

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (team, before, since, limit))
                return [dict(row) for row in cur.fetchall()]

    def get_pregame_injuries(
        self,
        team: str,
        before: datetime
    ) -> list[dict]:
        """
        Get injury reports before game time.

        Returns active injuries at prediction time.
        """
        query = """
        SELECT
            id,
            athlete_name,
            team,
            position,
            status,
            injury_type,
            injury_location,
            updated_at
        FROM semantic.injury_snapshots
        WHERE team = %s
          AND updated_at < %s
        ORDER BY updated_at DESC
        LIMIT 20
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (team, before))
                return [dict(row) for row in cur.fetchall()]

    def generate_sentiment_annotations(
        self,
        season: int,
        week: int
    ) -> list[dict]:
        """
        Generate sentiment training annotations from game outcomes.

        For each game:
        1. Get pre-game news for both teams
        2. Label with outcome (win/loss, cover/not)
        3. Create training examples

        Returns list of annotation dicts in JSONL format.
        """
        annotations = []
        games = self.get_games_with_outcomes(season, week)

        for game in games:
            kickoff = game.get("kickoff") or datetime.now()

            # Get news for home team
            home_news = self.get_pregame_news(
                game["home_team"], kickoff, hours=48
            )

            # Get news for away team
            away_news = self.get_pregame_news(
                game["away_team"], kickoff, hours=48
            )

            # Generate annotations for each news item
            for news in home_news:
                annotation = self._create_sentiment_annotation(
                    news=news,
                    team=game["home_team"],
                    opponent=game["away_team"],
                    is_home=True,
                    team_won=game["home_won"],
                    spread=game["actual_spread"],
                    game_id=game["game_id"]
                )
                if annotation:
                    annotations.append(annotation)

            for news in away_news:
                annotation = self._create_sentiment_annotation(
                    news=news,
                    team=game["away_team"],
                    opponent=game["home_team"],
                    is_home=False,
                    team_won=not game["home_won"] if game["home_won"] is not None else None,
                    spread=-game["actual_spread"],  # Invert for away
                    game_id=game["game_id"]
                )
                if annotation:
                    annotations.append(annotation)

        return annotations

    def _create_sentiment_annotation(
        self,
        news: dict,
        team: str,
        opponent: str,
        is_home: bool,
        team_won: bool | None,
        spread: float,
        game_id: str
    ) -> dict | None:
        """Create a single sentiment training annotation."""
        if not news.get("title"):
            return None

        # Combine title and description
        text = news["title"]
        if news.get("description"):
            text += " " + news["description"]

        # Determine sentiment label based on outcome
        # Positive news + team won = correct positive sentiment
        # Negative news + team lost = correct negative sentiment
        if team_won is None:
            outcome_label = "unknown"
        elif team_won:
            outcome_label = "positive_outcome"
        else:
            outcome_label = "negative_outcome"

        return {
            "id": f"sent_{news['id']}_{game_id}",
            "text": text,
            "team": team,
            "opponent": opponent,
            "is_home": is_home,
            "category": news.get("category", "general"),
            "outcome_label": outcome_label,
            "spread": spread,
            "game_id": game_id,
            "published_at": news["published_at"].isoformat() if news.get("published_at") else None,
            # Format for Ollama fine-tuning
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze the sentiment of this NFL news about {team}: {text}"
                },
                {
                    "role": "assistant",
                    "content": f"Sentiment analysis for {team}:\n"
                              f"- Category: {news.get('category', 'general')}\n"
                              f"- Outcome context: {'won' if team_won else 'lost' if team_won is not None else 'unknown'} "
                              f"by {abs(spread):.1f} points\n"
                              f"- Predictive signal: {'positive' if team_won else 'negative' if team_won is not None else 'neutral'}"
                }
            ]
        }

    def generate_ner_annotations(
        self,
        season: int,
        week: int
    ) -> list[dict]:
        """
        Generate NER training annotations from injury news.

        For injury news that proved predictive:
        1. Get injury-related news
        2. Extract player/injury mentions
        3. Create NER training examples

        Returns list of annotation dicts.
        """
        annotations = []
        games = self.get_games_with_outcomes(season, week)

        for game in games:
            kickoff = game.get("kickoff") or datetime.now()

            # Get injury news and actual injuries for both teams
            for team in [game["home_team"], game["away_team"]]:
                news = self.get_pregame_news(team, kickoff, hours=48)
                injuries = self.get_pregame_injuries(team, kickoff)

                # Filter to injury-related news
                injury_news = [n for n in news if n.get("category") == "injury"]

                for item in injury_news:
                    annotation = self._create_ner_annotation(
                        news=item,
                        injuries=injuries,
                        team=team,
                        game_id=game["game_id"]
                    )
                    if annotation:
                        annotations.append(annotation)

        return annotations

    def _create_ner_annotation(
        self,
        news: dict,
        injuries: list[dict],
        team: str,
        game_id: str
    ) -> dict | None:
        """Create a single NER training annotation."""
        if not news.get("title"):
            return None

        text = news["title"]
        if news.get("description"):
            text += " " + news["description"]

        # Extract entities from text based on known injuries
        entities = []
        for injury in injuries:
            player_name = injury.get("athlete_name", "")
            if player_name and player_name.lower() in text.lower():
                # Find position in text
                idx = text.lower().find(player_name.lower())
                entities.append({
                    "text": player_name,
                    "type": "PLAYER",
                    "start": idx,
                    "end": idx + len(player_name),
                    "status": injury.get("status"),
                    "injury_type": injury.get("injury_type")
                })

        # Also look for injury-related terms
        injury_terms = ["injured", "out", "questionable", "doubtful", "IR", "ankle", "knee", "hamstring", "concussion"]
        for term in injury_terms:
            if term.lower() in text.lower():
                idx = text.lower().find(term.lower())
                entities.append({
                    "text": term,
                    "type": "INJURY",
                    "start": idx,
                    "end": idx + len(term)
                })

        if not entities:
            return None

        return {
            "id": f"ner_{news['id']}_{game_id}",
            "text": text,
            "team": team,
            "game_id": game_id,
            "entities": entities,
            # Format for Ollama fine-tuning
            "messages": [
                {
                    "role": "user",
                    "content": f"Extract injury-related entities from: {text}"
                },
                {
                    "role": "assistant",
                    "content": json.dumps({"entities": entities}, indent=2)
                }
            ]
        }

    def export_training_batch(
        self,
        output_dir: Path,
        season: int | None = None,
        weeks: list[int] | None = None,
        format: str = "jsonl"
    ) -> AnnotationStats:
        """
        Export annotations for model fine-tuning.

        Args:
            output_dir: Directory to save files
            season: Season to export (default: current year)
            weeks: Weeks to export (default: all completed)
            format: Output format ('jsonl' or 'spacy')

        Returns:
            AnnotationStats with export summary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        season = season or 2025
        stats = AnnotationStats(season=season, week=0)

        # Determine weeks to process
        if weeks is None:
            # Get all completed weeks
            query = """
            SELECT DISTINCT week
            FROM games
            WHERE season = %s AND home_score IS NOT NULL
            ORDER BY week
            """
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (season,))
                    weeks = [row[0] for row in cur.fetchall()]

        all_sentiment = []
        all_ner = []

        for week in weeks:
            try:
                sentiment = self.generate_sentiment_annotations(season, week)
                ner = self.generate_ner_annotations(season, week)

                all_sentiment.extend(sentiment)
                all_ner.extend(ner)

                stats.games_processed += len(self.get_games_with_outcomes(season, week))
                stats.sentiment_annotations += len(sentiment)
                stats.ner_annotations += len(ner)

            except Exception as e:
                stats.errors.append(f"Week {week}: {e}")
                logger.warning(f"Error processing week {week}: {e}")

        stats.week = max(weeks) if weeks else 0

        # Export files
        if format == "jsonl":
            self._export_jsonl(all_sentiment, output_dir / "sentiment_train.jsonl")
            self._export_jsonl(all_ner, output_dir / "ner_train.jsonl")
        else:
            # SpaCy format for NER
            self._export_spacy(all_ner, output_dir / "ner_train.spacy")

        logger.info(f"Exported {stats.sentiment_annotations} sentiment, {stats.ner_annotations} NER annotations")

        return stats

    def _export_jsonl(self, annotations: list[dict], output_path: Path) -> None:
        """Export annotations as JSONL."""
        with open(output_path, "w") as f:
            for ann in annotations:
                f.write(json.dumps(ann) + "\n")

    def _export_spacy(self, annotations: list[dict], output_path: Path) -> None:
        """Export NER annotations in SpaCy format."""
        # Convert to SpaCy training format
        spacy_format = []
        for ann in annotations:
            entities = [
                (e["start"], e["end"], e["type"])
                for e in ann.get("entities", [])
            ]
            spacy_format.append((ann["text"], {"entities": entities}))

        with open(output_path, "w") as f:
            json.dump(spacy_format, f, indent=2)


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    generator = AnnotationGenerator()

    print("\n=== Testing Annotation Generator ===")

    # Get games with outcomes
    games = generator.get_games_with_outcomes(2025, 13)
    print(f"\nGames with outcomes in Week 13: {len(games)}")

    if games:
        game = games[0]
        print(f"\nSample game: {game['game_id']}")
        print(f"  Result: {game['home_team']} {game['home_score']} - {game['away_score']} {game['away_team']}")

        # Get pregame news
        kickoff = game.get("kickoff") or datetime.now()
        news = generator.get_pregame_news(game["home_team"], kickoff, hours=48)
        print(f"\nPre-game news for {game['home_team']}: {len(news)} articles")

        # Generate annotations
        sentiment = generator.generate_sentiment_annotations(2025, 13)
        ner = generator.generate_ner_annotations(2025, 13)

        print(f"\nSentiment annotations: {len(sentiment)}")
        print(f"NER annotations: {len(ner)}")

        if sentiment:
            print(f"\nSample sentiment annotation:")
            sample = sentiment[0]
            print(f"  Text: {sample['text'][:100]}...")
            print(f"  Outcome: {sample['outcome_label']}")
