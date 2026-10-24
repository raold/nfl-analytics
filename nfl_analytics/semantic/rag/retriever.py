"""
Game context retriever for RAG-augmented predictions.

Retrieves relevant historical game contexts based on semantic similarity.
Integrates with LLM inference for contextualized predictions.

Usage:
    retriever = GameContextRetriever()
    retriever.build_index(teams=['KC', 'BAL'], seasons=[2022, 2023, 2024])
    context = retriever.get_context('KC', 'BAL', 2025, 16)
"""

import logging
from pathlib import Path
from typing import Any, Optional

from nfl_analytics.semantic.rag.embeddings import EmbeddingGenerator
from nfl_analytics.semantic.rag.game_context import GameContext, GameContextBuilder
from nfl_analytics.semantic.rag.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class GameContextRetriever:
    """Retrieve relevant historical game contexts for RAG."""

    DEFAULT_PERSIST_PATH = Path("models/rag/game_context_store")

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize retriever.

        Args:
            persist_path: Path for persisting vector store
            embedding_model: Model name for embeddings
        """
        self.persist_path = persist_path or self.DEFAULT_PERSIST_PATH
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.vector_store = VectorStore(
            embedding_generator=self.embedding_generator,
            persist_path=self.persist_path,
        )
        self.context_builder = GameContextBuilder()
        self._indexed_games = set()

    def build_index(
        self,
        teams: Optional[list[str]] = None,
        seasons: Optional[list[int]] = None,
        include_injuries: bool = True,
    ) -> int:
        """
        Build vector index from historical game data.

        Args:
            teams: List of team abbreviations (None = all 32 teams)
            seasons: List of seasons to index (None = 2020-2024)
            include_injuries: Whether to include injury contexts

        Returns:
            Number of contexts indexed
        """
        if teams is None:
            teams = [
                "ARI",
                "ATL",
                "BAL",
                "BUF",
                "CAR",
                "CHI",
                "CIN",
                "CLE",
                "DAL",
                "DEN",
                "DET",
                "GB",
                "HOU",
                "IND",
                "JAX",
                "KC",
                "LA",
                "LAC",
                "LV",
                "MIA",
                "MIN",
                "NE",
                "NO",
                "NYG",
                "NYJ",
                "PHI",
                "PIT",
                "SEA",
                "SF",
                "TB",
                "TEN",
                "WAS",
            ]

        if seasons is None:
            seasons = [2020, 2021, 2022, 2023, 2024]

        logger.info(f"Building index for {len(teams)} teams across {len(seasons)} seasons...")

        all_contexts: list[GameContext] = []

        # Build game contexts for each team
        for team in teams:
            contexts = self.context_builder.build_team_contexts(
                team=team,
                seasons=seasons,
                limit=100,  # Last 100 games per team
            )
            all_contexts.extend(contexts)

            if include_injuries:
                injury_contexts = self.context_builder.build_injury_contexts(
                    team=team,
                    seasons=seasons,
                    limit=200,  # More injury records
                )
                all_contexts.extend(injury_contexts)

        # Deduplicate by game_id
        unique_contexts = {}
        for ctx in all_contexts:
            if ctx.game_id not in unique_contexts:
                unique_contexts[ctx.game_id] = ctx

        contexts_list = list(unique_contexts.values())
        logger.info(f"Built {len(contexts_list)} unique contexts")

        # Add to vector store
        if contexts_list:
            texts = [ctx.text for ctx in contexts_list]
            metadata = [
                ctx.metadata | {"game_id": ctx.game_id, "season": ctx.season, "week": ctx.week}
                for ctx in contexts_list
            ]
            self.vector_store.add(texts=texts, metadata=metadata)

            # Track indexed games
            self._indexed_games.update(ctx.game_id for ctx in contexts_list)

            # Persist
            self.vector_store.save()

        return len(contexts_list)

    def get_context(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        k: int = 10,
        include_types: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Get relevant historical context for a matchup.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Current season
            week: Current week
            k: Number of context items to retrieve
            include_types: Types to include ('game', 'injury')

        Returns:
            Dictionary with:
                - matchup_history: Past matchups between these teams
                - home_team_context: Recent home team performance
                - away_team_context: Recent away team performance
                - injury_context: Relevant injury history
                - combined_context: Formatted context for LLM
        """
        include_types = include_types or ["game", "injury"]

        # Build query
        query = f"{away_team} at {home_team} matchup {season} week {week} NFL football"

        # Search for relevant contexts
        results = self.vector_store.search(
            query=query,
            k=k * 2,  # Get more, then filter
            score_threshold=0.3,
        )

        # Categorize results
        matchup_history = []
        home_context = []
        away_context = []
        injury_context = []

        for result in results:
            meta = result.metadata
            ctx_type = meta.get("type", "game")

            # Skip future games
            if meta.get("season", 0) > season:
                continue
            if meta.get("season") == season and meta.get("week", 0) >= week:
                continue

            if ctx_type == "game":
                # Check if it's a matchup between these teams
                if home_team in result.text and away_team in result.text:
                    matchup_history.append(result)
                elif home_team in result.text:
                    home_context.append(result)
                elif away_team in result.text:
                    away_context.append(result)

            elif ctx_type == "injury" and "injury" in include_types:
                if home_team in result.text or away_team in result.text:
                    injury_context.append(result)

        # Build combined context string
        combined = self._format_combined_context(
            matchup_history=matchup_history[:3],
            home_context=home_context[:3],
            away_context=away_context[:3],
            injury_context=injury_context[:5],
            home_team=home_team,
            away_team=away_team,
        )

        return {
            "matchup_history": [self._result_to_dict(r) for r in matchup_history[:5]],
            "home_team_context": [self._result_to_dict(r) for r in home_context[:5]],
            "away_team_context": [self._result_to_dict(r) for r in away_context[:5]],
            "injury_context": [self._result_to_dict(r) for r in injury_context[:5]],
            "combined_context": combined,
            "query": query,
            "total_results": len(results),
        }

    def _format_combined_context(
        self,
        matchup_history: list[SearchResult],
        home_context: list[SearchResult],
        away_context: list[SearchResult],
        injury_context: list[SearchResult],
        home_team: str,
        away_team: str,
    ) -> str:
        """Format all contexts into a single string for LLM consumption."""
        parts = []

        if matchup_history:
            parts.append(f"## Historical Matchups ({home_team} vs {away_team})")
            for r in matchup_history:
                parts.append(f"- {r.text}")

        if home_context:
            parts.append(f"\n## {home_team} Recent Performance")
            for r in home_context:
                parts.append(f"- {r.text}")

        if away_context:
            parts.append(f"\n## {away_team} Recent Performance")
            for r in away_context:
                parts.append(f"- {r.text}")

        if injury_context:
            parts.append("\n## Relevant Injury History")
            for r in injury_context:
                parts.append(f"- {r.text}")

        return "\n".join(parts) if parts else "No historical context available."

    def _result_to_dict(self, result: SearchResult) -> dict[str, Any]:
        """Convert SearchResult to dictionary."""
        return {
            "text": result.text,
            "score": result.score,
            "metadata": result.metadata,
            "rank": result.rank,
        }

    def is_available(self) -> bool:
        """Check if retriever is available (has indexed data)."""
        return len(self.vector_store) > 0

    def load(self) -> bool:
        """Load persisted index."""
        try:
            self.vector_store.load()
            return len(self.vector_store) > 0
        except Exception as e:
            logger.warning(f"Could not load index: {e}")
            return False
