"""
RAG (Retrieval-Augmented Generation) module for NFL analytics.

Provides historical game context retrieval using semantic search:
- Embedding generation with sentence-transformers
- Vector store for efficient similarity search
- Game context retrieval for LLM augmentation

Usage:
    from nfl_analytics.semantic.rag import GameContextRetriever

    retriever = GameContextRetriever()
    context = retriever.get_context(home_team='KC', away_team='BAL', season=season, week=16)
    # Returns: List of relevant historical matchups, injuries, coaching notes
"""

from nfl_analytics.semantic.rag.embeddings import EmbeddingGenerator
from nfl_analytics.semantic.rag.vector_store import VectorStore
from nfl_analytics.semantic.rag.retriever import GameContextRetriever
from nfl_analytics.semantic.rag.game_context import GameContextBuilder

__all__ = [
    "EmbeddingGenerator",
    "VectorStore",
    "GameContextRetriever",
    "GameContextBuilder",
]
