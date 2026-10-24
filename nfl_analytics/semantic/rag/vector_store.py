"""
Vector store for semantic search.

Simple numpy-based implementation for semantic similarity search.
Can be upgraded to FAISS or Qdrant for larger scale deployments.

Usage:
    store = VectorStore()
    store.add(texts=["Game 1 context", "Game 2 context"], metadata=[...])
    results = store.search("Chiefs injury report", k=5)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from py.semantic.rag.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector similarity search."""

    text: str
    score: float
    metadata: dict[str, Any]
    rank: int


class VectorStore:
    """Simple numpy-based vector store for semantic search."""

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        persist_path: Optional[Path] = None,
    ):
        """
        Initialize vector store.

        Args:
            embedding_generator: EmbeddingGenerator instance (creates one if not provided)
            persist_path: Path to persist/load store (optional)
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.persist_path = persist_path

        # Storage
        self.embeddings: Optional[np.ndarray] = None
        self.texts: list[str] = []
        self.metadata: list[dict[str, Any]] = []

        # Load if persist path exists
        if persist_path and persist_path.exists():
            self.load()

    def add(
        self,
        texts: list[str],
        metadata: Optional[list[dict[str, Any]]] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add texts to the vector store.

        Args:
            texts: List of text strings
            metadata: Optional metadata for each text
            embeddings: Pre-computed embeddings (optional)
        """
        if not texts:
            return

        # Generate embeddings if not provided
        if embeddings is None:
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.embedding_generator.embed_texts(texts)

        # Initialize or append
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        # Store texts and metadata
        self.texts.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])

        logger.info(f"Vector store now contains {len(self.texts)} documents")

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
        filter_fn: Optional[callable] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            filter_fn: Optional function to filter metadata

        Returns:
            List of SearchResult objects
        """
        if self.embeddings is None or len(self.texts) == 0:
            logger.warning("Vector store is empty")
            return []

        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)

        # Compute cosine similarities (embeddings are L2 normalized)
        similarities = np.dot(self.embeddings, query_embedding)

        # Apply metadata filter if provided
        if filter_fn:
            mask = np.array([filter_fn(m) for m in self.metadata])
            similarities = np.where(mask, similarities, -np.inf)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            score = float(similarities[idx])
            if score < score_threshold:
                continue

            results.append(
                SearchResult(
                    text=self.texts[idx],
                    score=score,
                    metadata=self.metadata[idx],
                    rank=rank + 1,
                )
            )

        return results

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save vector store to disk.

        Args:
            path: Path to save (uses persist_path if not provided)
        """
        save_path = path or self.persist_path
        if save_path is None:
            raise ValueError("No path provided for saving")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        np.save(save_path.with_suffix(".npy"), self.embeddings)

        # Save texts and metadata
        with open(save_path.with_suffix(".json"), "w") as f:
            json.dump({"texts": self.texts, "metadata": self.metadata}, f)

        logger.info(f"Vector store saved to {save_path}")

    def load(self, path: Optional[Path] = None) -> None:
        """
        Load vector store from disk.

        Args:
            path: Path to load from (uses persist_path if not provided)
        """
        load_path = path or self.persist_path
        if load_path is None:
            raise ValueError("No path provided for loading")

        load_path = Path(load_path)

        # Load embeddings
        embeddings_path = load_path.with_suffix(".npy")
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)

        # Load texts and metadata
        json_path = load_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                self.texts = data.get("texts", [])
                self.metadata = data.get("metadata", [])

        logger.info(
            f"Vector store loaded: {len(self.texts)} documents, "
            f"embedding shape: {self.embeddings.shape if self.embeddings is not None else None}"
        )

    def clear(self) -> None:
        """Clear all stored data."""
        self.embeddings = None
        self.texts = []
        self.metadata = []
        logger.info("Vector store cleared")

    def __len__(self) -> int:
        """Return number of documents in store."""
        return len(self.texts)
