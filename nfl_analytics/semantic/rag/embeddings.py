"""
Embedding generation for RAG system.

Uses sentence-transformers for local embedding generation.
Model: all-MiniLM-L6-v2 (fast, 384 dimensions, good quality)

Usage:
    generator = EmbeddingGenerator()
    embeddings = generator.embed_texts(["Patrick Mahomes injury report"])
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers."""

    # Default model - small, fast, good quality
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: Hugging Face model name
            cache_dir: Directory for model cache
            device: Device for inference ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(self.cache_dir) if self.cache_dir else None,
                    device=self.device,
                )
                logger.info(f"Model loaded on device: {self._model.device}")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Install with: uv pip install sentence-transformers"
                )
                raise
        return self._model

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.EMBEDDING_DIM)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            query: Query text

        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.embed_texts([query])[0]

    def is_available(self) -> bool:
        """Check if sentence-transformers is available."""
        try:
            from sentence_transformers import SentenceTransformer

            return True
        except ImportError:
            return False

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension for the current model."""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return self.EMBEDDING_DIM
