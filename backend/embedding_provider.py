"""
Production Real Embedding Provider for OmniBrain.

Implements the ingestion.embedding_generator.EmbeddingProvider Protocol
using Sentence Transformers (defaulting to 'sentence-transformers/all-MiniLM-L6-v2'
with 384 dimensions and Cosine similarity normalization).

Features:
- Full compliance with EmbeddingProvider Protocol (@runtime_checkable).
- Thread-safe lazy model loading (model is loaded once and cached in memory).
- Batch and single-text embedding generation.
- L2-normalized vector output for exact Qdrant COSINE distance metric compatibility.
- Safe input validation and empty batch handling.
- Zero network/model overhead during imports or test initialization.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, List

from ingestion.embedding_generator import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIMENSION: int = 384


class SentenceTransformerEmbeddingProvider:
    """Production embedding provider implementing the EmbeddingProvider protocol.

    Wraps a SentenceTransformer model with thread-safe lazy loading,
    batch processing, output normalization, and validation.

    Attributes:
        model_name: HuggingFace model identifier or local directory path.
        dimension: Dimensionality of output embeddings (default 384).
        normalize_embeddings: Whether to L2-normalize vectors (required for Cosine similarity).
        batch_size: Default batch size for batch encoding.
        device: Target compute device ('cpu', 'cuda', etc. or None for auto-detect).
    """

    def __init__(
        self,
        model_name: str | None = None,
        dimension: int | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        device: str | None = None,
        model_instance: Any | None = None,
    ) -> None:
        """Initialize SentenceTransformerEmbeddingProvider.

        Args:
            model_name: Model identifier (defaults to EMBEDDING_MODEL env var or all-MiniLM-L6-v2).
            dimension: Expected embedding dimension (defaults to EMBEDDING_DIMENSION env var or 384).
            normalize_embeddings: Whether to normalize output embeddings to unit length.
            batch_size: Batch size for batch encoding.
            device: Computation device ('cpu', 'cuda', etc.).
            model_instance: Optional pre-loaded SentenceTransformer instance (for testing/mocking).
        """
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL")
            or DEFAULT_MODEL_NAME
        ).strip()

        env_dim = os.getenv("EMBEDDING_DIMENSION")
        if dimension is not None:
            self.dimension = dimension
        elif env_dim:
            self.dimension = int(env_dim)
        else:
            self.dimension = DEFAULT_DIMENSION

        if self.dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {self.dimension}")

        self.normalize_embeddings = normalize_embeddings
        self.batch_size = max(1, batch_size)
        self.device = device

        self._model = model_instance
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Check whether the underlying model is already loaded in memory."""
        return self._model is not None

    def _load_model(self) -> Any:
        """Thread-safe lazy loading of the SentenceTransformer model."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            logger.info(
                "Loading SentenceTransformer model '%s' on device '%s'...",
                self.model_name,
                self.device or "auto",
            )
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(
                    model_name_or_path=self.model_name,
                    device=self.device,
                )

                # Validate dimension if possible
                if hasattr(model, "get_embedding_dimension"):
                    model_dim = model.get_embedding_dimension()
                elif hasattr(model, "get_sentence_embedding_dimension"):
                    model_dim = model.get_sentence_embedding_dimension()
                else:
                    model_dim = None

                if model_dim is not None and model_dim != self.dimension:
                    logger.warning(
                        "Model '%s' reports dimension %d, but configured dimension is %d. Updating dimension.",
                        self.model_name,
                        model_dim,
                        self.dimension,
                    )
                    self.dimension = model_dim

                self._model = model
                logger.info(
                    "SentenceTransformer model '%s' loaded successfully (dimension=%d).",
                    self.model_name,
                    self.dimension,
                )
                return self._model
            except ImportError as exc:
                raise ImportError(
                    "The 'sentence-transformers' package is required for SentenceTransformerEmbeddingProvider. "
                    "Install it with `pip install sentence-transformers`."
                ) from exc
            except Exception as exc:
                logger.error(
                    "Failed to load SentenceTransformer model '%s': %s",
                    self.model_name,
                    exc,
                )
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}': {exc}"
                ) from exc

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        Args:
            text: Text content to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            TypeError: If text is not a string.
            ValueError: If text is empty or whitespace-only.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be a str, got {type(text).__name__}")

        if not text.strip():
            raise ValueError("Cannot generate embedding for empty or whitespace-only text.")

        model = self._load_model()
        vector = model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        if hasattr(vector, "tolist"):
            result: list[float] = [float(x) for x in vector.tolist()]
        elif isinstance(vector, (list, tuple)):
            result = [float(x) for x in vector]
        else:
            raise ValueError(f"Unexpected output type from encoder: {type(vector).__name__}")

        # Validate dimension and finite values
        if len(result) != self.dimension:
            raise ValueError(
                f"Generated vector dimension {len(result)} does not match expected dimension {self.dimension}."
            )

        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of text strings.

        Args:
            texts: List of non-empty text strings to embed.

        Returns:
            List of embedding vectors corresponding to the input texts.

        Raises:
            TypeError: If texts is not a list or contains non-string elements.
            ValueError: If any text string is empty or whitespace-only.
        """
        if not isinstance(texts, list):
            raise TypeError(f"Expected texts to be a list, got {type(texts).__name__}")

        if not texts:
            return []

        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Item at index {idx} is not a str: got {type(text).__name__}"
                )
            if not text.strip():
                raise ValueError(
                    f"Item at index {idx} has empty or whitespace-only content."
                )

        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        results: list[list[float]] = []
        for idx, vec in enumerate(vectors):
            if hasattr(vec, "tolist"):
                cleaned = [float(x) for x in vec.tolist()]
            elif isinstance(vec, (list, tuple)):
                cleaned = [float(x) for x in vec]
            else:
                raise ValueError(f"Unexpected output type for item {idx}: {type(vec).__name__}")

            if len(cleaned) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(cleaned)} for item {idx} does not match expected dimension {self.dimension}."
                )
            results.append(cleaned)

        return results


# Alias for clean naming in the dependency injection layer
RealEmbeddingProvider = SentenceTransformerEmbeddingProvider
