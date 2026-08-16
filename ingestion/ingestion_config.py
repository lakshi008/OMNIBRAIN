"""
Reusable, validated configuration layer for the OmniBrain ingestion pipeline.

Provides a single IngestionConfig dataclass consolidating chunking, retrieval,
and Qdrant collection settings, with strong validation and environment-variable support.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field


# ── Default Constants ─────────────────────────────────────────────────────

_DEFAULT_CHUNK_SIZE: int = 1000
_DEFAULT_CHUNK_OVERLAP: int = 200
_DEFAULT_RETRIEVAL_TOP_K: int = 5
_DEFAULT_RETRIEVAL_MIN_SCORE: float = 0.0
_DEFAULT_QDRANT_COLLECTION: str = "omnibrain_documents"
_DEFAULT_QDRANT_TIMEOUT: float = 10.0


def _validate_positive_int(value: object, name: str) -> int:
    """Return value if it is a strict positive non-bool integer, else raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"'{name}' must be a positive integer greater than 0, got {value!r}."
        )
    return value  # type: ignore[return-value]


def _validate_non_negative_int(value: object, name: str) -> int:
    """Return value if it is a non-negative non-bool integer, else raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"'{name}' must be a non-negative integer (>= 0), got {value!r}."
        )
    return value  # type: ignore[return-value]


def _validate_finite_float(value: object, name: str, low: float = 0.0, high: float = 1.0) -> float:
    """Return value if it is a finite float within [low, high], else raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a numeric value, got {value!r}.")
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        raise ValueError(f"'{name}' must be a finite number, got {fv!r}.")
    if not (low <= fv <= high):
        raise ValueError(
            f"'{name}' must be between {low} and {high} (inclusive), got {fv!r}."
        )
    return fv


def _validate_positive_float(value: object, name: str) -> float:
    """Return value if it is a finite positive float, else raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a numeric value, got {value!r}.")
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv) or fv <= 0:
        raise ValueError(
            f"'{name}' must be a positive finite number greater than 0, got {fv!r}."
        )
    return fv


def _validate_non_empty_string(value: object, name: str) -> str:
    """Return stripped value if it is a non-empty string, else raise ValueError."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"'{name}' must be a non-empty string, got {value!r}."
        )
    return value


@dataclass
class IngestionConfig:
    """Validated configuration object for the OmniBrain ingestion pipeline.

    Consolidates chunking, retrieval, and Qdrant collection settings into
    one reusable, env-aware configuration that can be passed into the ingestion service.

    Attributes:
        chunk_size: Target character size per text chunk. Must be > 0. Default: 1000.
        chunk_overlap: Overlap in characters between consecutive chunks.
            Must be >= 0 and < chunk_size. Default: 200.
        retrieval_top_k: Maximum number of nearest neighbours to retrieve. Must be > 0.
            Default: 5.
        retrieval_min_score: Minimum similarity score threshold for retrieved results.
            Must be a finite float in [0.0, 1.0]. Default: 0.0.
        qdrant_collection: Name of the Qdrant collection to store/search vectors in.
            Must be a non-empty string. Default: 'omnibrain_documents'.
        qdrant_timeout: Qdrant client request timeout in seconds. Must be > 0.
            Default: 10.0.
    """

    chunk_size: int = _DEFAULT_CHUNK_SIZE
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP
    retrieval_top_k: int = _DEFAULT_RETRIEVAL_TOP_K
    retrieval_min_score: float = _DEFAULT_RETRIEVAL_MIN_SCORE
    qdrant_collection: str = _DEFAULT_QDRANT_COLLECTION
    qdrant_timeout: float = _DEFAULT_QDRANT_TIMEOUT

    def __post_init__(self) -> None:
        """Validate all configuration values after construction."""
        self.chunk_size = _validate_positive_int(self.chunk_size, "chunk_size")
        self.chunk_overlap = _validate_non_negative_int(self.chunk_overlap, "chunk_overlap")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"'chunk_overlap' ({self.chunk_overlap}) must be strictly less than "
                f"'chunk_size' ({self.chunk_size})."
            )
        self.retrieval_top_k = _validate_positive_int(self.retrieval_top_k, "retrieval_top_k")
        self.retrieval_min_score = _validate_finite_float(
            self.retrieval_min_score, "retrieval_min_score", low=0.0, high=1.0
        )
        self.qdrant_collection = _validate_non_empty_string(
            self.qdrant_collection, "qdrant_collection"
        )
        self.qdrant_timeout = _validate_positive_float(self.qdrant_timeout, "qdrant_timeout")

    @classmethod
    def from_env(cls) -> IngestionConfig:
        """Construct an IngestionConfig by loading supported environment variables.

        Supported variables (all optional, safe defaults apply when absent):
            INGESTION_CHUNK_SIZE    — positive integer
            INGESTION_CHUNK_OVERLAP — non-negative integer
            INGESTION_TOP_K         — positive integer
            INGESTION_MIN_SCORE     — float in [0.0, 1.0]
            QDRANT_COLLECTION       — non-empty string
            QDRANT_TIMEOUT          — positive float (seconds)

        Invalid environment values raise descriptive ValueError rather than
        silently falling back to defaults.

        Returns:
            A validated IngestionConfig instance built from the environment.

        Raises:
            ValueError: If any environment variable contains an invalid value.
        """
        def _get_int(var: str, default: int) -> int:
            raw = os.getenv(var)
            if raw is None:
                return default
            try:
                return int(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Environment variable '{var}' must be a valid integer, got {raw!r}."
                )

        def _get_float(var: str, default: float) -> float:
            raw = os.getenv(var)
            if raw is None:
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Environment variable '{var}' must be a valid number, got {raw!r}."
                )

        def _get_str(var: str, default: str) -> str:
            raw = os.getenv(var)
            if raw is None:
                return default
            if not raw.strip():
                raise ValueError(
                    f"Environment variable '{var}' must not be empty or whitespace."
                )
            return raw

        return cls(
            chunk_size=_get_int("INGESTION_CHUNK_SIZE", _DEFAULT_CHUNK_SIZE),
            chunk_overlap=_get_int("INGESTION_CHUNK_OVERLAP", _DEFAULT_CHUNK_OVERLAP),
            retrieval_top_k=_get_int("INGESTION_TOP_K", _DEFAULT_RETRIEVAL_TOP_K),
            retrieval_min_score=_get_float("INGESTION_MIN_SCORE", _DEFAULT_RETRIEVAL_MIN_SCORE),
            qdrant_collection=_get_str("QDRANT_COLLECTION", _DEFAULT_QDRANT_COLLECTION),
            qdrant_timeout=_get_float("QDRANT_TIMEOUT", _DEFAULT_QDRANT_TIMEOUT),
        )
