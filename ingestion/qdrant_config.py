"""
Configuration module for Qdrant vector database integration.

Reads environment variables with safe defaults for local/in-memory development.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class QdrantConfig:
    """Configuration settings for Qdrant client connection.

    Attributes:
        url: Qdrant endpoint URL or ':memory:' for in-memory testing.
        api_key: Optional API key for authenticated instances.
        default_collection: Default collection name for document vectors.
        timeout: Request timeout in seconds.
    """

    url: str = ":memory:"
    api_key: str | None = None
    default_collection: str = "omnibrain_documents"
    timeout: float = 10.0

    @classmethod
    def from_env(cls) -> QdrantConfig:
        """Construct QdrantConfig from environment variables with safe fallbacks."""
        return cls(
            url=os.getenv("QDRANT_URL", ":memory:"),
            api_key=os.getenv("QDRANT_API_KEY") or None,
            default_collection=os.getenv("QDRANT_COLLECTION", "omnibrain_documents"),
            timeout=float(os.getenv("QDRANT_TIMEOUT", "10.0")),
        )
