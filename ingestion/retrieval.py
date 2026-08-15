"""
Retrieval layer for the OmniBrain ingestion pipeline.

Provides structured vector similarity retrieval from Qdrant vector stores
preserving complete citation lineage and score ranking.
"""

from __future__ import annotations

import math
from typing import Any

from ingestion.models import VectorSearchResult
from ingestion.qdrant_store import QdrantVectorStore


def retrieve(
    query_vector: list[float],
    store: QdrantVectorStore,
    collection_name: str,
    top_k: int = 5,
) -> list[VectorSearchResult]:
    """Retrieve top-k most similar chunks from a Qdrant collection.

    Args:
        query_vector: Dense query vector as a list of finite floats.
        store: QdrantVectorStore instance.
        collection_name: Name of target collection.
        top_k: Maximum number of ranked results to return (positive integer).

    Returns:
        List of VectorSearchResult objects ordered by similarity score descending.

    Raises:
        TypeError: If store is not a QdrantVectorStore or query_vector is not a list.
        ValueError: If query_vector is empty, non-numeric, contains NaN/inf,
                    collection_name is empty, or top_k <= 0.
    """
    if not isinstance(store, QdrantVectorStore):
        raise TypeError(f"store must be an instance of QdrantVectorStore, got {type(store).__name__}.")

    if not collection_name or not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError("collection_name must be a non-empty string.")

    if not isinstance(query_vector, list):
        raise TypeError(f"query_vector must be a list, got {type(query_vector).__name__}.")

    if len(query_vector) == 0:
        raise ValueError("query_vector cannot be empty.")

    cleaned_query: list[float] = []
    for idx, val in enumerate(query_vector):
        if not isinstance(val, (int, float)) or isinstance(val, bool) or math.isnan(val) or math.isinf(val):
            raise ValueError(f"query_vector contains invalid non-numeric or non-finite value at index {idx}: {val!r}")
        cleaned_query.append(float(val))

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer > 0, got {top_k!r}.")

    return store.search_records(
        collection_name=collection_name.strip(),
        query_vector=cleaned_query,
        limit=top_k,
    )
