"""
Retrieval service layer for the OmniBrain ingestion pipeline.

Combines Qdrant similarity search with score filtering, deduplication,
ranking, and citation context construction into a single high-level API.
"""

from __future__ import annotations

from ingestion.models import RetrievalServiceResult, VectorSearchResult
from ingestion.qdrant_store import QdrantVectorStore
from ingestion.retrieval import retrieve
from ingestion.retrieval_processor import (
    build_retrieval_context,
    process_retrieval_results,
)


def retrieve_context(
    query_vector: list[float],
    store: QdrantVectorStore,
    collection_name: str,
    top_k: int = 5,
    min_score: float = 0.0,
    max_results: int = 5,
) -> RetrievalServiceResult:
    """Retrieve top-k similar records, filter/deduplicate them, and construct context.

    Pipeline:
        1. Validates query vector, store, collection name, top_k via retrieve().
        2. Performs vector similarity search against Qdrant.
        3. Filters by min_score, deduplicates by chunk_id, and limits by max_results via process_retrieval_results().
        4. Formats into structured source context blocks via build_retrieval_context().
        5. Returns structured RetrievalServiceResult.

    Args:
        query_vector: Dense query vector as a list of floats.
        store: QdrantVectorStore instance.
        collection_name: Target Qdrant collection name.
        top_k: Initial number of nearest neighbours to retrieve from Qdrant (> 0).
        min_score: Minimum similarity score threshold (-1.0 to 1.0).
        max_results: Maximum final processed results to return (> 0).

    Returns:
        RetrievalServiceResult containing ranked results, query dimension, and formatted context.

    Raises:
        TypeError: If arguments are of incorrect types.
        ValueError: If arguments contain invalid or non-finite values.
    """
    # 1. Retrieve raw search results from Qdrant
    raw_results = retrieve(
        query_vector=query_vector,
        store=store,
        collection_name=collection_name,
        top_k=top_k,
    )

    # 2. Filter, deduplicate, and limit results
    processed_results = process_retrieval_results(
        results=raw_results,
        min_score=min_score,
        max_results=max_results,
    )

    # 3. Build textual context
    context = build_retrieval_context(processed_results)

    # 4. Construct high-level service result
    return RetrievalServiceResult(
        query_vector_dimension=len(query_vector),
        results=processed_results,
        context=context,
    )
