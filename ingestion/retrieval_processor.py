"""
Retrieval result processor for the OmniBrain ingestion pipeline.

Provides filtering, deduplication, ranking, and context serialization for
VectorSearchResult objects retrieved from the vector database.
"""

from __future__ import annotations

import math
from typing import Any

from ingestion.models import VectorSearchResult


def process_retrieval_results(
    results: list[VectorSearchResult],
    min_score: float = 0.0,
    max_results: int = 5,
) -> list[VectorSearchResult]:
    """Filter, deduplicate, rank, and limit retrieved vector search results.

    1. Validates all inputs and item types.
    2. Filters out results with similarity score < min_score.
    3. Deduplicates items sharing the same chunk_id, retaining the highest-scoring record.
    4. Sorts remaining results strictly by score descending.
    5. Limits output to max_results.
    6. Preserves complete citation/lineage metadata.

    Args:
        results: List of VectorSearchResult objects.
        min_score: Minimum similarity score threshold (-1.0 to 1.0).
        max_results: Maximum number of results to return (positive integer > 0).

    Returns:
        Deduplicated, filtered, and score-ranked list of VectorSearchResult objects.

    Raises:
        TypeError: If results is not a list or contains non-VectorSearchResult items.
        ValueError: If min_score is non-finite or outside [-1.0, 1.0], or if max_results <= 0.
    """
    if not isinstance(results, list):
        raise TypeError(f"results must be a list, got {type(results).__name__}.")

    for idx, item in enumerate(results):
        if not isinstance(item, VectorSearchResult):
            raise TypeError(
                f"Item at index {idx} is not an instance of VectorSearchResult: got {type(item).__name__}."
            )

    if (
        not isinstance(min_score, (int, float))
        or isinstance(min_score, bool)
        or not math.isfinite(min_score)
        or min_score < -1.0
        or min_score > 1.0
    ):
        raise ValueError(
            f"min_score must be a finite float between -1.0 and 1.0, got {min_score!r}."
        )

    if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results <= 0:
        raise ValueError(
            f"max_results must be a positive integer > 0, got {max_results!r}."
        )

    if not results:
        return []

    # 1. Filter by min_score
    filtered = [r for r in results if r.score >= min_score]

    if not filtered:
        return []

    # 2. Deduplicate by chunk_id, preserving the highest-scoring record
    best_by_chunk_id: dict[str, VectorSearchResult] = {}
    for r in filtered:
        if r.chunk_id not in best_by_chunk_id or r.score > best_by_chunk_id[r.chunk_id].score:
            best_by_chunk_id[r.chunk_id] = r

    # 3. Sort by score descending (secondary sort by chunk_index, then chunk_id for determinism)
    deduped = list(best_by_chunk_id.values())
    deduped.sort(key=lambda r: (-r.score, r.chunk_index, r.chunk_id))

    # 4. Limit to max_results
    return deduped[:max_results]


def build_retrieval_context(
    results: list[VectorSearchResult],
) -> str:
    """Format ranked retrieval results into structured textual context for downstream agents.

    Formats each source with citation metadata:
    - [Source N]
    - File: <filename>
    - Page: <page_number or N/A>
    - Type: <content_type>
    - Content: <content>

    Args:
        results: List of VectorSearchResult objects in relevance order.

    Returns:
        Clean, formatted context string, or empty string if results is empty.

    Raises:
        TypeError: If results is not a list or contains non-VectorSearchResult items.
    """
    if not isinstance(results, list):
        raise TypeError(f"results must be a list, got {type(results).__name__}.")

    if not results:
        return ""

    source_blocks: list[str] = []
    for idx, res in enumerate(results, start=1):
        if not isinstance(res, VectorSearchResult):
            raise TypeError(
                f"Item at index {idx - 1} is not a VectorSearchResult: got {type(res).__name__}."
            )

        page_str = str(res.page_number) if res.page_number is not None else "N/A"
        block = (
            f"[Source {idx}]\n"
            f"File: {res.filename}\n"
            f"Page: {page_str}\n"
            f"Type: {res.content_type}\n"
            f"Content:\n"
            f"{res.content}"
        )
        source_blocks.append(block)

    return "\n\n".join(source_blocks)
