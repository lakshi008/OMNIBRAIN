"""
Tests for the retrieval result processor module.

Tests cover filtering by min_score, deduplication, highest-score duplicate preservation,
max_results bounds, citation/lineage integrity, and context string construction.
"""

from __future__ import annotations

import math
import uuid
import pytest

from ingestion.models import VectorSearchResult
from ingestion.retrieval_processor import (
    build_retrieval_context,
    process_retrieval_results,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_results() -> list[VectorSearchResult]:
    """Create a sample list of VectorSearchResult objects."""
    return [
        VectorSearchResult(
            chunk_id="chunk-1",
            score=0.92,
            document_id="doc-1",
            filename="report_2025.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="OmniBrain architecture is built on specialized agent swarms.",
            metadata={"char_count": 60, "section": "intro"},
        ),
        VectorSearchResult(
            chunk_id="chunk-2",
            score=0.85,
            document_id="doc-1",
            filename="report_2025.pdf",
            page_number=2,
            chunk_index=1,
            content_type="table",
            content="| Agent | Latency |\n| Search | 45ms |",
            metadata={"table_index": 0, "rows": 2},
        ),
        VectorSearchResult(
            chunk_id="chunk-3",
            score=0.78,
            document_id="doc-2",
            filename="diagrams.pdf",
            page_number=3,
            chunk_index=0,
            content_type="image",
            content="[Image: Diagram of multi-modal ingestion pipeline]",
            metadata={"image_index": 0, "width": 1024},
        ),
        VectorSearchResult(
            chunk_id="chunk-4",
            score=0.45,
            document_id="doc-2",
            filename="diagrams.pdf",
            page_number=4,
            chunk_index=1,
            content_type="text",
            content="Low-relevance appendix notes.",
            metadata={"char_count": 28},
        ),
    ]


# ── Processing Tests ─────────────────────────────────────────────────────


class TestProcessRetrievalResults:
    """Tests for process_retrieval_results."""

    def test_successful_processing_default_args(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Processes and returns results in score descending order."""
        processed = process_retrieval_results(sample_results)
        assert len(processed) == 4
        assert [r.score for r in processed] == [0.92, 0.85, 0.78, 0.45]
        assert processed[0].chunk_id == "chunk-1"

    def test_empty_results_returns_empty(self) -> None:
        """Empty input list returns empty list."""
        assert process_retrieval_results([]) == []

    def test_min_score_filtering(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Filters out items with score < min_score."""
        processed = process_retrieval_results(sample_results, min_score=0.80)
        assert len(processed) == 2
        assert all(r.score >= 0.80 for r in processed)
        assert [r.chunk_id for r in processed] == ["chunk-1", "chunk-2"]

    def test_max_results_limiting(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Limits output count to max_results."""
        processed = process_retrieval_results(sample_results, max_results=2)
        assert len(processed) == 2
        assert [r.chunk_id for r in processed] == ["chunk-1", "chunk-2"]

    def test_descending_score_ordering_from_unsorted(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Unsorted input is ordered strictly descending by score."""
        unsorted = [sample_results[2], sample_results[0], sample_results[1]]
        processed = process_retrieval_results(unsorted)
        assert [r.score for r in processed] == [0.92, 0.85, 0.78]

    def test_duplicate_chunk_removal(self) -> None:
        """Removes duplicate chunk_id entries."""
        duplicates = [
            VectorSearchResult("c1", 0.90, "d1", "a.pdf", 1, 0, "text", "Content 1"),
            VectorSearchResult("c1", 0.85, "d1", "a.pdf", 1, 0, "text", "Content 1"),
            VectorSearchResult("c2", 0.80, "d1", "a.pdf", 2, 1, "text", "Content 2"),
        ]
        processed = process_retrieval_results(duplicates)
        assert len(processed) == 2
        assert [r.chunk_id for r in processed] == ["c1", "c2"]

    def test_highest_score_duplicate_preservation(self) -> None:
        """Retains the entry with highest score when duplicates occur."""
        duplicates = [
            VectorSearchResult("c1", 0.70, "d1", "a.pdf", 1, 0, "text", "Lower score"),
            VectorSearchResult("c1", 0.95, "d1", "a.pdf", 1, 0, "text", "Higher score"),
        ]
        processed = process_retrieval_results(duplicates)
        assert len(processed) == 1
        assert processed[0].score == 0.95
        assert processed[0].content == "Higher score"

    def test_multiple_documents_preserved(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Preserves results originating from multiple documents."""
        processed = process_retrieval_results(sample_results)
        doc_ids = {r.document_id for r in processed}
        assert doc_ids == {"doc-1", "doc-2"}

    def test_modalities_preserved(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Preserves text, table, and image content types."""
        processed = process_retrieval_results(sample_results)
        types = [r.content_type for r in processed]
        assert "text" in types
        assert "table" in types
        assert "image" in types

    def test_citation_lineage_preserved(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Preserves chunk_id, document_id, filename, page_number, chunk_index, metadata."""
        processed = process_retrieval_results(sample_results)
        top = processed[0]
        assert top.chunk_id == "chunk-1"
        assert top.document_id == "doc-1"
        assert top.filename == "report_2025.pdf"
        assert top.page_number == 1
        assert top.chunk_index == 0
        assert top.content_type == "text"
        assert top.content == "OmniBrain architecture is built on specialized agent swarms."
        assert top.metadata == {"char_count": 60, "section": "intro"}


# ── Validation & Error Handling Tests ────────────────────────────────────


class TestProcessingErrorHandling:
    """Tests for input validation in process_retrieval_results."""

    def test_invalid_results_type_raises(self) -> None:
        """Non-list results argument raises TypeError."""
        with pytest.raises(TypeError, match="results must be a list"):
            process_retrieval_results("not_a_list")  # type: ignore

    def test_invalid_result_item_raises(self) -> None:
        """List containing non-VectorSearchResult item raises TypeError."""
        with pytest.raises(TypeError, match="not an instance of VectorSearchResult"):
            process_retrieval_results(["not_a_result"])  # type: ignore

    def test_invalid_min_score_nan_raises(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """NaN min_score raises ValueError."""
        with pytest.raises(ValueError, match="min_score must be a finite float"):
            process_retrieval_results(sample_results, min_score=float("nan"))

    def test_invalid_min_score_inf_raises(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Infinite min_score raises ValueError."""
        with pytest.raises(ValueError, match="min_score must be a finite float"):
            process_retrieval_results(sample_results, min_score=float("inf"))

    def test_invalid_min_score_out_of_bounds_raises(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Out-of-bounds min_score raises ValueError."""
        with pytest.raises(ValueError, match="between -1.0 and 1.0"):
            process_retrieval_results(sample_results, min_score=1.5)

    def test_invalid_min_score_type_raises(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Non-numeric min_score raises ValueError."""
        with pytest.raises(ValueError, match="min_score must be a finite float"):
            process_retrieval_results(sample_results, min_score="0.5")  # type: ignore

    def test_invalid_max_results_zero_raises(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """max_results <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_results must be a positive integer"):
            process_retrieval_results(sample_results, max_results=0)

    def test_invalid_max_results_negative_raises(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Negative max_results raises ValueError."""
        with pytest.raises(ValueError, match="max_results must be a positive integer"):
            process_retrieval_results(sample_results, max_results=-3)

    def test_invalid_max_results_type_raises(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Non-integer max_results raises ValueError."""
        with pytest.raises(ValueError, match="max_results must be a positive integer"):
            process_retrieval_results(sample_results, max_results="5")  # type: ignore


# ── Context Generation Tests ─────────────────────────────────────────────


class TestBuildContext:
    """Tests for build_retrieval_context."""

    def test_context_generation_single_source(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Formats single result with source, file, page, type, and content."""
        context = build_retrieval_context([sample_results[0]])
        assert "[Source 1]" in context
        assert "File: report_2025.pdf" in context
        assert "Page: 1" in context
        assert "Type: text" in context
        assert "Content:\nOmniBrain architecture is built on specialized agent swarms." in context

    def test_empty_context_returns_empty_string(self) -> None:
        """Empty results list returns empty string."""
        assert build_retrieval_context([]) == ""

    def test_multiple_source_separation(
        self, sample_results: list[VectorSearchResult]
    ) -> None:
        """Multiple sources are numbered and separated clearly."""
        context = build_retrieval_context(sample_results[:2])
        assert "[Source 1]" in context
        assert "[Source 2]" in context
        assert "File: report_2025.pdf" in context
        assert "Type: table" in context
        assert "| Agent | Latency |" in context
        # Sources are separated by double newline
        parts = context.split("\n\n")
        assert len(parts) == 2

    def test_page_number_none_handled(self) -> None:
        """Page number None is displayed as N/A."""
        result = VectorSearchResult(
            chunk_id="c1",
            score=0.9,
            document_id="d1",
            filename="unknown.pdf",
            page_number=None,
            chunk_index=0,
            content_type="text",
            content="Some text",
        )
        context = build_retrieval_context([result])
        assert "Page: N/A" in context

    def test_invalid_context_input_type_raises(self) -> None:
        """Non-list input raises TypeError."""
        with pytest.raises(TypeError, match="results must be a list"):
            build_retrieval_context("invalid")  # type: ignore

    def test_invalid_context_item_raises(self) -> None:
        """Non-VectorSearchResult item raises TypeError."""
        with pytest.raises(TypeError, match="not a VectorSearchResult"):
            build_retrieval_context(["invalid_item"])  # type: ignore
