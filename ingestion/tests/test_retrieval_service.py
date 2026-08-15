"""
Tests for the retrieval service layer.

Tests verify end-to-end retrieval, filtering, deduplication, and context building
using QdrantVectorStore in-memory and RetrievalServiceResult dataclass helpers.
"""

from __future__ import annotations

import math
import uuid
import pytest

from ingestion.models import (
    EmbeddingGenerationResult,
    EmbeddingVectorRecord,
    RetrievalServiceResult,
    VectorSearchResult,
)
from ingestion.qdrant_config import QdrantConfig
from ingestion.qdrant_store import QdrantVectorStore
from ingestion.retrieval_service import retrieve_context


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def in_memory_store() -> QdrantVectorStore:
    """Create in-memory QdrantVectorStore."""
    config = QdrantConfig(url=":memory:")
    return QdrantVectorStore(config=config)


@pytest.fixture
def sample_doc_id() -> str:
    """Create sample document UUID."""
    return str(uuid.uuid4())


@pytest.fixture
def populated_service_store(
    in_memory_store: QdrantVectorStore, sample_doc_id: str
) -> tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]:
    """Populate an in-memory collection with multi-modal vectors."""
    col_name = "service_test_col"
    in_memory_store.create_collection(col_name, vector_dimension=4)

    records = [
        EmbeddingVectorRecord(
            chunk_id=str(uuid.uuid4()),
            document_id=sample_doc_id,
            filename="omnibrain_spec.pdf",
            chunk_index=0,
            page_number=1,
            content_type="text",
            vector=[1.0, 0.0, 0.0, 0.0],
            metadata={"char_count": 65, "content": "OmniBrain architecture orchestrates agent swarms."},
        ),
        EmbeddingVectorRecord(
            chunk_id=str(uuid.uuid4()),
            document_id=sample_doc_id,
            filename="omnibrain_spec.pdf",
            chunk_index=1,
            page_number=2,
            content_type="table",
            vector=[0.7, 0.7, 0.0, 0.0],
            metadata={"table_index": 0, "rows": 3, "content": "| Agent | Type |\n| Search | Retrieval |"},
        ),
        EmbeddingVectorRecord(
            chunk_id=str(uuid.uuid4()),
            document_id=sample_doc_id,
            filename="omnibrain_spec.pdf",
            chunk_index=2,
            page_number=3,
            content_type="image",
            vector=[0.0, 0.0, 1.0, 0.0],
            metadata={"image_index": 0, "width": 800, "content": "[Image on page 3: System Overview]"},
        ),
    ]

    gen_result = EmbeddingGenerationResult(
        document_id=sample_doc_id,
        filename="omnibrain_spec.pdf",
        items=records,
        dimension=4,
        is_ready=True,
    )
    in_memory_store.upsert_embeddings(col_name, gen_result)
    return in_memory_store, col_name, records


# ── Success Tests ────────────────────────────────────────────────────────


class TestRetrievalServiceSuccess:
    """Tests for successful execution of retrieve_context."""

    def test_retrieve_context_returns_retrieval_service_result(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """retrieve_context returns RetrievalServiceResult with all fields populated."""
        store, col_name, records = populated_service_store
        result = retrieve_context(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            store=store,
            collection_name=col_name,
            top_k=3,
            min_score=0.0,
            max_results=3,
        )

        assert isinstance(result, RetrievalServiceResult)
        assert result.query_vector_dimension == 4
        assert result.has_results is True
        assert result.total_results == 3
        assert len(result.results) == 3
        assert isinstance(result.context, str)
        assert len(result.context) > 0
        assert "[Source 1]" in result.context
        assert "File: omnibrain_spec.pdf" in result.context

    def test_multi_modal_counts_and_helpers(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """Helper properties and methods filter by modality and page."""
        store, col_name, records = populated_service_store
        result = retrieve_context(
            query_vector=[0.5, 0.5, 0.5, 0.0],
            store=store,
            collection_name=col_name,
            top_k=3,
            min_score=0.0,
            max_results=3,
        )

        assert result.text_results == 1
        assert result.table_results == 1
        assert result.image_results == 1
        assert len(result.get_results_by_type("text")) == 1
        assert len(result.get_results_by_type("table")) == 1
        assert len(result.get_results_by_type("image")) == 1
        assert len(result.get_results_on_page(1)) == 1
        assert len(result.get_results_on_page(2)) == 1
        assert len(result.get_results_on_page(99)) == 0

    def test_score_ordering(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """Results and context are strictly ordered by similarity score descending."""
        store, col_name, records = populated_service_store
        # Vector closest to text item ([1, 0, 0, 0])
        result = retrieve_context(
            query_vector=[0.95, 0.05, 0.0, 0.0],
            store=store,
            collection_name=col_name,
            top_k=3,
        )
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)
        assert result.results[0].chunk_id == records[0].chunk_id

    def test_min_score_filtering(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """min_score filters out lower-similarity items."""
        store, col_name, records = populated_service_store
        # Query matching [1, 0, 0, 0] perfectly, others lower
        result = retrieve_context(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            store=store,
            collection_name=col_name,
            top_k=3,
            min_score=0.85,
        )
        assert result.total_results == 1
        assert result.results[0].chunk_id == records[0].chunk_id

    def test_max_results_limiting(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """max_results restricts the number of returned items."""
        store, col_name, records = populated_service_store
        result = retrieve_context(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            store=store,
            collection_name=col_name,
            top_k=3,
            max_results=1,
        )
        assert result.total_results == 1

    def test_empty_collection_handling(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """Empty collection yields empty results without errors."""
        in_memory_store.create_collection("empty_col", vector_dimension=4)
        result = retrieve_context(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            store=in_memory_store,
            collection_name="empty_col",
        )
        assert result.has_results is False
        assert result.total_results == 0
        assert result.results == []
        assert result.context == ""

    def test_citation_metadata_preservation(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """All citation/lineage fields are preserved in results."""
        store, col_name, records = populated_service_store
        result = retrieve_context(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            store=store,
            collection_name=col_name,
            top_k=1,
        )
        top = result.results[0]
        assert top.chunk_id == records[0].chunk_id
        assert top.document_id == records[0].document_id
        assert top.filename == "omnibrain_spec.pdf"
        assert top.page_number == 1
        assert top.chunk_index == 0
        assert top.content_type == "text"
        assert top.content == "OmniBrain architecture orchestrates agent swarms."
        assert top.metadata["char_count"] == 65


# ── Validation & Error Handling Tests ────────────────────────────────────


class TestRetrievalServiceValidation:
    """Tests for input validation in retrieve_context."""

    def test_empty_query_vector_raises(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """Empty query vector raises ValueError."""
        store, col_name, _ = populated_service_store
        with pytest.raises(ValueError, match="cannot be empty"):
            retrieve_context([], store, col_name)

    def test_nan_in_query_vector_raises(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """NaN in query vector raises ValueError."""
        store, col_name, _ = populated_service_store
        with pytest.raises(ValueError, match="invalid non-numeric or non-finite"):
            retrieve_context([float("nan"), 0.0, 0.0, 0.0], store, col_name)

    def test_infinity_in_query_vector_raises(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """Infinity in query vector raises ValueError."""
        store, col_name, _ = populated_service_store
        with pytest.raises(ValueError, match="invalid non-numeric or non-finite"):
            retrieve_context([float("inf"), 0.0, 0.0, 0.0], store, col_name)

    def test_non_numeric_in_query_vector_raises(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """String element in query vector raises ValueError."""
        store, col_name, _ = populated_service_store
        with pytest.raises(ValueError, match="invalid non-numeric or non-finite"):
            retrieve_context(["1.0", 0.0, 0.0, 0.0], store, col_name)  # type: ignore

    def test_non_list_query_vector_raises(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """Non-list query vector raises TypeError."""
        store, col_name, _ = populated_service_store
        with pytest.raises(TypeError, match="must be a list"):
            retrieve_context("invalid", store, col_name)  # type: ignore

    def test_invalid_store_raises(self) -> None:
        """Invalid store instance raises TypeError."""
        with pytest.raises(TypeError, match="must be an instance of QdrantVectorStore"):
            retrieve_context([1.0, 0.0, 0.0, 0.0], "not_a_store", "col")  # type: ignore

    def test_empty_collection_name_raises(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """Empty collection name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            retrieve_context([1.0, 0.0, 0.0, 0.0], in_memory_store, "")

    def test_invalid_top_k_raises(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """top_k <= 0 raises ValueError."""
        store, col_name, _ = populated_service_store
        with pytest.raises(ValueError, match="positive integer > 0"):
            retrieve_context([1.0, 0.0, 0.0, 0.0], store, col_name, top_k=0)

    def test_invalid_min_score_raises(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """min_score outside [-1.0, 1.0] raises ValueError."""
        store, col_name, _ = populated_service_store
        with pytest.raises(ValueError, match="between -1.0 and 1.0"):
            retrieve_context([1.0, 0.0, 0.0, 0.0], store, col_name, min_score=2.0)

    def test_invalid_max_results_raises(
        self,
        populated_service_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]],
    ) -> None:
        """max_results <= 0 raises ValueError."""
        store, col_name, _ = populated_service_store
        with pytest.raises(ValueError, match="positive integer > 0"):
            retrieve_context([1.0, 0.0, 0.0, 0.0], store, col_name, max_results=-1)
