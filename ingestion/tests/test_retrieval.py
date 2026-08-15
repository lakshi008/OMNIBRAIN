"""
Tests for the retrieval layer.

Tests verify vector similarity retrieval from QdrantVectorStore, top_k ranking,
score ordering descending, citation/lineage metadata preservation, and validation.
"""

from __future__ import annotations

import math
import uuid
import pytest
from qdrant_client import QdrantClient

from ingestion.models import (
    EmbeddingGenerationResult,
    EmbeddingVectorRecord,
    VectorSearchResult,
)
from ingestion.qdrant_config import QdrantConfig
from ingestion.qdrant_store import QdrantVectorStore
from ingestion.retrieval import retrieve


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
def populated_store(in_memory_store: QdrantVectorStore, sample_doc_id: str) -> tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]:
    """Populate an in-memory collection with 4-dimensional vectors."""
    col_name = "test_retrieval_col"
    in_memory_store.create_collection(col_name, vector_dimension=4)

    records = [
        EmbeddingVectorRecord(
            chunk_id=str(uuid.uuid4()),
            document_id=sample_doc_id,
            filename="omni_report.pdf",
            chunk_index=0,
            page_number=1,
            content_type="text",
            vector=[1.0, 0.0, 0.0, 0.0],
            metadata={"char_count": 50, "content": "OmniBrain introduction to agents."},
        ),
        EmbeddingVectorRecord(
            chunk_id=str(uuid.uuid4()),
            document_id=sample_doc_id,
            filename="omni_report.pdf",
            chunk_index=1,
            page_number=2,
            content_type="table",
            vector=[0.7, 0.7, 0.0, 0.0],
            metadata={"table_index": 0, "rows": 3, "content": "| Model | Score |\n| Agent | 99% |"},
        ),
        EmbeddingVectorRecord(
            chunk_id=str(uuid.uuid4()),
            document_id=sample_doc_id,
            filename="omni_report.pdf",
            chunk_index=2,
            page_number=3,
            content_type="image",
            vector=[0.0, 0.0, 1.0, 0.0],
            metadata={"image_index": 0, "width": 800, "content": "[Image on page 3]"},
        ),
    ]

    gen_result = EmbeddingGenerationResult(
        document_id=sample_doc_id,
        filename="omni_report.pdf",
        items=records,
        dimension=4,
        is_ready=True,
    )
    in_memory_store.upsert_embeddings(col_name, gen_result)
    return in_memory_store, col_name, records


# ── Success Tests ────────────────────────────────────────────────────────


class TestRetrievalSuccess:
    """Tests for successful vector retrieval."""

    def test_retrieve_returns_vector_search_results(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """retrieve returns a list of VectorSearchResult objects."""
        store, col_name, records = populated_store
        results = retrieve(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            store=store,
            collection_name=col_name,
            top_k=2,
        )
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, VectorSearchResult)

    def test_score_ordering_descending(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """Results are ordered by similarity score descending."""
        store, col_name, records = populated_store
        # Query closest to record 0, then record 1, then record 2
        results = retrieve(
            query_vector=[0.9, 0.1, 0.0, 0.0],
            store=store,
            collection_name=col_name,
            top_k=3,
        )
        assert len(results) == 3
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].chunk_id == records[0].chunk_id

    def test_top_k_bounds(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """top_k limit bounds the number of returned records."""
        store, col_name, _ = populated_store
        res1 = retrieve([1.0, 0.0, 0.0, 0.0], store, col_name, top_k=1)
        assert len(res1) == 1

        res2 = retrieve([1.0, 0.0, 0.0, 0.0], store, col_name, top_k=3)
        assert len(res2) == 3

    def test_preserves_citation_metadata(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """Retrieval preserves complete citation fields for downstream RAG citations."""
        store, col_name, records = populated_store
        results = retrieve([1.0, 0.0, 0.0, 0.0], store, col_name, top_k=1)
        top = results[0]
        assert top.chunk_id == records[0].chunk_id
        assert top.document_id == records[0].document_id
        assert top.filename == "omni_report.pdf"
        assert top.page_number == 1
        assert top.chunk_index == 0
        assert top.content_type == "text"
        assert top.content == "OmniBrain introduction to agents."
        assert isinstance(top.metadata, dict)
        assert top.metadata["char_count"] == 50

    def test_retrieves_table_and_image_modalities(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """Retrieves table and image modality records correctly."""
        store, col_name, records = populated_store
        # Query matching image vector [0.0, 0.0, 1.0, 0.0]
        results = retrieve([0.0, 0.0, 1.0, 0.0], store, col_name, top_k=1)
        assert results[0].chunk_id == records[2].chunk_id
        assert results[0].content_type == "image"
        assert results[0].page_number == 3


# ── Empty & Edge Case Tests ──────────────────────────────────────────────


class TestEmptyAndEdgeCases:
    """Tests for empty collection and edge scenarios."""

    def test_empty_collection_returns_empty_list(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """Querying an empty collection returns empty list."""
        in_memory_store.create_collection("empty_col", vector_dimension=4)
        results = retrieve([1.0, 0.0, 0.0, 0.0], in_memory_store, "empty_col", top_k=5)
        assert results == []


# ── Validation & Error Handling Tests ────────────────────────────────────


class TestValidationAndErrorHandling:
    """Tests for strict query validation and error handling."""

    def test_empty_query_vector_raises(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """Empty query vector raises ValueError."""
        store, col_name, _ = populated_store
        with pytest.raises(ValueError, match="cannot be empty"):
            retrieve([], store, col_name)

    def test_non_list_query_vector_raises(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """Non-list query vector raises TypeError."""
        store, col_name, _ = populated_store
        with pytest.raises(TypeError, match="must be a list"):
            retrieve("invalid_vector", store, col_name)  # type: ignore

    def test_nan_in_query_vector_raises(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """NaN in query vector raises ValueError."""
        store, col_name, _ = populated_store
        with pytest.raises(ValueError, match="invalid non-numeric or non-finite"):
            retrieve([float("nan"), 0.0, 0.0, 0.0], store, col_name)

    def test_infinity_in_query_vector_raises(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """Infinity in query vector raises ValueError."""
        store, col_name, _ = populated_store
        with pytest.raises(ValueError, match="invalid non-numeric or non-finite"):
            retrieve([float("inf"), 0.0, 0.0, 0.0], store, col_name)

    def test_non_numeric_in_query_vector_raises(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """String element in query vector raises ValueError."""
        store, col_name, _ = populated_store
        with pytest.raises(ValueError, match="invalid non-numeric or non-finite"):
            retrieve(["1.0", 0.0, 0.0, 0.0], store, col_name)  # type: ignore

    def test_invalid_top_k_zero_raises(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """top_k <= 0 raises ValueError."""
        store, col_name, _ = populated_store
        with pytest.raises(ValueError, match="positive integer > 0"):
            retrieve([1.0, 0.0, 0.0, 0.0], store, col_name, top_k=0)

    def test_invalid_top_k_negative_raises(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """top_k < 0 raises ValueError."""
        store, col_name, _ = populated_store
        with pytest.raises(ValueError, match="positive integer > 0"):
            retrieve([1.0, 0.0, 0.0, 0.0], store, col_name, top_k=-5)

    def test_invalid_top_k_type_raises(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """Non-int top_k raises ValueError."""
        store, col_name, _ = populated_store
        with pytest.raises(ValueError, match="positive integer > 0"):
            retrieve([1.0, 0.0, 0.0, 0.0], store, col_name, top_k="5")  # type: ignore

    def test_invalid_store_instance_raises(self) -> None:
        """Non-QdrantVectorStore raises TypeError."""
        with pytest.raises(TypeError, match="must be an instance of QdrantVectorStore"):
            retrieve([1.0, 0.0, 0.0, 0.0], "not_a_store", "col")  # type: ignore

    def test_empty_collection_name_raises(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """Empty collection name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            retrieve([1.0, 0.0, 0.0, 0.0], in_memory_store, "")

    def test_missing_collection_raises(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """Querying nonexistent collection raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            retrieve([1.0, 0.0, 0.0, 0.0], in_memory_store, "nonexistent_col")

    def test_dimension_mismatch_raises(
        self, populated_store: tuple[QdrantVectorStore, str, list[EmbeddingVectorRecord]]
    ) -> None:
        """Query vector dimension mismatch with collection raises ValueError."""
        store, col_name, _ = populated_store
        with pytest.raises(ValueError, match="does not match collection dimension"):
            retrieve([1.0, 0.0], store, col_name)
