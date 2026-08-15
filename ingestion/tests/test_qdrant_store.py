"""
Tests for the Qdrant vector store integration.

Uses in-memory Qdrant client (:memory:) to thoroughly test collection management,
vector upserting, metadata/citation preservation, and cosine similarity search.
"""

from __future__ import annotations

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


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def in_memory_store() -> QdrantVectorStore:
    """Create a QdrantVectorStore backed by an in-memory client."""
    config = QdrantConfig(url=":memory:")
    return QdrantVectorStore(config=config)


@pytest.fixture
def sample_doc_id() -> str:
    """Create sample document UUID."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_vector_records(sample_doc_id: str) -> list[EmbeddingVectorRecord]:
    """Create sample list of 4-dimensional EmbeddingVectorRecord objects."""
    return [
        EmbeddingVectorRecord(
            chunk_id=str(uuid.uuid4()),
            document_id=sample_doc_id,
            filename="report.pdf",
            chunk_index=0,
            page_number=1,
            content_type="text",
            vector=[1.0, 0.0, 0.0, 0.0],
            metadata={"char_count": 55, "section": "intro", "content": "OmniBrain intro."},
        ),
        EmbeddingVectorRecord(
            chunk_id=str(uuid.uuid4()),
            document_id=sample_doc_id,
            filename="report.pdf",
            chunk_index=1,
            page_number=1,
            content_type="table",
            vector=[0.0, 1.0, 0.0, 0.0],
            metadata={"table_index": 0, "rows": 2, "content": "| Table 1 |"},
        ),
        EmbeddingVectorRecord(
            chunk_id=str(uuid.uuid4()),
            document_id=sample_doc_id,
            filename="report.pdf",
            chunk_index=2,
            page_number=2,
            content_type="image",
            vector=[0.0, 0.0, 1.0, 0.0],
            metadata={"image_index": 0, "width": 640, "content": "[Image 1]"},
        ),
    ]


@pytest.fixture
def sample_generation_result(
    sample_vector_records: list[EmbeddingVectorRecord], sample_doc_id: str
) -> EmbeddingGenerationResult:
    """Create sample EmbeddingGenerationResult."""
    return EmbeddingGenerationResult(
        document_id=sample_doc_id,
        filename="report.pdf",
        items=sample_vector_records,
        dimension=4,
        is_ready=True,
    )


# ── Initialization Tests ─────────────────────────────────────────────────


class TestClientInitialization:
    """Tests for store initialization and configuration."""

    def test_in_memory_initialization(self, in_memory_store: QdrantVectorStore) -> None:
        """Store initializes with in-memory client without external dependencies."""
        assert in_memory_store.client is not None

    def test_custom_client_injection(self) -> None:
        """Store accepts pre-configured QdrantClient instance."""
        custom_client = QdrantClient(location=":memory:")
        store = QdrantVectorStore(client=custom_client)
        assert store.client is custom_client

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """QdrantConfig parses environment variables accurately."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "secret-test-key")
        monkeypatch.setenv("QDRANT_COLLECTION", "custom_collection")
        monkeypatch.setenv("QDRANT_TIMEOUT", "15.5")

        config = QdrantConfig.from_env()
        assert config.url == "http://localhost:6333"
        assert config.api_key == "secret-test-key"
        assert config.default_collection == "custom_collection"
        assert config.timeout == 15.5


# ── Collection Management Tests ──────────────────────────────────────────


class TestCollectionManagement:
    """Tests for collection creation, lookup, inspection, and deletion."""

    def test_create_and_exists_collection(self, in_memory_store: QdrantVectorStore) -> None:
        """Creating a collection makes collection_exists return True."""
        assert in_memory_store.collection_exists("test_col") is False
        in_memory_store.create_collection("test_col", vector_dimension=4)
        assert in_memory_store.collection_exists("test_col") is True

    def test_get_collection_info(self, in_memory_store: QdrantVectorStore) -> None:
        """get_collection_info returns metadata and vector dimension."""
        in_memory_store.create_collection("info_col", vector_dimension=128)
        info = in_memory_store.get_collection_info("info_col")
        assert info["collection_name"] == "info_col"
        assert info["vector_dimension"] == 128
        assert info["distance"].lower() == "cosine"
        assert info["points_count"] == 0

    def test_delete_collection(self, in_memory_store: QdrantVectorStore) -> None:
        """delete_collection removes the collection completely."""
        in_memory_store.create_collection("del_col", vector_dimension=4)
        assert in_memory_store.collection_exists("del_col") is True
        in_memory_store.delete_collection("del_col")
        assert in_memory_store.collection_exists("del_col") is False

    def test_invalid_collection_name_raises(self, in_memory_store: QdrantVectorStore) -> None:
        """Empty collection name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            in_memory_store.create_collection("", vector_dimension=4)

    def test_invalid_dimension_raises(self, in_memory_store: QdrantVectorStore) -> None:
        """Non-positive dimension raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            in_memory_store.create_collection("bad_dim", vector_dimension=0)


# ── Vector Upsert Tests ──────────────────────────────────────────────────


class TestVectorUpsert:
    """Tests for upserting embeddings into Qdrant."""

    def test_successful_upsert(
        self,
        in_memory_store: QdrantVectorStore,
        sample_generation_result: EmbeddingGenerationResult,
    ) -> None:
        """Upserting returns count of points and updates collection points_count."""
        in_memory_store.create_collection("upsert_col", vector_dimension=4)
        count = in_memory_store.upsert_embeddings("upsert_col", sample_generation_result)
        assert count == 3
        info = in_memory_store.get_collection_info("upsert_col")
        assert info["points_count"] == 3

    def test_empty_generation_result(self, in_memory_store: QdrantVectorStore) -> None:
        """Empty result returns 0 and does not error."""
        in_memory_store.create_collection("empty_col", vector_dimension=4)
        empty_res = EmbeddingGenerationResult(
            document_id="doc1",
            filename="a.pdf",
            items=[],
            dimension=0,
            is_ready=True,
        )
        assert in_memory_store.upsert_embeddings("empty_col", empty_res) == 0

    def test_upsert_nonexistent_collection_raises(
        self,
        in_memory_store: QdrantVectorStore,
        sample_generation_result: EmbeddingGenerationResult,
    ) -> None:
        """Upserting to missing collection raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            in_memory_store.upsert_embeddings("missing_col", sample_generation_result)

    def test_dimension_mismatch_raises(
        self,
        in_memory_store: QdrantVectorStore,
        sample_generation_result: EmbeddingGenerationResult,
    ) -> None:
        """Mismatched vector dimension raises ValueError."""
        in_memory_store.create_collection("dim_mismatch_col", vector_dimension=8)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            in_memory_store.upsert_embeddings("dim_mismatch_col", sample_generation_result)

    def test_duplicate_upsert_overwrites(
        self,
        in_memory_store: QdrantVectorStore,
        sample_generation_result: EmbeddingGenerationResult,
    ) -> None:
        """Upserting the same points twice overwrites points without duplicating count."""
        in_memory_store.create_collection("dup_col", vector_dimension=4)
        in_memory_store.upsert_embeddings("dup_col", sample_generation_result)
        in_memory_store.upsert_embeddings("dup_col", sample_generation_result)
        info = in_memory_store.get_collection_info("dup_col")
        assert info["points_count"] == 3


# ── Similarity Search Tests ──────────────────────────────────────────────


class TestSimilaritySearch:
    """Tests for vector similarity search and citation metadata retrieval."""

    def test_similarity_search_ordering(
        self,
        in_memory_store: QdrantVectorStore,
        sample_generation_result: EmbeddingGenerationResult,
    ) -> None:
        """Search returns most similar vector with highest score first."""
        in_memory_store.create_collection("search_col", vector_dimension=4)
        in_memory_store.upsert_embeddings("search_col", sample_generation_result)

        # Query vector identical to item 0 ([1.0, 0.0, 0.0, 0.0])
        results = in_memory_store.search("search_col", query_vector=[1.0, 0.0, 0.0, 0.0], limit=3)
        assert len(results) == 3
        top_match = results[0]
        assert top_match["chunk_id"] == sample_generation_result.items[0].chunk_id
        assert top_match["score"] >= 0.99
        assert top_match["content_type"] == "text"
        assert top_match["page_number"] == 1
        assert top_match["filename"] == "report.pdf"

    def test_search_limit(
        self,
        in_memory_store: QdrantVectorStore,
        sample_generation_result: EmbeddingGenerationResult,
    ) -> None:
        """Limit parameter bounds the number of returned results."""
        in_memory_store.create_collection("limit_col", vector_dimension=4)
        in_memory_store.upsert_embeddings("limit_col", sample_generation_result)

        results = in_memory_store.search("limit_col", query_vector=[1.0, 0.0, 0.0, 0.0], limit=1)
        assert len(results) == 1

    def test_search_records_returns_dataclasses(
        self,
        in_memory_store: QdrantVectorStore,
        sample_generation_result: EmbeddingGenerationResult,
    ) -> None:
        """search_records returns list of VectorSearchResult instances."""
        in_memory_store.create_collection("records_col", vector_dimension=4)
        in_memory_store.upsert_embeddings("records_col", sample_generation_result)

        records = in_memory_store.search_records("records_col", query_vector=[0.0, 1.0, 0.0, 0.0], limit=2)
        assert len(records) == 2
        assert isinstance(records[0], VectorSearchResult)
        assert records[0].content_type == "table"
        assert records[0].chunk_id == sample_generation_result.items[1].chunk_id

    def test_search_dimension_mismatch_raises(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """Searching with wrong query vector dimension raises ValueError."""
        in_memory_store.create_collection("query_dim_col", vector_dimension=4)
        with pytest.raises(ValueError, match="does not match collection dimension"):
            in_memory_store.search("query_dim_col", query_vector=[1.0, 2.0])

    def test_search_empty_query_raises(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """Searching with empty query vector raises ValueError."""
        in_memory_store.create_collection("empty_query_col", vector_dimension=4)
        with pytest.raises(ValueError, match="non-empty list"):
            in_memory_store.search("empty_query_col", query_vector=[])
