"""
Tests for Day 19 ingestion pipeline validation and contract hardening.

Covers IngestionValidationResult, chunk contracts, embedding contracts,
search result contracts, lineage consistency, composite pipeline contracts,
and end-to-end integration across all modular layers (Days 1–18).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import pymupdf
import pytest

from ingestion.chunk_validator import normalize_chunks, validate_chunks
from ingestion.chunker import chunk_document
from ingestion.embedding_generator import EmbeddingProvider, generate_embeddings
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.ingestion_config import IngestionConfig
from ingestion.ingestion_health import check_ingestion_readiness
from ingestion.ingestion_logging import IngestionLogger
from ingestion.ingestion_metrics import IngestionMetrics
from ingestion.ingestion_service import run_ingestion
from ingestion.ingestion_status import IngestionStatus, PipelineStatus
from ingestion.ingestion_validation import (
    IngestionValidationResult,
    validate_chunk_contracts,
    validate_embedding_contracts,
    validate_pipeline_contracts,
    validate_pipeline_lineage,
    validate_search_result_contracts,
)
from ingestion.models import (
    ChunkingResult,
    DocumentChunk,
    EmbeddingGenerationResult,
    EmbeddingVectorRecord,
    RetrievalServiceResult,
    VectorSearchResult,
)
from ingestion.pdf_ingestion_pipeline import ingest_pdf
from ingestion.qdrant_config import QdrantConfig
from ingestion.qdrant_store import QdrantVectorStore
from ingestion.retrieval_service import retrieve_context


# ── Helpers & Fixtures ───────────────────────────────────────────────────


class _DeterministicProvider:
    """Deterministic embedding provider for tests."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        base = sum(ord(c) for c in text) % 997
        return [round(((base + i) % 100) / 100.0, 4) for i in range(self.dim)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def provider() -> _DeterministicProvider:
    return _DeterministicProvider(dim=4)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Multi-page sample PDF with text and table content."""
    pdf_path = tmp_path / "integration_validation_sample.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "OmniBrain Day 19 Integration Test - Page 1")
    p1.insert_text((50, 100), "First section describing the ingestion pipeline architecture.")

    p2 = doc.new_page()
    p2.insert_text((50, 50), "OmniBrain Day 19 Integration Test - Page 2")
    p2.insert_text((50, 100), "Second section detailing validation and contract compliance.")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def _make_sample_chunk(
    chunk_id: str = "chunk-1",
    doc_id: str = "doc-123",
    filename: str = "test.pdf",
    chunk_index: int = 0,
    page_number: int | None = 1,
    content_type: str = "text",
    content: str = "Sample content text.",
    metadata: dict[str, Any] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        document_id=doc_id,
        filename=filename,
        page_number=page_number,
        content=content,
        content_type=content_type,
        metadata=metadata if metadata is not None else {},
    )


# ── IngestionValidationResult Dataclass Tests ─────────────────────────────


class TestIngestionValidationResultModel:
    """Tests for IngestionValidationResult properties and helpers."""

    def test_default_validation_result_state(self) -> None:
        res = IngestionValidationResult()
        assert res.is_valid() is True
        assert res.status == "VALID"
        assert res.checks == {}
        assert res.errors == []
        assert res.warnings == []
        assert res.summary == {}

    def test_passed_failed_warning_filters(self) -> None:
        res = IngestionValidationResult(
            checks={
                "c1": "PASS",
                "c2": "FAIL",
                "c3": "WARN",
                "c4": "PASS",
            }
        )
        assert res.passed_checks() == ["c1", "c4"]
        assert res.failed_checks() == ["c2"]
        assert res.warning_checks() == ["c3"]


# ── Chunk Contract Validation Tests ───────────────────────────────────────


class TestChunkContractValidation:
    """Tests for validate_chunk_contracts()."""

    def test_valid_chunks_pass_validation(self) -> None:
        chunks = [
            _make_sample_chunk(chunk_id="c1", chunk_index=0),
            _make_sample_chunk(chunk_id="c2", chunk_index=1, page_number=2),
        ]
        res = validate_chunk_contracts(chunks)
        assert res.is_valid() is True
        assert res.status == "VALID"
        assert res.checks["chunk_attributes_contract"] == "PASS"
        assert res.summary["valid_chunks"] == 2

    def test_chunking_result_container_accepted(self) -> None:
        cr = ChunkingResult(
            document_id="doc-123",
            filename="test.pdf",
            chunks=[_make_sample_chunk()],
        )
        res = validate_chunk_contracts(cr)
        assert res.is_valid() is True

    def test_empty_chunks_produces_warning(self) -> None:
        res = validate_chunk_contracts([])
        assert res.is_valid() is True
        assert res.status == "WARNING"
        assert len(res.warnings) > 0
        assert res.checks["chunks_non_empty"] == "WARN"

    def test_invalid_container_type_fails(self) -> None:
        res = validate_chunk_contracts("invalid_container")
        assert res.is_valid() is False
        assert res.checks["chunk_container_type"] == "FAIL"

    def test_non_document_chunk_element_fails(self) -> None:
        res = validate_chunk_contracts([_make_sample_chunk(), "not_a_chunk"])  # type: ignore[list-item]
        assert res.is_valid() is False
        assert res.checks["chunk_attributes_contract"] == "FAIL"

    def test_empty_chunk_id_fails(self) -> None:
        chunk = _make_sample_chunk(chunk_id="")
        res = validate_chunk_contracts([chunk])
        assert res.is_valid() is False
        assert any("chunk_id" in err for err in res.errors)

    def test_invalid_chunk_index_fails(self) -> None:
        chunk = _make_sample_chunk(chunk_index=-1)
        res = validate_chunk_contracts([chunk])
        assert res.is_valid() is False
        assert any("chunk_index" in err for err in res.errors)

    def test_invalid_page_number_fails(self) -> None:
        chunk = _make_sample_chunk(page_number=0)
        res = validate_chunk_contracts([chunk])
        assert res.is_valid() is False
        assert any("page_number" in err for err in res.errors)

    def test_invalid_content_type_fails(self) -> None:
        chunk = _make_sample_chunk(content_type="audio")
        res = validate_chunk_contracts([chunk])
        assert res.is_valid() is False
        assert any("content_type" in err for err in res.errors)

    def test_empty_content_fails(self) -> None:
        chunk = _make_sample_chunk(content="   ")
        res = validate_chunk_contracts([chunk])
        assert res.is_valid() is False
        assert any("content" in err for err in res.errors)


# ── Embedding Contract Validation Tests ───────────────────────────────────


class TestEmbeddingContractValidation:
    """Tests for validate_embedding_contracts()."""

    def test_valid_embedding_generation_result_passes(self) -> None:
        item = EmbeddingVectorRecord(
            chunk_id="c1",
            document_id="doc-123",
            filename="test.pdf",
            chunk_index=0,
            page_number=1,
            content_type="text",
            vector=[0.1, 0.2, 0.3, 0.4],
        )
        egr = EmbeddingGenerationResult(
            document_id="doc-123",
            filename="test.pdf",
            items=[item],
            dimension=4,
            is_ready=True,
        )
        res = validate_embedding_contracts(egr)
        assert res.is_valid() is True
        assert res.checks["vector_records_contract"] == "PASS"

    def test_invalid_embedding_result_container_fails(self) -> None:
        res = validate_embedding_contracts({"items": []})
        assert res.is_valid() is False
        assert res.checks["embedding_result_type"] == "FAIL"

    def test_invalid_dimension_fails(self) -> None:
        egr = EmbeddingGenerationResult(
            document_id="doc-123",
            filename="test.pdf",
            items=[],
            dimension=0,
            is_ready=False,
        )
        res = validate_embedding_contracts(egr)
        assert res.is_valid() is False
        assert res.checks["embedding_dimension"] == "FAIL"

    def test_vector_dimension_mismatch_fails(self) -> None:
        item = EmbeddingVectorRecord(
            chunk_id="c1",
            document_id="doc-123",
            filename="test.pdf",
            chunk_index=0,
            page_number=1,
            content_type="text",
            vector=[0.1, 0.2],  # Length 2, dimension 4
        )
        egr = EmbeddingGenerationResult(
            document_id="doc-123",
            filename="test.pdf",
            items=[item],
            dimension=4,
            is_ready=True,
        )
        res = validate_embedding_contracts(egr)
        assert res.is_valid() is False
        assert any("dimension mismatch" in err for err in res.errors)

    def test_vector_with_nan_fails(self) -> None:
        item = EmbeddingVectorRecord(
            chunk_id="c1",
            document_id="doc-123",
            filename="test.pdf",
            chunk_index=0,
            page_number=1,
            content_type="text",
            vector=[0.1, float("nan"), 0.3, 0.4],
        )
        egr = EmbeddingGenerationResult(
            document_id="doc-123",
            filename="test.pdf",
            items=[item],
            dimension=4,
            is_ready=True,
        )
        res = validate_embedding_contracts(egr)
        assert res.is_valid() is False
        assert any("invalid numeric value" in err for err in res.errors)


# ── Search Result Contract Validation Tests ───────────────────────────────


class TestSearchResultContractValidation:
    """Tests for validate_search_result_contracts()."""

    def test_valid_search_results_pass(self) -> None:
        sr = VectorSearchResult(
            chunk_id="c1",
            score=0.95,
            document_id="doc-123",
            filename="test.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Retrieved sample text.",
        )
        res = validate_search_result_contracts([sr])
        assert res.is_valid() is True
        assert res.checks["search_result_contract"] == "PASS"

    def test_retrieval_service_result_container_accepted(self) -> None:
        sr = VectorSearchResult(
            chunk_id="c1",
            score=0.95,
            document_id="doc-123",
            filename="test.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Retrieved text.",
        )
        rsr = RetrievalServiceResult(
            query_vector_dimension=4,
            results=[sr],
            context="Formatted citation context",
        )
        res = validate_search_result_contracts(rsr)
        assert res.is_valid() is True

    def test_empty_search_results_produces_warning(self) -> None:
        res = validate_search_result_contracts([])
        assert res.is_valid() is True
        assert res.status == "WARNING"

    def test_invalid_score_fails(self) -> None:
        sr = VectorSearchResult(
            chunk_id="c1",
            score=float("nan"),
            document_id="doc-123",
            filename="test.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Retrieved text.",
        )
        res = validate_search_result_contracts([sr])
        assert res.is_valid() is False
        assert any("invalid score" in err for err in res.errors)

    def test_non_string_content_fails(self) -> None:
        sr = VectorSearchResult(
            chunk_id="c1",
            score=0.8,
            document_id="doc-123",
            filename="test.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content=12345,  # type: ignore[arg-type]
        )
        res = validate_search_result_contracts([sr])
        assert res.is_valid() is False
        assert any("content must be a string" in err for err in res.errors)


# ── Lineage Validation Tests ──────────────────────────────────────────────


class TestLineageValidation:
    """Tests for validate_pipeline_lineage()."""

    def test_matching_lineage_passes(self) -> None:
        chunks = [
            _make_sample_chunk(chunk_id="c1", doc_id="doc-123", filename="doc.pdf"),
            _make_sample_chunk(chunk_id="c2", doc_id="doc-123", filename="doc.pdf"),
        ]
        res = validate_pipeline_lineage("doc-123", "doc.pdf", chunks)
        assert res.is_valid() is True
        assert res.checks["lineage_consistency"] == "PASS"

    def test_document_id_mismatch_fails(self) -> None:
        chunks = [
            _make_sample_chunk(chunk_id="c1", doc_id="doc-123", filename="doc.pdf"),
            _make_sample_chunk(chunk_id="c2", doc_id="doc-OTHER", filename="doc.pdf"),
        ]
        res = validate_pipeline_lineage("doc-123", "doc.pdf", chunks)
        assert res.is_valid() is False
        assert any("document_id mismatch" in err for err in res.errors)

    def test_filename_mismatch_fails(self) -> None:
        chunks = [
            _make_sample_chunk(chunk_id="c1", doc_id="doc-123", filename="doc.pdf"),
            _make_sample_chunk(chunk_id="c2", doc_id="doc-123", filename="other.pdf"),
        ]
        res = validate_pipeline_lineage("doc-123", "doc.pdf", chunks)
        assert res.is_valid() is False
        assert any("filename mismatch" in err for err in res.errors)


# ── Composite Contract Validation Tests ───────────────────────────────────


class TestCompositeContractValidation:
    """Tests for validate_pipeline_contracts()."""

    def test_composite_validation_all_valid_artifacts(self) -> None:
        chunk = _make_sample_chunk(chunk_id="c1", doc_id="doc-123", filename="doc.pdf")
        vec = EmbeddingVectorRecord(
            chunk_id="c1",
            document_id="doc-123",
            filename="doc.pdf",
            chunk_index=0,
            page_number=1,
            content_type="text",
            vector=[0.1, 0.2, 0.3, 0.4],
        )
        egr = EmbeddingGenerationResult(
            document_id="doc-123",
            filename="doc.pdf",
            items=[vec],
            dimension=4,
            is_ready=True,
        )
        sr = VectorSearchResult(
            chunk_id="c1",
            score=0.88,
            document_id="doc-123",
            filename="doc.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Retrieved content",
        )

        res = validate_pipeline_contracts(
            chunks=[chunk],
            embedding_result=egr,
            search_results=[sr],
            source_document_id="doc-123",
            source_filename="doc.pdf",
        )
        assert res.is_valid() is True
        assert res.checks["chunk_attributes_contract"] == "PASS"
        assert res.checks["vector_records_contract"] == "PASS"
        assert res.checks["search_result_contract"] == "PASS"
        assert res.checks["lineage_consistency"] == "PASS"


# ── Full End-to-End Pipeline Contract Validation ─────────────────────────


class TestEndToEndPipelineIntegrationAndValidation:
    """Full end-to-end integration test validating all Day 1–18 components working together."""

    def test_end_to_end_flow_with_all_components(
        self, sample_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        # 1. Day 15 Config
        config = IngestionConfig(chunk_size=500, chunk_overlap=50)

        # 2. Day 18 Readiness Check
        readiness = check_ingestion_readiness(provider=provider, config=config)
        assert readiness.is_ready() is True

        # 3. Day 14 Status, Day 16 Metrics, Day 17 Logger
        tracker = IngestionStatus()
        metrics = IngestionMetrics()
        logger = IngestionLogger(logging.getLogger("test.day19_full_flow"))

        # 4. Day 13 Ingestion Execution
        generation_result = run_ingestion(
            pdf_path=sample_pdf,
            embedding_provider=provider,
            config=config,
            status_tracker=tracker,
            metrics=metrics,
            logger=logger,
        )

        assert tracker.status == PipelineStatus.COMPLETED
        assert metrics.status == "COMPLETED"
        assert generation_result.is_ready is True
        assert generation_result.dimension == provider.dim

        # 5. Day 9 & 10 Qdrant Vector Store & Search
        store = QdrantVectorStore()
        collection_name = "test_day19_validation"
        store.create_collection(collection_name, vector_dimension=provider.dim)
        upserted_count = store.upsert_embeddings(collection_name, generation_result)
        assert upserted_count == generation_result.total_items

        # 6. Day 12 Retrieval Service
        query_vector = provider.embed("pipeline architecture")
        retrieval_res = retrieve_context(
            query_vector=query_vector,
            store=store,
            collection_name=collection_name,
            top_k=3,
        )
        assert retrieval_res.has_results is True

        # 7. Day 19 Contract & Lineage Validation across artifacts
        validation = validate_pipeline_contracts(
            embedding_result=generation_result,
            search_results=retrieval_res,
            source_document_id=generation_result.document_id,
            source_filename=generation_result.filename,
        )
        assert validation.is_valid() is True
        assert validation.checks["vector_records_contract"] == "PASS"
        assert validation.checks["search_result_contract"] == "PASS"
        assert validation.checks["lineage_consistency"] == "PASS"

    def test_backward_compatibility_run_ingestion_minimal(
        self, sample_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """Verify run_ingestion works with minimal arguments (backward compatible with Day 13)."""
        res = run_ingestion(sample_pdf, provider)
        assert res.is_ready is True
        assert res.total_items > 0
