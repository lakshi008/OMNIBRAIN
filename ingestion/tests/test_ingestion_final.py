"""
Day 20 Final Regression, Integration & Delivery Readiness Test Suite.

Performs final hardening and quality validation across all modular components
of the OmniBrain Ingestion subsystem (Days 1–19).
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
import pymupdf
import pytest

import ingestion
from ingestion.chunk_validator import normalize_chunks, validate_chunks
from ingestion.chunker import chunk_document
from ingestion.embedding_generator import EmbeddingProvider, generate_embeddings
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
)
from ingestion.ingestion_config import IngestionConfig
from ingestion.ingestion_errors import (
    IngestionChunkingError,
    IngestionEmbeddingError,
    IngestionError,
    IngestionExtractionError,
    IngestionPipelineError,
    IngestionValidationError,
)
from ingestion.ingestion_health import (
    IngestionHealthResult,
    check_ingestion_health,
    check_ingestion_readiness,
)
from ingestion.ingestion_logging import IngestionLogger, get_ingestion_logger
from ingestion.ingestion_metrics import IngestionMetrics, StageMetrics
from ingestion.ingestion_service import run_ingestion
from ingestion.ingestion_status import (
    IngestionStatus,
    PipelineStage,
    PipelineStatus,
)
from ingestion.ingestion_validation import (
    IngestionValidationResult,
    validate_chunk_contracts,
    validate_embedding_contracts,
    validate_pipeline_contracts,
    validate_pipeline_lineage,
    validate_search_result_contracts,
)
from ingestion.models import (
    ChunkValidationResult,
    ChunkingResult,
    DocumentChunk,
    DocumentMetadata,
    EmbeddingGenerationResult,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    EmbeddingVectorRecord,
    ExtractedImage,
    ExtractedTable,
    ImageExtractionResult,
    IngestionResult,
    PageData,
    ParsedDocument,
    RetrievalServiceResult,
    TableExtractionResult,
    VectorSearchResult,
)
from ingestion.pdf_image_extractor import extract_images
from ingestion.pdf_ingestion_pipeline import ingest_pdf
from ingestion.pdf_table_extractor import extract_tables
from ingestion.pdf_text_extractor import extract_text, validate_pdf
from ingestion.qdrant_config import QdrantConfig
from ingestion.qdrant_store import QdrantVectorStore
from ingestion.retrieval import retrieve
from ingestion.retrieval_processor import (
    build_retrieval_context,
    process_retrieval_results,
)
from ingestion.retrieval_service import retrieve_context


# ── Fixtures & Deterministic Providers ───────────────────────────────────


class _DeterministicProvider:
    """Deterministic, pure-Python embedding provider for testing."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        base = sum(ord(c) for c in text) % 997
        return [round(((base * (i + 1)) % 100) / 100.0, 4) for i in range(self.dim)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _DimensionMismatchProvider:
    """Provider returning vectors with alternating length to trigger errors."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] if i == 0 else [0.1] for i in range(len(texts))]


@pytest.fixture
def provider() -> _DeterministicProvider:
    return _DeterministicProvider(dim=4)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Multi-page sample PDF with text and a formatted table."""
    pdf_path = tmp_path / "final_delivery_sample.pdf"
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text((50, 50), "OmniBrain Final Hardening Test Document")
    p1.insert_text((50, 100), "This document tests multi-modal ingestion and retrieval capabilities.")

    p2 = doc.new_page()
    p2.insert_text((50, 50), "Section 2: Verification and Delivery Readiness")
    p2.insert_text((50, 100), "All Day 1 through Day 19 components are validated.")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ── 1. Package & Exports Validation ──────────────────────────────────────


class TestPackageAndPublicExports:
    """Verifies that ingestion package exports are complete, clean, and without duplicates."""

    def test_import_ingestion_package(self) -> None:
        assert ingestion is not None

    def test_all_exports_present_on_package(self) -> None:
        for symbol in ingestion.__all__:
            assert hasattr(ingestion, symbol), f"Missing public symbol in ingestion: {symbol}"

    def test_no_duplicate_exports_in_all(self) -> None:
        assert len(ingestion.__all__) == len(set(ingestion.__all__)), "Duplicate exports found in __all__"

    def test_all_modules_importable(self) -> None:
        modules = [
            "ingestion.chunk_validator",
            "ingestion.chunker",
            "ingestion.embedding_generator",
            "ingestion.embedding_preparation",
            "ingestion.exceptions",
            "ingestion.ingestion_config",
            "ingestion.ingestion_errors",
            "ingestion.ingestion_health",
            "ingestion.ingestion_logging",
            "ingestion.ingestion_metrics",
            "ingestion.ingestion_service",
            "ingestion.ingestion_status",
            "ingestion.ingestion_validation",
            "ingestion.models",
            "ingestion.pdf_image_extractor",
            "ingestion.pdf_ingestion_pipeline",
            "ingestion.pdf_parser",
            "ingestion.pdf_table_extractor",
            "ingestion.pdf_text_extractor",
            "ingestion.qdrant_config",
            "ingestion.qdrant_store",
            "ingestion.retrieval",
            "ingestion.retrieval_processor",
            "ingestion.retrieval_service",
        ]
        for mod in modules:
            imported = importlib.import_module(mod)
            assert imported is not None


# ── 2. Backward Compatibility & API Signatures ───────────────────────────


class TestAPIBackwardCompatibility:
    """Verifies that run_ingestion() works with original minimal parameters as well as full suite."""

    def test_run_ingestion_minimal_signature(
        self, sample_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        result = run_ingestion(sample_pdf, provider)
        assert isinstance(result, EmbeddingGenerationResult)
        assert result.is_ready is True
        assert result.total_items > 0
        assert result.dimension == provider.dim

    def test_run_ingestion_custom_chunking_args(
        self, sample_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        result = run_ingestion(sample_pdf, provider, chunk_size=300, chunk_overlap=30)
        assert result.is_ready is True
        assert result.total_items > 0

    def test_run_ingestion_full_composable_options(
        self, sample_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        config = IngestionConfig(chunk_size=400, chunk_overlap=40)
        status_tracker = IngestionStatus()
        metrics = IngestionMetrics()
        logger = IngestionLogger(logging.getLogger("test.final_composable"))

        result = run_ingestion(
            pdf_path=sample_pdf,
            embedding_provider=provider,
            config=config,
            status_tracker=status_tracker,
            metrics=metrics,
            logger=logger,
        )
        assert result.is_ready is True
        assert status_tracker.status == PipelineStatus.COMPLETED
        assert metrics.status == "COMPLETED"
        assert metrics.total_vectors == result.total_items


# ── 3. Configuration Subsystem (Day 15) ───────────────────────────────────


class TestConfigurationSubsystem:
    """Verifies IngestionConfig validation, defaults, and env loading."""

    def test_default_config_values(self) -> None:
        cfg = IngestionConfig()
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.retrieval_top_k == 5
        assert cfg.retrieval_min_score == 0.0
        assert cfg.qdrant_collection == "omnibrain_documents"
        assert cfg.qdrant_timeout == 10.0

    def test_invalid_overlap_raises_error(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            IngestionConfig(chunk_size=200, chunk_overlap=200)

    def test_from_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INGESTION_CHUNK_SIZE", "600")
        monkeypatch.setenv("INGESTION_CHUNK_OVERLAP", "60")
        monkeypatch.setenv("QDRANT_COLLECTION", "custom_docs")
        cfg = IngestionConfig.from_env()
        assert cfg.chunk_size == 600
        assert cfg.chunk_overlap == 60
        assert cfg.qdrant_collection == "custom_docs"


# ── 4. Observability Subsystem (Days 14, 16, 17) ──────────────────────────


class TestObservabilitySubsystem:
    """Verifies Status, Metrics, and Logging coexist without interfering with pipeline results."""

    def test_status_tracker_lifecycle(self) -> None:
        tracker = IngestionStatus()
        assert tracker.status == PipelineStatus.PENDING
        tracker.start(PipelineStage.EXTRACTION)
        assert tracker.status == PipelineStatus.RUNNING
        assert tracker.current_stage == PipelineStage.EXTRACTION
        tracker.advance_stage(PipelineStage.CHUNKING)
        assert tracker.current_stage == PipelineStage.CHUNKING
        tracker.complete()
        assert tracker.status == PipelineStatus.COMPLETED

    def test_metrics_stage_and_counter_tracking(
        self, sample_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        metrics = IngestionMetrics()
        run_ingestion(sample_pdf, provider, metrics=metrics)
        assert metrics.status == "COMPLETED"
        assert metrics.total_duration_seconds > 0.0
        assert metrics.total_chunks > 0
        assert metrics.total_vectors == metrics.total_chunks
        assert len(metrics.stage_metrics) >= 6

    def test_logging_safety_and_event_records(
        self, sample_pdf: Path, provider: _DeterministicProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = IngestionLogger(logging.getLogger("test.final_logging"))
        with caplog.at_level(logging.DEBUG, logger="test.final_logging"):
            run_ingestion(sample_pdf, provider, logger=logger)
        messages = [r.message for r in caplog.records if r.name == "test.final_logging"]
        assert any("ingestion_start" in m for m in messages)
        assert any("stage_start" in m for m in messages)
        assert any("stage_complete" in m for m in messages)
        assert any("ingestion_complete" in m for m in messages)


# ── 5. Health & Readiness Subsystem (Day 18) ──────────────────────────────


class TestHealthAndReadinessSubsystem:
    """Verifies health check and readiness check isolation and safety."""

    def test_health_check_passes_with_defaults(self) -> None:
        health = check_ingestion_health()
        assert health.is_healthy() is True
        assert health.status == "READY"
        assert len(health.errors) == 0

    def test_readiness_check_fails_without_provider(self) -> None:
        readiness = check_ingestion_readiness(provider=None)
        assert readiness.is_healthy() is True
        assert readiness.is_ready() is False
        assert readiness.status == "HEALTHY"
        assert len(readiness.errors) > 0

    def test_readiness_check_passes_with_valid_provider(
        self, provider: _DeterministicProvider
    ) -> None:
        readiness = check_ingestion_readiness(provider=provider)
        assert readiness.is_healthy() is True
        assert readiness.is_ready() is True
        assert readiness.status == "READY"


# ── 6. End-to-End Retrieval Pipeline (Days 9–12) ─────────────────────────


class TestEndToEndRetrievalPipeline:
    """Verifies complete flow from document embedding to retrieval service context output."""

    def test_full_retrieval_flow(
        self, sample_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        # Ingest document
        gen_result = run_ingestion(sample_pdf, provider)
        assert gen_result.is_ready is True

        # Index in Qdrant store
        store = QdrantVectorStore()
        col_name = "test_final_delivery_col"
        store.create_collection(col_name, vector_dimension=provider.dim)
        upserted = store.upsert_embeddings(col_name, gen_result)
        assert upserted == gen_result.total_items

        # Retrieve and build context
        query_vec = provider.embed("OmniBrain Delivery Readiness")
        service_res = retrieve_context(
            query_vector=query_vec,
            store=store,
            collection_name=col_name,
            top_k=3,
        )

        assert isinstance(service_res, RetrievalServiceResult)
        assert service_res.has_results is True
        assert len(service_res.results) <= 3
        assert len(service_res.context) > 0
        # Lineage preserved in search results
        for r in service_res.results:
            assert r.document_id == gen_result.document_id
            assert r.filename == gen_result.filename


# ── 7. Error Handling & Isolation (Day 14) ────────────────────────────────


class TestErrorHandlingAndIsolation:
    """Verifies structured exceptions and failure propagation across all stages."""

    def test_missing_pdf_raises_not_found(self, tmp_path: Path, provider: _DeterministicProvider) -> None:
        tracker = IngestionStatus()
        metrics = IngestionMetrics()
        with pytest.raises((PDFNotFoundError, IngestionExtractionError)):
            run_ingestion(
                pdf_path=tmp_path / "non_existent_file.pdf",
                embedding_provider=provider,
                status_tracker=tracker,
                metrics=metrics,
            )
        assert tracker.status == PipelineStatus.FAILED
        assert metrics.status == "FAILED"

    def test_invalid_chunk_size_raises_validation_error(
        self, sample_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        with pytest.raises(IngestionValidationError):
            run_ingestion(sample_pdf, provider, chunk_size=-10)

    def test_invalid_provider_raises_type_error(self, sample_pdf: Path) -> None:
        with pytest.raises(TypeError, match="Invalid embedding provider"):
            run_ingestion(sample_pdf, embedding_provider="not_a_provider")  # type: ignore[arg-type]

    def test_broken_provider_raises_embedding_error(self, sample_pdf: Path) -> None:
        broken = _DimensionMismatchProvider()
        with pytest.raises(IngestionEmbeddingError):
            run_ingestion(sample_pdf, embedding_provider=broken)


# ── 8. Contract & Lineage Validation (Day 19) ─────────────────────────────


class TestContractAndLineageValidation:
    """Verifies pipeline contract validation across document chunks and vectors."""

    def test_validate_pipeline_contracts_end_to_end(
        self, sample_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        gen_result = run_ingestion(sample_pdf, provider)
        validation = validate_pipeline_contracts(
            embedding_result=gen_result,
            source_document_id=gen_result.document_id,
            source_filename=gen_result.filename,
        )
        assert validation.is_valid() is True
        assert validation.checks["embedding_result_type"] == "PASS"
        assert validation.checks["vector_records_contract"] == "PASS"
        assert validation.checks["lineage_consistency"] == "PASS"
