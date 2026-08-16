"""
Tests for the IngestionMetrics and StageMetrics classes.

Covers basic metrics construction, stage timing context manager,
counter population, pipeline integration, and regression protection
for Day 14 error/status and Day 15 configuration behavior.
"""

from __future__ import annotations

import time
from pathlib import Path

import pymupdf
import pytest

from ingestion.ingestion_metrics import IngestionMetrics, StageMetrics
from ingestion.ingestion_service import run_ingestion


# ── Helpers ───────────────────────────────────────────────────────────────


class _DeterministicProvider:
    """Deterministic 6-dimensional embedding provider for integration tests."""

    def embed(self, text: str) -> list[float]:
        base = sum(ord(c) for c in text) % 997
        return [round((base * (i + 1)) % 1000 / 1000.0, 4) for i in range(6)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _BrokenProvider:
    """Provider that emits inconsistent vector dimensions to trigger failures."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0] if i == 0 else [1.0] for i in range(len(texts))]


@pytest.fixture
def provider() -> _DeterministicProvider:
    return _DeterministicProvider()


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    """Two-page text-only PDF for integration tests."""
    pdf_path = tmp_path / "metrics_test.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 100), "OmniBrain ingestion metrics test page one content.")
    p2 = doc.new_page()
    p2.insert_text((50, 100), "OmniBrain ingestion metrics test page two content.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ── StageMetrics Tests ────────────────────────────────────────────────────


class TestStageMetrics:
    """Tests for the StageMetrics dataclass."""

    def test_default_stage_metrics_creation(self) -> None:
        """StageMetrics can be created with just a stage name."""
        sm = StageMetrics(stage="EXTRACTION")
        assert sm.stage == "EXTRACTION"
        assert sm.duration_seconds == 0.0
        assert sm.success is False
        assert sm.error is None

    def test_successful_stage_metrics(self) -> None:
        """StageMetrics correctly reflects a successful stage."""
        sm = StageMetrics(stage="CHUNKING", duration_seconds=0.5, success=True)
        assert sm.success is True
        assert sm.duration_seconds == 0.5
        assert sm.error is None

    def test_failed_stage_metrics(self) -> None:
        """StageMetrics correctly reflects a failed stage with an error message."""
        sm = StageMetrics(
            stage="EMBEDDING_GENERATION",
            duration_seconds=0.1,
            success=False,
            error="Provider dimension mismatch",
        )
        assert sm.success is False
        assert sm.error == "Provider dimension mismatch"

    def test_negative_duration_raises(self) -> None:
        """StageMetrics raises ValueError for negative duration."""
        with pytest.raises(ValueError, match="duration_seconds"):
            StageMetrics(stage="CHUNKING", duration_seconds=-0.001)


# ── IngestionMetrics Basic Tests ──────────────────────────────────────────


class TestIngestionMetricsBasic:
    """Tests for the default IngestionMetrics state and field initialization."""

    def test_default_metrics_creation(self) -> None:
        """IngestionMetrics can be created with all defaults."""
        m = IngestionMetrics()
        assert m is not None

    def test_initial_status_is_pending(self) -> None:
        m = IngestionMetrics()
        assert m.status == "PENDING"

    def test_initial_document_id_empty(self) -> None:
        m = IngestionMetrics()
        assert m.document_id == ""

    def test_initial_filename_empty(self) -> None:
        m = IngestionMetrics()
        assert m.filename == ""

    def test_initial_stage_metrics_empty(self) -> None:
        m = IngestionMetrics()
        assert m.stage_metrics == []

    def test_initial_counters_are_zero(self) -> None:
        m = IngestionMetrics()
        assert m.total_chunks == 0
        assert m.text_chunks == 0
        assert m.table_chunks == 0
        assert m.image_chunks == 0
        assert m.total_embedding_items == 0
        assert m.total_vectors == 0

    def test_start_pipeline_sets_running(self) -> None:
        m = IngestionMetrics()
        m.start_pipeline()
        assert m.status == "RUNNING"

    def test_finish_pipeline_success_sets_completed(self) -> None:
        m = IngestionMetrics()
        m.start_pipeline()
        m.finish_pipeline(success=True)
        assert m.status == "COMPLETED"
        assert m.succeeded is True
        assert m.failed is False

    def test_finish_pipeline_failure_sets_failed(self) -> None:
        m = IngestionMetrics()
        m.start_pipeline()
        m.finish_pipeline(success=False, error="Something broke")
        assert m.status == "FAILED"
        assert m.failed is True
        assert m.error == "Something broke"

    def test_total_duration_is_non_negative(self) -> None:
        m = IngestionMetrics()
        m.start_pipeline()
        m.finish_pipeline(success=True)
        assert m.total_duration_seconds >= 0.0


# ── Stage Timing Tests ────────────────────────────────────────────────────


class TestStageTimingContextManager:
    """Tests for IngestionMetrics.track_stage() context manager."""

    def test_successful_stage_timing(self) -> None:
        """A successful stage yields a StageMetrics with success=True and duration >= 0."""
        m = IngestionMetrics()
        m.start_pipeline()
        with m.track_stage("CHUNKING"):
            time.sleep(0.01)
        assert len(m.stage_metrics) == 1
        sm = m.stage_metrics[0]
        assert sm.stage == "CHUNKING"
        assert sm.success is True
        assert sm.duration_seconds >= 0.0
        assert sm.error is None

    def test_failed_stage_timing_records_error(self) -> None:
        """An exception in track_stage marks the stage as failed and propagates."""
        m = IngestionMetrics()
        m.start_pipeline()
        with pytest.raises(RuntimeError, match="boom"):
            with m.track_stage("EXTRACTION"):
                raise RuntimeError("boom")
        assert len(m.stage_metrics) == 1
        sm = m.stage_metrics[0]
        assert sm.stage == "EXTRACTION"
        assert sm.success is False
        assert "boom" in sm.error

    def test_stage_duration_is_non_negative_on_failure(self) -> None:
        """Duration is always >= 0 even when a stage raises immediately."""
        m = IngestionMetrics()
        m.start_pipeline()
        with pytest.raises(ValueError):
            with m.track_stage("VALIDATION"):
                raise ValueError("bad input")
        assert m.stage_metrics[0].duration_seconds >= 0.0

    def test_multiple_stages_recorded_in_order(self) -> None:
        """Multiple track_stage calls produce stages in execution order."""
        m = IngestionMetrics()
        m.start_pipeline()
        with m.track_stage("EXTRACTION"):
            pass
        with m.track_stage("CHUNKING"):
            pass
        with m.track_stage("NORMALIZATION"):
            pass
        assert [sm.stage for sm in m.stage_metrics] == [
            "EXTRACTION", "CHUNKING", "NORMALIZATION"
        ]

    def test_get_stage_returns_correct_stage(self) -> None:
        m = IngestionMetrics()
        m.start_pipeline()
        with m.track_stage("CHUNKING"):
            pass
        sm = m.get_stage("CHUNKING")
        assert sm is not None
        assert sm.stage == "CHUNKING"

    def test_get_stage_returns_none_for_missing(self) -> None:
        m = IngestionMetrics()
        assert m.get_stage("NONEXISTENT") is None

    def test_successful_stages_returns_only_successes(self) -> None:
        m = IngestionMetrics()
        m.start_pipeline()
        with m.track_stage("EXTRACTION"):
            pass
        with pytest.raises(RuntimeError):
            with m.track_stage("CHUNKING"):
                raise RuntimeError("fail")
        assert m.successful_stages() == ["EXTRACTION"]

    def test_failed_stage_returns_first_failure(self) -> None:
        m = IngestionMetrics()
        m.start_pipeline()
        with m.track_stage("EXTRACTION"):
            pass
        with pytest.raises(ValueError):
            with m.track_stage("CHUNKING"):
                raise ValueError("oops")
        failed = m.failed_stage()
        assert failed is not None
        assert failed.stage == "CHUNKING"


# ── Counter Tests ─────────────────────────────────────────────────────────


class TestMetricsCounters:
    """Tests for record_chunks() and record_embeddings() counter helpers."""

    def test_record_chunks_reads_properties(self) -> None:
        """record_chunks() reads total/text/table/image properties from the result."""

        class _FakeChunkingResult:
            total_chunks = 10
            text_chunks = 6
            table_chunks = 3
            image_chunks = 1

        m = IngestionMetrics()
        m.record_chunks(_FakeChunkingResult())
        assert m.total_chunks == 10
        assert m.text_chunks == 6
        assert m.table_chunks == 3
        assert m.image_chunks == 1

    def test_record_embeddings_reads_total_items(self) -> None:
        """record_embeddings() reads total_items from the EmbeddingGenerationResult."""

        class _FakeGenResult:
            total_items = 7

        m = IngestionMetrics()
        m.record_embeddings(_FakeGenResult())
        assert m.total_embedding_items == 7
        assert m.total_vectors == 7


# ── Pipeline Integration Tests ────────────────────────────────────────────


class TestIngestionMetricsPipelineIntegration:
    """Integration tests verifying metrics populated by run_ingestion()."""

    def test_successful_ingestion_produces_completed_metrics(
        self, text_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """After successful run_ingestion, metrics.status == 'COMPLETED'."""
        m = IngestionMetrics()
        run_ingestion(text_pdf, provider, metrics=m)
        assert m.status == "COMPLETED"
        assert m.succeeded is True
        assert m.total_duration_seconds >= 0.0

    def test_successful_ingestion_records_all_stages(
        self, text_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """All pipeline stages are recorded in metrics after successful ingestion."""
        m = IngestionMetrics()
        run_ingestion(text_pdf, provider, metrics=m)
        recorded = [sm.stage for sm in m.stage_metrics]
        for stage in ("EXTRACTION", "CHUNKING", "NORMALIZATION", "VALIDATION",
                      "EMBEDDING_PREPARATION", "EMBEDDING_GENERATION"):
            assert stage in recorded, f"Expected stage '{stage}' in {recorded}"

    def test_successful_ingestion_all_stages_successful(
        self, text_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """Every recorded stage has success=True on a clean ingestion run."""
        m = IngestionMetrics()
        run_ingestion(text_pdf, provider, metrics=m)
        for sm in m.stage_metrics:
            assert sm.success is True, f"Stage {sm.stage!r} unexpectedly failed."

    def test_successful_ingestion_chunk_counters_populated(
        self, text_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """Chunk counters are populated and consistent after successful ingestion."""
        m = IngestionMetrics()
        result = run_ingestion(text_pdf, provider, metrics=m)
        assert m.total_chunks == result.total_items
        assert m.total_chunks == m.text_chunks + m.table_chunks + m.image_chunks

    def test_successful_ingestion_vector_counters_populated(
        self, text_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """total_vectors matches the EmbeddingGenerationResult total_items."""
        m = IngestionMetrics()
        result = run_ingestion(text_pdf, provider, metrics=m)
        assert m.total_vectors == result.total_items
        assert m.total_embedding_items == result.total_items

    def test_successful_ingestion_document_metadata_recorded(
        self, text_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """document_id and filename are populated on the metrics object."""
        m = IngestionMetrics()
        result = run_ingestion(text_pdf, provider, metrics=m)
        assert m.document_id == result.document_id
        assert m.filename == "metrics_test.pdf"

    def test_failed_ingestion_sets_failed_status(
        self, text_pdf: Path
    ) -> None:
        """A failed run_ingestion (broken provider) sets metrics.status to 'FAILED'."""
        from ingestion.ingestion_errors import IngestionEmbeddingError
        broken = _BrokenProvider()
        m = IngestionMetrics()
        with pytest.raises(IngestionEmbeddingError):
            run_ingestion(text_pdf, broken, metrics=m)
        assert m.status == "FAILED"
        assert m.failed is True
        assert m.error is not None

    def test_no_metrics_remains_backward_compatible(
        self, text_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """run_ingestion without metrics= works exactly as before Day 16."""
        result = run_ingestion(text_pdf, provider)
        assert result.is_ready is True
        assert result.total_items > 0

    def test_day14_status_tracker_still_works_alongside_metrics(
        self, text_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """status_tracker and metrics can be used together without conflict."""
        from ingestion.ingestion_status import IngestionStatus, PipelineStatus
        tracker = IngestionStatus()
        m = IngestionMetrics()
        run_ingestion(text_pdf, provider, status_tracker=tracker, metrics=m)
        assert tracker.status == PipelineStatus.COMPLETED
        assert m.status == "COMPLETED"

    def test_day15_config_still_works_alongside_metrics(
        self, text_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """IngestionConfig and metrics can be used together without conflict."""
        from ingestion.ingestion_config import IngestionConfig
        config = IngestionConfig(chunk_size=500, chunk_overlap=50)
        m = IngestionMetrics()
        result = run_ingestion(text_pdf, provider, config=config, metrics=m)
        assert m.status == "COMPLETED"
        assert result.is_ready is True
