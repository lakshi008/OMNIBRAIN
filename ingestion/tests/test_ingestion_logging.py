"""
Tests for the ingestion structured logging layer.

Covers logger creation, handler deduplication, structured log events,
correct log levels, contextual fields, pipeline integration,
and backward compatibility with Day 14/15/16 components.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf
import pytest

from ingestion.ingestion_logging import (
    IngestionLogger,
    _LOGGER_NAME,
    _fmt,
    get_ingestion_logger,
)
from ingestion.ingestion_metrics import IngestionMetrics
from ingestion.ingestion_service import run_ingestion


# ── Helpers ───────────────────────────────────────────────────────────────


class _DeterministicProvider:
    def embed(self, text: str) -> list[float]:
        base = sum(ord(c) for c in text) % 997
        return [round((base * (i + 1)) % 1000 / 1000.0, 4) for i in range(4)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _BrokenProvider:
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0] if i == 0 else [1.0] for i in range(len(texts))]


@pytest.fixture
def provider() -> _DeterministicProvider:
    return _DeterministicProvider()


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "logging_test.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 100), "OmniBrain ingestion logging test page one content.")
    p2 = doc.new_page()
    p2.insert_text((50, 100), "OmniBrain ingestion logging test page two content.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ── Logger Creation Tests ─────────────────────────────────────────────────


class TestGetIngestionLogger:
    """Tests for get_ingestion_logger() and handler deduplication."""

    def test_logger_can_be_created(self) -> None:
        logger = get_ingestion_logger()
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_logger_has_expected_name(self) -> None:
        logger = get_ingestion_logger()
        assert logger.name == _LOGGER_NAME

    def test_repeated_calls_return_same_logger(self) -> None:
        logger1 = get_ingestion_logger()
        logger2 = get_ingestion_logger()
        assert logger1 is logger2

    def test_custom_name_creates_separate_logger(self) -> None:
        logger_a = get_ingestion_logger("ingestion.test_a")
        logger_b = get_ingestion_logger("ingestion.test_b")
        assert logger_a is not logger_b


class TestIngestionLoggerCreation:
    """Tests for IngestionLogger construction and initial state."""

    def test_ingestion_logger_can_be_created(self) -> None:
        il = IngestionLogger()
        assert il is not None

    def test_ingestion_logger_has_underlying_logger(self) -> None:
        il = IngestionLogger()
        assert isinstance(il.logger, logging.Logger)

    def test_ingestion_logger_initial_document_id_empty(self) -> None:
        il = IngestionLogger()
        assert il.document_id == ""

    def test_ingestion_logger_initial_filename_empty(self) -> None:
        il = IngestionLogger()
        assert il.filename == ""

    def test_custom_logger_injection(self) -> None:
        custom = logging.getLogger("ingestion.custom_test")
        il = IngestionLogger(logger=custom)
        assert il.logger is custom


# ── Log Event Tests ───────────────────────────────────────────────────────


class TestLogEvents:
    """Tests verifying structured log events at correct levels."""

    def test_ingestion_start_emits_info(self, caplog: pytest.LogCaptureFixture) -> None:
        il = IngestionLogger()
        with caplog.at_level(logging.INFO, logger=il.logger.name):
            il.log_ingestion_start("report.pdf")
        records = [r for r in caplog.records if r.name == il.logger.name]
        assert any("ingestion_start" in r.message for r in records)
        assert any(r.levelno == logging.INFO for r in records if "ingestion_start" in r.message)

    def test_ingestion_start_records_filename(self, caplog: pytest.LogCaptureFixture) -> None:
        il = IngestionLogger()
        with caplog.at_level(logging.INFO, logger=il.logger.name):
            il.log_ingestion_start("my_report.pdf")
        assert any("my_report.pdf" in r.message for r in caplog.records)

    def test_ingestion_complete_emits_info(self, caplog: pytest.LogCaptureFixture) -> None:
        il = IngestionLogger()
        with caplog.at_level(logging.INFO, logger=il.logger.name):
            il.log_ingestion_complete("doc-123", "report.pdf")
        assert any(
            "ingestion_complete" in r.message and r.levelno == logging.INFO
            for r in caplog.records
        )

    def test_ingestion_complete_includes_document_id(self, caplog: pytest.LogCaptureFixture) -> None:
        il = IngestionLogger()
        with caplog.at_level(logging.INFO, logger=il.logger.name):
            il.log_ingestion_complete("doc-uuid-999", "file.pdf", total_vectors=5)
        assert any("doc-uuid-999" in r.message for r in caplog.records)

    def test_ingestion_failed_emits_error(self, caplog: pytest.LogCaptureFixture) -> None:
        il = IngestionLogger()
        with caplog.at_level(logging.ERROR, logger=il.logger.name):
            il.log_ingestion_failed("broken.pdf", error="Corrupted PDF")
        assert any(
            "ingestion_failed" in r.message and r.levelno == logging.ERROR
            for r in caplog.records
        )

    def test_ingestion_failed_includes_error(self, caplog: pytest.LogCaptureFixture) -> None:
        il = IngestionLogger()
        with caplog.at_level(logging.ERROR, logger=il.logger.name):
            il.log_ingestion_failed("x.pdf", error="Missing file")
        assert any("Missing file" in r.message for r in caplog.records)

    def test_stage_start_emits_info(self, caplog: pytest.LogCaptureFixture) -> None:
        il = IngestionLogger()
        with caplog.at_level(logging.INFO, logger=il.logger.name):
            il.log_stage_start("CHUNKING")
        assert any(
            "stage_start" in r.message and "CHUNKING" in r.message and r.levelno == logging.INFO
            for r in caplog.records
        )

    def test_stage_complete_emits_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        il = IngestionLogger()
        with caplog.at_level(logging.DEBUG, logger=il.logger.name):
            il.log_stage_complete("EXTRACTION", duration_seconds=0.12)
        assert any(
            "stage_complete" in r.message and r.levelno == logging.DEBUG
            for r in caplog.records
        )

    def test_stage_complete_includes_duration(self, caplog: pytest.LogCaptureFixture) -> None:
        il = IngestionLogger()
        with caplog.at_level(logging.DEBUG, logger=il.logger.name):
            il.log_stage_complete("NORMALIZATION", duration_seconds=0.025)
        assert any("duration_seconds" in r.message for r in caplog.records)

    def test_stage_failed_emits_error(self, caplog: pytest.LogCaptureFixture) -> None:
        il = IngestionLogger()
        with caplog.at_level(logging.ERROR, logger=il.logger.name):
            il.log_stage_failed("CHUNKING", error="Out of memory")
        assert any(
            "stage_failed" in r.message and r.levelno == logging.ERROR
            for r in caplog.records
        )

    def test_stage_failed_includes_stage_name(self, caplog: pytest.LogCaptureFixture) -> None:
        il = IngestionLogger()
        with caplog.at_level(logging.ERROR, logger=il.logger.name):
            il.log_stage_failed("EMBEDDING_GENERATION", error="dim mismatch")
        assert any("EMBEDDING_GENERATION" in r.message for r in caplog.records)

    def test_log_from_metrics_emits_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        il = IngestionLogger()
        m = IngestionMetrics(
            document_id="doc-99",
            filename="test.pdf",
            total_chunks=5,
            text_chunks=4,
            table_chunks=1,
            total_vectors=5,
            status="COMPLETED",
        )
        with caplog.at_level(logging.DEBUG, logger=il.logger.name):
            il.log_from_metrics(m)
        assert any("metrics_summary" in r.message for r in caplog.records)


# ── Format Helper Tests ───────────────────────────────────────────────────


class TestFmtHelper:
    """Tests for the _fmt() formatting helper."""

    def test_fmt_basic_fields(self) -> None:
        result = _fmt(event="stage_start", stage="CHUNKING")
        assert "event='stage_start'" in result
        assert "stage='CHUNKING'" in result

    def test_fmt_none_values_excluded(self) -> None:
        result = _fmt(event="test", error=None)
        assert "error" not in result

    def test_fmt_zero_int_included(self) -> None:
        result = _fmt(total_chunks=0)
        assert "total_chunks=0" in result


# ── Pipeline Integration Tests ────────────────────────────────────────────


class TestLoggingPipelineIntegration:
    """Integration tests verifying logging cooperates with run_ingestion."""

    def test_successful_ingestion_logs_start_and_complete(
        self,
        text_pdf: Path,
        provider: _DeterministicProvider,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        il = IngestionLogger()
        with caplog.at_level(logging.DEBUG, logger=il.logger.name):
            run_ingestion(text_pdf, provider, logger=il)
        messages = " ".join(r.message for r in caplog.records if r.name == il.logger.name)
        assert "ingestion_start" in messages
        assert "ingestion_complete" in messages

    def test_successful_ingestion_logs_all_stages(
        self,
        text_pdf: Path,
        provider: _DeterministicProvider,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        il = IngestionLogger()
        with caplog.at_level(logging.DEBUG, logger=il.logger.name):
            run_ingestion(text_pdf, provider, logger=il)
        messages = " ".join(r.message for r in caplog.records if r.name == il.logger.name)
        for stage in ("EXTRACTION", "CHUNKING", "NORMALIZATION",
                      "VALIDATION", "EMBEDDING_PREPARATION", "EMBEDDING_GENERATION"):
            assert stage in messages, f"Expected stage '{stage}' in log messages"

    def test_failed_ingestion_logs_failure(
        self,
        text_pdf: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from ingestion.ingestion_errors import IngestionEmbeddingError
        il = IngestionLogger()
        broken = _BrokenProvider()
        with caplog.at_level(logging.ERROR, logger=il.logger.name):
            with pytest.raises(IngestionEmbeddingError):
                run_ingestion(text_pdf, broken, logger=il)
        error_msgs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("ingestion_failed" in m or "stage_failed" in m for m in error_msgs)

    def test_logging_does_not_change_returned_result(
        self,
        text_pdf: Path,
        provider: _DeterministicProvider,
    ) -> None:
        """run_ingestion with logger= returns the same result as without."""
        il = IngestionLogger()
        result_with_log = run_ingestion(text_pdf, provider, logger=il)
        result_without_log = run_ingestion(text_pdf, provider)
        assert result_with_log.total_items == result_without_log.total_items
        assert result_with_log.dimension == result_without_log.dimension

    def test_logging_does_not_swallow_exceptions(
        self,
        tmp_path: Path,
        provider: _DeterministicProvider,
    ) -> None:
        """Exceptions still propagate when logger is provided."""
        from ingestion.ingestion_errors import IngestionExtractionError
        il = IngestionLogger()
        with pytest.raises(IngestionExtractionError):
            run_ingestion(tmp_path / "missing.pdf", provider, logger=il)

    def test_no_logger_remains_backward_compatible(
        self,
        text_pdf: Path,
        provider: _DeterministicProvider,
    ) -> None:
        """run_ingestion without logger= works identically to before Day 17."""
        result = run_ingestion(text_pdf, provider)
        assert result.is_ready is True
        assert result.total_items > 0

    def test_day16_metrics_work_alongside_logger(
        self,
        text_pdf: Path,
        provider: _DeterministicProvider,
    ) -> None:
        """metrics= and logger= can be combined without conflict."""
        m = IngestionMetrics()
        il = IngestionLogger()
        run_ingestion(text_pdf, provider, metrics=m, logger=il)
        assert m.status == "COMPLETED"
        assert m.total_vectors > 0

    def test_plain_stdlib_logger_accepted(
        self,
        text_pdf: Path,
        provider: _DeterministicProvider,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A plain logging.Logger can be passed as the logger= argument."""
        stdlib_logger = logging.getLogger("ingestion.stdlib_test")
        with caplog.at_level(logging.DEBUG, logger=stdlib_logger.name):
            result = run_ingestion(text_pdf, provider, logger=stdlib_logger)
        assert result.is_ready is True
