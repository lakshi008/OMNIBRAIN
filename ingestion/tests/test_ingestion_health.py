"""
Tests for the Day 18 ingestion health and readiness layer.

Covers IngestionHealthResult dataclass, check_ingestion_health(),
check_ingestion_readiness(), failure handling, module checks, provider checks,
safety, and integration with the existing Day 1–17 ingestion architecture.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import pytest

from ingestion.ingestion_config import IngestionConfig
from ingestion.ingestion_health import (
    CHECK_CONFIG,
    CHECK_MODULES,
    CHECK_PROVIDER,
    CHECK_QDRANT_CONFIG,
    IngestionHealthResult,
    check_ingestion_health,
    check_ingestion_readiness,
)
from ingestion.ingestion_logging import IngestionLogger
from ingestion.ingestion_metrics import IngestionMetrics
from ingestion.ingestion_service import run_ingestion
from ingestion.qdrant_config import QdrantConfig


# ── Helpers & Fixtures ───────────────────────────────────────────────────


class _ValidSingleEmbedProvider:
    """Valid provider exposing only embed(text)."""

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class _ValidBatchEmbedProvider:
    """Valid provider exposing embed_batch(texts)."""

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _InvalidProviderNoMethods:
    """Invalid provider with no embed or embed_batch methods."""

    pass


class _InvalidProviderNonCallable:
    """Invalid provider where embed is not callable."""

    embed = "not_a_callable"


@pytest.fixture
def valid_provider() -> _ValidBatchEmbedProvider:
    return _ValidBatchEmbedProvider()


# ── IngestionHealthResult Dataclass Tests ─────────────────────────────────


class TestIngestionHealthResult:
    """Tests for the IngestionHealthResult dataclass and helper methods."""

    def test_default_health_result(self) -> None:
        res = IngestionHealthResult()
        assert res.healthy is True
        assert res.ready is True
        assert res.status == "READY"
        assert res.checks == {}
        assert res.errors == []

    def test_is_healthy_helper(self) -> None:
        res_true = IngestionHealthResult(healthy=True)
        res_false = IngestionHealthResult(healthy=False)
        assert res_true.is_healthy() is True
        assert res_false.is_healthy() is False

    def test_is_ready_helper(self) -> None:
        res_true = IngestionHealthResult(ready=True)
        res_false = IngestionHealthResult(ready=False)
        assert res_true.is_ready() is True
        assert res_false.is_ready() is False

    def test_passed_failed_skipped_checks_filtering(self) -> None:
        res = IngestionHealthResult(
            checks={
                "check_a": "PASS",
                "check_b": "FAIL",
                "check_c": "SKIP",
                "check_d": "PASS",
            }
        )
        assert res.passed_checks() == ["check_a", "check_d"]
        assert res.failed_checks() == ["check_b"]
        assert res.skipped_checks() == ["check_c"]


# ── check_ingestion_health() Tests ───────────────────────────────────────


class TestCheckIngestionHealth:
    """Tests for the health-checking function."""

    def test_default_health_check_passes(self) -> None:
        res = check_ingestion_health()
        assert res.is_healthy() is True
        assert res.status == "READY"
        assert res.checks[CHECK_MODULES] == "PASS"
        assert res.checks[CHECK_CONFIG] == "PASS"
        assert res.checks[CHECK_QDRANT_CONFIG] == "PASS"
        assert res.checks[CHECK_PROVIDER] == "SKIP"
        assert len(res.errors) == 0

    def test_health_check_with_valid_custom_config(self) -> None:
        config = IngestionConfig(chunk_size=500, chunk_overlap=50)
        res = check_ingestion_health(config=config)
        assert res.is_healthy() is True
        assert res.checks[CHECK_CONFIG] == "PASS"

    def test_health_check_with_valid_custom_qdrant_config(self) -> None:
        q_cfg = QdrantConfig(url=":memory:", timeout=15.0)
        res = check_ingestion_health(qdrant_config=q_cfg)
        assert res.is_healthy() is True
        assert res.checks[CHECK_QDRANT_CONFIG] == "PASS"

    def test_health_check_with_invalid_config_type(self) -> None:
        res = check_ingestion_health(config={"chunk_size": 500})  # type: ignore[arg-type]
        assert res.is_healthy() is False
        assert res.checks[CHECK_CONFIG] == "FAIL"
        assert any("IngestionConfig" in err for err in res.errors)
        assert res.status in ("DEGRADED", "UNHEALTHY")

    def test_health_check_with_invalid_qdrant_config_type(self) -> None:
        res = check_ingestion_health(qdrant_config="invalid_config")  # type: ignore[arg-type]
        assert res.is_healthy() is False
        assert res.checks[CHECK_QDRANT_CONFIG] == "FAIL"
        assert any("QdrantConfig" in err for err in res.errors)

    def test_health_check_with_missing_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original_import = __import__

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "ingestion.pdf_text_extractor":
                raise ImportError("Mocked missing module")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mocked_import)
        res = check_ingestion_health()
        assert res.is_healthy() is False
        assert res.checks[CHECK_MODULES] == "FAIL"
        assert any("Mocked missing module" in err for err in res.errors)


# ── check_ingestion_readiness() Tests ────────────────────────────────────


class TestCheckIngestionReadiness:
    """Tests for the readiness-checking function."""

    def test_readiness_with_valid_batch_provider_passes(
        self, valid_provider: _ValidBatchEmbedProvider
    ) -> None:
        res = check_ingestion_readiness(provider=valid_provider)
        assert res.is_healthy() is True
        assert res.is_ready() is True
        assert res.status == "READY"
        assert res.checks[CHECK_PROVIDER] == "PASS"
        assert len(res.errors) == 0

    def test_readiness_with_valid_single_embed_provider_passes(self) -> None:
        provider = _ValidSingleEmbedProvider()
        res = check_ingestion_readiness(provider=provider)
        assert res.is_ready() is True
        assert res.checks[CHECK_PROVIDER] == "PASS"

    def test_readiness_missing_provider_fails(self) -> None:
        res = check_ingestion_readiness(provider=None)
        assert res.is_healthy() is True
        assert res.is_ready() is False
        assert res.status == "HEALTHY"
        assert res.checks[CHECK_PROVIDER] == "FAIL"
        assert any("provider" in err.lower() for err in res.errors)

    def test_readiness_invalid_provider_no_methods_fails(self) -> None:
        provider = _InvalidProviderNoMethods()
        res = check_ingestion_readiness(provider=provider)
        assert res.is_ready() is False
        assert res.checks[CHECK_PROVIDER] == "FAIL"
        assert any("embed" in err for err in res.errors)

    def test_readiness_invalid_provider_non_callable_fails(self) -> None:
        provider = _InvalidProviderNonCallable()
        res = check_ingestion_readiness(provider=provider)
        assert res.is_ready() is False
        assert res.checks[CHECK_PROVIDER] == "FAIL"

    def test_readiness_with_invalid_config_fails_both(self) -> None:
        res = check_ingestion_readiness(
            provider=_ValidSingleEmbedProvider(),
            config="not_a_config",  # type: ignore[arg-type]
        )
        assert res.is_healthy() is False
        assert res.is_ready() is False
        assert res.checks[CHECK_CONFIG] == "FAIL"

    def test_readiness_with_invalid_qdrant_config_fails(
        self, valid_provider: _ValidBatchEmbedProvider
    ) -> None:
        res = check_ingestion_readiness(
            provider=valid_provider,
            qdrant_config=12345,  # type: ignore[arg-type]
        )
        assert res.is_ready() is False
        assert res.checks[CHECK_QDRANT_CONFIG] == "FAIL"

    def test_readiness_all_components_valid(
        self, valid_provider: _ValidBatchEmbedProvider
    ) -> None:
        config = IngestionConfig(chunk_size=800, chunk_overlap=100)
        q_cfg = QdrantConfig(url=":memory:")
        res = check_ingestion_readiness(
            provider=valid_provider,
            config=config,
            qdrant_config=q_cfg,
        )
        assert res.is_healthy() is True
        assert res.is_ready() is True
        assert res.status == "READY"
        assert len(res.failed_checks()) == 0


# ── Safety & Isolation Tests ─────────────────────────────────────────────


class TestHealthSafetyAndIsolation:
    """Tests verifying safety properties of health checks."""

    def test_health_check_does_not_modify_filesystem(self, tmp_path: Path) -> None:
        before_files = list(tmp_path.iterdir())
        check_ingestion_health()
        after_files = list(tmp_path.iterdir())
        assert before_files == after_files

    def test_readiness_check_does_not_call_embed(self) -> None:
        class _TrackingProvider:
            called = False

            def embed(self, text: str) -> list[float]:
                self.called = True
                return [0.0]

        tracker = _TrackingProvider()
        res = check_ingestion_readiness(provider=tracker)
        assert res.is_ready() is True
        assert tracker.called is False

    def test_health_check_handles_unexpected_exception_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_unexpected(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Unexpected internal boom")

        monkeypatch.setattr(
            "ingestion.ingestion_health._check_modules",
            lambda: ("FAIL", "Unexpected internal boom"),
        )
        res = check_ingestion_health()
        assert res.is_healthy() is False
        assert any("boom" in err for err in res.errors)


# ── Pipeline Integration Tests ───────────────────────────────────────────


class TestHealthPipelineIntegration:
    """Tests checking integration between health checks and pipeline execution."""

    def test_pre_flight_readiness_check_before_ingestion(
        self, tmp_path: Path, valid_provider: _ValidBatchEmbedProvider
    ) -> None:
        import pymupdf

        pdf_path = tmp_path / "flight_test.pdf"
        doc = pymupdf.open()
        p = doc.new_page()
        p.insert_text((50, 100), "Sample text for health integration.")
        doc.save(str(pdf_path))
        doc.close()

        config = IngestionConfig(chunk_size=500, chunk_overlap=50)
        # Pre-flight check
        readiness = check_ingestion_readiness(provider=valid_provider, config=config)
        assert readiness.is_ready() is True

        # Pipeline run
        metrics = IngestionMetrics()
        logger = IngestionLogger(logging.getLogger("test.health_integration"))
        result = run_ingestion(
            pdf_path=pdf_path,
            embedding_provider=valid_provider,
            config=config,
            metrics=metrics,
            logger=logger,
        )
        assert result.is_ready is True
        assert metrics.status == "COMPLETED"
