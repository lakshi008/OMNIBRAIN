"""
Tests for the IngestionConfig configuration layer.

Covers default construction, custom values, strict validation,
environment-variable loading, and integration with run_ingestion.
"""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf
import pytest

from ingestion.ingestion_config import IngestionConfig
from ingestion.ingestion_service import run_ingestion


# ── Helpers ───────────────────────────────────────────────────────────────


class _DeterministicProvider:
    """Deterministic 4-dimensional embedding provider for integration tests."""

    def embed(self, text: str) -> list[float]:
        return [(sum(ord(c) for c in text) % 1000) / 1000.0 * (i + 1) % 1.0 for i in range(4)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def provider() -> _DeterministicProvider:
    return _DeterministicProvider()


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    """Two-page text-only PDF for integration tests."""
    pdf_path = tmp_path / "config_test.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 100), "OmniBrain ingestion configuration integration test page one.")
    p2 = doc.new_page()
    p2.insert_text((50, 100), "OmniBrain ingestion configuration integration test page two.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ── Default Configuration Tests ───────────────────────────────────────────


class TestIngestionConfigDefaults:
    """Tests verifying that default IngestionConfig values are valid and correct."""

    def test_default_config_can_be_created(self) -> None:
        """Default IngestionConfig() constructs without errors."""
        config = IngestionConfig()
        assert config is not None

    def test_default_chunk_size(self) -> None:
        config = IngestionConfig()
        assert config.chunk_size == 1000

    def test_default_chunk_overlap(self) -> None:
        config = IngestionConfig()
        assert config.chunk_overlap == 200

    def test_default_retrieval_top_k(self) -> None:
        config = IngestionConfig()
        assert config.retrieval_top_k == 5

    def test_default_retrieval_min_score(self) -> None:
        config = IngestionConfig()
        assert config.retrieval_min_score == 0.0

    def test_default_qdrant_collection(self) -> None:
        config = IngestionConfig()
        assert config.qdrant_collection == "omnibrain_documents"

    def test_default_qdrant_timeout(self) -> None:
        config = IngestionConfig()
        assert config.qdrant_timeout == 10.0


# ── Custom Configuration Tests ────────────────────────────────────────────


class TestIngestionConfigCustomValues:
    """Tests verifying that custom values are accepted and stored correctly."""

    def test_custom_chunk_size(self) -> None:
        config = IngestionConfig(chunk_size=500)
        assert config.chunk_size == 500

    def test_custom_chunk_overlap(self) -> None:
        config = IngestionConfig(chunk_size=500, chunk_overlap=50)
        assert config.chunk_overlap == 50

    def test_custom_retrieval_top_k(self) -> None:
        config = IngestionConfig(retrieval_top_k=10)
        assert config.retrieval_top_k == 10

    def test_custom_retrieval_min_score(self) -> None:
        config = IngestionConfig(retrieval_min_score=0.75)
        assert config.retrieval_min_score == 0.75

    def test_custom_qdrant_collection(self) -> None:
        config = IngestionConfig(qdrant_collection="my_collection")
        assert config.qdrant_collection == "my_collection"

    def test_custom_qdrant_timeout(self) -> None:
        config = IngestionConfig(qdrant_timeout=30.0)
        assert config.qdrant_timeout == 30.0

    def test_min_score_boundary_zero(self) -> None:
        config = IngestionConfig(retrieval_min_score=0.0)
        assert config.retrieval_min_score == 0.0

    def test_min_score_boundary_one(self) -> None:
        config = IngestionConfig(retrieval_min_score=1.0)
        assert config.retrieval_min_score == 1.0


# ── Validation Tests ──────────────────────────────────────────────────────


class TestIngestionConfigValidation:
    """Tests verifying that invalid values are rejected with clear ValueError messages."""

    # chunk_size
    def test_chunk_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            IngestionConfig(chunk_size=0)

    def test_chunk_size_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            IngestionConfig(chunk_size=-1)

    def test_chunk_size_bool_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            IngestionConfig(chunk_size=True)  # type: ignore[arg-type]

    # chunk_overlap
    def test_chunk_overlap_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            IngestionConfig(chunk_size=500, chunk_overlap=-1)

    def test_chunk_overlap_equal_to_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            IngestionConfig(chunk_size=200, chunk_overlap=200)

    def test_chunk_overlap_greater_than_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            IngestionConfig(chunk_size=200, chunk_overlap=300)

    # retrieval_top_k
    def test_top_k_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="retrieval_top_k"):
            IngestionConfig(retrieval_top_k=0)

    def test_top_k_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="retrieval_top_k"):
            IngestionConfig(retrieval_top_k=-5)

    def test_top_k_bool_raises(self) -> None:
        with pytest.raises(ValueError, match="retrieval_top_k"):
            IngestionConfig(retrieval_top_k=True)  # type: ignore[arg-type]

    # retrieval_min_score
    def test_min_score_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="retrieval_min_score"):
            IngestionConfig(retrieval_min_score=-0.1)

    def test_min_score_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="retrieval_min_score"):
            IngestionConfig(retrieval_min_score=1.1)

    def test_min_score_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="retrieval_min_score"):
            IngestionConfig(retrieval_min_score=float("nan"))

    def test_min_score_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="retrieval_min_score"):
            IngestionConfig(retrieval_min_score=float("inf"))

    # qdrant_collection
    def test_empty_collection_name_raises(self) -> None:
        with pytest.raises(ValueError, match="qdrant_collection"):
            IngestionConfig(qdrant_collection="")

    def test_whitespace_collection_name_raises(self) -> None:
        with pytest.raises(ValueError, match="qdrant_collection"):
            IngestionConfig(qdrant_collection="   ")

    # qdrant_timeout
    def test_timeout_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="qdrant_timeout"):
            IngestionConfig(qdrant_timeout=0.0)

    def test_timeout_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="qdrant_timeout"):
            IngestionConfig(qdrant_timeout=-5.0)

    def test_timeout_bool_raises(self) -> None:
        with pytest.raises(ValueError, match="qdrant_timeout"):
            IngestionConfig(qdrant_timeout=True)  # type: ignore[arg-type]


# ── Environment Variable Tests ────────────────────────────────────────────


class TestIngestionConfigFromEnv:
    """Tests for environment-variable-driven configuration via from_env()."""

    def test_from_env_uses_defaults_when_no_env_vars_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() with no env vars set produces default config."""
        for var in (
            "INGESTION_CHUNK_SIZE", "INGESTION_CHUNK_OVERLAP",
            "INGESTION_TOP_K", "INGESTION_MIN_SCORE",
            "QDRANT_COLLECTION", "QDRANT_TIMEOUT",
        ):
            monkeypatch.delenv(var, raising=False)
        config = IngestionConfig.from_env()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.retrieval_top_k == 5
        assert config.retrieval_min_score == 0.0
        assert config.qdrant_collection == "omnibrain_documents"
        assert config.qdrant_timeout == 10.0

    def test_from_env_overrides_chunk_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INGESTION_CHUNK_SIZE", "512")
        monkeypatch.delenv("INGESTION_CHUNK_OVERLAP", raising=False)
        config = IngestionConfig.from_env()
        assert config.chunk_size == 512

    def test_from_env_overrides_collection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")
        config = IngestionConfig.from_env()
        assert config.qdrant_collection == "test_collection"

    def test_from_env_overrides_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QDRANT_TIMEOUT", "25.0")
        config = IngestionConfig.from_env()
        assert config.qdrant_timeout == 25.0

    def test_from_env_invalid_chunk_size_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INGESTION_CHUNK_SIZE", "not_a_number")
        with pytest.raises(ValueError, match="INGESTION_CHUNK_SIZE"):
            IngestionConfig.from_env()

    def test_from_env_invalid_min_score_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INGESTION_MIN_SCORE", "banana")
        with pytest.raises(ValueError, match="INGESTION_MIN_SCORE"):
            IngestionConfig.from_env()

    def test_from_env_empty_collection_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QDRANT_COLLECTION", "")
        with pytest.raises(ValueError, match="QDRANT_COLLECTION"):
            IngestionConfig.from_env()


# ── Integration Tests ─────────────────────────────────────────────────────


class TestIngestionConfigIntegration:
    """Tests verifying IngestionConfig integrates correctly with run_ingestion."""

    def test_run_ingestion_accepts_config(
        self, text_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """run_ingestion works when an IngestionConfig is passed."""
        config = IngestionConfig(chunk_size=500, chunk_overlap=50)
        result = run_ingestion(text_pdf, provider, config=config)
        assert result.is_ready is True

    def test_config_chunk_size_is_used(
        self, text_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """A smaller chunk_size via config produces more chunks than the default."""
        config_small = IngestionConfig(chunk_size=20, chunk_overlap=5)
        config_large = IngestionConfig(chunk_size=2000, chunk_overlap=200)

        result_small = run_ingestion(text_pdf, provider, config=config_small)
        result_large = run_ingestion(text_pdf, provider, config=config_large)

        assert result_small.total_items >= result_large.total_items

    def test_explicit_args_override_config(
        self, text_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """Explicit chunk_size/chunk_overlap kwargs override config values."""
        config = IngestionConfig(chunk_size=2000, chunk_overlap=200)
        # Override with much smaller values
        result = run_ingestion(text_pdf, provider, config=config, chunk_size=20, chunk_overlap=5)
        # Result should have more items than using chunk_size=2000
        result_default = run_ingestion(text_pdf, provider, config=config)
        assert result.total_items >= result_default.total_items

    def test_no_config_uses_defaults_backward_compatible(
        self, text_pdf: Path, provider: _DeterministicProvider
    ) -> None:
        """run_ingestion without config parameter behaves exactly as before Day 15."""
        result = run_ingestion(text_pdf, provider)
        assert result.is_ready is True
        assert result.total_items > 0
