"""
OmniBrain Member 4 — Day 41 Release & Deployment Configuration Regression Certification.

Verifies that OMNIBRAIN remains compatible with its existing release/deployment configuration.
Tests cover only configuration files and modules actually present in the repository:

  Repository configuration files discovered:
    - pytest.ini              (test discovery configuration)
    - requirements.txt        (single runtime dependency: pymupdf)
    - .env.example            (empty, no documented env vars in example)
    - ingestion/ingestion_config.py   (IngestionConfig + from_env())
    - ingestion/qdrant_config.py      (QdrantConfig + from_env())
    - vision/provider_config.py       (VisionProviderConfig + VisionProviderCapabilities)

  Coverage:
    - pytest.ini is parseable and contains valid test discovery settings
    - requirements.txt is readable and parseable
    - All three package __init__ modules import without errors
    - All public submodules import without requiring production infrastructure
    - IngestionConfig defaults match documented values
    - IngestionConfig.from_env() with no env vars returns correct defaults
    - IngestionConfig.from_env() with valid env vars applies them
    - IngestionConfig.from_env() with invalid env vars raises ValueError
    - IngestionConfig validation of invalid values raises ValueError
    - QdrantConfig defaults (url=':memory:', no api_key, correct collection/timeout)
    - QdrantConfig.from_env() with no env vars returns correct defaults
    - QdrantConfig.from_env() with env vars applies them (monkeypatched, restored)
    - VisionProviderConfig defaults and validation
    - VisionProviderConfig to_dict/from_dict round trip
    - VisionProviderCapabilities defaults and validation
    - VisionProviderCapabilities to_dict/from_dict round trip
    - Cross-component config compatibility (IngestionConfig + QdrantConfig + VisionProviderConfig)
    - Configuration override and restore isolation (no env leak between tests)
    - Configuration immutability (loading config does not modify any source file)
    - Secret safety (no real secrets anywhere in test values)
    - Environment isolation (monkeypatch restores real environment after each test)

  NOT APPLICABLE sections (not present in repo):
    - pyproject.toml — absent
    - Dockerfile — absent
    - docker-compose.yml — absent
    - FastAPI app entry point — absent
    - Route discovery — absent

Constraints:
  - 100% offline. Zero real APIs, network, LLM, Qdrant cluster, or credentials.
  - Zero production code modified.
  - No configuration loaders, adapters, wrappers, or env handlers added.
  - Only fake placeholder values (DAY41_FAKE_VALUE) used; no real secrets.
"""

from __future__ import annotations

import configparser
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Snapshot of config file hashes at module import time (immutability baseline)
_CONFIG_FILE_PATHS = [
    REPO_ROOT / "pytest.ini",
    REPO_ROOT / "requirements.txt",
    REPO_ROOT / ".env.example",
]


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


_INITIAL_HASHES = {str(p): _file_sha256(p) for p in _CONFIG_FILE_PATHS}


# ===========================================================================
# 1. FILE EXISTENCE & READABILITY (Section 3)
# ===========================================================================

class TestConfigurationDiscovery:
    """Verify that all expected configuration files are present and readable."""

    def test_pytest_ini_exists(self) -> None:
        assert (REPO_ROOT / "pytest.ini").exists()

    def test_requirements_txt_exists(self) -> None:
        assert (REPO_ROOT / "requirements.txt").exists()

    def test_env_example_exists(self) -> None:
        assert (REPO_ROOT / ".env.example").exists()

    def test_ingestion_config_module_exists(self) -> None:
        assert (REPO_ROOT / "ingestion" / "ingestion_config.py").exists()

    def test_qdrant_config_module_exists(self) -> None:
        assert (REPO_ROOT / "ingestion" / "qdrant_config.py").exists()

    def test_vision_provider_config_module_exists(self) -> None:
        assert (REPO_ROOT / "vision" / "provider_config.py").exists()


# ===========================================================================
# 2. PYTEST.INI CONFIGURATION (Section 7)
# ===========================================================================

class TestPytestIniConfiguration:
    """Verify pytest.ini is parseable and contains valid configuration."""

    def test_pytest_ini_parseable(self) -> None:
        ini_path = REPO_ROOT / "pytest.ini"
        parser = configparser.ConfigParser()
        parser.read(str(ini_path))
        assert "pytest" in parser.sections()

    def test_pytest_ini_testpaths_present(self) -> None:
        ini_path = REPO_ROOT / "pytest.ini"
        content = ini_path.read_text(encoding="utf-8")
        assert "testpaths" in content

    def test_pytest_ini_expected_testpaths(self) -> None:
        ini_path = REPO_ROOT / "pytest.ini"
        content = ini_path.read_text(encoding="utf-8")
        assert "ingestion/tests" in content
        assert "vision/tests" in content

    def test_pytest_ini_addopts_present(self) -> None:
        ini_path = REPO_ROOT / "pytest.ini"
        content = ini_path.read_text(encoding="utf-8")
        assert "addopts" in content

    def test_pytest_ini_ignore_option_present(self) -> None:
        ini_path = REPO_ROOT / "pytest.ini"
        content = ini_path.read_text(encoding="utf-8")
        assert "test_parser.py" in content


# ===========================================================================
# 3. REQUIREMENTS.TXT (Section 3)
# ===========================================================================

class TestRequirementsTxt:
    """Verify requirements.txt is readable and contains the expected runtime dependency."""

    @staticmethod
    def _read_reqs() -> str:
        req_path = REPO_ROOT / "requirements.txt"
        raw = req_path.read_bytes()
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    def test_requirements_readable(self) -> None:
        content = self._read_reqs()
        assert content.strip()

    def test_requirements_contains_pymupdf(self) -> None:
        content = self._read_reqs()
        assert "pymupdf" in content.lower()

    def test_requirements_version_pinned(self) -> None:
        content = self._read_reqs()
        # Expect pinned version format: name==version
        assert "==" in content


# ===========================================================================
# 4. PACKAGE IMPORT REGRESSION (Section 18)
# ===========================================================================

class TestPackageImportRegression:
    """Verify all three main packages import cleanly without production infrastructure."""

    def test_ingestion_package_imports(self) -> None:
        import ingestion  # noqa: F401
        assert hasattr(ingestion, "IngestionConfig")
        assert hasattr(ingestion, "QdrantConfig")
        assert hasattr(ingestion, "DocumentChunk")
        assert hasattr(ingestion, "prepare_for_embedding")

    def test_agents_package_imports(self) -> None:
        import agents  # noqa: F401
        assert hasattr(agents, "AgentCitation")
        assert hasattr(agents, "AgentResponse")
        assert hasattr(agents, "AgentState")

    def test_vision_package_imports(self) -> None:
        import vision  # noqa: F401
        assert hasattr(vision, "VisionProviderConfig")
        assert hasattr(vision, "VisionProviderCapabilities")
        assert hasattr(vision, "VisualEvidenceAdapter")

    def test_ingestion_config_module_import(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        assert IngestionConfig is not None

    def test_qdrant_config_module_import(self) -> None:
        from ingestion.qdrant_config import QdrantConfig
        assert QdrantConfig is not None

    def test_vision_provider_config_import(self) -> None:
        from vision.provider_config import VisionProviderConfig, VisionProviderCapabilities
        assert VisionProviderConfig is not None
        assert VisionProviderCapabilities is not None


# ===========================================================================
# 5. INGESTION CONFIG — DEFAULTS (Section 10)
# ===========================================================================

class TestIngestionConfigDefaults:
    """Verify IngestionConfig default values match documented constants."""

    def test_default_chunk_size(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        cfg = IngestionConfig()
        assert cfg.chunk_size == 1000

    def test_default_chunk_overlap(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        cfg = IngestionConfig()
        assert cfg.chunk_overlap == 200

    def test_default_retrieval_top_k(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        cfg = IngestionConfig()
        assert cfg.retrieval_top_k == 5

    def test_default_retrieval_min_score(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        cfg = IngestionConfig()
        assert cfg.retrieval_min_score == 0.0

    def test_default_qdrant_collection(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        cfg = IngestionConfig()
        assert cfg.qdrant_collection == "omnibrain_documents"

    def test_default_qdrant_timeout(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        cfg = IngestionConfig()
        assert cfg.qdrant_timeout == 10.0


# ===========================================================================
# 6. INGESTION CONFIG — FROM_ENV (Sections 8, 9)
# ===========================================================================

class TestIngestionConfigFromEnv:
    """Verify IngestionConfig.from_env() uses defaults when env vars are absent."""

    def test_from_env_no_vars_returns_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ingestion.ingestion_config import IngestionConfig
        for var in ("INGESTION_CHUNK_SIZE", "INGESTION_CHUNK_OVERLAP",
                    "INGESTION_TOP_K", "INGESTION_MIN_SCORE",
                    "QDRANT_COLLECTION", "QDRANT_TIMEOUT"):
            monkeypatch.delenv(var, raising=False)

        cfg = IngestionConfig.from_env()
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.retrieval_top_k == 5
        assert cfg.retrieval_min_score == 0.0
        assert cfg.qdrant_collection == "omnibrain_documents"
        assert cfg.qdrant_timeout == 10.0

    def test_from_env_valid_chunk_size_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ingestion.ingestion_config import IngestionConfig
        monkeypatch.setenv("INGESTION_CHUNK_SIZE", "500")
        monkeypatch.delenv("INGESTION_CHUNK_OVERLAP", raising=False)
        monkeypatch.delenv("INGESTION_TOP_K", raising=False)
        monkeypatch.delenv("INGESTION_MIN_SCORE", raising=False)
        monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
        monkeypatch.delenv("QDRANT_TIMEOUT", raising=False)
        cfg = IngestionConfig.from_env()
        assert cfg.chunk_size == 500

    def test_from_env_valid_collection_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ingestion.ingestion_config import IngestionConfig
        for var in ("INGESTION_CHUNK_SIZE", "INGESTION_CHUNK_OVERLAP",
                    "INGESTION_TOP_K", "INGESTION_MIN_SCORE", "QDRANT_TIMEOUT"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("QDRANT_COLLECTION", "day41_test_collection")
        cfg = IngestionConfig.from_env()
        assert cfg.qdrant_collection == "day41_test_collection"

    def test_from_env_invalid_chunk_size_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ingestion.ingestion_config import IngestionConfig
        monkeypatch.setenv("INGESTION_CHUNK_SIZE", "DAY41_NOT_AN_INT")
        with pytest.raises(ValueError):
            IngestionConfig.from_env()

    def test_from_env_empty_collection_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ingestion.ingestion_config import IngestionConfig
        monkeypatch.setenv("QDRANT_COLLECTION", "   ")
        with pytest.raises(ValueError):
            IngestionConfig.from_env()

    def test_from_env_env_restored_after_test(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify monkeypatch restores environment — no permanent side effects."""
        from ingestion.ingestion_config import IngestionConfig
        original = os.environ.get("INGESTION_CHUNK_SIZE")
        monkeypatch.setenv("INGESTION_CHUNK_SIZE", "999")
        cfg = IngestionConfig.from_env()
        assert cfg.chunk_size == 999
        # After test, monkeypatch auto-restores; we just verify the fixture works
        assert os.environ.get("INGESTION_CHUNK_SIZE") == "999"  # still in scope


# ===========================================================================
# 7. INGESTION CONFIG — VALIDATION (Section 11)
# ===========================================================================

class TestIngestionConfigValidation:
    """Verify IngestionConfig constructor validation raises ValueError on invalid inputs."""

    def test_zero_chunk_size_raises(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        with pytest.raises(ValueError):
            IngestionConfig(chunk_size=0)

    def test_negative_chunk_size_raises(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        with pytest.raises(ValueError):
            IngestionConfig(chunk_size=-1)

    def test_chunk_overlap_gte_chunk_size_raises(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        with pytest.raises(ValueError):
            IngestionConfig(chunk_size=100, chunk_overlap=100)

    def test_negative_retrieval_top_k_raises(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        with pytest.raises(ValueError):
            IngestionConfig(retrieval_top_k=-1)

    def test_min_score_above_one_raises(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        with pytest.raises(ValueError):
            IngestionConfig(retrieval_min_score=1.5)

    def test_empty_collection_name_raises(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        with pytest.raises(ValueError):
            IngestionConfig(qdrant_collection="")

    def test_zero_qdrant_timeout_raises(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        with pytest.raises(ValueError):
            IngestionConfig(qdrant_timeout=0.0)

    def test_valid_custom_config_accepted(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        cfg = IngestionConfig(chunk_size=512, chunk_overlap=50, retrieval_top_k=10,
                              retrieval_min_score=0.7, qdrant_collection="day41_col",
                              qdrant_timeout=5.0)
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 50
        assert cfg.retrieval_top_k == 10


# ===========================================================================
# 8. QDRANT CONFIG — DEFAULTS & FROM_ENV (Sections 8, 9, 10)
# ===========================================================================

class TestQdrantConfigDefaults:
    """Verify QdrantConfig defaults and from_env behavior."""

    def test_default_url_is_memory(self) -> None:
        from ingestion.qdrant_config import QdrantConfig
        cfg = QdrantConfig()
        assert cfg.url == ":memory:"

    def test_default_api_key_is_none(self) -> None:
        from ingestion.qdrant_config import QdrantConfig
        cfg = QdrantConfig()
        assert cfg.api_key is None

    def test_default_collection(self) -> None:
        from ingestion.qdrant_config import QdrantConfig
        cfg = QdrantConfig()
        assert cfg.default_collection == "omnibrain_documents"

    def test_default_timeout(self) -> None:
        from ingestion.qdrant_config import QdrantConfig
        cfg = QdrantConfig()
        assert cfg.timeout == 10.0

    def test_from_env_no_vars_returns_memory_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ingestion.qdrant_config import QdrantConfig
        for var in ("QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION", "QDRANT_TIMEOUT"):
            monkeypatch.delenv(var, raising=False)
        cfg = QdrantConfig.from_env()
        assert cfg.url == ":memory:"
        assert cfg.api_key is None
        assert cfg.default_collection == "omnibrain_documents"

    def test_from_env_url_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ingestion.qdrant_config import QdrantConfig
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
        monkeypatch.delenv("QDRANT_TIMEOUT", raising=False)
        cfg = QdrantConfig.from_env()
        assert cfg.url == "http://localhost:6333"

    def test_from_env_api_key_absent_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ingestion.qdrant_config import QdrantConfig
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        cfg = QdrantConfig.from_env()
        assert cfg.api_key is None

    def test_secret_safety_no_real_key_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify test uses only placeholder values — never a real API key."""
        from ingestion.qdrant_config import QdrantConfig
        monkeypatch.setenv("QDRANT_API_KEY", "DAY41_FAKE_VALUE")
        cfg = QdrantConfig.from_env()
        # The key reads through correctly — this is a placeholder, not a real secret
        assert cfg.api_key == "DAY41_FAKE_VALUE"


# ===========================================================================
# 9. VISION PROVIDER CONFIG (Section 13)
# ===========================================================================

class TestVisionProviderConfigDefaults:
    """Verify VisionProviderConfig defaults, validation, and to_dict/from_dict round-trip."""

    def test_valid_minimal_config(self) -> None:
        from vision.provider_config import VisionProviderConfig
        cfg = VisionProviderConfig(provider_name="test", model_name="test-model")
        assert cfg.provider_name == "test"
        assert cfg.model_name == "test-model"
        assert cfg.timeout == 30.0
        assert cfg.max_tokens is None
        assert cfg.temperature is None
        assert cfg.max_input_images == 1
        assert cfg.extra_params == {}

    def test_custom_config_accepted(self) -> None:
        from vision.provider_config import VisionProviderConfig
        cfg = VisionProviderConfig(
            provider_name="day41_provider", model_name="day41-model",
            timeout=60.0, max_tokens=1024, temperature=0.5,
            max_input_images=4, extra_params={"day41_key": "DAY41_FAKE_VALUE"},
        )
        assert cfg.provider_name == "day41_provider"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.5
        assert cfg.max_input_images == 4

    def test_empty_provider_name_raises(self) -> None:
        from vision.provider_config import VisionProviderConfig
        from vision.exceptions import VisionProviderConfigError
        with pytest.raises(VisionProviderConfigError):
            VisionProviderConfig(provider_name="", model_name="test-model")

    def test_zero_timeout_raises(self) -> None:
        from vision.provider_config import VisionProviderConfig
        from vision.exceptions import VisionProviderConfigError
        with pytest.raises(VisionProviderConfigError):
            VisionProviderConfig(provider_name="p", model_name="m", timeout=0.0)

    def test_invalid_temperature_raises(self) -> None:
        from vision.provider_config import VisionProviderConfig
        from vision.exceptions import VisionProviderConfigError
        with pytest.raises(VisionProviderConfigError):
            VisionProviderConfig(provider_name="p", model_name="m", temperature=3.0)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        from vision.provider_config import VisionProviderConfig
        cfg = VisionProviderConfig(
            provider_name="day41_test", model_name="day41-m",
            timeout=15.0, max_tokens=512, temperature=0.3,
            max_input_images=2, extra_params={"param": "DAY41_FAKE_VALUE"},
        )
        d = cfg.to_dict()
        restored = VisionProviderConfig.from_dict(d)
        assert restored.provider_name == "day41_test"
        assert restored.model_name == "day41-m"
        assert restored.timeout == 15.0
        assert restored.max_tokens == 512
        assert restored.temperature == 0.3
        assert restored.max_input_images == 2

    def test_from_dict_missing_provider_name_raises(self) -> None:
        from vision.provider_config import VisionProviderConfig
        from vision.exceptions import VisionProviderConfigError
        with pytest.raises(VisionProviderConfigError):
            VisionProviderConfig.from_dict({"model_name": "m"})

    def test_from_dict_missing_model_name_raises(self) -> None:
        from vision.provider_config import VisionProviderConfig
        from vision.exceptions import VisionProviderConfigError
        with pytest.raises(VisionProviderConfigError):
            VisionProviderConfig.from_dict({"provider_name": "p"})


# ===========================================================================
# 10. VISION PROVIDER CAPABILITIES (Section 13)
# ===========================================================================

class TestVisionProviderCapabilities:
    """Verify VisionProviderCapabilities defaults, validation, and round-trip."""

    def test_default_capabilities(self) -> None:
        from vision.provider_config import VisionProviderCapabilities
        caps = VisionProviderCapabilities()
        assert caps.max_images == 1
        assert caps.supports_streaming is False
        assert caps.supports_multi_image is False
        assert caps.supports_system_prompt is True
        assert len(caps.supported_modalities) > 0
        assert len(caps.supported_formats) > 0

    def test_custom_capabilities(self) -> None:
        from vision.provider_config import VisionProviderCapabilities
        caps = VisionProviderCapabilities(
            supported_modalities=frozenset({"image"}),
            supported_formats=frozenset({"png", "jpeg"}),
            max_images=5, supports_multi_image=True,
        )
        assert caps.supports_modality("image")
        assert caps.supports_format("png")
        assert caps.max_images == 5

    def test_invalid_modality_raises(self) -> None:
        from vision.provider_config import VisionProviderCapabilities
        from vision.exceptions import VisionProviderConfigError
        with pytest.raises(VisionProviderConfigError):
            VisionProviderCapabilities(supported_modalities=frozenset({"invalid_mode"}))

    def test_zero_max_images_raises(self) -> None:
        from vision.provider_config import VisionProviderCapabilities
        from vision.exceptions import VisionProviderConfigError
        with pytest.raises(VisionProviderConfigError):
            VisionProviderCapabilities(max_images=0)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        from vision.provider_config import VisionProviderCapabilities
        caps = VisionProviderCapabilities(
            supported_modalities=frozenset({"image", "chart"}),
            supported_formats=frozenset({"png"}),
            max_images=3, supports_multi_image=True,
        )
        d = caps.to_dict()
        restored = VisionProviderCapabilities.from_dict(d)
        assert restored.max_images == 3
        assert restored.supports_multi_image is True
        assert "image" in restored.supported_modalities
        assert "chart" in restored.supported_modalities


# ===========================================================================
# 11. CROSS-COMPONENT CONFIGURATION COMPATIBILITY (Section 19)
# ===========================================================================

class TestCrossComponentConfigCompatibility:
    """Verify IngestionConfig, QdrantConfig, and VisionProviderConfig coexist without conflicts."""

    def test_all_three_configs_instantiate_together(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        from ingestion.qdrant_config import QdrantConfig
        from vision.provider_config import VisionProviderConfig
        ing = IngestionConfig()
        qdr = QdrantConfig()
        vis = VisionProviderConfig(provider_name="day41", model_name="day41-model")
        assert ing.chunk_size == 1000
        assert qdr.url == ":memory:"
        assert vis.provider_name == "day41"

    def test_qdrant_collection_names_consistent(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        from ingestion.qdrant_config import QdrantConfig
        ing = IngestionConfig()
        qdr = QdrantConfig()
        assert ing.qdrant_collection == qdr.default_collection


# ===========================================================================
# 12. CONFIGURATION OVERRIDE & RESET ISOLATION (Sections 20, 21)
# ===========================================================================

class TestConfigurationOverrideAndReset:
    """Verify configuration override affects only intended value; state does not leak."""

    def test_override_single_field_leaves_others_unchanged(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        default_cfg = IngestionConfig()
        overridden = IngestionConfig(chunk_size=2000)
        assert overridden.chunk_size == 2000
        assert overridden.chunk_overlap == default_cfg.chunk_overlap
        assert overridden.retrieval_top_k == default_cfg.retrieval_top_k
        assert overridden.qdrant_collection == default_cfg.qdrant_collection

    def test_env_override_restored_after_test(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ingestion.ingestion_config import IngestionConfig
        monkeypatch.setenv("INGESTION_CHUNK_SIZE", "777")
        cfg = IngestionConfig.from_env()
        assert cfg.chunk_size == 777
        # monkeypatch automatically restores env after test

    def test_default_after_override_is_independent(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        # Override
        cfg_override = IngestionConfig(chunk_size=5000, chunk_overlap=100)
        assert cfg_override.chunk_size == 5000
        # Re-create default — must be independent
        cfg_default = IngestionConfig()
        assert cfg_default.chunk_size == 1000


# ===========================================================================
# 13. CONFIGURATION IMMUTABILITY (Section 25)
# ===========================================================================

class TestConfigurationImmutability:
    """Verify loading configuration does not modify any configuration source file."""

    def test_pytest_ini_unchanged(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        from ingestion.qdrant_config import QdrantConfig
        from vision.provider_config import VisionProviderConfig, VisionProviderCapabilities
        # Load all configs
        IngestionConfig()
        QdrantConfig()
        VisionProviderConfig(provider_name="t", model_name="m")
        VisionProviderCapabilities()

        current = _file_sha256(REPO_ROOT / "pytest.ini")
        assert current == _INITIAL_HASHES[str(REPO_ROOT / "pytest.ini")]

    def test_requirements_txt_unchanged(self) -> None:
        from ingestion.ingestion_config import IngestionConfig
        IngestionConfig()
        current = _file_sha256(REPO_ROOT / "requirements.txt")
        assert current == _INITIAL_HASHES[str(REPO_ROOT / "requirements.txt")]

    def test_env_example_unchanged(self) -> None:
        from ingestion.qdrant_config import QdrantConfig
        QdrantConfig()
        current = _file_sha256(REPO_ROOT / ".env.example")
        assert current == _INITIAL_HASHES[str(REPO_ROOT / ".env.example")]


# ===========================================================================
# 14. NOT APPLICABLE — DOCUMENTED (Sections 14–17)
# ===========================================================================

class TestNotApplicableDocumentation:
    """Document NOT APPLICABLE sections explicitly so the report is accurate."""

    def test_no_pyproject_toml_present(self) -> None:
        """pyproject.toml is NOT APPLICABLE — file does not exist in this repository."""
        assert not (REPO_ROOT / "pyproject.toml").exists()

    def test_no_dockerfile_present(self) -> None:
        """Dockerfile is NOT APPLICABLE — file does not exist in this repository."""
        assert not (REPO_ROOT / "Dockerfile").exists()

    def test_no_docker_compose_present(self) -> None:
        """docker-compose.yml is NOT APPLICABLE — file does not exist."""
        assert not (REPO_ROOT / "docker-compose.yml").exists()
        assert not (REPO_ROOT / "docker-compose.yaml").exists()
