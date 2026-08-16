"""
Lightweight health and readiness checks for the OmniBrain ingestion pipeline.

Provides deterministic, pure-Python health/readiness results that can be
evaluated before accepting an ingestion request.  No external APIs are called,
no documents or vectors are modified, and no exceptions leak from individual
check functions.

Concepts
--------
Health  — can the ingestion subsystem operate correctly?
            (modules importable, standard-library OK, config constructible)

Readiness — is the subsystem sufficiently configured to accept a real
            ingestion request right now?
            (valid IngestionConfig provided, usable EmbeddingProvider present)

Both are represented by a single :class:`IngestionHealthResult` dataclass so
callers can inspect both dimensions at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Health-check names ────────────────────────────────────────────────────

CHECK_MODULES = "modules_importable"
CHECK_CONFIG = "configuration_valid"
CHECK_PROVIDER = "embedding_provider_available"
CHECK_QDRANT_CONFIG = "qdrant_config_valid"

_PASS = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"


# ── Result dataclass ──────────────────────────────────────────────────────


@dataclass
class IngestionHealthResult:
    """Structured result of a health / readiness evaluation.

    Attributes:
        healthy: True when all *health* checks passed (modules available,
            default config constructible, optional dependencies present).
        ready: True when all *readiness* checks passed (valid config AND
            usable embedding provider supplied).
        status: Human-readable summary string
            ('READY', 'HEALTHY', 'DEGRADED', 'UNHEALTHY').
        checks: Mapping of check-name -> 'PASS' | 'FAIL' | 'SKIP'.
        errors: Ordered list of failure/error messages from failed checks.
    """

    healthy: bool = True
    ready: bool = True
    status: str = "READY"
    checks: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    # ── Convenience helpers ───────────────────────────────────────────────

    def is_healthy(self) -> bool:
        """Return True when all health checks passed."""
        return self.healthy

    def is_ready(self) -> bool:
        """Return True when all readiness checks passed."""
        return self.ready

    def failed_checks(self) -> list[str]:
        """Return names of checks that produced a 'FAIL' result."""
        return [name for name, result in self.checks.items() if result == _FAIL]

    def passed_checks(self) -> list[str]:
        """Return names of checks that produced a 'PASS' result."""
        return [name for name, result in self.checks.items() if result == _PASS]

    def skipped_checks(self) -> list[str]:
        """Return names of checks that were skipped ('SKIP')."""
        return [name for name, result in self.checks.items() if result == _SKIP]


# ── Individual check helpers (never raise) ────────────────────────────────


def _check_modules() -> tuple[str, str]:
    """Verify all required ingestion modules can be imported.

    Returns a (PASS|FAIL, error_message) tuple.
    """
    required = [
        "ingestion.chunker",
        "ingestion.chunk_validator",
        "ingestion.embedding_preparation",
        "ingestion.embedding_generator",
        "ingestion.pdf_ingestion_pipeline",
        "ingestion.pdf_text_extractor",
        "ingestion.ingestion_config",
        "ingestion.ingestion_errors",
        "ingestion.ingestion_status",
        "ingestion.ingestion_metrics",
        "ingestion.ingestion_logging",
        "ingestion.ingestion_service",
    ]
    missing: list[str] = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError as exc:
            missing.append(f"{mod}: {exc}")
    if missing:
        return _FAIL, "Missing modules: " + "; ".join(missing)
    return _PASS, ""


def _check_config(config: Any | None) -> tuple[str, str]:
    """Verify that the provided or default IngestionConfig is valid.

    If *config* is None a default ``IngestionConfig()`` is constructed.
    """
    try:
        from ingestion.ingestion_config import IngestionConfig

        if config is None:
            IngestionConfig()  # verify the default is constructible
        elif not isinstance(config, IngestionConfig):
            return _FAIL, f"'config' must be an IngestionConfig instance, got {type(config).__name__!r}."
        # IngestionConfig already validates in __post_init__; if we reach
        # here with a non-None config it has already been validated.
        return _PASS, ""
    except Exception as exc:  # noqa: BLE001
        return _FAIL, f"Configuration invalid: {exc}"


def _check_provider(provider: Any | None) -> tuple[str, str]:
    """Verify that the embedding provider exposes a usable interface.

    Does NOT call embed() or emit any real vectors.
    """
    if provider is None:
        return _SKIP, ""  # provider is optional for health; fail for readiness
    has_embed = hasattr(provider, "embed") and callable(getattr(provider, "embed"))
    has_batch = hasattr(provider, "embed_batch") and callable(getattr(provider, "embed_batch"))
    if not (has_embed or has_batch):
        return _FAIL, (
            "Embedding provider must implement 'embed(text)' or 'embed_batch(texts)'. "
            f"Got: {type(provider).__name__!r}."
        )
    return _PASS, ""


def _check_qdrant_config(qdrant_config: Any | None) -> tuple[str, str]:
    """Verify that the Qdrant configuration is valid.

    If *qdrant_config* is None a default ``QdrantConfig()`` is constructed.
    No real Qdrant connection is opened.
    """
    try:
        from ingestion.qdrant_config import QdrantConfig

        if qdrant_config is None:
            QdrantConfig()  # verify default is constructible
        elif not isinstance(qdrant_config, QdrantConfig):
            return _FAIL, (
                f"'qdrant_config' must be a QdrantConfig instance, got {type(qdrant_config).__name__!r}."
            )
        return _PASS, ""
    except Exception as exc:  # noqa: BLE001
        return _FAIL, f"QdrantConfig invalid: {exc}"


# ── Public API ────────────────────────────────────────────────────────────


def check_ingestion_health(
    config: Any | None = None,
    qdrant_config: Any | None = None,
) -> IngestionHealthResult:
    """Run all *health* checks for the ingestion subsystem.

    Health checks verify that the subsystem can operate correctly
    independently of any runtime request parameters.

    Checks performed:
    1. ``modules_importable``   — required ingestion modules can be imported.
    2. ``configuration_valid``  — IngestionConfig can be built with *config*
                                  or with safe defaults.
    3. ``qdrant_config_valid``  — QdrantConfig can be built with *qdrant_config*
                                  or with safe defaults.

    No embedding provider is required; no external API is called;
    no documents or vectors are touched.

    Args:
        config: Optional IngestionConfig to validate.  When None the default
            config is validated instead.
        qdrant_config: Optional QdrantConfig to validate.  When None the
            default config is validated instead.

    Returns:
        :class:`IngestionHealthResult` with ``healthy`` set to True only if
        all checks pass.
    """
    result = IngestionHealthResult()

    # 1. Modules
    mod_status, mod_err = _check_modules()
    result.checks[CHECK_MODULES] = mod_status
    if mod_status == _FAIL:
        result.errors.append(mod_err)
        result.healthy = False

    # 2. Configuration
    cfg_status, cfg_err = _check_config(config)
    result.checks[CHECK_CONFIG] = cfg_status
    if cfg_status == _FAIL:
        result.errors.append(cfg_err)
        result.healthy = False

    # 3. Qdrant config
    qcfg_status, qcfg_err = _check_qdrant_config(qdrant_config)
    result.checks[CHECK_QDRANT_CONFIG] = qcfg_status
    if qcfg_status == _FAIL:
        result.errors.append(qcfg_err)
        result.healthy = False

    # Provider is not checked in health — only in readiness
    result.checks[CHECK_PROVIDER] = _SKIP

    _assign_status(result)
    return result


def check_ingestion_readiness(
    provider: Any | None = None,
    config: Any | None = None,
    qdrant_config: Any | None = None,
) -> IngestionHealthResult:
    """Run all *readiness* checks for the ingestion subsystem.

    Readiness checks verify that the subsystem is sufficiently configured
    to accept and process an ingestion request right now.

    Checks performed:
    1. ``modules_importable``       — required ingestion modules can be imported.
    2. ``configuration_valid``      — IngestionConfig is valid.
    3. ``embedding_provider_available`` — provider implements the embedding interface.
    4. ``qdrant_config_valid``      — QdrantConfig is valid.

    No external API is called; no documents or vectors are touched.

    Args:
        provider: EmbeddingProvider instance to validate (mandatory for readiness).
        config: Optional IngestionConfig to validate.
        qdrant_config: Optional QdrantConfig to validate.

    Returns:
        :class:`IngestionHealthResult` with ``ready`` set to True only if
        all checks pass, including a non-None provider.
    """
    result = IngestionHealthResult()

    # 1. Modules
    mod_status, mod_err = _check_modules()
    result.checks[CHECK_MODULES] = mod_status
    if mod_status == _FAIL:
        result.errors.append(mod_err)
        result.healthy = False

    # 2. Configuration
    cfg_status, cfg_err = _check_config(config)
    result.checks[CHECK_CONFIG] = cfg_status
    if cfg_status == _FAIL:
        result.errors.append(cfg_err)
        result.healthy = False

    # 3. Embedding provider (mandatory for readiness)
    if provider is None:
        result.checks[CHECK_PROVIDER] = _FAIL
        result.errors.append("No embedding provider supplied — cannot accept ingestion requests.")
        result.ready = False
    else:
        prov_status, prov_err = _check_provider(provider)
        result.checks[CHECK_PROVIDER] = prov_status
        if prov_status == _FAIL:
            result.errors.append(prov_err)
            result.ready = False

    # 4. Qdrant config
    qcfg_status, qcfg_err = _check_qdrant_config(qdrant_config)
    result.checks[CHECK_QDRANT_CONFIG] = qcfg_status
    if qcfg_status == _FAIL:
        result.errors.append(qcfg_err)
        result.ready = False

    # Propagate health → readiness: not healthy ⇒ not ready
    if not result.healthy:
        result.ready = False

    _assign_status(result)
    return result


# ── Status assignment ─────────────────────────────────────────────────────


def _assign_status(result: IngestionHealthResult) -> None:
    """Set result.status based on healthy/ready flags."""
    if result.healthy and result.ready:
        result.status = "READY"
    elif result.healthy and not result.ready:
        result.status = "HEALTHY"
    elif not result.healthy and len(result.errors) < len(result.checks):
        result.status = "DEGRADED"
    else:
        result.status = "UNHEALTHY"
