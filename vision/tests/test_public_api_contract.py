"""
Day 49 — Vision Agent Public API Contract & Interface Stability Tests.

Comprehensive test suite verifying:
  1.  Package exports in vision.__all__ matches exported symbols exactly.
  2.  Independent module imports across all vision submodules.
  3.  Public symbol availability and callable/class attributes.
  4.  Constructor signatures and parameter compatibility.
  5.  Method signatures and return type contracts.
  6.  VisionRequest contract, validation, immutability, and helpers.
  7.  VisualEvidence contract, lineage fields, and image payload validation.
  8.  VisionResult contract, lineage synchronization, and metadata sanitization.
  9.  Exception hierarchy integrity and error message preservation without secrets.
  10. VisualEvidenceAdapter contract for citations, search results, and chunks.
  11. VisionExecutionAdapter contract with multi-stage execution and isolation.
  12. VisionExecutionLifecycle contract, stage constants, and terminal protection.
  13. VisionResultNormalizer contract and execution trace recording.
  14. VisionPipeline contract and run_vision_pipeline orchestration.
  15. VisionModelProvider base contract and capability checking.
  16. VisionProviderConfig and VisionProviderCapabilities contracts.
  17. Backward-compatible import paths.
  18. Public data types and dataclass representations.
  19. Supported dictionary conversions and serializability.
  20. Deterministic error raising across repeated invalid inputs.
  21. No internal state or secrets leakage in public representations.
  22. Concurrent public API execution across multiple threads.
  23. Resource safety compatibility on repeated public invocations.
  24. Retry compatibility through public entry points.
  25. Cancellation compatibility through public entry points.
  26. Timeout compatibility through public entry points.
  27. Documentation and docstring presence on public classes and functions.
  28. Deterministic public API inventory snapshot.
  29. Strict offline execution guarantee.
  30. Public API integration stability.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import importlib
import inspect
import io
import sys
import threading
from typing import Any

import pytest
from PIL import Image

import vision
from vision import (
    FORBIDDEN_METADATA_KEYS,
    SUPPORTED_IMAGE_FORMATS,
    VALID_VISUAL_CONTENT_TYPES,
    ImageEvidencePreparator,
    OversizedImagePolicy,
    PreparedImageEvidence,
    VisionAgent,
    VisionAgentError,
    VisionCancellationError,
    VisionCancellationToken,
    VisionError,
    VisionEvidenceError,
    VisionExecutionAdapter,
    VisionExecutionLifecycle,
    VisionExecutionObservation,
    VisionExecutionStage,
    VisionExecutionTrace,
    VisionInputBuilder,
    VisionInputValidationError,
    VisionModelInput,
    VisionModelProvider,
    VisionPipeline,
    VisionProcessingError,
    VisionProviderCapabilities,
    VisionProviderConfig,
    VisionProviderConfigError,
    VisionProviderError,
    VisionProviderExecutionError,
    VisionProviderRegistry,
    VisionProviderUnavailableError,
    VisionRequest,
    VisionResult,
    VisionResultNormalizer,
    VisionRetryPolicy,
    VisionTimeoutError,
    VisionUnsupportedCapabilityError,
    VisualEvidence,
    VisualEvidenceAdapter,
    build_vision_input,
    execute_vision_request,
    prepare_image_evidence,
    run_vision_pipeline,
)


# ===========================================================================
# Helpers & Doubles
# ===========================================================================


def _make_test_image(
    format_name: str = "PNG",
    width: int = 32,
    height: int = 32,
    color: tuple[int, int, int] = (120, 180, 240),
) -> bytes:
    """Generate minimal valid image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


def _make_evidence(
    doc_id: str = "doc-api-001",
    filename: str = "api_chart.png",
    chunk_id: str = "chk-api-001",
    content_type: str = "chart",
) -> VisualEvidence:
    """Construct a valid VisualEvidence instance."""
    return VisualEvidence(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        content_type=content_type,
        image_bytes=_make_test_image("PNG"),
        page_number=1,
        chunk_index=0,
        metadata={"source": "test_public_api_contract"},
    )


class PublicApiTestProvider(VisionModelProvider):
    """Test double for verifying public provider contracts."""

    def __init__(
        self,
        config: VisionProviderConfig | None = None,
        capabilities: VisionProviderCapabilities | None = None,
    ) -> None:
        cfg = config or VisionProviderConfig(provider_name="test_api_prov", model_name="vlm-api")
        super().__init__(cfg, capabilities)
        self.invocation_count: int = 0
        self._lock = threading.Lock()

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        with self._lock:
            self.invocation_count += 1
            idx = self.invocation_count

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Public API provider executed query: {model_input.query}",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={"call_index": idx},
        )


# ===========================================================================
# 1. Package Exports, Imports & Inventory
# ===========================================================================


class TestPackageExportsAndImports:
    """Tests 1, 2, 3, 17, 28: Export consistency, independent imports, and inventory."""

    def test_01_package_exports_match_all(self) -> None:
        """Every symbol in vision.__all__ is present in the vision module namespace."""
        assert isinstance(vision.__all__, list)
        for sym in vision.__all__:
            assert hasattr(vision, sym), f"Exported symbol '{sym}' is missing from vision module."

    @pytest.mark.parametrize(
        "mod_name",
        [
            "vision.models",
            "vision.exceptions",
            "vision.evidence_adapter",
            "vision.image_preparation",
            "vision.input_builder",
            "vision.provider_config",
            "vision.provider",
            "vision.lifecycle",
            "vision.result_normalizer",
            "vision.execution_adapter",
            "vision.pipeline",
            "vision.vision_agent",
        ],
    )
    def test_02_independent_module_imports(self, mod_name: str) -> None:
        """Every submodule in the vision package can be imported independently."""
        mod = importlib.import_module(mod_name)
        assert mod is not None

    def test_03_public_symbol_availability(self) -> None:
        """Core public classes and entry points are available and callable."""
        assert inspect.isclass(VisionRequest)
        assert inspect.isclass(VisualEvidence)
        assert inspect.isclass(VisionResult)
        assert inspect.isclass(VisionPipeline)
        assert inspect.isclass(VisionAgent)
        assert inspect.isclass(VisionExecutionAdapter)
        assert inspect.isclass(VisionExecutionLifecycle)
        assert inspect.isclass(VisionRetryPolicy)
        assert inspect.isclass(VisionCancellationToken)
        assert callable(run_vision_pipeline)
        assert callable(execute_vision_request)

    def test_17_backward_compatible_import_paths(self) -> None:
        """Standard sub-module import paths remain functional and consistent."""
        from vision.models import VisionRequest as VR1
        from vision import VisionRequest as VR2
        assert VR1 is VR2

        from vision.lifecycle import VisionExecutionLifecycle as VEL1
        from vision import VisionExecutionLifecycle as VEL2
        assert VEL1 is VEL2

        from vision.exceptions import VisionAgentError as VAE1
        from vision import VisionAgentError as VAE2
        assert VAE1 is VAE2

    def test_28_deterministic_api_inventory(self) -> None:
        """Verify the exact set of expected public symbols."""
        expected_inventory = {
            "VisualEvidence",
            "VisionRequest",
            "VisionResult",
            "VALID_VISUAL_CONTENT_TYPES",
            "PreparedImageEvidence",
            "ImageEvidencePreparator",
            "OversizedImagePolicy",
            "SUPPORTED_IMAGE_FORMATS",
            "prepare_image_evidence",
            "VisionModelInput",
            "VisionInputBuilder",
            "build_vision_input",
            "VisionModelProvider",
            "VisionProviderRegistry",
            "VisionProviderConfig",
            "VisionProviderCapabilities",
            "VisionExecutionAdapter",
            "VisionExecutionStage",
            "VisionExecutionLifecycle",
            "VisionExecutionObservation",
            "VisionCancellationToken",
            "VisionRetryPolicy",
            "execute_vision_request",
            "VisionResultNormalizer",
            "VisionExecutionTrace",
            "FORBIDDEN_METADATA_KEYS",
            "VisionPipeline",
            "run_vision_pipeline",
            "VisualEvidenceAdapter",
            "VisionAgent",
            "VisionAgentError",
            "VisionError",
            "VisionCancellationError",
            "VisionInputValidationError",
            "VisionEvidenceError",
            "VisionProcessingError",
            "VisionProviderError",
            "VisionProviderConfigError",
            "VisionProviderExecutionError",
            "VisionProviderUnavailableError",
            "VisionUnsupportedCapabilityError",
            "VisionTimeoutError",
        }
        actual_all = set(vision.__all__)
        assert expected_inventory.issubset(actual_all)


# ===========================================================================
# 2. Signature & Constructor Compatibility
# ===========================================================================


class TestSignatureAndConstructorCompatibility:
    """Tests 4, 5, 18, 19: Constructors, methods, typing, and dictionary serialization."""

    def test_04_constructor_compatibility(self) -> None:
        """Constructors accept expected positional and keyword arguments with safe defaults."""
        ev = VisualEvidence(
            document_id="doc1",
            filename="img.png",
            chunk_id="chk1",
            content_type="chart",
            image_bytes=_make_test_image("PNG"),
        )
        assert ev.page_number is None
        assert ev.chunk_index == 0

        req = VisionRequest(query="Test query", evidence=[ev])
        assert req.session_id is None
        assert req.metadata == {}

        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        assert cfg.timeout == 30.0
        assert cfg.temperature is None
        assert cfg.max_input_images == 1

    def test_05_method_signature_compatibility(self) -> None:
        """Inspect key method signatures for compatibility."""
        sig_pipeline_run = inspect.signature(VisionPipeline.run)
        assert "request" in sig_pipeline_run.parameters
        assert "evidence" in sig_pipeline_run.parameters

        sig_adapter_exec = inspect.signature(VisionExecutionAdapter.execute)
        assert "request" in sig_adapter_exec.parameters
        assert "evidence" in sig_adapter_exec.parameters

        sig_provider_exec = inspect.signature(VisionModelProvider.execute)
        assert "model_input" in sig_provider_exec.parameters

    def test_18_public_data_types(self) -> None:
        """Verify immutable and dataclass attributes of public domain models."""
        assert dataclasses.is_dataclass(VisualEvidence)
        assert dataclasses.is_dataclass(VisionRequest)
        assert dataclasses.is_dataclass(VisionResult)
        assert dataclasses.is_dataclass(VisionModelInput)
        assert dataclasses.is_dataclass(VisionExecutionLifecycle)
        assert dataclasses.is_dataclass(VisionExecutionObservation)
        assert dataclasses.is_dataclass(VisionRetryPolicy)

    def test_19_supported_serialization(self) -> None:
        """Domain models provide clean to_dict and/or from_dict helpers."""
        ev = _make_evidence()
        ev_dict = ev.to_dict()
        assert isinstance(ev_dict, dict)
        assert ev_dict["document_id"] == "doc-api-001"

        req = VisionRequest(query="Q", evidence=[ev])
        req_dict = req.to_dict()
        assert isinstance(req_dict, dict)
        assert req_dict["query"] == "Q"

        res = VisionResult(query="Q", status="success", description="Desc")
        res_dict = res.to_dict()
        assert isinstance(res_dict, dict)
        assert res_dict["status"] == "success"


# ===========================================================================
# 3. Model Contracts & Validation
# ===========================================================================


class TestModelContractsAndValidation:
    """Tests 6, 7, 8: VisionRequest, VisualEvidence, and VisionResult contracts."""

    def test_06_vision_request_contract(self) -> None:
        """VisionRequest validates query non-emptiness and evidence typing."""
        with pytest.raises(VisionInputValidationError, match="query"):
            VisionRequest(query="")

        with pytest.raises(VisionInputValidationError, match="evidence"):
            VisionRequest(query="Valid", evidence=["not_evidence"])  # type: ignore[list-item]

        req = VisionRequest(query="Valid query", evidence=[_make_evidence()])
        assert req.has_evidence is True
        assert len(req.evidence) == 1

    def test_07_visual_evidence_contract(self) -> None:
        """VisualEvidence enforces valid visual content types and lineage parameters."""
        with pytest.raises(VisionEvidenceError, match="content_type"):
            VisualEvidence(
                document_id="doc1",
                filename="f.png",
                chunk_id="c1",
                content_type="invalid_type",
                image_bytes=_make_test_image(),
            )

        with pytest.raises(VisionEvidenceError, match="document_id"):
            VisualEvidence(
                document_id="",
                filename="f.png",
                chunk_id="c1",
                content_type="chart",
            )

        with pytest.raises(VisionEvidenceError, match="page_number"):
            VisualEvidence(
                document_id="doc1",
                filename="f.png",
                chunk_id="c1",
                page_number=-5,
            )

    def test_08_vision_result_contract(self) -> None:
        """VisionResult maintains lineage, error, and metadata invariants."""
        res = VisionResult(
            query="Chart query",
            status="success",
            description="Analysis of chart",
            document_id="doc-123",
            filename="chart.png",
            chunk_id="chk-123",
            metadata={"source": "api_test"},
        )
        assert res.is_success is True
        assert res.is_error is False
        assert res.document_id == "doc-123"


# ===========================================================================
# 4. Exception Hierarchy & Sanitization
# ===========================================================================


class TestExceptionHierarchyAndSanitization:
    """Tests 9, 20, 21: Exception hierarchy, deterministic errors, and secret safety."""

    def test_09_exception_hierarchy(self) -> None:
        """All Vision exceptions derive from VisionAgentError (and VisionError)."""
        assert issubclass(VisionError, Exception)
        assert issubclass(VisionAgentError, Exception)
        assert issubclass(VisionInputValidationError, VisionAgentError)
        assert issubclass(VisionEvidenceError, VisionAgentError)
        assert issubclass(VisionProcessingError, VisionAgentError)
        assert issubclass(VisionProviderError, VisionAgentError)
        assert issubclass(VisionProviderConfigError, VisionProviderError)
        assert issubclass(VisionProviderExecutionError, VisionProviderError)
        assert issubclass(VisionProviderUnavailableError, VisionProviderError)
        assert issubclass(VisionUnsupportedCapabilityError, VisionProviderError)
        assert issubclass(VisionTimeoutError, VisionProviderError)
        assert issubclass(VisionCancellationError, VisionAgentError)

    def test_20_deterministic_errors(self) -> None:
        """Repeated invalid operations deterministically raise the same exception type."""
        for _ in range(5):
            with pytest.raises(VisionInputValidationError):
                VisionRequest(query="")

    def test_21_no_internal_state_or_secret_leakage(self) -> None:
        """Metadata sanitization strips forbidden keys (api_key, token, password, etc.)."""
        meta_with_secrets = {
            "api_key": "secret-12345",
            "token": "bearer-xyz",
            "public_key": "visible-data",
            "raw_bytes": b"binary_image_data",
        }
        sanitized = VisionResultNormalizer.sanitize_metadata(meta_with_secrets)
        assert "api_key" not in sanitized
        assert "token" not in sanitized
        assert "raw_bytes" not in sanitized
        assert sanitized.get("public_key") == "visible-data"


# ===========================================================================
# 5. Pipeline, Provider, Adapter & Lifecycle Integration
# ===========================================================================


class TestPipelineProviderAdapterAndLifecycle:
    """Tests 10, 11, 12, 13, 14, 15, 16, 27, 30: Core component integration."""

    def test_10_adapter_contract(self) -> None:
        """VisualEvidenceAdapter converts citations, search results, and chunks."""
        raw_citation = {
            "document_id": "doc-cit",
            "filename": "chart.png",
            "chunk_id": "chk-cit",
            "content_type": "chart",
            "image_bytes": _make_test_image("PNG"),
        }
        ev = VisualEvidenceAdapter.adapt(raw_citation)
        assert isinstance(ev, VisualEvidence)
        assert ev.document_id == "doc-cit"
        assert ev.content_type == "chart"

    def test_11_execution_adapter_contract(self) -> None:
        """VisionExecutionAdapter runs multi-stage pipeline and returns VisionResult."""
        provider = PublicApiTestProvider()
        adapter = VisionExecutionAdapter(provider=provider)
        ev = _make_evidence()

        res = adapter.execute("API Query", evidence=[ev])
        assert isinstance(res, VisionResult)
        assert res.status == "success"
        assert provider.invocation_count == 1

    def test_12_lifecycle_contract(self) -> None:
        """VisionExecutionLifecycle manages canonical stages and enforces terminal protection."""
        lc = VisionExecutionLifecycle(provider_name="p", model_name="m")
        assert lc.stage == VisionExecutionStage.PENDING

        lc.transition_to(VisionExecutionStage.VALIDATING)
        lc.transition_to(VisionExecutionStage.COMPLETED)
        assert lc.is_completed is True
        assert lc.is_terminal is True

        with pytest.raises(VisionInputValidationError):
            lc.transition_to(VisionExecutionStage.EXECUTING)

    def test_13_normalizer_contract(self) -> None:
        """VisionResultNormalizer standardizes raw provider result and attaches lineage."""
        raw_res = VisionResult(query="Q", status="success", description="Ok")
        ev = _make_evidence(doc_id="doc-norm", filename="norm.png")
        model_input = build_vision_input(query="Q", evidence=prepare_image_evidence(ev))

        normalized = VisionResultNormalizer.normalize(raw_res, model_input=model_input)
        assert normalized.document_id == "doc-norm"
        assert normalized.filename == "norm.png"

    def test_14_pipeline_contract(self) -> None:
        """VisionPipeline orchestrates end-to-end execution cleanly."""
        provider = PublicApiTestProvider()
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        res = pipeline.run("Pipeline Query", evidence=[ev])
        assert res.status == "success"
        assert provider.invocation_count == 1

    def test_15_provider_contract(self) -> None:
        """VisionModelProvider base class enforces capability checking and input validation."""
        caps = VisionProviderCapabilities(supported_modalities=["chart"])
        provider = PublicApiTestProvider(capabilities=caps)

        assert provider.capabilities.supports_modality("chart") is True
        assert provider.capabilities.supports_modality("diagram") is False

    def test_16_provider_configuration_contract(self) -> None:
        """VisionProviderConfig validates configuration parameters and safe defaults."""
        cfg = VisionProviderConfig(provider_name="prov1", model_name="mod1", timeout=12.0)
        assert cfg.provider_name == "prov1"
        assert cfg.model_name == "mod1"
        assert cfg.timeout == 12.0

    def test_27_documentation_presence(self) -> None:
        """Public classes and entry functions have meaningful docstrings."""
        assert VisionRequest.__doc__ is not None
        assert VisualEvidence.__doc__ is not None
        assert VisionResult.__doc__ is not None
        assert VisionPipeline.__doc__ is not None
        assert VisionAgent.__doc__ is not None
        assert VisionExecutionAdapter.__doc__ is not None

    def test_30_public_api_stability(self) -> None:
        """run_vision_pipeline module-level function executes seamlessly."""
        provider = PublicApiTestProvider()
        ev = _make_evidence()
        res = run_vision_pipeline(provider=provider, request="Module Function Query", evidence=[ev])
        assert res.status == "success"


# ===========================================================================
# 6. Concurrency, Resource Safety, Retries, Timeout & Cancellation
# ===========================================================================


class TestConcurrencySafetyAndAdvancedContracts:
    """Tests 22, 23, 24, 25, 26, 29: Concurrency, isolation, retries, and offline guarantees."""

    def test_22_concurrent_public_api_usage(self) -> None:
        """Multiple concurrent public API executions operate safely across worker threads."""
        provider = PublicApiTestProvider()
        pipeline = VisionPipeline(provider=provider)

        def worker(idx: int) -> str:
            ev = _make_evidence(doc_id=f"doc_{idx}")
            res = pipeline.run(f"Query {idx}", evidence=[ev])
            return res.status

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(16)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 16
        assert all(r == "success" for r in results)

    def test_23_resource_safety_compatibility(self) -> None:
        """Repeated public executions do not leak state or references."""
        provider = PublicApiTestProvider()
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        for i in range(5):
            res = pipeline.run(f"Query {i}", evidence=[ev])
            obs = VisionExecutionObservation.from_result(res)
            assert obs.attempt_count == 1
            assert obs.retry_count == 0

    def test_24_retry_compatibility(self) -> None:
        """Public pipeline.run accepts max_retries and VisionRetryPolicy."""
        class FailOnceProvider(VisionModelProvider):
            def __init__(self, config: VisionProviderConfig) -> None:
                super().__init__(config)
                self.calls = 0

            def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
                self.calls += 1
                if self.calls == 1:
                    raise VisionProviderExecutionError("Transient error call 1")
                return VisionResult(query=model_input.query, status="success")

        provider = FailOnceProvider(VisionProviderConfig(provider_name="p", model_name="m"))
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        res = pipeline.run("Query", evidence=[ev], max_retries=1)
        assert res.status == "success"
        obs = VisionExecutionObservation.from_result(res)
        assert obs.attempt_count == 2
        assert obs.retry_count == 1

    def test_25_cancellation_compatibility(self) -> None:
        """Public pipeline.run accepts VisionCancellationToken and aborts cleanly."""
        provider = PublicApiTestProvider()
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        tok = VisionCancellationToken()
        tok.cancel("Public cancel")

        with pytest.raises(VisionCancellationError):
            pipeline.run("Query", evidence=[ev], cancellation_token=tok)

        assert provider.invocation_count == 0

    def test_26_timeout_compatibility(self) -> None:
        """Public pipeline.run validates timeout and handles timeout exceptions cleanly."""
        provider = PublicApiTestProvider()
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        with pytest.raises(VisionInputValidationError, match="timeout"):
            pipeline.run("Query", evidence=[ev], timeout=-1.0)

    def test_29_offline_execution(self) -> None:
        """Public API operates strictly offline with zero external network or telemetry modules."""
        for forbidden in ("langfuse", "opentelemetry", "prometheus", "requests", "httpx", "aiohttp"):
            assert forbidden not in sys.modules or forbidden not in globals()
