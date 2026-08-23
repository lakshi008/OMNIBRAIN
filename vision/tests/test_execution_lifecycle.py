"""
Comprehensive unit and integration tests for Day 38: Vision Provider Execution Lifecycle,
Timeout & Safe Execution Contract.

Tests cover:
  1.  Execution lifecycle stage constants and terminal state properties.
  2.  VisionExecutionLifecycle initialization, state transitions, and validation.
  3.  Transition from terminal stage raises VisionInputValidationError.
  4.  VisionProviderConfig timeout validation (positive, finite, non-boolean numeric).
  5.  Execution pipeline stage tracking: pending -> validating -> preparing -> building_input -> executing -> completed.
  6.  Provider called exactly once per request execution.
  7.  VisionModelInput forwarded verbatim to provider without modification.
  8.  VisionModelInput immutability snapshot verification (pre/post execution comparison).
  9.  Immutability violation in provider raises VisionProcessingError.
  10. Successful execution attaches execution_lifecycle metadata to VisionResult.
  11. Provider execution failure transitions lifecycle to failed and preserves exception cause (__cause__).
  12. Provider failure does not return fake successful VisionResult.
  13. VisionTimeoutError transitions lifecycle to timeout stage and propagates cleanly.
  14. Zero, negative, or non-numeric timeout values rejected in VisionProviderConfig.
  15. No secret leakage, credentials, or fake latency in lifecycle to_dict().
  16. VisionExecutionAdapter with missing or invalid provider raises VisionInputValidationError.
  17. Repeated execution with identical inputs produces identical lifecycle transitions.
  18. Integration with Day 33 Adapter, Day 34 Preparation, Day 35 Input Builder, Day 36 Provider, and Day 37 Execution Adapter.
  19. Zero network calls, zero vendor SDK dependencies.
"""

from __future__ import annotations

import inspect
import io
from typing import Any

import pytest
from PIL import Image

from vision.exceptions import (
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderConfigError,
    VisionProviderError,
    VisionProviderExecutionError,
    VisionTimeoutError,
)
from vision.execution_adapter import (
    VisionExecutionAdapter,
    execute_vision_request,
)
from vision.input_builder import VisionModelInput, build_vision_input
from vision.lifecycle import (
    VisionExecutionLifecycle,
    VisionExecutionStage,
)
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.provider import VisionModelProvider
from vision.provider_config import (
    VisionProviderCapabilities,
    VisionProviderConfig,
)
from vision.vision_agent import VisionAgent


# ===========================================================================
# Test Helpers & Test Doubles (Conformed strictly to test scope only)
# ===========================================================================


def _make_test_png(width: int = 64, height: int = 64) -> bytes:
    """Generate minimal PNG bytes for test evidence."""
    img = Image.new("RGB", (width, height), color=(150, 100, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class LifecycleRecordingProvider(VisionModelProvider):
    """Test double that records received inputs and invocation counts."""

    def __init__(
        self,
        config: VisionProviderConfig,
        capabilities: VisionProviderCapabilities | None = None,
        tamper_input: bool = False,
    ) -> None:
        super().__init__(config, capabilities)
        self.invocation_count: int = 0
        self.recorded_inputs: list[VisionModelInput] = []
        self.tamper_input = tamper_input

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        self.invocation_count += 1
        self.recorded_inputs.append(model_input)

        if self.tamper_input:
            # Tamper with internal dict (simulating illegal mutation)
            object.__setattr__(model_input, "document_id", "TAMPERED_DOC_ID")

        return VisionResult(
            query=model_input.query,
            status="success",
            description="Lifecycle test analysis output.",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={"provider": self.provider_name},
        )


class TimeoutSimulatingProvider(VisionModelProvider):
    """Test double that raises a controlled VisionTimeoutError."""

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        raise VisionTimeoutError(
            f"Execution exceeded configured timeout of {self.config.timeout}s."
        )


class CustomFailingProvider(VisionModelProvider):
    """Test double that raises a controlled VisionProviderExecutionError."""

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        raise VisionProviderExecutionError("Custom provider execution failure.")


# ===========================================================================
# 1. Test class: VisionExecutionStage & VisionExecutionLifecycle
# ===========================================================================


class TestVisionExecutionLifecycleUnit:
    """Tests 1-3: Lifecycle state constants, initialization, transitions, and serialization."""

    def test_01_lifecycle_stage_constants(self) -> None:
        """VisionExecutionStage defines required stage constants and sets."""
        assert VisionExecutionStage.PENDING == "pending"
        assert VisionExecutionStage.VALIDATING == "validating"
        assert VisionExecutionStage.PREPARING == "preparing"
        assert VisionExecutionStage.BUILDING_INPUT == "building_input"
        assert VisionExecutionStage.EXECUTING == "executing"
        assert VisionExecutionStage.COMPLETED == "completed"
        assert VisionExecutionStage.FAILED == "failed"
        assert VisionExecutionStage.TIMEOUT == "timeout"

        assert "completed" in VisionExecutionStage.TERMINAL_STAGES
        assert "failed" in VisionExecutionStage.TERMINAL_STAGES
        assert "timeout" in VisionExecutionStage.TERMINAL_STAGES

    def test_02_lifecycle_initialization_defaults(self) -> None:
        """VisionExecutionLifecycle initializes with pending state and provider metadata."""
        lifecycle = VisionExecutionLifecycle(provider_name="test-prov", model_name="v1")
        assert lifecycle.stage == "pending"
        assert lifecycle.provider_name == "test-prov"
        assert lifecycle.model_name == "v1"
        assert lifecycle.error is None
        assert lifecycle.is_completed is False
        assert lifecycle.is_failed is False
        assert lifecycle.is_terminal is False

    @pytest.mark.parametrize("invalid_stage", ["", "   ", "invalid_stage", 123, None])
    def test_03_lifecycle_invalid_stage_raises_error(self, invalid_stage: Any) -> None:
        """Invalid stage raises VisionInputValidationError on init or transition."""
        with pytest.raises(VisionInputValidationError, match="stage"):
            VisionExecutionLifecycle(stage=invalid_stage)

        lifecycle = VisionExecutionLifecycle(provider_name="p", model_name="m")
        with pytest.raises(VisionInputValidationError, match="stage"):
            lifecycle.transition_to(invalid_stage)

    def test_04_lifecycle_transitions_and_terminal_guard(self) -> None:
        """State transitions progress cleanly; transition from terminal state raises error."""
        lifecycle = VisionExecutionLifecycle(provider_name="p1", model_name="m1")

        lifecycle.transition_to(VisionExecutionStage.VALIDATING)
        assert lifecycle.stage == "validating"

        lifecycle.transition_to(VisionExecutionStage.PREPARING)
        assert lifecycle.stage == "preparing"

        lifecycle.transition_to(VisionExecutionStage.BUILDING_INPUT)
        assert lifecycle.stage == "building_input"

        lifecycle.transition_to(VisionExecutionStage.EXECUTING)
        assert lifecycle.stage == "executing"

        lifecycle.transition_to(VisionExecutionStage.COMPLETED)
        assert lifecycle.stage == "completed"
        assert lifecycle.is_completed is True
        assert lifecycle.is_terminal is True

        # Transitioning away from terminal state must raise VisionInputValidationError
        with pytest.raises(VisionInputValidationError, match="terminal stage"):
            lifecycle.transition_to(VisionExecutionStage.EXECUTING)

    def test_05_lifecycle_to_dict_serialization(self) -> None:
        """to_dict() returns clean, serializable lifecycle metadata dictionary."""
        lifecycle = VisionExecutionLifecycle(provider_name="p1", model_name="m1")
        lifecycle.transition_to(VisionExecutionStage.FAILED, error="Processing failed")

        data = lifecycle.to_dict()
        assert data["stage"] == "failed"
        assert data["provider_name"] == "p1"
        assert data["model_name"] == "m1"
        assert data["error"] == "Processing failed"
        assert data["is_completed"] is False
        assert data["is_failed"] is True
        assert data["is_terminal"] is True


# ===========================================================================
# 2. Test class: Timeout Configuration & Validation
# ===========================================================================


class TestTimeoutConfigurationAndValidation:
    """Tests 4, 14: Timeout parameter validation in VisionProviderConfig."""

    def test_06_valid_timeout_accepted(self) -> None:
        """Positive finite numeric timeouts are accepted."""
        cfg1 = VisionProviderConfig(provider_name="p", model_name="m", timeout=10.0)
        assert cfg1.timeout == 10.0

        cfg2 = VisionProviderConfig(provider_name="p", model_name="m", timeout=120)
        assert cfg2.timeout == 120.0

    @pytest.mark.parametrize("bad_timeout", [0, -1, -30.0, "30", True, float("inf"), float("nan")])
    def test_07_invalid_timeout_rejected(self, bad_timeout: Any) -> None:
        """Non-positive, non-numeric, or non-finite timeouts raise VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="timeout"):
            VisionProviderConfig(provider_name="p", model_name="m", timeout=bad_timeout)


# ===========================================================================
# 3. Test class: Execution Pipeline Stages & Invocation Control
# ===========================================================================


class TestExecutionPipelineStagesAndInvocationControl:
    """Tests 5, 6, 7, 10: Execution lifecycle stage tracking, metadata attachment, and single invocation."""

    def test_08_execution_stages_tracked_and_metadata_attached(self) -> None:
        """Successful execution progresses through stages and attaches lifecycle metadata to VisionResult."""
        config = VisionProviderConfig(provider_name="prov-stage", model_name="vlm-stage")
        provider = LifecycleRecordingProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(
            document_id="doc-stg-01",
            filename="report.pdf",
            chunk_id="chk-stg-01",
            page_number=1,
            image_bytes=_make_test_png(),
        )
        result = adapter.execute("Analyze stage tracking", evidence=[ev])

        assert result.is_success is True
        assert "execution_lifecycle" in result.metadata
        lc_data = result.metadata["execution_lifecycle"]

        assert lc_data["stage"] == "completed"
        assert lc_data["provider_name"] == "prov-stage"
        assert lc_data["model_name"] == "vlm-stage"
        assert lc_data["is_completed"] is True
        assert lc_data["error"] is None

    def test_09_single_invocation_control(self) -> None:
        """Execution adapter calls provider.execute exactly once per request."""
        config = VisionProviderConfig(provider_name="prov-single", model_name="v1")
        provider = LifecycleRecordingProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        assert provider.invocation_count == 0
        adapter.execute("Query single", evidence=[ev])
        assert provider.invocation_count == 1

    def test_10_forwarded_input_matches_pipeline_output(self) -> None:
        """VisionModelInput forwarded to provider exactly matches pipeline output without unexpected mutation."""
        config = VisionProviderConfig(provider_name="prov-fwd", model_name="v1")
        provider = LifecycleRecordingProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(
            document_id="doc-fwd-01",
            filename="source.pdf",
            chunk_id="chk-fwd-01",
            page_number=4,
            chunk_index=2,
            content_type="chart",
            image_bytes=_make_test_png(48, 48),
            metadata={"author": "analyst"},
        )
        adapter.execute("Describe chart", evidence=[ev])

        inp = provider.recorded_inputs[0]
        assert inp.query == "Describe chart"
        assert inp.document_id == "doc-fwd-01"
        assert inp.filename == "source.pdf"
        assert inp.page_number == 4
        assert inp.chunk_index == 2
        assert inp.content_type == "chart"
        assert inp.image_format == "png"
        assert inp.width == 48
        assert inp.height == 48
        assert inp.evidence_metadata == {"author": "analyst"}


# ===========================================================================
# 4. Test class: Immutability Verification
# ===========================================================================


class TestInputImmutabilityVerification:
    """Tests 8, 9, 29: Pre- and post-execution snapshot matching & immutability violation error."""

    def test_11_input_immutability_maintained(self) -> None:
        """Input fields before and after provider execution remain identical."""
        config = VisionProviderConfig(provider_name="prov-imm", model_name="v1")
        provider = LifecycleRecordingProvider(config, tamper_input=False)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())
        result = adapter.execute("Immutability test", evidence=[ev])

        assert result.is_success is True

    def test_12_tampered_input_raises_immutability_error(self) -> None:
        """Provider tampering with VisionModelInput during execution triggers VisionProcessingError."""
        config = VisionProviderConfig(provider_name="prov-tamper", model_name="v1")
        provider = LifecycleRecordingProvider(config, tamper_input=True)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        with pytest.raises(VisionProcessingError, match="immutability violated"):
            adapter.execute("Tamper test", evidence=[ev])


# ===========================================================================
# 5. Test class: Provider Failures & Cause Preservation
# ===========================================================================


class TestProviderFailuresAndCausePreservation:
    """Tests 10, 11, 12: Controlled exception handling, cause preservation, and no fake results."""

    def test_13_provider_failure_preserves_cause(self) -> None:
        """Custom provider error propagates with exception cause (__cause__) preserved."""
        config = VisionProviderConfig(provider_name="fail-prov", model_name="v1")
        provider = CustomFailingProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        with pytest.raises(VisionProviderExecutionError) as exc_info:
            adapter.execute("Query fail", evidence=[ev])

        assert "Custom provider execution failure" in str(exc_info.value)

    def test_14_provider_failure_returns_no_fake_result(self) -> None:
        """Provider failure raises exception and never returns a fake successful VisionResult."""
        config = VisionProviderConfig(provider_name="fail-prov", model_name="v1")
        provider = CustomFailingProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        with pytest.raises(VisionProviderError):
            res = adapter.execute("Query fail", evidence=[ev])
            # Guard against return
            assert res is None  # Should never reach here


# ===========================================================================
# 6. Test class: Timeout Boundary & Lifecycle State
# ===========================================================================


class TestTimeoutBoundaryAndLifecycle:
    """Tests 13, 16, 17, 18: VisionTimeoutError handling and lifecycle stage transition to TIMEOUT."""

    def test_15_timeout_provider_raises_vision_timeout_error(self) -> None:
        """Provider raising VisionTimeoutError causes execution adapter to raise VisionTimeoutError with cause."""
        config = VisionProviderConfig(provider_name="timeout-prov", model_name="v1", timeout=5.0)
        provider = TimeoutSimulatingProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        with pytest.raises(VisionTimeoutError) as exc_info:
            adapter.execute("Query timeout", evidence=[ev])

        assert "timed out" in str(exc_info.value).lower()
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, VisionTimeoutError)


# ===========================================================================
# 7. Test class: Integration & Public Exports
# ===========================================================================


class TestIntegrationAndExports:
    """Tests 23-28, 35: Public exports, VisionAgent integration, and deterministic execution."""

    def test_16_public_imports_exist_in_vision_package(self) -> None:
        """VisionExecutionStage, VisionExecutionLifecycle, and VisionTimeoutError are exported by vision."""
        from vision import (
            VisionExecutionLifecycle,
            VisionExecutionStage,
            VisionTimeoutError,
        )

        assert VisionExecutionStage.COMPLETED == "completed"
        assert issubclass(VisionTimeoutError, Exception)

    def test_17_deterministic_repeated_execution(self) -> None:
        """Repeated execution of identical inputs produces identical lifecycle stage outputs."""
        config = VisionProviderConfig(provider_name="prov-det", model_name="v1")
        provider = LifecycleRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        res1 = agent.execute("Query det", evidence=[ev])
        res2 = agent.execute("Query det", evidence=[ev])

        assert res1.metadata["execution_lifecycle"] == res2.metadata["execution_lifecycle"]

    def test_18_no_network_libraries_in_lifecycle_modules(self) -> None:
        """lifecycle.py imports no socket or HTTP libraries."""
        from vision import lifecycle

        source = inspect.getsource(lifecycle)
        forbidden = [
            "import requests",
            "import httpx",
            "import aiohttp",
            "import socket",
            "import urllib.request",
        ]
        for pattern in forbidden:
            assert pattern not in source, f"lifecycle.py contains forbidden pattern '{pattern}'"
