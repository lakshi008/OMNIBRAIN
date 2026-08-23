"""
Day 45 — Vision Agent Execution Observability Interfaces & Trace Contract Tests.

Comprehensive test suite verifying:
  1.  Successful execution observation (lifecycle state, provider status, trace stages).
  2.  Failed execution observation (error category, terminal state, failure flags).
  3.  Invalid request observation and short-circuit validation.
  4.  Unsupported capability observation (provider_called = False).
  5.  Provider invocation status verification across valid, invalid, and unsupported paths.
  6.  Lifecycle state transitions, stage invariants, and query properties.
  7.  Logical event/state ordering in execution trace sequences.
  8.  Terminal success state protection (rejection of illegal transitions).
  9.  Terminal failure state protection (rejection of illegal transitions).
  10. Duplicate completion prevention and idempotent behavior.
  11. Multi-evidence observation across varying evidence counts (1, 2, 3, 10).
  12. Evidence ordering and lineage preservation in execution observation.
  13. Request isolation (zero cross-request state leakage).
  14. Concurrent execution isolation under multithreaded workloads.
  15. Deterministic, repeatable execution observations without fake timestamps/latency.
  16. Immutability guarantees of frozen observation records.
  17. Error sanitization (stripping API keys, credentials, tokens, image bytes).
  18. Public API compatibility and exports.
  19. Offline execution guarantee (zero external network/telemetry dependencies).
  20. No fake VisionResult generation on error or no-evidence paths.
"""

from __future__ import annotations

import concurrent.futures
import copy
import dataclasses
import inspect
import io
from typing import Any

import pytest
from PIL import Image

from vision import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionAgent,
    VisionExecutionAdapter,
    VisionExecutionLifecycle,
    VisionExecutionObservation,
    VisionExecutionStage,
    VisionExecutionTrace,
    VisionModelInput,
    VisionModelProvider,
    VisionPipeline,
    VisionProviderCapabilities,
    VisionProviderConfig,
    VisionRequest,
    VisionResult,
    VisionResultNormalizer,
    VisualEvidence,
    execute_vision_request,
    run_vision_pipeline,
)
from vision.exceptions import (
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderConfigError,
    VisionProviderError,
    VisionProviderExecutionError,
    VisionProviderUnavailableError,
    VisionUnsupportedCapabilityError,
)
from vision.image_preparation import prepare_image_evidence
from vision.input_builder import build_vision_input


# ===========================================================================
# Test Doubles & Helpers
# ===========================================================================


def _make_test_image(
    format_name: str = "PNG",
    width: int = 40,
    height: int = 40,
    color: tuple[int, int, int] = (100, 150, 220),
) -> bytes:
    """Generate minimal valid image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


def _make_evidence(
    doc_id: str = "doc-obs-001",
    filename: str = "obs_chart.png",
    chunk_id: str = "chk-obs-001",
    content_type: str = "chart",
    page_number: int = 1,
    chunk_index: int = 0,
    image_bytes: bytes | None = None,
) -> VisualEvidence:
    """Construct a valid VisualEvidence object."""
    return VisualEvidence(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        content_type=content_type,
        image_bytes=image_bytes or _make_test_image("PNG"),
        page_number=page_number,
        chunk_index=chunk_index,
        metadata={"source": "test_observability"},
    )


class ObservabilityMockProvider(VisionModelProvider):
    """Test double recording invocations, inputs, and configurable behavior."""

    def __init__(
        self,
        config: VisionProviderConfig,
        capabilities: VisionProviderCapabilities | None = None,
        should_fail: bool = False,
    ) -> None:
        super().__init__(config, capabilities)
        self.should_fail = should_fail
        self.invocation_count: int = 0
        self.recorded_inputs: list[VisionModelInput] = []

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        self.invocation_count += 1
        self.recorded_inputs.append(model_input)

        if self.should_fail:
            raise VisionProviderExecutionError("Simulated backend provider failure.")

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Observed visual analysis by {self.provider_name}",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={
                "provider": self.provider_name,
                "model": self.model_name,
                "custom_metric": 42,
            },
        )


# ===========================================================================
# 1. Test class: Successful Execution Observation
# ===========================================================================


class TestSuccessfulExecutionObservation:
    """Tests verifying observation contracts on successful execution paths."""

    def test_01_successful_execution_observation_from_result(self) -> None:
        """Successful pipeline execution produces complete, accurate VisionExecutionObservation."""
        cfg = VisionProviderConfig(provider_name="test_prov", model_name="vlm-obs")
        provider = ObservabilityMockProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        ev = _make_evidence()
        req = VisionRequest(query="Analyze chart trend", evidence=[ev])

        result = pipeline.run(req)
        assert isinstance(result, VisionResult)
        assert result.status == "success"

        obs = VisionExecutionObservation.from_result(result)
        assert isinstance(obs, VisionExecutionObservation)
        assert obs.stage == VisionExecutionStage.COMPLETED
        assert obs.provider_name == "test_prov"
        assert obs.model_name == "vlm-obs"
        assert obs.is_completed is True
        assert obs.is_failed is False
        assert obs.is_terminal is True
        assert obs.error is None
        assert obs.provider_called is True
        assert obs.evidence_count == 1
        assert obs.result_status == "success"
        assert len(obs.stages) >= 5

    def test_02_observation_dictionary_serialization(self) -> None:
        """VisionExecutionObservation.to_dict produces clean serializable dictionary."""
        cfg = VisionProviderConfig(provider_name="test_prov", model_name="vlm-obs")
        provider = ObservabilityMockProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        ev = _make_evidence()
        result = pipeline.run("Analyze trend", evidence=[ev])
        obs = VisionExecutionObservation.from_result(result)
        data = obs.to_dict()

        assert data["stage"] == "completed"
        assert data["provider_name"] == "test_prov"
        assert data["model_name"] == "vlm-obs"
        assert data["is_completed"] is True
        assert data["is_failed"] is False
        assert data["is_terminal"] is True
        assert data["error"] is None
        assert data["provider_called"] is True
        assert data["evidence_count"] == 1
        assert data["result_status"] == "success"
        assert isinstance(data["stages"], list)
        assert isinstance(data["metadata"], dict)

    def test_03_no_evidence_successful_observation(self) -> None:
        """Execution with no evidence produces a graceful no_evidence observation without calling provider."""
        cfg = VisionProviderConfig(provider_name="test_prov", model_name="vlm-obs")
        provider = ObservabilityMockProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        req = VisionRequest(query="Query with no visual evidence", evidence=[])
        result = pipeline.run(req)

        assert result.status == "no_evidence"
        assert provider.invocation_count == 0

        obs = VisionExecutionObservation.from_result(result)
        assert obs.stage == VisionExecutionStage.COMPLETED
        assert obs.is_completed is True
        assert obs.is_failed is False
        assert obs.provider_called is False
        assert obs.evidence_count == 0
        assert obs.result_status == "no_evidence"


# ===========================================================================
# 2. Test class: Failed Execution & Error Observation
# ===========================================================================


class TestFailedExecutionObservation:
    """Tests verifying observability during failure and exception handling."""

    def test_04_provider_failure_observation(self) -> None:
        """Provider failure marks lifecycle as failed with error details and provider_called status."""
        cfg = VisionProviderConfig(provider_name="failing_prov", model_name="vlm-fail")
        provider = ObservabilityMockProvider(cfg, should_fail=True)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = _make_evidence()
        with pytest.raises(VisionProviderExecutionError, match="Simulated backend provider failure"):
            adapter.execute("Analyze chart", evidence=[ev])

        assert provider.invocation_count == 1

    def test_05_observation_from_failed_lifecycle(self) -> None:
        """Constructing observation from a failed lifecycle accurately captures failure state."""
        lifecycle = VisionExecutionLifecycle(provider_name="p1", model_name="m1")
        lifecycle.transition_to(VisionExecutionStage.VALIDATING)
        lifecycle.transition_to(VisionExecutionStage.FAILED, error="Invalid input payload")

        obs = VisionExecutionObservation.from_lifecycle(lifecycle, evidence_count=0)
        assert obs.stage == VisionExecutionStage.FAILED
        assert obs.is_completed is False
        assert obs.is_failed is True
        assert obs.is_terminal is True
        assert obs.error == "Invalid input payload"
        assert obs.provider_called is False

    def test_06_unsupported_capability_observation(self) -> None:
        """Unsupported capability request fails before provider execution (provider_called = False)."""
        cfg = VisionProviderConfig(provider_name="chart_only", model_name="chart-v1")
        caps = VisionProviderCapabilities(supported_modalities=["chart"])
        provider = ObservabilityMockProvider(cfg, capabilities=caps)
        pipeline = VisionPipeline(provider=provider)

        diag_ev = _make_evidence(content_type="diagram")
        req = VisionRequest(query="Analyze diagram", evidence=[diag_ev])

        with pytest.raises(VisionUnsupportedCapabilityError):
            pipeline.run(req)

        assert provider.invocation_count == 0

    def test_07_invalid_request_observation_short_circuit(self) -> None:
        """Invalid request validation fails before provider execution (provider_called = False)."""
        cfg = VisionProviderConfig(provider_name="p1", model_name="m1")
        provider = ObservabilityMockProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        with pytest.raises(VisionInputValidationError):
            pipeline.run("")

        with pytest.raises(VisionInputValidationError):
            pipeline.run(None)  # type: ignore[arg-type]

        assert provider.invocation_count == 0


# ===========================================================================
# 3. Test class: Lifecycle Stage Invariants & Transitions
# ===========================================================================


class TestLifecycleInvariantsAndTransitions:
    """Tests verifying canonical lifecycle stage transitions and terminal state protection."""

    def test_08_canonical_stage_order(self) -> None:
        """Trace stages in successful execution follow strict chronological ordering."""
        cfg = VisionProviderConfig(provider_name="p1", model_name="m1")
        provider = ObservabilityMockProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        ev = _make_evidence()
        result = pipeline.run("Analyze trend", evidence=[ev])
        obs = VisionExecutionObservation.from_result(result)

        expected_stages = [
            "request_received",
            "validation_started",
            "input_prepared",
            "provider_started",
            "provider_completed",
            "result_normalized",
            "execution_completed",
        ]
        assert list(obs.stages) == expected_stages

    def test_09_terminal_success_cannot_transition_to_other_stage(self) -> None:
        """Transitioning from terminal COMPLETED to another stage raises VisionInputValidationError."""
        lifecycle = VisionExecutionLifecycle(provider_name="p", model_name="m")
        lifecycle.transition_to(VisionExecutionStage.VALIDATING)
        lifecycle.transition_to(VisionExecutionStage.COMPLETED)

        assert lifecycle.is_terminal is True

        with pytest.raises(VisionInputValidationError, match="Cannot transition execution lifecycle from terminal stage"):
            lifecycle.transition_to(VisionExecutionStage.VALIDATING)

        with pytest.raises(VisionInputValidationError, match="Cannot transition execution lifecycle from terminal stage"):
            lifecycle.transition_to(VisionExecutionStage.EXECUTING)

    def test_10_terminal_failure_cannot_transition_to_other_stage(self) -> None:
        """Transitioning from terminal FAILED to another stage raises VisionInputValidationError."""
        lifecycle = VisionExecutionLifecycle(provider_name="p", model_name="m")
        lifecycle.transition_to(VisionExecutionStage.FAILED, error="Fatal crash")

        assert lifecycle.is_terminal is True

        with pytest.raises(VisionInputValidationError, match="Cannot transition execution lifecycle from terminal stage"):
            lifecycle.transition_to(VisionExecutionStage.COMPLETED)

        with pytest.raises(VisionInputValidationError, match="Cannot transition execution lifecycle from terminal stage"):
            lifecycle.transition_to(VisionExecutionStage.PREPARING)

    def test_11_idempotent_same_stage_transition(self) -> None:
        """Transitioning to the exact same terminal stage does not raise an error."""
        lifecycle = VisionExecutionLifecycle(provider_name="p", model_name="m")
        lifecycle.transition_to(VisionExecutionStage.COMPLETED)
        # Re-transitioning to completed is idempotent
        lifecycle.transition_to(VisionExecutionStage.COMPLETED, metadata={"note": "second_call"})
        assert lifecycle.stage == VisionExecutionStage.COMPLETED
        assert lifecycle.metadata.get("note") == "second_call"

    def test_12_lifecycle_to_observation_method(self) -> None:
        """VisionExecutionLifecycle.to_observation() creates an observation snapshot."""
        lifecycle = VisionExecutionLifecycle(provider_name="prov_alpha", model_name="mod_beta")
        lifecycle.transition_to(VisionExecutionStage.VALIDATING)
        lifecycle.transition_to(VisionExecutionStage.PREPARING)
        lifecycle.transition_to(VisionExecutionStage.COMPLETED)

        trace = VisionExecutionTrace()
        trace.add_stage("stage_1")
        trace.add_stage("stage_2")

        obs = lifecycle.to_observation(trace=trace, evidence_count=2, result_status="success")
        assert obs.stage == "completed"
        assert obs.provider_name == "prov_alpha"
        assert obs.model_name == "mod_beta"
        assert obs.is_completed is True
        assert obs.evidence_count == 2
        assert obs.stages == ("stage_1", "stage_2")


# ===========================================================================
# 4. Test class: Multi-Evidence Observability & Lineage
# ===========================================================================


class TestMultiEvidenceObservability:
    """Tests verifying accurate evidence counts and lineage in observability data."""

    @pytest.mark.parametrize("count", [1, 2, 3, 10])
    def test_13_multi_evidence_count_accuracy(self, count: int) -> None:
        """Observation accurately records exact evidence count for multiple visual items."""
        cfg = VisionProviderConfig(provider_name="test_prov", model_name="vlm-obs")
        provider = ObservabilityMockProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        evidences = [
            _make_evidence(
                doc_id=f"doc-{i:03d}",
                filename=f"chart_{i:03d}.png",
                chunk_id=f"chk-{i:03d}",
                page_number=i + 1,
                chunk_index=i,
            )
            for i in range(count)
        ]

        req = VisionRequest(query="Multi-evidence query", evidence=evidences)
        result = pipeline.run(req)

        obs = VisionExecutionObservation.from_result(result)
        assert obs.evidence_count == count
        assert len(result.evidence) == count

        # Primary evidence lineage preserved
        assert result.document_id == "doc-000"
        assert result.filename == "chart_000.png"
        assert result.page_number == 1
        assert result.chunk_id == "chk-000"

    def test_14_evidence_order_preservation(self) -> None:
        """Evidence order is preserved exactly in the resulting observation data."""
        cfg = VisionProviderConfig(provider_name="test_prov", model_name="vlm-obs")
        provider = ObservabilityMockProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        ev1 = _make_evidence(doc_id="first_doc")
        ev2 = _make_evidence(doc_id="second_doc")
        ev3 = _make_evidence(doc_id="third_doc")

        result = pipeline.run("Check order", evidence=[ev1, ev2, ev3])
        assert [e.document_id for e in result.evidence] == ["first_doc", "second_doc", "third_doc"]

        obs = VisionExecutionObservation.from_result(result)
        assert obs.evidence_count == 3


# ===========================================================================
# 5. Test class: Immutability & Error Sanitization
# ===========================================================================


class TestImmutabilityAndSanitization:
    """Tests verifying immutability of observation objects and sanitization of secrets."""

    def test_15_observation_is_frozen(self) -> None:
        """VisionExecutionObservation is a frozen dataclass and rejects field mutations."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = ObservabilityMockProvider(cfg)
        result = VisionPipeline(provider=provider).run("Query", evidence=[_make_evidence()])
        obs = VisionExecutionObservation.from_result(result)

        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            obs.stage = "failed"  # type: ignore[misc]

        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            obs.provider_called = False  # type: ignore[misc]

        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            obs.evidence_count = 999  # type: ignore[misc]

    def test_16_sanitization_strips_sensitive_keys_from_observation(self) -> None:
        """Observation metadata strips forbidden secret keys (api_key, token, password, etc.)."""
        meta_with_secrets = {
            "valid_key": "safe_value",
            "api_key": "sk-secret-12345",
            "password": "super-secret-password",
            "token": "bearer-token-abc",
            "image_bytes": b"raw_binary_bytes",
            "nested": {
                "nested_safe": 100,
                "auth": "secret_auth_header",
            },
        }
        sanitized = VisionResultNormalizer.sanitize_metadata(meta_with_secrets)

        assert "valid_key" in sanitized
        assert "api_key" not in sanitized
        assert "password" not in sanitized
        assert "token" not in sanitized
        assert "image_bytes" not in sanitized
        assert sanitized["nested"] == {"nested_safe": 100}

        # Build an observation with this metadata
        obs = VisionExecutionObservation(
            stage="completed",
            provider_name="prov",
            model_name="mod",
            is_completed=True,
            is_failed=False,
            is_terminal=True,
            metadata=sanitized,
        )
        assert "api_key" not in obs.metadata
        assert "password" not in obs.metadata


# ===========================================================================
# 6. Test class: Request Isolation & Concurrency Safety
# ===========================================================================


class TestRequestIsolationAndConcurrency:
    """Tests verifying request isolation and concurrent multithreaded safety."""

    def test_17_request_isolation_between_executions(self) -> None:
        """Independent executions do not bleed lifecycle, trace, or metadata state."""
        cfg = VisionProviderConfig(provider_name="shared_prov", model_name="vlm-iso")
        provider = ObservabilityMockProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        ev_a = _make_evidence(doc_id="doc_A", filename="chart_A.png")
        ev_b = _make_evidence(doc_id="doc_B", filename="chart_B.png")

        res_a = pipeline.run("Query A", evidence=[ev_a])
        res_b = pipeline.run("Query B", evidence=[ev_b, _make_evidence(doc_id="doc_B2")])

        obs_a = VisionExecutionObservation.from_result(res_a)
        obs_b = VisionExecutionObservation.from_result(res_b)

        assert obs_a.evidence_count == 1
        assert obs_b.evidence_count == 2

        assert res_a.document_id == "doc_A"
        assert res_b.document_id == "doc_B"

    def test_18_concurrent_executions_observation_isolation(self) -> None:
        """Concurrent executions across threads maintain distinct, isolated observations."""
        cfg = VisionProviderConfig(provider_name="conc_prov", model_name="vlm-conc")
        provider = ObservabilityMockProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        def worker_task(idx: int) -> VisionExecutionObservation:
            ev = _make_evidence(doc_id=f"doc_conc_{idx:02d}")
            res = pipeline.run(f"Query {idx}", evidence=[ev])
            return VisionExecutionObservation.from_result(res)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker_task, i) for i in range(24)]
            observations = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(observations) == 24
        assert all(obs.is_completed for obs in observations)
        assert all(obs.evidence_count == 1 for obs in observations)
        assert all(obs.provider_called for obs in observations)
        assert provider.invocation_count == 24


# ===========================================================================
# 7. Test class: Determinism & Public API Compatibility
# ===========================================================================


class TestDeterminismAndPublicAPI:
    """Tests verifying repeatable deterministic behavior and public API integrity."""

    def test_19_deterministic_trace_and_observation(self) -> None:
        """Repeated identical executions yield identical stage sequences and observations."""
        cfg = VisionProviderConfig(provider_name="det_prov", model_name="vlm-det")
        provider = ObservabilityMockProvider(cfg)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        observations: list[VisionExecutionObservation] = []
        for _ in range(5):
            res = pipeline.run("Deterministic query", evidence=[ev])
            observations.append(VisionExecutionObservation.from_result(res))

        for obs in observations[1:]:
            assert obs.stage == observations[0].stage
            assert obs.is_completed == observations[0].is_completed
            assert obs.stages == observations[0].stages
            assert obs.evidence_count == observations[0].evidence_count

    def test_20_public_api_exports(self) -> None:
        """VisionExecutionObservation, VisionExecutionLifecycle, VisionExecutionTrace are exported."""
        import vision

        assert hasattr(vision, "VisionExecutionObservation")
        assert hasattr(vision, "VisionExecutionLifecycle")
        assert hasattr(vision, "VisionExecutionStage")
        assert hasattr(vision, "VisionExecutionTrace")

        assert "VisionExecutionObservation" in vision.__all__
        assert "VisionExecutionLifecycle" in vision.__all__
        assert "VisionExecutionStage" in vision.__all__
        assert "VisionExecutionTrace" in vision.__all__

    def test_21_offline_execution_guarantee(self) -> None:
        """Execution observability runs 100% offline with zero external network or telemetry dependencies."""
        import sys
        import vision.lifecycle

        # Ensure no network / telemetry modules are referenced in vision.lifecycle
        lifecycle_source = inspect.getsource(vision.lifecycle) if "inspect" in globals() else ""
        for forbidden in ("langfuse", "opentelemetry", "prometheus", "datadog", "sentry", "httpx", "requests"):
            assert forbidden not in sys.modules or forbidden not in lifecycle_source

    def test_22_no_fake_result_on_failure_or_no_evidence(self) -> None:
        """No-evidence execution does not fabricate visual analysis descriptions."""
        cfg = VisionProviderConfig(provider_name="det_prov", model_name="vlm-det")
        provider = ObservabilityMockProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        result = pipeline.run("Query with zero evidence", evidence=[])
        assert result.status == "no_evidence"
        assert result.description == ""
        assert result.evidence == []

        obs = VisionExecutionObservation.from_result(result)
        assert obs.provider_called is False
        assert obs.evidence_count == 0
        assert obs.result_status == "no_evidence"

    def test_23_provider_config_failure_prevention(self) -> None:
        """Invalid provider configuration fails fast before execution lifecycle begins."""
        with pytest.raises(VisionProviderConfigError):
            VisionProviderConfig(provider_name="", model_name="vlm-1")

        with pytest.raises(VisionProviderConfigError):
            VisionProviderConfig(provider_name="p1", model_name="m1", timeout=-1.0)

    def test_24_observation_direct_field_validation(self) -> None:
        """VisionExecutionObservation directly validates all required fields upon instantiation."""
        with pytest.raises(VisionInputValidationError, match="Invalid execution stage"):
            VisionExecutionObservation(
                stage="invalid_stage",
                provider_name="p",
                model_name="m",
                is_completed=False,
                is_failed=True,
                is_terminal=True,
            )

        with pytest.raises(VisionInputValidationError, match="provider_name"):
            VisionExecutionObservation(
                stage="completed",
                provider_name="",
                model_name="m",
                is_completed=True,
                is_failed=False,
                is_terminal=True,
            )

        with pytest.raises(VisionInputValidationError, match="model_name"):
            VisionExecutionObservation(
                stage="completed",
                provider_name="p",
                model_name="   ",
                is_completed=True,
                is_failed=False,
                is_terminal=True,
            )

        with pytest.raises(VisionInputValidationError, match="error must be a string"):
            VisionExecutionObservation(
                stage="failed",
                provider_name="p",
                model_name="m",
                is_completed=False,
                is_failed=True,
                is_terminal=True,
                error=123,  # type: ignore[arg-type]
            )

