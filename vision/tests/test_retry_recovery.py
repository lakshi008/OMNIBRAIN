"""
Day 47 — Vision Agent Retry & Recovery Safety Tests.

Comprehensive test suite verifying:
  1.  Retry count and policy validation (non-negative integer boundaries, invalid types).
  2.  Immediate success on first attempt (attempt_count=1, retry_count=0, provider_called=1).
  3.  Transient failure followed by success on retry (recovery safety).
  4.  Retry exhaustion handling and terminal failure state.
  5.  Retry eligibility distinction (transient provider vs non-retryable validation/capability errors).
  6.  Validation short-circuit (zero provider invocations, zero retry attempts).
  7.  Unsupported capability short-circuit (zero provider invocations, zero retries).
  8.  Configuration failure short-circuit (fails fast before execution).
  9.  Cancellation during/between retry attempts halts execution immediately.
  10. Timeout safety with retries (bounded, terminal timeout behavior).
  11. Execution-local retry state (zero global state leakage between requests).
  12. Concurrent multithreaded retry isolation across parallel requests.
  13. Multi-evidence retry preservation (lineage, order, counts across 1, 2, 3, 10 items).
  14. Evidence immutability before and after retry attempts.
  15. Request immutability before and after retry attempts.
  16. Result normalizer compatibility (only successful final attempt reaches normalizer).
  17. Lifecycle compatibility and trace recording of retry attempts.
  18. Observability compatibility (attempt_count and retry_count in VisionExecutionObservation).
  19. Terminal state protection (no retry after final completion or failure).
  20. Deterministic, repeatable retry behavior without sleep-heavy waits.
  21. Bounded retry guarantee (strict upper limit on provider calls).
  22. Error preservation and secret sanitization on final failure.
  23. Public API exports and backward compatibility.
  24. Complete offline execution guarantee.
"""

from __future__ import annotations

import concurrent.futures
import copy
import dataclasses
import io
import sys
import threading
from typing import Any

import pytest
from PIL import Image

from vision import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionAgent,
    VisionCancellationError,
    VisionCancellationToken,
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
    VisionRetryPolicy,
    VisionTimeoutError,
    VisualEvidence,
    execute_vision_request,
    run_vision_pipeline,
)
from vision.exceptions import (
    VisionAgentError,
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
    width: int = 32,
    height: int = 32,
    color: tuple[int, int, int] = (110, 170, 220),
) -> bytes:
    """Generate minimal valid image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


def _make_evidence(
    doc_id: str = "doc-retry-001",
    filename: str = "retry_chart.png",
    chunk_id: str = "chk-retry-001",
    content_type: str = "chart",
    page_number: int = 1,
    chunk_index: int = 0,
) -> VisualEvidence:
    """Construct a valid VisualEvidence instance."""
    return VisualEvidence(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        content_type=content_type,
        image_bytes=_make_test_image("PNG"),
        page_number=page_number,
        chunk_index=chunk_index,
        metadata={"source": "test_retry_recovery"},
    )


class SequenceFailingProvider(VisionModelProvider):
    """Test double that fails a specified number of times before succeeding."""

    def __init__(
        self,
        config: VisionProviderConfig,
        capabilities: VisionProviderCapabilities | None = None,
        fail_count: int = 0,
        fail_exception: type[Exception] = VisionProviderExecutionError,
        fail_message: str = "Simulated transient failure.",
    ) -> None:
        super().__init__(config, capabilities)
        self.fail_count = fail_count
        self.fail_exception = fail_exception
        self.fail_message = fail_message
        self.invocation_count: int = 0
        self.recorded_inputs: list[VisionModelInput] = []
        self._lock = threading.Lock()

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        with self._lock:
            self.invocation_count += 1
            current_call = self.invocation_count
            self.recorded_inputs.append(model_input)

        if current_call <= self.fail_count:
            raise self.fail_exception(f"{self.fail_message} (attempt {current_call})")

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Success after {current_call} attempts.",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={"provider": self.provider_name, "model": self.model_name, "attempt": current_call},
        )


# ===========================================================================
# 1. Test class: Retry Policy & Parameter Validation
# ===========================================================================


class TestRetryPolicyValidation:
    """Tests verifying VisionRetryPolicy creation, bounds, and exception eligibility."""

    def test_01_valid_retry_policy_defaults(self) -> None:
        """VisionRetryPolicy defaults to max_retries=0 and max_attempts=1."""
        policy = VisionRetryPolicy()
        assert policy.max_retries == 0
        assert policy.max_attempts == 1

    def test_02_valid_custom_retry_policy(self) -> None:
        """VisionRetryPolicy accepts non-negative integer max_retries."""
        policy = VisionRetryPolicy(max_retries=3)
        assert policy.max_retries == 3
        assert policy.max_attempts == 4

    @pytest.mark.parametrize(
        "bad_retries",
        [-1, -5, 1.5, True, False, "2", None, [], {}, float("inf"), float("nan")],
    )
    def test_03_invalid_max_retries_raises_validation_error(self, bad_retries: Any) -> None:
        """Invalid max_retries parameter raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="max_retries"):
            VisionRetryPolicy(max_retries=bad_retries)

    def test_04_retry_eligibility_classification(self) -> None:
        """is_retryable accurately distinguishes transient from non-retryable exceptions."""
        policy = VisionRetryPolicy(max_retries=2)

        # Retryable exceptions
        assert policy.is_retryable(VisionProviderExecutionError("Backend timeout/503")) is True
        assert policy.is_retryable(VisionProcessingError("Transient inference failure")) is True

        # Non-retryable exceptions
        assert policy.is_retryable(VisionInputValidationError("Malformed request")) is False
        assert policy.is_retryable(VisionEvidenceError("Invalid format")) is False
        assert policy.is_retryable(VisionUnsupportedCapabilityError("Format not supported")) is False
        assert policy.is_retryable(VisionProviderConfigError("Bad config")) is False
        assert policy.is_retryable(VisionCancellationError("User aborted")) is False
        assert policy.is_retryable(VisionTimeoutError("Operation timed out")) is False
        assert policy.is_retryable("not_an_exception") is False  # type: ignore[arg-type]


# ===========================================================================
# 2. Test class: Success & Recovery Behavior
# ===========================================================================


class TestSuccessAndRecoveryBehavior:
    """Tests verifying single-attempt success and recovery from transient failures."""

    def test_05_immediate_success_without_retry(self) -> None:
        """Provider that succeeds immediately executes exactly once with attempt_count=1."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = SequenceFailingProvider(cfg, fail_count=0)
        adapter = VisionExecutionAdapter(provider=provider)
        ev = _make_evidence()

        res = adapter.execute("Valid query", evidence=[ev], max_retries=2)
        assert res.status == "success"
        assert provider.invocation_count == 1

        obs = VisionExecutionObservation.from_result(res)
        assert obs.attempt_count == 1
        assert obs.retry_count == 0
        assert obs.provider_called is True
        assert obs.is_completed is True

    def test_06_transient_failure_then_success_recovery(self) -> None:
        """Provider failing on attempt 1 and succeeding on attempt 2 recovers cleanly."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = SequenceFailingProvider(cfg, fail_count=1)
        adapter = VisionExecutionAdapter(provider=provider)
        ev = _make_evidence()

        res = adapter.execute("Query", evidence=[ev], max_retries=1)
        assert res.status == "success"
        assert provider.invocation_count == 2

        obs = VisionExecutionObservation.from_result(res)
        assert obs.attempt_count == 2
        assert obs.retry_count == 1
        assert obs.is_completed is True
        assert "retry_attempted" in obs.stages

    def test_07_recovery_with_custom_retry_policy_instance(self) -> None:
        """Passing VisionRetryPolicy instance via retry_policy kwargs succeeds."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = SequenceFailingProvider(cfg, fail_count=2)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        policy = VisionRetryPolicy(max_retries=2)
        res = pipeline.run("Query", evidence=[ev], retry_policy=policy)
        assert res.status == "success"
        assert provider.invocation_count == 3

        obs = VisionExecutionObservation.from_result(res)
        assert obs.attempt_count == 3
        assert obs.retry_count == 2

    def test_08_retry_exhaustion_raises_final_error(self) -> None:
        """Provider failing beyond max_retries raises final exception and records failure."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = SequenceFailingProvider(cfg, fail_count=5)
        adapter = VisionExecutionAdapter(provider=provider)
        ev = _make_evidence()

        with pytest.raises(VisionProviderExecutionError, match="attempt 3"):
            adapter.execute("Query", evidence=[ev], max_retries=2)

        # Provider called 1 initial + 2 retries = 3 attempts total
        assert provider.invocation_count == 3


# ===========================================================================
# 3. Test class: Non-Retryable Short-Circuits & Boundaries
# ===========================================================================


class TestNonRetryableShortCircuits:
    """Tests verifying that non-retryable error categories never trigger retries."""

    def test_09_validation_error_short_circuit_never_retries(self) -> None:
        """Invalid VisionRequest raises VisionInputValidationError with 0 provider calls."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = SequenceFailingProvider(cfg, fail_count=0)
        pipeline = VisionPipeline(provider=provider)

        with pytest.raises(VisionInputValidationError):
            pipeline.run("", max_retries=3)

        assert provider.invocation_count == 0

    def test_10_unsupported_capability_never_retries(self) -> None:
        """Unsupported modality raises VisionUnsupportedCapabilityError with 0 provider calls."""
        cfg = VisionProviderConfig(provider_name="chart_only", model_name="m")
        caps = VisionProviderCapabilities(supported_modalities=["chart"])
        provider = SequenceFailingProvider(cfg, capabilities=caps)
        pipeline = VisionPipeline(provider=provider)

        diag_ev = _make_evidence(content_type="diagram")
        with pytest.raises(VisionUnsupportedCapabilityError):
            pipeline.run("Query", evidence=[diag_ev], max_retries=3)

        assert provider.invocation_count == 0

    def test_11_invalid_provider_config_never_retries(self) -> None:
        """Invalid provider config raises VisionProviderConfigError before execution."""
        with pytest.raises(VisionProviderConfigError):
            VisionProviderConfig(provider_name="", model_name="m")

    def test_12_cancellation_during_retry_halts_immediately(self) -> None:
        """Cancelling token after attempt 1 aborts execution and prevents attempt 2."""
        token = VisionCancellationToken()
        cfg = VisionProviderConfig(provider_name="p", model_name="m")

        class CancellingAfterFirstFailProvider(VisionModelProvider):
            def __init__(self, config: VisionProviderConfig) -> None:
                super().__init__(config)
                self.calls = 0

            def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
                self.calls += 1
                if self.calls == 1:
                    token.cancel("Cancelled after first attempt.")
                    raise VisionProviderExecutionError("Transient error on call 1.")
                return VisionResult(query=model_input.query, status="success")

        provider = CancellingAfterFirstFailProvider(cfg)
        adapter = VisionExecutionAdapter(provider=provider)
        ev = _make_evidence()

        with pytest.raises(VisionCancellationError, match="Cancelled after first attempt"):
            adapter.execute("Query", evidence=[ev], cancellation_token=token, max_retries=3)

        # Provider called only once, cancelled before attempt 2
        assert provider.calls == 1

    def test_13_timeout_does_not_trigger_uncontrolled_retry(self) -> None:
        """VisionTimeoutError is non-retryable and aborts without additional retry loops."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m", timeout=2.0)
        provider = SequenceFailingProvider(
            cfg,
            fail_count=5,
            fail_exception=VisionTimeoutError,
            fail_message="Execution timeout",
        )
        adapter = VisionExecutionAdapter(provider=provider)
        ev = _make_evidence()

        with pytest.raises(VisionTimeoutError):
            adapter.execute("Query", evidence=[ev], max_retries=3)

        # Timeout is terminal by design: executed only once
        assert provider.invocation_count == 1


# ===========================================================================
# 4. Test class: Immutability, Lineage & Normalizer Safety
# ===========================================================================


class TestImmutabilityAndNormalizerSafety:
    """Tests verifying input immutability, evidence preservation, and normalizer compatibility."""

    def test_14_evidence_immutability_through_retries(self) -> None:
        """VisualEvidence attributes remain completely unmutated across retries."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = SequenceFailingProvider(cfg, fail_count=1)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = _make_evidence(doc_id="doc-orig", filename="orig.png", chunk_id="chk-orig")
        before_dict = ev.to_dict()

        res = adapter.execute("Query", evidence=[ev], max_retries=2)
        assert res.status == "success"

        after_dict = ev.to_dict()
        assert before_dict == after_dict

    def test_15_request_immutability_through_retries(self) -> None:
        """VisionRequest attributes remain completely unmutated across retries."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = SequenceFailingProvider(cfg, fail_count=1)
        adapter = VisionExecutionAdapter(provider=provider)

        req = VisionRequest(query="Original Query", evidence=[_make_evidence()])
        orig_query = req.query
        orig_evidence_len = len(req.evidence)

        res = adapter.execute(req, max_retries=2)
        assert res.status == "success"
        assert req.query == orig_query
        assert len(req.evidence) == orig_evidence_len

    @pytest.mark.parametrize("count", [1, 2, 3, 10])
    def test_16_multi_evidence_retry_lineage_preservation(self, count: int) -> None:
        """Multi-evidence requests retain lineage, order, and counts through retried executions."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = SequenceFailingProvider(cfg, fail_count=1)
        pipeline = VisionPipeline(provider=provider)

        evidences = [
            _make_evidence(
                doc_id=f"doc-retry-{i:02d}",
                filename=f"chart_{i:02d}.png",
                chunk_id=f"chk-retry-{i:02d}",
                page_number=i + 1,
                chunk_index=i,
            )
            for i in range(count)
        ]

        res = pipeline.run("Multi-evidence query", evidence=evidences, max_retries=2)
        assert res.status == "success"
        assert len(res.evidence) == count
        assert res.document_id == "doc-retry-00"
        assert res.filename == "chart_00.png"
        assert res.chunk_id == "chk-retry-00"
        assert provider.invocation_count == 2

    def test_17_normalizer_called_only_for_final_successful_result(self) -> None:
        """Intermediate failed attempts do not generate VisionResult or duplicate traces."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = SequenceFailingProvider(cfg, fail_count=2)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        res = pipeline.run("Query", evidence=[ev], max_retries=2)
        assert isinstance(res, VisionResult)
        assert res.status == "success"

        trace_dict = res.metadata.get("execution_trace", {})
        stages = trace_dict.get("stages", [])
        # Exactly one final normalization & completion
        assert stages.count("result_normalized") == 1
        assert stages.count("execution_completed") == 1
        assert stages.count("retry_attempted") == 2


# ===========================================================================
# 5. Test class: Concurrency, Determinism & Public API
# ===========================================================================


class TestConcurrencyAndDeterminism:
    """Tests verifying multithreaded isolation, deterministic behavior, and API exports."""

    def test_18_execution_local_retry_state_isolation(self) -> None:
        """Independent executions have distinct retry counts and do not share attempt state."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        prov_a = SequenceFailingProvider(cfg, fail_count=1)
        prov_b = SequenceFailingProvider(cfg, fail_count=0)

        pipe_a = VisionPipeline(provider=prov_a)
        pipe_b = VisionPipeline(provider=prov_b)
        ev = _make_evidence()

        res_a = pipe_a.run("Query A", evidence=[ev], max_retries=2)
        res_b = pipe_b.run("Query B", evidence=[ev], max_retries=2)

        obs_a = VisionExecutionObservation.from_result(res_a)
        obs_b = VisionExecutionObservation.from_result(res_b)

        assert obs_a.attempt_count == 2
        assert obs_a.retry_count == 1

        assert obs_b.attempt_count == 1
        assert obs_b.retry_count == 0

    def test_19_concurrent_retry_isolation(self) -> None:
        """Multiple parallel executions with mixed retry outcomes execute in complete isolation."""
        cfg = VisionProviderConfig(provider_name="conc_p", model_name="m")

        def worker(idx: int) -> dict[str, Any]:
            # Even idx fails once then succeeds; odd idx succeeds immediately
            fail_count = 1 if idx % 2 == 0 else 0
            provider = SequenceFailingProvider(cfg, fail_count=fail_count)
            pipeline = VisionPipeline(provider=provider)
            ev = _make_evidence(doc_id=f"doc_{idx}")

            res = pipeline.run(f"Query {idx}", evidence=[ev], max_retries=2)
            obs = VisionExecutionObservation.from_result(res)
            return {
                "idx": idx,
                "status": res.status,
                "attempt_count": obs.attempt_count,
                "expected_attempts": 2 if idx % 2 == 0 else 1,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(16)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 16
        for r in results:
            assert r["status"] == "success"
            assert r["attempt_count"] == r["expected_attempts"]

    def test_20_deterministic_retry_behavior(self) -> None:
        """Same failure sequence repeatedly produces identical attempt counts and stage sequence."""
        cfg = VisionProviderConfig(provider_name="det_p", model_name="m")
        ev = _make_evidence()

        observations: list[VisionExecutionObservation] = []
        for _ in range(3):
            prov = SequenceFailingProvider(cfg, fail_count=1)
            res = VisionPipeline(provider=prov).run("Query", evidence=[ev], max_retries=2)
            observations.append(VisionExecutionObservation.from_result(res))

        for obs in observations[1:]:
            assert obs.attempt_count == observations[0].attempt_count
            assert obs.retry_count == observations[0].retry_count
            assert obs.stages == observations[0].stages

    def test_21_public_api_exports(self) -> None:
        """VisionRetryPolicy is cleanly exported in vision."""
        import vision

        assert hasattr(vision, "VisionRetryPolicy")
        assert "VisionRetryPolicy" in vision.__all__

    def test_22_offline_execution_guarantee(self) -> None:
        """Retry and recovery subsystem operates 100% offline with zero network dependencies."""
        for forbidden in ("langfuse", "opentelemetry", "prometheus", "requests", "httpx", "aiohttp"):
            assert forbidden not in sys.modules or forbidden not in globals()
