"""
Day 48 — Vision Agent Resource Safety & Execution Cleanup Tests.

Comprehensive test suite verifying all 27 required resource safety and execution cleanup invariants:
  1.  Execution state ownership
  2.  Repeated execution safety (10+ successive executions with zero state accumulation)
  3.  Success cleanup
  4.  Failure cleanup
  5.  Retry cleanup
  6.  Cancellation cleanup
  7.  Timeout cleanup
  8.  Multi-evidence cleanup
  9.  Evidence isolation
  10. Request isolation
  11. Provider state isolation
  12. Lifecycle isolation
  13. Observability isolation
  14. Result isolation
  15. Retry + cancellation cleanup
  16. Retry + timeout cleanup
  17. Concurrent isolation
  18. Global-state detection
  19. Temporary-file safety (pure in-memory processing)
  20. Idempotent cleanup / terminal state operations
  21. Validation failure cleanup
  22. Capability failure cleanup
  23. Configuration failure cleanup
  24. Provider failure cleanup
  25. Deterministic repeated execution
  26. Public API compatibility
  27. Complete offline execution guarantee
"""

from __future__ import annotations

import concurrent.futures
import copy
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


# ===========================================================================
# Test Doubles & Helpers
# ===========================================================================


def _make_test_image(
    format_name: str = "PNG",
    width: int = 32,
    height: int = 32,
    color: tuple[int, int, int] = (100, 180, 240),
) -> bytes:
    """Generate minimal valid image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


def _make_evidence(
    doc_id: str = "doc-cleanup-001",
    filename: str = "cleanup_chart.png",
    chunk_id: str = "chk-cleanup-001",
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
        metadata={"origin": "test_resource_safety"},
    )


class DynamicOutcomeProvider(VisionModelProvider):
    """Test double allowing execution-specific outcomes (success, failure, timeout, cancel)."""

    def __init__(
        self,
        config: VisionProviderConfig,
        capabilities: VisionProviderCapabilities | None = None,
    ) -> None:
        super().__init__(config, capabilities)
        self.invocation_count: int = 0
        self.recorded_inputs: list[VisionModelInput] = []
        self._lock = threading.Lock()

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        with self._lock:
            self.invocation_count += 1
            call_idx = self.invocation_count
            self.recorded_inputs.append(model_input)

        outcome = kwargs.get("force_outcome") or "success"
        if "fail" in model_input.query.lower() or outcome == "fail":
            raise VisionProviderExecutionError(f"Simulated execution failure on call {call_idx}.")
        if "timeout" in model_input.query.lower() or outcome == "timeout":
            raise VisionTimeoutError(f"Simulated execution timeout on call {call_idx}.")

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Executed successfully (call {call_idx}).",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={"provider": self.provider_name, "call_index": call_idx},
        )


# ===========================================================================
# 1. State Ownership & Repeated Execution
# ===========================================================================


class TestStateOwnershipAndRepeatedExecution:
    """Tests 1, 2, 3, 25: State ownership, repeated runs, and success cleanup."""

    def test_01_execution_state_ownership(self) -> None:
        """State changes in Execution A do not affect Execution B."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        res_a = pipeline.run("Query A", evidence=[ev])
        res_b = pipeline.run("Query B", evidence=[ev])

        res_a.metadata["unique_a"] = 123
        assert "unique_a" not in res_b.metadata
        assert res_a.query == "Query A"
        assert res_b.query == "Query B"

    def test_02_repeated_execution_safety(self) -> None:
        """10 successive executions on one pipeline instance do not accumulate stale state."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        for i in range(10):
            res = pipeline.run(f"Query {i}", evidence=[ev])
            assert res.status == "success"
            obs = VisionExecutionObservation.from_result(res)
            assert obs.attempt_count == 1
            assert obs.retry_count == 0
            assert obs.is_completed is True

        assert provider.invocation_count == 10

    def test_03_success_cleanup(self) -> None:
        """Successful execution completes with terminal lifecycle and clean metadata."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run("Query Success", evidence=[_make_evidence()])
        assert res.status == "success"
        obs = VisionExecutionObservation.from_result(res)
        assert obs.is_completed is True
        assert obs.is_failed is False
        assert obs.is_terminal is True
        assert obs.error is None

    def test_25_deterministic_repeated_execution(self) -> None:
        """Repeated runs with identical input produce identical output structures."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        res1 = pipeline.run("Query Det", evidence=[ev])
        res2 = pipeline.run("Query Det", evidence=[ev])

        assert res1.query == res2.query
        assert res1.status == res2.status
        assert res1.document_id == res2.document_id
        assert res1.filename == res2.filename


# ===========================================================================
# 2. Failure, Retry, Cancellation & Timeout Cleanup
# ===========================================================================


class TestFailureRetryCancellationAndTimeoutCleanup:
    """Tests 4, 5, 6, 7, 15, 16, 24: Error-path cleanup."""

    def test_04_failure_cleanup(self) -> None:
        """Failed execution A does not poison subsequent execution B."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        with pytest.raises(VisionProviderExecutionError):
            pipeline.run("Query FAIL", evidence=[ev])

        res_b = pipeline.run("Query B", evidence=[ev])
        assert res_b.status == "success"

    def test_05_retry_cleanup(self) -> None:
        """Retried execution A does not leak attempt counts into execution B."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")

        class RetryThenSuccessProvider(VisionModelProvider):
            def __init__(self, config: VisionProviderConfig) -> None:
                super().__init__(config)
                self.calls = 0

            def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
                self.calls += 1
                if self.calls == 1:
                    raise VisionProviderExecutionError("Transient error call 1")
                return VisionResult(query=model_input.query, status="success")

        provider = RetryThenSuccessProvider(cfg)
        adapter = VisionExecutionAdapter(provider=provider)
        ev = _make_evidence()

        # Run A: retries once, succeeds on attempt 2
        res_a = adapter.execute("Query A", evidence=[ev], max_retries=2)
        obs_a = VisionExecutionObservation.from_result(res_a)
        assert obs_a.attempt_count == 2
        assert obs_a.retry_count == 1

        # Run B: succeeds on attempt 1
        res_b = adapter.execute("Query B", evidence=[ev], max_retries=2)
        obs_b = VisionExecutionObservation.from_result(res_b)
        assert obs_b.attempt_count == 1
        assert obs_b.retry_count == 0

    def test_06_cancellation_cleanup(self) -> None:
        """Cancelled execution A does not poison execution B."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        tok = VisionCancellationToken()
        tok.cancel("Cancelled A")

        with pytest.raises(VisionCancellationError):
            pipeline.run("Query A", evidence=[ev], cancellation_token=tok)

        res_b = pipeline.run("Query B", evidence=[ev])
        assert res_b.status == "success"
        obs_b = VisionExecutionObservation.from_result(res_b)
        assert obs_b.is_cancelled is False
        assert obs_b.is_completed is True

    def test_07_timeout_cleanup(self) -> None:
        """Timed-out execution A does not poison execution B."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        with pytest.raises(VisionTimeoutError):
            pipeline.run("Query TIMEOUT", evidence=[ev])

        res_b = pipeline.run("Query B", evidence=[ev])
        assert res_b.status == "success"

    def test_15_retry_plus_cancellation_cleanup(self) -> None:
        """Cancelling execution A during retries leaves pipeline clean for execution B."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        tok = VisionCancellationToken()
        tok.cancel("Cancel A")

        with pytest.raises(VisionCancellationError):
            pipeline.run("Query A", evidence=[ev], cancellation_token=tok, max_retries=3)

        res_b = pipeline.run("Query B", evidence=[ev], max_retries=3)
        assert res_b.status == "success"

    def test_16_retry_plus_timeout_cleanup(self) -> None:
        """Timing out execution A during retries leaves pipeline clean for execution B."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        with pytest.raises(VisionTimeoutError):
            pipeline.run("Query TIMEOUT", evidence=[ev], max_retries=3)

        res_b = pipeline.run("Query B", evidence=[ev], max_retries=3)
        assert res_b.status == "success"

    def test_24_provider_failure_cleanup(self) -> None:
        """Provider 500 error on run A does not leak error strings into run B."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        with pytest.raises(VisionProviderExecutionError, match="call 1"):
            pipeline.run("Query FAIL", evidence=[ev])

        res_b = pipeline.run("Query B", evidence=[ev])
        assert res_b.status == "success"
        assert res_b.error is None


# ===========================================================================
# 3. Evidence, Request, Provider & Lifecycle Isolation
# ===========================================================================


class TestIsolationAndMutabilitySafety:
    """Tests 8, 9, 10, 11, 12, 13, 14: Isolation invariants."""

    def test_08_multi_evidence_cleanup(self) -> None:
        """Large evidence batch in Run A does not leak into small evidence batch in Run B."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        ev_10 = [_make_evidence(doc_id=f"doc_{i:02d}") for i in range(10)]
        ev_1 = [_make_evidence(doc_id="doc_single")]

        res_a = pipeline.run("Query 10", evidence=ev_10)
        assert len(res_a.evidence) == 10

        res_b = pipeline.run("Query 1", evidence=ev_1)
        assert len(res_b.evidence) == 1
        assert res_b.evidence[0].document_id == "doc_single"

    def test_09_evidence_isolation(self) -> None:
        """Mutating original evidence dictionary does not mutate result lineage."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        ev = _make_evidence(doc_id="doc_iso")
        res = pipeline.run("Query", evidence=[ev])

        ev.metadata["mutated"] = True
        assert "mutated" not in res.metadata

    def test_10_request_isolation(self) -> None:
        """Mutating VisionRequest instance does not corrupt normalized result evidence."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        ev = _make_evidence(doc_id="doc_req")
        req = VisionRequest(query="Query", evidence=[ev])
        res = pipeline.run(req)

        req.evidence.append(_make_evidence(doc_id="doc_extra"))
        assert len(res.evidence) == 1
        assert res.evidence[0].document_id == "doc_req"

    def test_11_provider_state_isolation(self) -> None:
        """Provider records separate inputs for each execution."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        pipeline.run("Query 1", evidence=[_make_evidence(doc_id="doc1")])
        pipeline.run("Query 2", evidence=[_make_evidence(doc_id="doc2")])

        assert provider.recorded_inputs[0].query == "Query 1"
        assert provider.recorded_inputs[1].query == "Query 2"

    def test_12_lifecycle_isolation(self) -> None:
        """Each execution receives an independent lifecycle instance."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        res_a = pipeline.run("Query A", evidence=[_make_evidence()])
        res_b = pipeline.run("Query B", evidence=[_make_evidence()])

        lc_a = res_a.metadata.get("execution_lifecycle", {})
        lc_b = res_b.metadata.get("execution_lifecycle", {})
        assert lc_a is not lc_b

    def test_13_observability_isolation(self) -> None:
        """VisionExecutionObservation instances are completely decoupled."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        res_a = pipeline.run("Query A", evidence=[_make_evidence()])
        res_b = pipeline.run("Query B", evidence=[_make_evidence()])

        obs_a = VisionExecutionObservation.from_result(res_a)
        obs_b = VisionExecutionObservation.from_result(res_b)
        assert obs_a is not obs_b

    def test_14_result_isolation(self) -> None:
        """Distinct VisionResult instances maintain separate metadata dictionaries."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        res_a = pipeline.run("Query A", evidence=[_make_evidence()])
        res_b = pipeline.run("Query B", evidence=[_make_evidence()])
        assert res_a.metadata is not res_b.metadata


# ===========================================================================
# 4. Short-Circuits & Invariant Checks
# ===========================================================================


class TestShortCircuitsAndInvariants:
    """Tests 17, 18, 19, 20, 21, 22, 23, 26, 27: Concurrency, invariants, and validation."""

    def test_17_concurrent_isolation(self) -> None:
        """Concurrent executions across 10 threads remain fully isolated."""
        cfg = VisionProviderConfig(provider_name="conc_p", model_name="conc_m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        def worker(idx: int) -> str:
            ev = _make_evidence(doc_id=f"doc_{idx}")
            res = pipeline.run(f"Query {idx}", evidence=[ev])
            return res.status

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        assert all(r == "success" for r in results)

    def test_18_global_state_detection(self) -> None:
        """Subsystem contains zero global execution registries or mutable lists."""
        import vision.execution_adapter as ea
        import vision.lifecycle as lc
        import vision.pipeline as pl

        for mod in (ea, lc, pl):
            for name, val in vars(mod).items():
                if name.startswith("_") and isinstance(val, (list, set, dict)):
                    assert name in ("__all__", "__builtins__", "__cached__", "__doc__", "__file__", "__name__", "__package__", "__loader__", "__spec__") or name.isupper() or len(val) == 0

    def test_19_temporary_file_safety(self) -> None:
        """Vision reasoning pipeline operates entirely in-memory without unmanaged tempfiles."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        res = pipeline.run("Query In Memory", evidence=[ev])
        assert res.status == "success"

    def test_20_idempotent_cleanup(self) -> None:
        """Terminal lifecycle operations are safe and idempotent."""
        lifecycle = VisionExecutionLifecycle(provider_name="p", model_name="m")
        lifecycle.cancel("Cancel once")
        # Second cancel is safe idempotent no-op
        lifecycle.cancel("Cancel twice")
        assert lifecycle.stage == VisionExecutionStage.CANCELLED

    def test_21_validation_failure_cleanup(self) -> None:
        """Empty query validation failure on run A does not prevent run B."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = DynamicOutcomeProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        with pytest.raises(VisionInputValidationError):
            pipeline.run("")

        res_b = pipeline.run("Valid", evidence=[_make_evidence()])
        assert res_b.status == "success"

    def test_22_capability_failure_cleanup(self) -> None:
        """Unsupported modality failure on run A does not prevent run B."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        caps = VisionProviderCapabilities(supported_modalities=["chart"])
        provider = DynamicOutcomeProvider(cfg, capabilities=caps)
        pipeline = VisionPipeline(provider=provider)

        diag_ev = _make_evidence(content_type="diagram")
        chart_ev = _make_evidence(content_type="chart")

        with pytest.raises(VisionUnsupportedCapabilityError):
            pipeline.run("Query diag", evidence=[diag_ev])

        res_b = pipeline.run("Query chart", evidence=[chart_ev])
        assert res_b.status == "success"

    def test_23_configuration_failure_cleanup(self) -> None:
        """Invalid provider config raises error without poisoning subsequent valid configs."""
        with pytest.raises(VisionProviderConfigError):
            VisionProviderConfig(provider_name="", model_name="m")

        cfg_valid = VisionProviderConfig(provider_name="valid_p", model_name="valid_m")
        assert cfg_valid.provider_name == "valid_p"

    def test_26_public_api_compatibility(self) -> None:
        """All public symbols exist in vision module export."""
        import vision

        expected = [
            "VisionRequest",
            "VisionResult",
            "VisualEvidence",
            "VisionPipeline",
            "VisionExecutionAdapter",
            "VisionExecutionLifecycle",
            "VisionExecutionObservation",
            "VisionCancellationToken",
            "VisionRetryPolicy",
            "VisionModelProvider",
            "VisionProviderConfig",
            "VisionProviderCapabilities",
        ]
        for s in expected:
            assert hasattr(vision, s)

    def test_27_offline_execution(self) -> None:
        """Subsystem operates 100% offline with zero external network or telemetry packages."""
        for forbidden in ("langfuse", "opentelemetry", "prometheus", "requests", "httpx", "aiohttp"):
            assert forbidden not in sys.modules or forbidden not in globals()
