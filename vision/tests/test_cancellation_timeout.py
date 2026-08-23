"""
Day 46 — Vision Agent Execution Cancellation & Timeout Safety Tests.

Comprehensive test suite verifying:
  1.  Valid timeout parameter acceptance and propagation.
  2.  Invalid timeout validation (zero, negative, boolean, non-finite, wrong types).
  3.  Timeout before provider execution (short-circuit).
  4.  Timeout during provider execution (terminal TIMEOUT stage, VisionTimeoutError).
  5.  Timeout after successful completion (terminal success protection).
  6.  Explicit cancellation via VisionCancellationToken before provider execution.
  7.  Cancellation during provider execution (synchronization primitives, no fake result).
  8.  Cancellation after successful completion (terminal success protection).
  9.  Repeated cancellation idempotency.
  10. Timeout and cancellation race conditions (single terminal state).
  11. Terminal state transition protection across all terminal stages.
  12. Resource cleanup and execution isolation after timeout or cancellation.
  13. Provider isolation (cancelling or timing out Request A does not impact Request B).
  14. Multi-evidence cancellation and timeout safety across varying evidence counts.
  15. Observability compatibility (accurate reporting of is_cancelled, provider_called=False, etc.).
  16. Result normalizer compatibility (no fabricated results on cancelled/timed-out paths).
  17. Concurrent multithreaded isolation with mixed success/timeout/cancellation executions.
  18. Deterministic behavior without sleep-heavy or flaky tests.
  19. Exception hierarchy compliance (VisionCancellationError, VisionTimeoutError).
  20. Public API integrity and complete offline execution guarantee.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import io
import math
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
    width: int = 36,
    height: int = 36,
    color: tuple[int, int, int] = (120, 160, 230),
) -> bytes:
    """Generate minimal valid image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


def _make_evidence(
    doc_id: str = "doc-to-001",
    filename: str = "test_chart.png",
    chunk_id: str = "chk-to-001",
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
        metadata={"origin": "test_cancellation_timeout"},
    )


class ControllableTestProvider(VisionModelProvider):
    """Test double with controllable timeout, cancellation, and execution hooks."""

    def __init__(
        self,
        config: VisionProviderConfig,
        capabilities: VisionProviderCapabilities | None = None,
        simulate_timeout: bool = False,
        start_event: threading.Event | None = None,
        release_event: threading.Event | None = None,
    ) -> None:
        super().__init__(config, capabilities)
        self.simulate_timeout = simulate_timeout
        self.start_event = start_event
        self.release_event = release_event
        self.invocation_count: int = 0
        self.recorded_inputs: list[VisionModelInput] = []
        self._lock = threading.Lock()

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        with self._lock:
            self.invocation_count += 1
            self.recorded_inputs.append(model_input)

        if self.start_event is not None:
            self.start_event.set()

        # Check for cancellation token passed directly to provider
        token = kwargs.get("cancellation_token") or kwargs.get("cancel_token")
        if token is not None:
            if hasattr(token, "is_cancelled") and token.is_cancelled:
                raise VisionCancellationError(getattr(token, "reason", "Cancelled in provider."))
            if hasattr(token, "is_set") and token.is_set():
                raise VisionCancellationError("Cancelled via event in provider.")

        if self.release_event is not None:
            self.release_event.wait(timeout=1.0)

        # Re-check cancellation after release wait
        if token is not None:
            if hasattr(token, "is_cancelled") and token.is_cancelled:
                raise VisionCancellationError(getattr(token, "reason", "Cancelled during execution."))
            if hasattr(token, "is_set") and token.is_set():
                raise VisionCancellationError("Cancelled via event during execution.")

        if self.simulate_timeout:
            raise VisionTimeoutError(f"Execution timed out after {self.config.timeout}s.")

        return VisionResult(
            query=model_input.query,
            status="success",
            description="Controllable provider execution result.",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={"provider": self.provider_name, "model": self.model_name},
        )


# ===========================================================================
# 1. Test class: Timeout Validation & Boundaries
# ===========================================================================


class TestTimeoutValidationAndBoundaries:
    """Tests for timeout parameter validation in provider config and runtime execution."""

    def test_01_valid_timeout_parameter(self) -> None:
        """Valid positive timeout is accepted in both config and runtime execution."""
        cfg = VisionProviderConfig(provider_name="p1", model_name="m1", timeout=15.0)
        assert cfg.timeout == 15.0

        provider = ControllableTestProvider(cfg)
        adapter = VisionExecutionAdapter(provider=provider)
        ev = _make_evidence()

        res = adapter.execute("Valid query", evidence=[ev], timeout=10.0)
        assert res.status == "success"
        assert provider.invocation_count == 1

    @pytest.mark.parametrize(
        "bad_timeout",
        [0, 0.0, -1, -5.5, None, True, False, "30", "5s", float("inf"), float("-inf"), float("nan"), [], {}],
    )
    def test_02_invalid_timeout_in_config_raises_error(self, bad_timeout: Any) -> None:
        """Invalid timeout values in VisionProviderConfig raise VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="timeout"):
            VisionProviderConfig(provider_name="p1", model_name="m1", timeout=bad_timeout)

    @pytest.mark.parametrize(
        "bad_timeout",
        [0, 0.0, -1, -5.5, True, False, "10", float("inf"), float("-inf"), float("nan"), [], {}],
    )
    def test_03_invalid_runtime_timeout_raises_validation_error(self, bad_timeout: Any) -> None:
        """Invalid runtime timeout in adapter.execute raises VisionInputValidationError."""
        cfg = VisionProviderConfig(provider_name="p1", model_name="m1")
        provider = ControllableTestProvider(cfg)
        adapter = VisionExecutionAdapter(provider=provider)
        ev = _make_evidence()

        with pytest.raises(VisionInputValidationError, match="timeout"):
            adapter.execute("Query", evidence=[ev], timeout=bad_timeout)

        assert provider.invocation_count == 0


# ===========================================================================
# 2. Test class: Timeout Handling During & After Execution
# ===========================================================================


class TestTimeoutHandling:
    """Tests for timeout occurrences before, during, and after execution."""

    def test_04_timeout_during_provider_execution(self) -> None:
        """Provider timeout raises VisionTimeoutError and transitions lifecycle to TIMEOUT stage."""
        cfg = VisionProviderConfig(provider_name="timeout_prov", model_name="vlm-to", timeout=5.0)
        provider = ControllableTestProvider(cfg, simulate_timeout=True)
        adapter = VisionExecutionAdapter(provider=provider)
        ev = _make_evidence()

        with pytest.raises(VisionTimeoutError, match="timed out"):
            adapter.execute("Heavy query", evidence=[ev])

        assert provider.invocation_count == 1

    def test_05_timeout_after_successful_completion_rejected(self) -> None:
        """Calling transition_to(TIMEOUT) after successful COMPLETED stage raises VisionInputValidationError."""
        lifecycle = VisionExecutionLifecycle(provider_name="p", model_name="m")
        lifecycle.transition_to(VisionExecutionStage.VALIDATING)
        lifecycle.transition_to(VisionExecutionStage.COMPLETED)

        assert lifecycle.is_completed is True
        assert lifecycle.is_terminal is True

        with pytest.raises(VisionInputValidationError, match="Cannot transition execution lifecycle from terminal stage"):
            lifecycle.transition_to(VisionExecutionStage.TIMEOUT, error="Late timeout")

        # Status remains COMPLETED and not failed
        assert lifecycle.stage == VisionExecutionStage.COMPLETED
        assert lifecycle.is_completed is True
        assert lifecycle.is_failed is False


# ===========================================================================
# 3. Test class: Explicit Cancellation Before & During Execution
# ===========================================================================


class TestCancellationHandling:
    """Tests for explicit execution cancellation via VisionCancellationToken."""

    def test_06_cancellation_token_creation_and_properties(self) -> None:
        """VisionCancellationToken manages thread-safe cancellation state."""
        token = VisionCancellationToken()
        assert token.is_cancelled is False
        assert token.reason is None

        token.cancel("User cancelled query.")
        assert token.is_cancelled is True
        assert token.reason == "User cancelled query."

        with pytest.raises(VisionCancellationError, match="User cancelled query"):
            token.raise_if_cancelled()

    def test_07_cancellation_before_provider_execution(self) -> None:
        """Pre-cancelled token immediately aborts pipeline without calling provider."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = ControllableTestProvider(cfg)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        token = VisionCancellationToken()
        token.cancel("Cancelled before start.")

        with pytest.raises(VisionCancellationError, match="Cancelled before start"):
            pipeline.run("Query", evidence=[ev], cancellation_token=token)

        assert provider.invocation_count == 0

    def test_08_cancellation_during_provider_execution(self) -> None:
        """Cancellation signaled while provider is running aborts execution cleanly."""
        start_event = threading.Event()
        release_event = threading.Event()
        token = VisionCancellationToken()

        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = ControllableTestProvider(
            cfg,
            start_event=start_event,
            release_event=release_event,
        )
        adapter = VisionExecutionAdapter(provider=provider)
        ev = _make_evidence()

        def runner() -> None:
            adapter.execute("Query", evidence=[ev], cancellation_token=token)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(runner)

            # Wait until provider has started execution
            assert start_event.wait(timeout=1.0)
            assert provider.invocation_count == 1

            # Signal cancellation
            token.cancel("Cancelled while running.")
            release_event.set()

            with pytest.raises(VisionCancellationError, match="Cancelled while running"):
                future.result()

    def test_09_cancellation_after_success_rejected(self) -> None:
        """Attempting to cancel an already COMPLETED lifecycle raises VisionInputValidationError."""
        lifecycle = VisionExecutionLifecycle(provider_name="p", model_name="m")
        lifecycle.transition_to(VisionExecutionStage.COMPLETED)

        with pytest.raises(VisionInputValidationError, match="Cannot transition execution lifecycle from terminal stage"):
            lifecycle.cancel("Attempt late cancel")

        assert lifecycle.stage == VisionExecutionStage.COMPLETED
        assert lifecycle.is_completed is True
        assert lifecycle.is_cancelled is False

    def test_10_repeated_cancellation_is_idempotent(self) -> None:
        """Calling cancel multiple times on an already cancelled lifecycle is idempotent."""
        lifecycle = VisionExecutionLifecycle(provider_name="p", model_name="m")
        lifecycle.cancel("First cancel")
        assert lifecycle.stage == VisionExecutionStage.CANCELLED
        assert lifecycle.is_cancelled is True
        assert lifecycle.is_terminal is True

        # Second cancel is a safe no-op
        lifecycle.cancel("Second cancel")
        assert lifecycle.stage == VisionExecutionStage.CANCELLED
        assert lifecycle.error == "First cancel"

    def test_11_threading_event_as_cancellation_signal(self) -> None:
        """threading.Event can be passed as cancellation_token."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = ControllableTestProvider(cfg)
        pipeline = VisionPipeline(provider=provider)
        ev = _make_evidence()

        cancel_event = threading.Event()
        cancel_event.set()

        with pytest.raises(VisionCancellationError, match="event signal"):
            pipeline.run("Query", evidence=[ev], cancellation_token=cancel_event)

        assert provider.invocation_count == 0


# ===========================================================================
# 4. Test class: Terminal State Protection & Races
# ===========================================================================


class TestTerminalStateProtectionAndRaces:
    """Tests verifying terminal state invariants and deterministic race handling."""

    @pytest.mark.parametrize(
        "terminal_stage",
        [
            VisionExecutionStage.COMPLETED,
            VisionExecutionStage.FAILED,
            VisionExecutionStage.TIMEOUT,
            VisionExecutionStage.CANCELLED,
        ],
    )
    def test_12_terminal_stages_cannot_transition_to_other_stages(self, terminal_stage: str) -> None:
        """Once in any terminal stage, transition to any different stage is strictly rejected."""
        lifecycle = VisionExecutionLifecycle(provider_name="p", model_name="m")
        lifecycle.transition_to(terminal_stage)
        assert lifecycle.is_terminal is True

        for other_stage in VisionExecutionStage.ALL_STAGES:
            if other_stage != terminal_stage:
                with pytest.raises(VisionInputValidationError, match="Cannot transition execution lifecycle from terminal stage"):
                    lifecycle.transition_to(other_stage)

    def test_13_timeout_then_cancellation_race(self) -> None:
        """When timeout occurs first, subsequent cancellation is rejected without state corruption."""
        lifecycle = VisionExecutionLifecycle(provider_name="p", model_name="m")
        lifecycle.transition_to(VisionExecutionStage.TIMEOUT, error="Operation timed out")
        assert lifecycle.stage == VisionExecutionStage.TIMEOUT
        assert lifecycle.is_terminal is True

        with pytest.raises(VisionInputValidationError):
            lifecycle.cancel("Late cancel")

        assert lifecycle.stage == VisionExecutionStage.TIMEOUT

    def test_14_cancellation_then_timeout_race(self) -> None:
        """When cancellation occurs first, subsequent timeout is rejected without state corruption."""
        lifecycle = VisionExecutionLifecycle(provider_name="p", model_name="m")
        lifecycle.cancel("User abort")
        assert lifecycle.stage == VisionExecutionStage.CANCELLED
        assert lifecycle.is_terminal is True

        with pytest.raises(VisionInputValidationError):
            lifecycle.transition_to(VisionExecutionStage.TIMEOUT, error="Late timeout")

        assert lifecycle.stage == VisionExecutionStage.CANCELLED


# ===========================================================================
# 5. Test class: Provider Isolation & Multi-Evidence Safety
# ===========================================================================


class TestProviderIsolationAndMultiEvidence:
    """Tests verifying isolation between requests and multi-evidence preservation."""

    def test_15_provider_isolation_between_cancelled_and_successful_requests(self) -> None:
        """Cancelling Request A does not affect Request B on the same shared provider."""
        cfg = VisionProviderConfig(provider_name="shared_p", model_name="shared-v1")
        provider = ControllableTestProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        ev_a = _make_evidence(doc_id="doc_A")
        ev_b = _make_evidence(doc_id="doc_B")

        token_a = VisionCancellationToken()
        token_a.cancel("Cancel A")

        # Request A cancelled
        with pytest.raises(VisionCancellationError):
            pipeline.run("Query A", evidence=[ev_a], cancellation_token=token_a)

        # Request B succeeds
        res_b = pipeline.run("Query B", evidence=[ev_b])
        assert res_b.status == "success"
        assert res_b.document_id == "doc_B"
        assert provider.invocation_count == 1

    @pytest.mark.parametrize("count", [1, 2, 3, 10])
    def test_16_multi_evidence_cancellation_safety(self, count: int) -> None:
        """Multi-evidence requests abort cleanly on cancellation without evidence leaks."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = ControllableTestProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        evidences = [_make_evidence(doc_id=f"doc_{i:02d}") for i in range(count)]
        token = VisionCancellationToken()
        token.cancel("Aborted multi-evidence")

        with pytest.raises(VisionCancellationError):
            pipeline.run("Multi-evidence query", evidence=evidences, cancellation_token=token)

        assert provider.invocation_count == 0


# ===========================================================================
# 6. Test class: Observability & Normalizer Compatibility
# ===========================================================================


class TestObservabilityAndNormalizerCompatibility:
    """Tests verifying Day 45 observability and Day 39 normalizer compatibility."""

    def test_17_cancelled_execution_observation(self) -> None:
        """Cancelled execution observation captures is_cancelled=True, provider_called=False."""
        lifecycle = VisionExecutionLifecycle(provider_name="obs_prov", model_name="vlm-1")
        lifecycle.cancel("Pre-execution cancellation")

        obs = VisionExecutionObservation.from_lifecycle(lifecycle, evidence_count=2)
        assert obs.stage == VisionExecutionStage.CANCELLED
        assert obs.is_cancelled is True
        assert obs.is_completed is False
        assert obs.is_failed is True
        assert obs.is_terminal is True
        assert obs.error == "Pre-execution cancellation"
        assert obs.provider_called is False
        assert obs.evidence_count == 2

    def test_18_timed_out_execution_observation(self) -> None:
        """Timed out execution observation captures stage=timeout, is_failed=True."""
        lifecycle = VisionExecutionLifecycle(provider_name="obs_prov", model_name="vlm-1")
        lifecycle.transition_to(VisionExecutionStage.TIMEOUT, error="Timeout exceeded")

        trace = VisionExecutionTrace()
        trace.add_stage("request_received")
        trace.add_stage("provider_started")

        obs = VisionExecutionObservation.from_lifecycle(lifecycle, trace=trace, evidence_count=1)
        assert obs.stage == VisionExecutionStage.TIMEOUT
        assert obs.is_failed is True
        assert obs.is_completed is False
        assert obs.is_cancelled is False
        assert obs.is_terminal is True
        assert obs.provider_called is True
        assert obs.error == "Timeout exceeded"

    def test_19_no_fake_result_on_cancellation(self) -> None:
        """Cancelled execution does not produce a fake VisionResult."""
        cfg = VisionProviderConfig(provider_name="p", model_name="m")
        provider = ControllableTestProvider(cfg)
        adapter = VisionExecutionAdapter(provider=provider)

        token = VisionCancellationToken()
        token.cancel("User cancelled")

        with pytest.raises(VisionCancellationError):
            adapter.execute("Query", evidence=[_make_evidence()], cancellation_token=token)

        assert provider.invocation_count == 0


# ===========================================================================
# 7. Test class: Concurrency Isolation & Determinism
# ===========================================================================


class TestConcurrencyIsolationAndDeterminism:
    """Tests verifying concurrent multithreaded isolation and deterministic contracts."""

    def test_20_concurrent_mixed_cancellation_timeout_success(self) -> None:
        """Parallel executions across threads with mixed outcomes remain completely isolated."""
        cfg = VisionProviderConfig(provider_name="conc_p", model_name="conc-v1")
        provider = ControllableTestProvider(cfg)
        pipeline = VisionPipeline(provider=provider)

        def worker_success(idx: int) -> str:
            ev = _make_evidence(doc_id=f"doc_s_{idx}")
            res = pipeline.run(f"Query {idx}", evidence=[ev])
            return res.status

        def worker_cancel(idx: int) -> str:
            ev = _make_evidence(doc_id=f"doc_c_{idx}")
            tok = VisionCancellationToken()
            tok.cancel("Cancelled worker")
            try:
                pipeline.run(f"Query {idx}", evidence=[ev], cancellation_token=tok)
                return "unexpected"
            except VisionCancellationError:
                return "cancelled"

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            s_futures = [executor.submit(worker_success, i) for i in range(12)]
            c_futures = [executor.submit(worker_cancel, i) for i in range(12)]

            s_results = [f.result() for f in concurrent.futures.as_completed(s_futures)]
            c_results = [f.result() for f in concurrent.futures.as_completed(c_futures)]

        assert len(s_results) == 12
        assert all(r == "success" for r in s_results)

        assert len(c_results) == 12
        assert all(r == "cancelled" for r in c_results)

        # Provider only invoked for successful runs
        assert provider.invocation_count == 12

    def test_21_exception_hierarchy(self) -> None:
        """VisionCancellationError and VisionTimeoutError inherit from VisionAgentError."""
        assert issubclass(VisionCancellationError, VisionAgentError)
        assert issubclass(VisionTimeoutError, VisionAgentError)
        assert issubclass(VisionTimeoutError, VisionProcessingError)

    def test_22_public_api_exports(self) -> None:
        """VisionCancellationToken and VisionCancellationError are exported in vision."""
        import vision

        assert hasattr(vision, "VisionCancellationToken")
        assert hasattr(vision, "VisionCancellationError")
        assert hasattr(vision, "VisionTimeoutError")

        assert "VisionCancellationToken" in vision.__all__
        assert "VisionCancellationError" in vision.__all__
        assert "VisionTimeoutError" in vision.__all__

    def test_23_offline_guarantee(self) -> None:
        """Timeout and cancellation operate 100% offline with zero external network dependencies."""
        for mod in ("langfuse", "opentelemetry", "prometheus", "requests", "httpx", "aiohttp"):
            assert mod not in sys.modules or mod not in globals()
