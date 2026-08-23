"""
Vision Execution Lifecycle and State Tracking for OmniBrain Member 3.

Defines deterministic execution state machines, execution stages, and state tracking
contracts for the Vision Model Provider execution pipeline.

Day 38 Scope:
  - VisionExecutionStage: Canonical execution stage constants.
  - VisionExecutionLifecycle: Deterministic, vendor-agnostic execution state tracking container.
  - Stage transitions: pending -> validating -> preparing -> building_input -> executing -> completed/failed/timeout.
  - Pure execution tracking -- zero secrets, zero fake timing, zero network calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import threading
from typing import Any

from vision.exceptions import VisionInputValidationError


class VisionExecutionStage:
    """Canonical stage identifiers for the Vision Provider Execution Lifecycle."""

    PENDING: str = "pending"
    VALIDATING: str = "validating"
    PREPARING: str = "preparing"
    BUILDING_INPUT: str = "building_input"
    EXECUTING: str = "executing"
    COMPLETED: str = "completed"
    FAILED: str = "failed"
    TIMEOUT: str = "timeout"
    CANCELLED: str = "cancelled"

    ALL_STAGES: frozenset[str] = frozenset({
        "pending",
        "validating",
        "preparing",
        "building_input",
        "executing",
        "completed",
        "failed",
        "timeout",
        "cancelled",
    })

    TERMINAL_STAGES: frozenset[str] = frozenset({
        "completed",
        "failed",
        "timeout",
        "cancelled",
    })


# ---------------------------------------------------------------------------
# VisionCancellationToken (Day 46 Cancellation Contract)
# ---------------------------------------------------------------------------


class VisionCancellationToken:
    """Lightweight, thread-safe cancellation token for Vision execution.

    Allows callers to explicitly request cancellation of an ongoing or pending
    vision reasoning pipeline execution without global mutable state.
    """

    def __init__(self) -> None:
        self._is_cancelled: bool = False
        self._reason: str | None = None
        self._lock = threading.Lock()

    @property
    def is_cancelled(self) -> bool:
        """Return True if cancellation has been requested."""
        with self._lock:
            return self._is_cancelled

    @property
    def reason(self) -> str | None:
        """Return the cancellation reason string if cancelled."""
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "Vision execution was cancelled by caller.") -> None:
        """Signal cancellation for this token.

        Args:
            reason: Optional description of why cancellation was requested.
        """
        with self._lock:
            self._is_cancelled = True
            self._reason = reason

    def raise_if_cancelled(self) -> None:
        """Raise VisionCancellationError if cancellation was requested.

        Raises:
            VisionCancellationError: If is_cancelled is True.
        """
        if self.is_cancelled:
            from vision.exceptions import VisionCancellationError

            raise VisionCancellationError(self.reason or "Vision execution was cancelled.")


# ---------------------------------------------------------------------------
# VisionRetryPolicy (Day 47 Retry & Recovery Contract)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisionRetryPolicy:
    """Immutable configuration policy for controlled execution retry and recovery.

    Defines maximum attempt thresholds and determines whether specific
    exceptions are eligible for bounded, execution-local retries.

    Attributes:
        max_retries: Maximum number of retry attempts beyond the initial attempt (>= 0).
    """

    max_retries: int = 0

    def __post_init__(self) -> None:
        """Validate retry policy parameters."""
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            raise VisionInputValidationError(
                f"max_retries must be a non-negative integer (>= 0), got {self.max_retries!r}."
            )

    @property
    def max_attempts(self) -> int:
        """Total allowable attempts (1 initial attempt + max_retries)."""
        return 1 + self.max_retries

    def is_retryable(self, exception: Exception) -> bool:
        """Determine whether a given exception is eligible for retry.

        Non-retryable exceptions:
          - VisionInputValidationError (invalid query/evidence/parameters)
          - VisionEvidenceError (corrupted/invalid evidence format)
          - VisionUnsupportedCapabilityError (unsupported modality/format)
          - VisionProviderConfigError (misconfigured provider)
          - VisionCancellationError (explicit cancellation)
          - VisionTimeoutError (timeout is terminal by design)

        Retryable exceptions:
          - VisionProviderExecutionError (transient backend failure)
          - VisionProcessingError (generic transient processing issue)

        Args:
            exception: Exception instance raised during execution.

        Returns:
            True if exception is eligible for retry, False otherwise.
        """
        if not isinstance(exception, Exception):
            return False

        from vision.exceptions import (
            VisionCancellationError,
            VisionEvidenceError,
            VisionInputValidationError,
            VisionProcessingError,
            VisionProviderConfigError,
            VisionProviderExecutionError,
            VisionTimeoutError,
            VisionUnsupportedCapabilityError,
        )

        # Strictly non-retryable categories
        if isinstance(
            exception,
            (
                VisionInputValidationError,
                VisionEvidenceError,
                VisionUnsupportedCapabilityError,
                VisionProviderConfigError,
                VisionCancellationError,
                VisionTimeoutError,
            ),
        ):
            return False

        # Retryable categories
        if isinstance(exception, (VisionProviderExecutionError, VisionProcessingError)):
            return True

        return False


@dataclass
class VisionExecutionLifecycle:
    """Deterministic, vendor-agnostic container for tracking provider execution lifecycle states.

    Maintains current execution stage, provider identity, error status, attempt counts,
    and stage-specific metadata without storing secrets, credentials, or fake latency data.

    Attributes:
        stage: Current execution stage string (must be in VisionExecutionStage.ALL_STAGES).
        provider_name: Identifier of the target vision provider.
        model_name: Designated vision model name.
        error: Error message string if execution reached a failed, timeout, or cancelled state.
        attempt_count: Number of provider execution attempts made.
        retry_count: Number of retry attempts made beyond initial execution.
        metadata: Stage transition metadata dictionary.
    """

    stage: str = VisionExecutionStage.PENDING
    provider_name: str = "unknown"
    model_name: str = "unknown"
    error: str | None = None
    attempt_count: int = 1
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate lifecycle fields."""
        if not isinstance(self.stage, str) or self.stage.strip().lower() not in VisionExecutionStage.ALL_STAGES:
            raise VisionInputValidationError(
                f"Invalid execution stage '{self.stage}'. "
                f"Must be one of {sorted(VisionExecutionStage.ALL_STAGES)}."
            )
        self.stage = self.stage.strip().lower()

        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise VisionInputValidationError("provider_name must be a non-empty string.")
        self.provider_name = self.provider_name.strip()

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise VisionInputValidationError("model_name must be a non-empty string.")
        self.model_name = self.model_name.strip()

        if self.error is not None and not isinstance(self.error, str):
            raise VisionInputValidationError("error must be a string or None.")

        if not isinstance(self.metadata, (dict, Mapping)):
            raise VisionInputValidationError("metadata must be a dictionary.")
        self.metadata = dict(self.metadata)

    @property
    def is_completed(self) -> bool:
        """Whether execution completed successfully."""
        return self.stage == VisionExecutionStage.COMPLETED and self.error is None

    @property
    def is_cancelled(self) -> bool:
        """Whether execution was cancelled."""
        return self.stage == VisionExecutionStage.CANCELLED

    @property
    def is_failed(self) -> bool:
        """Whether execution reached a failed, timeout, or cancelled stage."""
        return (
            self.stage in (VisionExecutionStage.FAILED, VisionExecutionStage.TIMEOUT, VisionExecutionStage.CANCELLED)
            or self.error is not None
        )

    @property
    def is_terminal(self) -> bool:
        """Whether execution reached a terminal state."""
        return self.stage in VisionExecutionStage.TERMINAL_STAGES

    def cancel(self, reason: str = "Vision execution was cancelled by caller.") -> None:
        """Cancel execution lifecycle.

        If execution is already in a terminal state (COMPLETED, FAILED, TIMEOUT),
        raises VisionInputValidationError to prevent altering terminal state.
        If already CANCELLED, this is an idempotent no-op.
        """
        if self.is_terminal:
            if self.stage == VisionExecutionStage.CANCELLED:
                return
            raise VisionInputValidationError(
                f"Cannot transition execution lifecycle from terminal stage '{self.stage}' to 'cancelled'."
            )
        self.transition_to(VisionExecutionStage.CANCELLED, error=reason)

    def transition_to(
        self,
        next_stage: str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Transition execution state to next_stage.

        Args:
            next_stage: Target stage identifier from VisionExecutionStage.ALL_STAGES.
            error: Optional error description string if transitioning to failed/timeout.
            metadata: Optional additional metadata to merge into lifecycle metadata.

        Raises:
            VisionInputValidationError: If next_stage is invalid or transition from terminal state occurs.
        """
        if not isinstance(next_stage, str) or next_stage.strip().lower() not in VisionExecutionStage.ALL_STAGES:
            raise VisionInputValidationError(
                f"Invalid target stage '{next_stage}'. Must be one of {sorted(VisionExecutionStage.ALL_STAGES)}."
            )

        clean_stage = next_stage.strip().lower()

        if self.is_terminal and clean_stage != self.stage:
            raise VisionInputValidationError(
                f"Cannot transition execution lifecycle from terminal stage '{self.stage}' to '{clean_stage}'."
            )

        self.stage = clean_stage
        if error is not None:
            if not isinstance(error, str):
                raise VisionInputValidationError("error must be a string or None.")
            self.error = error

        if metadata:
            if not isinstance(metadata, (dict, Mapping)):
                raise VisionInputValidationError("metadata must be a dictionary.")
            self.metadata.update(metadata)

    def to_observation(
        self,
        trace: Any = None,
        evidence_count: int = 0,
        result_status: str | None = None,
    ) -> VisionExecutionObservation:
        """Create an immutable VisionExecutionObservation from this lifecycle instance."""
        return VisionExecutionObservation.from_lifecycle(
            lifecycle=self,
            trace=trace,
            evidence_count=evidence_count,
            result_status=result_status,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return serializable dictionary representation of execution lifecycle state."""
        return {
            "stage": self.stage,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "error": self.error,
            "attempt_count": self.attempt_count,
            "retry_count": self.retry_count,
            "is_completed": self.is_completed,
            "is_cancelled": self.is_cancelled,
            "is_failed": self.is_failed,
            "is_terminal": self.is_terminal,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# VisionExecutionObservation (Day 45 Observability Contract)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisionExecutionObservation:
    """Immutable, structured execution observation container for the Vision subsystem.

    Provides lightweight, offline observability over the execution lifecycle,
    trace stages, provider invocation status, attempt counts, and result metadata without
    external telemetry, network dependencies, or production overhead.

    Attributes:
        stage: Canonical lifecycle stage identifier.
        provider_name: Target provider identifier.
        model_name: Target model identifier.
        is_completed: Whether execution reached successful completion.
        is_failed: Whether execution reached a failed or timeout state.
        is_terminal: Whether execution reached a terminal state.
        is_cancelled: Whether execution was cancelled.
        error: Optional error description string.
        attempt_count: Total provider execution attempts made.
        retry_count: Number of retried attempts made beyond initial execution.
        provider_called: Whether the provider backend was invoked during execution.
        evidence_count: Number of visual evidence items processed.
        result_status: Status string from VisionResult if available.
        stages: Tuple of ordered trace stage identifiers.
        metadata: Execution metadata dictionary.
    """

    stage: str
    provider_name: str
    model_name: str
    is_completed: bool
    is_failed: bool
    is_terminal: bool
    is_cancelled: bool = False
    error: str | None = None
    attempt_count: int = 1
    retry_count: int = 0
    provider_called: bool = False
    evidence_count: int = 0
    result_status: str | None = None
    stages: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate observation fields."""
        if not isinstance(self.stage, str) or self.stage.strip().lower() not in VisionExecutionStage.ALL_STAGES:
            raise VisionInputValidationError(
                f"Invalid execution stage '{self.stage}'. "
                f"Must be one of {sorted(VisionExecutionStage.ALL_STAGES)}."
            )
        object.__setattr__(self, "stage", self.stage.strip().lower())

        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise VisionInputValidationError("provider_name must be a non-empty string.")
        object.__setattr__(self, "provider_name", self.provider_name.strip())

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise VisionInputValidationError("model_name must be a non-empty string.")
        object.__setattr__(self, "model_name", self.model_name.strip())

        if self.error is not None and not isinstance(self.error, str):
            raise VisionInputValidationError("error must be a string or None.")

        if (
            isinstance(self.attempt_count, bool)
            or not isinstance(self.attempt_count, int)
            or self.attempt_count < 1
        ):
            raise VisionInputValidationError("attempt_count must be a positive integer (>= 1).")

        if (
            isinstance(self.retry_count, bool)
            or not isinstance(self.retry_count, int)
            or self.retry_count < 0
        ):
            raise VisionInputValidationError("retry_count must be a non-negative integer (>= 0).")

        if isinstance(self.stages, (list, set, frozenset)):
            object.__setattr__(self, "stages", tuple(str(s) for s in self.stages))
        elif not isinstance(self.stages, tuple):
            raise VisionInputValidationError("stages must be a tuple or list of strings.")

        if not isinstance(self.metadata, (dict, Mapping)):
            raise VisionInputValidationError("metadata must be a dictionary.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Convert observation to serializable dictionary representation."""
        return {
            "stage": self.stage,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "is_completed": self.is_completed,
            "is_cancelled": self.is_cancelled,
            "is_failed": self.is_failed,
            "is_terminal": self.is_terminal,
            "error": self.error,
            "attempt_count": self.attempt_count,
            "retry_count": self.retry_count,
            "provider_called": self.provider_called,
            "evidence_count": self.evidence_count,
            "result_status": self.result_status,
            "stages": list(self.stages),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_result(cls, result: Any) -> VisionExecutionObservation:
        """Construct an immutable VisionExecutionObservation from a VisionResult or dict.

        Args:
            result: VisionResult instance or result dictionary.

        Returns:
            Immutable VisionExecutionObservation snapshot.

        Raises:
            VisionInputValidationError: If result is None or invalid type.
        """
        if result is None:
            raise VisionInputValidationError("result cannot be None.")

        meta: dict[str, Any]
        status: str | None
        evidence_count: int
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            meta = result.metadata
            status = getattr(result, "status", None)
            ev = getattr(result, "evidence", [])
            evidence_count = len(ev) if isinstance(ev, (list, tuple)) else 0
        elif isinstance(result, dict):
            meta = result.get("metadata", {})
            status = result.get("status")
            ev = result.get("evidence", [])
            evidence_count = len(ev) if isinstance(ev, (list, tuple)) else 0
        else:
            raise VisionInputValidationError(
                f"Expected VisionResult or dict, got {type(result).__name__}."
            )

        lifecycle_data = meta.get("execution_lifecycle", {})
        if not isinstance(lifecycle_data, dict):
            lifecycle_data = {}

        trace_data = meta.get("execution_trace", {})
        stages_list: list[str] = []
        if isinstance(trace_data, dict):
            raw_stages = trace_data.get("stages", [])
            if isinstance(raw_stages, list):
                stages_list = [str(s) for s in raw_stages]

        provider_called = "provider_completed" in stages_list or "provider_started" in stages_list

        stage = str(lifecycle_data.get("stage", VisionExecutionStage.COMPLETED))
        provider_name = str(lifecycle_data.get("provider_name", "unknown"))
        model_name = str(lifecycle_data.get("model_name", "unknown"))
        error = lifecycle_data.get("error")
        if error is not None:
            error = str(error)

        attempt_count = int(lifecycle_data.get("attempt_count", 1))
        retry_count = int(lifecycle_data.get("retry_count", 0))

        is_completed = bool(
            lifecycle_data.get(
                "is_completed",
                stage == VisionExecutionStage.COMPLETED and error is None and status != "error",
            )
        )
        is_cancelled = bool(
            lifecycle_data.get("is_cancelled", stage == VisionExecutionStage.CANCELLED)
        )
        is_failed = bool(
            lifecycle_data.get(
                "is_failed",
                stage in (VisionExecutionStage.FAILED, VisionExecutionStage.TIMEOUT, VisionExecutionStage.CANCELLED)
                or error is not None
                or status == "error",
            )
        )
        is_terminal = bool(
            lifecycle_data.get("is_terminal", stage in VisionExecutionStage.TERMINAL_STAGES)
        )

        return cls(
            stage=stage,
            provider_name=provider_name,
            model_name=model_name,
            is_completed=is_completed,
            is_cancelled=is_cancelled,
            is_failed=is_failed,
            is_terminal=is_terminal,
            error=error,
            attempt_count=attempt_count,
            retry_count=retry_count,
            provider_called=provider_called,
            evidence_count=evidence_count,
            result_status=str(status) if status is not None else None,
            stages=tuple(stages_list),
            metadata=dict(meta),
        )

    @classmethod
    def from_lifecycle(
        cls,
        lifecycle: VisionExecutionLifecycle,
        trace: Any = None,
        evidence_count: int = 0,
        result_status: str | None = None,
    ) -> VisionExecutionObservation:
        """Construct an immutable VisionExecutionObservation from a VisionExecutionLifecycle instance.

        Args:
            lifecycle: Active or completed VisionExecutionLifecycle instance.
            trace: Optional VisionExecutionTrace or dictionary containing trace stages.
            evidence_count: Number of visual evidence items processed.
            result_status: Optional result status string.

        Returns:
            Immutable VisionExecutionObservation snapshot.

        Raises:
            VisionInputValidationError: If lifecycle is not a VisionExecutionLifecycle instance.
        """
        if not isinstance(lifecycle, VisionExecutionLifecycle):
            raise VisionInputValidationError(
                f"lifecycle must be a VisionExecutionLifecycle instance, got {type(lifecycle).__name__}."
            )

        stages_list: list[str] = []
        if trace is not None:
            if hasattr(trace, "stages") and isinstance(trace.stages, list):
                stages_list = [str(s) for s in trace.stages]
            elif isinstance(trace, dict) and "stages" in trace:
                raw_s = trace.get("stages", [])
                if isinstance(raw_s, list):
                    stages_list = [str(s) for s in raw_s]

        provider_called = "provider_completed" in stages_list or "provider_started" in stages_list

        return cls(
            stage=lifecycle.stage,
            provider_name=lifecycle.provider_name,
            model_name=lifecycle.model_name,
            is_completed=lifecycle.is_completed,
            is_cancelled=lifecycle.is_cancelled,
            is_failed=lifecycle.is_failed,
            is_terminal=lifecycle.is_terminal,
            error=lifecycle.error,
            attempt_count=lifecycle.attempt_count,
            retry_count=lifecycle.retry_count,
            provider_called=provider_called,
            evidence_count=evidence_count,
            result_status=result_status,
            stages=tuple(stages_list),
            metadata=dict(lifecycle.metadata),
        )
