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

    ALL_STAGES: frozenset[str] = frozenset({
        "pending",
        "validating",
        "preparing",
        "building_input",
        "executing",
        "completed",
        "failed",
        "timeout",
    })

    TERMINAL_STAGES: frozenset[str] = frozenset({"completed", "failed", "timeout"})


@dataclass
class VisionExecutionLifecycle:
    """Deterministic, vendor-agnostic container for tracking provider execution lifecycle states.

    Maintains current execution stage, provider identity, error status, and stage-specific
    metadata without storing secrets, credentials, or fake latency data.

    Attributes:
        stage: Current execution stage string (must be in VisionExecutionStage.ALL_STAGES).
        provider_name: Identifier of the target vision provider.
        model_name: Designated vision model name.
        error: Error message string if execution reached a failed or timeout state.
        metadata: Stage transition metadata dictionary.
    """

    stage: str = VisionExecutionStage.PENDING
    provider_name: str = "unknown"
    model_name: str = "unknown"
    error: str | None = None
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
    def is_failed(self) -> bool:
        """Whether execution reached a failed or timeout stage."""
        return self.stage in (VisionExecutionStage.FAILED, VisionExecutionStage.TIMEOUT) or self.error is not None

    @property
    def is_terminal(self) -> bool:
        """Whether execution reached a terminal state."""
        return self.stage in VisionExecutionStage.TERMINAL_STAGES

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
            "is_completed": self.is_completed,
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
    trace stages, provider invocation status, and result metadata without
    external telemetry, network dependencies, or production overhead.

    Attributes:
        stage: Canonical lifecycle stage identifier.
        provider_name: Target provider identifier.
        model_name: Target model identifier.
        is_completed: Whether execution reached successful completion.
        is_failed: Whether execution reached a failed or timeout state.
        is_terminal: Whether execution reached a terminal state.
        error: Optional error description string.
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
    error: str | None = None
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
            "is_failed": self.is_failed,
            "is_terminal": self.is_terminal,
            "error": self.error,
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

        is_completed = bool(
            lifecycle_data.get(
                "is_completed",
                stage == VisionExecutionStage.COMPLETED and error is None and status != "error",
            )
        )
        is_failed = bool(
            lifecycle_data.get(
                "is_failed",
                stage in (VisionExecutionStage.FAILED, VisionExecutionStage.TIMEOUT)
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
            is_failed=is_failed,
            is_terminal=is_terminal,
            error=error,
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
            is_failed=lifecycle.is_failed,
            is_terminal=lifecycle.is_terminal,
            error=lifecycle.error,
            provider_called=provider_called,
            evidence_count=evidence_count,
            result_status=result_status,
            stages=tuple(stages_list),
            metadata=dict(lifecycle.metadata),
        )
