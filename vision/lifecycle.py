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
