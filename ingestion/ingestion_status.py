"""
Structured pipeline status and stage tracking for OmniBrain ingestion.

Provides deterministic status transitions, stage progression history,
and error tracking across the ingestion lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PipelineStatus(str, Enum):
    """Execution status of the ingestion pipeline."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PipelineStage(str, Enum):
    """Individual stages within the ingestion pipeline."""

    EXTRACTION = "EXTRACTION"
    CHUNKING = "CHUNKING"
    NORMALIZATION = "NORMALIZATION"
    VALIDATION = "VALIDATION"
    EMBEDDING_PREPARATION = "EMBEDDING_PREPARATION"
    EMBEDDING_GENERATION = "EMBEDDING_GENERATION"
    COMPLETED = "COMPLETED"


@dataclass
class IngestionStatus:
    """Tracks status, current stage, completed stages, and error information.

    Attributes:
        status: Current pipeline execution status (PENDING, RUNNING, COMPLETED, FAILED).
        current_stage: The currently active or last reached pipeline stage.
        completed_stages: Ordered list of successfully completed stages.
        error: Descriptive error message if the pipeline failed.
        original_error: The underlying Exception instance if failed.
        document_id: Unique UUID identifier of the document once extracted.
        filename: Name of the source file.
    """

    status: PipelineStatus = PipelineStatus.PENDING
    current_stage: PipelineStage = PipelineStage.EXTRACTION
    completed_stages: list[PipelineStage] = field(default_factory=list)
    error: str | None = None
    original_error: Exception | None = None
    document_id: str | None = None
    filename: str | None = None

    def start(self, stage: PipelineStage = PipelineStage.EXTRACTION) -> None:
        """Start the pipeline from PENDING status.

        Args:
            stage: Initial stage to enter (default: EXTRACTION).

        Raises:
            ValueError: If the pipeline is already RUNNING, COMPLETED, or FAILED.
        """
        if self.status != PipelineStatus.PENDING:
            raise ValueError(f"Cannot start pipeline from status '{self.status.value}'.")
        self.status = PipelineStatus.RUNNING
        self.current_stage = stage

    def advance_stage(self, next_stage: PipelineStage) -> None:
        """Advance the pipeline to the next stage and record the completed stage.

        Args:
            next_stage: The next PipelineStage to transition into.

        Raises:
            ValueError: If the pipeline is not in RUNNING status.
        """
        if self.status != PipelineStatus.RUNNING:
            raise ValueError(
                f"Cannot advance stage while pipeline status is '{self.status.value}'."
            )
        if self.current_stage not in self.completed_stages:
            self.completed_stages.append(self.current_stage)
        self.current_stage = next_stage

    def complete(self) -> None:
        """Mark the pipeline as successfully COMPLETED.

        Raises:
            ValueError: If the pipeline is not in RUNNING status.
        """
        if self.status != PipelineStatus.RUNNING:
            raise ValueError(
                f"Cannot complete pipeline while status is '{self.status.value}'."
            )
        if (
            self.current_stage != PipelineStage.COMPLETED
            and self.current_stage not in self.completed_stages
        ):
            self.completed_stages.append(self.current_stage)
        self.status = PipelineStatus.COMPLETED
        self.current_stage = PipelineStage.COMPLETED

    def fail(self, error_message: str, original_error: Exception | None = None) -> None:
        """Mark the pipeline as FAILED and record error details.

        Args:
            error_message: Explanation of the failure.
            original_error: Optional underlying Exception.
        """
        self.status = PipelineStatus.FAILED
        self.error = error_message
        self.original_error = original_error
