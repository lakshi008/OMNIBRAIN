"""
Tests for IngestionStatus tracking.

Verifies lifecycle state transitions (PENDING -> RUNNING -> COMPLETED / FAILED),
stage progression history, error recording, and transition validation.
"""

from __future__ import annotations

import pytest

from ingestion.ingestion_status import (
    IngestionStatus,
    PipelineStage,
    PipelineStatus,
)


class TestIngestionStatusLifecycle:
    """Tests for status tracking lifecycle and state transitions."""

    def test_initial_state_is_pending(self) -> None:
        """New tracker starts in PENDING status with empty completed_stages."""
        tracker = IngestionStatus()
        assert tracker.status == PipelineStatus.PENDING
        assert tracker.current_stage == PipelineStage.EXTRACTION
        assert tracker.completed_stages == []
        assert tracker.error is None
        assert tracker.original_error is None

    def test_start_transitions_to_running(self) -> None:
        """start() moves tracker to RUNNING and sets initial stage."""
        tracker = IngestionStatus()
        tracker.start(PipelineStage.EXTRACTION)
        assert tracker.status == PipelineStatus.RUNNING
        assert tracker.current_stage == PipelineStage.EXTRACTION

    def test_advance_stage_records_completed_stages(self) -> None:
        """advance_stage() appends previous stage to completed_stages."""
        tracker = IngestionStatus()
        tracker.start(PipelineStage.EXTRACTION)

        tracker.advance_stage(PipelineStage.CHUNKING)
        assert tracker.current_stage == PipelineStage.CHUNKING
        assert tracker.completed_stages == [PipelineStage.EXTRACTION]

        tracker.advance_stage(PipelineStage.NORMALIZATION)
        assert tracker.current_stage == PipelineStage.NORMALIZATION
        assert tracker.completed_stages == [
            PipelineStage.EXTRACTION,
            PipelineStage.CHUNKING,
        ]

    def test_complete_marks_pipeline_completed(self) -> None:
        """complete() moves status and current_stage to COMPLETED."""
        tracker = IngestionStatus()
        tracker.start(PipelineStage.EXTRACTION)
        tracker.advance_stage(PipelineStage.EMBEDDING_GENERATION)
        tracker.complete()

        assert tracker.status == PipelineStatus.COMPLETED
        assert tracker.current_stage == PipelineStage.COMPLETED
        assert PipelineStage.EMBEDDING_GENERATION in tracker.completed_stages

    def test_fail_records_error_details(self) -> None:
        """fail() transitions status to FAILED and records message and exception."""
        tracker = IngestionStatus()
        tracker.start(PipelineStage.EXTRACTION)
        root_cause = RuntimeError("Disk full")

        tracker.fail("Failed during extraction", original_error=root_cause)
        assert tracker.status == PipelineStatus.FAILED
        assert tracker.error == "Failed during extraction"
        assert tracker.original_error is root_cause

    def test_document_id_and_filename_tracked(self) -> None:
        """document_id and filename can be assigned and inspected."""
        tracker = IngestionStatus(
            document_id="doc-uuid-123",
            filename="quarterly_report.pdf",
        )
        assert tracker.document_id == "doc-uuid-123"
        assert tracker.filename == "quarterly_report.pdf"


class TestIngestionStatusValidation:
    """Tests for rejecting invalid state transitions."""

    def test_start_when_not_pending_raises(self) -> None:
        """start() on an already RUNNING tracker raises ValueError."""
        tracker = IngestionStatus()
        tracker.start()
        with pytest.raises(ValueError, match="Cannot start pipeline from status 'RUNNING'"):
            tracker.start()

    def test_advance_when_pending_raises(self) -> None:
        """advance_stage() before start() raises ValueError."""
        tracker = IngestionStatus()
        with pytest.raises(ValueError, match="Cannot advance stage while pipeline status is 'PENDING'"):
            tracker.advance_stage(PipelineStage.CHUNKING)

    def test_advance_when_completed_raises(self) -> None:
        """advance_stage() after complete() raises ValueError."""
        tracker = IngestionStatus()
        tracker.start()
        tracker.complete()
        with pytest.raises(ValueError, match="Cannot advance stage while pipeline status is 'COMPLETED'"):
            tracker.advance_stage(PipelineStage.EXTRACTION)

    def test_complete_when_pending_raises(self) -> None:
        """complete() on a PENDING tracker raises ValueError."""
        tracker = IngestionStatus()
        with pytest.raises(ValueError, match="Cannot complete pipeline while status is 'PENDING'"):
            tracker.complete()
