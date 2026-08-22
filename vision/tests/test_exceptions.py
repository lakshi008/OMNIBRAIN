"""
Unit tests for Member 3 Vision Agent domain exceptions.
"""

from __future__ import annotations

import pytest

from vision.exceptions import (
    VisionAgentError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
)


class TestVisionExceptions:
    """Test suite for Vision Agent exception hierarchy and error propagation."""

    def test_01_vision_agent_error_base(self) -> None:
        """VisionAgentError inherits from Exception and stores message."""
        err = VisionAgentError("Base error message")
        assert isinstance(err, Exception)
        assert str(err) == "Base error message"
        assert err.message == "Base error message"

    def test_02_vision_input_validation_error_inheritance(self) -> None:
        """VisionInputValidationError inherits from VisionAgentError."""
        err = VisionInputValidationError("Invalid query parameter")
        assert isinstance(err, VisionAgentError)
        assert isinstance(err, Exception)
        assert "Invalid query" in str(err)

    def test_03_vision_evidence_error_inheritance(self) -> None:
        """VisionEvidenceError inherits from VisionAgentError."""
        err = VisionEvidenceError("Corrupted image lineage")
        assert isinstance(err, VisionAgentError)
        assert "Corrupted image" in str(err)

    def test_04_vision_processing_error_inheritance(self) -> None:
        """VisionProcessingError inherits from VisionAgentError."""
        err = VisionProcessingError("Model inference failed")
        assert isinstance(err, VisionAgentError)
        assert "Model inference failed" in str(err)

    def test_05_default_exception_messages(self) -> None:
        """Exceptions construct with sensible default messages."""
        assert "error occurred" in str(VisionAgentError())
        assert "Invalid input" in str(VisionInputValidationError())
        assert "lineage or format" in str(VisionEvidenceError())
        assert "processing operation failed" in str(VisionProcessingError())
