"""
Exceptions for OmniBrain Member 3 Vision Agent subsystem.

Defines the domain exception hierarchy for visual evidence validation,
request formatting, lineage integrity, and vision processing workflows.
"""

from __future__ import annotations


class VisionAgentError(Exception):
    """Base exception for all Member 3 Vision Agent errors."""

    def __init__(self, message: str = "A vision agent error occurred.") -> None:
        super().__init__(message)
        self.message = message


class VisionInputValidationError(VisionAgentError):
    """Raised when query, visual evidence, or parameters fail validation."""

    def __init__(self, message: str = "Invalid input provided to Vision Agent.") -> None:
        super().__init__(message)


class VisionEvidenceError(VisionAgentError):
    """Raised when visual evidence lineage, data, or format is corrupted or missing."""

    def __init__(self, message: str = "Visual evidence lineage or format error.") -> None:
        super().__init__(message)


class VisionProcessingError(VisionAgentError):
    """Raised when visual evidence processing, model inference, or extraction fails."""

    def __init__(self, message: str = "Vision processing operation failed.") -> None:
        super().__init__(message)
