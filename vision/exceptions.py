"""
Exceptions for OmniBrain Member 3 Vision Agent subsystem.

Defines the domain exception hierarchy for visual evidence validation,
request formatting, lineage integrity, vision processing workflows,
vision model provider abstraction, and execution lifecycle/timeout boundaries.
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


# ---------------------------------------------------------------------------
# Provider-specific exception hierarchy (Day 36 & Day 38)
# ---------------------------------------------------------------------------


class VisionProviderError(VisionAgentError):
    """Base exception for all Vision Model Provider errors."""

    def __init__(self, message: str = "A vision provider error occurred.") -> None:
        super().__init__(message)


class VisionProviderConfigError(VisionProviderError, VisionInputValidationError):
    """Raised when provider configuration is invalid or missing required parameters."""

    def __init__(self, message: str = "Invalid vision provider configuration.") -> None:
        super().__init__(message)


class VisionProviderExecutionError(VisionProviderError, VisionProcessingError):
    """Raised when a vision provider encounters an error during execution or inference."""

    def __init__(self, message: str = "Vision provider execution failed.") -> None:
        super().__init__(message)


class VisionProviderUnavailableError(VisionProviderError, VisionProcessingError):
    """Raised when a requested vision provider is not available, not installed, or not reachable."""

    def __init__(self, message: str = "Vision provider is unavailable.") -> None:
        super().__init__(message)


class VisionUnsupportedCapabilityError(VisionProviderError, VisionProcessingError):
    """Raised when a vision input requires a capability not supported by the provider."""

    def __init__(self, message: str = "Requested capability is not supported by vision provider.") -> None:
        super().__init__(message)


class VisionTimeoutError(VisionProviderError, VisionProcessingError):
    """Raised when vision provider execution exceeds the configured timeout threshold."""

    def __init__(self, message: str = "Vision provider execution timed out.") -> None:
        super().__init__(message)


class VisionCancellationError(VisionAgentError):
    """Raised when vision execution is cancelled before or during processing."""

    def __init__(self, message: str = "Vision execution was cancelled.") -> None:
        super().__init__(message)

