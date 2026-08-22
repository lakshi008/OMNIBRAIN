"""
OmniBrain Member 3 Vision Agent Subsystem.

Provides visual evidence models, exceptions, and contracts for
image, chart, and diagram reasoning and multi-modal integration.
"""

from vision.exceptions import (
    VisionAgentError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.vision_agent import VisionAgent

__all__ = [
    # Domain Models
    "VisualEvidence",
    "VisionRequest",
    "VisionResult",
    "VALID_VISUAL_CONTENT_TYPES",
    # Agents
    "VisionAgent",
    # Domain Exceptions
    "VisionAgentError",
    "VisionInputValidationError",
    "VisionEvidenceError",
    "VisionProcessingError",
]
