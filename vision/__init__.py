"""
OmniBrain Member 3 Vision Agent Subsystem.

Provides visual evidence models, exceptions, and contracts for
image, chart, and diagram reasoning and multi-modal integration.
"""

from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionAgentError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
)
from vision.image_preparation import (
    SUPPORTED_IMAGE_FORMATS,
    ImageEvidencePreparator,
    OversizedImagePolicy,
    PreparedImageEvidence,
    prepare_image_evidence,
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
    # Image Preparation (Day 34)
    "PreparedImageEvidence",
    "ImageEvidencePreparator",
    "OversizedImagePolicy",
    "SUPPORTED_IMAGE_FORMATS",
    "prepare_image_evidence",
    # Adapters
    "VisualEvidenceAdapter",
    # Agents
    "VisionAgent",
    # Domain Exceptions
    "VisionAgentError",
    "VisionInputValidationError",
    "VisionEvidenceError",
    "VisionProcessingError",
]

