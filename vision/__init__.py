"""
OmniBrain Member 3 Vision Agent Subsystem.

Provides visual evidence models, image preparation, input building,
provider abstractions, execution adapters, and contracts for multi-modal reasoning.
"""

from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionAgentError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderConfigError,
    VisionProviderError,
    VisionProviderExecutionError,
    VisionProviderUnavailableError,
    VisionUnsupportedCapabilityError,
)
from vision.execution_adapter import (
    VisionExecutionAdapter,
    execute_vision_request,
)
from vision.image_preparation import (
    SUPPORTED_IMAGE_FORMATS,
    ImageEvidencePreparator,
    OversizedImagePolicy,
    PreparedImageEvidence,
    prepare_image_evidence,
)
from vision.input_builder import (
    VisionInputBuilder,
    VisionModelInput,
    build_vision_input,
)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.provider import (
    VisionModelProvider,
    VisionProviderRegistry,
)
from vision.provider_config import (
    VisionProviderCapabilities,
    VisionProviderConfig,
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
    # Vision Input Builder (Day 35)
    "VisionModelInput",
    "VisionInputBuilder",
    "build_vision_input",
    # Vision Model Provider Abstraction (Day 36)
    "VisionModelProvider",
    "VisionProviderRegistry",
    "VisionProviderConfig",
    "VisionProviderCapabilities",
    # Execution Adapter (Day 37)
    "VisionExecutionAdapter",
    "execute_vision_request",
    # Adapters
    "VisualEvidenceAdapter",
    # Agents
    "VisionAgent",
    # Domain Exceptions
    "VisionAgentError",
    "VisionInputValidationError",
    "VisionEvidenceError",
    "VisionProcessingError",
    "VisionProviderError",
    "VisionProviderConfigError",
    "VisionProviderExecutionError",
    "VisionProviderUnavailableError",
    "VisionUnsupportedCapabilityError",
]
