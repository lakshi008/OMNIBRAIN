"""
Image Evidence Preparation and Validation for OmniBrain Member 3 Vision Agent.

Accepts VisualEvidence objects (carrying image_path or image_bytes) and validates,
inspects, and wraps them into PreparedImageEvidence — a fully validated, lineage-
preserved container ready for future Vision Model inference.

Day 34 Scope:
  - Validate visual modality
  - Validate image source (path or bytes present)
  - Load and inspect image using Pillow
  - Validate image format (PNG, JPEG, WEBP)
  - Validate image dimensions (width > 0, height > 0)
  - Capture technical metadata (format, mode, width, height, size_bytes)
  - Enforce configurable oversized-image policy
  - Preserve full source lineage

Out of Scope:
  - Vision model inference
  - Image captioning or OCR
  - Image resizing or transformation
  - External API calls
  - New PDF extraction (Member 1 owns that)
  - Query embedding or retrieval (Member 2 owns that)
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any

from vision.exceptions import (
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
)
from vision.models import VALID_VISUAL_CONTENT_TYPES, VisualEvidence

# ---------------------------------------------------------------------------
# Supported image formats
# ---------------------------------------------------------------------------

#: Set of normalised format strings accepted by the preparation pipeline.
SUPPORTED_IMAGE_FORMATS: frozenset[str] = frozenset({"png", "jpeg", "webp"})

#: Pillow format name → normalised format string mapping.
_PILLOW_FORMAT_MAP: dict[str, str] = {
    "PNG": "png",
    "JPEG": "jpeg",
    "MPO": "jpeg",   # JPEG-compatible multi-picture extension
    "WEBP": "webp",
}

# ---------------------------------------------------------------------------
# Oversized-image policy
# ---------------------------------------------------------------------------

#: Default maximum dimension (width or height) in pixels.
#: Images with either dimension exceeding this value trigger the oversized policy.
DEFAULT_MAX_DIMENSION: int = 16_000

#: Default maximum total pixel count (width × height).
DEFAULT_MAX_PIXELS: int = 100_000_000  # 100 MP


@dataclass
class OversizedImagePolicy:
    """Configurable policy for images that exceed size thresholds.

    Attributes:
        max_dimension: Maximum allowed value for width or height (pixels).
        max_pixels: Maximum allowed total pixel count (width × height).
        reject: If True (default), oversized images raise VisionEvidenceError.
                If False, oversized images are accepted and flagged in metadata.
    """

    max_dimension: int = DEFAULT_MAX_DIMENSION
    max_pixels: int = DEFAULT_MAX_PIXELS
    reject: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.max_dimension, int) or self.max_dimension <= 0:
            raise VisionInputValidationError(
                "OversizedImagePolicy.max_dimension must be a positive integer."
            )
        if not isinstance(self.max_pixels, int) or self.max_pixels <= 0:
            raise VisionInputValidationError(
                "OversizedImagePolicy.max_pixels must be a positive integer."
            )
        if not isinstance(self.reject, bool):
            raise VisionInputValidationError(
                "OversizedImagePolicy.reject must be a boolean."
            )


# ---------------------------------------------------------------------------
# PreparedImageEvidence
# ---------------------------------------------------------------------------


@dataclass
class PreparedImageEvidence:
    """A fully validated and inspection-ready image evidence container.

    Wraps the source VisualEvidence and augments it with confirmed technical
    image attributes (format, dimensions, mode, size).  All lineage fields from
    the source are mirrored directly on this object for convenient access.

    Attributes:
        source: Original VisualEvidence from which this was prepared.
        document_id: Preserved source document identifier.
        filename: Preserved source filename.
        chunk_id: Preserved source chunk identifier.
        page_number: Preserved source page number (1-indexed, or None).
        chunk_index: Preserved source chunk index (0-indexed).
        content_type: Preserved visual modality ('image', 'chart', 'diagram').
        metadata: Preserved source metadata dictionary (shallow copy).
        image_format: Detected and normalised image format ('png', 'jpeg', 'webp').
        width: Confirmed image width in pixels.
        height: Confirmed image height in pixels.
        mode: PIL/Pillow image mode string (e.g. 'RGB', 'RGBA', 'L').
        size_bytes: Total size of the raw image data in bytes.
        is_oversized: True if the image exceeded the oversized policy thresholds
                      but the policy was configured to allow it (reject=False).
    """

    source: VisualEvidence
    document_id: str
    filename: str
    chunk_id: str
    page_number: int | None
    chunk_index: int
    content_type: str
    metadata: dict[str, Any]
    image_format: str
    width: int
    height: int
    mode: str
    size_bytes: int
    is_oversized: bool = False

    def __post_init__(self) -> None:
        """Validate that lineage fields exactly match the source VisualEvidence."""
        if not isinstance(self.source, VisualEvidence):
            raise VisionInputValidationError(
                f"source must be a VisualEvidence instance, got {type(self.source).__name__}."
            )
        # Enforce exact lineage match — no synthesized identifiers.
        if self.document_id != self.source.document_id:
            raise VisionEvidenceError(
                "PreparedImageEvidence.document_id must match source.document_id."
            )
        if self.filename != self.source.filename:
            raise VisionEvidenceError(
                "PreparedImageEvidence.filename must match source.filename."
            )
        if self.chunk_id != self.source.chunk_id:
            raise VisionEvidenceError(
                "PreparedImageEvidence.chunk_id must match source.chunk_id."
            )
        if self.chunk_index != self.source.chunk_index:
            raise VisionEvidenceError(
                "PreparedImageEvidence.chunk_index must match source.chunk_index."
            )
        if self.content_type != self.source.content_type:
            raise VisionEvidenceError(
                "PreparedImageEvidence.content_type must match source.content_type."
            )
        if self.page_number != self.source.page_number:
            raise VisionEvidenceError(
                "PreparedImageEvidence.page_number must match source.page_number."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable dictionary representation."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "content_type": self.content_type,
            "metadata": dict(self.metadata),
            "image_format": self.image_format,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "size_bytes": self.size_bytes,
            "is_oversized": self.is_oversized,
        }


# ---------------------------------------------------------------------------
# ImageEvidencePreparator
# ---------------------------------------------------------------------------


class ImageEvidencePreparator:
    """Prepares and validates VisualEvidence for future Vision Model processing.

    Responsibilities:
      1. Accept a VisualEvidence instance.
      2. Validate it carries a supported visual modality.
      3. Verify an image source (path or bytes) is present.
      4. Load the image using Pillow to detect format and dimensions.
      5. Validate format is in SUPPORTED_IMAGE_FORMATS.
      6. Validate dimensions are positive.
      7. Apply the configured OversizedImagePolicy.
      8. Collect technical metadata (format, mode, width, height, size_bytes).
      9. Preserve all lineage fields unchanged.
     10. Return PreparedImageEvidence.

    This class performs NO inference, NO captioning, NO OCR, and NO external
    API calls.  It only validates and structures existing image data.
    """

    def __init__(
        self,
        oversized_policy: OversizedImagePolicy | None = None,
    ) -> None:
        """Initialise the preparator with an optional oversized-image policy.

        Args:
            oversized_policy: Policy controlling how oversized images are handled.
                              Defaults to OversizedImagePolicy() (reject=True).
        """
        if oversized_policy is not None and not isinstance(oversized_policy, OversizedImagePolicy):
            raise VisionInputValidationError(
                "oversized_policy must be an OversizedImagePolicy instance or None."
            )
        self.oversized_policy: OversizedImagePolicy = oversized_policy or OversizedImagePolicy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, evidence: VisualEvidence) -> PreparedImageEvidence:
        """Validate and prepare a VisualEvidence instance for Vision processing.

        Args:
            evidence: VisualEvidence instance carrying image_path or image_bytes.

        Returns:
            PreparedImageEvidence with confirmed format, dimensions, and lineage.

        Raises:
            VisionInputValidationError: If evidence is None or wrong type,
                                        or if the evidence is non-visual.
            VisionEvidenceError: If no image source is present, the format is
                                 unsupported, or the oversized policy rejects it.
            VisionProcessingError: If the image data is corrupted or cannot
                                   be opened by Pillow.
        """
        self._validate_evidence_type(evidence)
        self._validate_visual_modality(evidence)

        image_data, size_bytes = self._load_image_source(evidence)
        pil_image = self._open_image(image_data, evidence)
        image_format = self._validate_format(pil_image, evidence)
        width, height = self._validate_dimensions(pil_image, evidence)
        mode = pil_image.mode
        is_oversized = self._apply_oversized_policy(width, height, evidence)

        return PreparedImageEvidence(
            source=evidence,
            document_id=evidence.document_id,
            filename=evidence.filename,
            chunk_id=evidence.chunk_id,
            page_number=evidence.page_number,
            chunk_index=evidence.chunk_index,
            content_type=evidence.content_type,
            metadata=dict(evidence.metadata),
            image_format=image_format,
            width=width,
            height=height,
            mode=mode,
            size_bytes=size_bytes,
            is_oversized=is_oversized,
        )

    # ------------------------------------------------------------------
    # Private validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_evidence_type(evidence: Any) -> None:
        """Raise VisionInputValidationError if evidence is not a VisualEvidence."""
        if evidence is None:
            raise VisionInputValidationError(
                "evidence cannot be None; expected a VisualEvidence instance."
            )
        if not isinstance(evidence, VisualEvidence):
            raise VisionInputValidationError(
                f"evidence must be a VisualEvidence instance, got {type(evidence).__name__}."
            )

    @staticmethod
    def _validate_visual_modality(evidence: VisualEvidence) -> None:
        """Raise VisionInputValidationError if evidence is not a supported visual type."""
        if evidence.content_type not in VALID_VISUAL_CONTENT_TYPES:
            raise VisionInputValidationError(
                f"evidence.content_type '{evidence.content_type}' is not a supported visual "
                f"modality. Expected one of {sorted(VALID_VISUAL_CONTENT_TYPES)}."
            )

    @staticmethod
    def _load_image_source(evidence: VisualEvidence) -> tuple[bytes, int]:
        """Resolve and return raw image bytes and their size.

        Preference order: image_bytes → image_path.

        Args:
            evidence: The VisualEvidence to load image data from.

        Returns:
            Tuple of (raw_bytes, size_in_bytes).

        Raises:
            VisionEvidenceError: If no image source is present, the path does not
                                 exist, or the path/bytes are empty.
        """
        # 1. image_bytes takes priority
        if evidence.image_bytes is not None:
            if not isinstance(evidence.image_bytes, (bytes, bytearray)):
                raise VisionEvidenceError(
                    "evidence.image_bytes must be bytes or bytearray."
                )
            if len(evidence.image_bytes) == 0:
                raise VisionEvidenceError(
                    "evidence.image_bytes is empty (0 bytes); cannot prepare image."
                )
            raw: bytes = bytes(evidence.image_bytes)
            return raw, len(raw)

        # 2. image_path fallback
        if evidence.image_path is not None:
            if not isinstance(evidence.image_path, str) or not evidence.image_path.strip():
                raise VisionEvidenceError(
                    "evidence.image_path must be a non-empty string."
                )
            path = evidence.image_path.strip()
            if not os.path.exists(path):
                raise VisionEvidenceError(
                    f"Image file not found at path: '{path}'."
                )
            if not os.path.isfile(path):
                raise VisionEvidenceError(
                    f"Image path is not a file: '{path}'."
                )
            file_size = os.path.getsize(path)
            if file_size == 0:
                raise VisionEvidenceError(
                    f"Image file is empty (0 bytes) at path: '{path}'."
                )
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError as exc:
                raise VisionEvidenceError(
                    f"Cannot read image file at path '{path}': {exc}"
                ) from exc
            return data, len(data)

        # 3. No image source at all
        raise VisionEvidenceError(
            "VisualEvidence has no image source: both image_bytes and image_path are None. "
            "Supply one before calling prepare()."
        )

    @staticmethod
    def _open_image(image_data: bytes, evidence: VisualEvidence):  # type: ignore[return]
        """Open raw image bytes with Pillow and return the Image object.

        Args:
            image_data: Raw image bytes.
            evidence: Source VisualEvidence (used in error messages).

        Returns:
            PIL.Image.Image instance (loaded into memory).

        Raises:
            VisionProcessingError: If Pillow cannot open or decode the image.
        """
        try:
            from PIL import Image, UnidentifiedImageError  # type: ignore[import-untyped]
        except ImportError as exc:
            raise VisionProcessingError(
                "Pillow (PIL) is required for image evidence preparation but is not installed."
            ) from exc

        try:
            pil_image = Image.open(io.BytesIO(image_data))
            pil_image.load()  # Force full decode to catch corruption early
            return pil_image
        except UnidentifiedImageError as exc:
            raise VisionProcessingError(
                f"Cannot identify image format for evidence chunk_id='{evidence.chunk_id}': {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise VisionProcessingError(
                f"Corrupted or unreadable image for evidence chunk_id='{evidence.chunk_id}': {exc}"
            ) from exc

    @staticmethod
    def _validate_format(pil_image: Any, evidence: VisualEvidence) -> str:
        """Detect and validate image format from a Pillow Image.

        Args:
            pil_image: Loaded PIL.Image.Image instance.
            evidence: Source VisualEvidence (used in error messages).

        Returns:
            Normalised format string ('png', 'jpeg', or 'webp').

        Raises:
            VisionEvidenceError: If the format is unsupported or undetectable.
        """
        raw_format: str | None = pil_image.format  # may be None for synthetic images
        if raw_format is None:
            # Attempt to derive from evidence.image_format hint
            hint = (evidence.image_format or "").strip().upper()
            if hint in _PILLOW_FORMAT_MAP:
                return _PILLOW_FORMAT_MAP[hint]
            raise VisionEvidenceError(
                f"Cannot detect image format for evidence chunk_id='{evidence.chunk_id}'. "
                "Pillow returned no format and no image_format hint was provided."
            )

        normalised = _PILLOW_FORMAT_MAP.get(raw_format)
        if normalised is None:
            raise VisionEvidenceError(
                f"Unsupported image format '{raw_format}' for evidence "
                f"chunk_id='{evidence.chunk_id}'. "
                f"Supported formats: {sorted(SUPPORTED_IMAGE_FORMATS)}."
            )
        return normalised

    def _validate_dimensions(
        self, pil_image: Any, evidence: VisualEvidence
    ) -> tuple[int, int]:
        """Extract and validate image dimensions from a Pillow Image.

        Args:
            pil_image: Loaded PIL.Image.Image instance.
            evidence: Source VisualEvidence (used in error messages).

        Returns:
            Tuple (width, height) both > 0.

        Raises:
            VisionEvidenceError: If width or height is zero or negative.
        """
        width, height = pil_image.size  # (width, height)

        if not isinstance(width, int) or width <= 0:
            raise VisionEvidenceError(
                f"Image has invalid width ({width}) for evidence "
                f"chunk_id='{evidence.chunk_id}'. Width must be > 0."
            )
        if not isinstance(height, int) or height <= 0:
            raise VisionEvidenceError(
                f"Image has invalid height ({height}) for evidence "
                f"chunk_id='{evidence.chunk_id}'. Height must be > 0."
            )
        return width, height

    def _apply_oversized_policy(
        self, width: int, height: int, evidence: VisualEvidence
    ) -> bool:
        """Apply the configured oversized-image policy.

        Args:
            width: Confirmed image width in pixels.
            height: Confirmed image height in pixels.
            evidence: Source VisualEvidence (used in error messages).

        Returns:
            True if image is oversized but policy allows it, False otherwise.

        Raises:
            VisionEvidenceError: If the image is oversized and policy.reject is True.
        """
        policy = self.oversized_policy
        total_pixels = width * height

        dimension_exceeded = (width > policy.max_dimension or height > policy.max_dimension)
        pixels_exceeded = total_pixels > policy.max_pixels

        if dimension_exceeded or pixels_exceeded:
            reason_parts = []
            if dimension_exceeded:
                reason_parts.append(
                    f"max dimension {max(width, height)}px > allowed {policy.max_dimension}px"
                )
            if pixels_exceeded:
                reason_parts.append(
                    f"total pixels {total_pixels:,} > allowed {policy.max_pixels:,}"
                )
            reason = "; ".join(reason_parts)

            if policy.reject:
                raise VisionEvidenceError(
                    f"Image is oversized for evidence chunk_id='{evidence.chunk_id}' "
                    f"({reason}). Rejected by OversizedImagePolicy."
                )
            return True  # allowed but flagged

        return False


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def prepare_image_evidence(
    evidence: VisualEvidence,
    *,
    oversized_policy: OversizedImagePolicy | None = None,
) -> PreparedImageEvidence:
    """Validate and prepare a VisualEvidence instance for Vision Model processing.

    This is the primary public entry point for Day 34 image preparation.
    It delegates to ImageEvidencePreparator with the supplied policy.

    Args:
        evidence: VisualEvidence carrying image_path or image_bytes.
        oversized_policy: Optional policy for oversized image handling.
                          Defaults to OversizedImagePolicy() (reject=True).

    Returns:
        PreparedImageEvidence with confirmed format, dimensions, and full lineage.

    Raises:
        VisionInputValidationError: If evidence is invalid or non-visual.
        VisionEvidenceError: If image source, format, or dimensions are invalid.
        VisionProcessingError: If the image data is corrupted or unreadable.
    """
    preparator = ImageEvidencePreparator(oversized_policy=oversized_policy)
    return preparator.prepare(evidence)
