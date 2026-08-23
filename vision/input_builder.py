"""
Vision Input Builder for OmniBrain Member 3 Vision Agent.

Converts a validated PreparedImageEvidence (Day 34) and a user query string
into a standardized, lineage-preserving VisionModelInput contract ready for
consumption by a future Vision Model provider.

Day 35 Scope:
  - Define VisionModelInput frozen dataclass contract
  - Implement VisionInputBuilder with deterministic build() method
  - Query validation (non-empty, type-checked, stripped)
  - PreparedImageEvidence type enforcement (Day 34 only; raw VisualEvidence rejected)
  - Visual modality enforcement (image, chart, diagram only)
  - Full source lineage preservation (document_id, filename, page_number,
    chunk_id, chunk_index, content_type, metadata)
  - Technical image metadata preservation (image_format, width, height,
    mode, size_bytes, is_oversized)
  - Deterministic: same inputs -> identical VisionModelInput.to_dict() output
  - No external API calls, no LLM calls, no inference, no OCR, no captioning

Out of Scope (future days):
  - Vision model inference (OpenAI, Gemini, Claude, HuggingFace)
  - Query rewriting or LLM prompt optimization
  - Image resizing or transformation
  - OCR / captioning / chart interpretation
  - LangGraph / Supervisor / FastAPI / Streamlit
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vision.exceptions import (
    VisionEvidenceError,
    VisionInputValidationError,
)
from vision.image_preparation import PreparedImageEvidence
from vision.models import VALID_VISUAL_CONTENT_TYPES


# ---------------------------------------------------------------------------
# VisionModelInput
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisionModelInput:
    """Standardized, lineage-preserving input contract for a Vision Model provider.

    Produced exclusively by VisionInputBuilder from a validated PreparedImageEvidence
    (Day 34) and a user query.  All lineage and technical image metadata fields are
    copied verbatim from the source evidence; no values are generated or synthesised.

    This dataclass is frozen (immutable after construction) to prevent accidental
    mutation of critical lineage fields.

    Attributes:
        query: Validated, stripped user query or visual instruction.
        document_id: Source document identifier (preserved from PreparedImageEvidence).
        filename: Source filename (preserved from PreparedImageEvidence).
        page_number: Source 1-indexed page number, or None (preserved from source).
        chunk_id: Source chunk identifier (preserved from PreparedImageEvidence).
        chunk_index: Source 0-indexed chunk position (preserved from source).
        content_type: Visual modality tag ('image', 'chart', or 'diagram').
        image_format: Validated image format ('png', 'jpeg', or 'webp').
        width: Confirmed image width in pixels.
        height: Confirmed image height in pixels.
        mode: PIL image mode string (e.g. 'RGB', 'RGBA', 'L').
        size_bytes: Raw image data size in bytes.
        is_oversized: True if image was accepted despite exceeding the oversized policy.
        evidence_metadata: Source evidence metadata dictionary (shallow copy).
        builder_metadata: Optional caller-supplied metadata for routing/tracing.
    """

    query: str
    document_id: str
    filename: str
    page_number: int | None
    chunk_id: str
    chunk_index: int
    content_type: str
    image_format: str
    width: int
    height: int
    mode: str
    size_bytes: int
    is_oversized: bool
    evidence_metadata: dict[str, Any]
    builder_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate VisionModelInput field types and values."""
        # query
        if not isinstance(self.query, str) or not self.query.strip():
            raise VisionInputValidationError(
                "VisionModelInput.query must be a non-empty string."
            )

        # lineage strings
        for attr in ("document_id", "filename", "chunk_id", "content_type",
                     "image_format", "mode"):
            val = getattr(self, attr)
            if not isinstance(val, str) or not val.strip():
                raise VisionInputValidationError(
                    f"VisionModelInput.{attr} must be a non-empty string."
                )

        # page_number
        if self.page_number is not None:
            if (
                not isinstance(self.page_number, int)
                or isinstance(self.page_number, bool)
                or self.page_number <= 0
            ):
                raise VisionInputValidationError(
                    f"VisionModelInput.page_number must be a positive integer or None, "
                    f"got {self.page_number!r}."
                )

        # chunk_index
        if not isinstance(self.chunk_index, int) or isinstance(self.chunk_index, bool) or self.chunk_index < 0:
            raise VisionInputValidationError(
                f"VisionModelInput.chunk_index must be a non-negative integer, "
                f"got {self.chunk_index!r}."
            )

        # width / height / size_bytes
        for dim_attr in ("width", "height", "size_bytes"):
            val = getattr(self, dim_attr)
            if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
                raise VisionInputValidationError(
                    f"VisionModelInput.{dim_attr} must be a positive integer, got {val!r}."
                )

        # is_oversized
        if not isinstance(self.is_oversized, bool):
            raise VisionInputValidationError(
                "VisionModelInput.is_oversized must be a boolean."
            )

        # content_type must be a recognised visual modality
        if self.content_type not in VALID_VISUAL_CONTENT_TYPES:
            raise VisionEvidenceError(
                f"VisionModelInput.content_type '{self.content_type}' is not a valid "
                f"visual modality. Expected one of {sorted(VALID_VISUAL_CONTENT_TYPES)}."
            )

        # metadata dicts
        for meta_attr in ("evidence_metadata", "builder_metadata"):
            val = getattr(self, meta_attr)
            if not isinstance(val, (dict, Mapping)):
                raise VisionInputValidationError(
                    f"VisionModelInput.{meta_attr} must be a dictionary, "
                    f"got {type(val).__name__}."
                )

        # Freeze metadata dicts as plain dicts (frozen dataclass prevents direct
        # reassignment, so we use object.__setattr__)
        object.__setattr__(self, "evidence_metadata", dict(self.evidence_metadata))
        object.__setattr__(self, "builder_metadata", dict(self.builder_metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a fully serialisable dictionary representation.

        The returned dictionary contains exactly the documented schema fields --
        no extra synthesised keys.
        """
        return {
            "query": self.query,
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "content_type": self.content_type,
            "image_format": self.image_format,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "size_bytes": self.size_bytes,
            "is_oversized": self.is_oversized,
            "evidence_metadata": dict(self.evidence_metadata),
            "builder_metadata": dict(self.builder_metadata),
        }


# ---------------------------------------------------------------------------
# VisionInputBuilder
# ---------------------------------------------------------------------------


class VisionInputBuilder:
    """Builds a standardized VisionModelInput from a PreparedImageEvidence and query.

    The builder is deterministic: the same query + PreparedImageEvidence always
    produces an identical VisionModelInput.to_dict() output.

    Responsibilities:
      1. Validate the user query (non-empty string, stripped).
      2. Validate the evidence is a PreparedImageEvidence (Day 34 only).
      3. Enforce visual modality (image, chart, diagram).
      4. Copy all lineage fields verbatim from evidence -> VisionModelInput.
      5. Copy all technical image metadata verbatim from evidence -> VisionModelInput.
      6. Attach optional caller-supplied builder_metadata.
      7. Return a frozen VisionModelInput.

    This builder performs NO inference, NO API calls, NO LLM calls,
    NO OCR, NO captioning, and NO image transformation.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        query: str,
        evidence: PreparedImageEvidence,
        *,
        builder_metadata: dict[str, Any] | None = None,
    ) -> VisionModelInput:
        """Build a VisionModelInput from a query and validated PreparedImageEvidence.

        Args:
            query: Natural language query or visual instruction for the Vision Model.
                   Must be a non-empty string; leading/trailing whitespace is stripped.
            evidence: A fully validated PreparedImageEvidence from Day 34.
                      Raw VisualEvidence is NOT accepted here.
            builder_metadata: Optional caller-supplied metadata dictionary for routing
                              or tracing.  Defaults to empty dict if None.

        Returns:
            Frozen VisionModelInput with all lineage and technical fields preserved.

        Raises:
            VisionInputValidationError: If query or builder_metadata are invalid.
            VisionEvidenceError: If evidence is None, wrong type, or non-visual.
        """
        validated_query = self._validate_query(query)
        self._validate_evidence(evidence)
        meta = self._validate_builder_metadata(builder_metadata)

        return VisionModelInput(
            query=validated_query,
            document_id=evidence.document_id,
            filename=evidence.filename,
            page_number=evidence.page_number,
            chunk_id=evidence.chunk_id,
            chunk_index=evidence.chunk_index,
            content_type=evidence.content_type,
            image_format=evidence.image_format,
            width=evidence.width,
            height=evidence.height,
            mode=evidence.mode,
            size_bytes=evidence.size_bytes,
            is_oversized=evidence.is_oversized,
            evidence_metadata=dict(evidence.metadata),
            builder_metadata=meta,
        )

    # ------------------------------------------------------------------
    # Private validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(query: Any) -> str:
        """Validate and normalise the user query.

        Args:
            query: Raw query input from caller.

        Returns:
            Stripped, validated query string.

        Raises:
            VisionInputValidationError: If query is None, not a string, empty,
                                        or whitespace-only.
        """
        if query is None:
            raise VisionInputValidationError(
                "query cannot be None; a non-empty string is required."
            )
        if not isinstance(query, str):
            raise VisionInputValidationError(
                f"query must be a string, got {type(query).__name__}."
            )
        stripped = query.strip()
        if not stripped:
            raise VisionInputValidationError(
                "query cannot be empty or whitespace-only."
            )
        return stripped

    @staticmethod
    def _validate_evidence(evidence: Any) -> None:
        """Validate that evidence is a PreparedImageEvidence (Day 34 only).

        Raw VisualEvidence or any other type is rejected to enforce that the
        Day 34 preparation pipeline has been executed before building the input.

        Args:
            evidence: Candidate evidence object from caller.

        Raises:
            VisionEvidenceError: If evidence is None or not a PreparedImageEvidence.
            VisionInputValidationError: If evidence content_type is non-visual
                                        (defensive check; VisualEvidence already
                                        blocks this, but preserved for safety).
        """
        from vision.models import VisualEvidence  # local import avoids circularity

        if evidence is None:
            raise VisionEvidenceError(
                "evidence cannot be None; a PreparedImageEvidence instance is required."
            )

        # Explicitly detect raw VisualEvidence and provide a clear error message
        if isinstance(evidence, VisualEvidence):
            raise VisionEvidenceError(
                "Raw VisualEvidence is not accepted by VisionInputBuilder. "
                "Run the Day 34 image preparation pipeline first: "
                "prepare_image_evidence(visual_evidence) -> PreparedImageEvidence."
            )

        if not isinstance(evidence, PreparedImageEvidence):
            raise VisionEvidenceError(
                f"evidence must be a PreparedImageEvidence instance (Day 34), "
                f"got {type(evidence).__name__}."
            )

        # Visual modality check (defensive; PreparedImageEvidence already enforces
        # this through its source VisualEvidence, but validated here explicitly)
        if evidence.content_type not in VALID_VISUAL_CONTENT_TYPES:
            raise VisionEvidenceError(
                f"evidence.content_type '{evidence.content_type}' is not a valid "
                f"visual modality. Expected one of {sorted(VALID_VISUAL_CONTENT_TYPES)}."
            )

    @staticmethod
    def _validate_builder_metadata(metadata: Any) -> dict[str, Any]:
        """Validate and normalise the optional builder_metadata argument.

        Args:
            metadata: Raw metadata from caller (None is accepted -> empty dict).

        Returns:
            Plain dict (may be empty).

        Raises:
            VisionInputValidationError: If metadata is provided but is not a dict.
        """
        if metadata is None:
            return {}
        if not isinstance(metadata, (dict, Mapping)):
            raise VisionInputValidationError(
                f"builder_metadata must be a dictionary or None, "
                f"got {type(metadata).__name__}."
            )
        return dict(metadata)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def build_vision_input(
    query: str,
    evidence: PreparedImageEvidence,
    *,
    builder_metadata: dict[str, Any] | None = None,
) -> VisionModelInput:
    """Build a VisionModelInput from a query and validated PreparedImageEvidence.

    This is the primary public entry point for Day 35 input building.
    It delegates to VisionInputBuilder with the supplied arguments.

    Args:
        query: Natural language query or visual instruction.
        evidence: Validated PreparedImageEvidence (Day 34 output).
        builder_metadata: Optional metadata dictionary for routing or tracing.

    Returns:
        Frozen VisionModelInput with full lineage and technical metadata preserved.

    Raises:
        VisionInputValidationError: If query or builder_metadata are invalid.
        VisionEvidenceError: If evidence is None, wrong type, or non-visual.
    """
    builder = VisionInputBuilder()
    return builder.build(query, evidence, builder_metadata=builder_metadata)
