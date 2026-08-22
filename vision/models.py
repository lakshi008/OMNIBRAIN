"""
Domain models for OmniBrain Member 3 Vision Agent subsystem.

Provides structured request, visual evidence, and result contracts
for visual evidence analysis, chart interpretation, and diagram reasoning
while strictly preserving end-to-end document lineage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agents.models import AgentCitation
from ingestion.models import VectorSearchResult
from vision.exceptions import VisionEvidenceError, VisionInputValidationError

# Recognized visual evidence content types in OmniBrain
VALID_VISUAL_CONTENT_TYPES: frozenset[str] = frozenset({"image", "chart", "diagram"})


@dataclass
class VisualEvidence:
    """Represents a unit of visual evidence (PDF image, chart, or diagram) with lineage.

    Attributes:
        document_id: Unique parent document identifier.
        filename: Name of source document file.
        chunk_id: Unique chunk identifier for this visual evidence unit.
        page_number: Source 1-indexed page number if known, or None.
        chunk_index: Non-negative sequential index within document.
        content_type: Visual modality tag ('image', 'chart', or 'diagram').
        image_path: Optional local filesystem path to extracted image file.
        image_bytes: Optional raw byte payload of the image.
        image_format: Optional format string ('png', 'jpeg', 'webp', etc.).
        width: Optional pixel width.
        height: Optional pixel height.
        description: Optional textual description or OCR text.
        metadata: Arbitrary metadata dictionary preserving source attributes.
    """

    document_id: str
    filename: str
    chunk_id: str
    page_number: int | None = None
    chunk_index: int = 0
    content_type: str = "image"
    image_path: str | None = None
    image_bytes: bytes | None = None
    image_format: str | None = None
    width: int | None = None
    height: int | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate visual evidence lineage and attributes."""
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise VisionEvidenceError("document_id must be a non-empty string.")

        if not isinstance(self.filename, str) or not self.filename.strip():
            raise VisionEvidenceError("filename must be a non-empty string.")

        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise VisionEvidenceError("chunk_id must be a non-empty string.")

        if self.page_number is not None:
            if not isinstance(self.page_number, int) or isinstance(self.page_number, bool) or self.page_number <= 0:
                raise VisionEvidenceError(
                    f"page_number must be a positive integer > 0 or None, got {self.page_number!r}."
                )

        if not isinstance(self.chunk_index, int) or isinstance(self.chunk_index, bool) or self.chunk_index < 0:
            raise VisionEvidenceError(
                f"chunk_index must be a non-negative integer >= 0, got {self.chunk_index!r}."
            )

        if not isinstance(self.content_type, str) or not self.content_type.strip():
            raise VisionEvidenceError("content_type must be a non-empty string.")

        normalized_type = self.content_type.strip().lower()
        if normalized_type not in VALID_VISUAL_CONTENT_TYPES:
            raise VisionEvidenceError(
                f"Invalid visual content_type '{self.content_type}'. "
                f"Must be one of {sorted(VALID_VISUAL_CONTENT_TYPES)}."
            )
        self.content_type = normalized_type

        if self.width is not None and (not isinstance(self.width, int) or isinstance(self.width, bool) or self.width <= 0):
            raise VisionEvidenceError(f"width must be a positive integer > 0 or None, got {self.width!r}.")

        if self.height is not None and (not isinstance(self.height, int) or isinstance(self.height, bool) or self.height <= 0):
            raise VisionEvidenceError(f"height must be a positive integer > 0 or None, got {self.height!r}.")

        if not isinstance(self.metadata, (dict, Mapping)):
            raise VisionEvidenceError(f"metadata must be a dictionary, got {type(self.metadata).__name__}.")

        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Convert VisualEvidence to a serializable dictionary."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "content_type": self.content_type,
            "image_path": self.image_path,
            "image_format": self.image_format,
            "width": self.width,
            "height": self.height,
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisualEvidence:
        """Construct VisualEvidence from dictionary."""
        if not isinstance(data, dict):
            raise VisionEvidenceError("Input data must be a dictionary.")

        return cls(
            document_id=data.get("document_id", ""),
            filename=data.get("filename", ""),
            chunk_id=data.get("chunk_id", ""),
            page_number=data.get("page_number"),
            chunk_index=data.get("chunk_index", 0),
            content_type=data.get("content_type", "image"),
            image_path=data.get("image_path"),
            image_format=data.get("image_format"),
            width=data.get("width"),
            height=data.get("height"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_citation(
        cls,
        citation: AgentCitation,
        image_path: str | None = None,
        image_bytes: bytes | None = None,
        image_format: str | None = None,
    ) -> VisualEvidence:
        """Construct VisualEvidence from an AgentCitation produced by Member 2 SearchAgent.

        Args:
            citation: AgentCitation with content_type in ('image', 'chart', 'diagram').
            image_path: Optional path to extracted image file.
            image_bytes: Optional raw image byte payload.
            image_format: Optional image format.

        Returns:
            VisualEvidence preserving all citation lineage fields.
        """
        if not isinstance(citation, AgentCitation):
            raise VisionEvidenceError(
                f"Expected AgentCitation instance, got {type(citation).__name__}."
            )

        chunk_idx = citation.metadata.get("chunk_index", 0) if isinstance(citation.metadata, dict) else 0
        return cls(
            document_id=citation.document_id,
            filename=citation.filename,
            chunk_id=citation.chunk_id,
            page_number=citation.page_number,
            chunk_index=chunk_idx if isinstance(chunk_idx, int) and not isinstance(chunk_idx, bool) and chunk_idx >= 0 else 0,
            content_type=citation.content_type,
            image_path=image_path,
            image_bytes=image_bytes,
            image_format=image_format,
            metadata=dict(citation.metadata),
        )

    @classmethod
    def from_search_result(
        cls,
        result: VectorSearchResult,
        image_path: str | None = None,
        image_bytes: bytes | None = None,
        image_format: str | None = None,
    ) -> VisualEvidence:
        """Construct VisualEvidence from a Member 1 VectorSearchResult.

        Args:
            result: VectorSearchResult with content_type in ('image', 'chart', 'diagram').
            image_path: Optional path to extracted image file.
            image_bytes: Optional raw image byte payload.
            image_format: Optional image format.

        Returns:
            VisualEvidence preserving all vector search lineage fields.
        """
        if not isinstance(result, VectorSearchResult):
            raise VisionEvidenceError(
                f"Expected VectorSearchResult instance, got {type(result).__name__}."
            )

        return cls(
            document_id=result.document_id,
            filename=result.filename,
            chunk_id=result.chunk_id,
            page_number=result.page_number,
            chunk_index=result.chunk_index,
            content_type=result.content_type,
            image_path=image_path,
            image_bytes=image_bytes,
            image_format=image_format,
            description=result.content,
            metadata=dict(result.metadata),
        )


@dataclass
class VisionRequest:
    """Request contract for the Vision Agent.

    Attributes:
        query: Natural language query or prompt describing visual analysis goal.
        evidence: List of VisualEvidence items to analyze.
        metadata: Optional metadata dictionary.
        session_id: Optional session identifier for conversation tracing.
    """

    query: str
    evidence: list[VisualEvidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None

    def __post_init__(self) -> None:
        """Validate vision request attributes."""
        if not isinstance(self.query, str):
            raise VisionInputValidationError(
                f"query must be a string, got {type(self.query).__name__}."
            )
        cleaned_query = self.query.strip()
        if not cleaned_query:
            raise VisionInputValidationError("query cannot be empty or whitespace-only.")
        self.query = cleaned_query

        if not isinstance(self.evidence, list):
            raise VisionInputValidationError(
                f"evidence must be a list of VisualEvidence, got {type(self.evidence).__name__}."
            )

        for idx, item in enumerate(self.evidence):
            if not isinstance(item, VisualEvidence):
                raise VisionInputValidationError(
                    f"Item at index {idx} in evidence is not a VisualEvidence instance: got {type(item).__name__}."
                )

        if not isinstance(self.metadata, (dict, Mapping)):
            raise VisionInputValidationError(
                f"metadata must be a dictionary, got {type(self.metadata).__name__}."
            )
        self.metadata = dict(self.metadata)

        if self.session_id is not None:
            if not isinstance(self.session_id, str) or not self.session_id.strip():
                raise VisionInputValidationError("session_id must be a non-empty string or None.")
            self.session_id = self.session_id.strip()

    @property
    def has_evidence(self) -> bool:
        """Whether request includes at least one visual evidence item."""
        return len(self.evidence) > 0

    @property
    def total_evidence(self) -> int:
        """Count of visual evidence items in request."""
        return len(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        """Convert request to dictionary representation."""
        return {
            "query": self.query,
            "evidence": [e.to_dict() for e in self.evidence],
            "metadata": dict(self.metadata),
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisionRequest:
        """Construct VisionRequest from dictionary representation."""
        if not isinstance(data, dict):
            raise VisionInputValidationError("Input data must be a dictionary.")

        raw_evidence = data.get("evidence", [])
        evidence: list[VisualEvidence] = []
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                if isinstance(item, VisualEvidence):
                    evidence.append(item)
                elif isinstance(item, dict):
                    evidence.append(VisualEvidence.from_dict(item))
                else:
                    raise VisionInputValidationError(
                        f"Invalid visual evidence in data: expected dict or VisualEvidence, got {type(item).__name__}."
                    )

        return cls(
            query=data.get("query", ""),
            evidence=evidence,
            metadata=data.get("metadata", {}),
            session_id=data.get("session_id"),
        )


@dataclass
class VisionResult:
    """Output contract produced by Vision Agent preserving lineage and analysis.

    Attributes:
        query: Original user query.
        status: Execution status ('success', 'no_evidence', 'error', 'not_implemented').
        description: Extracted textual analysis, description, or chart summary.
        evidence: List of VisualEvidence items analyzed.
        document_id: Parent document identifier (from primary visual evidence, if present).
        filename: Source filename (from primary visual evidence, if present).
        page_number: Source page number (from primary visual evidence, if present).
        chunk_id: Source chunk identifier (from primary visual evidence, if present).
        content_type: Modality tag ('image', 'chart', 'diagram').
        metadata: Processing metadata and statistics.
        error: Error message string if processing encountered a failure.
    """

    query: str
    status: str = "success"
    description: str = ""
    evidence: list[VisualEvidence] = field(default_factory=list)
    document_id: str = ""
    filename: str = ""
    page_number: int | None = None
    chunk_id: str = ""
    content_type: str = "image"
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate vision result attributes and lineage."""
        if not isinstance(self.query, str) or not self.query.strip():
            raise VisionInputValidationError("query must be a non-empty string.")

        if not isinstance(self.status, str) or not self.status.strip():
            raise VisionInputValidationError("status must be a non-empty string.")

        if not isinstance(self.description, str):
            raise VisionInputValidationError("description must be a string.")

        if not isinstance(self.evidence, list):
            raise VisionInputValidationError("evidence must be a list.")

        for idx, item in enumerate(self.evidence):
            if not isinstance(item, VisualEvidence):
                raise VisionInputValidationError(
                    f"Item at index {idx} in evidence is not VisualEvidence: got {type(item).__name__}."
                )

        if not isinstance(self.metadata, (dict, Mapping)):
            raise VisionInputValidationError("metadata must be a dictionary.")

        self.metadata = dict(self.metadata)

        if self.error is not None and not isinstance(self.error, str):
            raise VisionInputValidationError("error must be a string or None.")

        # If primary lineage fields not explicitly supplied, inherit from first evidence item
        if self.evidence:
            first = self.evidence[0]
            if not self.document_id:
                self.document_id = first.document_id
            if not self.filename:
                self.filename = first.filename
            if self.page_number is None:
                self.page_number = first.page_number
            if not self.chunk_id:
                self.chunk_id = first.chunk_id
            if self.content_type == "image" and first.content_type != "image":
                self.content_type = first.content_type

    @property
    def is_success(self) -> bool:
        """Whether result represents successful processing without errors."""
        return self.status == "success" and self.error is None

    @property
    def is_error(self) -> bool:
        """Whether result represents an error condition."""
        return self.status == "error" or self.error is not None

    @property
    def has_evidence(self) -> bool:
        """Whether result is backed by at least one visual evidence item."""
        return len(self.evidence) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert VisionResult to dictionary representation."""
        return {
            "query": self.query,
            "status": self.status,
            "description": self.description,
            "evidence": [e.to_dict() for e in self.evidence],
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "content_type": self.content_type,
            "metadata": dict(self.metadata),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisionResult:
        """Construct VisionResult from dictionary."""
        if not isinstance(data, dict):
            raise VisionInputValidationError("Input data must be a dictionary.")

        raw_evidence = data.get("evidence", [])
        evidence: list[VisualEvidence] = []
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                if isinstance(item, VisualEvidence):
                    evidence.append(item)
                elif isinstance(item, dict):
                    evidence.append(VisualEvidence.from_dict(item))
                else:
                    raise VisionInputValidationError(
                        f"Invalid visual evidence in data: expected dict or VisualEvidence, got {type(item).__name__}."
                    )

        return cls(
            query=data.get("query", ""),
            status=data.get("status", "success"),
            description=data.get("description", ""),
            evidence=evidence,
            document_id=data.get("document_id", ""),
            filename=data.get("filename", ""),
            page_number=data.get("page_number"),
            chunk_id=data.get("chunk_id", ""),
            content_type=data.get("content_type", "image"),
            metadata=data.get("metadata", {}),
            error=data.get("error"),
        )
