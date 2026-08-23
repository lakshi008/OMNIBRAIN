"""
Visual Evidence Adapter for OmniBrain Member 3 Vision Agent subsystem.

Converts, validates, and normalizes retrieval evidence from Member 1 (VectorSearchResult, Chunk)
and Member 2 (AgentCitation, SearchResult, AgentResponse) into Vision-specific VisualEvidence
instances while strictly preserving source document lineage.
"""

from __future__ import annotations

from typing import Any

from agents.models import AgentCitation, AgentResponse, SearchResult
from ingestion.models import DocumentChunk, VectorSearchResult
from vision.exceptions import VisionEvidenceError, VisionInputValidationError
from vision.models import VALID_VISUAL_CONTENT_TYPES, VisualEvidence


class VisualEvidenceAdapter:
    """Adapter for validating and converting retrieval evidence into VisualEvidence contracts.

    Ensures that only genuine visual modalities (image, chart, diagram) are adapted,
    while guaranteeing that source lineage fields (document_id, filename, page_number,
    chunk_id, chunk_index, content_type, and metadata) are never modified, synthesized,
    or lost.
    """

    @staticmethod
    def is_visual_content_type(content_type: Any) -> bool:
        """Check whether a content_type string represents a supported visual modality.

        Args:
            content_type: String content type to check.

        Returns:
            True if content_type is in ('image', 'chart', 'diagram'), False otherwise.
        """
        if not isinstance(content_type, str):
            return False
        return content_type.strip().lower() in VALID_VISUAL_CONTENT_TYPES

    @classmethod
    def is_visual(cls, item: Any) -> bool:
        """Check whether an evidence object or dictionary represents visual evidence.

        Args:
            item: AgentCitation, VectorSearchResult, Chunk, VisualEvidence, or dict.

        Returns:
            True if item has a visual content_type, False otherwise.
        """
        if item is None:
            return False

        if isinstance(item, VisualEvidence):
            return True

        if isinstance(item, (AgentCitation, VectorSearchResult, DocumentChunk)):
            return cls.is_visual_content_type(item.content_type)

        if isinstance(item, dict):
            return cls.is_visual_content_type(item.get("content_type"))

        return False

    @classmethod
    def adapt_citation(
        cls,
        citation: AgentCitation,
        image_path: str | None = None,
        image_bytes: bytes | None = None,
        image_format: str | None = None,
    ) -> VisualEvidence:
        """Adapt a Member 2 AgentCitation into a VisualEvidence instance.

        Args:
            citation: AgentCitation with visual content_type ('image', 'chart', 'diagram').
            image_path: Optional filesystem path to image file.
            image_bytes: Optional raw image byte payload.
            image_format: Optional image format ('png', 'jpeg', etc.).

        Returns:
            VisualEvidence preserving all citation lineage fields.

        Raises:
            VisionInputValidationError: If citation is None or invalid type.
            VisionEvidenceError: If citation content_type is not visual or lineage is invalid.
        """
        if citation is None:
            raise VisionInputValidationError("Citation cannot be None.")

        if not isinstance(citation, AgentCitation):
            raise VisionInputValidationError(
                f"Expected AgentCitation instance, got {type(citation).__name__}."
            )

        if not cls.is_visual_content_type(citation.content_type):
            raise VisionEvidenceError(
                f"Unsupported content_type '{citation.content_type}' for visual adaptation. "
                f"Must be one of {sorted(VALID_VISUAL_CONTENT_TYPES)}."
            )

        return VisualEvidence.from_citation(
            citation=citation,
            image_path=image_path,
            image_bytes=image_bytes,
            image_format=image_format,
        )

    @classmethod
    def adapt_search_result(
        cls,
        result: VectorSearchResult,
        image_path: str | None = None,
        image_bytes: bytes | None = None,
        image_format: str | None = None,
    ) -> VisualEvidence:
        """Adapt a Member 1 VectorSearchResult into a VisualEvidence instance.

        Args:
            result: VectorSearchResult with visual content_type.
            image_path: Optional filesystem path to image file.
            image_bytes: Optional raw image byte payload.
            image_format: Optional image format.

        Returns:
            VisualEvidence preserving all vector search result lineage fields.

        Raises:
            VisionInputValidationError: If result is None or invalid type.
            VisionEvidenceError: If result content_type is not visual or lineage is invalid.
        """
        if result is None:
            raise VisionInputValidationError("VectorSearchResult cannot be None.")

        if not isinstance(result, VectorSearchResult):
            raise VisionInputValidationError(
                f"Expected VectorSearchResult instance, got {type(result).__name__}."
            )

        if not cls.is_visual_content_type(result.content_type):
            raise VisionEvidenceError(
                f"Unsupported content_type '{result.content_type}' for visual adaptation. "
                f"Must be one of {sorted(VALID_VISUAL_CONTENT_TYPES)}."
            )

        return VisualEvidence.from_search_result(
            result=result,
            image_path=image_path,
            image_bytes=image_bytes,
            image_format=image_format,
        )

    @classmethod
    def adapt_chunk(
        cls,
        chunk: DocumentChunk,
        image_path: str | None = None,
        image_bytes: bytes | None = None,
        image_format: str | None = None,
    ) -> VisualEvidence:
        """Adapt a Member 1 DocumentChunk into a VisualEvidence instance.

        Args:
            chunk: DocumentChunk with visual content_type.
            image_path: Optional filesystem path to image file.
            image_bytes: Optional raw image byte payload.
            image_format: Optional image format.

        Returns:
            VisualEvidence preserving all chunk lineage fields.

        Raises:
            VisionInputValidationError: If chunk is None or invalid type.
            VisionEvidenceError: If chunk content_type is not visual or lineage is invalid.
        """
        if chunk is None:
            raise VisionInputValidationError("Chunk cannot be None.")

        if not isinstance(chunk, DocumentChunk):
            raise VisionInputValidationError(
                f"Expected DocumentChunk instance, got {type(chunk).__name__}."
            )

        if not cls.is_visual_content_type(chunk.content_type):
            raise VisionEvidenceError(
                f"Unsupported content_type '{chunk.content_type}' for visual adaptation. "
                f"Must be one of {sorted(VALID_VISUAL_CONTENT_TYPES)}."
            )

        return VisualEvidence(
            document_id=chunk.document_id,
            filename=chunk.filename,
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content_type=chunk.content_type,
            image_path=image_path,
            image_bytes=image_bytes,
            image_format=image_format,
            description=chunk.content,
            metadata=dict(chunk.metadata),
        )

    @classmethod
    def adapt(cls, item: Any, **kwargs: Any) -> VisualEvidence:
        """Universal adapter method for converting a single evidence object to VisualEvidence.

        Args:
            item: VisualEvidence, AgentCitation, VectorSearchResult, DocumentChunk, or dict.
            **kwargs: Optional overrides (image_path, image_bytes, image_format).

        Returns:
            VisualEvidence instance.

        Raises:
            VisionInputValidationError: If item is None or an unsupported type.
            VisionEvidenceError: If item fails visual content type or lineage validation.
        """
        if item is None:
            raise VisionInputValidationError("Evidence item cannot be None.")

        if isinstance(item, VisualEvidence):
            return item

        if isinstance(item, AgentCitation):
            return cls.adapt_citation(item, **kwargs)

        if isinstance(item, VectorSearchResult):
            return cls.adapt_search_result(item, **kwargs)

        if isinstance(item, DocumentChunk):
            return cls.adapt_chunk(item, **kwargs)

        if isinstance(item, dict):
            content_type = item.get("content_type")
            if not cls.is_visual_content_type(content_type):
                raise VisionEvidenceError(
                    f"Unsupported content_type '{content_type}' in dictionary for visual adaptation."
                )
            return VisualEvidence.from_dict(item)

        raise VisionInputValidationError(
            f"Unsupported evidence object type for visual adaptation: {type(item).__name__}."
        )

    @classmethod
    def adapt_search_package(
        cls,
        package: SearchResult | AgentResponse,
        strict: bool = False,
    ) -> list[VisualEvidence]:
        """Extract and adapt visual evidence from a SearchResult or AgentResponse package.

        Args:
            package: SearchResult or AgentResponse instance.
            strict: If True, any non-visual citation raises VisionEvidenceError.
                    If False, non-visual citations are safely filtered out.

        Returns:
            List of VisualEvidence objects in original rank order.

        Raises:
            VisionInputValidationError: If package is None or invalid type.
            VisionEvidenceError: If strict=True and non-visual citations are encountered.
        """
        if package is None:
            raise VisionInputValidationError("Search package cannot be None.")

        if not isinstance(package, (SearchResult, AgentResponse)):
            raise VisionInputValidationError(
                f"Expected SearchResult or AgentResponse, got {type(package).__name__}."
            )

        visual_evidence_list: list[VisualEvidence] = []
        for idx, citation in enumerate(package.citations):
            if cls.is_visual_content_type(citation.content_type):
                visual_evidence_list.append(cls.adapt_citation(citation))
            elif strict:
                raise VisionEvidenceError(
                    f"Citation at index {idx} has non-visual content_type '{citation.content_type}' "
                    "in strict adaptation mode."
                )

        return visual_evidence_list

    @classmethod
    def adapt_batch(
        cls,
        items: list[Any],
        strict: bool = False,
        **kwargs: Any,
    ) -> list[VisualEvidence]:
        """Adapt a batch of evidence items into a list of VisualEvidence objects.

        Args:
            items: List of evidence items (AgentCitation, VectorSearchResult, Chunk, etc.).
            strict: If True, non-visual items raise VisionEvidenceError.
                    If False, non-visual items are safely skipped.
            **kwargs: Keyword arguments passed to individual adapter calls.

        Returns:
            List of VisualEvidence instances.

        Raises:
            VisionInputValidationError: If items is not a list.
            VisionEvidenceError: If strict=True and a non-visual item is encountered.
        """
        if not isinstance(items, list):
            raise VisionInputValidationError(
                f"items must be a list, got {type(items).__name__}."
            )

        results: list[VisualEvidence] = []
        for idx, item in enumerate(items):
            if cls.is_visual(item):
                results.append(cls.adapt(item, **kwargs))
            elif strict:
                raise VisionEvidenceError(
                    f"Item at index {idx} is not visual evidence in strict adaptation mode."
                )

        return results
