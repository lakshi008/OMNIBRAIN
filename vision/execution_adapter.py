"""
Vision Execution Adapter for OmniBrain Member 3 Vision Agent.

Orchestrates the multi-stage pipeline connecting incoming VisionRequest inputs,
retrieval evidence adaptation (Day 33), image preparation (Day 34),
input building (Day 35), and provider execution (Day 36) into a unified workflow.

Execution Pipeline:
    VisionRequest
          ↓
    Evidence Adaptation (VisualEvidenceAdapter)
          ↓
    Image Preparation (PreparedImageEvidence)
          ↓
    Input Building (VisionModelInput)
          ↓
    Provider Execution (VisionModelProvider.execute)
          ↓
    Standardized Result (VisionResult)
"""

from __future__ import annotations

from typing import Any

from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderError,
    VisionProviderExecutionError,
)
from vision.image_preparation import PreparedImageEvidence, prepare_image_evidence
from vision.input_builder import VisionModelInput, build_vision_input
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.provider import VisionModelProvider


class VisionExecutionAdapter:
    """Orchestrator between VisionAgent requests and VisionModelProvider backends.

    Encapsulates the complete transformation and validation lifecycle:
    1. Input request normalization and validation.
    2. Evidence normalization and adaptation via VisualEvidenceAdapter.
    3. Image inspection, format validation, and preparation via Day 34 pipeline.
    4. VisionModelInput construction and lineage locking via Day 35 pipeline.
    5. Delegated execution to an injected VisionModelProvider via Day 36 contract.
    6. Return of structured VisionResult without vendor-specific leakage.
    """

    def __init__(
        self,
        provider: VisionModelProvider,
        evidence_adapter: type[VisualEvidenceAdapter] = VisualEvidenceAdapter,
    ) -> None:
        """Initialize the execution adapter with an injected provider backend.

        Args:
            provider: Concrete implementation of VisionModelProvider.
            evidence_adapter: Adapter class for normalizing retrieval evidence.

        Raises:
            VisionInputValidationError: If provider is not a VisionModelProvider instance.
        """
        if provider is None:
            raise VisionInputValidationError("provider cannot be None.")

        if not isinstance(provider, VisionModelProvider):
            raise VisionInputValidationError(
                f"provider must be a VisionModelProvider instance, got {type(provider).__name__}."
            )

        self._provider = provider
        self._evidence_adapter = evidence_adapter

    @property
    def provider(self) -> VisionModelProvider:
        """Return the configured VisionModelProvider backend."""
        return self._provider

    def _normalize_request(
        self,
        request: str | VisionRequest,
        evidence: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VisionRequest:
        """Normalize raw input into a validated VisionRequest instance."""
        if request is None:
            raise VisionInputValidationError("Request cannot be None.")

        if isinstance(request, str):
            cleaned = request.strip()
            if not cleaned:
                raise VisionInputValidationError("Query cannot be empty or whitespace-only.")
            raw_evidence = list(evidence) if evidence is not None else []
            adapted_evidence = self._normalize_evidence_list(raw_evidence)
            req_meta = dict(metadata or {})
            return VisionRequest(query=cleaned, evidence=adapted_evidence, metadata=req_meta)

        if isinstance(request, VisionRequest):
            combined_evidence = list(request.evidence)
            if evidence is not None:
                combined_evidence.extend(self._normalize_evidence_list(list(evidence)))

            merged_meta = dict(request.metadata)
            if metadata:
                merged_meta.update(metadata)

            return VisionRequest(
                query=request.query,
                evidence=combined_evidence,
                metadata=merged_meta,
                session_id=request.session_id,
            )

        raise VisionInputValidationError(
            f"Expected string or VisionRequest, got {type(request).__name__}."
        )

    def _normalize_evidence_list(self, raw_items: list[Any]) -> list[VisualEvidence]:
        """Convert a list of arbitrary evidence items into validated VisualEvidence objects."""
        adapted: list[VisualEvidence] = []
        for idx, item in enumerate(raw_items):
            if isinstance(item, VisualEvidence):
                adapted.append(item)
            elif self._evidence_adapter.is_visual(item):
                adapted.append(self._evidence_adapter.adapt(item))
            else:
                raise VisionEvidenceError(
                    f"Evidence item at index {idx} ({type(item).__name__}) is not a supported visual modality."
                )
        return adapted

    def execute(
        self,
        request: str | VisionRequest,
        evidence: list[Any] | None = None,
        **kwargs: Any,
    ) -> VisionResult:
        """Execute the multi-stage vision reasoning pipeline.

        Args:
            request: Query string or structured VisionRequest.
            evidence: Optional list of VisualEvidence, citations, search results, or chunks.
            **kwargs: Runtime arguments (e.g. builder_metadata, extra provider parameters).

        Returns:
            Structured VisionResult containing provider output, query, and source lineage.

        Raises:
            VisionInputValidationError: If request format or parameters are invalid.
            VisionEvidenceError: If visual evidence lineage or preparation fails.
            VisionProviderExecutionError: If provider execution fails.
            VisionProviderError: If provider contract or capabilities are violated.
        """
        # Step 1: Normalize and validate request and evidence
        vision_req = self._normalize_request(
            request, evidence=evidence, metadata=kwargs.get("metadata")
        )

        # Step 2: Handle no-evidence requests gracefully
        if not vision_req.has_evidence:
            return VisionResult(
                query=vision_req.query,
                status="no_evidence",
                description="",
                evidence=[],
                metadata=dict(vision_req.metadata),
            )

        # Step 3: Prepare visual evidence via Day 34 pipeline
        primary_ev = vision_req.evidence[0]
        prepared_evidence = prepare_image_evidence(primary_ev)

        # Step 4: Construct validated, lineage-locked VisionModelInput via Day 35 pipeline
        model_input = build_vision_input(
            query=vision_req.query,
            evidence=prepared_evidence,
            builder_metadata=kwargs.get("builder_metadata"),
        )

        # Step 5: Execute provider backend via Day 36 abstraction
        result = self._provider.execute(model_input, **kwargs)

        # Step 6: Validate result contract
        if not isinstance(result, VisionResult):
            raise VisionProcessingError(
                f"Provider returned unexpected result type: {type(result).__name__}, expected VisionResult."
            )

        return result


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def execute_vision_request(
    provider: VisionModelProvider,
    request: str | VisionRequest,
    evidence: list[Any] | None = None,
    **kwargs: Any,
) -> VisionResult:
    """Execute a vision request through the full multi-stage execution pipeline.

    Args:
        provider: Injected VisionModelProvider instance.
        request: Query string or VisionRequest.
        evidence: Optional list of visual evidence items.
        **kwargs: Runtime execution parameters.

    Returns:
        Standardized VisionResult preserving query and lineage.
    """
    adapter = VisionExecutionAdapter(provider=provider)
    return adapter.execute(request, evidence=evidence, **kwargs)
