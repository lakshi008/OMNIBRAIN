"""
Vision Agent for OmniBrain Member 3 Multi-Modal subsystem.

Coordinates visual evidence validation, image lineage tracking, chart/diagram
interpretation, and structured VisionResult delivery via provider abstraction.
"""

from __future__ import annotations

from typing import Any

from vision.exceptions import (
    VisionAgentError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
)
from vision.image_preparation import PreparedImageEvidence, prepare_image_evidence
from vision.input_builder import VisionModelInput, build_vision_input
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.provider import VisionModelProvider


class VisionAgent:
    """Agent responsible for visual evidence reasoning and diagram/chart analysis.

    Coordinates between retrieval evidence, Day 34 image preparation, Day 35 input
    building, and Day 36 VisionModelProvider abstraction without tight coupling to
    any specific model vendor.
    """

    def __init__(
        self,
        agent_name: str = "VisionAgent",
        model_name: str = "default-vision-model",
        provider: VisionModelProvider | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize VisionAgent with agent identity, configuration, and optional provider.

        Args:
            agent_name: Identifier for this vision agent instance.
            model_name: Designated vision model name or descriptor.
            provider: Optional VisionModelProvider instance for backend execution.
            metadata: Optional configuration metadata dictionary.

        Raises:
            VisionInputValidationError: If agent_name, model_name, or provider is invalid.
        """
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise VisionInputValidationError("agent_name must be a non-empty string.")

        if not isinstance(model_name, str) or not model_name.strip():
            raise VisionInputValidationError("model_name must be a non-empty string.")

        if provider is not None and not isinstance(provider, VisionModelProvider):
            raise VisionInputValidationError(
                f"provider must be a VisionModelProvider instance or None, "
                f"got {type(provider).__name__}."
            )

        self.agent_name = agent_name.strip()
        self.model_name = model_name.strip()
        self.provider: VisionModelProvider | None = provider
        self.metadata = dict(metadata or {})

    def _prepare_request(
        self,
        request: str | VisionRequest,
        evidence: list[VisualEvidence] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VisionRequest:
        """Normalize raw input into a validated VisionRequest instance."""
        if request is None:
            raise VisionInputValidationError("Request cannot be None.")

        if isinstance(request, str):
            cleaned = request.strip()
            if not cleaned:
                raise VisionInputValidationError("Query cannot be empty or whitespace-only.")
            evidence_list = list(evidence) if evidence is not None else []
            req_meta = dict(metadata or {})
            return VisionRequest(query=cleaned, evidence=evidence_list, metadata=req_meta)

        if isinstance(request, VisionRequest):
            if evidence is not None:
                # Merge evidence lists while preserving order
                combined_evidence = list(request.evidence) + list(evidence)
            else:
                combined_evidence = list(request.evidence)

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

    def analyze(
        self,
        request: str | VisionRequest,
        evidence: list[VisualEvidence] | None = None,
        **kwargs: Any,
    ) -> VisionResult:
        """Execute visual analysis for query and visual evidence.

        Args:
            request: Query string or structured VisionRequest.
            evidence: Optional list of VisualEvidence items.
            **kwargs: Additional runtime execution parameters.

        Returns:
            Structured VisionResult preserving source lineage and visual analysis.

        Raises:
            VisionInputValidationError: If input validation fails.
            VisionEvidenceError: If visual evidence lineage fails.
            VisionProcessingError: If visual model inference fails or is not configured.
        """
        vision_req = self._prepare_request(
            request, evidence=evidence, metadata=kwargs.get("metadata")
        )

        # If a concrete provider is configured, delegate execution through the provider contract
        if self.provider is not None:
            if not vision_req.has_evidence:
                return VisionResult(
                    query=vision_req.query,
                    status="no_evidence",
                    description="",
                    evidence=[],
                    metadata=dict(vision_req.metadata),
                )

            primary_ev = vision_req.evidence[0]
            # Convert raw evidence to PreparedImageEvidence via Day 34 pipeline
            prepared = prepare_image_evidence(primary_ev)
            # Build standardized VisionModelInput via Day 35 pipeline
            model_input = build_vision_input(
                query=vision_req.query,
                evidence=prepared,
                builder_metadata=kwargs.get("builder_metadata"),
            )
            # Execute through provider abstraction
            return self.provider.execute(model_input, **kwargs)

        # Default Day 32 behavior when no provider backend is configured:
        # Explicitly signals that model inference backend is not implemented without a provider.
        raise VisionProcessingError(
            f"Vision model inference backend for '{self.model_name}' is not implemented for Day 32 foundation."
        )

    def process(
        self,
        request: str | VisionRequest,
        evidence: list[VisualEvidence] | None = None,
        **kwargs: Any,
    ) -> VisionResult:
        """Alias for analyze method."""
        return self.analyze(request, evidence=evidence, **kwargs)

    def __call__(
        self,
        request: str | VisionRequest,
        evidence: list[VisualEvidence] | None = None,
        **kwargs: Any,
    ) -> VisionResult:
        """Allow calling VisionAgent instance directly as a callable."""
        return self.analyze(request, evidence=evidence, **kwargs)
