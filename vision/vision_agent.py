"""
Vision Agent for OmniBrain Member 3 Multi-Modal subsystem.

Coordinates visual evidence validation, image lineage tracking, chart/diagram
interpretation, and structured VisionResult delivery.
"""

from __future__ import annotations

from typing import Any

from vision.exceptions import (
    VisionAgentError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
)
from vision.models import VisionRequest, VisionResult, VisualEvidence


class VisionAgent:
    """Agent responsible for visual evidence reasoning and diagram/chart analysis.

    For Day 32: Establishes architecture, domain contracts, validation rules,
    and dependency boundaries without running external API calls or generating
    fake production results.
    """

    def __init__(
        self,
        agent_name: str = "VisionAgent",
        model_name: str = "default-vision-model",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize VisionAgent with agent identity and configuration.

        Args:
            agent_name: Identifier for this vision agent instance.
            model_name: Designated vision model name or descriptor.
            metadata: Optional configuration metadata dictionary.

        Raises:
            VisionInputValidationError: If agent_name or model_name is invalid.
        """
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise VisionInputValidationError("agent_name must be a non-empty string.")

        if not isinstance(model_name, str) or not model_name.strip():
            raise VisionInputValidationError("model_name must be a non-empty string.")

        self.agent_name = agent_name.strip()
        self.model_name = model_name.strip()
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
            VisionProcessingError: If visual model inference fails or is not implemented.
        """
        vision_req = self._prepare_request(request, evidence=evidence, metadata=kwargs.get("metadata"))

        # Day 32 contract: Architecture, validation, and contract layer.
        # Explicitly signals that model inference backend is not implemented for Day 32 foundation,
        # never returning fake or hallucinated production descriptions.
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
