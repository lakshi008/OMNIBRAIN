"""
Vision End-to-End Pipeline Contract for OmniBrain Member 3 Vision Agent Subsystem.

Formalizes the complete multi-stage integration contract connecting:
  1. Member 1 & Member 2 retrieval citations/search results (Search/Citation Evidence)
  2. VisualEvidenceAdapter (Day 33 evidence normalization & lineage preservation)
  3. ImageEvidencePreparator (Day 34 image validation & format inspection)
  4. VisionInputBuilder (Day 35 VisionModelInput construction & lineage locking)
  5. VisionModelProvider (Day 36 vendor-agnostic provider execution boundary)
  6. VisionExecutionLifecycle (Day 38 state tracking & stage hardening)
  7. VisionResultNormalizer (Day 39 output validation, trace, & metadata sanitization)
  8. VisionResult (Standardized output contract)

Day 40 Scope:
  - VisionPipeline: End-to-end integration contract orchestrator.
  - run_vision_pipeline: High-level convenience entry point.
  - Pure integration contract -- zero network, zero LLM, zero fake Vision results in production.
"""

from __future__ import annotations

from typing import Any

from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionInputValidationError,
    VisionProcessingError,
)
from vision.execution_adapter import VisionExecutionAdapter
from vision.models import VisionRequest, VisionResult
from vision.provider import VisionModelProvider


class VisionPipeline:
    """End-to-End Vision Pipeline Orchestrator and Contract Validator.

    Provides a clean, unified integration boundary linking all Member 3 Vision
    subsystem components (Days 32–39) through one production-safe interface.
    """

    def __init__(self, provider: VisionModelProvider) -> None:
        """Initialize VisionPipeline with an injected VisionModelProvider backend.

        Args:
            provider: Validated VisionModelProvider instance.

        Raises:
            VisionInputValidationError: If provider is None or invalid type.
        """
        if provider is None:
            raise VisionInputValidationError("provider cannot be None.")

        if not isinstance(provider, VisionModelProvider):
            raise VisionInputValidationError(
                f"provider must be a VisionModelProvider instance, got {type(provider).__name__}."
            )

        self._adapter = VisionExecutionAdapter(provider=provider)

    @property
    def provider(self) -> VisionModelProvider:
        """Return the underlying VisionModelProvider."""
        return self._adapter.provider

    @property
    def execution_adapter(self) -> VisionExecutionAdapter:
        """Return the underlying VisionExecutionAdapter."""
        return self._adapter

    def run(
        self,
        request: str | VisionRequest,
        evidence: list[Any] | None = None,
        **kwargs: Any,
    ) -> VisionResult:
        """Execute the complete end-to-end visual reasoning pipeline.

        Args:
            request: Query string or structured VisionRequest.
            evidence: Optional list of visual evidence, citations, search results, or chunks.
            **kwargs: Additional runtime parameters (e.g. builder_metadata).

        Returns:
            Standardized VisionResult maintaining complete document lineage and execution trace.

        Raises:
            VisionInputValidationError: If request or evidence fails validation.
            VisionEvidenceError: If evidence adaptation or preparation fails.
            VisionProviderExecutionError: If provider execution fails.
            VisionProcessingError: If result normalization fails.
        """
        return self._adapter.execute(request, evidence=evidence, **kwargs)


def run_vision_pipeline(
    provider: VisionModelProvider,
    request: str | VisionRequest,
    evidence: list[Any] | None = None,
    **kwargs: Any,
) -> VisionResult:
    """Convenience entry point for running the end-to-end Vision pipeline.

    Args:
        provider: Injected VisionModelProvider instance.
        request: Query string or VisionRequest.
        evidence: Optional list of evidence items.
        **kwargs: Runtime execution parameters.

    Returns:
        Standardized VisionResult contract.
    """
    pipeline = VisionPipeline(provider=provider)
    return pipeline.run(request, evidence=evidence, **kwargs)
