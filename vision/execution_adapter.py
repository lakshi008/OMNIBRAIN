"""
Vision Execution Adapter for OmniBrain Member 3 Vision Agent.

Orchestrates the multi-stage pipeline connecting incoming VisionRequest inputs,
retrieval evidence adaptation (Day 33), image preparation (Day 34),
input building (Day 35), provider execution (Day 36), execution lifecycle (Day 38),
and result normalization & execution trace layer (Day 39) into a unified workflow.

Execution Pipeline:
    VisionRequest
          ↓
    Validating (VisionExecutionStage.VALIDATING)
          ↓
    Evidence Adaptation (VisualEvidenceAdapter)
          ↓
    Preparing (VisionExecutionStage.PREPARING)
          ↓
    Building Input (VisionExecutionStage.BUILDING_INPUT)
          ↓
    Executing (VisionExecutionStage.EXECUTING) -> Immutability & Timeout Boundary
          ↓
    Result Normalization & Trace (VisionResultNormalizer)
          ↓
    Completed / Failed / Timeout (VisionExecutionStage.COMPLETED)
          ↓
    Standardized VisionResult
"""

from __future__ import annotations

from typing import Any

from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionAgentError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderError,
    VisionProviderExecutionError,
    VisionTimeoutError,
)
from vision.image_preparation import PreparedImageEvidence, prepare_image_evidence
from vision.input_builder import VisionModelInput, build_vision_input
from vision.lifecycle import VisionExecutionLifecycle, VisionExecutionStage
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.provider import VisionModelProvider
from vision.result_normalizer import (
    VisionExecutionTrace,
    VisionResultNormalizer,
)


class VisionExecutionAdapter:
    """Orchestrator between VisionAgent requests and VisionModelProvider backends.

    Encapsulates the complete transformation, validation, lifecycle hardening, and normalization:
    1. Input request normalization and stage validation.
    2. Evidence normalization and adaptation via VisualEvidenceAdapter.
    3. Image inspection, format validation, and preparation via Day 34 pipeline.
    4. VisionModelInput construction and lineage locking via Day 35 pipeline.
    5. Input immutability snapshot verification (before and after execution).
    6. Delegated single-invocation execution to an injected VisionModelProvider via Day 36 contract.
    7. Execution lifecycle tracking (pending -> validating -> preparing -> building_input -> executing -> completed/failed/timeout).
    8. Production-safe result normalization, metadata sanitization, and execution trace recording via Day 39 pipeline.
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

    @staticmethod
    def _create_input_snapshot(model_input: VisionModelInput) -> dict[str, Any]:
        """Create a field snapshot of VisionModelInput to verify post-execution immutability."""
        return {
            "query": model_input.query,
            "document_id": model_input.document_id,
            "filename": model_input.filename,
            "page_number": model_input.page_number,
            "chunk_id": model_input.chunk_id,
            "chunk_index": model_input.chunk_index,
            "content_type": model_input.content_type,
            "image_format": model_input.image_format,
            "width": model_input.width,
            "height": model_input.height,
            "mode": model_input.mode,
            "size_bytes": model_input.size_bytes,
            "is_oversized": model_input.is_oversized,
            "evidence_metadata": dict(model_input.evidence_metadata),
            "builder_metadata": dict(model_input.builder_metadata),
        }

    def execute(
        self,
        request: str | VisionRequest,
        evidence: list[Any] | None = None,
        **kwargs: Any,
    ) -> VisionResult:
        """Execute the multi-stage vision reasoning pipeline with lifecycle & trace hardening.

        Args:
            request: Query string or structured VisionRequest.
            evidence: Optional list of VisualEvidence, citations, search results, or chunks.
            **kwargs: Runtime arguments (e.g. builder_metadata, extra provider parameters).

        Returns:
            Structured, normalized VisionResult containing provider output and source lineage.

        Raises:
            VisionInputValidationError: If request format or parameters are invalid.
            VisionEvidenceError: If visual evidence lineage or preparation fails.
            VisionProviderExecutionError: If provider execution fails.
            VisionTimeoutError: If execution exceeds provider timeout limits.
            VisionProcessingError: If result normalization or immutability contract fails.
        """
        lifecycle = VisionExecutionLifecycle(
            provider_name=self._provider.provider_name,
            model_name=self._provider.model_name,
        )
        trace = VisionExecutionTrace()
        trace.add_stage("request_received")

        try:
            # Stage 1: VALIDATING — Normalize and validate request and evidence
            lifecycle.transition_to(VisionExecutionStage.VALIDATING)
            trace.add_stage("validation_started")
            vision_req = self._normalize_request(
                request, evidence=evidence, metadata=kwargs.get("metadata")
            )

            # Stage 2: Handle no-evidence requests gracefully
            if not vision_req.has_evidence:
                lifecycle.transition_to(
                    VisionExecutionStage.COMPLETED,
                    metadata={"notice": "no_evidence_supplied"},
                )
                trace.add_stage("execution_completed")
                res_meta = VisionResultNormalizer.sanitize_metadata(vision_req.metadata)
                res_meta["execution_lifecycle"] = lifecycle.to_dict()
                res_meta["execution_trace"] = trace.to_dict()
                return VisionResult(
                    query=vision_req.query,
                    status="no_evidence",
                    description="",
                    evidence=[],
                    metadata=res_meta,
                )

            # Stage 3: PREPARING — Prepare visual evidence via Day 34 pipeline
            lifecycle.transition_to(VisionExecutionStage.PREPARING)
            prepared_evidences = [prepare_image_evidence(ev) for ev in vision_req.evidence]
            primary_prepared = prepared_evidences[0]

            # Stage 4: BUILDING_INPUT — Construct validated, lineage-locked VisionModelInput via Day 35 pipeline
            lifecycle.transition_to(VisionExecutionStage.BUILDING_INPUT)
            trace.add_stage("input_prepared")
            b_meta = dict(kwargs.get("builder_metadata") or {})
            b_meta["total_evidence_count"] = len(prepared_evidences)
            b_meta["all_evidence_lineage"] = [e.to_dict() for e in prepared_evidences]
            model_input = build_vision_input(
                query=vision_req.query,
                evidence=primary_prepared,
                builder_metadata=b_meta,
            )

            # Stage 5: EXECUTING — Verify immutability & execute single provider invocation
            lifecycle.transition_to(VisionExecutionStage.EXECUTING)
            trace.add_stage("provider_started")
            before_snapshot = self._create_input_snapshot(model_input)

            raw_result = self._provider.execute(model_input, **kwargs)
            trace.add_stage("provider_completed")

            after_snapshot = self._create_input_snapshot(model_input)
            if before_snapshot != after_snapshot:
                raise VisionProcessingError(
                    "VisionModelInput immutability violated during provider execution."
                )

            # Stage 6: RESULT NORMALIZATION — Validate, reconcile lineage, and sanitize metadata via Day 39 normalizer
            normalized_result = VisionResultNormalizer.normalize(
                result=raw_result,
                request=vision_req,
                model_input=model_input,
                trace=trace,
            )

            # Stage 7: COMPLETED — Mark lifecycle complete and attach final lifecycle state
            lifecycle.transition_to(VisionExecutionStage.COMPLETED)
            res_meta = dict(normalized_result.metadata)
            res_meta["execution_lifecycle"] = lifecycle.to_dict()
            normalized_result.metadata = res_meta

            return normalized_result

        except VisionTimeoutError as err:
            lifecycle.transition_to(VisionExecutionStage.TIMEOUT, error=str(err))
            raise VisionTimeoutError(f"Vision provider execution timed out: {err}") from err

        except (VisionInputValidationError, VisionEvidenceError, VisionProviderError, VisionProcessingError) as err:
            if not lifecycle.is_terminal:
                lifecycle.transition_to(VisionExecutionStage.FAILED, error=str(err))
            raise

        except Exception as err:
            if not lifecycle.is_terminal:
                lifecycle.transition_to(VisionExecutionStage.FAILED, error=str(err))
            raise VisionProviderExecutionError(
                f"Unexpected failure during provider execution lifecycle: {err}"
            ) from err


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
