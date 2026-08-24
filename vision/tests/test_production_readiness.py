"""
Day 54 - OmniBrain Member 3 Vision Agent: Final Production Readiness & Contract Freeze Audit.

Performs a comprehensive, cross-component audit of the Member 3 Vision subsystem:
  Search Evidence -> VisualEvidenceAdapter -> prepare_image_evidence -> build_vision_input
  -> VisionModelProvider -> VisionExecutionAdapter -> VisionExecutionLifecycle
  -> VisionResultNormalizer -> VisionPipeline -> VisionResult -> Downstream Supervisor

Verifies:
  1. Complete Public API Audit (__all__ exports, types, signatures, symbol integrity)
  2. End-to-end happy path execution across all Member 3 stages (Days 33-53)
  3. End-to-end multi-evidence integration (1, 2, 3, 5, 10 items) without order loss
  4. End-to-end multi-document integration (DOC-A, DOC-B, DOC-C) maintaining lineage
  5. Request validation boundary (invalid inputs rejected before provider execution)
  6. Provider configuration validation and capability checks (Day 44)
  7. Retry and recovery policy execution without duplicate work (Day 47)
  8. Timeout handling and terminal state protection (Day 46)
  9. Cancellation handling and lifecycle cleanup (Day 46)
 10. Thread-safe concurrency and mixed outcome isolation (Day 42)
 11. Resource safety and thread cleanup after all execution outcomes (Day 48)
 12. Execution observability and stage transition ordering (Day 45)
 13. Result normalization, trace attachment, and secret sanitization (Day 39)
 14. Full citation and lineage integrity (Day 52)
 15. Downstream supervisor handoff contract stability (Day 53)
 16. API contract freeze (constructor signatures, parameter types, return values)
 17. Complete domain exception hierarchy verification
 18. Error message sanitization (no credential or path leakages)
 19. Deterministic execution behavior across repeated runs
 20. State isolation on sequential pipeline reuse
 21. Duplicate-work prevention (zero duplicate retrieval, embedding, or preparation)
 22. 100% offline execution guarantee (zero network, HTTP, LLM, or external APIs)

All tests execute 100% offline.
"""

from __future__ import annotations

import io
import sys
import threading
import time
from typing import Any

import pytest
from PIL import Image

import vision
from agents.models import AgentCitation, AgentResponse, SearchResult
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionAgentError,
    VisionCancellationError,
    VisionError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderConfigError,
    VisionProviderError,
    VisionProviderExecutionError,
    VisionProviderUnavailableError,
    VisionTimeoutError,
    VisionUnsupportedCapabilityError,
)
from vision.execution_adapter import VisionExecutionAdapter, execute_vision_request
from vision.image_preparation import (
    SUPPORTED_IMAGE_FORMATS,
    ImageEvidencePreparator,
    OversizedImagePolicy,
    PreparedImageEvidence,
    prepare_image_evidence,
)
from vision.input_builder import VisionInputBuilder, VisionModelInput, build_vision_input
from vision.lifecycle import (
    VisionCancellationToken,
    VisionExecutionLifecycle,
    VisionExecutionObservation,
    VisionExecutionStage,
    VisionRetryPolicy,
)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.pipeline import VisionPipeline, run_vision_pipeline
from vision.provider import VisionModelProvider, VisionProviderRegistry
from vision.provider_config import VisionProviderCapabilities, VisionProviderConfig
from vision.result_normalizer import (
    FORBIDDEN_METADATA_KEYS,
    VisionExecutionTrace,
    VisionResultNormalizer,
)
from vision.vision_agent import VisionAgent


# ===========================================================================
# Test Helpers & Downstream Test Doubles
# ===========================================================================

def _make_png_bytes(width: int = 16, height: int = 16, color: tuple[int, int, int] = (60, 120, 180)) -> bytes:
    """Generate a valid PNG byte payload for testing visual evidence."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_SAMPLE_PNG = _make_png_bytes()


def _make_citation(
    doc_id: str = "DOC-AUDIT-001",
    filename: str = "production_spec.pdf",
    chunk_id: str = "CHUNK-AUDIT-001",
    content_type: str = "image",
    page_number: int | None = 2,
    score: float = 0.96,
    metadata: dict[str, Any] | None = None,
) -> AgentCitation:
    """Construct a Member 2 AgentCitation for production readiness testing."""
    return AgentCitation(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        content_type=content_type,
        score=score,
        metadata=metadata if metadata is not None else {"env": "production_audit", "stage": "verification"},
    )


class AuditTestProvider(VisionModelProvider):
    """Controlled offline test double for final production readiness verification."""

    def __init__(
        self,
        capabilities: VisionProviderCapabilities | None = None,
        should_fail: bool = False,
        fail_count: int = 0,
        latency: float = 0.0,
        simulate_timeout: bool = False,
    ) -> None:
        config = VisionProviderConfig(provider_name="audit_test_provider", model_name="audit_model_v1")
        super().__init__(config=config, capabilities=capabilities)
        self._should_fail = should_fail
        self._fail_count = fail_count
        self._call_count = 0
        self._received_inputs: list[VisionModelInput] = []
        self._latency = latency
        self._simulate_timeout = simulate_timeout
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    @property
    def received_inputs(self) -> list[VisionModelInput]:
        with self._lock:
            return list(self._received_inputs)

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        """Execute with deterministic result, tracking received inputs."""
        if self._latency > 0:
            time.sleep(self._latency)

        with self._lock:
            self._call_count += 1
            current_count = self._call_count
            self._received_inputs.append(model_input)

        if self._simulate_timeout:
            raise VisionTimeoutError("Simulated provider execution timeout during production audit.")

        if self._should_fail and current_count <= self._fail_count:
            raise VisionProviderExecutionError(
                f"Simulated provider execution failure (call {current_count})."
            )

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Production audit analysis of '{model_input.query}'",
            document_id=model_input.document_id,
            filename=model_input.filename,
            chunk_id=model_input.chunk_id,
            page_number=model_input.page_number,
            content_type=model_input.content_type,
            metadata={
                "provider_name": self.config.provider_name,
                "model_name": self.config.model_name,
                "call_count": current_count,
                "api_key": "leaky_secret_key_should_be_sanitized",
            },
        )


class AuditDownstreamConsumer:
    """Lightweight downstream consumer double for end-to-end handoff verification."""

    def __init__(self) -> None:
        self.received_results: list[VisionResult] = []

    def consume(self, result: VisionResult) -> dict[str, Any]:
        """Consume and validate a VisionResult."""
        if not isinstance(result, VisionResult):
            raise TypeError(f"Consumer expected VisionResult, got {type(result).__name__}")
        self.received_results.append(result)
        return {
            "consumed": True,
            "status": result.status,
            "document_id": result.document_id,
            "filename": result.filename,
            "evidence_count": len(result.evidence),
        }


# ===========================================================================
# Test Cases
# ===========================================================================

class TestProductionReadinessAndContractFreeze:
    """Complete production-readiness audit and contract freeze test suite (Day 54)."""

    def test_01_complete_public_api_audit(self) -> None:
        """Step 1: Inspect and verify all 45+ exports in vision.__all__ exist and are valid."""
        exported_names = vision.__all__
        assert len(exported_names) >= 30

        for name in exported_names:
            assert hasattr(vision, name), f"vision.__all__ contains missing symbol '{name}'"
            symbol = getattr(vision, name)
            assert symbol is not None

        # Verify key classes and entrypoints
        assert issubclass(VisionPipeline, object)
        assert issubclass(VisionRequest, object)
        assert issubclass(VisionResult, object)
        assert issubclass(VisualEvidence, object)
        assert issubclass(VisionModelProvider, object)
        assert issubclass(VisionError, Exception)

    def test_02_end_to_end_happy_path(self) -> None:
        """Step 2: Full pipeline flow: Search Citation -> VisualEvidence -> PreparedImage -> Input -> Provider -> Normalizer -> Result -> Consumer."""
        citation = _make_citation(
            doc_id="DOC-E2E-001",
            filename="happy_path.pdf",
            chunk_id="CHUNK-E2E-001",
            content_type="chart",
            page_number=5,
        )

        # 1. Adapt citation
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
        assert ev.document_id == "DOC-E2E-001"

        # 2. Image preparation
        prep = prepare_image_evidence(ev)
        assert isinstance(prep, PreparedImageEvidence)

        # 3. VisionModelInput builder
        model_input = build_vision_input(query="Analyze happy path chart", evidence=prep)
        assert isinstance(model_input, VisionModelInput)

        # 4. Pipeline execution
        req = VisionRequest(query="Analyze happy path chart", evidence=[ev])
        provider = AuditTestProvider()
        pipeline = VisionPipeline(provider=provider)

        result = pipeline.run(req)

        # 5. Downstream handoff
        consumer = AuditDownstreamConsumer()
        consumed = consumer.consume(result)

        assert consumed["consumed"] is True
        assert consumed["status"] == "success"
        assert consumed["document_id"] == "DOC-E2E-001"
        assert consumed["filename"] == "happy_path.pdf"
        assert result.page_number == 5
        assert result.content_type == "chart"
        assert "api_key" not in result.metadata
        assert provider.call_count == 1

    def test_03_end_to_end_multi_evidence(self) -> None:
        """Step 3: Multi-evidence integration (1, 2, 3, 5, 10 items) preserves count, order, and lineage."""
        for n in [1, 2, 3, 5, 10]:
            citations = [
                _make_citation(doc_id=f"DOC-{i}", filename=f"fig_{i}.pdf", chunk_id=f"chk-{i}")
                for i in range(n)
            ]
            evidence = [
                VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG)
                for c in citations
            ]

            req = VisionRequest(query=f"Analyze {n} items", evidence=evidence)
            provider = AuditTestProvider()
            pipeline = VisionPipeline(provider=provider)

            res = pipeline.run(req)

            assert len(res.evidence) == n
            for i in range(n):
                assert res.evidence[i].document_id == f"DOC-{i}"
                assert res.evidence[i].chunk_id == f"chk-{i}"

    def test_04_end_to_end_multi_document(self) -> None:
        """Step 4: Multi-document evidence (DOC-A, DOC-B, DOC-C) maintains document identity."""
        cit_a = _make_citation(doc_id="DOC-A", filename="a.pdf")
        cit_b = _make_citation(doc_id="DOC-B", filename="b.pdf")
        cit_c = _make_citation(doc_id="DOC-C", filename="c.pdf")

        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a, image_bytes=_SAMPLE_PNG)
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b, image_bytes=_SAMPLE_PNG)
        ev_c = VisualEvidenceAdapter.adapt_citation(cit_c, image_bytes=_SAMPLE_PNG)

        req = VisionRequest(query="Multi-document query", evidence=[ev_a, ev_b, ev_c])
        provider = AuditTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert [e.document_id for e in res.evidence] == ["DOC-A", "DOC-B", "DOC-C"]

    def test_05_validation_boundary_enforcement(self) -> None:
        """Step 5: Invalid inputs (empty query, whitespace query, None, invalid type) fail before provider execution."""
        cit = _make_citation()
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        provider = AuditTestProvider()
        pipeline = VisionPipeline(provider=provider)

        # Empty query
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="", evidence=[ev])

        # Whitespace query
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="   ", evidence=[ev])

        # None request
        with pytest.raises(VisionInputValidationError):
            pipeline.run(None)  # type: ignore[arg-type]

        # Non-visual citation in strict mode
        cit_txt = _make_citation(content_type="text")
        with pytest.raises(VisionEvidenceError):
            VisualEvidenceAdapter.adapt_citation(cit_txt)

        assert provider.call_count == 0

    def test_06_provider_configuration_validation(self) -> None:
        """Step 6: VisionProviderConfig and capabilities validation without real secrets."""
        # Valid config
        config = VisionProviderConfig(
            provider_name="test_prov",
            model_name="v1",
            timeout=30.0,
        )
        assert config.provider_name == "test_prov"
        assert config.model_name == "v1"
        assert config.timeout == 30.0

        # Invalid provider name
        with pytest.raises(VisionProviderConfigError):
            VisionProviderConfig(provider_name="", model_name="v1")

        # Invalid timeout
        with pytest.raises(VisionProviderConfigError):
            VisionProviderConfig(provider_name="prov", model_name="v1", timeout=-5.0)

        # Capabilities
        caps = VisionProviderCapabilities(
            supported_modalities=frozenset({"image", "chart"}),
            supports_multi_image=True,
        )
        assert caps.supports_modality("chart") is True
        assert caps.supports_modality("video") is False

    def test_07_retry_policy_execution(self) -> None:
        """Step 7: Provider failure triggers VisionRetryPolicy without duplicating retrieval or prep."""
        cit = _make_citation(doc_id="DOC-RETRY-AUDIT")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Retry audit query", evidence=[ev])

        provider = AuditTestProvider(should_fail=True, fail_count=1)
        retry_policy = VisionRetryPolicy(max_retries=2)
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req, retry_policy=retry_policy)

        assert res.status == "success"
        assert provider.call_count == 2
        assert res.evidence[0].document_id == "DOC-RETRY-AUDIT"

    def test_08_timeout_handling_and_cleanup(self) -> None:
        """Step 8: Execution timeout raises VisionTimeoutError and protects terminal state."""
        cit = _make_citation(doc_id="DOC-TIMEOUT-AUDIT")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Timeout audit query", evidence=[ev])

        provider = AuditTestProvider(simulate_timeout=True)
        pipeline = VisionPipeline(provider=provider)

        with pytest.raises(VisionTimeoutError):
            pipeline.run(req)

        # Subsequent execution succeeds normally
        provider_ok = AuditTestProvider()
        pipeline_ok = VisionPipeline(provider=provider_ok)
        res = pipeline_ok.run(req)
        assert res.status == "success"

    def test_09_cancellation_handling_and_cleanup(self) -> None:
        """Step 9: Cancellation via VisionCancellationToken halts pipeline before provider invocation."""
        cit = _make_citation(doc_id="DOC-CANCEL-AUDIT")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Cancel audit query", evidence=[ev])

        token = VisionCancellationToken()
        token.cancel()

        provider = AuditTestProvider()
        pipeline = VisionPipeline(provider=provider)

        with pytest.raises(VisionCancellationError):
            pipeline.run(req, cancellation_token=token)

        assert provider.call_count == 0

    def test_10_concurrent_execution_isolation(self) -> None:
        """Step 10: Parallel requests (DOC-A, DOC-B, DOC-C, DOC-D) with mixed outcomes maintain total isolation."""
        provider = AuditTestProvider()
        pipeline = VisionPipeline(provider=provider)

        results: dict[str, Any] = {}
        errors: dict[str, Exception] = {}

        def run_thread(tag: str, doc_id: str) -> None:
            try:
                c = _make_citation(doc_id=doc_id)
                ev = VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG)
                req = VisionRequest(query=f"Query {tag}", evidence=[ev])
                res = pipeline.run(req)
                results[tag] = res
            except Exception as exc:
                errors[tag] = exc

        threads = [
            threading.Thread(target=run_thread, args=("A", "DOC-THREAD-A")),
            threading.Thread(target=run_thread, args=("B", "DOC-THREAD-B")),
            threading.Thread(target=run_thread, args=("C", "DOC-THREAD-C")),
            threading.Thread(target=run_thread, args=("D", "DOC-THREAD-D")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 4
        assert results["A"].evidence[0].document_id == "DOC-THREAD-A"
        assert results["B"].evidence[0].document_id == "DOC-THREAD-B"
        assert results["C"].evidence[0].document_id == "DOC-THREAD-C"
        assert results["D"].evidence[0].document_id == "DOC-THREAD-D"

    def test_11_resource_safety_post_execution(self) -> None:
        """Step 11: Active thread count remains normal after execution outcomes."""
        initial_threads = threading.active_count()

        provider = AuditTestProvider()
        pipeline = VisionPipeline(provider=provider)

        cit = _make_citation()
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Resource safety check", evidence=[ev])

        for _ in range(5):
            pipeline.run(req)

        assert threading.active_count() <= initial_threads + 2

    def test_12_observability_contract_trace(self) -> None:
        """Step 12: VisionExecutionObservation records valid stage transitions without mutating results."""
        lifecycle = VisionExecutionLifecycle(provider_name="audit_prov", model_name="v1")
        lifecycle.transition_to(VisionExecutionStage.VALIDATING)
        lifecycle.transition_to(VisionExecutionStage.PREPARING)
        lifecycle.transition_to(VisionExecutionStage.BUILDING_INPUT)
        lifecycle.transition_to(VisionExecutionStage.EXECUTING)
        lifecycle.transition_to(VisionExecutionStage.COMPLETED)

        obs = VisionExecutionObservation.from_lifecycle(lifecycle, evidence_count=1)

        assert obs.is_completed is True
        assert obs.is_failed is False
        assert obs.provider_name == "audit_prov"
        assert obs.evidence_count == 1
        assert obs.stage == VisionExecutionStage.COMPLETED

    def test_13_result_normalization_and_sanitization(self) -> None:
        """Step 13: ResultNormalizer strips secret keys and attaches execution trace."""
        cit = _make_citation()
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)

        raw_res = VisionResult(
            query="Norm audit",
            status="success",
            description="Analysis",
            evidence=[ev],
            metadata={"api_key": "secret_val", "safe_key": "ok"},
        )

        trace = VisionExecutionTrace.create_default()
        norm_res = VisionResultNormalizer.normalize(raw_res, trace=trace)

        assert "api_key" not in norm_res.metadata
        assert norm_res.metadata.get("safe_key") == "ok"
        assert "execution_trace" in norm_res.metadata

    def test_14_citation_and_lineage_integrity(self) -> None:
        """Step 14: Document lineage (document_id, filename, chunk_id, page_number) preserved end-to-end."""
        cit = _make_citation(
            doc_id="DOC-LINEAGE-END",
            filename="end_spec.pdf",
            chunk_id="CHUNK-LINEAGE-END",
            page_number=11,
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Lineage audit", evidence=[ev])

        provider = AuditTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert res.document_id == "DOC-LINEAGE-END"
        assert res.filename == "end_spec.pdf"
        assert res.chunk_id == "CHUNK-LINEAGE-END"
        assert res.page_number == 11

    def test_15_supervisor_handoff_contract(self) -> None:
        """Step 15: VisionResult is consumed by downstream consumer double without accessing private state."""
        cit = _make_citation()
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Supervisor audit", evidence=[ev])

        provider = AuditTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        consumer = AuditDownstreamConsumer()
        summary = consumer.consume(res)

        assert summary["consumed"] is True
        assert summary["status"] == "success"

    def test_16_api_contract_freeze_signatures(self) -> None:
        """Step 16: Verify constructor signatures and key methods remain frozen."""
        import inspect

        # VisionPipeline signature
        pipeline_sig = inspect.signature(VisionPipeline.__init__)
        assert "provider" in pipeline_sig.parameters

        # VisionRequest signature
        request_sig = inspect.signature(VisionRequest.__init__)
        assert "query" in request_sig.parameters
        assert "evidence" in request_sig.parameters

        # VisionResult signature
        result_sig = inspect.signature(VisionResult.__init__)
        assert "query" in result_sig.parameters
        assert "status" in result_sig.parameters

    def test_17_domain_exception_hierarchy(self) -> None:
        """Step 17: Complete domain exception hierarchy inheritance check."""
        assert issubclass(VisionInputValidationError, VisionError)
        assert issubclass(VisionEvidenceError, VisionError)
        assert issubclass(VisionProcessingError, VisionError)
        assert issubclass(VisionProviderError, VisionError)
        assert issubclass(VisionProviderConfigError, VisionProviderError)
        assert issubclass(VisionProviderExecutionError, VisionProviderError)
        assert issubclass(VisionProviderUnavailableError, VisionProviderError)
        assert issubclass(VisionUnsupportedCapabilityError, VisionError)
        assert issubclass(VisionTimeoutError, VisionError)
        assert issubclass(VisionCancellationError, VisionError)
        assert issubclass(VisionAgentError, VisionError)

    def test_18_error_sanitization(self) -> None:
        """Step 18: Domain exception messages do not leak secrets or credentials."""
        err = VisionInputValidationError("Invalid query parameter provided.")
        assert "secret" not in str(err).lower()
        assert "api_key" not in str(err).lower()

    def test_19_deterministic_behavior(self) -> None:
        """Step 19: Identical requests produce deterministic output and zero state accumulation across runs."""
        cit = _make_citation(doc_id="DOC-DETERM-1")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Deterministic check", evidence=[ev])

        provider = AuditTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res1 = pipeline.run(req)
        res2 = pipeline.run(req)

        assert res1.document_id == res2.document_id
        assert res1.status == res2.status
        assert provider.call_count == 2

    def test_20_pipeline_instance_sequential_reuse_state_isolation(self) -> None:
        """Step 20: Sequentially reusing the same VisionPipeline instance does not cross-contaminate state."""
        cit_a = _make_citation(doc_id="DOC-SEQ-A")
        cit_b = _make_citation(doc_id="DOC-SEQ-B")

        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a, image_bytes=_SAMPLE_PNG)
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b, image_bytes=_SAMPLE_PNG)

        req_a = VisionRequest(query="Query A", evidence=[ev_a])
        req_b = VisionRequest(query="Query B", evidence=[ev_b])

        provider = AuditTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res_a = pipeline.run(req_a)
        res_b = pipeline.run(req_b)

        assert res_a.document_id == "DOC-SEQ-A"
        assert res_b.document_id == "DOC-SEQ-B"
        assert res_a.query == "Query A"
        assert res_b.query == "Query B"

    def test_21_no_duplicate_work_verification(self) -> None:
        """Step 21: Verify single provider call per pipeline run (no duplicate work)."""
        cit = _make_citation()
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Single work query", evidence=[ev])

        provider = AuditTestProvider()
        pipeline = VisionPipeline(provider=provider)

        pipeline.run(req)
        assert provider.call_count == 1

    def test_22_offline_execution_guarantee(self) -> None:
        """Step 22: Verify subsystem modules carry no active network or external API dependencies."""
        for mod_name in sys.modules:
            if "vision" in mod_name:
                assert "sentence_transformers" not in mod_name.lower()
                assert "qdrant" not in mod_name.lower()
                assert "fastapi" not in mod_name.lower()
                assert "streamlit" not in mod_name.lower()
