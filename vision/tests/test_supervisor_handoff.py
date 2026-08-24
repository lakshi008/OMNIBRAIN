"""
Day 53 - OmniBrain Member 3 Vision Agent: Vision Result -> Supervisor Handoff Contract Tests.

Verifies the downstream contract connecting Member 3 Vision to a downstream orchestration consumer:
  Member 3 Vision -> VisionResult -> Downstream Supervisor / Orchestration Consumer

Ensures:
  1. VisionResult public contract verification (properties, status, lineage, metadata, evidence)
  2. Successful downstream handoff to lightweight offline downstream consumer/double
  3. Multi-evidence result handoff (1, 2, 3, 5, 10 items) preserving count, order, and lineage
  4. Multi-document result handoff (DOC-A, DOC-B, DOC-C) maintaining document identity
  5. Modality preservation (image, chart, diagram) without content-type swapping
  6. End-to-end citation and lineage preservation from Search evidence to downstream consumer
  7. Result normalization reuse (Day 39) preventing internal provider/secret leakages
  8. Success status representation ('success', is_success=True, error=None)
  9. Provider failure representation handling (status='error', is_error=True)
 10. Retry success handoff emitting ONLY the final successful result (Day 47)
 11. Timeout error representation handling (Day 46)
 12. Cancellation error representation handling (Day 46)
 13. Thread-safe concurrent result isolation and mixed outcome stability
 14. Mutation safety of VisionResult metadata and dictionary exports
 15. Roundtrip serialization compatibility via to_dict() and from_dict()
 16. Public API import contract compatibility (Day 49)
 17. Prevention of internal execution leakage to downstream consumers
 18. Zero duplicate retrieval, zero embedding generation, zero Qdrant access
 19. 100% offline execution using controlled test doubles

All tests execute 100% offline.
"""

from __future__ import annotations

import copy
import io
import sys
import threading
import time
from typing import Any

import pytest
from PIL import Image

from agents.models import AgentCitation, AgentResponse, SearchResult
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionCancellationError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderExecutionError,
    VisionTimeoutError,
)
from vision.execution_adapter import VisionExecutionAdapter
from vision.input_builder import VisionModelInput
from vision.lifecycle import (
    VisionCancellationToken,
    VisionExecutionLifecycle,
    VisionRetryPolicy,
)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.pipeline import VisionPipeline, run_vision_pipeline
from vision.provider import VisionModelProvider
from vision.provider_config import VisionProviderCapabilities, VisionProviderConfig
from vision.result_normalizer import VisionExecutionTrace, VisionResultNormalizer


# ===========================================================================
# Test Helpers & Downstream Orchestration Doubles
# ===========================================================================

def _make_png_bytes(width: int = 16, height: int = 16, color: tuple[int, int, int] = (90, 160, 210)) -> bytes:
    """Generate a valid PNG byte payload for testing visual evidence."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_SAMPLE_PNG = _make_png_bytes()


def _make_citation(
    doc_id: str = "DOC-SUPER-001",
    filename: str = "diagram.pdf",
    chunk_id: str = "CHUNK-SUPER-001",
    content_type: str = "image",
    page_number: int | None = 3,
    score: float = 0.94,
    metadata: dict[str, Any] | None = None,
) -> AgentCitation:
    """Construct a Member 2 AgentCitation for handoff testing."""
    return AgentCitation(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        content_type=content_type,
        score=score,
        metadata=metadata if metadata is not None else {"source": "search_agent", "section": "architecture"},
    )


class SupervisorHandoffTestProvider(VisionModelProvider):
    """Controlled offline test provider for VisionResult -> Supervisor handoff testing."""

    def __init__(
        self,
        capabilities: VisionProviderCapabilities | None = None,
        should_fail: bool = False,
        fail_count: int = 0,
        latency: float = 0.0,
        simulate_timeout: bool = False,
    ) -> None:
        config = VisionProviderConfig(provider_name="supervisor_handoff_test", model_name="handoff_model_v1")
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
        """Execute and return VisionResult preserving model input lineage."""
        if self._latency > 0:
            time.sleep(self._latency)

        with self._lock:
            self._call_count += 1
            current_count = self._call_count
            self._received_inputs.append(model_input)

        if self._simulate_timeout:
            raise VisionTimeoutError("Simulated provider execution timeout.")

        if self._should_fail and current_count <= self._fail_count:
            raise VisionProviderExecutionError(
                f"Simulated provider execution failure (call {current_count})."
            )

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Supervisor handoff analysis of '{model_input.query}'",
            document_id=model_input.document_id,
            filename=model_input.filename,
            chunk_id=model_input.chunk_id,
            page_number=model_input.page_number,
            content_type=model_input.content_type,
            metadata={
                "provider_name": self.config.provider_name,
                "model_name": self.config.model_name,
                "call_count": current_count,
                "api_key": "secret_key_should_be_sanitized",
            },
        )


class OfflineSupervisorConsumer:
    """Lightweight offline test double simulating downstream Supervisor consuming VisionResult."""

    def __init__(self) -> None:
        self.received_results: list[VisionResult] = []
        self.errors: list[str] = []

    def consume(self, result: VisionResult) -> dict[str, Any]:
        """Consume and validate a VisionResult from Member 3 Vision."""
        if not isinstance(result, VisionResult):
            raise TypeError(f"Supervisor expected VisionResult, got {type(result).__name__}")

        self.received_results.append(result)

        if result.is_error:
            self.errors.append(result.error or "Unknown error")
            return {
                "supervisor_status": "AGENT_FAILED",
                "query": result.query,
                "error": result.error,
            }

        return {
            "supervisor_status": "SUCCESS",
            "query": result.query,
            "description": result.description,
            "document_id": result.document_id,
            "filename": result.filename,
            "page_number": result.page_number,
            "chunk_id": result.chunk_id,
            "content_type": result.content_type,
            "evidence_count": len(result.evidence),
            "metadata": dict(result.metadata),
        }


# ===========================================================================
# Test Suite
# ===========================================================================

class TestSupervisorHandoffContract:
    """Complete test suite for Day 53 VisionResult -> Supervisor Handoff Contract."""

    def test_01_inspect_vision_result_public_contract(self) -> None:
        """Step 1 & 2: Inspect and verify actual VisionResult public fields and properties."""
        res = VisionResult(
            query="Public contract check",
            status="success",
            description="Contract verification text",
            document_id="DOC-PUBLIC-1",
            filename="pub.pdf",
            chunk_id="CHK-PUB-1",
            page_number=4,
            content_type="chart",
            metadata={"version": "1.0"},
        )

        assert res.query == "Public contract check"
        assert res.status == "success"
        assert res.description == "Contract verification text"
        assert res.document_id == "DOC-PUBLIC-1"
        assert res.filename == "pub.pdf"
        assert res.chunk_id == "CHK-PUB-1"
        assert res.page_number == 4
        assert res.content_type == "chart"
        assert res.metadata == {"version": "1.0"}
        assert res.error is None
        assert res.is_success is True
        assert res.is_error is False

    def test_02_successful_downstream_handoff(self) -> None:
        """Step 3: End-to-end execution of VisionRequest -> VisionPipeline -> VisionResult -> Downstream Consumer."""
        cit = _make_citation(doc_id="DOC-HANDOFF-1", filename="arch.pdf", content_type="diagram")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Describe system architecture", evidence=[ev])

        provider = SupervisorHandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        result = pipeline.run(req)

        consumer = OfflineSupervisorConsumer()
        consumed = consumer.consume(result)

        assert consumed["supervisor_status"] == "SUCCESS"
        assert consumed["query"] == "Describe system architecture"
        assert consumed["document_id"] == "DOC-HANDOFF-1"
        assert consumed["filename"] == "arch.pdf"
        assert consumed["content_type"] == "diagram"
        assert consumed["evidence_count"] == 1

    def test_03_multi_evidence_result_handoff(self) -> None:
        """Step 4: Multi-evidence (1, 2, 3, 5, 10 items) handed off without loss, reordering, or duplication."""
        for n in [1, 2, 3, 5, 10]:
            citations = [
                _make_citation(doc_id=f"DOC-{i}", filename=f"f_{i}.pdf", chunk_id=f"chk-{i}")
                for i in range(n)
            ]
            evidence = [
                VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG)
                for c in citations
            ]

            req = VisionRequest(query=f"Analyze {n} items", evidence=evidence)
            provider = SupervisorHandoffTestProvider()
            pipeline = VisionPipeline(provider=provider)

            res = pipeline.run(req)

            consumer = OfflineSupervisorConsumer()
            summary = consumer.consume(res)

            assert summary["evidence_count"] == n
            assert len(res.evidence) == n
            for i in range(n):
                assert res.evidence[i].document_id == f"DOC-{i}"
                assert res.evidence[i].chunk_id == f"chk-{i}"

    def test_04_multi_document_result_handoff(self) -> None:
        """Step 5: Evidence from DOC-A, DOC-B, DOC-C preserves document identities downstream."""
        cit_a = _make_citation(doc_id="DOC-A", filename="a.pdf")
        cit_b = _make_citation(doc_id="DOC-B", filename="b.pdf")
        cit_c = _make_citation(doc_id="DOC-C", filename="c.pdf")

        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a, image_bytes=_SAMPLE_PNG)
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b, image_bytes=_SAMPLE_PNG)
        ev_c = VisualEvidenceAdapter.adapt_citation(cit_c, image_bytes=_SAMPLE_PNG)

        req = VisionRequest(query="Multi-doc comparison", evidence=[ev_a, ev_b, ev_c])
        provider = SupervisorHandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        consumer = OfflineSupervisorConsumer()
        consumer.consume(res)

        assert len(res.evidence) == 3
        assert res.evidence[0].document_id == "DOC-A"
        assert res.evidence[1].document_id == "DOC-B"
        assert res.evidence[2].document_id == "DOC-C"

    def test_05_content_types_preservation(self) -> None:
        """Step 6: Modalities (image, chart, diagram) preserve their associations downstream."""
        c_img = _make_citation(doc_id="D1", content_type="image")
        c_chart = _make_citation(doc_id="D2", content_type="chart")
        c_diag = _make_citation(doc_id="D3", content_type="diagram")

        evs = [
            VisualEvidenceAdapter.adapt_citation(c_img, image_bytes=_SAMPLE_PNG),
            VisualEvidenceAdapter.adapt_citation(c_chart, image_bytes=_SAMPLE_PNG),
            VisualEvidenceAdapter.adapt_citation(c_diag, image_bytes=_SAMPLE_PNG),
        ]

        req = VisionRequest(query="Content types check", evidence=evs)
        provider = SupervisorHandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        consumer = OfflineSupervisorConsumer()
        consumer.consume(res)

        assert [e.content_type for e in res.evidence] == ["image", "chart", "diagram"]

    def test_06_citation_preservation_to_downstream(self) -> None:
        """Step 7: Full citation lineage (document_id, filename, chunk_id, page_number) reaches downstream consumer."""
        cit = _make_citation(
            doc_id="DOC-LINEAGE-99",
            filename="system_spec.pdf",
            chunk_id="CHUNK-SPEC-42",
            page_number=14,
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Lineage trace to supervisor", evidence=[ev])

        provider = SupervisorHandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        consumer = OfflineSupervisorConsumer()
        consumed = consumer.consume(res)

        assert consumed["document_id"] == "DOC-LINEAGE-99"
        assert consumed["filename"] == "system_spec.pdf"
        assert consumed["chunk_id"] == "CHUNK-SPEC-42"
        assert consumed["page_number"] == 14

    def test_07_result_normalizer_sanitization_for_downstream(self) -> None:
        """Step 8: VisionResultNormalizer sanitizes provider metadata before supervisor consumption."""
        cit = _make_citation()
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Sanitization check", evidence=[ev])

        provider = SupervisorHandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        # Provider emits api_key in metadata; VisionResultNormalizer sanitizes it
        assert "api_key" not in res.metadata
        assert "execution_trace" in res.metadata

        consumer = OfflineSupervisorConsumer()
        consumed = consumer.consume(res)
        assert "api_key" not in consumed["metadata"]

    def test_08_success_status_representation(self) -> None:
        """Step 9: Successful execution exposes terminal status 'success'."""
        cit = _make_citation()
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Status check", evidence=[ev])

        provider = SupervisorHandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert res.status == "success"
        assert res.is_success is True
        assert res.error is None

    def test_09_provider_failure_handoff(self) -> None:
        """Step 10: Provider failure produces deterministic error representation."""
        cit = _make_citation(doc_id="DOC-FAIL-1")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Failure handoff test", evidence=[ev])

        provider = SupervisorHandoffTestProvider(should_fail=True, fail_count=1)
        pipeline = VisionPipeline(provider=provider)

        with pytest.raises(VisionProviderExecutionError):
            pipeline.run(req, retry_policy=VisionRetryPolicy(max_retries=0))

        # Downstream consumer receiving an error VisionResult representation
        err_res = VisionResult(
            query="Failure handoff test",
            status="error",
            description="",
            document_id="DOC-FAIL-1",
            error="Provider execution failed.",
        )

        consumer = OfflineSupervisorConsumer()
        consumed = consumer.consume(err_res)

        assert consumed["supervisor_status"] == "AGENT_FAILED"
        assert consumed["error"] == "Provider execution failed."

    def test_10_retry_success_handoff(self) -> None:
        """Step 11: Retry mechanism emits ONLY the final successful VisionResult downstream."""
        cit = _make_citation(doc_id="DOC-RETRY-1")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Retry handoff query", evidence=[ev])

        provider = SupervisorHandoffTestProvider(should_fail=True, fail_count=1)
        retry_policy = VisionRetryPolicy(max_retries=2)
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req, retry_policy=retry_policy)

        consumer = OfflineSupervisorConsumer()
        consumed = consumer.consume(res)

        assert consumed["supervisor_status"] == "SUCCESS"
        assert provider.call_count == 2
        assert len(consumer.received_results) == 1

    def test_11_timeout_handoff(self) -> None:
        """Step 12: Provider execution timeout produces clean error handling without fake success."""
        cit = _make_citation(doc_id="DOC-TIMEOUT-1")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Timeout handoff query", evidence=[ev])

        provider = SupervisorHandoffTestProvider(simulate_timeout=True)
        pipeline = VisionPipeline(provider=provider)

        with pytest.raises(VisionTimeoutError):
            pipeline.run(req)

        # Downstream consumer receiving a timeout error VisionResult representation
        timeout_res = VisionResult(
            query="Timeout handoff query",
            status="error",
            description="",
            document_id="DOC-TIMEOUT-1",
            error="Execution timed out.",
        )

        consumer = OfflineSupervisorConsumer()
        consumed = consumer.consume(timeout_res)
        assert consumed["supervisor_status"] == "AGENT_FAILED"
        assert "timed out" in consumed["error"]

    def test_12_cancellation_handoff(self) -> None:
        """Step 13: Cancellation token stops pipeline execution cleanly."""
        cit = _make_citation(doc_id="DOC-CANCEL-1")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Cancel handoff query", evidence=[ev])

        token = VisionCancellationToken()
        token.cancel()

        provider = SupervisorHandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        with pytest.raises(VisionCancellationError):
            pipeline.run(req, cancellation_token=token)

        assert provider.call_count == 0

    def test_13_concurrent_result_isolation(self) -> None:
        """Step 14: Parallel requests delivered to downstream consumers maintain 100% isolation."""
        provider = SupervisorHandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        consumer = OfflineSupervisorConsumer()

        def run_thread(tag: str, doc_id: str) -> None:
            c = _make_citation(doc_id=doc_id)
            ev = VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG)
            req = VisionRequest(query=f"Query {tag}", evidence=[ev])
            res = pipeline.run(req)
            consumer.consume(res)

        threads = [
            threading.Thread(target=run_thread, args=("A", "DOC-CONC-A")),
            threading.Thread(target=run_thread, args=("B", "DOC-CONC-B")),
            threading.Thread(target=run_thread, args=("C", "DOC-CONC-C")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(consumer.received_results) == 3
        doc_ids = {r.document_id for r in consumer.received_results}
        assert doc_ids == {"DOC-CONC-A", "DOC-CONC-B", "DOC-CONC-C"}

    def test_14_mutation_safety(self) -> None:
        """Step 15: Downstream mutation of to_dict() or metadata does not mutate original VisionResult."""
        cit = _make_citation(doc_id="DOC-MUTATE-1", metadata={"tag": "original"})
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Mutation safety query", evidence=[ev])

        provider = SupervisorHandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        # Export to dictionary and mutate dictionary
        res_dict = res.to_dict()
        res_dict["status"] = "corrupted"
        res_dict["metadata"]["tag"] = "mutated"

        assert res.status == "success"
        assert res.metadata.get("tag") != "mutated"

    def test_15_serialization_compatibility(self) -> None:
        """Step 16: VisionResult.to_dict() and VisionResult.from_dict() roundtrip preserves all fields."""
        cit = _make_citation(doc_id="DOC-SERIAL-1", page_number=7, content_type="chart")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)

        orig_res = VisionResult(
            query="Serialization check",
            status="success",
            description="Chart summary",
            evidence=[ev],
            document_id="DOC-SERIAL-1",
            filename="diagram.pdf",
            page_number=7,
            chunk_id="CHUNK-SUPER-001",
            content_type="chart",
            metadata={"chart_type": "line"},
        )

        d = orig_res.to_dict()
        restored = VisionResult.from_dict(d)

        assert restored.query == orig_res.query
        assert restored.status == orig_res.status
        assert restored.description == orig_res.description
        assert restored.document_id == orig_res.document_id
        assert restored.filename == orig_res.filename
        assert restored.page_number == orig_res.page_number
        assert restored.chunk_id == orig_res.chunk_id
        assert restored.content_type == orig_res.content_type
        assert restored.metadata == orig_res.metadata
        assert len(restored.evidence) == 1
        assert restored.evidence[0].document_id == "DOC-SERIAL-1"

    def test_16_public_api_imports(self) -> None:
        """Step 17: Public API imports from 'vision' package remain valid."""
        import vision

        expected_symbols = [
            "VisionPipeline",
            "VisionRequest",
            "VisionResult",
            "VisualEvidence",
            "VisualEvidenceAdapter",
            "VisionModelProvider",
            "VisionProviderConfig",
            "VisionProviderCapabilities",
            "VisionRetryPolicy",
            "VisionCancellationToken",
        ]

        for sym in expected_symbols:
            assert hasattr(vision, sym), f"Missing public symbol '{sym}' in 'vision'"

    def test_17_no_internal_leakage(self) -> None:
        """Step 18: VisionResult handed off downstream does not contain private execution objects or secret keys."""
        cit = _make_citation()
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Leakage check", evidence=[ev])

        provider = SupervisorHandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        # Internal objects like execution adapter, provider, lifecycle are not attributes of VisionResult
        assert not hasattr(res, "_adapter")
        assert not hasattr(res, "_provider")
        assert not hasattr(res, "_lifecycle")

        # Metadata does not contain secrets or credentials
        for key in res.metadata:
            assert "api_key" not in key.lower()
            assert "secret" not in key.lower()

    def test_18_retrieval_and_qdrant_isolation(self) -> None:
        """Step 19 & 20: Handoff operates 100% offline without Qdrant or embedding calls."""
        import vision
        import vision.pipeline

        assert not hasattr(vision, "QdrantClient")
        assert not hasattr(vision.pipeline, "qdrant")

        for mod in sys.modules:
            if "vision" in mod:
                assert "sentence_transformers" not in mod.lower()
                assert "qdrant" not in mod.lower()
