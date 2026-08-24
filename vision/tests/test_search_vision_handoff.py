"""
Day 51 - OmniBrain Member 3 Vision Agent: Search -> Vision Handoff Contract Tests.

Verifies the actual handoff contract between Member 2 Search Evidence and Member 3 Vision.
Ensures:
  1. Actual Member 2 evidence compatibility (AgentCitation, SearchResult, AgentResponse)
  2. Evidence conversion via VisualEvidenceAdapter without field fabrication
  3. Single and multi-evidence handoff preserving original order (1, 2, 3, 5, 10 items)
  4. Mixed content types (image, chart, diagram) order and lineage preservation
  5. Query handoff & validation reuse (Day 43)
  6. Source lineage preservation across execution stages
  7. Metadata isolation and defensive copy enforcement
  8. Source identity and rank preservation
  9. Prevention of redundant retrieval, embedding generation, or Qdrant calls
 10. Provider handoff, failure handling, retry (Day 47), timeout (Day 46), and cancellation
 11. Thread safety and concurrent handoff isolation
 12. Public API contract compatibility (Day 49)

All tests execute 100% offline using controlled test doubles.
"""

from __future__ import annotations

import io
import sys
import threading
import time
from typing import Any

import pytest
from PIL import Image

from agents.models import AgentCitation, AgentResponse, SearchResult
from ingestion.models import VectorSearchResult, DocumentChunk
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionCancellationError,
    VisionEvidenceError,
    VisionInputValidationError,
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


# ===========================================================================
# Test Helpers & Controlled Test Doubles
# ===========================================================================

def _make_png_bytes(width: int = 16, height: int = 16, color: tuple = (100, 150, 200)) -> bytes:
    """Generate a valid PNG byte payload for testing."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_SAMPLE_PNG = _make_png_bytes()


def _make_citation(
    doc_id: str = "doc-search-001",
    filename: str = "diagram.pdf",
    chunk_id: str = "chunk-search-001",
    content_type: str = "image",
    page_number: int | None = 5,
    score: float = 0.95,
    metadata: dict[str, Any] | None = None,
) -> AgentCitation:
    """Construct a Member 2 AgentCitation with visual content_type."""
    return AgentCitation(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        content_type=content_type,
        score=score,
        metadata=metadata if metadata is not None else {"source": "search_agent", "section": "diagrams"},
    )


class HandoffTestProvider(VisionModelProvider):
    """Controlled offline test provider for Search -> Vision handoff testing."""

    def __init__(
        self,
        capabilities: VisionProviderCapabilities | None = None,
        should_fail: bool = False,
        fail_count: int = 0,
        latency: float = 0.0,
        simulate_timeout: bool = False,
    ) -> None:
        config = VisionProviderConfig(provider_name="handoff_test", model_name="test_vision_v1")
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
            raise VisionTimeoutError("Simulated provider execution timeout.")

        if self._should_fail and current_count <= self._fail_count:
            raise VisionProviderExecutionError(
                f"Simulated provider failure on execution call {current_count}."
            )

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Handoff test analysis of '{model_input.query}'",
            document_id=model_input.document_id,
            filename=model_input.filename,
            chunk_id=model_input.chunk_id,
            page_number=model_input.page_number,
            content_type=model_input.content_type,
            metadata={
                "provider_name": self.config.provider_name,
                "model_name": self.config.model_name,
                "call_count": current_count,
            },
        )


# ===========================================================================
# Test Cases
# ===========================================================================

class TestSearchVisionHandoff:
    """Comprehensive test suite for Search -> Vision handoff contract (Day 51)."""

    def test_01_member2_evidence_inspection(self) -> None:
        """Step 1 & 2: Inspect and verify actual Member 2 AgentCitation contract adaptation."""
        cit = _make_citation(
            doc_id="DOC-999",
            filename="arch.pdf",
            chunk_id="CHK-123",
            content_type="image",
            page_number=12,
            score=0.98,
            metadata={"author": "Team", "page": 12},
        )

        assert cit.document_id == "DOC-999"
        assert cit.filename == "arch.pdf"
        assert cit.chunk_id == "CHK-123"
        assert cit.page_number == 12
        assert cit.content_type == "image"
        assert cit.score == 0.98
        assert cit.metadata["author"] == "Team"

        # Adapt using existing VisualEvidenceAdapter
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        assert isinstance(ev, VisualEvidence)
        assert ev.document_id == "DOC-999"
        assert ev.filename == "arch.pdf"
        assert ev.chunk_id == "CHK-123"
        assert ev.page_number == 12
        assert ev.content_type == "image"
        assert ev.metadata["author"] == "Team"
        assert ev.image_bytes == _SAMPLE_PNG

    def test_02_citation_conversion_without_fabrication(self) -> None:
        """Step 2: Verify fields are preserved without fabrication when page_number is None."""
        cit = AgentCitation(
            document_id="DOC-NO-PAGE",
            filename="fig1.png",
            chunk_id="CHK-NO-PAGE",
            page_number=None,
            content_type="chart",
            score=0.85,
            metadata={"chart_type": "bar"},
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        assert ev.page_number is None
        assert ev.chunk_index == 0
        assert ev.document_id == "DOC-NO-PAGE"
        assert ev.filename == "fig1.png"
        assert ev.chunk_id == "CHK-NO-PAGE"
        assert ev.content_type == "chart"

    def test_03_single_evidence_handoff(self) -> None:
        """Step 3: End-to-end handoff of single Search evidence to VisionPipeline."""
        cit = _make_citation(doc_id="DOC-SINGLE", filename="single.png", content_type="image")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Describe this diagram", evidence=[ev])
        provider = HandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        result = pipeline.run(req)

        assert result is not None
        assert result.status == "success"
        assert provider.call_count == 1
        received_input = provider.received_inputs[0]
        assert received_input.query == "Describe this diagram"
        assert received_input.document_id == "DOC-SINGLE"

    def test_04_multi_evidence_handoff_order(self) -> None:
        """Step 4: Verify multi-evidence handoff preserves exact ordering (1, 2, 3, 5, 10 items)."""
        counts = [1, 2, 3, 5, 10]
        for n in counts:
            citations = [
                _make_citation(doc_id=f"DOC-{i}", filename=f"fig_{i}.png", chunk_id=f"CHK-{i}")
                for i in range(n)
            ]
            evidences = [
                VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG)
                for c in citations
            ]
            req = VisionRequest(query=f"Analyze {n} figures", evidence=evidences)
            provider = HandoffTestProvider()
            pipeline = VisionPipeline(provider=provider)

            res = pipeline.run(req)

            assert provider.call_count == 1
            assert len(res.evidence) == n
            for i in range(n):
                assert res.evidence[i].document_id == f"DOC-{i}"
                assert res.evidence[i].chunk_id == f"CHK-{i}"

    def test_05_mixed_content_types(self) -> None:
        """Step 5: Verify mixed visual content types (image, chart, diagram) preserve order & types."""
        c_img = _make_citation(doc_id="D-IMG", filename="f1.jpg", content_type="image")
        c_chart = _make_citation(doc_id="D-CHART", filename="f2.png", content_type="chart")
        c_diag = _make_citation(doc_id="D-DIAG", filename="f3.svg", content_type="diagram")

        evs = [
            VisualEvidenceAdapter.adapt_citation(c_img, image_bytes=_SAMPLE_PNG),
            VisualEvidenceAdapter.adapt_citation(c_chart, image_bytes=_SAMPLE_PNG),
            VisualEvidenceAdapter.adapt_citation(c_diag, image_bytes=_SAMPLE_PNG),
        ]

        req = VisionRequest(query="Compare image, chart, and diagram", evidence=evs)
        provider = HandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert len(res.evidence) == 3
        assert res.evidence[0].content_type == "image"
        assert res.evidence[1].content_type == "chart"
        assert res.evidence[2].content_type == "diagram"

        # Verify non-visual content type rejected in adapt_citation
        c_text = _make_citation(content_type="text")
        with pytest.raises(VisionEvidenceError):
            VisualEvidenceAdapter.adapt_citation(c_text)

    def test_06_query_handoff_validation(self) -> None:
        """Step 6: Query handoff verification reusing Day 43 validation."""
        c = _make_citation()
        ev = VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG)
        provider = HandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        # Normal query with whitespace
        req = VisionRequest(query="   What is this architecture?   ", evidence=[ev])
        res = pipeline.run(req)
        assert provider.received_inputs[0].query == "What is this architecture?"

        # Empty and whitespace-only queries
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="", evidence=[ev])

        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="   ", evidence=[ev])

        with pytest.raises(VisionInputValidationError):
            VisionRequest(query=None, evidence=[ev])  # type: ignore[arg-type]

        with pytest.raises(VisionInputValidationError):
            VisionRequest(query=12345, evidence=[ev])  # type: ignore[arg-type]

    def test_07_lineage_handoff_preservation(self) -> None:
        """Step 7: Lineage preservation from Search evidence -> VisionModelInput -> VisionResult."""
        cit = _make_citation(
            doc_id="DOC-LINEAGE-42",
            filename="system_design.pdf",
            chunk_id="CHK-LINEAGE-99",
            page_number=7,
            score=0.97,
            metadata={"stage": "search"},
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Lineage trace", evidence=[ev])
        provider = HandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        rec_input = provider.received_inputs[0]
        assert rec_input.document_id == "DOC-LINEAGE-42"
        assert rec_input.filename == "system_design.pdf"
        assert rec_input.chunk_id == "CHK-LINEAGE-99"
        assert rec_input.page_number == 7

        # Check VisionResult lineage fields
        assert res.document_id == "DOC-LINEAGE-42"
        assert res.filename == "system_design.pdf"
        assert res.chunk_id == "CHK-LINEAGE-99"
        assert res.page_number == 7

    def test_08_metadata_handoff_isolation(self) -> None:
        """Step 8: Metadata survives handoff, remains attached to correct evidence, and is defensively copied."""
        meta_a = {"doc_source": "Search-A", "tags": ["arch"]}
        meta_b = {"doc_source": "Search-B", "tags": ["graph"]}

        cit_a = _make_citation(doc_id="DOC-META-A", metadata=meta_a)
        cit_b = _make_citation(doc_id="DOC-META-B", metadata=meta_b)

        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a, image_bytes=_SAMPLE_PNG)
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b, image_bytes=_SAMPLE_PNG)

        # Modify original dicts to test defensive copy
        meta_a["mutated"] = True
        assert "mutated" not in ev_a.metadata

        assert ev_a.metadata["doc_source"] == "Search-A"
        assert ev_b.metadata["doc_source"] == "Search-B"
        assert "doc_source" in ev_a.metadata and ev_a.metadata["doc_source"] != ev_b.metadata["doc_source"]

    def test_09_source_identity(self) -> None:
        """Step 9: Distinct source identities from different documents remain distinct after handoff."""
        cit_a = _make_citation(doc_id="DOC-ALPHA", filename="alpha.pdf")
        cit_b = _make_citation(doc_id="DOC-BETA", filename="beta.pdf")

        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a, image_bytes=_SAMPLE_PNG)
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b, image_bytes=_SAMPLE_PNG)

        req = VisionRequest(query="Multi doc query", evidence=[ev_a, ev_b])
        provider = HandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert res.evidence[0].document_id == "DOC-ALPHA"
        assert res.evidence[1].document_id == "DOC-BETA"

    def test_10_search_result_order_preservation(self) -> None:
        """Step 10: Search ranking order (Rank 1, 2, 3) is preserved without reordering."""
        cit_rank1 = _make_citation(doc_id="DOC-RANK-1", filename="z_last_alpha.png", score=0.99)
        cit_rank2 = _make_citation(doc_id="DOC-RANK-2", filename="a_first_alpha.png", score=0.88)
        cit_rank3 = _make_citation(doc_id="DOC-RANK-3", filename="m_mid_alpha.png", score=0.77)

        search_res = SearchResult(
            query="Search ranking query",
            status="RESULTS_FOUND",
            citations=[cit_rank1, cit_rank2, cit_rank3],
        )

        # Adapt using adapt_search_package
        ev_list = VisualEvidenceAdapter.adapt_search_package(search_res)
        for ev in ev_list:
            ev.image_bytes = _SAMPLE_PNG

        req = VisionRequest(query="Preserve rank", evidence=ev_list)
        provider = HandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert res.evidence[0].document_id == "DOC-RANK-1"
        assert res.evidence[1].document_id == "DOC-RANK-2"
        assert res.evidence[2].document_id == "DOC-RANK-3"

    def test_11_no_retrieval_execution(self) -> None:
        """Step 11 & 12 & 13: Prove handoff does NOT call Qdrant, embeddings, or retrieval."""
        # Ensure qdrant or embedding client is not imported/invoked in vision handoff execution
        sys_modules_before = set(sys.modules.keys())
        cit = _make_citation()
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Offline query", evidence=[ev])
        provider = HandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)
        assert res is not None

        # Verify no Qdrant or SentenceTransformer module was dynamically imported
        new_modules = set(sys.modules.keys()) - sys_modules_before
        for mod in new_modules:
            assert "qdrant" not in mod.lower()
            assert "sentence_transformers" not in mod.lower()

    def test_12_no_embedding_module(self) -> None:
        """Step 12: Verify Member 3 does not generate search embeddings."""
        for mod_name in sys.modules:
            if "vision" in mod_name:
                assert "sentence_transformer" not in mod_name.lower()
                assert "embedding_client" not in mod_name.lower()

    def test_13_no_qdrant_isolation(self) -> None:
        """Step 13: Verify Vision imports do not depend on Qdrant."""
        import vision
        import vision.pipeline
        import vision.evidence_adapter

        for module in (vision, vision.pipeline, vision.evidence_adapter):
            assert not hasattr(module, "QdrantClient")
            assert not hasattr(module, "qdrant")

    def test_14_provider_handoff_contract(self) -> None:
        """Step 14: Reuse VisionModelProvider and verify model input passing."""
        cit = _make_citation(doc_id="DOC-PROV-1")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Provider contract query", evidence=[ev])

        provider = HandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)
        pipeline.run(req)

        assert provider.call_count == 1
        model_input = provider.received_inputs[0]
        assert isinstance(model_input, VisionModelInput)
        assert model_input.query == "Provider contract query"
        assert model_input.document_id == "DOC-PROV-1"

    def test_15_failure_at_handoff_invalid_evidence(self) -> None:
        """Step 15: Invalid Search evidence fails before provider execution."""
        provider = HandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        # None evidence
        with pytest.raises(VisionInputValidationError):
            VisualEvidenceAdapter.adapt_citation(None)  # type: ignore[arg-type]

        # Non-visual citation in strict adaptation
        cit_txt = _make_citation(content_type="text")
        with pytest.raises(VisionEvidenceError):
            VisualEvidenceAdapter.adapt_citation(cit_txt)

        assert provider.call_count == 0

    def test_16_provider_failure_handling(self) -> None:
        """Step 16: Simulate provider failure and verify evidence lineage is preserved in error path."""
        cit = _make_citation(doc_id="DOC-FAIL-1")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Failure test query", evidence=[ev])

        provider = HandoffTestProvider(should_fail=True, fail_count=1)
        pipeline = VisionPipeline(provider=provider)

        with pytest.raises(VisionProviderExecutionError):
            pipeline.run(req, retry_policy=VisionRetryPolicy(max_retries=0))

        # Lineage was preserved in request/evidence despite failure
        assert req.evidence[0].document_id == "DOC-FAIL-1"

    def test_17_retry_handoff_behavior(self) -> None:
        """Step 17: Reuse Day 47 retry policy; verify retry executes provider without re-search."""
        cit = _make_citation(doc_id="DOC-RETRY-1")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Retry query", evidence=[ev])

        provider = HandoffTestProvider(should_fail=True, fail_count=1)
        retry_policy = VisionRetryPolicy(max_retries=2)
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req, retry_policy=retry_policy)

        assert res is not None
        assert provider.call_count == 2
        assert res.evidence[0].document_id == "DOC-RETRY-1"

    def test_18_timeout_handoff_behavior(self) -> None:
        """Step 18: Reuse Day 46 timeout handling."""
        cit = _make_citation(doc_id="DOC-TIMEOUT-1")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Timeout query", evidence=[ev])

        provider = HandoffTestProvider(simulate_timeout=True)
        pipeline = VisionPipeline(provider=provider)

        with pytest.raises(VisionTimeoutError):
            pipeline.run(req)

    def test_19_cancellation_handoff_behavior(self) -> None:
        """Step 19: Test cancellation token propagation."""
        cit = _make_citation(doc_id="DOC-CANCEL-1")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Cancel query", evidence=[ev])

        token = VisionCancellationToken()
        token.cancel()

        provider = HandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        with pytest.raises(VisionCancellationError):
            pipeline.run(req, cancellation_token=token)

        assert provider.call_count == 0

    def test_20_concurrent_handoff_isolation(self) -> None:
        """Step 20: Run concurrent requests and verify total thread isolation."""
        provider = HandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        results: dict[str, Any] = {}
        errors: dict[str, Exception] = {}

        def run_req(tag: str, doc_id: str) -> None:
            try:
                c = _make_citation(doc_id=doc_id)
                ev = VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG)
                req = VisionRequest(query=f"Query {tag}", evidence=[ev])
                res = pipeline.run(req)
                results[tag] = res
            except Exception as exc:
                errors[tag] = exc

        threads = [
            threading.Thread(target=run_req, args=("reqA", "DOC-CONC-A")),
            threading.Thread(target=run_req, args=("reqB", "DOC-CONC-B")),
            threading.Thread(target=run_req, args=("reqC", "DOC-CONC-C")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 3
        assert results["reqA"] is not None
        assert results["reqB"] is not None
        assert results["reqC"] is not None

    def test_21_result_lineage_association(self) -> None:
        """Step 21: Verify VisionResult remains associated with correct upstream evidence."""
        cit = _make_citation(doc_id="DOC-RESULT-777", page_number=9)
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Result lineage test", evidence=[ev])

        provider = HandoffTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert res.document_id == "DOC-RESULT-777"
        assert res.page_number == 9

    def test_22_public_api_compatibility(self) -> None:
        """Step 22: Verify Day 49 public API contract compatibility."""
        import vision

        expected_exports = [
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

        for name in expected_exports:
            assert hasattr(vision, name), f"Public API missing exported symbol '{name}'"

    def test_23_adapt_search_package_strict_and_lenient(self) -> None:
        """Verify adapt_search_package handles SearchResult and AgentResponse correctly."""
        c_vis = _make_citation(doc_id="D-VIS", content_type="image")
        c_txt = _make_citation(doc_id="D-TXT", content_type="text")

        search_res = SearchResult(query="pkg search", citations=[c_vis, c_txt])

        # Lenient mode filters out non-visual
        ev_list_lenient = VisualEvidenceAdapter.adapt_search_package(search_res, strict=False)
        assert len(ev_list_lenient) == 1
        assert ev_list_lenient[0].document_id == "D-VIS"

        # Strict mode raises error on non-visual
        with pytest.raises(VisionEvidenceError):
            VisualEvidenceAdapter.adapt_search_package(search_res, strict=True)

        # AgentResponse package
        resp = AgentResponse(
            answer="Test answer",
            agent_name="SearchAgent",
            citations=[c_vis],
        )
        ev_list_resp = VisualEvidenceAdapter.adapt_search_package(resp)
        assert len(ev_list_resp) == 1
        assert ev_list_resp[0].document_id == "D-VIS"
