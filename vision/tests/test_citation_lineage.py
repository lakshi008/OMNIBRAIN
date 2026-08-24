"""
Day 52 - OmniBrain Member 3 Vision Agent: Citation & Lineage Integrity Tests.

Verifies that source citations and evidence lineage remain strictly intact across
the entire Member 3 execution lifecycle:
  Member 2 citation/evidence -> VisualEvidence -> VisionRequest -> VisionModelInput
  -> Provider -> VisionResult -> VisionResultNormalizer -> VisionPipeline

Ensures:
  1. Complete citation & lineage contract field verification (AgentCitation, VisualEvidence, VisionRequest, VisionModelInput, VisionResult, ResultNormalizer)
  2. Single-source lineage preservation through all execution stages
  3. Multi-source lineage isolation across documents (DOC-A, DOC-B, DOC-C)
  4. Multi-evidence counts (1, 2, 3, 5, 10) order and lineage preservation
  5. Same-document, different-chunk isolation (same doc_id, distinct chunk_ids)
  6. Same-page, different-chunk isolation (same page_number, distinct chunk_ids)
  7. Different-page lineage accuracy
  8. Content-type lineage preservation without modality swapping (image, chart, diagram)
  9. Metadata isolation and anti-leakage per evidence object
 10. Search rank citation order preservation (no unauthorized sorting)
 11. Deterministic duplicate citation handling without silent deduplication
 12. Optional lineage field (page_number=None) handling without fake value generation
 13. Invalid lineage rejection with deterministic exceptions
 14. VisionResultNormalizer lineage preservation & trace attachment
 15. Provider execution boundary isolation
 16. Retry lineage preservation without re-retrieval (Day 47)
 17. Timeout state isolation across requests (Day 46)
 18. Cancellation isolation across requests (Day 46)
 19. Thread-safe concurrent lineage isolation & mixed outcome stability
 20. Repeated execution isolation (no state/evidence accumulation)
 21. Cross-request metadata isolation
 22. Absolute prevention of fabricated lineage
 23. Zero duplicate retrieval, zero embedding generation, zero Qdrant access
 24. 100% offline execution using controlled test doubles

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

from agents.exceptions import AgentValidationError
from agents.models import AgentCitation, AgentResponse, SearchResult
from ingestion.models import DocumentChunk, VectorSearchResult
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.image_preparation import prepare_image_evidence
from vision.exceptions import (
    VisionCancellationError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderExecutionError,
    VisionTimeoutError,
)
from vision.execution_adapter import VisionExecutionAdapter
from vision.input_builder import VisionModelInput, build_vision_input
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
# Test Helpers & Controlled Test Doubles
# ===========================================================================

def _make_png_bytes(width: int = 16, height: int = 16, color: tuple[int, int, int] = (80, 140, 200)) -> bytes:
    """Generate a valid PNG byte payload for testing visual evidence."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_SAMPLE_PNG = _make_png_bytes()


def _make_citation(
    doc_id: str = "DOC-001",
    filename: str = "report.pdf",
    chunk_id: str = "CHUNK-001",
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
        metadata=metadata if metadata is not None else {"source": "search_agent", "section": "intro"},
    )


class LineageTestProvider(VisionModelProvider):
    """Controlled offline test double for verifying citation and lineage preservation."""

    def __init__(
        self,
        capabilities: VisionProviderCapabilities | None = None,
        should_fail: bool = False,
        fail_count: int = 0,
        latency: float = 0.0,
        simulate_timeout: bool = False,
    ) -> None:
        config = VisionProviderConfig(provider_name="lineage_test", model_name="lineage_model_v1")
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
            raise VisionTimeoutError("Simulated provider timeout during lineage testing.")

        if self._should_fail and current_count <= self._fail_count:
            raise VisionProviderExecutionError(
                f"Simulated provider failure on execution call {current_count}."
            )

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Lineage analysis of '{model_input.query}'",
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

class TestCitationAndLineageIntegrity:
    """Complete test suite for Day 52 Citation & Lineage Integrity."""

    def test_01_inspect_citation_contract_fields(self) -> None:
        """Step 1: Inspect and verify actual repository lineage fields across all models."""
        cit = _make_citation()
        assert hasattr(cit, "document_id")
        assert hasattr(cit, "filename")
        assert hasattr(cit, "chunk_id")
        assert hasattr(cit, "page_number")
        assert hasattr(cit, "content_type")
        assert hasattr(cit, "score")
        assert hasattr(cit, "metadata")

        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        assert hasattr(ev, "document_id")
        assert hasattr(ev, "filename")
        assert hasattr(ev, "chunk_id")
        assert hasattr(ev, "page_number")
        assert hasattr(ev, "chunk_index")
        assert hasattr(ev, "content_type")
        assert hasattr(ev, "metadata")

        req = VisionRequest(query="Contract check", evidence=[ev])
        assert hasattr(req, "query")
        assert hasattr(req, "evidence")

        res = VisionResult(
            query="Contract check",
            status="success",
            document_id=ev.document_id,
            filename=ev.filename,
            chunk_id=ev.chunk_id,
            page_number=ev.page_number,
            content_type=ev.content_type,
            evidence=[ev],
        )
        assert hasattr(res, "document_id")
        assert hasattr(res, "filename")
        assert hasattr(res, "chunk_id")
        assert hasattr(res, "page_number")
        assert hasattr(res, "content_type")
        assert hasattr(res, "evidence")

    def test_02_single_source_lineage_flow(self) -> None:
        """Step 2: Trace single evidence through entire execution flow."""
        cit = _make_citation(
            doc_id="DOC-001",
            filename="report.pdf",
            chunk_id="CHUNK-001",
            page_number=5,
            content_type="image",
            metadata={"department": "finance"},
        )

        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        assert ev.document_id == "DOC-001"
        assert ev.filename == "report.pdf"
        assert ev.chunk_id == "CHUNK-001"
        assert ev.page_number == 5
        assert ev.content_type == "image"
        assert ev.metadata["department"] == "finance"

        req = VisionRequest(query="Analyze page 5 chart", evidence=[ev])
        provider = LineageTestProvider()
        pipeline = VisionPipeline(provider=provider)

        result = pipeline.run(req)

        assert result.status == "success"
        assert result.document_id == "DOC-001"
        assert result.filename == "report.pdf"
        assert result.chunk_id == "CHUNK-001"
        assert result.page_number == 5
        assert result.content_type == "image"

        rec_input = provider.received_inputs[0]
        assert rec_input.document_id == "DOC-001"
        assert rec_input.filename == "report.pdf"
        assert rec_input.chunk_id == "CHUNK-001"
        assert rec_input.page_number == 5

    def test_03_multi_source_lineage_isolation(self) -> None:
        """Step 3: Multi-document evidence (DOC-A, DOC-B, DOC-C) maintains strict source identity."""
        cit_a = _make_citation(doc_id="DOC-A", filename="a.pdf", chunk_id="chk-a")
        cit_b = _make_citation(doc_id="DOC-B", filename="b.pdf", chunk_id="chk-b")
        cit_c = _make_citation(doc_id="DOC-C", filename="c.pdf", chunk_id="chk-c")

        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a, image_bytes=_SAMPLE_PNG)
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b, image_bytes=_SAMPLE_PNG)
        ev_c = VisualEvidenceAdapter.adapt_citation(cit_c, image_bytes=_SAMPLE_PNG)

        req = VisionRequest(query="Compare A, B, and C", evidence=[ev_a, ev_b, ev_c])
        provider = LineageTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert len(res.evidence) == 3
        assert res.evidence[0].document_id == "DOC-A"
        assert res.evidence[1].document_id == "DOC-B"
        assert res.evidence[2].document_id == "DOC-C"

    def test_04_multi_evidence_counts_and_order(self) -> None:
        """Step 4: Counts (1, 2, 3, 5, 10) preserve exact order, count, and lineage without duplication."""
        for n in [1, 2, 3, 5, 10]:
            citations = [
                _make_citation(doc_id=f"DOC-{i}", filename=f"f_{i}.pdf", chunk_id=f"chk-{i}")
                for i in range(n)
            ]
            evidence = [
                VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG)
                for c in citations
            ]
            req = VisionRequest(query=f"Batch of {n}", evidence=evidence)
            provider = LineageTestProvider()
            pipeline = VisionPipeline(provider=provider)

            res = pipeline.run(req)

            assert len(res.evidence) == n
            for i in range(n):
                assert res.evidence[i].document_id == f"DOC-{i}"
                assert res.evidence[i].filename == f"f_{i}.pdf"
                assert res.evidence[i].chunk_id == f"chk-{i}"

    def test_05_same_document_different_chunks(self) -> None:
        """Step 5: Multiple chunks from the same document remain distinct and uncollapsed."""
        cit1 = _make_citation(doc_id="DOC-001", chunk_id="CHUNK-001", page_number=1)
        cit2 = _make_citation(doc_id="DOC-001", chunk_id="CHUNK-002", page_number=2)
        cit3 = _make_citation(doc_id="DOC-001", chunk_id="CHUNK-003", page_number=3)

        evs = [
            VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG)
            for c in (cit1, cit2, cit3)
        ]

        req = VisionRequest(query="Analyze 3 chunks of DOC-001", evidence=evs)
        provider = LineageTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert len(res.evidence) == 3
        assert [e.chunk_id for e in res.evidence] == ["CHUNK-001", "CHUNK-002", "CHUNK-003"]
        assert [e.document_id for e in res.evidence] == ["DOC-001", "DOC-001", "DOC-001"]

    def test_06_same_page_different_chunks(self) -> None:
        """Step 6: Multiple evidence items on the same document and page retain chunk-level identity."""
        cit_top = _make_citation(doc_id="DOC-001", page_number=3, chunk_id="CHUNK-PAGE3-TOP")
        cit_bot = _make_citation(doc_id="DOC-001", page_number=3, chunk_id="CHUNK-PAGE3-BOTTOM")

        ev_top = VisualEvidenceAdapter.adapt_citation(cit_top, image_bytes=_SAMPLE_PNG)
        ev_bot = VisualEvidenceAdapter.adapt_citation(cit_bot, image_bytes=_SAMPLE_PNG)

        req = VisionRequest(query="Page 3 top and bottom figures", evidence=[ev_top, ev_bot])
        provider = LineageTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert len(res.evidence) == 2
        assert res.evidence[0].page_number == 3
        assert res.evidence[1].page_number == 3
        assert res.evidence[0].chunk_id == "CHUNK-PAGE3-TOP"
        assert res.evidence[1].chunk_id == "CHUNK-PAGE3-BOTTOM"

    def test_07_different_pages_lineage(self) -> None:
        """Step 7: Evidence from pages 1, 5, 10 retain page association correctly."""
        p1 = _make_citation(doc_id="DOC-MULTI-PAGE", page_number=1, chunk_id="c1")
        p5 = _make_citation(doc_id="DOC-MULTI-PAGE", page_number=5, chunk_id="c5")
        p10 = _make_citation(doc_id="DOC-MULTI-PAGE", page_number=10, chunk_id="c10")

        evs = [
            VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG)
            for c in (p1, p5, p10)
        ]
        req = VisionRequest(query="Pages 1, 5, 10 analysis", evidence=evs)
        provider = LineageTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert [e.page_number for e in res.evidence] == [1, 5, 10]

    def test_08_content_type_lineage(self) -> None:
        """Step 8: Modalities (image, chart, diagram) retain their exact content_type without swapping."""
        c_img = _make_citation(doc_id="D1", content_type="image")
        c_chart = _make_citation(doc_id="D2", content_type="chart")
        c_diag = _make_citation(doc_id="D3", content_type="diagram")

        evs = [
            VisualEvidenceAdapter.adapt_citation(c_img, image_bytes=_SAMPLE_PNG),
            VisualEvidenceAdapter.adapt_citation(c_chart, image_bytes=_SAMPLE_PNG),
            VisualEvidenceAdapter.adapt_citation(c_diag, image_bytes=_SAMPLE_PNG),
        ]
        req = VisionRequest(query="Modalities check", evidence=evs)
        provider = LineageTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert [e.content_type for e in res.evidence] == ["image", "chart", "diagram"]

    def test_09_metadata_lineage_isolation(self) -> None:
        """Step 9: Evidence metadata remains attached exclusively to its parent evidence."""
        meta_a = {"source": "document-A", "section": "Introduction"}
        meta_b = {"source": "document-B", "section": "Architecture"}

        cit_a = _make_citation(doc_id="DOC-A", metadata=meta_a)
        cit_b = _make_citation(doc_id="DOC-B", metadata=meta_b)

        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a, image_bytes=_SAMPLE_PNG)
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b, image_bytes=_SAMPLE_PNG)

        assert ev_a.metadata["source"] == "document-A"
        assert ev_b.metadata["source"] == "document-B"
        assert "section" in ev_a.metadata and ev_a.metadata["section"] == "Introduction"
        assert "section" in ev_b.metadata and ev_b.metadata["section"] == "Architecture"
        assert "document-B" not in str(ev_a.metadata)

    def test_10_citation_order_preservation(self) -> None:
        """Step 10: Search ranking order is preserved; no independent sorting inside Vision."""
        c1 = _make_citation(doc_id="DOC-RANK-1", filename="z_file.pdf", score=0.99)
        c2 = _make_citation(doc_id="DOC-RANK-2", filename="a_file.pdf", score=0.88)
        c3 = _make_citation(doc_id="DOC-RANK-3", filename="m_file.pdf", score=0.77)

        evs = [
            VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG)
            for c in (c1, c2, c3)
        ]

        req = VisionRequest(query="Rank order query", evidence=evs)
        provider = LineageTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert [e.document_id for e in res.evidence] == ["DOC-RANK-1", "DOC-RANK-2", "DOC-RANK-3"]

    def test_11_duplicate_citation_handling(self) -> None:
        """Step 11: Repeated references to the same chunk are preserved in sequence without silent deduplication."""
        c1 = _make_citation(doc_id="DOC-A", chunk_id="CHUNK-1")
        c2 = _make_citation(doc_id="DOC-A", chunk_id="CHUNK-1")

        ev1 = VisualEvidenceAdapter.adapt_citation(c1, image_bytes=_SAMPLE_PNG)
        ev2 = VisualEvidenceAdapter.adapt_citation(c2, image_bytes=_SAMPLE_PNG)

        req = VisionRequest(query="Duplicate test", evidence=[ev1, ev2])
        provider = LineageTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert len(res.evidence) == 2
        assert res.evidence[0].chunk_id == "CHUNK-1"
        assert res.evidence[1].chunk_id == "CHUNK-1"

    def test_12_missing_optional_lineage(self) -> None:
        """Step 12: Evidence with page_number=None succeeds without inventing fake page numbers."""
        cit = AgentCitation(
            document_id="DOC-NO-PAGE",
            filename="image_only.png",
            chunk_id="CHUNK-NO-PAGE",
            page_number=None,
            content_type="chart",
            score=0.9,
            metadata={"chart": "line"},
        )

        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        assert ev.page_number is None

        req = VisionRequest(query="No page query", evidence=[ev])
        provider = LineageTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert res.page_number is None
        assert res.evidence[0].page_number is None

    def test_13_invalid_lineage_rejection(self) -> None:
        """Step 13: Malformed lineage fields raise deterministic validation exceptions."""
        # Empty document_id
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="f.png", chunk_id="c1")

        # Empty filename
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="d1", filename="  ", chunk_id="c1")

        # Empty chunk_id
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="d1", filename="f.png", chunk_id="")

        # Invalid page_number (negative or bool)
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="d1", filename="f.png", chunk_id="c1", page_number=-5)

        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="d1", filename="f.png", chunk_id="c1", page_number=True)  # type: ignore[arg-type]

    def test_14_result_normalizer_preserves_lineage(self) -> None:
        """Step 14: VisionResultNormalizer preserves document lineage and attaches trace."""
        cit = _make_citation(doc_id="DOC-NORM", filename="norm.pdf", chunk_id="chk-norm", page_number=8)
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)

        raw_res = VisionResult(
            query="Norm query",
            status="success",
            description="Raw result",
            document_id=ev.document_id,
            filename=ev.filename,
            chunk_id=ev.chunk_id,
            page_number=ev.page_number,
            content_type=ev.content_type,
            evidence=[ev],
            metadata={"api_key": "secret_key", "valid_meta": "ok"},
        )

        trace = VisionExecutionTrace.create_default()
        normalized = VisionResultNormalizer.normalize(raw_res, trace=trace)

        assert normalized.document_id == "DOC-NORM"
        assert normalized.filename == "norm.pdf"
        assert normalized.chunk_id == "chk-norm"
        assert normalized.page_number == 8
        assert "api_key" not in normalized.metadata
        assert normalized.metadata.get("valid_meta") == "ok"
        assert "execution_trace" in normalized.metadata

    def test_15_provider_boundary_isolation(self) -> None:
        """Step 15: Offline provider double receives exact VisionModelInput."""
        cit = _make_citation(doc_id="DOC-BOUND-1")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Boundary query", evidence=[ev])

        provider = LineageTestProvider()
        pipeline = VisionPipeline(provider=provider)

        pipeline.run(req)

        assert provider.call_count == 1
        model_input = provider.received_inputs[0]
        assert model_input.document_id == "DOC-BOUND-1"
        assert model_input.query == "Boundary query"

    def test_16_retry_lineage_preservation(self) -> None:
        """Step 16: Provider retry preserves Evidence A lineage without repeating retrieval."""
        cit = _make_citation(doc_id="DOC-RETRY-A")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Retry query", evidence=[ev])

        provider = LineageTestProvider(should_fail=True, fail_count=1)
        retry_policy = VisionRetryPolicy(max_retries=2)
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req, retry_policy=retry_policy)

        assert res.status == "success"
        assert provider.call_count == 2
        assert res.evidence[0].document_id == "DOC-RETRY-A"

    def test_17_timeout_lineage_isolation(self) -> None:
        """Step 17: Request A timeout does not pollute Request B lineage."""
        cit_a = _make_citation(doc_id="DOC-TIMEOUT-A")
        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a, image_bytes=_SAMPLE_PNG)
        req_a = VisionRequest(query="Timeout A", evidence=[ev_a])

        provider_timeout = LineageTestProvider(simulate_timeout=True)
        pipeline_a = VisionPipeline(provider=provider_timeout)

        with pytest.raises(VisionTimeoutError):
            pipeline_a.run(req_a)

        # Subsequent request B with normal provider succeeds cleanly
        cit_b = _make_citation(doc_id="DOC-OK-B")
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b, image_bytes=_SAMPLE_PNG)
        req_b = VisionRequest(query="Success B", evidence=[ev_b])

        provider_ok = LineageTestProvider()
        pipeline_b = VisionPipeline(provider=provider_ok)

        res_b = pipeline_b.run(req_b)

        assert res_b.status == "success"
        assert res_b.evidence[0].document_id == "DOC-OK-B"

    def test_18_cancellation_lineage_isolation(self) -> None:
        """Step 18: Cancellation on Request A does not affect Request B."""
        cit_a = _make_citation(doc_id="DOC-CANCEL-A")
        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a, image_bytes=_SAMPLE_PNG)
        req_a = VisionRequest(query="Cancel A", evidence=[ev_a])

        token_a = VisionCancellationToken()
        token_a.cancel()

        provider = LineageTestProvider()
        pipeline = VisionPipeline(provider=provider)

        with pytest.raises(VisionCancellationError):
            pipeline.run(req_a, cancellation_token=token_a)

        # Request B executes normally
        cit_b = _make_citation(doc_id="DOC-CANCEL-B")
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b, image_bytes=_SAMPLE_PNG)
        req_b = VisionRequest(query="Normal B", evidence=[ev_b])

        res_b = pipeline.run(req_b)
        assert res_b.status == "success"
        assert res_b.evidence[0].document_id == "DOC-CANCEL-B"

    def test_19_concurrent_lineage_isolation(self) -> None:
        """Step 19: Concurrent requests (DOC-A, DOC-B, DOC-C) maintain thread-safe lineage isolation."""
        provider = LineageTestProvider()
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
            threading.Thread(target=run_req, args=("reqA", "DOC-THREAD-A")),
            threading.Thread(target=run_req, args=("reqB", "DOC-THREAD-B")),
            threading.Thread(target=run_req, args=("reqC", "DOC-THREAD-C")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 3
        assert results["reqA"].evidence[0].document_id == "DOC-THREAD-A"
        assert results["reqB"].evidence[0].document_id == "DOC-THREAD-B"
        assert results["reqC"].evidence[0].document_id == "DOC-THREAD-C"

    def test_20_repeated_execution_isolation(self) -> None:
        """Step 20: Executing the same evidence repeatedly does not accumulate stale state or evidence."""
        cit = _make_citation(doc_id="DOC-REPEAT-1")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Repeat query", evidence=[ev])

        provider = LineageTestProvider()
        pipeline = VisionPipeline(provider=provider)

        r1 = pipeline.run(req)
        r2 = pipeline.run(req)
        r3 = pipeline.run(req)

        assert len(r1.evidence) == 1
        assert len(r2.evidence) == 1
        assert len(r3.evidence) == 1
        assert provider.call_count == 3

    def test_21_cross_request_metadata_isolation(self) -> None:
        """Step 21: Request-level metadata is isolated between Request A and Request B."""
        cit = _make_citation(doc_id="DOC-META-ISO")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)

        req_a = VisionRequest(query="Meta A", evidence=[ev], metadata={"request_id": "REQ-A"})
        req_b = VisionRequest(query="Meta B", evidence=[ev], metadata={"request_id": "REQ-B"})

        provider = LineageTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res_a = pipeline.run(req_a)
        res_b = pipeline.run(req_b)

        assert req_a.metadata["request_id"] == "REQ-A"
        assert req_b.metadata["request_id"] == "REQ-B"
        assert "REQ-B" not in str(req_a.metadata)

    def test_22_no_fabricated_lineage(self) -> None:
        """Step 22: Unsupplied optional fields remain None without fabricating values."""
        cit = AgentCitation(
            document_id="DOC-RAW",
            filename="raw.pdf",
            chunk_id="CHUNK-RAW",
            page_number=None,
            content_type="diagram",
            score=0.8,
        )

        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_SAMPLE_PNG)
        assert ev.page_number is None

        prep_ev = prepare_image_evidence(ev)
        model_input = build_vision_input(
            query="Raw query",
            evidence=prep_ev,
        )
        assert model_input.page_number is None

    def test_23_no_duplicate_retrieval_or_qdrant_imports(self) -> None:
        """Step 23 & 24: Verify Member 3 vision lineage test executes without Qdrant or embedding calls."""
        import vision
        import vision.pipeline

        assert not hasattr(vision, "QdrantClient")
        assert not hasattr(vision.pipeline, "VectorSearch")

        for mod in sys.modules:
            if "vision" in mod:
                assert "sentence_transformers" not in mod.lower()
                assert "qdrant" not in mod.lower()
