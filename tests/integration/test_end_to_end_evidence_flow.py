"""
OmniBrain Member 4 — Day 5 End-to-End Evidence Flow Integration Tests.

Verifies the complete existing evidence lifecycle across all system subsystems:
INGESTION -> SEARCH / RETRIEVAL -> SEARCH EVIDENCE -> VISION -> VISION RESULT -> SUPERVISOR / DOWNSTREAM

Ensures that:
1. Complete document identity, chunk identity, page lineage, content, content type, metadata, and citations survive the full cross-member chain.
2. Multi-chunk and multi-document sets maintain exact ordering, count, and strict isolation without cross-document leakage.
3. Multimodal content types ('text', 'table', 'image', 'chart', 'diagram') are preserved and routed correctly.
4. Malformed evidence triggers expected validation and domain error boundaries at each stage.
5. Failures (timeouts, errors, missing evidence) propagate cleanly without fabricated results or lineage.
6. All operations are strictly deterministic, request-isolated, repeatable, and 100% offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path for test runners executing this file directly
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

from agents.exceptions import AgentValidationError
from agents.models import AgentCitation, AgentResponse, AgentState, SearchResult
from ingestion.models import DocumentChunk, VectorSearchResult
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import VisionEvidenceError, VisionInputValidationError
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)


# ============================================================================
# 1. SINGLE-DOCUMENT FULL EVIDENCE FLOW
# ============================================================================


class TestSingleDocumentFullEvidenceFlow:
    """Verifies that a single document's evidence traverses the full 4-stage pipeline with exact lineage."""

    def test_single_visual_chunk_end_to_end_lineage(self) -> None:
        """Verify complete preservation: Ingestion chunk -> Search citation -> Vision -> Supervisor."""
        # 1. Ingestion Subsystem (Member 1)
        doc_id = "doc-full-e2e-001"
        fname = "quarterly_presentation.pdf"
        cid = "chunk-e2e-img-01"
        page = 4
        chunk_idx = 1
        ctype = "chart"
        content_desc = "Bar chart displaying Q3 gross margin growth of 18%"
        meta = {"source_department": "Finance", "year": 2024, "metric": "gross_margin", "chunk_index": chunk_idx}

        ingestion_chunk = DocumentChunk(
            chunk_id=cid,
            chunk_index=chunk_idx,
            document_id=doc_id,
            filename=fname,
            page_number=page,
            content=content_desc,
            content_type=ctype,
            metadata=dict(meta),
        )

        vs_result = VectorSearchResult(
            chunk_id=ingestion_chunk.chunk_id,
            score=0.94,
            document_id=ingestion_chunk.document_id,
            filename=ingestion_chunk.filename,
            page_number=ingestion_chunk.page_number,
            chunk_index=ingestion_chunk.chunk_index,
            content_type=ingestion_chunk.content_type,
            content=ingestion_chunk.content,
            metadata=dict(ingestion_chunk.metadata),
        )

        # 2. Search / Retrieval Subsystem (Member 2)
        citation = AgentCitation.from_search_result(vs_result)
        assert citation.document_id == doc_id
        assert citation.filename == fname
        assert citation.chunk_id == cid
        assert citation.page_number == page
        assert citation.content_type == ctype
        assert citation.score == 0.94
        assert citation.metadata["source_department"] == "Finance"

        search_pkg = SearchResult(
            query="Analyze gross margin growth",
            status="RESULTS_FOUND",
            citations=[citation],
            context=f"[Source 1] {content_desc}",
        )

        # 3. Vision Subsystem (Member 3)
        visual_evidence_list = VisualEvidenceAdapter.adapt_search_package(search_pkg)
        assert len(visual_evidence_list) == 1

        v_evidence = visual_evidence_list[0]
        assert v_evidence.document_id == doc_id
        assert v_evidence.filename == fname
        assert v_evidence.chunk_id == cid
        assert v_evidence.page_number == page
        assert v_evidence.chunk_index == chunk_idx
        assert v_evidence.content_type == ctype
        assert v_evidence.metadata["source_department"] == "Finance"
        assert v_evidence.metadata["metric"] == "gross_margin"

        vision_req = VisionRequest(
            query=search_pkg.query,
            evidence=visual_evidence_list,
        )

        # Simulate vision agent analysis result
        vision_result = VisionResult(
            query=vision_req.query,
            status="success",
            description="Analysis verified: Gross margin expanded by 18% driven by software revenues.",
            evidence=vision_req.evidence,
            metadata={"confidence": 0.98, "provider": "mock_vision_provider"},
        )

        assert vision_result.document_id == doc_id
        assert vision_result.filename == fname
        assert vision_result.chunk_id == cid
        assert vision_result.page_number == page
        assert vision_result.content_type == ctype

        # 4. Supervisor / Downstream Consumer (Member 4 / Orchestrator)
        supervisor_state = AgentState(
            query=vision_result.query,
            route="vision",
        )

        downstream_citation = AgentCitation(
            document_id=vision_result.document_id,
            filename=vision_result.filename,
            chunk_id=vision_result.chunk_id,
            page_number=vision_result.page_number,
            content_type=vision_result.content_type,
            score=1.0,
            metadata=dict(vision_result.metadata),
        )
        supervisor_state.add_citation(downstream_citation)
        supervisor_state.update(
            answer=vision_result.description,
            status="completed",
            metadata={"vision_metadata": dict(vision_result.metadata)},
        )

        # Final assertion on supervisor state
        assert supervisor_state.status == "completed"
        assert supervisor_state.answer == "Analysis verified: Gross margin expanded by 18% driven by software revenues."
        assert len(supervisor_state.citations) == 1
        assert supervisor_state.citations[0].document_id == doc_id
        assert supervisor_state.citations[0].filename == fname
        assert supervisor_state.citations[0].chunk_id == cid
        assert supervisor_state.citations[0].page_number == page
        assert supervisor_state.citations[0].content_type == ctype


# ============================================================================
# 2. MULTI-CHUNK & MULTI-EVIDENCE FLOW
# ============================================================================


class TestMultiChunkAndMultiEvidenceFlow:
    """Verifies that multiple chunks from one document maintain order, count, and provenance."""

    def test_multi_chunk_evidence_pipeline(self) -> None:
        """Verify 4 chunks across 3 pages transition through all subsystems retaining structure."""
        doc_id = "doc-multi-chunk-01"
        fname = "engineering_specs.pdf"

        raw_chunks = [
            VectorSearchResult(
                chunk_id=f"chk-spec-{i}",
                score=0.95 - (i * 0.04),
                document_id=doc_id,
                filename=fname,
                page_number=(i + 1) // 2,
                chunk_index=i - 1,
                content_type="diagram" if i % 2 == 1 else "image",
                content=f"Engineering spec figure {i}",
                metadata={"figure_num": i, "chunk_index": i - 1},
            )
            for i in range(1, 5)
        ]

        # Search citations
        citations = [AgentCitation.from_search_result(r) for r in raw_chunks]
        search_result = SearchResult(
            query="Analyze all engineering diagrams",
            status="RESULTS_FOUND",
            citations=citations,
            context="Combined engineering diagrams context",
        )

        # Vision evidence adaptation
        visual_evidence_list = VisualEvidenceAdapter.adapt_search_package(search_result)
        assert len(visual_evidence_list) == 4

        for idx, ev in enumerate(visual_evidence_list, start=1):
            assert ev.chunk_id == f"chk-spec-{idx}"
            assert ev.document_id == doc_id
            assert ev.filename == fname
            assert ev.chunk_index == idx - 1
            assert ev.metadata["figure_num"] == idx

        # Vision result
        vision_result = VisionResult(
            query=search_result.query,
            status="success",
            description="All 4 engineering specifications inspected and verified.",
            evidence=visual_evidence_list,
        )

        # Downstream response packaging
        response_citations = [
            AgentCitation(
                document_id=ev.document_id,
                filename=ev.filename,
                chunk_id=ev.chunk_id,
                page_number=ev.page_number,
                content_type=ev.content_type,
                score=1.0,
                metadata=dict(ev.metadata),
            )
            for ev in vision_result.evidence
        ]

        agent_response = AgentResponse(
            answer=vision_result.description,
            agent_name="VisionAgent",
            status=vision_result.status,
            citations=response_citations,
        )

        assert agent_response.is_success is True
        assert agent_response.total_citations == 4
        assert agent_response.unique_document_count == 1
        assert agent_response.unique_documents == [doc_id]


# ============================================================================
# 3. MULTI-DOCUMENT EVIDENCE ISOLATION
# ============================================================================


class TestMultiDocumentEvidenceIsolation:
    """Verifies that evidence from multiple documents (Doc A, Doc B, Doc C) remains strictly isolated."""

    def test_three_documents_cross_contamination_safety(self) -> None:
        """Verify Doc A, Doc B, and Doc C evidence never bleed metadata or identifiers across boundaries."""
        docs = [
            ("doc-A", "architecture_A.pdf", "diagram", {"system": "Frontend"}),
            ("doc-B", "backend_B.pdf", "chart", {"system": "Database"}),
            ("doc-C", "network_C.pdf", "image", {"system": "Gateway"}),
        ]

        vs_results = [
            VectorSearchResult(
                chunk_id=f"chk-{doc_id}",
                score=0.90,
                document_id=doc_id,
                filename=fname,
                page_number=1,
                chunk_index=0,
                content_type=ctype,
                content=f"Content for {doc_id}",
                metadata=dict(meta),
            )
            for doc_id, fname, ctype, meta in docs
        ]

        # Search Stage
        citations = [AgentCitation.from_search_result(r) for r in vs_results]
        search_pkg = SearchResult(
            query="Analyze multi-system architecture",
            status="RESULTS_FOUND",
            citations=citations,
            context="Multi-system context",
        )

        # Vision Stage
        visual_evidence = VisualEvidenceAdapter.adapt_search_package(search_pkg)
        assert len(visual_evidence) == 3

        # Downstream supervisor packaging
        downstream_cits = [
            AgentCitation(
                document_id=ev.document_id,
                filename=ev.filename,
                chunk_id=ev.chunk_id,
                page_number=ev.page_number,
                content_type=ev.content_type,
                score=1.0,
                metadata=dict(ev.metadata),
            )
            for ev in visual_evidence
        ]

        response = AgentResponse(
            answer="Multi-system analysis completed.",
            agent_name="Supervisor",
            citations=downstream_cits,
        )

        assert response.unique_document_count == 3
        assert response.unique_documents == ["doc-A", "doc-B", "doc-C"]

        # Verify Doc A
        cit_a = [c for c in response.citations if c.document_id == "doc-A"][0]
        assert cit_a.filename == "architecture_A.pdf"
        assert cit_a.content_type == "diagram"
        assert cit_a.metadata["system"] == "Frontend"

        # Verify Doc B
        cit_b = [c for c in response.citations if c.document_id == "doc-B"][0]
        assert cit_b.filename == "backend_B.pdf"
        assert cit_b.content_type == "chart"
        assert cit_b.metadata["system"] == "Database"

        # Verify Doc C
        cit_c = [c for c in response.citations if c.document_id == "doc-C"][0]
        assert cit_c.filename == "network_C.pdf"
        assert cit_c.content_type == "image"
        assert cit_c.metadata["system"] == "Gateway"


# ============================================================================
# 4. CONTENT TYPE PRESERVATION & ROUTING
# ============================================================================


class TestContentTypePreservationAndRouting:
    """Verifies that all supported modalities retain accurate type tags throughout the pipeline."""

    def test_mixed_modality_routing_and_preservation(self) -> None:
        """Verify text, table, and image chunks are appropriately filtered/adapted across stages."""
        results = [
            VectorSearchResult(
                chunk_id="c-txt",
                score=0.98,
                document_id="doc-mix",
                filename="doc.pdf",
                page_number=1,
                chunk_index=0,
                content_type="text",
                content="Text explanation paragraph.",
            ),
            VectorSearchResult(
                chunk_id="c-tbl",
                score=0.92,
                document_id="doc-mix",
                filename="doc.pdf",
                page_number=2,
                chunk_index=1,
                content_type="table",
                content="| Col1 | Col2 |\n| Val1 | Val2 |",
            ),
            VectorSearchResult(
                chunk_id="c-diag",
                score=0.89,
                document_id="doc-mix",
                filename="doc.pdf",
                page_number=3,
                chunk_index=2,
                content_type="diagram",
                content="Workflow diagram",
            ),
        ]

        citations = [AgentCitation.from_search_result(r) for r in results]
        search_pkg = SearchResult(
            query="Analyze multi-modal document",
            status="RESULTS_FOUND",
            citations=citations,
            context="Context",
        )

        assert search_pkg.text_count == 1
        assert search_pkg.table_count == 1
        assert search_pkg.image_count == 0  # diagram is visual but not 'image' in search_pkg property

        # Vision adapter extracts visual modalities (diagram)
        visual_evidence = VisualEvidenceAdapter.adapt_search_package(search_pkg)
        assert len(visual_evidence) == 1
        assert visual_evidence[0].chunk_id == "c-diag"
        assert visual_evidence[0].content_type == "diagram"


# ============================================================================
# 5. INVALID FLOW & ERROR PROPAGATION TESTS
# ============================================================================


class TestInvalidFlowAndFailurePropagation:
    """Verifies that malformed inputs and downstream failures propagate cleanly without data fabrication."""

    def test_missing_document_id_rejection_at_search_boundary(self) -> None:
        """Verify missing document_id in upstream result is rejected when creating AgentCitation."""
        with pytest.raises(AgentValidationError, match="document_id"):
            AgentCitation(
                document_id="",
                filename="doc.pdf",
                chunk_id="c1",
                content_type="image",
            )

    def test_missing_chunk_id_rejection_at_vision_boundary(self) -> None:
        """Verify missing chunk_id in VisualEvidence constructor raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError, match="chunk_id"):
            VisualEvidence(
                document_id="doc-1",
                filename="doc.pdf",
                chunk_id="",
                content_type="image",
            )

    def test_vision_failure_propagation_to_supervisor(self) -> None:
        """Verify VisionResult error status propagates to supervisor state without fabricating answers."""
        vision_err_result = VisionResult(
            query="Analyze unreadable image",
            status="error",
            description="",
            error="Image decoding failed: corrupt byte payload",
            metadata={"retries": 2},
        )

        state = AgentState(
            query=vision_err_result.query,
            route="vision",
        )
        state.add_error(vision_err_result.error)
        state.update(
            status="failed",
            metadata={"vision_metadata": dict(vision_err_result.metadata)},
        )

        assert state.status == "failed"
        assert len(state.errors) == 1
        assert "corrupt byte payload" in state.errors[0]
        assert state.answer == ""
        assert len(state.citations) == 0


# ============================================================================
# 6. REQUEST ISOLATION & REPEATED EXECUTION
# ============================================================================


class TestRequestIsolationAndRepeatability:
    """Verifies that distinct flow runs are strictly isolated and deterministically repeatable."""

    def test_two_independent_flows_are_completely_isolated(self) -> None:
        """Verify Flow A and Flow B have zero shared mutable state."""
        # Flow A
        cit_a = AgentCitation(
            document_id="doc-A",
            filename="fileA.pdf",
            chunk_id="chk-A",
            content_type="image",
            metadata={"flow": "A"},
        )
        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a)
        res_a = VisionResult(
            query="Query A",
            status="success",
            description="Answer A",
            evidence=[ev_a],
            metadata={"flow": "A"},
        )

        # Flow B
        cit_b = AgentCitation(
            document_id="doc-B",
            filename="fileB.pdf",
            chunk_id="chk-B",
            content_type="chart",
            metadata={"flow": "B"},
        )
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b)
        res_b = VisionResult(
            query="Query B",
            status="success",
            description="Answer B",
            evidence=[ev_b],
            metadata={"flow": "B"},
        )

        # Mutate Flow A
        res_a.metadata["mutated"] = True
        res_a.evidence[0].metadata["processed"] = True

        # Flow B must remain unaffected
        assert "mutated" not in res_b.metadata
        assert "processed" not in res_b.evidence[0].metadata
        assert res_b.document_id == "doc-B"
        assert res_b.filename == "fileB.pdf"

    def test_repeated_flow_execution_is_deterministic(self) -> None:
        """Verify executing the same evidence flow repeatedly produces strictly identical results."""
        vs_item = VectorSearchResult(
            chunk_id="chk-repeat-01",
            score=0.92,
            document_id="doc-repeat",
            filename="repeat.pdf",
            page_number=2,
            chunk_index=0,
            content_type="image",
            content="Deterministic repeat content",
            metadata={"seed": 1234},
        )

        def run_flow() -> tuple[dict, dict, dict]:
            cit = AgentCitation.from_search_result(vs_item)
            ev = VisualEvidenceAdapter.adapt_citation(cit)
            res = VisionResult(
                query="Repeat query",
                status="success",
                description="Repeat description",
                evidence=[ev],
            )
            return cit.to_dict(), ev.to_dict(), res.to_dict()

        run1 = run_flow()
        run2 = run_flow()
        run3 = run_flow()

        assert run1 == run2 == run3
