"""
OmniBrain Member 4 — Day 4 Vision -> Supervisor Handoff Contract Integration Tests.

Verifies the existing handoff and data contracts between:
- Member 3 (Vision subsystem: VisualEvidence, VisionRequest, VisionResult)
- Downstream / Supervisor Consumer (AgentState, AgentResponse, AgentCitation)

Ensures that:
1. Valid VisionResult outputs are cleanly consumed by downstream supervisor state and response models.
2. Document identity, chunk identity, page lineage, content type, descriptions, and metadata are preserved.
3. Multi-evidence and multi-document results preserve exact ordering, count, and strict provenance isolation.
4. All supported public result states ('success', 'error', 'no_evidence', 'not_implemented') and error contracts are handled properly.
5. Roundtrip serialization via to_dict() and from_dict() preserves the complete contract.
6. Handoff operations are strictly immutable, request-isolated, deterministic, and offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path for test runners executing this file directly
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

from agents.models import AgentCitation, AgentResponse, AgentState
from vision.exceptions import VisionInputValidationError
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionResult,
    VisualEvidence,
)


# ============================================================================
# 1. SUCCESSFUL HANDOFF & LINEAGE INHERITANCE
# ============================================================================


class TestSuccessfulVisionToSupervisorHandoff:
    """Verifies that a successful VisionResult is correctly consumed by supervisor/downstream boundaries."""

    def test_single_evidence_lineage_preservation_to_agent_state(self) -> None:
        """Verify VisionResult with single visual evidence updates AgentState with exact lineage."""
        evidence = VisualEvidence(
            document_id="doc-sup-01",
            filename="quarterly_report.pdf",
            chunk_id="chunk-img-001",
            page_number=4,
            chunk_index=2,
            content_type="chart",
            image_path="B:/tmp/q3_chart.png",
            description="Q3 Revenue chart showing 15% increase",
            metadata={"source": "Finance Dept", "metric": "revenue"},
        )

        vision_result = VisionResult(
            query="Analyze Q3 revenue trends from the chart",
            status="success",
            description="Revenue grew consistently across all four quarters.",
            evidence=[evidence],
            metadata={"latency_ms": 120, "confidence": 0.96},
        )

        # Verify lineage inheritance on VisionResult
        assert vision_result.document_id == "doc-sup-01"
        assert vision_result.filename == "quarterly_report.pdf"
        assert vision_result.chunk_id == "chunk-img-001"
        assert vision_result.page_number == 4
        assert vision_result.content_type == "chart"
        assert vision_result.is_success is True
        assert vision_result.is_error is False

        # Downstream Supervisor consumption into AgentState
        state = AgentState(
            query=vision_result.query,
            route="vision",
        )

        citation = AgentCitation(
            document_id=vision_result.document_id,
            filename=vision_result.filename,
            chunk_id=vision_result.chunk_id,
            page_number=vision_result.page_number,
            content_type=vision_result.content_type,
            score=1.0,
            metadata=dict(vision_result.metadata),
        )
        state.add_citation(citation)
        state.update(
            answer=vision_result.description,
            status="completed",
            metadata={"vision_metadata": dict(vision_result.metadata)},
        )

        assert state.status == "completed"
        assert state.answer == "Revenue grew consistently across all four quarters."
        assert len(state.citations) == 1
        assert state.citations[0].document_id == "doc-sup-01"
        assert state.citations[0].filename == "quarterly_report.pdf"
        assert state.citations[0].chunk_id == "chunk-img-001"
        assert state.citations[0].page_number == 4
        assert state.citations[0].content_type == "chart"
        assert state.citations[0].metadata["confidence"] == 0.96

    def test_vision_result_to_agent_response_package(self) -> None:
        """Verify constructing an AgentResponse from VisionResult preserving citations and status."""
        evidence = VisualEvidence(
            document_id="doc-arch-002",
            filename="system_architecture.pdf",
            chunk_id="chunk-arch-99",
            page_number=1,
            content_type="diagram",
            image_path="B:/tmp/arch.png",
        )

        vision_result = VisionResult(
            query="Explain system architecture",
            status="success",
            description="The architecture depicts a 4-member modular RAG system.",
            evidence=[evidence],
            metadata={"model": "offline-vision-provider"},
        )

        citation = AgentCitation(
            document_id=vision_result.document_id,
            filename=vision_result.filename,
            chunk_id=vision_result.chunk_id,
            page_number=vision_result.page_number,
            content_type=vision_result.content_type,
            score=1.0,
            metadata={"provider": "offline-vision-provider"},
        )

        response = AgentResponse(
            answer=vision_result.description,
            agent_name="VisionAgent",
            status=vision_result.status,
            citations=[citation],
            metadata=dict(vision_result.metadata),
        )

        assert response.is_success is True
        assert response.agent_name == "VisionAgent"
        assert response.answer == "The architecture depicts a 4-member modular RAG system."
        assert response.total_citations == 1
        assert response.citations[0].chunk_id == "chunk-arch-99"
        assert response.citations[0].content_type == "diagram"


# ============================================================================
# 2. MULTI-EVIDENCE HANDOFF
# ============================================================================


class TestMultiEvidenceSupervisorHandoff:
    """Verifies that multiple evidence items in a VisionResult retain ordering and distinct provenance."""

    def test_multi_evidence_ordering_and_citations(self) -> None:
        """Verify downstream conversion of multiple evidence items preserves count and sequence."""
        evidence_list = [
            VisualEvidence(
                document_id="doc-multi-ev",
                filename="analysis.pdf",
                chunk_id=f"c-img-{i}",
                page_number=i,
                chunk_index=i - 1,
                content_type="image" if i % 2 == 1 else "chart",
                metadata={"rank": i},
            )
            for i in range(1, 6)
        ]

        vision_result = VisionResult(
            query="Summarize all 5 visual figures",
            status="success",
            description="All 5 figures analyzed successfully.",
            evidence=evidence_list,
        )

        assert vision_result.has_evidence is True
        assert len(vision_result.evidence) == 5

        # Primary lineage inherited from the first item
        assert vision_result.document_id == "doc-multi-ev"
        assert vision_result.chunk_id == "c-img-1"
        assert vision_result.page_number == 1

        # Downstream supervisor converts all evidence to citations
        citations = [
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

        state = AgentState(query=vision_result.query)
        for c in citations:
            state.add_citation(c)

        assert len(state.citations) == 5
        for idx, c in enumerate(state.citations, start=1):
            assert c.chunk_id == f"c-img-{idx}"
            assert c.page_number == idx
            assert c.metadata["rank"] == idx


# ============================================================================
# 3. MULTI-DOCUMENT HANDOFF
# ============================================================================


class TestMultiDocumentSupervisorHandoff:
    """Verifies that evidence spanning multiple documents maintains document isolation downstream."""

    def test_multi_document_lineage_segregation(self) -> None:
        """Verify evidence from Doc A and Doc B retain separate document IDs and filenames."""
        ev_a = VisualEvidence(
            document_id="doc-A-uuid",
            filename="policy_A.pdf",
            chunk_id="chunk-A-01",
            page_number=2,
            content_type="diagram",
            metadata={"project": "Alpha"},
        )
        ev_b = VisualEvidence(
            document_id="doc-B-uuid",
            filename="financials_B.pdf",
            chunk_id="chunk-B-05",
            page_number=8,
            content_type="chart",
            metadata={"project": "Beta"},
        )

        vision_result = VisionResult(
            query="Compare diagram from Doc A and chart from Doc B",
            status="success",
            description="Comparison between Alpha workflow and Beta financials.",
            evidence=[ev_a, ev_b],
        )

        citations = [
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

        response = AgentResponse(
            answer=vision_result.description,
            agent_name="VisionAgent",
            citations=citations,
        )

        assert response.total_citations == 2
        assert response.unique_document_count == 2
        assert response.unique_documents == ["doc-A-uuid", "doc-B-uuid"]

        doc_a_cits = [c for c in response.citations if c.document_id == "doc-A-uuid"]
        doc_b_cits = [c for c in response.citations if c.document_id == "doc-B-uuid"]

        assert len(doc_a_cits) == 1
        assert doc_a_cits[0].filename == "policy_A.pdf"
        assert doc_a_cits[0].metadata["project"] == "Alpha"

        assert len(doc_b_cits) == 1
        assert doc_b_cits[0].filename == "financials_B.pdf"
        assert doc_b_cits[0].metadata["project"] == "Beta"


# ============================================================================
# 4. RESULT STATUS & ERROR CONTRACT TESTING
# ============================================================================


class TestResultStatusAndErrorContracts:
    """Verifies that public result states and error messages are faithfully represented downstream."""

    def test_error_status_representation_and_downstream_handling(self) -> None:
        """Verify error VisionResult propagates error details to AgentState without crash."""
        vision_result = VisionResult(
            query="Analyze corrupted image",
            status="error",
            description="",
            error="Vision provider timed out after 30s.",
            metadata={"retry_count": 3},
        )

        assert vision_result.is_success is False
        assert vision_result.is_error is True
        assert vision_result.error == "Vision provider timed out after 30s."

        state = AgentState(query=vision_result.query, route="vision")
        state.add_error(vision_result.error)
        state.update(
            status="failed",
            metadata={"vision_metadata": dict(vision_result.metadata)},
        )

        assert state.status == "failed"
        assert len(state.errors) == 1
        assert "timed out" in state.errors[0]

    def test_no_evidence_status_representation(self) -> None:
        """Verify no_evidence status is cleanly handled by downstream agent response."""
        vision_result = VisionResult(
            query="Find chart on text-only document",
            status="no_evidence",
            description="No visual evidence found for the requested query.",
            evidence=[],
        )

        assert vision_result.status == "no_evidence"
        assert vision_result.has_evidence is False

        response = AgentResponse(
            answer=vision_result.description,
            agent_name="VisionAgent",
            status=vision_result.status,
            citations=[],
        )

        assert response.status == "no_evidence"
        assert response.has_citations is False
        assert "No visual evidence" in response.answer

    def test_not_implemented_status_representation(self) -> None:
        """Verify not_implemented status is cleanly represented."""
        vision_result = VisionResult(
            query="Process 3D video stream",
            status="not_implemented",
            description="Video modalities not supported.",
        )

        assert vision_result.status == "not_implemented"
        assert vision_result.is_success is False


# ============================================================================
# 5. SERIALIZATION & ROUNDTRIP
# ============================================================================


class TestSerializationRoundtrip:
    """Verifies that to_dict() and from_dict() roundtrip preserves all fields for supervisor transport."""

    def test_vision_result_roundtrip_serialization(self) -> None:
        """Verify complete dictionary serialization and deserialization."""
        evidence = [
            VisualEvidence(
                document_id="doc-roundtrip",
                filename="roundtrip.pdf",
                chunk_id="chunk-rt-1",
                page_number=3,
                chunk_index=1,
                content_type="diagram",
                image_path="B:/tmp/rt.png",
                image_format="png",
                width=1024,
                height=768,
                description="Roundtrip diagram",
                metadata={"key1": "val1", "tag": 99},
            )
        ]

        original = VisionResult(
            query="Verify roundtrip serialization",
            status="success",
            description="Roundtrip verified successfully.",
            evidence=evidence,
            metadata={"pipeline": "standard", "trace_id": "trace-123"},
        )

        data = original.to_dict()
        assert isinstance(data, dict)
        assert data["query"] == "Verify roundtrip serialization"
        assert data["status"] == "success"
        assert data["document_id"] == "doc-roundtrip"
        assert len(data["evidence"]) == 1

        restored = VisionResult.from_dict(data)
        assert restored.query == original.query
        assert restored.status == original.status
        assert restored.description == original.description
        assert restored.document_id == original.document_id
        assert restored.filename == original.filename
        assert restored.page_number == original.page_number
        assert restored.chunk_id == original.chunk_id
        assert restored.content_type == original.content_type
        assert restored.metadata == original.metadata
        assert len(restored.evidence) == 1
        assert restored.evidence[0].document_id == "doc-roundtrip"
        assert restored.evidence[0].image_path == "B:/tmp/rt.png"


# ============================================================================
# 6. IMMUTABILITY & MUTATION SAFETY
# ============================================================================


class TestImmutabilitySafety:
    """Verifies that downstream consumption does not alter the upstream VisionResult."""

    def test_vision_result_remains_unchanged_during_downstream_use(self) -> None:
        """Verify mutating downstream AgentState metadata does not affect VisionResult."""
        evidence = VisualEvidence(
            document_id="doc-imm-01",
            filename="imm.pdf",
            chunk_id="c-imm-1",
            content_type="image",
        )
        vision_result = VisionResult(
            query="Immutable query",
            status="success",
            description="Original description",
            evidence=[evidence],
            metadata={"original_key": "original_val"},
        )

        original_dict = vision_result.to_dict()

        # Downstream supervisor takes metadata and modifies its local copy
        state = AgentState(
            query=vision_result.query,
            metadata=dict(vision_result.metadata),
        )
        state.metadata["original_key"] = "mutated_in_state"
        state.metadata["new_supervisor_field"] = "added"

        # VisionResult must remain unaltered
        assert vision_result.to_dict() == original_dict
        assert vision_result.metadata["original_key"] == "original_val"
        assert "new_supervisor_field" not in vision_result.metadata


# ============================================================================
# 7. REQUEST ISOLATION
# ============================================================================


class TestRequestIsolation:
    """Verifies that independent VisionResult instances do not share state."""

    def test_independent_vision_results_isolated(self) -> None:
        """Verify Request A and Request B results are strictly decoupled."""
        res_a = VisionResult(
            query="Query A",
            status="success",
            description="Description A",
            evidence=[VisualEvidence(document_id="doc-A", filename="fA.pdf", chunk_id="cA", content_type="image")],
            metadata={"req": "A"},
        )
        res_b = VisionResult(
            query="Query B",
            status="success",
            description="Description B",
            evidence=[VisualEvidence(document_id="doc-B", filename="fB.pdf", chunk_id="cB", content_type="chart")],
            metadata={"req": "B"},
        )

        res_a.metadata["extra"] = "alpha_only"
        res_a.evidence[0].metadata["processed"] = True

        assert "extra" not in res_b.metadata
        assert "processed" not in res_b.evidence[0].metadata
        assert res_b.document_id == "doc-B"
        assert res_b.query == "Query B"
