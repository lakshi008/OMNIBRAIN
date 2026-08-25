"""
OmniBrain Member 4 — Day 3 Search -> Vision Contract Integration Tests.

Verifies the existing handoff and data contracts between:
- Member 2 (Search subsystem: AgentCitation, SearchResult, AgentResponse)
- Member 3 (Vision subsystem: VisualEvidence, VisualEvidenceAdapter, VisionRequest, VisionResult)

Ensures that:
1. Valid Search citations and result packages correctly adapt into VisualEvidence and VisionRequests.
2. Complete document identity, chunk identity, page lineage, content type, descriptions, and metadata are preserved.
3. Multi-evidence and multi-document sets maintain exact ordering, count, and strict isolation without cross-document leakage.
4. All supported visual content types ('image', 'chart', 'diagram') are preserved, while non-visual types are filtered or rejected according to contract.
5. Malformed or invalid evidence triggers expected validation and domain error boundaries.
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

from agents.models import AgentCitation, AgentResponse, SearchResult
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import VisionEvidenceError, VisionInputValidationError
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)


# ============================================================================
# 1. SINGLE EVIDENCE HANDOFF TEST
# ============================================================================


class TestSingleEvidenceHandoff:
    """Verifies that a single Search AgentCitation adapts to VisualEvidence preserving all lineage."""

    def test_single_image_citation_to_visual_evidence(self) -> None:
        """Verify single visual citation adapts with document_id, chunk_id, page_number, and metadata."""
        citation = AgentCitation(
            document_id="doc-img-101",
            filename="quarterly_overview.pdf",
            chunk_id="chunk-img-001",
            page_number=4,
            content_type="image",
            score=0.94,
            metadata={
                "chunk_index": 2,
                "image_path": "B:/tmp/overview_chart.png",
                "image_format": "png",
                "caption": "Quarterly overview plot",
                "author": "Analytics Team",
            },
        )

        assert VisualEvidenceAdapter.is_visual(citation) is True
        evidence = VisualEvidenceAdapter.adapt_citation(citation)

        assert isinstance(evidence, VisualEvidence)
        assert evidence.document_id == "doc-img-101"
        assert evidence.filename == "quarterly_overview.pdf"
        assert evidence.chunk_id == "chunk-img-001"
        assert evidence.page_number == 4
        assert evidence.chunk_index == 2
        assert evidence.content_type == "image"
        assert evidence.image_path == "B:/tmp/overview_chart.png"
        assert evidence.image_format == "png"
        assert evidence.metadata["caption"] == "Quarterly overview plot"
        assert evidence.metadata["author"] == "Analytics Team"

    def test_single_diagram_citation_with_override_arguments(self) -> None:
        """Verify adapt_citation supports explicit image payload and format overrides."""
        citation = AgentCitation(
            document_id="doc-diag-202",
            filename="system_architecture.pdf",
            chunk_id="chunk-diag-042",
            page_number=12,
            content_type="diagram",
            score=0.89,
            metadata={"chunk_index": 5},
        )

        dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        evidence = VisualEvidenceAdapter.adapt_citation(
            citation,
            image_path="B:/tmp/diag_override.png",
            image_bytes=dummy_bytes,
            image_format="png",
        )

        assert evidence.document_id == "doc-diag-202"
        assert evidence.filename == "system_architecture.pdf"
        assert evidence.chunk_id == "chunk-diag-042"
        assert evidence.page_number == 12
        assert evidence.chunk_index == 5
        assert evidence.content_type == "diagram"
        assert evidence.image_path == "B:/tmp/diag_override.png"
        assert evidence.image_bytes == dummy_bytes
        assert evidence.image_format == "png"


# ============================================================================
# 2. MULTI-EVIDENCE ORDER & COUNT PRESERVATION
# ============================================================================


class TestMultiEvidenceHandoff:
    """Verifies that multiple evidence items maintain exact count, ordering, and attribution."""

    def test_multi_evidence_adaptation_order_and_count(self) -> None:
        """Verify adapt_search_package extracts all visual citations in rank order."""
        citations = [
            AgentCitation(
                document_id="doc-multi",
                filename="report.pdf",
                chunk_id=f"c-img-{i}",
                page_number=i,
                content_type="image" if i % 2 == 1 else "chart",
                score=0.95 - (i * 0.05),
                metadata={"chunk_index": i - 1, "rank": i},
            )
            for i in range(1, 6)
        ]

        search_pkg = SearchResult(
            query="Extract all diagrams and charts",
            status="RESULTS_FOUND",
            citations=citations,
            context="Visual summary",
        )

        visual_evidence_list = VisualEvidenceAdapter.adapt_search_package(search_pkg)

        assert len(visual_evidence_list) == 5
        for i, ev in enumerate(visual_evidence_list, start=1):
            assert ev.chunk_id == f"c-img-{i}"
            assert ev.page_number == i
            assert ev.chunk_index == i - 1
            assert ev.metadata["rank"] == i
            if i % 2 == 1:
                assert ev.content_type == "image"
            else:
                assert ev.content_type == "chart"

    def test_mixed_modality_search_package_filtering(self) -> None:
        """Verify non-visual citations (text, table) are safely filtered out in standard mode."""
        citations = [
            AgentCitation(
                document_id="doc-mix",
                filename="f.pdf",
                chunk_id="c-text-1",
                page_number=1,
                content_type="text",
                score=0.98,
            ),
            AgentCitation(
                document_id="doc-mix",
                filename="f.pdf",
                chunk_id="c-img-1",
                page_number=2,
                content_type="image",
                score=0.92,
            ),
            AgentCitation(
                document_id="doc-mix",
                filename="f.pdf",
                chunk_id="c-tbl-1",
                page_number=3,
                content_type="table",
                score=0.88,
            ),
            AgentCitation(
                document_id="doc-mix",
                filename="f.pdf",
                chunk_id="c-chart-1",
                page_number=4,
                content_type="chart",
                score=0.84,
            ),
        ]

        response_pkg = AgentResponse(
            answer="Multi-modal answer summary",
            agent_name="SearchAgent",
            status="success",
            citations=citations,
        )

        adapted = VisualEvidenceAdapter.adapt_search_package(response_pkg, strict=False)
        assert len(adapted) == 2
        assert adapted[0].chunk_id == "c-img-1"
        assert adapted[0].content_type == "image"
        assert adapted[1].chunk_id == "c-chart-1"
        assert adapted[1].content_type == "chart"


# ============================================================================
# 3. MULTI-DOCUMENT ISOLATION
# ============================================================================


class TestMultiDocumentEvidenceIsolation:
    """Verifies that evidence from multiple documents does not cross-contaminate."""

    def test_multi_document_lineage_and_metadata_isolation(self) -> None:
        """Verify Document A and Document B visual evidence retain distinct provenance and metadata."""
        doc_a_cit = AgentCitation(
            document_id="doc-A-unique-uuid",
            filename="document_A.pdf",
            chunk_id="chunk-A-01",
            page_number=3,
            content_type="chart",
            score=0.95,
            metadata={"source_project": "Alpha", "sensitivity": "restricted"},
        )

        doc_b_cit = AgentCitation(
            document_id="doc-B-unique-uuid",
            filename="document_B.pdf",
            chunk_id="chunk-B-99",
            page_number=7,
            content_type="diagram",
            score=0.87,
            metadata={"source_project": "Beta", "sensitivity": "public"},
        )

        ev_a = VisualEvidenceAdapter.adapt_citation(doc_a_cit)
        ev_b = VisualEvidenceAdapter.adapt_citation(doc_b_cit)

        # Verify Doc A
        assert ev_a.document_id == "doc-A-unique-uuid"
        assert ev_a.filename == "document_A.pdf"
        assert ev_a.chunk_id == "chunk-A-01"
        assert ev_a.page_number == 3
        assert ev_a.content_type == "chart"
        assert ev_a.metadata["source_project"] == "Alpha"
        assert ev_a.metadata["sensitivity"] == "restricted"

        # Verify Doc B
        assert ev_b.document_id == "doc-B-unique-uuid"
        assert ev_b.filename == "document_B.pdf"
        assert ev_b.chunk_id == "chunk-B-99"
        assert ev_b.page_number == 7
        assert ev_b.content_type == "diagram"
        assert ev_b.metadata["source_project"] == "Beta"
        assert ev_b.metadata["sensitivity"] == "public"

        # Verify zero cross-talk
        assert ev_a.document_id != ev_b.document_id
        assert ev_a.filename != ev_b.filename
        assert ev_a.chunk_id != ev_b.chunk_id
        assert ev_a.metadata["source_project"] != ev_b.metadata["source_project"]


# ============================================================================
# 4. CONTENT TYPE VERIFICATION
# ============================================================================


class TestContentTypePreservation:
    """Verifies that all supported visual modalities are preserved and non-visual types are classified correctly."""

    @pytest.mark.parametrize("content_type", ["image", "chart", "diagram", "IMAGE", "Chart", "DIAGRAM"])
    def test_all_supported_visual_content_types(self, content_type: str) -> None:
        """Verify image, chart, and diagram modalities (case-insensitive) adapt correctly."""
        citation = AgentCitation(
            document_id="doc-type-test",
            filename="modalities.pdf",
            chunk_id=f"chk-{content_type.lower()}",
            page_number=1,
            content_type=content_type,
            score=0.90,
        )

        assert VisualEvidenceAdapter.is_visual(citation) is True
        ev = VisualEvidenceAdapter.adapt_citation(citation)
        assert ev.content_type == content_type.strip().lower()
        assert ev.content_type in VALID_VISUAL_CONTENT_TYPES

    @pytest.mark.parametrize("non_visual_type", ["text", "table", "audio", "video", "code"])
    def test_non_visual_content_types_identified_and_rejected(self, non_visual_type: str) -> None:
        """Verify non-visual content types return False on is_visual and raise VisionEvidenceError on adapt_citation."""
        citation = AgentCitation(
            document_id="doc-non-visual",
            filename="text.pdf",
            chunk_id="chk-non-vis",
            page_number=1,
            content_type=non_visual_type,
            score=0.85,
        )

        assert VisualEvidenceAdapter.is_visual(citation) is False
        with pytest.raises(VisionEvidenceError, match="Unsupported content_type"):
            VisualEvidenceAdapter.adapt_citation(citation)


# ============================================================================
# 5. INVALID EVIDENCE & ERROR BOUNDARY TESTS
# ============================================================================


class TestInvalidEvidenceHandling:
    """Verifies that malformed or invalid search evidence is rejected cleanly at the Vision boundary."""

    def test_adapt_citation_rejects_none(self) -> None:
        """Verify adapt_citation raises VisionInputValidationError when citation is None."""
        with pytest.raises(VisionInputValidationError, match="Citation cannot be None"):
            VisualEvidenceAdapter.adapt_citation(None)  # type: ignore[arg-type]

    def test_adapt_citation_rejects_invalid_type(self) -> None:
        """Verify adapt_citation raises VisionInputValidationError on unexpected object types."""
        with pytest.raises(VisionInputValidationError, match="Expected AgentCitation instance"):
            VisualEvidenceAdapter.adapt_citation("not_a_citation")  # type: ignore[arg-type]

    def test_strict_mode_rejects_non_visual_in_package(self) -> None:
        """Verify adapt_search_package in strict=True mode raises on non-visual citations."""
        pkg = SearchResult(
            query="Strict check",
            status="RESULTS_FOUND",
            citations=[
                AgentCitation(
                    document_id="doc1",
                    filename="f.pdf",
                    chunk_id="c1",
                    page_number=1,
                    content_type="text",  # Non-visual
                )
            ],
            context="context",
        )
        with pytest.raises(VisionEvidenceError, match="has non-visual content_type"):
            VisualEvidenceAdapter.adapt_search_package(pkg, strict=True)

    def test_vision_request_validation_rejection(self) -> None:
        """Verify VisionRequest rejects empty query or malformed evidence list."""
        valid_ev = VisualEvidence(
            document_id="doc1",
            filename="f.pdf",
            chunk_id="c1",
            content_type="image",
        )

        # Empty query
        with pytest.raises(VisionInputValidationError, match="query cannot be empty"):
            VisionRequest(query="   ", evidence=[valid_ev])

        # Non-list evidence
        with pytest.raises(VisionInputValidationError, match="evidence must be a list"):
            VisionRequest(query="Valid query", evidence="not_a_list")  # type: ignore[arg-type]

        # List with non-VisualEvidence item
        with pytest.raises(VisionInputValidationError, match="not a VisualEvidence instance"):
            VisionRequest(query="Valid query", evidence=[valid_ev, "invalid_item"])  # type: ignore[list-item]


# ============================================================================
# 6. IMMUTABILITY & MUTATION SAFETY
# ============================================================================


class TestImmutabilityAndMutationSafety:
    """Verifies that adapting Search evidence to Vision never mutates original Search objects."""

    def test_citation_remains_unchanged_after_adaptation(self) -> None:
        """Verify original AgentCitation is immutable and unaffected by VisualEvidence operations."""
        original_meta = {"author": "Original Author", "tags": ["tag1", "tag2"]}
        citation = AgentCitation(
            document_id="doc-imm-01",
            filename="original.pdf",
            chunk_id="chunk-imm-01",
            page_number=5,
            content_type="image",
            score=0.91,
            metadata=dict(original_meta),
        )

        # Snapshot representation before adaptation
        before_dict = citation.to_dict()

        evidence = VisualEvidenceAdapter.adapt_citation(citation)

        # Mutate the resulting VisualEvidence
        evidence.metadata["author"] = "Mutated In Evidence"
        evidence.metadata["new_key"] = "added_value"

        # Original citation must remain completely unchanged
        assert citation.to_dict() == before_dict
        assert citation.metadata["author"] == "Original Author"
        assert "new_key" not in citation.metadata

    def test_search_package_immutability(self) -> None:
        """Verify SearchResult package and citations are not modified by adapt_search_package."""
        citations = [
            AgentCitation(
                document_id="doc-pkg-imm",
                filename="pkg.pdf",
                chunk_id="c-pkg-1",
                page_number=1,
                content_type="image",
                score=0.95,
                metadata={"key": "initial"},
            )
        ]
        pkg = SearchResult(
            query="Analyze chart",
            status="RESULTS_FOUND",
            citations=citations,
            context="initial context",
            metadata={"engine": "qdrant"},
        )

        before_pkg_dict = pkg.to_dict()
        adapted = VisualEvidenceAdapter.adapt_search_package(pkg)
        adapted[0].metadata["key"] = "tampered"

        assert pkg.to_dict() == before_pkg_dict
        assert pkg.citations[0].metadata["key"] == "initial"


# ============================================================================
# 7. REQUEST ISOLATION
# ============================================================================


class TestRequestIsolation:
    """Verifies that separate Search -> Vision handoffs maintain complete runtime isolation."""

    def test_two_independent_requests_do_not_share_state(self) -> None:
        """Verify two independent VisionRequest instances have isolated evidence and metadata."""
        cit_a = AgentCitation(
            document_id="doc-req-A",
            filename="file_A.pdf",
            chunk_id="chk-A",
            page_number=1,
            content_type="image",
            metadata={"session": "session_A"},
        )
        cit_b = AgentCitation(
            document_id="doc-req-B",
            filename="file_B.pdf",
            chunk_id="chk-B",
            page_number=2,
            content_type="diagram",
            metadata={"session": "session_B"},
        )

        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a)
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b)

        req_a = VisionRequest(
            query="Explain diagram A",
            evidence=[ev_a],
            session_id="sess-A",
            metadata={"request_id": "req-1"},
        )

        req_b = VisionRequest(
            query="Explain diagram B",
            evidence=[ev_b],
            session_id="sess-B",
            metadata={"request_id": "req-2"},
        )

        # Mutate Request A metadata and evidence
        req_a.metadata["extra"] = "alpha_only"
        req_a.evidence[0].metadata["processed"] = True

        # Request B must be unaffected
        assert "extra" not in req_b.metadata
        assert "processed" not in req_b.evidence[0].metadata
        assert req_b.session_id == "sess-B"
        assert req_b.evidence[0].document_id == "doc-req-B"
        assert req_b.evidence[0].chunk_id == "chk-B"
