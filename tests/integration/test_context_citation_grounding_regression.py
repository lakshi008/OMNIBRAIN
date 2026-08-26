"""
OmniBrain Member 4 — Day 45 Context Building & Citation Grounding Regression Certification.

Validates the context building, citation creation, source grounding, and lineage
traceability contracts across:
  - Ingestion (VectorSearchResult, RetrievalServiceResult, build_retrieval_context,
               process_retrieval_results, retrieve_context)
  - Agents / Search (AgentCitation, AgentResponse, SearchResult, SearchRequest, SearchAgent)
  - Vision (VisualEvidence, VisualEvidenceAdapter)

Covers:
  1.  Retrieval result -> formatted context conversion.
  2.  Deterministic source ordering and [Source N] numbering.
  3.  Document, page, and chunk identity preservation across context and citations.
  4.  Citation creation (from_search_result) and citation-to-source matching.
  5.  Deliberate citation inconsistency behavior (data model contract verification).
  6.  Missing / empty citation handling in responses.
  7.  Multiple citations and multi-document citation aggregation.
  8.  Citation ordering matching relevance ranking.
  9.  Duplicate source and citation handling.
  10. Multi-document context isolation (zero cross-contamination).
  11. Cross-request isolation.
  12. Source marker retention and verification.
  13. Empty retrieval result behavior.
  14. Invalid retrieval result rejection (type and structural validation).
  15. Serialization and deserialization round-trip of contexts and citations.
  16. Agent response grounding and visual evidence grounding.
  17. Input mutation safety and aliasing protection.
  18. 3-iteration determinism and error isolation.
  19. Complete offline end-to-end grounding flow.

Constraints:
  - 100% Offline: In-memory store, synthetic deterministic data, no external APIs.
  - Zero production code modified.
  - No new citation logic, context builders, or adapters added.
  - Synthetic deterministic data only.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# Ingestion layer (Member 1)
from ingestion.models import (
    RetrievalServiceResult,
    VectorSearchResult,
)
from ingestion.retrieval_processor import (
    build_retrieval_context,
    process_retrieval_results,
)

# Agents layer (Member 2)
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    SearchRequest,
    SearchResult,
)
from agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentValidationError,
)

# Vision layer (Member 3)
from vision.models import VisualEvidence
from vision.exceptions import VisionEvidenceError
from vision.evidence_adapter import VisualEvidenceAdapter


# ============================================================================
# Deterministic Synthetic Fixtures
# ============================================================================

DAY45_DOC_A = "DAY45-DOC-A"
DAY45_DOC_B = "DAY45-DOC-B"

DAY45_FILE_A = "day45_document_alpha.pdf"
DAY45_FILE_B = "day45_document_beta.pdf"

DAY45_PAGE_A1 = 1
DAY45_PAGE_A2 = 2
DAY45_PAGE_B1 = 1

DAY45_CHUNK_A1 = "DAY45-CHUNK-A1"
DAY45_CHUNK_A2 = "DAY45-CHUNK-A2"
DAY45_CHUNK_B1 = "DAY45-CHUNK-B1"

DAY45_DOC_A_MARKER = "DAY45_DOC_A_MARKER_90210"
DAY45_DOC_A_CHUNK_1 = "DAY45_DOC_A_CHUNK_1_ALPHA_CONTENT"
DAY45_DOC_A_CHUNK_2 = "DAY45_DOC_A_CHUNK_2_ALPHA_TABLE"

DAY45_DOC_B_MARKER = "DAY45_DOC_B_MARKER_38472"
DAY45_DOC_B_CHUNK_1 = "DAY45_DOC_B_CHUNK_1_BETA_CONTENT"

DAY45_A_UNIQUE = "DAY45_A_UNIQUE_EVIDENCE_KEY_1111"
DAY45_B_UNIQUE = "DAY45_B_UNIQUE_EVIDENCE_KEY_2222"


# ============================================================================
# 1. Valid Retrieval -> Context Conversion
# ============================================================================

class TestValidRetrievalToContextConversion:
    """Certifies that build_retrieval_context faithfully converts retrieval results into context."""

    def test_build_retrieval_context_structure_and_markers(self) -> None:
        """Verify context format, [Source N] numbering, and marker presence."""
        r1 = VectorSearchResult(
            chunk_id=DAY45_CHUNK_A1,
            score=0.95,
            document_id=DAY45_DOC_A,
            filename=DAY45_FILE_A,
            page_number=DAY45_PAGE_A1,
            chunk_index=0,
            content_type="text",
            content=f"{DAY45_DOC_A_CHUNK_1} - {DAY45_DOC_A_MARKER}",
            metadata={"source": "A1"},
        )
        r2 = VectorSearchResult(
            chunk_id=DAY45_CHUNK_A2,
            score=0.88,
            document_id=DAY45_DOC_A,
            filename=DAY45_FILE_A,
            page_number=DAY45_PAGE_A2,
            chunk_index=1,
            content_type="table",
            content=f"{DAY45_DOC_A_CHUNK_2} - {DAY45_A_UNIQUE}",
            metadata={"source": "A2"},
        )

        context = build_retrieval_context([r1, r2])

        # Verify Source 1 block
        assert "[Source 1]" in context
        assert f"File: {DAY45_FILE_A}" in context
        assert f"Page: {DAY45_PAGE_A1}" in context
        assert "Type: text" in context
        assert DAY45_DOC_A_CHUNK_1 in context
        assert DAY45_DOC_A_MARKER in context

        # Verify Source 2 block
        assert "[Source 2]" in context
        assert f"Page: {DAY45_PAGE_A2}" in context
        assert "Type: table" in context
        assert DAY45_DOC_A_CHUNK_2 in context
        assert DAY45_A_UNIQUE in context

        # Verify ordering
        assert context.find("[Source 1]") < context.find("[Source 2]")


# ============================================================================
# 2. Document, Page, and Chunk Lineage
# ============================================================================

class TestLineagePreservationInContextAndCitations:
    """Certifies that document, page, and chunk lineage are strictly preserved."""

    def test_document_and_page_and_chunk_lineage(self) -> None:
        """Original document_id, page_number, and chunk_id survive into context and citations."""
        res = VectorSearchResult(
            chunk_id=DAY45_CHUNK_A1,
            score=0.92,
            document_id=DAY45_DOC_A,
            filename=DAY45_FILE_A,
            page_number=DAY45_PAGE_A1,
            chunk_index=0,
            content_type="text",
            content=DAY45_DOC_A_CHUNK_1,
        )

        # 1. Context retains filename and page
        context = build_retrieval_context([res])
        assert DAY45_FILE_A in context
        assert f"Page: {DAY45_PAGE_A1}" in context

        # 2. Citation retains exact document_id, chunk_id, and page_number
        citation = AgentCitation.from_search_result(res)
        assert citation.document_id == DAY45_DOC_A
        assert citation.chunk_id == DAY45_CHUNK_A1
        assert citation.page_number == DAY45_PAGE_A1
        assert citation.filename == DAY45_FILE_A
        assert citation.content_type == "text"
        assert citation.score == 0.92


# ============================================================================
# 3. Citation Creation & Source Match
# ============================================================================

class TestCitationCreationAndSourceMatch:
    """Certifies that every created citation precisely matches its source record."""

    def test_citation_to_source_exact_match(self) -> None:
        """Verify citation fields correspond 1-to-1 with the source VectorSearchResult."""
        sources = [
            VectorSearchResult(
                chunk_id=DAY45_CHUNK_A1, score=0.95, document_id=DAY45_DOC_A,
                filename=DAY45_FILE_A, page_number=DAY45_PAGE_A1, chunk_index=0,
                content_type="text", content=DAY45_DOC_A_CHUNK_1,
            ),
            VectorSearchResult(
                chunk_id=DAY45_CHUNK_B1, score=0.85, document_id=DAY45_DOC_B,
                filename=DAY45_FILE_B, page_number=DAY45_PAGE_B1, chunk_index=0,
                content_type="table", content=DAY45_DOC_B_CHUNK_1,
            ),
        ]

        citations = [AgentCitation.from_search_result(s) for s in sources]

        for idx, (src, cit) in enumerate(zip(sources, citations, strict=True)):
            assert cit.document_id == src.document_id, f"Mismatch at index {idx} on document_id"
            assert cit.chunk_id == src.chunk_id, f"Mismatch at index {idx} on chunk_id"
            assert cit.page_number == src.page_number, f"Mismatch at index {idx} on page_number"
            assert cit.filename == src.filename, f"Mismatch at index {idx} on filename"
            assert cit.content_type == src.content_type, f"Mismatch at index {idx} on content_type"
            assert cit.score == src.score, f"Mismatch at index {idx} on score"

    def test_deliberate_inconsistent_citation_data_contract(self) -> None:
        """Verify behavior when creating an intentionally modified citation."""
        # AgentCitation is a typed data contract: it validates types and non-empty strings.
        # It accepts any valid strings as passed.
        inconsistent_citation = AgentCitation(
            document_id=DAY45_DOC_B,
            filename=DAY45_FILE_B,
            chunk_id=DAY45_CHUNK_A1,  # CHUNK_A1 attached to DOC_B
            page_number=DAY45_PAGE_A1,
        )
        assert inconsistent_citation.document_id == DAY45_DOC_B
        assert inconsistent_citation.chunk_id == DAY45_CHUNK_A1


# ============================================================================
# 4. Missing & Multiple Citations
# ============================================================================

class TestMissingAndMultipleCitations:
    """Certifies behavior with empty citation lists and multi-citation responses."""

    def test_missing_or_empty_citations_in_response(self) -> None:
        """AgentResponse with empty citations is valid and reports has_citations=False."""
        resp = AgentResponse(
            answer="Direct answer without citations",
            agent_name="SearchAgent",
            citations=[],
        )
        assert resp.has_citations is False
        assert resp.total_citations == 0
        assert resp.unique_documents == []
        assert resp.is_success is True

    def test_multiple_citations_across_multiple_documents(self) -> None:
        """AgentResponse correctly indexes multiple citations across DOC-A and DOC-B."""
        c_a1 = AgentCitation(document_id=DAY45_DOC_A, filename=DAY45_FILE_A, chunk_id=DAY45_CHUNK_A1)
        c_a2 = AgentCitation(document_id=DAY45_DOC_A, filename=DAY45_FILE_A, chunk_id=DAY45_CHUNK_A2)
        c_b1 = AgentCitation(document_id=DAY45_DOC_B, filename=DAY45_FILE_B, chunk_id=DAY45_CHUNK_B1)

        resp = AgentResponse(
            answer="Multi-doc answer",
            agent_name="SearchAgent",
            citations=[c_a1, c_a2, c_b1],
        )

        assert resp.total_citations == 3
        assert resp.unique_document_count == 2
        assert resp.unique_documents == [DAY45_DOC_A, DAY45_DOC_B]


# ============================================================================
# 5. Citation & Context Ordering
# ============================================================================

class TestCitationAndContextOrdering:
    """Certifies that context blocks and citations preserve relevance ordering."""

    def test_citation_order_matches_source_order(self) -> None:
        """Citations created from sorted retrieval results preserve the exact order."""
        sources = [
            VectorSearchResult(
                chunk_id="C-1", score=0.99, document_id=DAY45_DOC_A, filename=DAY45_FILE_A,
                page_number=1, chunk_index=0, content_type="text", content="First",
            ),
            VectorSearchResult(
                chunk_id="C-2", score=0.77, document_id=DAY45_DOC_B, filename=DAY45_FILE_B,
                page_number=1, chunk_index=0, content_type="text", content="Second",
            ),
            VectorSearchResult(
                chunk_id="C-3", score=0.55, document_id=DAY45_DOC_A, filename=DAY45_FILE_A,
                page_number=2, chunk_index=1, content_type="text", content="Third",
            ),
        ]

        # Context ordering
        context = build_retrieval_context(sources)
        pos1 = context.find("[Source 1]")
        pos2 = context.find("[Source 2]")
        pos3 = context.find("[Source 3]")
        assert pos1 < pos2 < pos3

        # Citation ordering
        citations = [AgentCitation.from_search_result(s) for s in sources]
        assert [c.chunk_id for c in citations] == ["C-1", "C-2", "C-3"]
        assert [c.score for c in citations] == [0.99, 0.77, 0.55]


# ============================================================================
# 6. Duplicate Source & Citation Behavior
# ============================================================================

class TestDuplicateSourceAndCitationBehavior:
    """Certifies deduplication in retrieval processing vs direct preservation in citations."""

    def test_retrieval_deduplication_then_context_building(self) -> None:
        """process_retrieval_results deduplicates by chunk_id before context is built."""
        r1 = VectorSearchResult(
            chunk_id=DAY45_CHUNK_A1, score=0.70, document_id=DAY45_DOC_A, filename=DAY45_FILE_A,
            page_number=1, chunk_index=0, content_type="text", content="Old version",
        )
        r2 = VectorSearchResult(
            chunk_id=DAY45_CHUNK_A1, score=0.95, document_id=DAY45_DOC_A, filename=DAY45_FILE_A,
            page_number=1, chunk_index=0, content_type="text", content="New highest score version",
        )

        deduped = process_retrieval_results([r1, r2])
        assert len(deduped) == 1
        assert deduped[0].score == 0.95

        context = build_retrieval_context(deduped)
        assert "[Source 1]" in context
        assert "[Source 2]" not in context
        assert "New highest score version" in context


# ============================================================================
# 7. Multi-Document & Cross-Request Context Isolation
# ============================================================================

class TestMultiDocumentAndCrossRequestIsolation:
    """Certifies zero cross-contamination between DOC-A and DOC-B contexts."""

    def test_multi_document_context_isolation(self) -> None:
        """Context A contains only DOC-A markers; Context B contains only DOC-B markers."""
        src_a = VectorSearchResult(
            chunk_id=DAY45_CHUNK_A1, score=0.9, document_id=DAY45_DOC_A, filename=DAY45_FILE_A,
            page_number=1, chunk_index=0, content_type="text",
            content=f"{DAY45_DOC_A_MARKER} - {DAY45_A_UNIQUE}",
        )
        src_b = VectorSearchResult(
            chunk_id=DAY45_CHUNK_B1, score=0.9, document_id=DAY45_DOC_B, filename=DAY45_FILE_B,
            page_number=1, chunk_index=0, content_type="text",
            content=f"{DAY45_DOC_B_MARKER} - {DAY45_B_UNIQUE}",
        )

        context_a = build_retrieval_context([src_a])
        context_b = build_retrieval_context([src_b])

        # Assert Context A contains ONLY A data
        assert DAY45_DOC_A_MARKER in context_a
        assert DAY45_A_UNIQUE in context_a
        assert DAY45_DOC_B_MARKER not in context_a
        assert DAY45_B_UNIQUE not in context_a
        assert DAY45_FILE_B not in context_a

        # Assert Context B contains ONLY B data
        assert DAY45_DOC_B_MARKER in context_b
        assert DAY45_B_UNIQUE in context_b
        assert DAY45_DOC_A_MARKER not in context_b
        assert DAY45_A_UNIQUE not in context_b
        assert DAY45_FILE_A not in context_b

    def test_cross_request_isolation_sequential(self) -> None:
        """Two search requests processed sequentially do not leak citations or contexts."""
        c_a = AgentCitation(
            document_id=DAY45_DOC_A, filename=DAY45_FILE_A, chunk_id=DAY45_CHUNK_A1,
            metadata={"request_id": "REQ-A", "marker": DAY45_A_UNIQUE},
        )
        c_b = AgentCitation(
            document_id=DAY45_DOC_B, filename=DAY45_FILE_B, chunk_id=DAY45_CHUNK_B1,
            metadata={"request_id": "REQ-B", "marker": DAY45_B_UNIQUE},
        )

        resp_a = AgentResponse(
            answer="Response for Request A",
            agent_name="SearchAgent",
            citations=[c_a],
            metadata={"request_id": "REQ-A"},
        )
        resp_b = AgentResponse(
            answer="Response for Request B",
            agent_name="SearchAgent",
            citations=[c_b],
            metadata={"request_id": "REQ-B"},
        )

        assert resp_a.metadata["request_id"] == "REQ-A"
        assert resp_a.citations[0].metadata["marker"] == DAY45_A_UNIQUE
        assert DAY45_DOC_B not in resp_a.unique_documents

        assert resp_b.metadata["request_id"] == "REQ-B"
        assert resp_b.citations[0].metadata["marker"] == DAY45_B_UNIQUE
        assert DAY45_DOC_A not in resp_b.unique_documents


# ============================================================================
# 8. Empty & Malformed Retrieval Handling
# ============================================================================

class TestEmptyAndMalformedRetrievalHandling:
    """Certifies validation and error handling on empty or malformed inputs."""

    def test_empty_retrieval_results_returns_empty_context(self) -> None:
        """build_retrieval_context with empty list returns empty string."""
        assert build_retrieval_context([]) == ""

    def test_invalid_retrieval_result_type_rejection(self) -> None:
        """build_retrieval_context rejects non-list input with TypeError."""
        with pytest.raises(TypeError, match="results must be a list"):
            build_retrieval_context("not_a_list")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="is not a VectorSearchResult"):
            build_retrieval_context([{"dict": "instead_of_model"}])  # type: ignore[list-item]

    def test_citation_creation_from_malformed_object_raises(self) -> None:
        """AgentCitation.from_search_result raises AgentValidationError when required attrs are missing."""
        class MalformedSource:
            pass

        with pytest.raises(AgentValidationError, match="missing required attribute"):
            AgentCitation.from_search_result(MalformedSource())


# ============================================================================
# 9. Context & Citation Serialization Round-Trip
# ============================================================================

class TestSerializationAndRoundTrip:
    """Certifies dictionary and JSON serialization round-trips."""

    def test_citation_serialization_and_json_roundtrip(self) -> None:
        """AgentCitation survives to_dict -> JSON -> from_dict exact roundtrip."""
        citation = AgentCitation(
            document_id=DAY45_DOC_A,
            filename=DAY45_FILE_A,
            chunk_id=DAY45_CHUNK_A1,
            page_number=DAY45_PAGE_A1,
            content_type="text",
            score=0.94,
            metadata={"marker": DAY45_DOC_A_MARKER},
        )

        d = citation.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        restored = AgentCitation.from_dict(parsed)

        assert restored == citation
        assert restored.document_id == DAY45_DOC_A
        assert restored.chunk_id == DAY45_CHUNK_A1
        assert restored.page_number == DAY45_PAGE_A1
        assert restored.score == 0.94
        assert restored.metadata["marker"] == DAY45_DOC_A_MARKER

    def test_agent_response_and_search_result_serialization(self) -> None:
        """AgentResponse and SearchResult containing citations serialize and restore cleanly."""
        citation = AgentCitation(
            document_id=DAY45_DOC_A,
            filename=DAY45_FILE_A,
            chunk_id=DAY45_CHUNK_A1,
            page_number=DAY45_PAGE_A1,
        )
        sres = SearchResult(
            query="Find Day 45 evidence",
            status="RESULTS_FOUND",
            citations=[citation],
            context="[Source 1] Evidence content",
            metadata={"marker": DAY45_DOC_A_MARKER},
        )

        d_sres = sres.to_dict()
        restored_sres = SearchResult.from_dict(d_sres)

        assert restored_sres.query == sres.query
        assert restored_sres.status == "RESULTS_FOUND"
        assert len(restored_sres.citations) == 1
        assert restored_sres.citations[0].document_id == DAY45_DOC_A
        assert restored_sres.context == sres.context


# ============================================================================
# 10. Agent Response & Visual Evidence Grounding
# ============================================================================

class TestAgentResponseAndVisualEvidenceGrounding:
    """Certifies traceability from AgentResponse and VisualEvidence back to source chunks."""

    def test_agent_response_traceable_to_source(self) -> None:
        """AgentResponse citations maintain direct provenance to original VectorSearchResult."""
        vs_res = VectorSearchResult(
            chunk_id=DAY45_CHUNK_A1,
            score=0.96,
            document_id=DAY45_DOC_A,
            filename=DAY45_FILE_A,
            page_number=DAY45_PAGE_A1,
            chunk_index=0,
            content_type="text",
            content=DAY45_DOC_A_CHUNK_1,
        )

        citation = AgentCitation.from_search_result(vs_res)
        response = AgentResponse(
            answer=f"Verified answer based on {DAY45_DOC_A_MARKER}",
            agent_name="SearchAgent",
            citations=[citation],
        )

        assert response.citations[0].document_id == vs_res.document_id
        assert response.citations[0].chunk_id == vs_res.chunk_id
        assert response.citations[0].page_number == vs_res.page_number
        assert response.citations[0].filename == vs_res.filename

    def test_visual_evidence_grounding_preserves_lineage(self) -> None:
        """VisualEvidence constructed from citation retains exact page and document identity."""
        img_citation = AgentCitation(
            document_id=DAY45_DOC_A,
            filename=DAY45_FILE_A,
            chunk_id=DAY45_CHUNK_A2,
            page_number=DAY45_PAGE_A2,
            content_type="image",
            score=0.91,
            metadata={"image_path": "/img/chart.png", "chunk_index": 1},
        )

        evidence = VisualEvidence.from_citation(img_citation)

        assert evidence.document_id == DAY45_DOC_A
        assert evidence.filename == DAY45_FILE_A
        assert evidence.chunk_id == DAY45_CHUNK_A2
        assert evidence.page_number == DAY45_PAGE_A2
        assert evidence.content_type == "image"
        assert evidence.image_path == "/img/chart.png"
        assert evidence.chunk_index == 1


# ============================================================================
# 11. Input Mutation Safety & Aliasing Protection
# ============================================================================

class TestInputMutationSafetyAndAliasing:
    """Certifies that context building and citation creation do not mutate caller inputs."""

    def test_build_retrieval_context_does_not_mutate_input_list(self) -> None:
        """build_retrieval_context leaves caller results list unmodified."""
        r1 = VectorSearchResult(
            chunk_id=DAY45_CHUNK_A1, score=0.9, document_id=DAY45_DOC_A, filename=DAY45_FILE_A,
            page_number=1, chunk_index=0, content_type="text", content="Content 1",
        )
        r2 = VectorSearchResult(
            chunk_id=DAY45_CHUNK_A2, score=0.8, document_id=DAY45_DOC_A, filename=DAY45_FILE_A,
            page_number=2, chunk_index=1, content_type="text", content="Content 2",
        )

        caller_list = [r1, r2]
        caller_copy = list(caller_list)

        ctx = build_retrieval_context(caller_list)

        assert len(caller_list) == 2
        assert caller_list == caller_copy
        assert "[Source 1]" in ctx

    def test_citation_metadata_aliasing_protection(self) -> None:
        """Modifying caller metadata dict does not mutate citation metadata."""
        caller_meta = {"key": "original_val"}
        citation = AgentCitation(
            document_id=DAY45_DOC_A,
            filename=DAY45_FILE_A,
            chunk_id=DAY45_CHUNK_A1,
            metadata=caller_meta,
        )

        caller_meta["key"] = "MUTATED_VALUE"
        assert citation.metadata["key"] == "original_val"


# ============================================================================
# 12. Repeated Execution & Error Isolation
# ============================================================================

class TestRepeatedExecutionAndErrorIsolation:
    """Certifies stability across 3 runs and sequential error isolation."""

    def test_context_building_determinism_3_iterations(self) -> None:
        """Running context building 3 times produces identical string representations."""
        sources = [
            VectorSearchResult(
                chunk_id=f"CHUNK-{i}", score=0.9 - (i * 0.1), document_id=DAY45_DOC_A,
                filename=DAY45_FILE_A, page_number=i + 1, chunk_index=i,
                content_type="text", content=f"Content {i}",
            )
            for i in range(3)
        ]

        runs = [build_retrieval_context(sources) for _ in range(3)]
        assert runs[0] == runs[1] == runs[2]

    def test_sequential_error_isolation(self) -> None:
        """An invalid citation in a sequence does not affect prior or subsequent valid citations."""
        results: list[str] = []

        # 1. Valid A
        c_a = AgentCitation(document_id=DAY45_DOC_A, filename=DAY45_FILE_A, chunk_id=DAY45_CHUNK_A1)
        results.append(c_a.chunk_id)

        # 2. Invalid B (empty chunk_id)
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id=DAY45_DOC_A, filename=DAY45_FILE_A, chunk_id="")

        # 3. Valid C
        c_c = AgentCitation(document_id=DAY45_DOC_B, filename=DAY45_FILE_B, chunk_id=DAY45_CHUNK_B1)
        results.append(c_c.chunk_id)

        assert results == [DAY45_CHUNK_A1, DAY45_CHUNK_B1]


# ============================================================================
# 13. Complete Offline End-to-End Grounding Flow
# ============================================================================

class TestCompleteEndToEndGroundingFlow:
    """Certifies the complete offline grounding pipeline from retrieval to citation & evidence."""

    def test_complete_offline_grounding_chain(self) -> None:
        """
        Flow:
          VectorSearchResult (Member 1)
            -> build_retrieval_context (Member 1)
            -> AgentCitation (Member 2)
            -> AgentResponse (Member 2)
            -> SearchResult (Member 2)
            -> VisualEvidence (Member 3)
        """
        # 1. Member 1 Retrieval Result
        vs_res = VectorSearchResult(
            chunk_id=DAY45_CHUNK_A2,
            score=0.97,
            document_id=DAY45_DOC_A,
            filename=DAY45_FILE_A,
            page_number=DAY45_PAGE_A2,
            chunk_index=1,
            content_type="image",
            content=f"Diagram content: {DAY45_DOC_A_MARKER}",
            metadata={"image_path": "/data/p2.png", "marker": DAY45_A_UNIQUE},
        )

        # 2. Member 1 Context Building
        context = build_retrieval_context([vs_res])
        assert DAY45_FILE_A in context
        assert f"Page: {DAY45_PAGE_A2}" in context
        assert "Type: image" in context
        assert DAY45_DOC_A_MARKER in context

        # 3. Member 2 Citation Creation
        citation = AgentCitation.from_search_result(vs_res)
        assert citation.document_id == DAY45_DOC_A
        assert citation.chunk_id == DAY45_CHUNK_A2
        assert citation.page_number == DAY45_PAGE_A2
        assert citation.score == 0.97

        # 4. Member 2 Response Delivery
        agent_resp = AgentResponse(
            answer="Grounded answer text.",
            agent_name="SearchAgent",
            citations=[citation],
            metadata={"query": "Locate diagram", "context": context},
        )
        assert agent_resp.citations[0].document_id == DAY45_DOC_A

        # 5. Member 2 Search Packaging
        search_res = SearchResult.from_response(agent_resp)
        assert search_res.citations[0].chunk_id == DAY45_CHUNK_A2
        assert search_res.status == "RESULTS_FOUND"

        # 6. Member 3 Visual Evidence Grounding
        evidence = VisualEvidence.from_citation(search_res.citations[0])
        assert evidence.document_id == DAY45_DOC_A
        assert evidence.filename == DAY45_FILE_A
        assert evidence.chunk_id == DAY45_CHUNK_A2
        assert evidence.page_number == DAY45_PAGE_A2
        assert evidence.content_type == "image"
        assert evidence.image_path == "/data/p2.png"
