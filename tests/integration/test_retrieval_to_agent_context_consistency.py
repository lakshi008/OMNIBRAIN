"""
OmniBrain Member 4 — Day 48 Retrieval-to-Agent Contract & Context Consistency Regression.

Validates the complete contract pipeline:
  Retrieval (VectorSearchResult, process_retrieval_results)
    ↓
  Context Building (build_retrieval_context)
    ↓
  Agent Input (AgentRequest, SearchRequest)
    ↓
  Agent Response & Search Packaging (AgentResponse, SearchResult, AgentCitation)

Covers:
  1.  Valid retrieval result schema validation and field integrity.
  2.  Retrieval result -> formatted context conversion.
  3.  Deterministic context ordering matching ranking order.
  4.  Document, page, and chunk lineage preservation across the pipeline.
  5.  Metadata preservation from retrieval to citation and response.
  6.  Retrieval -> Agent input construction and context handoff.
  7.  Agent response construction and citation consistency.
  8.  Multi-document context isolation and source contamination prevention.
  9.  Cross-request isolation and execution order reversal (A->B vs B->A).
  10. Empty retrieval and malformed retrieval input handling.
  11. Duplicate retrieval result handling (deduplication preserving highest score).
  12. Retrieval order stability and 3-iteration determinism.
  13. Input mutation safety and context/response object isolation.
  14. Serialization and deserialization round-trips for all retrieval/agent models.
  15. Error isolation across sequential valid/invalid processing.
  16. Complete offline end-to-end retrieval-to-agent pipeline verification.

Constraints:
  - 100% Offline: Synthetic deterministic models, no network, no external APIs.
  - Zero production code modified.
  - No new agent logic, ranking logic, adapters, or wrappers added.
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


# ============================================================================
# Deterministic Synthetic Fixtures
# ============================================================================

DAY48_DOC_A = "DAY48-DOC-A"
DAY48_DOC_B = "DAY48-DOC-B"

DAY48_FILE_A = "day48_document_alpha.pdf"
DAY48_FILE_B = "day48_document_beta.pdf"

DAY48_PAGE_A1 = 1
DAY48_PAGE_A2 = 2
DAY48_PAGE_B1 = 1

DAY48_CHUNK_A1 = "DAY48-CHUNK-A1"
DAY48_CHUNK_A2 = "DAY48-CHUNK-A2"
DAY48_CHUNK_B1 = "DAY48-CHUNK-B1"

DAY48_A_CHUNK_1 = "DAY48_A_CHUNK_1: Primary architectural specifications for OmniBrain."
DAY48_A_CHUNK_2 = "DAY48_A_CHUNK_2: Tabular performance benchmarks for ingestion."
DAY48_B_CHUNK_1 = "DAY48_B_CHUNK_1: Beta subsystem network configuration parameters."

DAY48_A_ONLY = "DAY48_A_ONLY_EXCLUSIVE_MARKER_99"
DAY48_B_ONLY = "DAY48_B_ONLY_EXCLUSIVE_MARKER_88"

DAY48_META_A: dict[str, Any] = {
    "day48_source": "A",
    "day48_test": "contract",
    "marker": DAY48_A_ONLY,
}
DAY48_META_B: dict[str, Any] = {
    "day48_source": "B",
    "day48_test": "contract",
    "marker": DAY48_B_ONLY,
}


# ============================================================================
# 1. Valid Retrieval Result Structure
# ============================================================================

class TestValidRetrievalResultStructure:
    """Certifies that VectorSearchResult models retain all contractual fields."""

    def test_vector_search_result_fields(self) -> None:
        """VectorSearchResult holds all document, chunk, page, and score fields."""
        res = VectorSearchResult(
            chunk_id=DAY48_CHUNK_A1,
            score=0.96,
            document_id=DAY48_DOC_A,
            filename=DAY48_FILE_A,
            page_number=DAY48_PAGE_A1,
            chunk_index=0,
            content_type="text",
            content=f"{DAY48_A_CHUNK_1} - {DAY48_A_ONLY}",
            metadata=DAY48_META_A,
        )

        assert res.chunk_id == DAY48_CHUNK_A1
        assert res.score == 0.96
        assert res.document_id == DAY48_DOC_A
        assert res.filename == DAY48_FILE_A
        assert res.page_number == DAY48_PAGE_A1
        assert res.chunk_index == 0
        assert res.content_type == "text"
        assert DAY48_A_CHUNK_1 in res.content
        assert DAY48_A_ONLY in res.content
        assert res.metadata == DAY48_META_A


# ============================================================================
# 2. Retrieval -> Context Conversion & Ordering
# ============================================================================

class TestRetrievalToContextConversionAndOrdering:
    """Certifies build_retrieval_context formatting, numbering, and order preservation."""

    def test_retrieval_to_context_preserves_content_and_metadata(self) -> None:
        """Context string includes [Source N] headers, file, page, type, and markers."""
        r1 = VectorSearchResult(
            chunk_id=DAY48_CHUNK_A1,
            score=0.95,
            document_id=DAY48_DOC_A,
            filename=DAY48_FILE_A,
            page_number=DAY48_PAGE_A1,
            chunk_index=0,
            content_type="text",
            content=f"{DAY48_A_CHUNK_1} - {DAY48_A_ONLY}",
        )
        r2 = VectorSearchResult(
            chunk_id=DAY48_CHUNK_A2,
            score=0.85,
            document_id=DAY48_DOC_A,
            filename=DAY48_FILE_A,
            page_number=DAY48_PAGE_A2,
            chunk_index=1,
            content_type="table",
            content=DAY48_A_CHUNK_2,
        )
        r3 = VectorSearchResult(
            chunk_id=DAY48_CHUNK_B1,
            score=0.75,
            document_id=DAY48_DOC_B,
            filename=DAY48_FILE_B,
            page_number=DAY48_PAGE_B1,
            chunk_index=0,
            content_type="text",
            content=f"{DAY48_B_CHUNK_1} - {DAY48_B_ONLY}",
        )

        context = build_retrieval_context([r1, r2, r3])

        # Verify Source 1
        assert "[Source 1]" in context
        assert f"File: {DAY48_FILE_A}" in context
        assert f"Page: {DAY48_PAGE_A1}" in context
        assert "Type: text" in context
        assert DAY48_A_CHUNK_1 in context
        assert DAY48_A_ONLY in context

        # Verify Source 2
        assert "[Source 2]" in context
        assert f"Page: {DAY48_PAGE_A2}" in context
        assert "Type: table" in context
        assert DAY48_A_CHUNK_2 in context

        # Verify Source 3
        assert "[Source 3]" in context
        assert f"File: {DAY48_FILE_B}" in context
        assert f"Page: {DAY48_PAGE_B1}" in context
        assert DAY48_B_CHUNK_1 in context
        assert DAY48_B_ONLY in context

        # Verify strictly increasing order
        pos1 = context.find("[Source 1]")
        pos2 = context.find("[Source 2]")
        pos3 = context.find("[Source 3]")
        assert pos1 < pos2 < pos3


# ============================================================================
# 3. Document, Page, and Chunk Lineage
# ============================================================================

class TestDocumentPageChunkLineage:
    """Certifies that document, page, and chunk identity remain 100% consistent."""

    def test_lineage_preservation_through_citation_conversion(self) -> None:
        """VectorSearchResult -> AgentCitation retains exact document, page, and chunk IDs."""
        res_a = VectorSearchResult(
            chunk_id=DAY48_CHUNK_A1,
            score=0.92,
            document_id=DAY48_DOC_A,
            filename=DAY48_FILE_A,
            page_number=DAY48_PAGE_A1,
            chunk_index=0,
            content_type="text",
            content=DAY48_A_CHUNK_1,
            metadata=DAY48_META_A,
        )
        res_b = VectorSearchResult(
            chunk_id=DAY48_CHUNK_B1,
            score=0.88,
            document_id=DAY48_DOC_B,
            filename=DAY48_FILE_B,
            page_number=DAY48_PAGE_B1,
            chunk_index=0,
            content_type="text",
            content=DAY48_B_CHUNK_1,
            metadata=DAY48_META_B,
        )

        cit_a = AgentCitation.from_search_result(res_a)
        cit_b = AgentCitation.from_search_result(res_b)

        # DOC-A Lineage
        assert cit_a.document_id == DAY48_DOC_A
        assert cit_a.filename == DAY48_FILE_A
        assert cit_a.chunk_id == DAY48_CHUNK_A1
        assert cit_a.page_number == DAY48_PAGE_A1
        assert cit_a.score == 0.92
        assert cit_a.metadata["day48_source"] == "A"

        # DOC-B Lineage
        assert cit_b.document_id == DAY48_DOC_B
        assert cit_b.filename == DAY48_FILE_B
        assert cit_b.chunk_id == DAY48_CHUNK_B1
        assert cit_b.page_number == DAY48_PAGE_B1
        assert cit_b.score == 0.88
        assert cit_b.metadata["day48_source"] == "B"


# ============================================================================
# 4. Retrieval -> Agent Input & Agent Response Delivery
# ============================================================================

class TestRetrievalToAgentHandoff:
    """Certifies handoff from context building into AgentRequest and AgentResponse."""

    def test_agent_request_and_response_consistency(self) -> None:
        """Context and citations package cleanly into AgentResponse and SearchResult."""
        res = VectorSearchResult(
            chunk_id=DAY48_CHUNK_A1,
            score=0.94,
            document_id=DAY48_DOC_A,
            filename=DAY48_FILE_A,
            page_number=DAY48_PAGE_A1,
            chunk_index=0,
            content_type="text",
            content=DAY48_A_CHUNK_1,
            metadata=DAY48_META_A,
        )

        context = build_retrieval_context([res])
        citation = AgentCitation.from_search_result(res)

        # AgentRequest
        req = AgentRequest(
            query="Analyze architecture specs",
            session_id="SESS-DAY48",
            metadata={"context": context, "source": "synthetic"},
        )
        assert req.metadata["context"] == context

        # AgentResponse
        resp = AgentResponse(
            answer=f"Verified specs based on {DAY48_A_CHUNK_1}",
            agent_name="SearchAgent",
            citations=[citation],
            metadata={"query": req.query, "context": context},
        )

        assert resp.citations[0].document_id == DAY48_DOC_A
        assert resp.citations[0].chunk_id == DAY48_CHUNK_A1
        assert resp.citations[0].page_number == DAY48_PAGE_A1
        assert resp.citations[0].score == 0.94

        # SearchResult packaging
        packaged = SearchResult.from_response(resp)
        assert packaged.status == "RESULTS_FOUND"
        assert packaged.total_results == 1
        assert packaged.has_results is True
        assert packaged.citations[0].document_id == DAY48_DOC_A


# ============================================================================
# 5. Multi-Document Isolation & Source Contamination Prevention
# ============================================================================

class TestMultiDocumentIsolationAndContamination:
    """Certifies zero cross-contamination between DOC-A and DOC-B contexts."""

    def test_multi_document_context_and_marker_isolation(self) -> None:
        """Context A contains only DOC-A markers; Context B contains only DOC-B markers."""
        r_a = VectorSearchResult(
            chunk_id=DAY48_CHUNK_A1, score=0.9, document_id=DAY48_DOC_A, filename=DAY48_FILE_A,
            page_number=1, chunk_index=0, content_type="text", content=f"{DAY48_A_CHUNK_1} - {DAY48_A_ONLY}",
        )
        r_b = VectorSearchResult(
            chunk_id=DAY48_CHUNK_B1, score=0.9, document_id=DAY48_DOC_B, filename=DAY48_FILE_B,
            page_number=1, chunk_index=0, content_type="text", content=f"{DAY48_B_CHUNK_1} - {DAY48_B_ONLY}",
        )

        ctx_a = build_retrieval_context([r_a])
        ctx_b = build_retrieval_context([r_b])

        # Assert Context A contains ONLY A data
        assert DAY48_A_ONLY in ctx_a
        assert DAY48_A_CHUNK_1 in ctx_a
        assert DAY48_B_ONLY not in ctx_a
        assert DAY48_B_CHUNK_1 not in ctx_a
        assert DAY48_FILE_B not in ctx_a

        # Assert Context B contains ONLY B data
        assert DAY48_B_ONLY in ctx_b
        assert DAY48_B_CHUNK_1 in ctx_b
        assert DAY48_A_ONLY not in ctx_b
        assert DAY48_A_CHUNK_1 not in ctx_b
        assert DAY48_FILE_A not in ctx_b


# ============================================================================
# 6. Cross-Request Isolation & Order Reversal
# ============================================================================

class TestCrossRequestIsolationAndOrderReversal:
    """Certifies that request order (A->B vs B->A) produces identical isolated outputs."""

    def test_request_order_reversal_stability(self) -> None:
        """Running A->B then B->A produces identical isolated outputs."""
        c_a = AgentCitation(
            document_id=DAY48_DOC_A, filename=DAY48_FILE_A, chunk_id=DAY48_CHUNK_A1,
            metadata={"marker": DAY48_A_ONLY},
        )
        c_b = AgentCitation(
            document_id=DAY48_DOC_B, filename=DAY48_FILE_B, chunk_id=DAY48_CHUNK_B1,
            metadata={"marker": DAY48_B_ONLY},
        )

        # Run A then B
        resp_a1 = AgentResponse(answer="A", agent_name="Agent", citations=[c_a])
        resp_b1 = AgentResponse(answer="B", agent_name="Agent", citations=[c_b])

        # Run B then A
        resp_b2 = AgentResponse(answer="B", agent_name="Agent", citations=[c_b])
        resp_a2 = AgentResponse(answer="A", agent_name="Agent", citations=[c_a])

        assert resp_a1.to_dict() == resp_a2.to_dict()
        assert resp_b1.to_dict() == resp_b2.to_dict()
        assert DAY48_DOC_B not in resp_a2.unique_documents
        assert DAY48_DOC_A not in resp_b2.unique_documents


# ============================================================================
# 7. Empty, Invalid, and Duplicate Retrieval Handling
# ============================================================================

class TestEmptyInvalidAndDuplicateRetrievalHandling:
    """Certifies validation and deduplication contracts."""

    def test_empty_retrieval_returns_empty_context(self) -> None:
        """Passing empty list to build_retrieval_context returns empty string."""
        assert build_retrieval_context([]) == ""

    def test_invalid_retrieval_input_type_rejection(self) -> None:
        """build_retrieval_context rejects non-list and non-model elements."""
        with pytest.raises(TypeError, match="results must be a list"):
            build_retrieval_context("invalid_string")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="is not a VectorSearchResult"):
            build_retrieval_context([{"dict": "not_model"}])  # type: ignore[list-item]

    def test_duplicate_retrieval_deduplication(self) -> None:
        """process_retrieval_results deduplicates identical chunk_id retaining highest score."""
        r1 = VectorSearchResult(
            chunk_id=DAY48_CHUNK_A1, score=0.60, document_id=DAY48_DOC_A, filename=DAY48_FILE_A,
            page_number=1, chunk_index=0, content_type="text", content="Lower score version",
        )
        r2 = VectorSearchResult(
            chunk_id=DAY48_CHUNK_A1, score=0.95, document_id=DAY48_DOC_A, filename=DAY48_FILE_A,
            page_number=1, chunk_index=0, content_type="text", content="Highest score version",
        )

        deduped = process_retrieval_results([r1, r2])
        assert len(deduped) == 1
        assert deduped[0].score == 0.95
        assert deduped[0].content == "Highest score version"


# ============================================================================
# 8. Input Mutation Safety & Object Isolation
# ============================================================================

class TestMutationSafetyAndObjectIsolation:
    """Certifies that caller input structures are not mutated and context objects are independent."""

    def test_build_retrieval_context_does_not_mutate_caller_list(self) -> None:
        """Caller results list remains unchanged after context building."""
        r1 = VectorSearchResult(
            chunk_id="C-1", score=0.9, document_id="D-1", filename="f.pdf",
            page_number=1, chunk_index=0, content_type="text", content="C1",
        )
        r2 = VectorSearchResult(
            chunk_id="C-2", score=0.8, document_id="D-1", filename="f.pdf",
            page_number=2, chunk_index=1, content_type="text", content="C2",
        )

        caller_list = [r1, r2]
        caller_copy = list(caller_list)

        ctx = build_retrieval_context(caller_list)

        assert len(caller_list) == 2
        assert caller_list == caller_copy
        assert "[Source 1]" in ctx

    def test_context_and_response_mutation_independence(self) -> None:
        """Modifying one response dictionary does not alter another."""
        c_a = AgentCitation(document_id=DAY48_DOC_A, filename=DAY48_FILE_A, chunk_id="C-A")
        c_b = AgentCitation(document_id=DAY48_DOC_B, filename=DAY48_FILE_B, chunk_id="C-B")

        resp_a = AgentResponse(answer="A", agent_name="Agent", citations=[c_a])
        resp_b = AgentResponse(answer="B", agent_name="Agent", citations=[c_b])

        d_a = resp_a.to_dict()
        d_a["answer"] = "MUTATED"

        assert resp_a.answer == "A"
        assert resp_b.answer == "B"


# ============================================================================
# 9. Serialization Round-Trips
# ============================================================================

class TestSerializationRoundTrips:
    """Certifies dictionary and JSON serialization round-trips."""

    def test_vector_search_result_and_citation_serialization(self) -> None:
        """VectorSearchResult and AgentCitation survive serialization cleanly."""
        res = VectorSearchResult(
            chunk_id=DAY48_CHUNK_A1,
            score=0.93,
            document_id=DAY48_DOC_A,
            filename=DAY48_FILE_A,
            page_number=DAY48_PAGE_A1,
            chunk_index=0,
            content_type="text",
            content=DAY48_A_CHUNK_1,
            metadata=DAY48_META_A,
        )

        # Dataclass serialization
        d_res = dataclasses.asdict(res)
        assert d_res["chunk_id"] == DAY48_CHUNK_A1
        assert d_res["score"] == 0.93
        assert d_res["metadata"]["day48_source"] == "A"

        # Citation serialization & JSON roundtrip
        citation = AgentCitation.from_search_result(res)
        d_cit = citation.to_dict()
        json_cit = json.dumps(d_cit)
        restored_cit = AgentCitation.from_dict(json.loads(json_cit))

        assert restored_cit == citation
        assert restored_cit.document_id == DAY48_DOC_A
        assert restored_cit.chunk_id == DAY48_CHUNK_A1
        assert restored_cit.score == 0.93

    def test_agent_response_and_search_result_serialization(self) -> None:
        """AgentResponse and SearchResult survive JSON serialization roundtrip."""
        citation = AgentCitation(
            document_id=DAY48_DOC_A,
            filename=DAY48_FILE_A,
            chunk_id=DAY48_CHUNK_A1,
            page_number=DAY48_PAGE_A1,
        )
        resp = AgentResponse(
            answer="Verified serializable answer",
            agent_name="SearchAgent",
            citations=[citation],
            metadata={"query": "Find specs"},
        )

        d = resp.to_dict()
        json_str = json.dumps(d)
        restored = AgentResponse.from_dict(json.loads(json_str))

        assert restored.answer == resp.answer
        assert len(restored.citations) == 1
        assert restored.citations[0].document_id == DAY48_DOC_A


# ============================================================================
# 10. Repeated Execution & Error Isolation
# ============================================================================

class TestRepeatedExecutionAndErrorIsolation:
    """Certifies 3-iteration determinism and error isolation."""

    def test_retrieval_to_context_determinism_3_iterations(self) -> None:
        """3 identical executions yield identical context output."""
        sources = [
            VectorSearchResult(
                chunk_id=f"CHUNK-{i}", score=0.9 - (i * 0.1), document_id=DAY48_DOC_A,
                filename=DAY48_FILE_A, page_number=i + 1, chunk_index=i,
                content_type="text", content=f"Content {i}",
            )
            for i in range(3)
        ]

        runs = [build_retrieval_context(sources) for _ in range(3)]
        assert runs[0] == runs[1] == runs[2]

    def test_sequential_error_isolation(self) -> None:
        """An invalid request in a sequence does not affect subsequent valid requests."""
        results: list[str] = []

        # 1. Valid A
        req_a = AgentRequest(query="Valid Query A")
        results.append(req_a.query)

        # 2. Invalid B (empty query)
        with pytest.raises(AgentValidationError):
            AgentRequest(query="")

        # 3. Valid C
        req_c = AgentRequest(query="Valid Query C")
        results.append(req_c.query)

        assert results == ["Valid Query A", "Valid Query C"]


# ============================================================================
# 11. End-to-End Pipeline Lineage Verification
# ============================================================================

class TestEndToEndPipelineLineage:
    """Certifies complete retrieval -> context -> agent response -> citation flow."""

    def test_complete_retrieval_to_agent_flow(self) -> None:
        """
        Flow:
          VectorSearchResult (Member 1)
            -> build_retrieval_context (Member 1)
            -> AgentCitation.from_search_result (Member 2)
            -> AgentResponse (Member 2)
            -> SearchResult.from_response (Member 2)
        """
        vs_res = VectorSearchResult(
            chunk_id=DAY48_CHUNK_A1,
            score=0.98,
            document_id=DAY48_DOC_A,
            filename=DAY48_FILE_A,
            page_number=DAY48_PAGE_A1,
            chunk_index=0,
            content_type="text",
            content=f"{DAY48_A_CHUNK_1} - {DAY48_A_ONLY}",
            metadata=DAY48_META_A,
        )

        # Context
        context = build_retrieval_context([vs_res])
        assert DAY48_A_CHUNK_1 in context
        assert DAY48_A_ONLY in context

        # Citation
        citation = AgentCitation.from_search_result(vs_res)
        assert citation.document_id == DAY48_DOC_A
        assert citation.chunk_id == DAY48_CHUNK_A1
        assert citation.page_number == DAY48_PAGE_A1
        assert citation.score == 0.98

        # AgentResponse
        resp = AgentResponse(
            answer=f"Verified specs based on {DAY48_A_CHUNK_1}",
            agent_name="SearchAgent",
            citations=[citation],
            metadata={"query": "Locate specs", "context": context},
        )
        assert resp.citations[0].document_id == DAY48_DOC_A

        # SearchResult
        packaged = SearchResult.from_response(resp)
        assert packaged.status == "RESULTS_FOUND"
        assert packaged.citations[0].chunk_id == DAY48_CHUNK_A1
        assert packaged.citations[0].metadata["marker"] == DAY48_A_ONLY
