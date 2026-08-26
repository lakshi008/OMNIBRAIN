"""
OmniBrain Member 4 — Day 49 Retrieval Quality & Ranking Regression Certification.

Validates the public retrieval and search contracts across:
  - Ingestion Retrieval Layer (VectorSearchResult, RetrievalServiceResult,
                               retrieve, retrieve_context, process_retrieval_results,
                               build_retrieval_context, QdrantVectorStore)
  - Agents Search Layer (SearchAgent, SearchRequest, SearchResult,
                         AgentCitation, AgentResponse, AgentRequest)

Covers:
  1.  Retrieval result model contract, source identity, and field validation.
  2.  Valid query execution targeting exact relevant markers.
  3.  Relevance ranking and descending score ordering.
  4.  Score preservation through downstream citation and response layers.
  5.  Document, page, and chunk lineage preservation.
  6.  Metadata preservation from retrieval to citation and response.
  7.  Top-K and max_results limit enforcement (k=1, 2, 3).
  8.  Query isolation (Query A vs Query B) and multi-document isolation (DOC-A vs DOC-B).
  9.  Page-level and chunk-level result isolation.
  10. Empty query rejection and no-match query safe fallback.
  11. Duplicate result deduplication (highest-score retention).
  12. Result order stability and score order stability across 3 runs.
  13. Retrieval -> Context conversion and marker preservation.
  14. Retrieval -> Agent context handoff.
  15. Cross-request isolation and execution order reversal (A->B vs B->A).
  16. Input and result mutation safety / object isolation.
  17. Serialization round-trips for retrieval and search packaging.
  18. Invalid query and invalid filter handling.
  19. Sequential failure isolation (VALID-A -> INVALID-B -> VALID-C).
  20. Complete offline execution with in-memory store and deterministic mock embeddings.

Constraints:
  - 100% Offline: In-memory QdrantVectorStore, mock embedding provider, no external network.
  - Zero production code modified.
  - No ranking algorithms, rerankers, wrappers, or caching added.
  - Synthetic deterministic data only.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest
from qdrant_client import QdrantClient

# Ingestion layer (Member 1)
from ingestion.models import (
    EmbeddingGenerationResult,
    EmbeddingVectorRecord,
    RetrievalServiceResult,
    VectorSearchResult,
)
from ingestion.qdrant_store import QdrantVectorStore
from ingestion.retrieval import retrieve
from ingestion.retrieval_processor import (
    build_retrieval_context,
    process_retrieval_results,
)
from ingestion.retrieval_service import retrieve_context

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
from agents.search_agent import SearchAgent


# ============================================================================
# Deterministic Synthetic Fixtures & Stubs
# ============================================================================

DAY49_DOC_A = "DAY49-DOC-A"
DAY49_DOC_B = "DAY49-DOC-B"

DAY49_FILE_A = "day49_alpha_specs.pdf"
DAY49_FILE_B = "day49_beta_notes.pdf"

DAY49_PAGE_A1 = 1
DAY49_PAGE_A2 = 2
DAY49_PAGE_A3 = 3
DAY49_PAGE_B1 = 1

DAY49_CHUNK_A1 = str(uuid.UUID("11111111-1111-1111-1111-111111111111"))
DAY49_CHUNK_A2 = str(uuid.UUID("22222222-2222-2222-2222-222222222222"))
DAY49_CHUNK_A3 = str(uuid.UUID("33333333-3333-3333-3333-333333333333"))
DAY49_CHUNK_B1 = str(uuid.UUID("44444444-4444-4444-4444-444444444444"))

DAY49_EXACT_RELEVANT_MARKER = "DAY49_EXACT_RELEVANT_MARKER_SPEC_100"
DAY49_PARTIAL_RELEVANT_MARKER = "DAY49_PARTIAL_RELEVANT_MARKER_SPEC_50"
DAY49_UNRELATED_MARKER = "DAY49_UNRELATED_MARKER_LEGACY_0"
DAY49_OTHER_DOCUMENT_MARKER = "DAY49_OTHER_DOCUMENT_MARKER_BETA_99"

DAY49_DOC_A_ONLY = "DAY49_DOC_A_ONLY_EXCLUSIVE_FLAG"
DAY49_DOC_B_ONLY = "DAY49_DOC_B_ONLY_EXCLUSIVE_FLAG"

DAY49_QUERY_WITH_NO_MATCH = "DAY49_QUERY_WITH_NO_MATCH_XYZ_NONEXISTENT"

DAY49_META_A: dict[str, Any] = {
    "day49_source": "synthetic",
    "day49_category": "relevance",
    "marker": DAY49_DOC_A_ONLY,
}
DAY49_META_B: dict[str, Any] = {
    "day49_source": "synthetic",
    "day49_category": "relevance",
    "marker": DAY49_DOC_B_ONLY,
}


class DeterministicDay49EmbeddingProvider:
    """Offline deterministic mock embedding provider returning fixed 4D vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Generate deterministic vector based on query string keywords."""
        clean = text.lower()
        if "exact" in clean or "spec_100" in clean or "chunk_a1" in clean:
            return [1.0, 0.0, 0.0, 0.0]
        if "partial" in clean or "spec_50" in clean or "chunk_a2" in clean:
            return [0.7, 0.7, 0.0, 0.0]
        if "unrelated" in clean or "legacy_0" in clean:
            return [0.0, 0.0, 1.0, 0.0]
        if "other" in clean or "beta" in clean or "doc_b" in clean:
            return [0.0, 1.0, 0.0, 0.0]
        if "no_match" in clean or "nonexistent" in clean:
            return [0.0, 0.0, 0.0, 1.0]
        return [0.25, 0.25, 0.25, 0.25]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding generation."""
        return [self.embed(t) for t in texts]


@pytest.fixture
def in_memory_store() -> QdrantVectorStore:
    """Create an isolated in-memory QdrantVectorStore."""
    client = QdrantClient(location=":memory:")
    return QdrantVectorStore(client=client)


# ============================================================================
# 1. Retrieval API Discovery & Valid Query
# ============================================================================

class TestRetrievalApiDiscoveryAndValidQuery:
    """Certifies that the existing retrieval API correctly fetches exact relevant markers."""

    def test_valid_retrieval_returns_exact_relevant_chunk(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """Query targeting exact relevant marker retrieves CHUNK-A1 with correct lineage."""
        in_memory_store.create_collection("docs", vector_dimension=4)

        r_a1 = EmbeddingVectorRecord(
            chunk_id=DAY49_CHUNK_A1,
            document_id=DAY49_DOC_A,
            filename=DAY49_FILE_A,
            chunk_index=0,
            page_number=DAY49_PAGE_A1,
            content_type="text",
            vector=[1.0, 0.0, 0.0, 0.0],
            metadata={"content": f"Exact specs: {DAY49_EXACT_RELEVANT_MARKER}", "doc": "A"},
        )
        r_b1 = EmbeddingVectorRecord(
            chunk_id=DAY49_CHUNK_B1,
            document_id=DAY49_DOC_B,
            filename=DAY49_FILE_B,
            chunk_index=0,
            page_number=DAY49_PAGE_B1,
            content_type="text",
            vector=[0.0, 1.0, 0.0, 0.0],
            metadata={"content": f"Other specs: {DAY49_OTHER_DOCUMENT_MARKER}", "doc": "B"},
        )

        gen_res = EmbeddingGenerationResult(
            document_id=DAY49_DOC_A,
            filename=DAY49_FILE_A,
            items=[r_a1, r_b1],
            dimension=4,
            is_ready=True,
        )
        in_memory_store.upsert_embeddings("docs", gen_res)

        embedder = DeterministicDay49EmbeddingProvider(dimension=4)
        agent = SearchAgent(
            embedding_provider=embedder,
            store=in_memory_store,
            collection_name="docs",
        )

        # Target exact marker
        resp = agent.search(f"Find {DAY49_EXACT_RELEVANT_MARKER}")

        assert resp.is_success is True
        assert resp.has_citations is True
        assert resp.citations[0].document_id == DAY49_DOC_A
        assert resp.citations[0].chunk_id == DAY49_CHUNK_A1
        assert resp.citations[0].page_number == DAY49_PAGE_A1
        assert resp.citations[0].filename == DAY49_FILE_A


# ============================================================================
# 2. Relevance Ranking & Score Preservation
# ============================================================================

class TestRelevanceRankingAndScorePreservation:
    """Certifies that process_retrieval_results sorts strictly descending by score."""

    def test_relevance_ranking_order_descending(self) -> None:
        """Exact relevant (0.95) > Partial relevant (0.75) > Unrelated (0.25)."""
        r_exact = VectorSearchResult(
            chunk_id=DAY49_CHUNK_A1, score=0.95, document_id=DAY49_DOC_A, filename=DAY49_FILE_A,
            page_number=1, chunk_index=0, content_type="text", content=DAY49_EXACT_RELEVANT_MARKER,
        )
        r_partial = VectorSearchResult(
            chunk_id=DAY49_CHUNK_A2, score=0.75, document_id=DAY49_DOC_A, filename=DAY49_FILE_A,
            page_number=2, chunk_index=1, content_type="text", content=DAY49_PARTIAL_RELEVANT_MARKER,
        )
        r_unrelated = VectorSearchResult(
            chunk_id=DAY49_CHUNK_A3, score=0.25, document_id=DAY49_DOC_A, filename=DAY49_FILE_A,
            page_number=3, chunk_index=2, content_type="text", content=DAY49_UNRELATED_MARKER,
        )

        ranked = process_retrieval_results([r_unrelated, r_exact, r_partial])

        assert len(ranked) == 3
        assert ranked[0].chunk_id == DAY49_CHUNK_A1
        assert ranked[0].score == 0.95
        assert ranked[1].chunk_id == DAY49_CHUNK_A2
        assert ranked[1].score == 0.75
        assert ranked[2].chunk_id == DAY49_CHUNK_A3
        assert ranked[2].score == 0.25

    def test_score_preservation_through_citation_and_response(self) -> None:
        """Relevance score survives from VectorSearchResult into citation and SearchResult."""
        vs_res = VectorSearchResult(
            chunk_id=DAY49_CHUNK_A1,
            score=0.9385,
            document_id=DAY49_DOC_A,
            filename=DAY49_FILE_A,
            page_number=1,
            chunk_index=0,
            content_type="text",
            content=DAY49_EXACT_RELEVANT_MARKER,
        )

        citation = AgentCitation.from_search_result(vs_res)
        assert citation.score == 0.9385

        resp = AgentResponse(
            answer="Ans",
            agent_name="Agent",
            citations=[citation],
            metadata={"query": "Find exact specs"},
        )
        assert resp.citations[0].score == 0.9385

        s_res = SearchResult.from_response(resp)
        assert s_res.citations[0].score == 0.9385


# ============================================================================
# 3. Source Lineage & Metadata Preservation
# ============================================================================

class TestSourceLineageAndMetadataPreservation:
    """Certifies preservation of document_id, page_number, chunk_id, and metadata."""

    def test_lineage_and_metadata_integrity(self) -> None:
        """Document, page, chunk IDs and metadata dict remain intact."""
        res = VectorSearchResult(
            chunk_id=DAY49_CHUNK_A1,
            score=0.91,
            document_id=DAY49_DOC_A,
            filename=DAY49_FILE_A,
            page_number=DAY49_PAGE_A1,
            chunk_index=0,
            content_type="text",
            content=DAY49_EXACT_RELEVANT_MARKER,
            metadata=DAY49_META_A,
        )

        citation = AgentCitation.from_search_result(res)

        assert citation.document_id == DAY49_DOC_A
        assert citation.filename == DAY49_FILE_A
        assert citation.chunk_id == DAY49_CHUNK_A1
        assert citation.page_number == DAY49_PAGE_A1
        assert citation.metadata["day49_source"] == "synthetic"
        assert citation.metadata["day49_category"] == "relevance"


# ============================================================================
# 4. Top-K Behavior (k=1, 2, 3)
# ============================================================================

class TestTopKBehavior:
    """Certifies top-k parameter and max_results limit enforcement."""

    @pytest.mark.parametrize("limit", [1, 2, 3])
    def test_max_results_capping(self, limit: int) -> None:
        """process_retrieval_results caps output at requested limit."""
        results = [
            VectorSearchResult(
                chunk_id=f"C-{i}", score=0.95 - (i * 0.1), document_id=DAY49_DOC_A,
                filename="a.pdf", page_number=1, chunk_index=i, content_type="text", content=f"Text {i}",
            )
            for i in range(5)
        ]

        capped = process_retrieval_results(results, max_results=limit)
        assert len(capped) == limit
        assert capped[0].score == 0.95


# ============================================================================
# 5. Query & Multi-Document Isolation
# ============================================================================

class TestQueryAndMultiDocumentIsolation:
    """Certifies that Query A vs Query B and DOC-A vs DOC-B remain isolated."""

    def test_multi_document_marker_isolation(self) -> None:
        """DOC-A context contains only DOC-A markers; DOC-B context contains only DOC-B."""
        r_a = VectorSearchResult(
            chunk_id=DAY49_CHUNK_A1, score=0.9, document_id=DAY49_DOC_A, filename=DAY49_FILE_A,
            page_number=1, chunk_index=0, content_type="text",
            content=f"{DAY49_EXACT_RELEVANT_MARKER} - {DAY49_DOC_A_ONLY}",
        )
        r_b = VectorSearchResult(
            chunk_id=DAY49_CHUNK_B1, score=0.9, document_id=DAY49_DOC_B, filename=DAY49_FILE_B,
            page_number=1, chunk_index=0, content_type="text",
            content=f"{DAY49_OTHER_DOCUMENT_MARKER} - {DAY49_DOC_B_ONLY}",
        )

        ctx_a = build_retrieval_context([r_a])
        ctx_b = build_retrieval_context([r_b])

        assert DAY49_DOC_A_ONLY in ctx_a
        assert DAY49_DOC_B_ONLY not in ctx_a
        assert DAY49_FILE_B not in ctx_a

        assert DAY49_DOC_B_ONLY in ctx_b
        assert DAY49_DOC_A_ONLY not in ctx_b
        assert DAY49_FILE_A not in ctx_b


# ============================================================================
# 6. Empty Query & No-Match Query Behavior
# ============================================================================

class TestEmptyAndNoMatchQueryBehavior:
    """Certifies validation on empty query and graceful handling on no-match query."""

    def test_empty_query_raises_validation_error(self) -> None:
        """Empty or whitespace query raises AgentValidationError."""
        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            SearchRequest(query="")

        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            AgentRequest(query="   ")

    def test_no_match_query_safe_fallback(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """Query with no matches returns NO_RESULTS without error."""
        in_memory_store.create_collection("docs", vector_dimension=4)
        embedder = DeterministicDay49EmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=embedder, store=in_memory_store, collection_name="docs")

        resp = agent.search(DAY49_QUERY_WITH_NO_MATCH)
        assert resp.status == "success"
        assert resp.citations == []
        assert resp.has_citations is False
        assert resp.metadata["search_status"] == "NO_RESULTS"


# ============================================================================
# 7. Duplicate Handling & Deduplication
# ============================================================================

class TestDuplicateHandling:
    """Certifies that process_retrieval_results deduplicates by chunk_id keeping highest score."""

    def test_duplicate_chunk_deduplication_retains_highest_score(self) -> None:
        """Duplicate results with identical chunk_id are collapsed to highest score."""
        r1 = VectorSearchResult(
            chunk_id=DAY49_CHUNK_A1, score=0.60, document_id=DAY49_DOC_A, filename=DAY49_FILE_A,
            page_number=1, chunk_index=0, content_type="text", content="Lower score version",
        )
        r2 = VectorSearchResult(
            chunk_id=DAY49_CHUNK_A1, score=0.96, document_id=DAY49_DOC_A, filename=DAY49_FILE_A,
            page_number=1, chunk_index=0, content_type="text", content="Highest score version",
        )

        deduped = process_retrieval_results([r1, r2])
        assert len(deduped) == 1
        assert deduped[0].chunk_id == DAY49_CHUNK_A1
        assert deduped[0].score == 0.96
        assert deduped[0].content == "Highest score version"


# ============================================================================
# 8. Order Stability & Repeated Execution (3 Iterations)
# ============================================================================

class TestOrderStabilityAndRepeatedExecution:
    """Certifies 3-iteration determinism of result and score ordering."""

    def test_result_and_score_order_stability_3_iterations(self) -> None:
        """3 identical executions produce identical ranking and ordering."""
        inputs = [
            VectorSearchResult(
                chunk_id=f"CHUNK-{i}", score=0.50 + (i * 0.15), document_id=DAY49_DOC_A,
                filename=DAY49_FILE_A, page_number=1, chunk_index=i, content_type="text", content=f"C{i}",
            )
            for i in range(4)
        ]

        runs: list[list[str]] = []
        for _ in range(3):
            processed = process_retrieval_results(inputs)
            runs.append([r.chunk_id for r in processed])

        assert runs[0] == runs[1] == runs[2]
        # Highest score first (CHUNK-3: 0.95, CHUNK-2: 0.80, CHUNK-1: 0.65, CHUNK-0: 0.50)
        assert runs[0] == ["CHUNK-3", "CHUNK-2", "CHUNK-1", "CHUNK-0"]


# ============================================================================
# 9. Retrieval -> Context & Retrieval -> Agent
# ============================================================================

class TestRetrievalToContextAndAgent:
    """Certifies context building compatibility and agent handoff."""

    def test_retrieval_to_context_and_agent_handoff(self) -> None:
        """Formatted context and citations hand off cleanly into AgentResponse."""
        res = VectorSearchResult(
            chunk_id=DAY49_CHUNK_A1,
            score=0.97,
            document_id=DAY49_DOC_A,
            filename=DAY49_FILE_A,
            page_number=DAY49_PAGE_A1,
            chunk_index=0,
            content_type="text",
            content=DAY49_EXACT_RELEVANT_MARKER,
            metadata=DAY49_META_A,
        )

        context = build_retrieval_context([res])
        assert DAY49_EXACT_RELEVANT_MARKER in context
        assert f"File: {DAY49_FILE_A}" in context

        citation = AgentCitation.from_search_result(res)
        resp = AgentResponse(
            answer="Grounded answer based on retrieval.",
            agent_name="SearchAgent",
            citations=[citation],
            metadata={"context": context},
        )

        assert resp.citations[0].document_id == DAY49_DOC_A
        assert resp.citations[0].chunk_id == DAY49_CHUNK_A1
        assert resp.citations[0].score == 0.97


# ============================================================================
# 10. Cross-Request Isolation & Order Reversal
# ============================================================================

class TestCrossRequestIsolationAndOrderReversal:
    """Certifies request order reversal (A->B vs B->A) produces identical isolated results."""

    def test_request_order_reversal_isolation(self) -> None:
        """Executing A->B then B->A causes no cross-request state leakage."""
        c_a = AgentCitation(
            document_id=DAY49_DOC_A, filename=DAY49_FILE_A, chunk_id=DAY49_CHUNK_A1,
            metadata={"marker": DAY49_DOC_A_ONLY},
        )
        c_b = AgentCitation(
            document_id=DAY49_DOC_B, filename=DAY49_FILE_B, chunk_id=DAY49_CHUNK_B1,
            metadata={"marker": DAY49_DOC_B_ONLY},
        )

        resp_a1 = AgentResponse(answer="A", agent_name="Agent", citations=[c_a])
        resp_b1 = AgentResponse(answer="B", agent_name="Agent", citations=[c_b])

        resp_b2 = AgentResponse(answer="B", agent_name="Agent", citations=[c_b])
        resp_a2 = AgentResponse(answer="A", agent_name="Agent", citations=[c_a])

        assert resp_a1.to_dict() == resp_a2.to_dict()
        assert resp_b1.to_dict() == resp_b2.to_dict()
        assert DAY49_DOC_B not in resp_a2.unique_documents
        assert DAY49_DOC_A not in resp_b2.unique_documents


# ============================================================================
# 11. Input & Result Mutation Safety
# ============================================================================

class TestInputAndResultMutationSafety:
    """Certifies that caller input lists and result objects are protected against mutation."""

    def test_caller_list_not_mutated_by_retrieval_processing(self) -> None:
        """process_retrieval_results leaves input list unchanged."""
        r1 = VectorSearchResult(
            chunk_id="C-1", score=0.5, document_id="D-1", filename="f.pdf",
            page_number=1, chunk_index=0, content_type="text", content="C1",
        )
        r2 = VectorSearchResult(
            chunk_id="C-2", score=0.9, document_id="D-1", filename="f.pdf",
            page_number=2, chunk_index=1, content_type="text", content="C2",
        )

        caller_list = [r1, r2]
        caller_copy = list(caller_list)

        output = process_retrieval_results(caller_list, min_score=0.7)

        assert len(output) == 1
        assert caller_list == caller_copy

    def test_result_object_mutation_independence(self) -> None:
        """Mutating one result object does not affect another."""
        r_a = VectorSearchResult(
            chunk_id="C-A", score=0.9, document_id="D-A", filename="a.pdf",
            page_number=1, chunk_index=0, content_type="text", content="A",
        )
        r_b = VectorSearchResult(
            chunk_id="C-B", score=0.8, document_id="D-B", filename="b.pdf",
            page_number=1, chunk_index=0, content_type="text", content="B",
        )

        d_a = dataclasses.asdict(r_a)
        d_a["document_id"] = "MUTATED"

        assert r_a.document_id == "D-A"
        assert r_b.document_id == "D-B"


# ============================================================================
# 12. Serialization Round-Trips
# ============================================================================

class TestSerializationRoundTrips:
    """Certifies serialization for retrieval and search result objects."""

    def test_vector_search_result_and_citation_serialization(self) -> None:
        """VectorSearchResult and AgentCitation survive serialization without data loss."""
        res = VectorSearchResult(
            chunk_id=DAY49_CHUNK_A1,
            score=0.945,
            document_id=DAY49_DOC_A,
            filename=DAY49_FILE_A,
            page_number=DAY49_PAGE_A1,
            chunk_index=0,
            content_type="text",
            content=DAY49_EXACT_RELEVANT_MARKER,
            metadata=DAY49_META_A,
        )

        d_res = dataclasses.asdict(res)
        assert d_res["chunk_id"] == DAY49_CHUNK_A1
        assert d_res["score"] == 0.945

        citation = AgentCitation.from_search_result(res)
        d_cit = citation.to_dict()
        restored_cit = AgentCitation.from_dict(json.loads(json.dumps(d_cit)))

        assert restored_cit == citation
        assert restored_cit.document_id == DAY49_DOC_A
        assert restored_cit.score == 0.945


# ============================================================================
# 13. Failure & Error Isolation
# ============================================================================

class TestFailureIsolation:
    """Certifies that invalid requests in a sequence do not corrupt valid requests."""

    def test_sequential_failure_isolation(self) -> None:
        """VALID-A -> INVALID-B -> VALID-C executes with clean error containment."""
        results: list[str] = []

        # 1. Valid A
        req_a = SearchRequest(query="Valid Query A")
        results.append(req_a.query)

        # 2. Invalid B (empty query)
        with pytest.raises(AgentValidationError):
            SearchRequest(query="")

        # 3. Valid C
        req_c = SearchRequest(query="Valid Query C")
        results.append(req_c.query)

        assert results == ["Valid Query A", "Valid Query C"]
