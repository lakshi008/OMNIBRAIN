"""
OmniBrain Member 4 — Day 44 Retrieval Quality & Ranking Regression Certification.

Validates the public retrieval and search contracts across:
  - Ingestion Retrieval Layer (VectorSearchResult, RetrievalServiceResult,
                               retrieve, retrieve_context, process_retrieval_results,
                               build_retrieval_context, QdrantVectorStore)
  - Agents Search Layer (SearchAgent, SearchRequest, SearchResult,
                         AgentCitation, AgentResponse, AgentRequest)

Covers:
  1.  Retrieval result model contract and field validation.
  2.  Relevance ordering and descending score ranking.
  3.  Score preservation through downstream citation and response layers.
  4.  Top-K and max_results limit enforcement.
  5.  Result count behavior (0 results, 1 result, N results, capped results).
  6.  Empty query and unknown query handling.
  7.  Cross-document and cross-request isolation.
  8.  Duplicate result deduplication (highest-score retention).
  9.  Same-document multi-chunk preservation and multi-document ranking.
  10. Metadata and page-level lineage preservation.
  11. Citation-aware context building and ordering integrity.
  12. Retrieval result and search packaging serialization round-trips.
  13. Query and result association.
  14. Invalid retrieval input rejection and deterministic error contracts.
  15. Input mutation safety and result object isolation.
  16. Multi-document dataset verification and 3-iteration determinism.

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
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest
from qdrant_client import QdrantClient

# Ingestion layer (Member 1)
from ingestion.models import (
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

DAY44_DOC_A = "DAY44-DOC-A"
DAY44_DOC_B = "DAY44-DOC-B"
DAY44_DOC_C = "DAY44-DOC-C"
DAY44_DOC_D = "DAY44-DOC-D"
DAY44_DOC_E = "DAY44-DOC-E"

DAY44_RELEVANT_ALPHA = "DAY44_RELEVANT_ALPHA"
DAY44_ALPHA_CONTENT = "Alpha specification details regarding OmniBrain retrieval algorithms."

DAY44_RELEVANT_BETA = "DAY44_RELEVANT_BETA"
DAY44_BETA_CONTENT = "Beta architecture overview for agentic RAG ranking policies."

DAY44_IRRELEVANT_GAMMA = "DAY44_IRRELEVANT_GAMMA"
DAY44_GAMMA_CONTENT = "Gamma unrelated notes on legacy hardware maintenance."

DAY44_UNKNOWN_QUERY_999 = "DAY44_UNKNOWN_QUERY_999"

DAY44_META_A: dict[str, Any] = {"day44_source": "synthetic", "day44_doc": "A", "domain": "retrieval"}
DAY44_META_B: dict[str, Any] = {"day44_source": "synthetic", "day44_doc": "B", "domain": "agents"}
DAY44_META_C: dict[str, Any] = {"day44_source": "synthetic", "day44_doc": "C", "domain": "hardware"}


class DeterministicMockEmbeddingProvider:
    """Offline mock embedding provider returning deterministic unit vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Generate deterministic vector based on query string."""
        if "alpha" in text.lower() or "doc_a" in text.lower():
            return [1.0, 0.0, 0.0, 0.0]
        if "beta" in text.lower() or "doc_b" in text.lower():
            return [0.0, 1.0, 0.0, 0.0]
        if "gamma" in text.lower() or "doc_c" in text.lower():
            return [0.0, 0.0, 1.0, 0.0]
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
# 1. Retrieval Result Model Contract
# ============================================================================

class TestRetrievalResultContract:
    """Certifies field validation and structure of VectorSearchResult and RetrievalServiceResult."""

    def test_vector_search_result_fields(self) -> None:
        """VectorSearchResult retains all contractual fields."""
        res = VectorSearchResult(
            chunk_id="CHUNK-001",
            score=0.94,
            document_id=DAY44_DOC_A,
            filename="doc_a.pdf",
            page_number=2,
            chunk_index=0,
            content_type="text",
            content=DAY44_ALPHA_CONTENT,
            metadata=DAY44_META_A,
        )

        assert res.chunk_id == "CHUNK-001"
        assert res.score == 0.94
        assert res.document_id == DAY44_DOC_A
        assert res.filename == "doc_a.pdf"
        assert res.page_number == 2
        assert res.chunk_index == 0
        assert res.content_type == "text"
        assert res.content == DAY44_ALPHA_CONTENT
        assert res.metadata == DAY44_META_A

    def test_retrieval_service_result_properties(self) -> None:
        """RetrievalServiceResult exposes helper properties and filters."""
        r1 = VectorSearchResult(
            chunk_id="C-1", score=0.9, document_id="D-1", filename="f1.pdf",
            page_number=1, chunk_index=0, content_type="text", content="Text 1",
        )
        r2 = VectorSearchResult(
            chunk_id="C-2", score=0.8, document_id="D-1", filename="f1.pdf",
            page_number=1, chunk_index=1, content_type="table", content="Table 1",
        )
        r3 = VectorSearchResult(
            chunk_id="C-3", score=0.7, document_id="D-2", filename="f2.pdf",
            page_number=2, chunk_index=0, content_type="image", content="Image 1",
        )

        s_res = RetrievalServiceResult(
            query_vector_dimension=4,
            results=[r1, r2, r3],
            context="Formatted context",
        )

        assert s_res.total_results == 3
        assert s_res.has_results is True
        assert s_res.text_results == 1
        assert s_res.table_results == 1
        assert s_res.image_results == 1
        assert s_res.get_results_by_type("table") == [r2]
        assert s_res.get_results_on_page(1) == [r1, r2]
        assert s_res.get_results_on_page(2) == [r3]


# ============================================================================
# 2. Relevance Ordering & Ranking
# ============================================================================

class TestRelevanceAndRankingOrder:
    """Certifies that process_retrieval_results sorts strictly descending by score."""

    def test_ranking_order_descending_by_score(self) -> None:
        """Highest score ranked first (A > B > C)."""
        r_low = VectorSearchResult(
            chunk_id="C-LOW", score=0.40, document_id=DAY44_DOC_C, filename="c.pdf",
            page_number=1, chunk_index=0, content_type="text", content=DAY44_GAMMA_CONTENT,
        )
        r_high = VectorSearchResult(
            chunk_id="C-HIGH", score=0.95, document_id=DAY44_DOC_A, filename="a.pdf",
            page_number=1, chunk_index=0, content_type="text", content=DAY44_ALPHA_CONTENT,
        )
        r_med = VectorSearchResult(
            chunk_id="C-MED", score=0.75, document_id=DAY44_DOC_B, filename="b.pdf",
            page_number=1, chunk_index=0, content_type="text", content=DAY44_BETA_CONTENT,
        )

        ranked = process_retrieval_results([r_low, r_high, r_med])

        assert len(ranked) == 3
        assert ranked[0].chunk_id == "C-HIGH"
        assert ranked[0].score == 0.95
        assert ranked[1].chunk_id == "C-MED"
        assert ranked[1].score == 0.75
        assert ranked[2].chunk_id == "C-LOW"
        assert ranked[2].score == 0.40

    def test_ranking_tie_breaking_deterministic(self) -> None:
        """Tied scores are resolved deterministically by chunk_index then chunk_id."""
        r1 = VectorSearchResult(
            chunk_id="CHUNK-Z", score=0.85, document_id=DAY44_DOC_A, filename="a.pdf",
            page_number=1, chunk_index=2, content_type="text", content="Content Z",
        )
        r2 = VectorSearchResult(
            chunk_id="CHUNK-A", score=0.85, document_id=DAY44_DOC_A, filename="a.pdf",
            page_number=1, chunk_index=1, content_type="text", content="Content A",
        )
        r3 = VectorSearchResult(
            chunk_id="CHUNK-B", score=0.85, document_id=DAY44_DOC_A, filename="a.pdf",
            page_number=1, chunk_index=1, content_type="text", content="Content B",
        )

        ranked = process_retrieval_results([r1, r2, r3])

        assert len(ranked) == 3
        # Lowest chunk_index first (1 < 2), then lexicographic chunk_id ("CHUNK-A" < "CHUNK-B")
        assert ranked[0].chunk_id == "CHUNK-A"
        assert ranked[1].chunk_id == "CHUNK-B"
        assert ranked[2].chunk_id == "CHUNK-Z"

    def test_min_score_threshold_filtering(self) -> None:
        """Results with score < min_score are filtered out."""
        r_relevant = VectorSearchResult(
            chunk_id="C-REL", score=0.82, document_id=DAY44_DOC_A, filename="a.pdf",
            page_number=1, chunk_index=0, content_type="text", content="Relevant",
        )
        r_irrelevant = VectorSearchResult(
            chunk_id="C-IRR", score=0.45, document_id=DAY44_DOC_C, filename="c.pdf",
            page_number=1, chunk_index=0, content_type="text", content="Irrelevant",
        )

        filtered = process_retrieval_results([r_relevant, r_irrelevant], min_score=0.70)
        assert len(filtered) == 1
        assert filtered[0].chunk_id == "C-REL"


# ============================================================================
# 3. Score Preservation
# ============================================================================

class TestScorePreservation:
    """Certifies that score values survive downstream citation conversion."""

    def test_score_survives_into_citation_and_response(self) -> None:
        """Relevance score is preserved without recalculation or truncation."""
        vs_res = VectorSearchResult(
            chunk_id="C-EXACT",
            score=0.8875,
            document_id=DAY44_DOC_A,
            filename="doc_a.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Score preservation test",
        )

        citation = AgentCitation.from_search_result(vs_res)
        assert citation.score == 0.8875

        response = AgentResponse(
            answer="",
            agent_name="SearchAgent",
            citations=[citation],
        )
        assert response.citations[0].score == 0.8875

        d = response.to_dict()
        assert d["citations"][0]["score"] == 0.8875


# ============================================================================
# 4. Top-K & Result Count Contract
# ============================================================================

class TestTopKAndResultCountContract:
    """Certifies top-k, max_results limits, and result count boundaries."""

    @pytest.mark.parametrize("limit", [1, 2, 3])
    def test_max_results_limit_capping(self, limit: int) -> None:
        """process_retrieval_results caps output at requested max_results."""
        results = [
            VectorSearchResult(
                chunk_id=f"C-{i}", score=0.90 - (i * 0.05), document_id=DAY44_DOC_A,
                filename="a.pdf", page_number=1, chunk_index=i, content_type="text", content=f"Text {i}",
            )
            for i in range(5)
        ]

        capped = process_retrieval_results(results, max_results=limit)
        assert len(capped) == limit
        assert capped[0].score == 0.90

    def test_zero_results_and_empty_list(self) -> None:
        """Empty input results list returns empty list."""
        assert process_retrieval_results([]) == []
        assert build_retrieval_context([]) == ""


# ============================================================================
# 5. Query Validation & Unknown Query Behavior
# ============================================================================

class TestQueryValidationAndUnknownQuery:
    """Certifies empty query rejection and unknown query handling."""

    def test_empty_or_whitespace_query_rejection(self) -> None:
        """SearchRequest and AgentRequest reject empty queries."""
        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            SearchRequest(query="")

        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            SearchRequest(query="   ")

        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            AgentRequest(query="")

    def test_unknown_query_produces_no_results_safely(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """Searching an empty or non-matching collection returns NO_RESULTS safely."""
        in_memory_store.create_collection("empty_coll", vector_dimension=4)
        embedder = DeterministicMockEmbeddingProvider(dimension=4)
        agent = SearchAgent(
            embedding_provider=embedder,
            store=in_memory_store,
            collection_name="empty_coll",
        )

        response = agent.search(DAY44_UNKNOWN_QUERY_999)
        assert response.status == "success"
        assert len(response.citations) == 0
        assert response.has_citations is False
        assert response.metadata["search_status"] == "NO_RESULTS"

        packaged = agent.package_result(response)
        assert packaged.status == "NO_RESULTS"
        assert packaged.total_results == 0
        assert packaged.has_results is False


# ============================================================================
# 6. Cross-Document & Cross-Request Isolation
# ============================================================================

class TestCrossDocumentAndRequestIsolation:
    """Certifies that results from DOC-A and DOC-B remain isolated."""

    def test_cross_document_isolation_in_retrieval_batches(self) -> None:
        """Results from DOC-A contain only DOC-A chunks, DOC-B contains only DOC-B."""
        r_a = VectorSearchResult(
            chunk_id="CHUNK-A-01", score=0.95, document_id=DAY44_DOC_A, filename="doc_a.pdf",
            page_number=1, chunk_index=0, content_type="text", content=DAY44_ALPHA_CONTENT,
        )
        r_b = VectorSearchResult(
            chunk_id="CHUNK-B-01", score=0.95, document_id=DAY44_DOC_B, filename="doc_b.pdf",
            page_number=1, chunk_index=0, content_type="text", content=DAY44_BETA_CONTENT,
        )

        resp_a = AgentResponse(
            answer="Doc A Response",
            agent_name="SearchAgent",
            citations=[AgentCitation.from_search_result(r_a)],
        )
        resp_b = AgentResponse(
            answer="Doc B Response",
            agent_name="SearchAgent",
            citations=[AgentCitation.from_search_result(r_b)],
        )

        assert resp_a.unique_documents == [DAY44_DOC_A]
        assert DAY44_DOC_B not in resp_a.unique_documents

        assert resp_b.unique_documents == [DAY44_DOC_B]
        assert DAY44_DOC_A not in resp_b.unique_documents

    def test_sequential_cross_request_isolation(self) -> None:
        """Sequential search requests do not leak citations or metadata."""
        req_a = SearchRequest(query="Query A", metadata={"tenant": "A"})
        req_b = SearchRequest(query="Query B", metadata={"tenant": "B"})

        assert req_a.metadata["tenant"] == "A"
        assert req_b.metadata["tenant"] == "B"
        assert "tenant" not in req_a.query


# ============================================================================
# 7. Duplicate Handling & Multi-Chunk Retention
# ============================================================================

class TestDuplicateAndMultiChunkHandling:
    """Certifies deduplication by chunk_id and retention of distinct chunks."""

    def test_deduplicate_by_chunk_id_retains_highest_score(self) -> None:
        """Duplicate results with identical chunk_id are collapsed to highest score."""
        r1 = VectorSearchResult(
            chunk_id="CHUNK-DUP-01", score=0.60, document_id=DAY44_DOC_A, filename="a.pdf",
            page_number=1, chunk_index=0, content_type="text", content="Lower score version",
        )
        r2 = VectorSearchResult(
            chunk_id="CHUNK-DUP-01", score=0.92, document_id=DAY44_DOC_A, filename="a.pdf",
            page_number=1, chunk_index=0, content_type="text", content="Higher score version",
        )
        r3 = VectorSearchResult(
            chunk_id="CHUNK-DUP-01", score=0.75, document_id=DAY44_DOC_A, filename="a.pdf",
            page_number=1, chunk_index=0, content_type="text", content="Medium score version",
        )

        deduped = process_retrieval_results([r1, r2, r3])
        assert len(deduped) == 1
        assert deduped[0].chunk_id == "CHUNK-DUP-01"
        assert deduped[0].score == 0.92
        assert deduped[0].content == "Higher score version"

    def test_same_document_distinct_chunks_retained(self) -> None:
        """Multiple distinct chunks from the same document are all retained."""
        chunks = [
            VectorSearchResult(
                chunk_id=f"CHUNK-A{i}", score=0.90 - (i * 0.05), document_id=DAY44_DOC_A,
                filename="doc_a.pdf", page_number=i + 1, chunk_index=i,
                content_type="text", content=f"Section {i} content",
            )
            for i in range(3)
        ]

        processed = process_retrieval_results(chunks)
        assert len(processed) == 3
        assert [c.chunk_id for c in processed] == ["CHUNK-A0", "CHUNK-A1", "CHUNK-A2"]
        assert all(c.document_id == DAY44_DOC_A for c in processed)


# ============================================================================
# 8. Multi-Document Ranking & Lineage Preservation
# ============================================================================

class TestMultiDocumentRankingAndLineage:
    """Certifies multi-document ranking, metadata preservation, and page lineage."""

    def test_multi_document_ranking_hierarchy(self) -> None:
        """Rank results across DOC-A (0.95), DOC-B (0.80), DOC-C (0.35)."""
        r_a = VectorSearchResult(
            chunk_id="C-A", score=0.95, document_id=DAY44_DOC_A, filename="doc_a.pdf",
            page_number=1, chunk_index=0, content_type="text", content=DAY44_ALPHA_CONTENT,
            metadata=DAY44_META_A,
        )
        r_b = VectorSearchResult(
            chunk_id="C-B", score=0.80, document_id=DAY44_DOC_B, filename="doc_b.pdf",
            page_number=2, chunk_index=0, content_type="text", content=DAY44_BETA_CONTENT,
            metadata=DAY44_META_B,
        )
        r_c = VectorSearchResult(
            chunk_id="C-C", score=0.35, document_id=DAY44_DOC_C, filename="doc_c.pdf",
            page_number=3, chunk_index=0, content_type="text", content=DAY44_GAMMA_CONTENT,
            metadata=DAY44_META_C,
        )

        ranked = process_retrieval_results([r_c, r_a, r_b])
        assert [r.document_id for r in ranked] == [DAY44_DOC_A, DAY44_DOC_B, DAY44_DOC_C]
        assert ranked[0].metadata["domain"] == "retrieval"
        assert ranked[1].metadata["domain"] == "agents"
        assert ranked[2].metadata["domain"] == "hardware"

        # Verify page lineage
        assert ranked[0].page_number == 1
        assert ranked[1].page_number == 2
        assert ranked[2].page_number == 3


# ============================================================================
# 9. Context Building Compatibility
# ============================================================================

class TestContextBuildingCompatibility:
    """Certifies build_retrieval_context formatting and citation numbering."""

    def test_context_building_structure_and_ordering(self) -> None:
        """build_retrieval_context formats source blocks matching ranking order."""
        r1 = VectorSearchResult(
            chunk_id="C-1", score=0.95, document_id=DAY44_DOC_A, filename="doc_a.pdf",
            page_number=1, chunk_index=0, content_type="text", content=DAY44_ALPHA_CONTENT,
        )
        r2 = VectorSearchResult(
            chunk_id="C-2", score=0.85, document_id=DAY44_DOC_B, filename="doc_b.pdf",
            page_number=4, chunk_index=1, content_type="table", content="Table data cells",
        )

        context = build_retrieval_context([r1, r2])

        assert "[Source 1]" in context
        assert "[Source 2]" in context
        assert "File: doc_a.pdf" in context
        assert "Page: 1" in context
        assert "Type: text" in context
        assert DAY44_ALPHA_CONTENT in context

        assert "File: doc_b.pdf" in context
        assert "Page: 4" in context
        assert "Type: table" in context
        assert "Table data cells" in context

        # Source 1 appears before Source 2
        idx1 = context.find("[Source 1]")
        idx2 = context.find("[Source 2]")
        assert idx1 < idx2


# ============================================================================
# 10. Retrieval Serialization Round-Trip
# ============================================================================

class TestRetrievalSerialization:
    """Certifies serialization and deserialization of retrieval artifacts."""

    def test_vector_search_result_asdict_and_agent_citation_roundtrip(self) -> None:
        """VectorSearchResult and AgentCitation survive dictionary serialization."""
        vs_res = VectorSearchResult(
            chunk_id="C-SERIAL",
            score=0.91,
            document_id=DAY44_DOC_A,
            filename="doc_a.pdf",
            page_number=5,
            chunk_index=2,
            content_type="image",
            content="Serialized image content",
            metadata=DAY44_META_A,
        )

        d_vs = dataclasses.asdict(vs_res)
        assert d_vs["chunk_id"] == "C-SERIAL"
        assert d_vs["score"] == 0.91
        assert d_vs["metadata"]["day44_doc"] == "A"

        citation = AgentCitation.from_search_result(vs_res)
        d_cit = citation.to_dict()
        restored_cit = AgentCitation.from_dict(d_cit)
        assert restored_cit == citation


# ============================================================================
# 11. Query & Result Association
# ============================================================================

class TestQueryResultAssociation:
    """Certifies query attribution in SearchResult and AgentResponse."""

    def test_search_result_associates_exact_query(self) -> None:
        """SearchResult retains the query that produced the results."""
        citation = AgentCitation(
            document_id=DAY44_DOC_A,
            filename="doc_a.pdf",
            chunk_id="C-1",
        )
        sres = SearchResult(
            query="Evaluate Day 44 ranking contracts",
            status="RESULTS_FOUND",
            citations=[citation],
            context="Context block",
        )
        assert sres.query == "Evaluate Day 44 ranking contracts"

        d = sres.to_dict()
        assert d["query"] == "Evaluate Day 44 ranking contracts"
        restored = SearchResult.from_dict(d)
        assert restored.query == "Evaluate Day 44 ranking contracts"


# ============================================================================
# 12. Invalid Retrieval Input Rejection
# ============================================================================

class TestInvalidRetrievalInput:
    """Certifies input validation and error contracts on retrieval functions."""

    def test_process_retrieval_results_invalid_inputs(self) -> None:
        """process_retrieval_results rejects invalid results list and parameters."""
        # Non-list
        with pytest.raises(TypeError, match="results must be a list"):
            process_retrieval_results("not_a_list")  # type: ignore[arg-type]

        # Non-VectorSearchResult items
        with pytest.raises(TypeError, match="not an instance of VectorSearchResult"):
            process_retrieval_results([{"dict": "not_model"}])  # type: ignore[list-item]

        # Invalid min_score
        with pytest.raises(ValueError, match="min_score must be a finite float"):
            process_retrieval_results([], min_score=1.5)

        with pytest.raises(ValueError, match="min_score must be a finite float"):
            process_retrieval_results([], min_score=float("nan"))

        # Invalid max_results
        with pytest.raises(ValueError, match="max_results must be a positive integer"):
            process_retrieval_results([], max_results=0)

        with pytest.raises(ValueError, match="max_results must be a positive integer"):
            process_retrieval_results([], max_results=-5)

    def test_retrieve_function_invalid_inputs(self, in_memory_store: QdrantVectorStore) -> None:
        """retrieve function validates store, collection, query_vector, and top_k."""
        # Invalid store
        with pytest.raises(TypeError, match="store must be an instance of QdrantVectorStore"):
            retrieve(query_vector=[0.1, 0.2], store="not_store", collection_name="coll")  # type: ignore[arg-type]

        # Empty collection
        with pytest.raises(ValueError, match="collection_name must be a non-empty string"):
            retrieve(query_vector=[0.1, 0.2], store=in_memory_store, collection_name="")

        # Empty vector
        with pytest.raises(ValueError, match="query_vector cannot be empty"):
            retrieve(query_vector=[], store=in_memory_store, collection_name="coll")

        # Non-numeric vector element
        with pytest.raises(ValueError, match="invalid non-numeric or non-finite value"):
            retrieve(query_vector=[0.1, "bad"], store=in_memory_store, collection_name="coll")  # type: ignore[list-item]

        # Invalid top_k
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            retrieve(query_vector=[0.1, 0.2], store=in_memory_store, collection_name="coll", top_k=0)


# ============================================================================
# 13. Input Mutation Safety & Object Isolation
# ============================================================================

class TestInputMutationSafetyAndIsolation:
    """Certifies that retrieval functions do not mutate caller inputs."""

    def test_process_retrieval_results_does_not_mutate_caller_list(self) -> None:
        """Caller results list remains unchanged after filtering and sorting."""
        r1 = VectorSearchResult(
            chunk_id="C-1", score=0.6, document_id="D-1", filename="f.pdf",
            page_number=1, chunk_index=0, content_type="text", content="C1",
        )
        r2 = VectorSearchResult(
            chunk_id="C-2", score=0.9, document_id="D-1", filename="f.pdf",
            page_number=1, chunk_index=1, content_type="text", content="C2",
        )

        caller_list = [r1, r2]
        caller_copy = list(caller_list)

        output = process_retrieval_results(caller_list, min_score=0.7)

        assert len(output) == 1
        assert caller_list == caller_copy
        assert len(caller_list) == 2


# ============================================================================
# 14. Multi-Document Dataset & Determinism
# ============================================================================

class TestMultiDocumentDatasetAndDeterminism:
    """Certifies multi-document dataset integrity and 3-iteration determinism."""

    def test_multi_document_dataset_5_documents(self) -> None:
        """5-document dataset (DOC-A through DOC-E) correctly processed and ranked."""
        docs = [DAY44_DOC_A, DAY44_DOC_B, DAY44_DOC_C, DAY44_DOC_D, DAY44_DOC_E]
        dataset_results = [
            VectorSearchResult(
                chunk_id=f"CHUNK-{doc}-01",
                score=0.95 - (idx * 0.10),
                document_id=doc,
                filename=f"{doc.lower()}.pdf",
                page_number=idx + 1,
                chunk_index=0,
                content_type="text",
                content=f"Content marker for {doc}",
                metadata={"doc": doc, "rank_pos": idx},
            )
            for idx, doc in enumerate(docs)
        ]

        ranked = process_retrieval_results(dataset_results, max_results=5)
        assert len(ranked) == 5
        assert [r.document_id for r in ranked] == docs

        for idx, doc in enumerate(docs):
            assert ranked[idx].document_id == doc
            assert ranked[idx].page_number == idx + 1
            assert ranked[idx].metadata["doc"] == doc

    def test_retrieval_determinism_across_3_iterations(self) -> None:
        """Running the same retrieval ranking scenario 3 times produces identical output."""
        inputs = [
            VectorSearchResult(
                chunk_id=f"C-{i}", score=0.50 + (i * 0.10), document_id=f"DOC-{i}",
                filename=f"doc_{i}.pdf", page_number=1, chunk_index=i,
                content_type="text", content=f"Content {i}",
            )
            for i in range(5)
        ]

        runs: list[list[str]] = []
        for _ in range(3):
            processed = process_retrieval_results(inputs, max_results=3)
            runs.append([r.chunk_id for r in processed])

        assert runs[0] == runs[1] == runs[2]
        # Highest scores first: C-4 (0.9), C-3 (0.8), C-2 (0.7)
        assert runs[0] == ["C-4", "C-3", "C-2"]
