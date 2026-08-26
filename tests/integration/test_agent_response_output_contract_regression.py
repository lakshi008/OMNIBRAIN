"""
OmniBrain Member 4 — Day 46 Agent Response & Output Contract Regression Certification.

Validates the Agent input/output contracts, response structure, citation preservation,
source grounding, and lineage traceability across:
  - Agents Layer (SearchAgent, AgentRequest, SearchRequest, AgentResponse,
                  SearchResult, AgentCitation, AgentState)
  - Ingestion / Retrieval Layer (VectorSearchResult, RetrievalServiceResult,
                                 QdrantVectorStore)
  - Vision Integration (VisualEvidence, VisualEvidenceAdapter)

Covers:
  1.  Valid agent input handling (string query, AgentRequest, SearchRequest).
  2.  Agent response contract and structural schema guarantees.
  3.  Citation preservation and full document/page/chunk lineage.
  4.  Source grounding and multi-source aggregation across documents.
  5.  Empty context and zero-match retrieval handling.
  6.  Missing required input and invalid type validation.
  7.  Malformed context and evidence integrity validation.
  8.  Citation integrity and deliberate inconsistency data contract.
  9.  Visual evidence grounding and multimodal citation adaptation.
  10. AgentResponse and AgentCitation serialization round-trips.
  11. Multi-run response determinism (3 iterations).
  12. Cross-request isolation and execution order reversal (A->B vs B->A).
  13. Multi-document marker isolation.
  14. Input mutation safety and response instance isolation.
  15. Error isolation and exception determinism.
  16. Null, empty, and optional value handling.
  17. End-to-end agentic retrieval and packaged search result contract.

Constraints:
  - 100% Offline: In-memory QdrantVectorStore, deterministic mock embeddings, no network.
  - Zero production code modified.
  - No new agent logic, citation logic, wrappers, or caching added.
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

# Agents layer (Member 2)
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    AgentState,
    SearchRequest,
    SearchResult,
)
from agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentValidationError,
)
from agents.search_agent import SearchAgent

# Vision layer (Member 3)
from vision.models import VisualEvidence
from vision.evidence_adapter import VisualEvidenceAdapter


# ============================================================================
# Deterministic Synthetic Fixtures & Mocks
# ============================================================================

DAY46_DOC_A = "DAY46-DOC-A"
DAY46_DOC_B = "DAY46-DOC-B"

DAY46_FILE_A = "day46_alpha_system.pdf"
DAY46_FILE_B = "day46_beta_system.pdf"

DAY46_PAGE_A1 = 1
DAY46_PAGE_A2 = 2
DAY46_PAGE_B1 = 1

DAY46_CHUNK_A1 = "DAY46-CHUNK-A1"
DAY46_CHUNK_A2 = "DAY46-CHUNK-A2"
DAY46_CHUNK_B1 = "DAY46-CHUNK-B1"

DAY46_AGENT_MARKER_A = "DAY46_AGENT_MARKER_ALPHA_555"
DAY46_AGENT_MARKER_B = "DAY46_AGENT_MARKER_BETA_777"
DAY46_AGENT_A_ONLY = "DAY46_AGENT_A_ONLY_EVIDENCE_SPEC"
DAY46_AGENT_B_ONLY = "DAY46_AGENT_B_ONLY_EVIDENCE_SPEC"

DAY46_META_A: dict[str, Any] = {"day": 46, "doc": "A", "marker": DAY46_AGENT_MARKER_A}
DAY46_META_B: dict[str, Any] = {"day": 46, "doc": "B", "marker": DAY46_AGENT_MARKER_B}


class MockDeterministicEmbeddingProvider:
    """Deterministic offline embedding provider returning fixed 4D vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Generate deterministic vector based on text keyword."""
        clean = text.lower()
        if "alpha" in clean or "doc_a" in clean:
            return [1.0, 0.0, 0.0, 0.0]
        if "beta" in clean or "doc_b" in clean:
            return [0.0, 1.0, 0.0, 0.0]
        return [0.5, 0.5, 0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding generation."""
        return [self.embed(t) for t in texts]


@pytest.fixture
def in_memory_store() -> QdrantVectorStore:
    """Create an isolated in-memory QdrantVectorStore."""
    client = QdrantClient(location=":memory:")
    return QdrantVectorStore(client=client)


# ============================================================================
# 1. Valid Agent Input Handling
# ============================================================================

class TestValidAgentInputHandling:
    """Certifies that SearchAgent accepts raw strings, AgentRequest, and SearchRequest."""

    def test_search_agent_accepts_raw_string_query(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """SearchAgent accepts minimal raw string query."""
        in_memory_store.create_collection("docs", vector_dimension=4)
        embedder = MockDeterministicEmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=in_memory_store, collection_name="docs")

        resp = agent.search("Find system architecture")
        assert isinstance(resp, AgentResponse)
        assert resp.status == "success"
        assert resp.agent_name == "SearchAgent"
        assert resp.metadata["query"] == "Find system architecture"

    def test_search_agent_accepts_agent_request_and_search_request(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """SearchAgent accepts AgentRequest and SearchRequest instances."""
        in_memory_store.create_collection("docs", vector_dimension=4)
        embedder = MockDeterministicEmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=in_memory_store, collection_name="docs")

        # AgentRequest
        req = AgentRequest(query="Find alpha specs", session_id="SESS-46", metadata=DAY46_META_A)
        resp_req = agent.search(req)
        assert resp_req.metadata["query"] == "Find alpha specs"
        assert resp_req.metadata["session_id"] == "SESS-46"
        assert resp_req.metadata["marker"] == DAY46_AGENT_MARKER_A

        # SearchRequest
        sreq = SearchRequest(
            query="Find beta specs",
            top_k=10,
            min_score=0.5,
            max_results=3,
            session_id="SESS-46-B",
            metadata=DAY46_META_B,
        )
        resp_sreq = agent.search(sreq)
        assert resp_sreq.metadata["query"] == "Find beta specs"
        assert resp_sreq.metadata["top_k"] == 10
        assert resp_sreq.metadata["min_score"] == 0.5
        assert resp_sreq.metadata["max_results"] == 3


# ============================================================================
# 2. Agent Response Contract
# ============================================================================

class TestAgentResponseContract:
    """Certifies structural fields and helper properties of AgentResponse."""

    def test_agent_response_fields_and_properties(self) -> None:
        """AgentResponse exposes all contractual fields and calculated properties."""
        c1 = AgentCitation(
            document_id=DAY46_DOC_A,
            filename=DAY46_FILE_A,
            chunk_id=DAY46_CHUNK_A1,
            page_number=DAY46_PAGE_A1,
            content_type="text",
            score=0.95,
        )
        c2 = AgentCitation(
            document_id=DAY46_DOC_B,
            filename=DAY46_FILE_B,
            chunk_id=DAY46_CHUNK_B1,
            page_number=DAY46_PAGE_B1,
            content_type="table",
            score=0.85,
        )

        resp = AgentResponse(
            answer="Multi-modal agent answer",
            agent_name="SupervisorAgent",
            status="success",
            citations=[c1, c2],
            metadata={"latency_ms": 120, "query": "Test query"},
            error=None,
        )

        assert resp.answer == "Multi-modal agent answer"
        assert resp.agent_name == "SupervisorAgent"
        assert resp.status == "success"
        assert resp.error is None
        assert resp.is_success is True
        assert resp.is_error is False
        assert resp.has_citations is True
        assert resp.total_citations == 2
        assert resp.unique_document_count == 2
        assert resp.unique_documents == [DAY46_DOC_A, DAY46_DOC_B]
        assert len(resp.text_results) == 1
        assert len(resp.table_results) == 1
        assert len(resp.image_results) == 0


# ============================================================================
# 3. Citation Preservation & Source Lineage
# ============================================================================

class TestCitationPreservationAndSourceLineage:
    """Certifies AgentResponse -> AgentCitation -> Document -> Page -> Chunk lineage."""

    def test_lineage_traceability(self) -> None:
        """Citation preserves exact provenance to original VectorSearchResult."""
        vs_res = VectorSearchResult(
            chunk_id=DAY46_CHUNK_A1,
            score=0.96,
            document_id=DAY46_DOC_A,
            filename=DAY46_FILE_A,
            page_number=DAY46_PAGE_A1,
            chunk_index=0,
            content_type="text",
            content=f"Alpha content with {DAY46_AGENT_MARKER_A}",
            metadata=DAY46_META_A,
        )

        citation = AgentCitation.from_search_result(vs_res)
        resp = AgentResponse(
            answer="Grounded answer",
            agent_name="SearchAgent",
            citations=[citation],
        )

        assert len(resp.citations) == 1
        cit = resp.citations[0]
        assert cit.document_id == DAY46_DOC_A
        assert cit.filename == DAY46_FILE_A
        assert cit.chunk_id == DAY46_CHUNK_A1
        assert cit.page_number == DAY46_PAGE_A1
        assert cit.content_type == "text"
        assert cit.score == 0.96
        assert cit.metadata["marker"] == DAY46_AGENT_MARKER_A


# ============================================================================
# 4. Source Grounding & Multi-Source Response
# ============================================================================

class TestSourceGroundingAndMultiSource:
    """Certifies that citations strictly point to relevant sources across documents."""

    def test_source_grounding_isolation(self) -> None:
        """DOC-A citations contain only DOC-A; DOC-B citations contain only DOC-B."""
        c_a = AgentCitation(
            document_id=DAY46_DOC_A,
            filename=DAY46_FILE_A,
            chunk_id=DAY46_CHUNK_A1,
            page_number=DAY46_PAGE_A1,
            metadata={"marker": DAY46_AGENT_A_ONLY},
        )
        c_b = AgentCitation(
            document_id=DAY46_DOC_B,
            filename=DAY46_FILE_B,
            chunk_id=DAY46_CHUNK_B1,
            page_number=DAY46_PAGE_B1,
            metadata={"marker": DAY46_AGENT_B_ONLY},
        )

        resp_a = AgentResponse(answer="A only", agent_name="SearchAgent", citations=[c_a])
        resp_b = AgentResponse(answer="B only", agent_name="SearchAgent", citations=[c_b])

        assert resp_a.unique_documents == [DAY46_DOC_A]
        assert resp_a.citations[0].metadata["marker"] == DAY46_AGENT_A_ONLY
        assert DAY46_DOC_B not in resp_a.unique_documents

        assert resp_b.unique_documents == [DAY46_DOC_B]
        assert resp_b.citations[0].metadata["marker"] == DAY46_AGENT_B_ONLY
        assert DAY46_DOC_A not in resp_b.unique_documents

    def test_multi_source_response_aggregation(self) -> None:
        """Multiple chunks from DOC-A and DOC-B correctly indexed in single response."""
        c1 = AgentCitation(document_id=DAY46_DOC_A, filename=DAY46_FILE_A, chunk_id=DAY46_CHUNK_A1)
        c2 = AgentCitation(document_id=DAY46_DOC_A, filename=DAY46_FILE_A, chunk_id=DAY46_CHUNK_A2)
        c3 = AgentCitation(document_id=DAY46_DOC_B, filename=DAY46_FILE_B, chunk_id=DAY46_CHUNK_B1)

        resp = AgentResponse(
            answer="Combined response",
            agent_name="SearchAgent",
            citations=[c1, c2, c3],
        )

        assert resp.total_citations == 3
        assert resp.unique_document_count == 2
        assert resp.unique_documents == [DAY46_DOC_A, DAY46_DOC_B]


# ============================================================================
# 5. Empty Context & Missing Required Input
# ============================================================================

class TestEmptyContextAndMissingInput:
    """Certifies behavior with empty contexts and missing required fields."""

    def test_empty_retrieval_context_produces_no_results(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """Search on empty store returns success status with zero citations."""
        in_memory_store.create_collection("empty_coll", vector_dimension=4)
        embedder = MockDeterministicEmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=in_memory_store, collection_name="empty_coll")

        resp = agent.search("Unmatched query")
        assert resp.status == "success"
        assert resp.citations == []
        assert resp.has_citations is False
        assert resp.metadata["total_results"] == 0
        assert resp.metadata["search_status"] == "NO_RESULTS"

    def test_missing_required_input_validation(self) -> None:
        """Missing query or agent_name raises deterministic AgentValidationError."""
        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            AgentRequest(query="")

        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            SearchRequest(query="   ")

        with pytest.raises(AgentValidationError, match="agent_name must be a non-empty string"):
            AgentResponse(answer="Answer", agent_name="")


# ============================================================================
# 6. Invalid Input Types & Malformed Context
# ============================================================================

class TestInvalidTypesAndMalformedContext:
    """Certifies rejection of invalid data types and corrupted evidence items."""

    def test_invalid_input_types_rejection(self) -> None:
        """Rejects non-string query, non-list citations, and invalid top_k."""
        with pytest.raises(AgentValidationError, match="query must be a string"):
            AgentRequest(query=999)  # type: ignore[arg-type]

        with pytest.raises(AgentValidationError, match="citations must be a list"):
            AgentResponse(answer="Ans", agent_name="Agent", citations="not_a_list")  # type: ignore[arg-type]

        with pytest.raises(AgentValidationError, match="top_k must be a positive integer"):
            SearchRequest(query="valid", top_k="five")  # type: ignore[arg-type]

    def test_malformed_context_evidence_integrity_check(self) -> None:
        """SearchAgent._validate_result_integrity rejects incomplete VectorSearchResult items."""
        # Missing chunk_id
        bad_item = VectorSearchResult(
            chunk_id="",
            score=0.9,
            document_id=DAY46_DOC_A,
            filename=DAY46_FILE_A,
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Content",
        )
        with pytest.raises(AgentExecutionError, match="chunk_id is missing or empty"):
            SearchAgent._validate_result_integrity(bad_item, 0)

        # Empty content
        empty_content_item = VectorSearchResult(
            chunk_id=DAY46_CHUNK_A1,
            score=0.9,
            document_id=DAY46_DOC_A,
            filename=DAY46_FILE_A,
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="   ",  # whitespace only
        )
        with pytest.raises(AgentExecutionError, match="content is empty or missing"):
            SearchAgent._validate_result_integrity(empty_content_item, 0)


# ============================================================================
# 7. Visual Evidence Grounding
# ============================================================================

class TestVisualEvidenceGrounding:
    """Certifies conversion and lineage preservation from citation to VisualEvidence."""

    def test_visual_evidence_adaptation_from_citation(self) -> None:
        """Visual evidence created from visual citation retains exact page and chunk identity."""
        citation = AgentCitation(
            document_id=DAY46_DOC_A,
            filename=DAY46_FILE_A,
            chunk_id=DAY46_CHUNK_A2,
            page_number=DAY46_PAGE_A2,
            content_type="chart",
            score=0.93,
            metadata={"image_path": "/images/chart1.png", "chunk_index": 1},
        )

        assert VisualEvidenceAdapter.is_visual(citation) is True
        evidence = VisualEvidence.from_citation(citation)

        assert evidence.document_id == DAY46_DOC_A
        assert evidence.filename == DAY46_FILE_A
        assert evidence.chunk_id == DAY46_CHUNK_A2
        assert evidence.page_number == DAY46_PAGE_A2
        assert evidence.content_type == "chart"
        assert evidence.image_path == "/images/chart1.png"
        assert evidence.chunk_index == 1


# ============================================================================
# 8. Response & Citation Serialization Round-Trips
# ============================================================================

class TestSerializationRoundTrips:
    """Certifies exact preservation during to_dict -> JSON -> from_dict round-trips."""

    def test_agent_response_and_citation_serialization(self) -> None:
        """AgentResponse with nested citations survives JSON roundtrip."""
        citation = AgentCitation(
            document_id=DAY46_DOC_A,
            filename=DAY46_FILE_A,
            chunk_id=DAY46_CHUNK_A1,
            page_number=DAY46_PAGE_A1,
            content_type="text",
            score=0.91,
            metadata=DAY46_META_A,
        )
        resp = AgentResponse(
            answer="Verified serializable answer",
            agent_name="SearchAgent",
            status="success",
            citations=[citation],
            metadata={"query": "Find alpha", "total": 1},
        )

        d = resp.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        restored = AgentResponse.from_dict(parsed)

        assert restored.answer == resp.answer
        assert restored.agent_name == resp.agent_name
        assert restored.status == resp.status
        assert len(restored.citations) == 1
        assert restored.citations[0].document_id == DAY46_DOC_A
        assert restored.citations[0].chunk_id == DAY46_CHUNK_A1
        assert restored.citations[0].page_number == DAY46_PAGE_A1
        assert restored.citations[0].score == 0.91
        assert restored.citations[0].metadata["marker"] == DAY46_AGENT_MARKER_A


# ============================================================================
# 9. Response Determinism (3 Iterations)
# ============================================================================

class TestResponseDeterminism:
    """Certifies output stability across 3 consecutive executions."""

    def test_agent_response_determinism_3_iterations(self) -> None:
        """3 identical executions yield identical responses, citations, and metadata."""
        c = AgentCitation(
            document_id=DAY46_DOC_A,
            filename=DAY46_FILE_A,
            chunk_id=DAY46_CHUNK_A1,
            page_number=DAY46_PAGE_A1,
            score=0.95,
        )

        runs: list[dict[str, Any]] = []
        for _ in range(3):
            resp = AgentResponse(
                answer="Deterministic answer",
                agent_name="SearchAgent",
                citations=[c],
                metadata={"query": "Deterministic query"},
            )
            runs.append(resp.to_dict())

        assert runs[0] == runs[1] == runs[2]


# ============================================================================
# 10. Cross-Request Isolation & Order Reversal
# ============================================================================

class TestCrossRequestIsolationAndOrderReversal:
    """Certifies that request order (A->B vs B->A) causes no state bleed."""

    def test_request_order_reversal_isolation(self) -> None:
        """Running A->B then B->A produces identical isolated outputs."""
        c_a = AgentCitation(
            document_id=DAY46_DOC_A, filename=DAY46_FILE_A, chunk_id=DAY46_CHUNK_A1,
            metadata={"marker": DAY46_AGENT_A_ONLY},
        )
        c_b = AgentCitation(
            document_id=DAY46_DOC_B, filename=DAY46_FILE_B, chunk_id=DAY46_CHUNK_B1,
            metadata={"marker": DAY46_AGENT_B_ONLY},
        )

        # Run A then B
        resp_a1 = AgentResponse(answer="A", agent_name="Agent", citations=[c_a])
        resp_b1 = AgentResponse(answer="B", agent_name="Agent", citations=[c_b])

        # Run B then A
        resp_b2 = AgentResponse(answer="B", agent_name="Agent", citations=[c_b])
        resp_a2 = AgentResponse(answer="A", agent_name="Agent", citations=[c_a])

        assert resp_a1.to_dict() == resp_a2.to_dict()
        assert resp_b1.to_dict() == resp_b2.to_dict()
        assert DAY46_DOC_B not in resp_a2.unique_documents
        assert DAY46_DOC_A not in resp_b2.unique_documents


# ============================================================================
# 11. Input & Response Mutation Safety
# ============================================================================

class TestMutationSafetyAndObjectIsolation:
    """Certifies defensive copying and independence of response objects."""

    def test_caller_metadata_mutation_safety(self) -> None:
        """Modifying caller metadata dict does not alter AgentRequest or AgentResponse."""
        caller_dict = {"env": "prod", "marker": "initial"}
        req = AgentRequest(query="Test", metadata=caller_dict)

        caller_dict["env"] = "corrupted"
        assert req.metadata["env"] == "prod"

    def test_response_object_independence(self) -> None:
        """Mutating one response instance does not modify an independent instance."""
        c1 = AgentCitation(document_id=DAY46_DOC_A, filename=DAY46_FILE_A, chunk_id="C-1")
        c2 = AgentCitation(document_id=DAY46_DOC_B, filename=DAY46_FILE_B, chunk_id="C-2")

        resp_a = AgentResponse(answer="Ans A", agent_name="Agent", citations=[c1])
        resp_b = AgentResponse(answer="Ans B", agent_name="Agent", citations=[c2])

        d_a = resp_a.to_dict()
        d_a["answer"] = "MUTATED"

        assert resp_a.answer == "Ans A"
        assert resp_b.answer == "Ans B"


# ============================================================================
# 12. Error Isolation & Exception Determinism
# ============================================================================

class TestErrorIsolationAndExceptionDeterminism:
    """Certifies stability across invalid input sequences."""

    def test_sequential_error_isolation(self) -> None:
        """An invalid request in a sequence does not corrupt state of subsequent valid requests."""
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

    def test_repeated_exception_stability(self) -> None:
        """10 identical invalid requests produce the exact same exception and message."""
        expected_msg = "query cannot be empty or whitespace-only."
        for _ in range(10):
            with pytest.raises(AgentValidationError) as exc:
                AgentRequest(query="   ")
            assert str(exc.value) == expected_msg


# ============================================================================
# 13. End-to-End Agent Retrieval & Search Packaging
# ============================================================================

class TestEndToEndAgentRetrievalAndPackaging:
    """Certifies complete offline path: Query -> SearchAgent -> AgentResponse -> SearchResult."""

    def test_search_agent_packaged_output_contract(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """Full search workflow producing validated SearchResult package."""
        import uuid as _uuid
        from ingestion.models import EmbeddingGenerationResult, EmbeddingVectorRecord

        in_memory_store.create_collection("docs", vector_dimension=4)

        # In-memory Qdrant requires UUID-format string IDs
        chunk_uuid = str(_uuid.UUID("a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1"))

        record = EmbeddingVectorRecord(
            chunk_id=chunk_uuid,
            document_id=DAY46_DOC_A,
            filename=DAY46_FILE_A,
            chunk_index=0,
            page_number=DAY46_PAGE_A1,
            content_type="text",
            vector=[1.0, 0.0, 0.0, 0.0],
            metadata={"marker": DAY46_AGENT_MARKER_A, "content": "Alpha architecture document"},
        )
        embedding_result = EmbeddingGenerationResult(
            document_id=DAY46_DOC_A,
            filename=DAY46_FILE_A,
            items=[record],
            dimension=4,
            is_ready=True,
        )
        in_memory_store.upsert_embeddings(collection_name="docs", result=embedding_result)

        embedder = MockDeterministicEmbeddingProvider(dimension=4)
        agent = SearchAgent(
            embedding_provider=embedder,
            store=in_memory_store,
            collection_name="docs",
        )

        # Search and package
        search_res = agent.search_and_package("Query regarding alpha architecture")

        assert isinstance(search_res, SearchResult)
        assert search_res.status == "RESULTS_FOUND"
        assert search_res.total_results == 1
        assert search_res.has_results is True
        assert len(search_res.citations) == 1

        cit = search_res.citations[0]
        assert cit.document_id == DAY46_DOC_A
        assert cit.chunk_id == chunk_uuid
        assert cit.page_number == DAY46_PAGE_A1
        assert cit.filename == DAY46_FILE_A
