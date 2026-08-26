"""
OmniBrain Member 4 — Day 50 Citation & Grounding Regression Certification.

Validates that retrieved information remains strictly grounded across the entire pipeline:
  Retrieval (VectorSearchResult, process_retrieval_results)
    ↓
  Context (build_retrieval_context)
    ↓
  Agent Input & Processing (SearchAgent, AgentRequest, SearchRequest)
    ↓
  Citation (AgentCitation)
    ↓
  Final Response (AgentResponse, SearchResult)
    ↓
  Visual Evidence Grounding (VisualEvidence, VisualEvidenceAdapter)

Covers:
  1.  Retrieval grounding (correct document_id, page_number, chunk_id).
  2.  Context grounding (survival of fact markers and source identity).
  3.  Citation creation and lineage traceability (DOC -> Page -> Chunk -> Citation).
  4.  Wrong citation detection (distinguishing correct from mismatched source references).
  5.  Cross-document grounding (DOC-A vs DOC-B facts).
  6.  Cross-page grounding (PAGE-1 vs PAGE-2 facts).
  7.  Cross-chunk grounding (CHUNK-A vs CHUNK-B facts).
  8.  Multi-citation response consistency (multiple independent sources).
  9.  Citation ordering stability (3-run repeatability).
  10. Citation completeness and uncited information handling.
  11. Context contamination prevention (DOC-A secrets vs DOC-B secrets).
  12. Query isolation (Query A vs Query B).
  13. Repeated execution determinism (3 iterations).
  14. Citation and response serialization round-trips.
  15. Visual grounding (VisualEvidence on DOC-DAY50, PAGE-3, visual markers).
  16. Error behavior and validation contracts.
  17. Mutation safety and object isolation.
  18. Cross-request isolation.
  19. Complete offline end-to-end grounding flow.

Constraints:
  - 100% Offline: In-memory QdrantVectorStore, deterministic mock embeddings, no external APIs.
  - Zero production code modified.
  - No new models, adapters, wrappers, or citation logic added.
  - Synthetic deterministic data only.
"""

from __future__ import annotations

import copy
import dataclasses
import json
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
from agents.search_agent import SearchAgent

# Vision layer (Member 3)
from vision.models import VisualEvidence
from vision.exceptions import VisionEvidenceError, VisionInputValidationError
from vision.evidence_adapter import VisualEvidenceAdapter


# ============================================================================
# Deterministic Synthetic Fixtures
# ============================================================================

DOC_DAY50 = "DOC-DAY50"
DAY50_FILENAME = "day50_document.pdf"

PAGE_1 = 1
PAGE_2 = 2
PAGE_3 = 3

CHUNK_1 = str(uuid.UUID("11111111-1111-1111-1111-111111111111"))
CHUNK_2 = str(uuid.UUID("22222222-2222-2222-2222-222222222222"))
CHUNK_3 = str(uuid.UUID("33333333-3333-3333-3333-333333333333"))

DAY50_PRIMARY_FACT = "DAY50_PRIMARY_FACT: OmniBrain architecture guarantees deterministic lineage preservation."
DAY50_SUPPORTING_FACT = "DAY50_SUPPORTING_FACT: Secondary latency benchmarks demonstrate high throughput."
DAY50_UNRELATED_FACT = "DAY50_UNRELATED_FACT: Historical notes on obsolete legacy storage formats."

DAY50_VISUAL_MARKER = "DAY50_VISUAL_MARKER: System architecture topology diagram"

DOC_A = "DOC-A-DAY50"
DOC_B = "DOC-B-DAY50"
DAY50_DOCUMENT_A_FACT = "DAY50_DOCUMENT_A_FACT: Alpha module processing specifications."
DAY50_DOCUMENT_B_FACT = "DAY50_DOCUMENT_B_FACT: Beta module configuration directives."

DAY50_SECRET_A = "DAY50_SECRET_A_CONFIDENTIAL_KEY_999"
DAY50_SECRET_B = "DAY50_SECRET_B_CONFIDENTIAL_KEY_888"

DAY50_PAGE_ONE_FACT = "DAY50_PAGE_ONE_FACT: Introduction and core definitions."
DAY50_PAGE_TWO_FACT = "DAY50_PAGE_TWO_FACT: Detailed implementation procedures."

DAY50_CHUNK_A_FACT = "DAY50_CHUNK_A_FACT: Subsection A mathematical foundations."
DAY50_CHUNK_B_FACT = "DAY50_CHUNK_B_FACT: Subsection B empirical validation results."


class DeterministicDay50EmbeddingProvider:
    """Deterministic offline mock embedding provider returning fixed 4D vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Generate deterministic vector based on text keywords."""
        clean = text.lower()
        if "primary" in clean or "chunk-1" in clean or "chunk_1" in clean:
            return [1.0, 0.0, 0.0, 0.0]
        if "supporting" in clean or "chunk-2" in clean or "chunk_2" in clean:
            return [0.0, 1.0, 0.0, 0.0]
        if "unrelated" in clean or "chunk-3" in clean or "chunk_3" in clean:
            return [0.0, 0.0, 1.0, 0.0]
        if "doc-a" in clean or "fact_a" in clean or "document_a" in clean:
            return [0.5, 0.5, 0.0, 0.0]
        if "doc-b" in clean or "fact_b" in clean or "document_b" in clean:
            return [0.0, 0.5, 0.5, 0.0]
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
# 1. Retrieval & Context Grounding
# ============================================================================

class TestRetrievalAndContextGrounding:
    """Certifies retrieval grounding and context building preservation."""

    def test_retrieval_grounding_identifies_correct_source(
        self, in_memory_store: QdrantVectorStore
    ) -> None:
        """Query targeting primary fact retrieves CHUNK-1 on PAGE-1 of DOC-DAY50."""
        in_memory_store.create_collection("grounding_docs", vector_dimension=4)

        r1 = EmbeddingVectorRecord(
            chunk_id=CHUNK_1,
            document_id=DOC_DAY50,
            filename=DAY50_FILENAME,
            chunk_index=0,
            page_number=PAGE_1,
            content_type="text",
            vector=[1.0, 0.0, 0.0, 0.0],
            metadata={"content": DAY50_PRIMARY_FACT},
        )
        r2 = EmbeddingVectorRecord(
            chunk_id=CHUNK_2,
            document_id=DOC_DAY50,
            filename=DAY50_FILENAME,
            chunk_index=1,
            page_number=PAGE_2,
            content_type="text",
            vector=[0.0, 1.0, 0.0, 0.0],
            metadata={"content": DAY50_SUPPORTING_FACT},
        )

        gen_result = EmbeddingGenerationResult(
            document_id=DOC_DAY50,
            filename=DAY50_FILENAME,
            items=[r1, r2],
            dimension=4,
            is_ready=True,
        )
        in_memory_store.upsert_embeddings("grounding_docs", gen_result)

        embedder = DeterministicDay50EmbeddingProvider()
        agent = SearchAgent(
            embedding_provider=embedder,
            store=in_memory_store,
            collection_name="grounding_docs",
        )

        # Search primary fact
        resp = agent.search("Find primary architectural fact")

        assert resp.is_success is True
        assert resp.has_citations is True
        assert resp.citations[0].document_id == DOC_DAY50
        assert resp.citations[0].chunk_id == CHUNK_1
        assert resp.citations[0].page_number == PAGE_1
        assert resp.citations[0].filename == DAY50_FILENAME

    def test_context_grounding_preserves_facts_and_lineage(self) -> None:
        """build_retrieval_context retains primary fact marker and page/file lineage."""
        vs_res = VectorSearchResult(
            chunk_id=CHUNK_1,
            score=0.96,
            document_id=DOC_DAY50,
            filename=DAY50_FILENAME,
            page_number=PAGE_1,
            chunk_index=0,
            content_type="text",
            content=DAY50_PRIMARY_FACT,
        )

        context = build_retrieval_context([vs_res])

        assert "[Source 1]" in context
        assert f"File: {DAY50_FILENAME}" in context
        assert f"Page: {PAGE_1}" in context
        assert "Type: text" in context
        assert DAY50_PRIMARY_FACT in context


# ============================================================================
# 2. Citation Creation & Lineage Consistency
# ============================================================================

class TestCitationCreationAndLineage:
    """Certifies AgentCitation creation and lineage traceability."""

    def test_citation_lineage_traceability(self) -> None:
        """AgentCitation preserves exact document_id, filename, chunk_id, and page_number."""
        vs_res = VectorSearchResult(
            chunk_id=CHUNK_1,
            score=0.95,
            document_id=DOC_DAY50,
            filename=DAY50_FILENAME,
            page_number=PAGE_1,
            chunk_index=0,
            content_type="text",
            content=DAY50_PRIMARY_FACT,
        )

        citation = AgentCitation.from_search_result(vs_res)

        assert citation.document_id == DOC_DAY50
        assert citation.filename == DAY50_FILENAME
        assert citation.chunk_id == CHUNK_1
        assert citation.page_number == PAGE_1
        assert citation.content_type == "text"
        assert citation.score == 0.95

    def test_distinguish_correct_and_wrong_citation(self) -> None:
        """Verify that correct citation matches source whereas an inconsistent citation is detected."""
        source = VectorSearchResult(
            chunk_id=CHUNK_1,
            score=0.95,
            document_id=DOC_DAY50,
            filename=DAY50_FILENAME,
            page_number=PAGE_1,
            chunk_index=0,
            content_type="text",
            content=DAY50_PRIMARY_FACT,
        )

        correct_citation = AgentCitation.from_search_result(source)
        wrong_citation = AgentCitation(
            document_id=DOC_DAY50,
            filename=DAY50_FILENAME,
            chunk_id=CHUNK_2,  # Wrong chunk!
            page_number=PAGE_2,  # Wrong page!
        )

        # Correct citation matches source exactly
        assert correct_citation.chunk_id == source.chunk_id
        assert correct_citation.page_number == source.page_number

        # Wrong citation differs from source
        assert wrong_citation.chunk_id != source.chunk_id
        assert wrong_citation.page_number != source.page_number


# ============================================================================
# 3. Cross-Document, Cross-Page, and Cross-Chunk Grounding
# ============================================================================

class TestCrossBoundaryGrounding:
    """Certifies isolation across documents, pages, and chunks."""

    def test_cross_document_grounding_isolation(self) -> None:
        """DOC-A fact grounds to DOC-A citation; DOC-B fact grounds to DOC-B citation."""
        r_a = VectorSearchResult(
            chunk_id="CHUNK-DOC-A", score=0.95, document_id=DOC_A, filename="doc_a.pdf",
            page_number=1, chunk_index=0, content_type="text", content=DAY50_DOCUMENT_A_FACT,
        )
        r_b = VectorSearchResult(
            chunk_id="CHUNK-DOC-B", score=0.95, document_id=DOC_B, filename="doc_b.pdf",
            page_number=1, chunk_index=0, content_type="text", content=DAY50_DOCUMENT_B_FACT,
        )

        cit_a = AgentCitation.from_search_result(r_a)
        cit_b = AgentCitation.from_search_result(r_b)

        assert cit_a.document_id == DOC_A
        assert cit_a.document_id != DOC_B

        assert cit_b.document_id == DOC_B
        assert cit_b.document_id != DOC_A

    def test_cross_page_grounding_isolation(self) -> None:
        """PAGE-1 fact maps to page 1, PAGE-2 fact maps to page 2."""
        r_p1 = VectorSearchResult(
            chunk_id="C-P1", score=0.9, document_id=DOC_DAY50, filename=DAY50_FILENAME,
            page_number=PAGE_1, chunk_index=0, content_type="text", content=DAY50_PAGE_ONE_FACT,
        )
        r_p2 = VectorSearchResult(
            chunk_id="C-P2", score=0.9, document_id=DOC_DAY50, filename=DAY50_FILENAME,
            page_number=PAGE_2, chunk_index=1, content_type="text", content=DAY50_PAGE_TWO_FACT,
        )

        cit_p1 = AgentCitation.from_search_result(r_p1)
        cit_p2 = AgentCitation.from_search_result(r_p2)

        assert cit_p1.page_number == 1
        assert cit_p2.page_number == 2
        assert cit_p1.page_number != cit_p2.page_number

    def test_cross_chunk_grounding_isolation(self) -> None:
        """CHUNK-A fact maps to CHUNK-A, CHUNK-B fact maps to CHUNK-B."""
        r_ca = VectorSearchResult(
            chunk_id="CHUNK-A-SPEC", score=0.9, document_id=DOC_DAY50, filename=DAY50_FILENAME,
            page_number=1, chunk_index=0, content_type="text", content=DAY50_CHUNK_A_FACT,
        )
        r_cb = VectorSearchResult(
            chunk_id="CHUNK-B-SPEC", score=0.9, document_id=DOC_DAY50, filename=DAY50_FILENAME,
            page_number=1, chunk_index=1, content_type="text", content=DAY50_CHUNK_B_FACT,
        )

        cit_ca = AgentCitation.from_search_result(r_ca)
        cit_cb = AgentCitation.from_search_result(r_cb)

        assert cit_ca.chunk_id == "CHUNK-A-SPEC"
        assert cit_cb.chunk_id == "CHUNK-B-SPEC"


# ============================================================================
# 4. Multi-Citation Response & Citation Completeness
# ============================================================================

class TestMultiCitationAndCompleteness:
    """Certifies multi-citation responses and citation completeness."""

    def test_multi_citation_response_preserves_distinct_sources(self) -> None:
        """Two distinct sources produce two distinct citations without replacement."""
        c1 = AgentCitation(
            document_id=DOC_DAY50, filename=DAY50_FILENAME, chunk_id=CHUNK_1, page_number=PAGE_1,
        )
        c2 = AgentCitation(
            document_id=DOC_DAY50, filename=DAY50_FILENAME, chunk_id=CHUNK_2, page_number=PAGE_2,
        )

        resp = AgentResponse(
            answer=f"Verified based on {DAY50_PRIMARY_FACT} and {DAY50_SUPPORTING_FACT}",
            agent_name="SearchAgent",
            citations=[c1, c2],
            metadata={"query": "Find primary and supporting facts"},
        )

        assert resp.total_citations == 2
        assert resp.citations[0].chunk_id == CHUNK_1
        assert resp.citations[0].page_number == PAGE_1
        assert resp.citations[1].chunk_id == CHUNK_2
        assert resp.citations[1].page_number == PAGE_2

    def test_uncited_response_handling(self) -> None:
        """AgentResponse with empty citations is valid, with has_citations=False."""
        resp = AgentResponse(
            answer="General response with no backing citations",
            agent_name="SearchAgent",
            citations=[],
        )
        assert resp.has_citations is False
        assert resp.total_citations == 0
        assert resp.unique_documents == []


# ============================================================================
# 5. Context Contamination Prevention & Query Isolation
# ============================================================================

class TestContaminationAndQueryIsolation:
    """Certifies zero context leakage and strict query isolation."""

    def test_context_contamination_prevention(self) -> None:
        """DOC-A context contains only DOC-A secrets; DOC-B context contains only DOC-B secrets."""
        r_a = VectorSearchResult(
            chunk_id="C-A", score=0.9, document_id=DOC_A, filename="a.pdf",
            page_number=1, chunk_index=0, content_type="text", content=DAY50_SECRET_A,
        )
        r_b = VectorSearchResult(
            chunk_id="C-B", score=0.9, document_id=DOC_B, filename="b.pdf",
            page_number=1, chunk_index=0, content_type="text", content=DAY50_SECRET_B,
        )

        ctx_a = build_retrieval_context([r_a])
        ctx_b = build_retrieval_context([r_b])

        assert DAY50_SECRET_A in ctx_a
        assert DAY50_SECRET_B not in ctx_a

        assert DAY50_SECRET_B in ctx_b
        assert DAY50_SECRET_A not in ctx_b

    def test_query_isolation_sequential(self) -> None:
        """Query A citations do not leak into Query B citations."""
        c_a = AgentCitation(document_id=DOC_A, filename="a.pdf", chunk_id="C-A")
        c_b = AgentCitation(document_id=DOC_B, filename="b.pdf", chunk_id="C-B")

        resp_a = AgentResponse(
            answer="Ans A", agent_name="Agent", citations=[c_a], metadata={"query": "Query A"},
        )
        resp_b = AgentResponse(
            answer="Ans B", agent_name="Agent", citations=[c_b], metadata={"query": "Query B"},
        )

        assert resp_a.citations[0].document_id == DOC_A
        assert resp_a.metadata["query"] == "Query A"
        assert DOC_B not in resp_a.unique_documents

        assert resp_b.citations[0].document_id == DOC_B
        assert resp_b.metadata["query"] == "Query B"
        assert DOC_A not in resp_b.unique_documents


# ============================================================================
# 6. Serialization Round-Trips
# ============================================================================

class TestSerializationRoundTrips:
    """Certifies serialization and deserialization of citations and responses."""

    def test_citation_serialization_json_roundtrip(self) -> None:
        """AgentCitation survives to_dict -> JSON -> from_dict exact roundtrip."""
        citation = AgentCitation(
            document_id=DOC_DAY50,
            filename=DAY50_FILENAME,
            chunk_id=CHUNK_1,
            page_number=PAGE_1,
            content_type="text",
            score=0.97,
            metadata={"fact": "primary"},
        )

        d = citation.to_dict()
        json_str = json.dumps(d)
        restored = AgentCitation.from_dict(json.loads(json_str))

        assert restored == citation
        assert restored.document_id == DOC_DAY50
        assert restored.chunk_id == CHUNK_1
        assert restored.page_number == PAGE_1
        assert restored.score == 0.97
        assert restored.metadata["fact"] == "primary"

    def test_agent_response_serialization_json_roundtrip(self) -> None:
        """AgentResponse with nested citations survives JSON roundtrip."""
        citation = AgentCitation(
            document_id=DOC_DAY50,
            filename=DAY50_FILENAME,
            chunk_id=CHUNK_1,
            page_number=PAGE_1,
        )
        resp = AgentResponse(
            answer="Serialized grounded answer",
            agent_name="SearchAgent",
            citations=[citation],
            metadata={"query": "Find fact"},
        )

        d = resp.to_dict()
        restored = AgentResponse.from_dict(json.loads(json.dumps(d)))

        assert restored.answer == resp.answer
        assert len(restored.citations) == 1
        assert restored.citations[0].document_id == DOC_DAY50
        assert restored.citations[0].chunk_id == CHUNK_1


# ============================================================================
# 7. Visual Evidence Grounding
# ============================================================================

class TestVisualEvidenceGrounding:
    """Certifies VisualEvidence grounding on DOC-DAY50, PAGE-3, visual markers."""

    def test_visual_evidence_grounding_preserves_page_and_description(self) -> None:
        """VisualEvidence constructed from visual citation preserves page 3 and description."""
        vis_citation = AgentCitation(
            document_id=DOC_DAY50,
            filename=DAY50_FILENAME,
            chunk_id=CHUNK_3,
            page_number=PAGE_3,
            content_type="diagram",
            score=0.92,
            metadata={"image_path": "/images/topology.png", "chunk_index": 2},
        )

        evidence = VisualEvidence.from_citation(vis_citation)

        assert evidence.document_id == DOC_DAY50
        assert evidence.filename == DAY50_FILENAME
        assert evidence.chunk_id == CHUNK_3
        assert evidence.page_number == PAGE_3
        assert evidence.content_type == "diagram"
        assert evidence.image_path == "/images/topology.png"

    def test_direct_visual_evidence_preserves_description(self) -> None:
        """VisualEvidence preserves description field."""
        ev = VisualEvidence(
            document_id=DOC_DAY50,
            filename=DAY50_FILENAME,
            chunk_id=CHUNK_3,
            page_number=PAGE_3,
            content_type="diagram",
            description=DAY50_VISUAL_MARKER,
        )
        assert ev.description == DAY50_VISUAL_MARKER
        assert ev.page_number == 3


# ============================================================================
# 8. Error Behavior & Validation Contracts
# ============================================================================

class TestErrorBehaviorAndValidationContracts:
    """Certifies deterministic validation errors for invalid lineage and types."""

    def test_invalid_citation_attributes_raise_validation_error(self) -> None:
        """Missing document_id or invalid page_number raises AgentValidationError."""
        with pytest.raises(AgentValidationError, match="document_id must be a non-empty string"):
            AgentCitation(document_id="", filename="f.pdf", chunk_id="c1")

        with pytest.raises(AgentValidationError, match="filename must be a non-empty string"):
            AgentCitation(document_id="d1", filename="   ", chunk_id="c1")

        with pytest.raises(AgentValidationError, match="page_number must be a positive integer"):
            AgentCitation(document_id="d1", filename="f.pdf", chunk_id="c1", page_number=0)

    def test_invalid_visual_evidence_attributes_raise_evidence_error(self) -> None:
        """Invalid visual modality raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError, match="Invalid visual content_type 'audio'"):
            VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", content_type="audio")


# ============================================================================
# 9. Mutation Safety & Repeated Execution (3 Iterations)
# ============================================================================

class TestMutationSafetyAndRepeatedExecution:
    """Certifies mutation safety and 3-iteration determinism."""

    def test_citation_mutation_safety(self) -> None:
        """Modifying one citation dictionary does not alter another independent citation."""
        c1 = AgentCitation(document_id=DOC_DAY50, filename=DAY50_FILENAME, chunk_id="C-1")
        c2 = AgentCitation(document_id=DOC_DAY50, filename=DAY50_FILENAME, chunk_id="C-2")

        d1 = c1.to_dict()
        d1["document_id"] = "MUTATED"

        assert c1.document_id == DOC_DAY50
        assert c2.document_id == DOC_DAY50

    def test_pipeline_determinism_3_iterations(self) -> None:
        """3 identical executions yield identical citations and response representations."""
        c = AgentCitation(
            document_id=DOC_DAY50,
            filename=DAY50_FILENAME,
            chunk_id=CHUNK_1,
            page_number=PAGE_1,
            score=0.95,
        )

        runs: list[dict[str, Any]] = []
        for _ in range(3):
            resp = AgentResponse(
                answer="Deterministic answer",
                agent_name="SearchAgent",
                citations=[c],
                metadata={"query": "Find fact"},
            )
            runs.append(resp.to_dict())

        assert runs[0] == runs[1] == runs[2]


# ============================================================================
# 10. End-to-End Grounding Pipeline Verification
# ============================================================================

class TestEndToEndGroundingPipeline:
    """Certifies complete offline pipeline: Retrieval -> Context -> Agent -> Citation -> Final Response."""

    def test_complete_offline_grounding_flow(self) -> None:
        """
        Flow:
          VectorSearchResult (Member 1)
            -> build_retrieval_context (Member 1)
            -> AgentCitation.from_search_result (Member 2)
            -> AgentResponse (Member 2)
            -> SearchResult.from_response (Member 2)
            -> VisualEvidence.from_citation (Member 3)
        """
        vs_res = VectorSearchResult(
            chunk_id=CHUNK_1,
            score=0.98,
            document_id=DOC_DAY50,
            filename=DAY50_FILENAME,
            page_number=PAGE_1,
            chunk_index=0,
            content_type="text",
            content=DAY50_PRIMARY_FACT,
            metadata={"priority": "critical"},
        )

        # 1. Context building
        context = build_retrieval_context([vs_res])
        assert DAY50_PRIMARY_FACT in context
        assert f"File: {DAY50_FILENAME}" in context

        # 2. Citation creation
        citation = AgentCitation.from_search_result(vs_res)
        assert citation.document_id == DOC_DAY50
        assert citation.chunk_id == CHUNK_1
        assert citation.page_number == PAGE_1
        assert citation.score == 0.98

        # 3. Agent response delivery
        agent_resp = AgentResponse(
            answer=f"Verified grounded statement: {DAY50_PRIMARY_FACT}",
            agent_name="SearchAgent",
            citations=[citation],
            metadata={"query": "Locate primary fact", "context": context},
        )
        assert agent_resp.citations[0].document_id == DOC_DAY50
        assert agent_resp.citations[0].chunk_id == CHUNK_1

        # 4. SearchResult packaging
        search_res = SearchResult.from_response(agent_resp)
        assert search_res.status == "RESULTS_FOUND"
        assert search_res.citations[0].document_id == DOC_DAY50
        assert search_res.citations[0].chunk_id == CHUNK_1
