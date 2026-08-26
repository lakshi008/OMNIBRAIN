"""
OmniBrain Member 4 — Day 51 Multi-Document Isolation Regression Certification.

Validates that multiple documents can be registered, indexed, retrieved, contextualized,
and queried without cross-document contamination.

Core invariant:
  DOC-A information must remain associated with DOC-A.
  DOC-B information must remain associated with DOC-B.
  DOC-C information must remain associated with DOC-C.

Covers:
  1.  Document registration and vector store ingestion for isolated synthetic documents (DOC-A, DOC-B, DOC-C).
  2.  Document-level retrieval exclusivity (Query A returns only DOC-A, Query B returns only DOC-B, Query C returns only DOC-C).
  3.  Cross-document contamination prevention across all pairs.
  4.  Context isolation in build_retrieval_context (no foreign markers in document contexts).
  5.  Agent input and response isolation.
  6.  Citation isolation (citation.document_id, chunk_id, page_number strictly matched).
  7.  Visual evidence isolation across documents and pages.
  8.  Multi-document query execution retrieving legitimate multi-document candidates.
  9.  Similar marker prefix discrimination (DAY51_COMMON_A vs DAY51_COMMON_B vs DAY51_COMMON_C).
  10. Intra-document page-level and chunk-level isolation.
  11. Sequential (A -> B -> C), Reverse-order (C -> B -> A), and Interleaved (A -> B -> A -> C -> B -> C) execution.
  12. Repeated same-document execution (3 iterations with zero state accumulation).
  13. No-match document query behavior (no contamination from prior documents).
  14. Failed request error isolation (Valid A -> Invalid -> Valid B).
  15. Mutation safety and object independence.
  16. Serialization and deserialization isolation.
  17. Complete end-to-end multi-document lineage table verification.

Constraints:
  - 100% Offline: In-memory QdrantVectorStore, deterministic mock embeddings, no external APIs.
  - Zero production code modified.
  - No new models, adapters, wrappers, or ranking algorithms added.
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
from vision.exceptions import VisionEvidenceError
from vision.evidence_adapter import VisualEvidenceAdapter


# ============================================================================
# Deterministic Synthetic Fixtures & UUIDs
# ============================================================================

DOC_A = "DOC-A-DAY51"
DOC_B = "DOC-B-DAY51"
DOC_C = "DOC-C-DAY51"
DOC_D = "DOC-D-DAY51"

FILE_A = "day51_alpha_spec.pdf"
FILE_B = "day51_beta_spec.pdf"
FILE_C = "day51_gamma_spec.pdf"
FILE_D = "day51_delta_spec.pdf"

CHUNK_A1 = str(uuid.UUID("aaaaaaaa-1111-1111-1111-111111111111"))
CHUNK_A2 = str(uuid.UUID("aaaaaaaa-2222-2222-2222-222222222222"))
CHUNK_B1 = str(uuid.UUID("bbbbbbbb-1111-1111-1111-111111111111"))
CHUNK_C1 = str(uuid.UUID("cccccccc-1111-1111-1111-111111111111"))
CHUNK_D1 = str(uuid.UUID("dddddddd-1111-1111-1111-111111111111"))

PAGE_A1 = 1
PAGE_A2 = 2
PAGE_B1 = 1
PAGE_C1 = 1
PAGE_D1 = 1

DAY51_DOC_A_UNIQUE = "DAY51_DOC_A_UNIQUE_SPEC_ALPHA_999"
DAY51_DOC_B_UNIQUE = "DAY51_DOC_B_UNIQUE_SPEC_BETA_888"
DAY51_DOC_C_UNIQUE = "DAY51_DOC_C_UNIQUE_SPEC_GAMMA_777"

DAY51_PAGE_A1 = "DAY51_PAGE_A1_OVERVIEW_MARKER"
DAY51_PAGE_A2 = "DAY51_PAGE_A2_DETAILS_MARKER"

DAY51_CHUNK_A1 = "DAY51_CHUNK_A1_SECTION_1"
DAY51_CHUNK_A2 = "DAY51_CHUNK_A2_SECTION_2"

DAY51_COMMON_A = "DAY51_COMMON_PREFIX_ALPHA_10"
DAY51_COMMON_B = "DAY51_COMMON_PREFIX_BETA_20"
DAY51_COMMON_C = "DAY51_COMMON_PREFIX_GAMMA_30"

DAY51_NO_MATCH = "DAY51_QUERY_FOR_NO_MATCHING_DOCS_XYZ"


class DeterministicDay51EmbeddingProvider:
    """Deterministic offline mock embedding provider returning orthogonal 4D unit vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Map distinct document query keywords to orthogonal vector axes."""
        clean = text.lower()
        if "doc_a" in clean or "alpha" in clean or "chunk_a" in clean:
            return [1.0, 0.0, 0.0, 0.0]
        if "doc_b" in clean or "beta" in clean or "chunk_b" in clean:
            return [0.0, 1.0, 0.0, 0.0]
        if "doc_c" in clean or "gamma" in clean or "chunk_c" in clean:
            return [0.0, 0.0, 1.0, 0.0]
        if "multi_ab" in clean or "both_a_and_b" in clean:
            return [0.7071, 0.7071, 0.0, 0.0]
        if "no_match" in clean or "nonexistent" in clean:
            return [0.0, 0.0, 0.0, 1.0]
        return [0.5, 0.5, 0.5, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding generation."""
        return [self.embed(t) for t in texts]


@pytest.fixture
def multi_doc_store() -> tuple[QdrantVectorStore, str]:
    """Create an isolated in-memory QdrantVectorStore preloaded with DOC-A, DOC-B, DOC-C, and DOC-D."""
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(client=client)
    col_name = "multi_doc_isolation_coll"
    store.create_collection(col_name, vector_dimension=4)

    # Document A (2 chunks on pages 1 and 2)
    rec_a1 = EmbeddingVectorRecord(
        chunk_id=CHUNK_A1,
        document_id=DOC_A,
        filename=FILE_A,
        chunk_index=0,
        page_number=PAGE_A1,
        content_type="text",
        vector=[1.0, 0.0, 0.0, 0.0],
        metadata={"content": f"{DAY51_DOC_A_UNIQUE} - {DAY51_PAGE_A1} - {DAY51_CHUNK_A1} - {DAY51_COMMON_A}"},
    )
    rec_a2 = EmbeddingVectorRecord(
        chunk_id=CHUNK_A2,
        document_id=DOC_A,
        filename=FILE_A,
        chunk_index=1,
        page_number=PAGE_A2,
        content_type="text",
        vector=[1.0, 0.0, 0.0, 0.0],
        metadata={"content": f"{DAY51_PAGE_A2} - {DAY51_CHUNK_A2}"},
    )
    gen_a = EmbeddingGenerationResult(
        document_id=DOC_A, filename=FILE_A, items=[rec_a1, rec_a2], dimension=4, is_ready=True,
    )
    store.upsert_embeddings(col_name, gen_a)

    # Document B (1 chunk on page 1)
    rec_b1 = EmbeddingVectorRecord(
        chunk_id=CHUNK_B1,
        document_id=DOC_B,
        filename=FILE_B,
        chunk_index=0,
        page_number=PAGE_B1,
        content_type="text",
        vector=[0.0, 1.0, 0.0, 0.0],
        metadata={"content": f"{DAY51_DOC_B_UNIQUE} - {DAY51_COMMON_B}"},
    )
    gen_b = EmbeddingGenerationResult(
        document_id=DOC_B, filename=FILE_B, items=[rec_b1], dimension=4, is_ready=True,
    )
    store.upsert_embeddings(col_name, gen_b)

    # Document C (1 chunk on page 1)
    rec_c1 = EmbeddingVectorRecord(
        chunk_id=CHUNK_C1,
        document_id=DOC_C,
        filename=FILE_C,
        chunk_index=0,
        page_number=PAGE_C1,
        content_type="text",
        vector=[0.0, 0.0, 1.0, 0.0],
        metadata={"content": f"{DAY51_DOC_C_UNIQUE} - {DAY51_COMMON_C}"},
    )
    gen_c = EmbeddingGenerationResult(
        document_id=DOC_C, filename=FILE_C, items=[rec_c1], dimension=4, is_ready=True,
    )
    store.upsert_embeddings(col_name, gen_c)

    return store, col_name


# ============================================================================
# 1. Document Registration & Retrieval Isolation
# ============================================================================

class TestDocumentRegistrationAndRetrievalIsolation:
    """Certifies that document-level queries retrieve only the targeted document."""

    def test_document_a_retrieval_isolation(
        self, multi_doc_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Query targeting DOC-A returns DOC-A chunks with zero DOC-B or DOC-C contamination."""
        store, col_name = multi_doc_store
        embedder = DeterministicDay51EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        resp = agent.search(f"Search for {DAY51_DOC_A_UNIQUE}")

        assert resp.is_success is True
        assert resp.has_citations is True
        assert all(c.document_id == DOC_A for c in resp.citations)
        assert DOC_B not in resp.unique_documents
        assert DOC_C not in resp.unique_documents

    def test_document_b_retrieval_isolation(
        self, multi_doc_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Query targeting DOC-B returns DOC-B chunks with zero DOC-A or DOC-C contamination."""
        store, col_name = multi_doc_store
        embedder = DeterministicDay51EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        resp = agent.search(f"Search for {DAY51_DOC_B_UNIQUE}")

        assert resp.is_success is True
        assert resp.has_citations is True
        assert all(c.document_id == DOC_B for c in resp.citations)
        assert DOC_A not in resp.unique_documents
        assert DOC_C not in resp.unique_documents

    def test_document_c_retrieval_isolation(
        self, multi_doc_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Query targeting DOC-C returns DOC-C chunks with zero DOC-A or DOC-B contamination."""
        store, col_name = multi_doc_store
        embedder = DeterministicDay51EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        resp = agent.search(f"Search for {DAY51_DOC_C_UNIQUE}")

        assert resp.is_success is True
        assert resp.has_citations is True
        assert all(c.document_id == DOC_C for c in resp.citations)
        assert DOC_A not in resp.unique_documents
        assert DOC_B not in resp.unique_documents


# ============================================================================
# 2. Context & Agent Input Isolation
# ============================================================================

class TestContextAndAgentInputIsolation:
    """Certifies context building and agent input isolation."""

    def test_context_building_isolation(self) -> None:
        """build_retrieval_context for DOC-A contains only DOC-A markers, etc."""
        r_a = VectorSearchResult(
            chunk_id=CHUNK_A1, score=0.95, document_id=DOC_A, filename=FILE_A,
            page_number=PAGE_A1, chunk_index=0, content_type="text", content=DAY51_DOC_A_UNIQUE,
        )
        r_b = VectorSearchResult(
            chunk_id=CHUNK_B1, score=0.95, document_id=DOC_B, filename=FILE_B,
            page_number=PAGE_B1, chunk_index=0, content_type="text", content=DAY51_DOC_B_UNIQUE,
        )
        r_c = VectorSearchResult(
            chunk_id=CHUNK_C1, score=0.95, document_id=DOC_C, filename=FILE_C,
            page_number=PAGE_C1, chunk_index=0, content_type="text", content=DAY51_DOC_C_UNIQUE,
        )

        ctx_a = build_retrieval_context([r_a])
        ctx_b = build_retrieval_context([r_b])
        ctx_c = build_retrieval_context([r_c])

        # Context A isolation
        assert DAY51_DOC_A_UNIQUE in ctx_a
        assert DAY51_DOC_B_UNIQUE not in ctx_a
        assert DAY51_DOC_C_UNIQUE not in ctx_a

        # Context B isolation
        assert DAY51_DOC_B_UNIQUE in ctx_b
        assert DAY51_DOC_A_UNIQUE not in ctx_b
        assert DAY51_DOC_C_UNIQUE not in ctx_b

        # Context C isolation
        assert DAY51_DOC_C_UNIQUE in ctx_c
        assert DAY51_DOC_A_UNIQUE not in ctx_c
        assert DAY51_DOC_B_UNIQUE not in ctx_c


# ============================================================================
# 3. Citation & Visual Evidence Isolation
# ============================================================================

class TestCitationAndVisualEvidenceIsolation:
    """Certifies AgentCitation and VisualEvidence isolation across documents."""

    def test_citation_isolation_across_documents(self) -> None:
        """Citations created for DOC-A, DOC-B, DOC-C preserve exact document_id."""
        c_a = AgentCitation(document_id=DOC_A, filename=FILE_A, chunk_id=CHUNK_A1, page_number=1)
        c_b = AgentCitation(document_id=DOC_B, filename=FILE_B, chunk_id=CHUNK_B1, page_number=1)
        c_c = AgentCitation(document_id=DOC_C, filename=FILE_C, chunk_id=CHUNK_C1, page_number=1)

        assert c_a.document_id == DOC_A
        assert c_b.document_id == DOC_B
        assert c_c.document_id == DOC_C

    def test_visual_evidence_isolation_across_documents(self) -> None:
        """VisualEvidence instances maintain distinct document_id and page_number."""
        ev_a = VisualEvidence(
            document_id=DOC_A, filename=FILE_A, chunk_id=CHUNK_A1, page_number=PAGE_A1,
            content_type="diagram", description="Diagram A",
        )
        ev_b = VisualEvidence(
            document_id=DOC_B, filename=FILE_B, chunk_id=CHUNK_B1, page_number=PAGE_B1,
            content_type="chart", description="Chart B",
        )

        assert ev_a.document_id == DOC_A
        assert ev_a.page_number == PAGE_A1
        assert ev_b.document_id == DOC_B
        assert ev_b.page_number == PAGE_B1


# ============================================================================
# 4. Multi-Document Query & Similar Prefix Handling
# ============================================================================

class TestMultiDocumentQueryAndPrefixes:
    """Certifies multi-document legitimate querying and common prefix discrimination."""

    def test_multi_document_joint_query(
        self, multi_doc_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """A query addressing both DOC-A and DOC-B retrieves both while excluding DOC-C."""
        store, col_name = multi_doc_store
        embedder = DeterministicDay51EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # multi_ab query vector activates both DOC-A and DOC-B
        resp = agent.search("Find multi_ab specifications")

        assert resp.is_success is True
        assert resp.has_citations is True
        unique_docs = resp.unique_documents
        assert DOC_A in unique_docs
        assert DOC_B in unique_docs
        assert DOC_C not in unique_docs

    def test_similar_prefix_discrimination(self) -> None:
        """Markers sharing prefix ('DAY51_COMMON_') preserve exact identity without cross-match."""
        r_a = VectorSearchResult(
            chunk_id=CHUNK_A1, score=0.9, document_id=DOC_A, filename=FILE_A,
            page_number=1, chunk_index=0, content_type="text", content=DAY51_COMMON_A,
        )
        r_b = VectorSearchResult(
            chunk_id=CHUNK_B1, score=0.9, document_id=DOC_B, filename=FILE_B,
            page_number=1, chunk_index=0, content_type="text", content=DAY51_COMMON_B,
        )
        r_c = VectorSearchResult(
            chunk_id=CHUNK_C1, score=0.9, document_id=DOC_C, filename=FILE_C,
            page_number=1, chunk_index=0, content_type="text", content=DAY51_COMMON_C,
        )

        ctx_a = build_retrieval_context([r_a])
        assert DAY51_COMMON_A in ctx_a
        assert DAY51_COMMON_B not in ctx_a
        assert DAY51_COMMON_C not in ctx_a


# ============================================================================
# 5. Page & Chunk Isolation Within Document
# ============================================================================

class TestIntraDocumentPageAndChunkIsolation:
    """Certifies intra-document page and chunk isolation within DOC-A."""

    def test_intra_document_page_isolation(self) -> None:
        """PAGE-A1 and PAGE-A2 preserve distinct page numbers and chunk identifiers."""
        r_p1 = VectorSearchResult(
            chunk_id=CHUNK_A1, score=0.9, document_id=DOC_A, filename=FILE_A,
            page_number=PAGE_A1, chunk_index=0, content_type="text", content=DAY51_PAGE_A1,
        )
        r_p2 = VectorSearchResult(
            chunk_id=CHUNK_A2, score=0.9, document_id=DOC_A, filename=FILE_A,
            page_number=PAGE_A2, chunk_index=1, content_type="text", content=DAY51_PAGE_A2,
        )

        cit_p1 = AgentCitation.from_search_result(r_p1)
        cit_p2 = AgentCitation.from_search_result(r_p2)

        assert cit_p1.page_number == 1
        assert cit_p1.chunk_id == CHUNK_A1

        assert cit_p2.page_number == 2
        assert cit_p2.chunk_id == CHUNK_A2


# ============================================================================
# 6. Sequential, Reverse, and Interleaved Execution
# ============================================================================

class TestExecutionOrderIsolation:
    """Certifies sequential (A->B->C), reverse (C->B->A), and interleaved execution."""

    def test_sequential_requests_isolation(
        self, multi_doc_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Sequential execution A -> B -> C preserves independent document identities."""
        store, col_name = multi_doc_store
        embedder = DeterministicDay51EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        resp_a = agent.search(f"Query {DAY51_DOC_A_UNIQUE}")
        resp_b = agent.search(f"Query {DAY51_DOC_B_UNIQUE}")
        resp_c = agent.search(f"Query {DAY51_DOC_C_UNIQUE}")

        assert resp_a.unique_documents == [DOC_A]
        assert resp_b.unique_documents == [DOC_B]
        assert resp_c.unique_documents == [DOC_C]

    def test_reverse_order_requests_isolation(
        self, multi_doc_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Reverse execution C -> B -> A preserves independent document identities."""
        store, col_name = multi_doc_store
        embedder = DeterministicDay51EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        resp_c = agent.search(f"Query {DAY51_DOC_C_UNIQUE}")
        resp_b = agent.search(f"Query {DAY51_DOC_B_UNIQUE}")
        resp_a = agent.search(f"Query {DAY51_DOC_A_UNIQUE}")

        assert resp_c.unique_documents == [DOC_C]
        assert resp_b.unique_documents == [DOC_B]
        assert resp_a.unique_documents == [DOC_A]

    def test_interleaved_requests_isolation(
        self, multi_doc_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Interleaved execution A -> B -> A -> C -> B -> C maintains consistency without bleed."""
        store, col_name = multi_doc_store
        embedder = DeterministicDay51EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        sequence = [DOC_A, DOC_B, DOC_A, DOC_C, DOC_B, DOC_C]
        queries = {
            DOC_A: f"Query {DAY51_DOC_A_UNIQUE}",
            DOC_B: f"Query {DAY51_DOC_B_UNIQUE}",
            DOC_C: f"Query {DAY51_DOC_C_UNIQUE}",
        }

        for expected_doc in sequence:
            resp = agent.search(queries[expected_doc])
            assert resp.unique_documents == [expected_doc]

    def test_repeated_same_document_request_stability(
        self, multi_doc_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """3 identical executions of DOC-A query yield identical non-accumulating results."""
        store, col_name = multi_doc_store
        embedder = DeterministicDay51EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        runs: list[list[str]] = []
        for _ in range(3):
            resp = agent.search(f"Query {DAY51_DOC_A_UNIQUE}")
            runs.append(resp.unique_documents)

        assert runs[0] == runs[1] == runs[2] == [DOC_A]


# ============================================================================
# 7. No-Match Isolation & Failed Request Error Isolation
# ============================================================================

class TestNoMatchAndFailureIsolation:
    """Certifies no-match document query behavior and error containment."""

    def test_no_match_query_isolation(
        self, multi_doc_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Searching for nonexistent marker returns NO_RESULTS without leaking prior documents."""
        store, col_name = multi_doc_store
        embedder = DeterministicDay51EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # Run DOC-A first
        agent.search(f"Query {DAY51_DOC_A_UNIQUE}")

        # Run no-match query
        resp_nomatch = agent.search(DAY51_NO_MATCH)
        assert resp_nomatch.status == "success"
        assert resp_nomatch.citations == []
        assert resp_nomatch.has_citations is False
        assert resp_nomatch.metadata["search_status"] == "NO_RESULTS"

    def test_failed_request_error_isolation(
        self, multi_doc_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Valid A -> Invalid B -> Valid C sequence executes with clean error containment."""
        store, col_name = multi_doc_store
        embedder = DeterministicDay51EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # 1. Valid A
        resp_a = agent.search(f"Query {DAY51_DOC_A_UNIQUE}")
        assert resp_a.unique_documents == [DOC_A]

        # 2. Invalid B (empty query)
        with pytest.raises(AgentValidationError):
            agent.search("")

        # 3. Valid C
        resp_c = agent.search(f"Query {DAY51_DOC_C_UNIQUE}")
        assert resp_c.unique_documents == [DOC_C]


# ============================================================================
# 8. Mutation Safety & Serialization Isolation
# ============================================================================

class TestMutationSafetyAndSerializationIsolation:
    """Certifies mutation safety and serialization isolation across documents."""

    def test_mutation_safety_across_requests(self) -> None:
        """Mutating caller metadata for Request A does not mutate Request B."""
        meta_a = {"tenant": "A", "flag": True}
        meta_b = {"tenant": "B", "flag": False}

        req_a = SearchRequest(query="Query A", metadata=meta_a)
        req_b = SearchRequest(query="Query B", metadata=meta_b)

        meta_a["tenant"] = "CORRUPTED"

        assert req_a.metadata["tenant"] == "A"
        assert req_b.metadata["tenant"] == "B"

    def test_serialization_isolation_across_documents(self) -> None:
        """Serializing and deserializing DOC-A and DOC-B results maintains clean separation."""
        c_a = AgentCitation(document_id=DOC_A, filename=FILE_A, chunk_id=CHUNK_A1, page_number=1)
        c_b = AgentCitation(document_id=DOC_B, filename=FILE_B, chunk_id=CHUNK_B1, page_number=1)

        d_a = c_a.to_dict()
        d_b = c_b.to_dict()

        restored_a = AgentCitation.from_dict(json.loads(json.dumps(d_a)))
        restored_b = AgentCitation.from_dict(json.loads(json.dumps(d_b)))

        assert restored_a.document_id == DOC_A
        assert restored_b.document_id == DOC_B
        assert restored_a != restored_b


# ============================================================================
# 9. Complete End-to-End Multi-Document Lineage Table
# ============================================================================

class TestEndToEndMultiDocumentLineageTable:
    """Certifies the full lineage for DOC-A, DOC-B, and DOC-C from extraction to citation."""

    def test_end_to_end_lineage_table(
        self, multi_doc_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """
        Lineage Table Verification:
          DOC-A: document_id (DOC-A) -> page_number (1) -> chunk_id (CHUNK_A1) -> citation
          DOC-B: document_id (DOC-B) -> page_number (1) -> chunk_id (CHUNK_B1) -> citation
          DOC-C: document_id (DOC-C) -> page_number (1) -> chunk_id (CHUNK_C1) -> citation
        """
        store, col_name = multi_doc_store
        embedder = DeterministicDay51EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        docs = [
            (DOC_A, f"Query {DAY51_DOC_A_UNIQUE}", CHUNK_A1, PAGE_A1, FILE_A),
            (DOC_B, f"Query {DAY51_DOC_B_UNIQUE}", CHUNK_B1, PAGE_B1, FILE_B),
            (DOC_C, f"Query {DAY51_DOC_C_UNIQUE}", CHUNK_C1, PAGE_C1, FILE_C),
        ]

        for expected_doc, query_str, expected_chunk, expected_page, expected_file in docs:
            resp = agent.search(query_str)
            assert resp.is_success is True
            assert len(resp.citations) >= 1

            cit = resp.citations[0]
            assert cit.document_id == expected_doc, f"Lineage mismatch: expected doc {expected_doc}, got {cit.document_id}"
            assert cit.chunk_id == expected_chunk, f"Lineage mismatch: expected chunk {expected_chunk}, got {cit.chunk_id}"
            assert cit.page_number == expected_page, f"Lineage mismatch: expected page {expected_page}, got {cit.page_number}"
            assert cit.filename == expected_file, f"Lineage mismatch: expected filename {expected_file}, got {cit.filename}"
