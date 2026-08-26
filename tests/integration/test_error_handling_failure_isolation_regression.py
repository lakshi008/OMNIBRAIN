"""
OmniBrain Member 4 — Day 53 Error Handling & Failure Isolation Regression Certification.

Validates that errors are handled deterministically across the OmniBrain pipeline and that
failed requests do not corrupt or affect subsequent successful requests.

Core Invariants:
  1. Failed requests fail cleanly and deterministically according to established contracts.
  2. Failed requests leave zero residual state or partial mutations in shared fixtures/stores.
  3. Subsequent and concurrent valid requests succeed with complete lineage integrity.

Covers:
  1.  Valid baseline execution (DOC-A, DOC-B, DOC-C).
  2.  Invalid input rejection across models (empty query, invalid page, missing ID).
  3.  Failed request -> Successful request transition.
  4.  Success -> Failure -> Success sequence isolation.
  5.  Cross-document failure isolation (DOC-A failure does not bleed into DOC-B).
  6.  Retrieval error isolation (vector search failure -> valid search success).
  7.  Context error isolation (build_retrieval_context invalid type -> valid context building).
  8.  Agent error isolation (SearchAgent invalid input -> valid input).
  9.  Citation error isolation (AgentCitation validation failure -> valid citation creation).
  10. Serialization error isolation (from_dict corruption -> valid round-trip).
  11. Repeated failure determinism (identical errors on repeated invalid calls).
  12. Multi-failure recovery (Invalid -> Invalid -> Valid A -> Valid B).
  13. Interleaved failure and success execution (A-valid -> B-invalid -> C-valid -> A-invalid -> B-valid -> C-invalid).
  14. Concurrent failure isolation (Valid A + Invalid B + Valid C in ThreadPoolExecutor).
  15. Context contamination check with secret tokens (DAY53_SECRET_A vs DAY53_SECRET_B).
  16. Citation state check after error.
  17. Request state isolation and mutation safety.
  18. Complete offline end-to-end error path and recovery.

Constraints:
  - 100% Offline: In-memory QdrantVectorStore, mock embeddings, ThreadPoolExecutor.
  - Zero production code modified.
  - No new models, adapters, wrappers, or caching added.
  - Synthetic deterministic data only.
"""

from __future__ import annotations

import concurrent.futures
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


# ============================================================================
# Deterministic Synthetic Fixtures
# ============================================================================

DOC_A = "DOC-A-DAY53"
DOC_B = "DOC-B-DAY53"
DOC_C = "DOC-C-DAY53"

FILE_A = "day53_alpha.pdf"
FILE_B = "day53_beta.pdf"
FILE_C = "day53_gamma.pdf"

CHUNK_A1 = str(uuid.UUID("11111111-1111-1111-1111-aaaaaaaaaaaa"))
CHUNK_B1 = str(uuid.UUID("22222222-2222-2222-2222-bbbbbbbbbbbb"))
CHUNK_C1 = str(uuid.UUID("33333333-3333-3333-3333-cccccccccccc"))

PAGE_A1 = 1
PAGE_B1 = 1
PAGE_C1 = 1

DAY53_VALID_A = "DAY53_VALID_A_PRIMARY_SPECIFICATION"
DAY53_VALID_B = "DAY53_VALID_B_PRIMARY_SPECIFICATION"
DAY53_VALID_C = "DAY53_VALID_C_PRIMARY_SPECIFICATION"

DAY53_SECRET_A = "DAY53_SECRET_A_CONFIDENTIAL_TOKEN_999"
DAY53_SECRET_B = "DAY53_SECRET_B_CONFIDENTIAL_TOKEN_888"


class DeterministicDay53EmbeddingProvider:
    """Thread-safe deterministic offline mock embedding provider returning orthogonal 4D unit vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Map query string keywords to distinct vector directions."""
        clean = text.lower()
        if "alpha" in clean or "doc_a" in clean or "valid_a" in clean:
            return [1.0, 0.0, 0.0, 0.0]
        if "beta" in clean or "doc_b" in clean or "valid_b" in clean:
            return [0.0, 1.0, 0.0, 0.0]
        if "gamma" in clean or "doc_c" in clean or "valid_c" in clean:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding generation."""
        return [self.embed(t) for t in texts]


@pytest.fixture
def isolation_store() -> tuple[QdrantVectorStore, str]:
    """Create an isolated in-memory QdrantVectorStore preloaded with DOC-A, DOC-B, DOC-C."""
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(client=client)
    col_name = "error_isolation_coll"
    store.create_collection(col_name, vector_dimension=4)

    # Document A
    rec_a1 = EmbeddingVectorRecord(
        chunk_id=CHUNK_A1,
        document_id=DOC_A,
        filename=FILE_A,
        chunk_index=0,
        page_number=PAGE_A1,
        content_type="text",
        vector=[1.0, 0.0, 0.0, 0.0],
        metadata={"content": f"{DAY53_VALID_A} - {DAY53_SECRET_A}"},
    )
    gen_a = EmbeddingGenerationResult(
        document_id=DOC_A, filename=FILE_A, items=[rec_a1], dimension=4, is_ready=True,
    )
    store.upsert_embeddings(col_name, gen_a)

    # Document B
    rec_b1 = EmbeddingVectorRecord(
        chunk_id=CHUNK_B1,
        document_id=DOC_B,
        filename=FILE_B,
        chunk_index=0,
        page_number=PAGE_B1,
        content_type="text",
        vector=[0.0, 1.0, 0.0, 0.0],
        metadata={"content": f"{DAY53_VALID_B} - {DAY53_SECRET_B}"},
    )
    gen_b = EmbeddingGenerationResult(
        document_id=DOC_B, filename=FILE_B, items=[rec_b1], dimension=4, is_ready=True,
    )
    store.upsert_embeddings(col_name, gen_b)

    # Document C
    rec_c1 = EmbeddingVectorRecord(
        chunk_id=CHUNK_C1,
        document_id=DOC_C,
        filename=FILE_C,
        chunk_index=0,
        page_number=PAGE_C1,
        content_type="text",
        vector=[0.0, 0.0, 1.0, 0.0],
        metadata={"content": DAY53_VALID_C},
    )
    gen_c = EmbeddingGenerationResult(
        document_id=DOC_C, filename=FILE_C, items=[rec_c1], dimension=4, is_ready=True,
    )
    store.upsert_embeddings(col_name, gen_c)

    return store, col_name


# ============================================================================
# 1. Valid Baseline Execution
# ============================================================================

class TestValidBaseline:
    """Certifies that independent valid requests succeed with complete lineage."""

    def test_valid_baseline_execution(
        self, isolation_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """DOC-A, DOC-B, DOC-C requests succeed independently with correct markers."""
        store, col_name = isolation_store
        embedder = DeterministicDay53EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        resp_a = agent.search(f"Find {DAY53_VALID_A}")
        resp_b = agent.search(f"Find {DAY53_VALID_B}")
        resp_c = agent.search(f"Find {DAY53_VALID_C}")

        assert resp_a.unique_documents == [DOC_A]
        assert resp_b.unique_documents == [DOC_B]
        assert resp_c.unique_documents == [DOC_C]


# ============================================================================
# 2. Invalid Input Validation Contracts
# ============================================================================

class TestInvalidInputContracts:
    """Certifies deterministic validation errors across request, citation, and visual models."""

    def test_empty_and_whitespace_queries_rejected(self) -> None:
        """SearchRequest and AgentRequest reject empty and whitespace strings."""
        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            SearchRequest(query="")

        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            SearchRequest(query="   ")

        with pytest.raises(AgentValidationError, match="query must be a string"):
            SearchRequest(query=1234)  # type: ignore[arg-type]

    def test_invalid_citation_attributes_rejected(self) -> None:
        """AgentCitation rejects missing document_id, whitespace filename, and page <= 0."""
        with pytest.raises(AgentValidationError, match="document_id must be a non-empty string"):
            AgentCitation(document_id="", filename="f.pdf", chunk_id="c1")

        with pytest.raises(AgentValidationError, match="filename must be a non-empty string"):
            AgentCitation(document_id="doc1", filename="   ", chunk_id="c1")

        with pytest.raises(AgentValidationError, match="page_number must be a positive integer"):
            AgentCitation(document_id="doc1", filename="f.pdf", chunk_id="c1", page_number=0)

    def test_invalid_visual_evidence_rejected(self) -> None:
        """VisualEvidence rejects unsupported content types and non-positive page numbers."""
        with pytest.raises(VisionEvidenceError, match="Invalid visual content_type 'audio'"):
            VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", content_type="audio")

        with pytest.raises(VisionEvidenceError, match="page_number must be a positive integer"):
            VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", page_number=-2)


# ============================================================================
# 3. Sequential Failure and Success Transitions
# ============================================================================

class TestFailureAndSuccessTransitions:
    """Certifies that failed requests do not pollute following successful requests."""

    def test_failed_request_then_successful_request(
        self, isolation_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Invalid request failure does not prevent subsequent DOC-A request from succeeding."""
        store, col_name = isolation_store
        embedder = DeterministicDay53EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # 1. Invalid request fails
        with pytest.raises(AgentValidationError):
            agent.search("")

        # 2. Valid DOC-A request succeeds cleanly
        resp_a = agent.search(f"Find {DAY53_VALID_A}")
        assert resp_a.is_success is True
        assert resp_a.unique_documents == [DOC_A]
        assert resp_a.citations[0].document_id == DOC_A

    def test_success_then_failure_then_success(
        self, isolation_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """DOC-A success -> failure -> DOC-B success: neither successful response is corrupted."""
        store, col_name = isolation_store
        embedder = DeterministicDay53EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # 1. DOC-A succeeds
        resp_a = agent.search(f"Find {DAY53_VALID_A}")
        assert resp_a.unique_documents == [DOC_A]

        # 2. Failure occurs
        with pytest.raises(AgentValidationError):
            agent.search("   ")

        # 3. DOC-B succeeds
        resp_b = agent.search(f"Find {DAY53_VALID_B}")
        assert resp_b.unique_documents == [DOC_B]
        assert DOC_A not in resp_b.unique_documents

    def test_cross_document_failure_isolation(
        self, isolation_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Invalid DOC-A query failure does not leak into subsequent DOC-B query."""
        store, col_name = isolation_store
        embedder = DeterministicDay53EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # Attempt invalid SearchRequest with empty query
        with pytest.raises(AgentValidationError):
            SearchRequest(query="")

        # Query DOC-B
        resp_b = agent.search(f"Find {DAY53_VALID_B}")
        assert resp_b.unique_documents == [DOC_B]
        assert resp_b.citations[0].document_id == DOC_B


# ============================================================================
# 4. Component-Level Error Isolation (Retrieval, Context, Agent, Citation, Serialization)
# ============================================================================

class TestComponentLevelErrorIsolation:
    """Certifies error containment within retrieval, context, citation, and serialization."""

    def test_retrieval_error_isolation(
        self, isolation_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Querying an invalid collection raises ValueError, while query on valid collection succeeds."""
        store, col_name = isolation_store

        # 1. Search non-existent collection
        with pytest.raises(ValueError, match="Collection 'missing_coll' does not exist"):
            store.search("missing_coll", query_vector=[1.0, 0.0, 0.0, 0.0])

        # 2. Search valid collection succeeds
        results = store.search(col_name, query_vector=[1.0, 0.0, 0.0, 0.0])
        assert len(results) >= 1
        assert results[0]["document_id"] == DOC_A

    def test_context_building_error_isolation(self) -> None:
        """Passing invalid input to build_retrieval_context raises TypeError without breaking subsequent valid calls."""
        # 1. Invalid input raises TypeError
        with pytest.raises(TypeError, match="results must be a list"):
            build_retrieval_context("not_a_list")  # type: ignore[arg-type]

        # 2. Valid DOC-B context builds cleanly
        r_b = VectorSearchResult(
            chunk_id=CHUNK_B1, score=0.9, document_id=DOC_B, filename=FILE_B,
            page_number=PAGE_B1, chunk_index=0, content_type="text", content=DAY53_VALID_B,
        )
        ctx = build_retrieval_context([r_b])
        assert DAY53_VALID_B in ctx
        assert f"File: {FILE_B}" in ctx

    def test_citation_error_isolation(self) -> None:
        """Malformed AgentCitation creation fails, followed by valid citation creation."""
        # 1. Invalid citation fails
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="f.pdf", chunk_id="c1")

        # 2. Valid citation for DOC-B succeeds
        cit_b = AgentCitation(
            document_id=DOC_B, filename=FILE_B, chunk_id=CHUNK_B1, page_number=PAGE_B1,
        )
        assert cit_b.document_id == DOC_B
        assert cit_b.chunk_id == CHUNK_B1

    def test_serialization_error_isolation(self) -> None:
        """Deserializing corrupted dictionary raises AgentValidationError without affecting valid round-trips."""
        # 1. Corrupted dictionary deserialization raises error
        with pytest.raises(AgentValidationError):
            AgentCitation.from_dict({"document_id": "", "filename": "f.pdf", "chunk_id": "c1"})

        # 2. Valid citation round-trip succeeds
        valid_cit = AgentCitation(
            document_id=DOC_A, filename=FILE_A, chunk_id=CHUNK_A1, page_number=PAGE_A1,
        )
        restored = AgentCitation.from_dict(json.loads(json.dumps(valid_cit.to_dict())))
        assert restored.document_id == DOC_A
        assert restored.chunk_id == CHUNK_A1


# ============================================================================
# 5. Repeated Failure Determinism & Multi-Failure Recovery
# ============================================================================

class TestRepeatedFailureAndRecovery:
    """Certifies deterministic error behavior and multi-failure recovery."""

    def test_repeated_failure_determinism(self) -> None:
        """Executing the same invalid request 3 times yields identical AgentValidationError."""
        for _ in range(3):
            with pytest.raises(AgentValidationError, match="query cannot be empty"):
                SearchRequest(query="")

    def test_multi_failure_recovery(
        self, isolation_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Invalid -> Invalid -> Valid DOC-A -> Valid DOC-B all execute correctly."""
        store, col_name = isolation_store
        embedder = DeterministicDay53EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # 2 consecutive failures
        with pytest.raises(AgentValidationError):
            agent.search("")

        with pytest.raises(AgentValidationError):
            agent.search("   ")

        # 2 consecutive successes
        resp_a = agent.search(f"Find {DAY53_VALID_A}")
        resp_b = agent.search(f"Find {DAY53_VALID_B}")

        assert resp_a.unique_documents == [DOC_A]
        assert resp_b.unique_documents == [DOC_B]


# ============================================================================
# 6. Interleaved and Concurrent Failure Isolation
# ============================================================================

class TestInterleavedAndConcurrentFailureIsolation:
    """Certifies interleaved and concurrent failure containment."""

    def test_interleaved_failure_and_success_sequence(
        self, isolation_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Sequence: A-valid -> B-invalid -> C-valid -> A-invalid -> B-valid -> C-invalid."""
        store, col_name = isolation_store
        embedder = DeterministicDay53EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # 1. A-valid
        assert agent.search(f"Find {DAY53_VALID_A}").unique_documents == [DOC_A]

        # 2. B-invalid
        with pytest.raises(AgentValidationError):
            agent.search("")

        # 3. C-valid
        assert agent.search(f"Find {DAY53_VALID_C}").unique_documents == [DOC_C]

        # 4. A-invalid
        with pytest.raises(AgentValidationError):
            agent.search("   ")

        # 5. B-valid
        assert agent.search(f"Find {DAY53_VALID_B}").unique_documents == [DOC_B]

        # 6. C-invalid
        with pytest.raises(AgentValidationError):
            SearchRequest(query="")

    def test_concurrent_failure_isolation(
        self, isolation_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Valid A + Invalid B + Valid C executed concurrently in ThreadPoolExecutor."""
        store, col_name = isolation_store
        embedder = DeterministicDay53EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        def _run_query(query_str: str) -> list[str]:
            return agent.search(query_str).unique_documents

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f_a = executor.submit(_run_query, f"Find {DAY53_VALID_A}")
            f_b = executor.submit(_run_query, "")  # Invalid
            f_c = executor.submit(_run_query, f"Find {DAY53_VALID_C}")

            assert f_a.result() == [DOC_A]
            assert f_c.result() == [DOC_C]

            with pytest.raises(AgentValidationError):
                f_b.result()


# ============================================================================
# 7. Context Contamination Check with Secret Tokens
# ============================================================================

class TestContextContaminationAndSecrets:
    """Certifies that failed requests involving DOC-A do not leak DOC-A secrets into DOC-B."""

    def test_secret_token_isolation_after_failure(
        self, isolation_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """A failure on DOC-A query does not leak DAY53_SECRET_A into DOC-B response."""
        store, col_name = isolation_store
        embedder = DeterministicDay53EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # Trigger failure
        with pytest.raises(AgentValidationError):
            agent.search("")

        # Run DOC-B query
        resp_b = agent.search(f"Find {DAY53_VALID_B}")
        assert resp_b.unique_documents == [DOC_B]
        assert resp_b.has_citations is True

        content_b = resp_b.citations[0].metadata.get("content", "")
        assert DAY53_SECRET_B in content_b
        assert DAY53_SECRET_A not in content_b


# ============================================================================
# 8. Request State Isolation & Mutation Safety
# ============================================================================

class TestRequestStateIsolationAndMutation:
    """Certifies mutation safety and request state independence under failure."""

    def test_request_state_isolation_on_mutation(self) -> None:
        """Mutating or failing request_a does not affect independent request_b."""
        req_a = SearchRequest(query="Query Alpha", metadata={"token": "A"})
        req_b = SearchRequest(query="Query Beta", metadata={"token": "B"})

        req_a.metadata["token"] = "CORRUPTED"

        assert req_a.metadata["token"] == "CORRUPTED"
        assert req_b.metadata["token"] == "B"

    def test_citation_state_check_after_error(self) -> None:
        """After a failed citation operation, creating a valid citation retains exact fields."""
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="bad.pdf", chunk_id="c1")

        valid_cit = AgentCitation(
            document_id=DOC_C,
            filename=FILE_C,
            chunk_id=CHUNK_C1,
            page_number=PAGE_C1,
            score=0.98,
        )

        assert valid_cit.document_id == DOC_C
        assert valid_cit.chunk_id == CHUNK_C1
        assert valid_cit.page_number == PAGE_C1
        assert valid_cit.score == 0.98


# ============================================================================
# 9. End-to-End Error Path & Recovery
# ============================================================================

class TestEndToEndErrorPathAndRecovery:
    """Certifies end-to-end flow with failure injection and subsequent full pipeline execution."""

    def test_end_to_end_error_recovery_pipeline(
        self, isolation_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """
        Flow:
          1. Trigger SearchAgent invalid query error.
          2. Execute full pipeline for DOC-A:
             SearchAgent.search_and_package -> SearchResult -> AgentCitation -> Lineage Verification.
        """
        store, col_name = isolation_store
        embedder = DeterministicDay53EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # 1. Error path
        with pytest.raises(AgentValidationError):
            agent.search_and_package("")

        # 2. Full pipeline recovery
        packaged_res = agent.search_and_package(f"Find {DAY53_VALID_A}")

        assert isinstance(packaged_res, SearchResult)
        assert packaged_res.status == "RESULTS_FOUND"
        assert packaged_res.total_results >= 1
        assert packaged_res.has_results is True

        cit = packaged_res.citations[0]
        assert cit.document_id == DOC_A
        assert cit.chunk_id == CHUNK_A1
        assert cit.page_number == PAGE_A1
        assert cit.filename == FILE_A
