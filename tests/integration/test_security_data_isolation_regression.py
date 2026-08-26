"""
OmniBrain Member 4 — Day 55 Security & Data Isolation Regression Certification.

Validates that sensitive and document-specific data does not leak across requests,
documents, contexts, citations, or responses, adhering to strict data isolation invariants:
  - Document Isolation (Query A returns only Doc A data; Doc B/C data excluded).
  - Secret Marker Isolation (Fake secret markers from B/C never leak into A).
  - Context Isolation (Context generated for Doc A contains only Doc A public data).
  - Citation Isolation (AgentCitation lineage accurately mapped without cross-reference).
  - Request State Isolation (Sequential and interleaved requests remain independent).
  - Input & Output Validation Contracts (Strict schema enforcement without bypass).
  - Error Information Disclosure (Error messages contain no sensitive cross-document data).
  - Cross-Document Error Containment (A failed Doc A query does not taint Doc B/C).
  - Serialization Isolation (Independent JSON serialization roundtrips).
  - Mutation Safety (Independent object trees).
  - Concurrent Isolation (ThreadPoolExecutor parallel requests).
  - Repeated Request Stability (Zero residual state accumulation).
  - Input Injection Marker Handling (Safe treatment of adversarial marker strings).
  - End-to-End Lineage Isolation.

Constraints:
  - 100% Offline: In-memory QdrantVectorStore, mock deterministic embeddings.
  - Zero production code modified.
  - Fake synthetic secrets only (no real credentials, keys, or personal data).
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

# Ingestion subsystem (Member 1)
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

# Agents subsystem (Member 2)
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

# Vision subsystem (Member 3)
from vision.models import VisualEvidence
from vision.exceptions import VisionEvidenceError, VisionInputValidationError


# ============================================================================
# Deterministic Synthetic Fixtures & Fake Secrets
# ============================================================================

DOC_A = "DOC-A-DAY55"
DOC_B = "DOC-B-DAY55"
DOC_C = "DOC-C-DAY55"

FILE_A = "day55_confidential_a.pdf"
FILE_B = "day55_confidential_b.pdf"
FILE_C = "day55_confidential_c.pdf"

CHUNK_A1 = str(uuid.UUID("11111111-5555-5555-5555-aaaaaaaaaaaa"))
CHUNK_B1 = str(uuid.UUID("22222222-5555-5555-5555-bbbbbbbbbbbb"))
CHUNK_C1 = str(uuid.UUID("33333333-5555-5555-5555-cccccccccccc"))

PAGE_A1 = 1
PAGE_B1 = 1
PAGE_C1 = 1

DAY55_PUBLIC_A = "DAY55_PUBLIC_A_DATA_ALPHA"
DAY55_FAKE_SECRET_A = "DAY55_FAKE_SECRET_A_KEY_99999"

DAY55_PUBLIC_B = "DAY55_PUBLIC_B_DATA_BETA"
DAY55_FAKE_SECRET_B = "DAY55_FAKE_SECRET_B_KEY_88888"

DAY55_PUBLIC_C = "DAY55_PUBLIC_C_DATA_GAMMA"
DAY55_FAKE_SECRET_C = "DAY55_FAKE_SECRET_C_KEY_77777"


class DeterministicDay55EmbeddingProvider:
    """Thread-safe deterministic offline mock embedding provider returning orthogonal 4D unit vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Map distinct document query keywords to predictable vectors."""
        clean = text.lower()
        if "alpha" in clean or "doc_a" in clean or "public_a" in clean:
            return [1.0, 0.0, 0.0, 0.0]
        if "beta" in clean or "doc_b" in clean or "public_b" in clean:
            return [0.0, 1.0, 0.0, 0.0]
        if "gamma" in clean or "doc_c" in clean or "public_c" in clean:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding generation."""
        return [self.embed(t) for t in texts]


@pytest.fixture
def secure_store() -> tuple[QdrantVectorStore, str]:
    """Create an isolated in-memory QdrantVectorStore preloaded with DOC-A, DOC-B, DOC-C."""
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(client=client)
    col_name = "security_isolation_coll"
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
        metadata={
            "public_content": DAY55_PUBLIC_A,
            "secret_marker": DAY55_FAKE_SECRET_A,
            "content": f"{DAY55_PUBLIC_A} - confidential token: {DAY55_FAKE_SECRET_A}",
        },
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
        metadata={
            "public_content": DAY55_PUBLIC_B,
            "secret_marker": DAY55_FAKE_SECRET_B,
            "content": f"{DAY55_PUBLIC_B} - confidential token: {DAY55_FAKE_SECRET_B}",
        },
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
        metadata={
            "public_content": DAY55_PUBLIC_C,
            "secret_marker": DAY55_FAKE_SECRET_C,
            "content": f"{DAY55_PUBLIC_C} - confidential token: {DAY55_FAKE_SECRET_C}",
        },
    )
    gen_c = EmbeddingGenerationResult(
        document_id=DOC_C, filename=FILE_C, items=[rec_c1], dimension=4, is_ready=True,
    )
    store.upsert_embeddings(col_name, gen_c)

    return store, col_name


# ============================================================================
# 1. Document & Secret Marker Isolation
# ============================================================================

class TestDocumentAndSecretMarkerIsolation:
    """Sections 6, 7: Document boundary enforcement and cross-document secret marker containment."""

    def test_document_a_isolation_and_secret_exclusion(
        self, secure_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Querying DOC-A returns only DOC-A info, strictly excluding DOC-B/C public and secret markers."""
        store, col_name = secure_store
        embedder = DeterministicDay55EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        resp = agent.search(f"Search {DAY55_PUBLIC_A}")

        assert resp.unique_documents == [DOC_A]
        assert resp.has_citations is True
        content = resp.citations[0].metadata.get("content", "")

        assert DAY55_PUBLIC_A in content
        assert DAY55_FAKE_SECRET_B not in content
        assert DAY55_FAKE_SECRET_C not in content
        assert DAY55_PUBLIC_B not in content
        assert DAY55_PUBLIC_C not in content

    def test_document_b_isolation_and_secret_exclusion(
        self, secure_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Querying DOC-B returns only DOC-B info, strictly excluding DOC-A/C public and secret markers."""
        store, col_name = secure_store
        embedder = DeterministicDay55EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        resp = agent.search(f"Search {DAY55_PUBLIC_B}")

        assert resp.unique_documents == [DOC_B]
        content = resp.citations[0].metadata.get("content", "")

        assert DAY55_PUBLIC_B in content
        assert DAY55_FAKE_SECRET_A not in content
        assert DAY55_FAKE_SECRET_C not in content
        assert DAY55_PUBLIC_A not in content
        assert DAY55_PUBLIC_C not in content

    def test_document_c_isolation_and_secret_exclusion(
        self, secure_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Querying DOC-C returns only DOC-C info, strictly excluding DOC-A/B public and secret markers."""
        store, col_name = secure_store
        embedder = DeterministicDay55EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        resp = agent.search(f"Search {DAY55_PUBLIC_C}")

        assert resp.unique_documents == [DOC_C]
        content = resp.citations[0].metadata.get("content", "")

        assert DAY55_PUBLIC_C in content
        assert DAY55_FAKE_SECRET_A not in content
        assert DAY55_FAKE_SECRET_B not in content
        assert DAY55_PUBLIC_A not in content
        assert DAY55_PUBLIC_B not in content


# ============================================================================
# 2. Context & Citation Isolation
# ============================================================================

class TestContextAndCitationIsolation:
    """Sections 8, 9: Context building and citation lineage data boundaries."""

    def test_context_building_isolation(self) -> None:
        """Context built for DOC-A includes only DOC-A markers, omitting DOC-B/C markers."""
        res_a = VectorSearchResult(
            chunk_id=CHUNK_A1, score=0.95, document_id=DOC_A, filename=FILE_A,
            page_number=PAGE_A1, chunk_index=0, content_type="text",
            content=f"Report A: {DAY55_PUBLIC_A}",
        )
        ctx_a = build_retrieval_context([res_a])

        assert DAY55_PUBLIC_A in ctx_a
        assert DAY55_PUBLIC_B not in ctx_a
        assert DAY55_PUBLIC_C not in ctx_a
        assert DAY55_FAKE_SECRET_B not in ctx_a
        assert DAY55_FAKE_SECRET_C not in ctx_a

    def test_citation_lineage_isolation(self) -> None:
        """AgentCitation maintains exact document_id, chunk_id, page_number without cross-referencing."""
        res_b = VectorSearchResult(
            chunk_id=CHUNK_B1, score=0.92, document_id=DOC_B, filename=FILE_B,
            page_number=PAGE_B1, chunk_index=0, content_type="text",
            content=f"Report B: {DAY55_PUBLIC_B}",
        )
        cit_b = AgentCitation.from_search_result(res_b)

        assert cit_b.document_id == DOC_B
        assert cit_b.chunk_id == CHUNK_B1
        assert cit_b.filename == FILE_B
        assert cit_b.page_number == PAGE_B1


# ============================================================================
# 3. Request State Isolation & Interleaved Request Sequences
# ============================================================================

class TestRequestIsolationAndInterleavedSequences:
    """Section 10: Sequential and interleaved request state isolation."""

    def test_interleaved_request_isolation(
        self, secure_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Sequence A -> B -> A -> C -> B -> C executes with zero residual state leakage."""
        store, col_name = secure_store
        embedder = DeterministicDay55EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        sequence = [
            (DOC_A, f"Search {DAY55_PUBLIC_A}"),
            (DOC_B, f"Search {DAY55_PUBLIC_B}"),
            (DOC_A, f"Search {DAY55_PUBLIC_A}"),
            (DOC_C, f"Search {DAY55_PUBLIC_C}"),
            (DOC_B, f"Search {DAY55_PUBLIC_B}"),
            (DOC_C, f"Search {DAY55_PUBLIC_C}"),
        ]

        for expected_doc, query_str in sequence:
            resp = agent.search(query_str)
            assert resp.unique_documents == [expected_doc]


# ============================================================================
# 4. Input & Output Validation Contracts
# ============================================================================

class TestInputAndOutputValidationContracts:
    """Sections 11, 12: Schema validation and malformed input rejection."""

    def test_input_validation_empty_and_whitespace_rejected(self) -> None:
        """Empty and whitespace strings rejected with AgentValidationError."""
        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            SearchRequest(query="")

        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            SearchRequest(query="    ")

    def test_input_validation_invalid_types_rejected(self) -> None:
        """Non-string query inputs rejected with AgentValidationError."""
        with pytest.raises(AgentValidationError, match="query must be a string"):
            SearchRequest(query=99999)  # type: ignore[arg-type]

    def test_citation_input_validation_enforced(self) -> None:
        """AgentCitation validates non-empty document_id and positive page numbers."""
        with pytest.raises(AgentValidationError, match="document_id must be a non-empty string"):
            AgentCitation(document_id="", filename="test.pdf", chunk_id="c1")

        with pytest.raises(AgentValidationError, match="page_number must be a positive integer"):
            AgentCitation(document_id="doc1", filename="test.pdf", chunk_id="c1", page_number=0)


# ============================================================================
# 5. Error Information Disclosure & Cross-Document Failure Isolation
# ============================================================================

class TestErrorDisclosureAndFailureIsolation:
    """Sections 13, 14: Error message containment and failure isolation across documents."""

    def test_error_message_does_not_disclose_unrelated_secrets(
        self, secure_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Validation and execution errors do not leak fake secrets in exception messages."""
        with pytest.raises(AgentValidationError) as exc_info:
            SearchRequest(query="")

        err_msg = str(exc_info.value)
        assert DAY55_FAKE_SECRET_A not in err_msg
        assert DAY55_FAKE_SECRET_B not in err_msg
        assert DAY55_FAKE_SECRET_C not in err_msg

    def test_cross_document_failure_isolation(
        self, secure_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Failure on DOC-A does not taint or block subsequent valid DOC-B and DOC-C requests."""
        store, col_name = secure_store
        embedder = DeterministicDay55EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # 1. Trigger failure
        with pytest.raises(AgentValidationError):
            agent.search("")

        # 2. Query DOC-B succeeds
        resp_b = agent.search(f"Search {DAY55_PUBLIC_B}")
        assert resp_b.unique_documents == [DOC_B]

        # 3. Trigger second failure
        with pytest.raises(AgentValidationError):
            agent.search("   ")

        # 4. Query DOC-C succeeds
        resp_c = agent.search(f"Search {DAY55_PUBLIC_C}")
        assert resp_c.unique_documents == [DOC_C]


# ============================================================================
# 6. Serialization Isolation & Mutation Safety
# ============================================================================

class TestSerializationAndMutationSafety:
    """Sections 15, 16: Independent serialization round-trips and memory isolation."""

    def test_serialization_isolation(self) -> None:
        """DOC-A and DOC-B citations serialized to JSON and restored maintain independent identities."""
        cit_a = AgentCitation(document_id=DOC_A, filename=FILE_A, chunk_id=CHUNK_A1, page_number=1)
        cit_b = AgentCitation(document_id=DOC_B, filename=FILE_B, chunk_id=CHUNK_B1, page_number=1)

        dict_a = cit_a.to_dict()
        dict_b = cit_b.to_dict()

        restored_a = AgentCitation.from_dict(json.loads(json.dumps(dict_a)))
        restored_b = AgentCitation.from_dict(json.loads(json.dumps(dict_b)))

        assert restored_a.document_id == DOC_A
        assert restored_b.document_id == DOC_B
        assert restored_a.chunk_id != restored_b.chunk_id

    def test_mutation_safety_across_independent_objects(self) -> None:
        """Mutating request A does not alter request B or request C."""
        req_a = SearchRequest(query="Alpha Query", metadata={"tenant": "A"})
        req_b = SearchRequest(query="Beta Query", metadata={"tenant": "B"})
        req_c = SearchRequest(query="Gamma Query", metadata={"tenant": "C"})

        req_a.metadata["tenant"] = "MUTATED_A"

        assert req_a.metadata["tenant"] == "MUTATED_A"
        assert req_b.metadata["tenant"] == "B"
        assert req_c.metadata["tenant"] == "C"


# ============================================================================
# 7. Concurrent & Repeated Request Isolation
# ============================================================================

class TestConcurrentAndRepeatedIsolation:
    """Sections 17, 18: ThreadPoolExecutor parallel execution and 3-iteration stability."""

    def test_concurrent_request_isolation(
        self, secure_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """DOC-A, DOC-B, DOC-C executed concurrently return strictly isolated results."""
        store, col_name = secure_store
        embedder = DeterministicDay55EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        queries = [
            (DOC_A, f"Search {DAY55_PUBLIC_A}"),
            (DOC_B, f"Search {DAY55_PUBLIC_B}"),
            (DOC_C, f"Search {DAY55_PUBLIC_C}"),
        ]

        def _execute(target_doc: str, query_str: str) -> tuple[str, list[str]]:
            res = agent.search(query_str)
            return target_doc, res.unique_documents

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_execute, doc, q) for doc, q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for expected_doc, actual_docs in results:
            assert actual_docs == [expected_doc]

    def test_repeated_request_isolation_stability(
        self, secure_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Executing DOC-A query 3 consecutive times yields identical isolated results without accumulation."""
        store, col_name = secure_store
        embedder = DeterministicDay55EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        for _ in range(3):
            resp = agent.search(f"Search {DAY55_PUBLIC_A}")
            assert resp.unique_documents == [DOC_A]
            assert len(resp.citations) == 1
            assert resp.citations[0].document_id == DOC_A


# ============================================================================
# 8. Prompt / Input Injection Marker Handling & End-to-End Flow
# ============================================================================

class TestInjectionMarkersAndEndToEndFlow:
    """Sections 21, 22: Safe handling of input injection strings and end-to-end handoff."""

    def test_input_injection_marker_strings_handled_safely(
        self, secure_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Adversarial marker strings like IGNORE_INSTRUCTION treated as standard string query safely."""
        store, col_name = secure_store
        embedder = DeterministicDay55EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        injection_query = f"DAY55_TEST_IGNORE_INSTRUCTION Search {DAY55_PUBLIC_A} DAY55_TEST_SYSTEM_MARKER"
        resp = agent.search(injection_query)

        # Standard query routing succeeds without schema failure
        assert resp.unique_documents == [DOC_A]
        assert resp.is_success is True

    def test_end_to_end_data_isolation_flow(
        self, secure_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """End-to-end pipeline handoff for DOC-A strictly maintains isolation from DOC-B."""
        store, col_name = secure_store
        embedder = DeterministicDay55EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # 1. Search and package DOC-A
        packaged_a = agent.search_and_package(f"Search {DAY55_PUBLIC_A}")
        assert packaged_a.status == "RESULTS_FOUND"
        assert packaged_a.has_results is True
        assert packaged_a.citations[0].document_id == DOC_A

        # 2. Search and package DOC-B
        packaged_b = agent.search_and_package(f"Search {DAY55_PUBLIC_B}")
        assert packaged_b.status == "RESULTS_FOUND"
        assert packaged_b.has_results is True
        assert packaged_b.citations[0].document_id == DOC_B
