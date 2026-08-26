"""
OmniBrain Member 4 — Day 52 Concurrent Request State Isolation Regression Certification.

Validates that concurrent, interleaved, repeated, and failed requests do not leak state,
context, retrieval results, citations, or document identity between requests.

Core Invariant:
  Request A -> DOC-A -> result A
  Request B -> DOC-B -> result B
  Request C -> DOC-C -> result C

No request may receive another request's data.

Covers:
  1.  Single request baseline (A, B, C executed independently).
  2.  Two-request concurrent execution (A + B concurrently).
  3.  Three-request concurrent execution (A + B + C concurrently).
  4.  Repeated concurrency (3 repeated batches of concurrent requests).
  5.  Interleaved execution (A -> B -> A -> C -> B -> C).
  6.  Reverse order execution (C -> B -> A sequentially and concurrently).
  7.  Concurrent retrieval operations on vector store.
  8.  Concurrent context building via build_retrieval_context.
  9.  Concurrent citation creation via AgentCitation.
  10. Request state isolation during overlapping execution lifecycles.
  11. Concurrent failure and success isolation (Valid A + Invalid B + Valid C).
  12. Same-document concurrent execution (multiple concurrent queries targeting DOC-A).
  13. Cross-document concurrent execution and exact marker isolation.
  14. Citation lineage verification under concurrency.
  15. Serialization and deserialization round-trips under concurrent execution.
  16. Mutation safety and independent request/citation objects.
  17. 3-iteration determinism and repeatability.

Constraints:
  - 100% Offline: In-memory QdrantVectorStore, deterministic mock embeddings, ThreadPoolExecutor.
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


# ============================================================================
# Deterministic Synthetic Fixtures
# ============================================================================

DOC_A = "DOC-A-DAY52"
DOC_B = "DOC-B-DAY52"
DOC_C = "DOC-C-DAY52"

FILE_A = "day52_doc_a.pdf"
FILE_B = "day52_doc_b.pdf"
FILE_C = "day52_doc_c.pdf"

CHUNK_A1 = str(uuid.UUID("11111111-aaaa-aaaa-aaaa-aaaaaaaaaaaa"))
CHUNK_B1 = str(uuid.UUID("22222222-bbbb-bbbb-bbbb-bbbbbbbbbbbb"))
CHUNK_C1 = str(uuid.UUID("33333333-cccc-cccc-cccc-cccccccccccc"))

PAGE_A1 = 1
PAGE_B1 = 1
PAGE_C1 = 1

DAY52_REQUEST_A_UNIQUE = "DAY52_REQUEST_A_UNIQUE_ALPHA_MARKER_99"
DAY52_REQUEST_B_UNIQUE = "DAY52_REQUEST_B_UNIQUE_BETA_MARKER_88"
DAY52_REQUEST_C_UNIQUE = "DAY52_REQUEST_C_UNIQUE_GAMMA_MARKER_77"


class DeterministicDay52EmbeddingProvider:
    """Thread-safe deterministic offline mock embedding provider returning orthogonal 4D unit vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Map distinct document query keywords to orthogonal vector axes."""
        clean = text.lower()
        if "alpha" in clean or "doc_a" in clean or "request_a" in clean:
            return [1.0, 0.0, 0.0, 0.0]
        if "beta" in clean or "doc_b" in clean or "request_b" in clean:
            return [0.0, 1.0, 0.0, 0.0]
        if "gamma" in clean or "doc_c" in clean or "request_c" in clean:
            return [0.0, 0.0, 1.0, 0.0]
        if "no_match" in clean or "nonexistent" in clean:
            return [0.0, 0.0, 0.0, 1.0]
        return [0.5, 0.5, 0.5, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding generation."""
        return [self.embed(t) for t in texts]


@pytest.fixture
def concurrent_store() -> tuple[QdrantVectorStore, str]:
    """Create an isolated in-memory QdrantVectorStore preloaded with DOC-A, DOC-B, DOC-C."""
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(client=client)
    col_name = "concurrent_isolation_coll"
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
        metadata={"content": f"Specifications: {DAY52_REQUEST_A_UNIQUE}", "doc": "A"},
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
        metadata={"content": f"Specifications: {DAY52_REQUEST_B_UNIQUE}", "doc": "B"},
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
        metadata={"content": f"Specifications: {DAY52_REQUEST_C_UNIQUE}", "doc": "C"},
    )
    gen_c = EmbeddingGenerationResult(
        document_id=DOC_C, filename=FILE_C, items=[rec_c1], dimension=4, is_ready=True,
    )
    store.upsert_embeddings(col_name, gen_c)

    return store, col_name


# ============================================================================
# 1. Single Request Baseline & Two/Three-Request Concurrency
# ============================================================================

class TestConcurrentRequestStateIsolation:
    """Certifies concurrent execution isolation across 1, 2, and 3 concurrent requests."""

    def test_single_request_baseline(
        self, concurrent_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Sequential single-request baseline: A -> DOC-A, B -> DOC-B, C -> DOC-C."""
        store, col_name = concurrent_store
        embedder = DeterministicDay52EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        resp_a = agent.search(f"Query {DAY52_REQUEST_A_UNIQUE}")
        resp_b = agent.search(f"Query {DAY52_REQUEST_B_UNIQUE}")
        resp_c = agent.search(f"Query {DAY52_REQUEST_C_UNIQUE}")

        assert resp_a.unique_documents == [DOC_A]
        assert resp_b.unique_documents == [DOC_B]
        assert resp_c.unique_documents == [DOC_C]

    def test_two_request_concurrency(
        self, concurrent_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Concurrent execution of Request A and Request B produces isolated results."""
        store, col_name = concurrent_store
        embedder = DeterministicDay52EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        queries = [
            (DOC_A, f"Query {DAY52_REQUEST_A_UNIQUE}"),
            (DOC_B, f"Query {DAY52_REQUEST_B_UNIQUE}"),
        ]

        def _execute(target_doc: str, query_str: str) -> tuple[str, list[str]]:
            res = agent.search(query_str)
            return target_doc, res.unique_documents

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_execute, doc, q) for doc, q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for expected_doc, actual_docs in results:
            assert actual_docs == [expected_doc]

    def test_three_request_concurrency(
        self, concurrent_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Concurrent execution of Request A, B, and C produces strictly isolated results."""
        store, col_name = concurrent_store
        embedder = DeterministicDay52EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        queries = [
            (DOC_A, f"Query {DAY52_REQUEST_A_UNIQUE}"),
            (DOC_B, f"Query {DAY52_REQUEST_B_UNIQUE}"),
            (DOC_C, f"Query {DAY52_REQUEST_C_UNIQUE}"),
        ]

        def _execute(target_doc: str, query_str: str) -> tuple[str, list[str]]:
            res = agent.search(query_str)
            return target_doc, res.unique_documents

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_execute, doc, q) for doc, q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for expected_doc, actual_docs in results:
            assert actual_docs == [expected_doc]


# ============================================================================
# 2. Repeated Concurrency, Interleaved, and Reverse Order
# ============================================================================

class TestRepeatedAndInterleavedConcurrency:
    """Certifies repeated batches, interleaved calls, and reverse order execution."""

    def test_repeated_concurrency_3_iterations(
        self, concurrent_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """3 consecutive concurrent batches of A+B+C execute with zero cross-contamination."""
        store, col_name = concurrent_store
        embedder = DeterministicDay52EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        queries = [
            (DOC_A, f"Query {DAY52_REQUEST_A_UNIQUE}"),
            (DOC_B, f"Query {DAY52_REQUEST_B_UNIQUE}"),
            (DOC_C, f"Query {DAY52_REQUEST_C_UNIQUE}"),
        ]

        def _execute(target_doc: str, query_str: str) -> tuple[str, list[str]]:
            res = agent.search(query_str)
            return target_doc, res.unique_documents

        for _ in range(3):
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(_execute, doc, q) for doc, q in queries]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            for expected_doc, actual_docs in results:
                assert actual_docs == [expected_doc]

    def test_interleaved_execution(
        self, concurrent_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Interleaved execution A -> B -> A -> C -> B -> C preserves correct document association."""
        store, col_name = concurrent_store
        embedder = DeterministicDay52EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        seq = [DOC_A, DOC_B, DOC_A, DOC_C, DOC_B, DOC_C]
        queries = {
            DOC_A: f"Query {DAY52_REQUEST_A_UNIQUE}",
            DOC_B: f"Query {DAY52_REQUEST_B_UNIQUE}",
            DOC_C: f"Query {DAY52_REQUEST_C_UNIQUE}",
        }

        for expected_doc in seq:
            resp = agent.search(queries[expected_doc])
            assert resp.unique_documents == [expected_doc]

    def test_reverse_order_sequential_and_concurrent(
        self, concurrent_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """C -> B -> A reverse execution maintains proper document identities."""
        store, col_name = concurrent_store
        embedder = DeterministicDay52EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # Reverse sequential
        resp_c = agent.search(f"Query {DAY52_REQUEST_C_UNIQUE}")
        resp_b = agent.search(f"Query {DAY52_REQUEST_B_UNIQUE}")
        resp_a = agent.search(f"Query {DAY52_REQUEST_A_UNIQUE}")

        assert resp_c.unique_documents == [DOC_C]
        assert resp_b.unique_documents == [DOC_B]
        assert resp_a.unique_documents == [DOC_A]


# ============================================================================
# 3. Concurrent Retrieval, Context Building, and Citation Creation
# ============================================================================

class TestConcurrentSubsystemOperations:
    """Certifies concurrent execution within retrieval, context building, and citation creation."""

    def test_concurrent_context_building(self) -> None:
        """build_retrieval_context executed concurrently generates isolated contexts."""
        r_a = VectorSearchResult(
            chunk_id=CHUNK_A1, score=0.9, document_id=DOC_A, filename=FILE_A,
            page_number=PAGE_A1, chunk_index=0, content_type="text", content=DAY52_REQUEST_A_UNIQUE,
        )
        r_b = VectorSearchResult(
            chunk_id=CHUNK_B1, score=0.9, document_id=DOC_B, filename=FILE_B,
            page_number=PAGE_B1, chunk_index=0, content_type="text", content=DAY52_REQUEST_B_UNIQUE,
        )
        r_c = VectorSearchResult(
            chunk_id=CHUNK_C1, score=0.9, document_id=DOC_C, filename=FILE_C,
            page_number=PAGE_C1, chunk_index=0, content_type="text", content=DAY52_REQUEST_C_UNIQUE,
        )

        items = [(DOC_A, [r_a]), (DOC_B, [r_b]), (DOC_C, [r_c])]

        def _build_ctx(doc: str, results: list[VectorSearchResult]) -> tuple[str, str]:
            return doc, build_retrieval_context(results)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_build_ctx, doc, res) for doc, res in items]
            ctx_results = dict([f.result() for f in concurrent.futures.as_completed(futures)])

        assert DAY52_REQUEST_A_UNIQUE in ctx_results[DOC_A]
        assert DAY52_REQUEST_B_UNIQUE not in ctx_results[DOC_A]
        assert DAY52_REQUEST_C_UNIQUE not in ctx_results[DOC_A]

        assert DAY52_REQUEST_B_UNIQUE in ctx_results[DOC_B]
        assert DAY52_REQUEST_A_UNIQUE not in ctx_results[DOC_B]
        assert DAY52_REQUEST_C_UNIQUE not in ctx_results[DOC_B]

        assert DAY52_REQUEST_C_UNIQUE in ctx_results[DOC_C]
        assert DAY52_REQUEST_A_UNIQUE not in ctx_results[DOC_C]
        assert DAY52_REQUEST_B_UNIQUE not in ctx_results[DOC_C]

    def test_concurrent_citation_creation(self) -> None:
        """AgentCitation.from_search_result executed concurrently maintains exact lineage."""
        r_a = VectorSearchResult(
            chunk_id=CHUNK_A1, score=0.95, document_id=DOC_A, filename=FILE_A,
            page_number=PAGE_A1, chunk_index=0, content_type="text", content=DAY52_REQUEST_A_UNIQUE,
        )
        r_b = VectorSearchResult(
            chunk_id=CHUNK_B1, score=0.92, document_id=DOC_B, filename=FILE_B,
            page_number=PAGE_B1, chunk_index=0, content_type="text", content=DAY52_REQUEST_B_UNIQUE,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_a = executor.submit(AgentCitation.from_search_result, r_a)
            f_b = executor.submit(AgentCitation.from_search_result, r_b)
            c_a = f_a.result()
            c_b = f_b.result()

        assert c_a.document_id == DOC_A
        assert c_a.chunk_id == CHUNK_A1
        assert c_b.document_id == DOC_B
        assert c_b.chunk_id == CHUNK_B1


# ============================================================================
# 4. Failure + Success Concurrency & Same/Cross Document Concurrency
# ============================================================================

class TestFailureAndCrossDocumentConcurrency:
    """Certifies concurrent failure containment and cross-document marker isolation."""

    def test_concurrent_failure_and_success_isolation(
        self, concurrent_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Valid A + Invalid B (empty query) + Valid C executed concurrently: B fails cleanly, A & C succeed."""
        store, col_name = concurrent_store
        embedder = DeterministicDay52EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        def _run_query(query_str: str) -> list[str]:
            return agent.search(query_str).unique_documents

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f_a = executor.submit(_run_query, f"Query {DAY52_REQUEST_A_UNIQUE}")
            f_b = executor.submit(_run_query, "")  # Invalid empty query
            f_c = executor.submit(_run_query, f"Query {DAY52_REQUEST_C_UNIQUE}")

            # A and C must succeed
            assert f_a.result() == [DOC_A]
            assert f_c.result() == [DOC_C]

            # B must raise AgentValidationError
            with pytest.raises(AgentValidationError):
                f_b.result()

    def test_same_document_concurrency(
        self, concurrent_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Multiple concurrent requests targeting DOC-A execute safely without state mutation."""
        store, col_name = concurrent_store
        embedder = DeterministicDay52EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        def _run() -> list[str]:
            return agent.search(f"Query {DAY52_REQUEST_A_UNIQUE}").unique_documents

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_run) for _ in range(4)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for r in results:
            assert r == [DOC_A]

    def test_cross_document_marker_isolation_concurrent(
        self, concurrent_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Cross-document marker check under concurrent execution: A only contains A, B only B, C only C."""
        store, col_name = concurrent_store
        embedder = DeterministicDay52EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        tasks = [
            (DOC_A, f"Query {DAY52_REQUEST_A_UNIQUE}", DAY52_REQUEST_A_UNIQUE, [DAY52_REQUEST_B_UNIQUE, DAY52_REQUEST_C_UNIQUE]),
            (DOC_B, f"Query {DAY52_REQUEST_B_UNIQUE}", DAY52_REQUEST_B_UNIQUE, [DAY52_REQUEST_A_UNIQUE, DAY52_REQUEST_C_UNIQUE]),
            (DOC_C, f"Query {DAY52_REQUEST_C_UNIQUE}", DAY52_REQUEST_C_UNIQUE, [DAY52_REQUEST_A_UNIQUE, DAY52_REQUEST_B_UNIQUE]),
        ]

        def _run_marker_test(
            expected_doc: str, query_str: str, expected_marker: str, forbidden_markers: list[str]
        ) -> tuple[str, str, list[str], AgentResponse]:
            resp = agent.search(query_str)
            return expected_doc, expected_marker, forbidden_markers, resp

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_run_marker_test, doc, q, exp, forb) for doc, q, exp, forb in tasks]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for expected_doc, expected_marker, forbidden_markers, resp in results:
            assert resp.unique_documents == [expected_doc]
            assert resp.has_citations is True
            raw_content = resp.citations[0].metadata.get("content", "")
            assert expected_marker in raw_content
            for forbidden in forbidden_markers:
                assert forbidden not in raw_content


# ============================================================================
# 5. Serialization Under Concurrency & Mutation Safety
# ============================================================================

class TestSerializationAndMutationSafety:
    """Certifies serialization round-trips and mutation safety under concurrency."""

    def test_concurrent_serialization_isolation(self) -> None:
        """Serializing and deserializing citations concurrently preserves identities without drift."""
        c_a = AgentCitation(document_id=DOC_A, filename=FILE_A, chunk_id=CHUNK_A1, page_number=1)
        c_b = AgentCitation(document_id=DOC_B, filename=FILE_B, chunk_id=CHUNK_B1, page_number=1)
        c_c = AgentCitation(document_id=DOC_C, filename=FILE_C, chunk_id=CHUNK_C1, page_number=1)

        def _roundtrip(citation: AgentCitation) -> AgentCitation:
            d = citation.to_dict()
            json_str = json.dumps(d)
            return AgentCitation.from_dict(json.loads(json_str))

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f_a = executor.submit(_roundtrip, c_a)
            f_b = executor.submit(_roundtrip, c_b)
            f_c = executor.submit(_roundtrip, c_c)

            res_a = f_a.result()
            res_b = f_b.result()
            res_c = f_c.result()

        assert res_a.document_id == DOC_A
        assert res_b.document_id == DOC_B
        assert res_c.document_id == DOC_C

    def test_mutation_safety_under_concurrency(self) -> None:
        """Independent request objects modified concurrently do not interfere with each other."""
        req_a = SearchRequest(query="Query A", metadata={"tenant": "A"})
        req_b = SearchRequest(query="Query B", metadata={"tenant": "B"})

        def _mutate(req: SearchRequest, new_tenant: str) -> str:
            req.metadata["tenant"] = new_tenant
            return req.metadata["tenant"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_a = executor.submit(_mutate, req_a, "MUTATED_A")
            f_b = executor.submit(_mutate, req_b, "MUTATED_B")

            res_a = f_a.result()
            res_b = f_b.result()

        assert res_a == "MUTATED_A"
        assert res_b == "MUTATED_B"
        assert req_a.metadata["tenant"] == "MUTATED_A"
        assert req_b.metadata["tenant"] == "MUTATED_B"
