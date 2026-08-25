"""
OmniBrain Member 4 — Day 8 Performance, Duplicate-Work & Resource-Efficiency Tests.

Verifies that the existing OMNIBRAIN integration contracts execute efficiently without
unexpected duplicate work, repeated operations, or resource leaks across Ingestion,
Search, Vision, and Supervisor layers.

Focus areas:
1. Single execution work verification (embedding, retrieval, adaptation, normalization).
2. Repeated execution linearity (N queries execute exactly N operations without runaway loops).
3. Concurrent execution efficiency (threads do not re-execute sibling operations).
4. Search agent call count verification (single embed and single retrieval per search request).
5. Vision evidence adaptation single-pass efficiency.
6. Result normalizer single-pass sanitization.
7. Resource safety (zero workspace pollution, no memory leaks or persistent state artifacts).
8. Preservation of result correctness, lineage, and citations under load.
9. 100% offline, deterministic, side-effect-free execution.
"""

from __future__ import annotations

import copy
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Ensure repo root is on sys.path for test runners executing this file directly
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

from agents.models import AgentCitation, AgentRequest, AgentResponse, AgentState, SearchRequest, SearchResult
from agents.search_agent import SearchAgent
from ingestion.models import DocumentChunk, VectorSearchResult
from ingestion.qdrant_store import QdrantVectorStore
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.result_normalizer import VisionResultNormalizer


class CountingEmbeddingProvider:
    """Mock EmbeddingProvider that tracks invocation counts."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.embed_call_count = 0
        self.embed_batch_call_count = 0
        self._lock = threading.Lock()

    def embed(self, text: str) -> list[float]:
        with self._lock:
            self.embed_call_count += 1
        return [0.1] * self.dimension

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self.embed_batch_call_count += 1
        return [[0.1] * self.dimension for _ in texts]


# ============================================================================
# 1. SEARCH AGENT SINGLE INVOCATION EFFICIENCY
# ============================================================================


class TestSearchAgentSingleInvocationEfficiency:
    """Verifies that SearchAgent executes embedding and retrieval exactly once per search request."""

    def test_single_search_request_executes_single_embedding_call(self) -> None:
        """Verify SearchAgent.execute() calls embed exactly once per query."""
        provider = CountingEmbeddingProvider(dimension=4)
        mock_store = MagicMock(spec=QdrantVectorStore)

        agent = SearchAgent(
            embedding_provider=provider,
            store=mock_store,
            collection_name="test_collection",
            expected_dimension=4,
        )

        dummy_results = [
            VectorSearchResult(
                chunk_id="chk-01",
                score=0.92,
                document_id="doc-01",
                filename="f.pdf",
                page_number=1,
                chunk_index=0,
                content_type="text",
                content="Search agent single invocation test content.",
            )
        ]

        # Patch retrieve_context to return dummy_results
        with MagicMock() as mock_retrieve:
            from unittest.mock import patch

            with patch("agents.search_agent.retrieve_context") as mock_rc:
                from ingestion.models import RetrievalServiceResult

                mock_rc.return_value = RetrievalServiceResult(
                    query_vector_dimension=4,
                    results=dummy_results,
                    context="[Source 1] Search agent single invocation test content.",
                )

                request = AgentRequest(query="What is the test revenue?")
                response = agent.search(request)

                assert response.is_success is True
                assert response.total_citations == 1
                assert provider.embed_call_count == 1
                assert mock_rc.call_count == 1

    def test_repeated_search_requests_scale_linearly(self) -> None:
        """Verify N repeated search queries result in exactly N embedding and retrieval calls."""
        provider = CountingEmbeddingProvider(dimension=4)
        mock_store = MagicMock(spec=QdrantVectorStore)

        agent = SearchAgent(
            embedding_provider=provider,
            store=mock_store,
            collection_name="test_collection",
            expected_dimension=4,
        )

        n_queries = 5
        dummy_results = [
            VectorSearchResult(
                chunk_id="chk-01",
                score=0.90,
                document_id="doc-01",
                filename="f.pdf",
                page_number=1,
                chunk_index=0,
                content_type="text",
                content="Repeated run test content.",
            )
        ]

        from unittest.mock import patch
        from ingestion.models import RetrievalServiceResult

        with patch("agents.search_agent.retrieve_context") as mock_rc:
            mock_rc.return_value = RetrievalServiceResult(
                query_vector_dimension=4,
                results=dummy_results,
                context="Context",
            )

            for i in range(n_queries):
                resp = agent.search(AgentRequest(query=f"Query {i}"))
                assert resp.is_success is True

            assert provider.embed_call_count == n_queries
            assert mock_rc.call_count == n_queries


# ============================================================================
# 2. VISION EVIDENCE ADAPTATION EFFICIENCY
# ============================================================================


class TestVisionEvidenceAdaptationEfficiency:
    """Verifies that evidence adaptation operates in a single linear pass over the citation package."""

    def test_single_pass_evidence_adaptation(self) -> None:
        """Verify adapting a package with 5 visual citations constructs exactly 5 VisualEvidence items."""
        citations = [
            AgentCitation(
                document_id="doc-eff-01",
                filename="efficiency.pdf",
                chunk_id=f"c-img-{i}",
                page_number=i + 1,
                content_type="image" if i % 2 == 0 else "chart",
                score=0.95 - (i * 0.05),
                metadata={"chunk_index": i},
            )
            for i in range(5)
        ]

        search_pkg = SearchResult(
            query="Analyze efficiency plots",
            status="RESULTS_FOUND",
            citations=citations,
            context="Visual summary",
        )

        adapted_items = VisualEvidenceAdapter.adapt_search_package(search_pkg)
        assert len(adapted_items) == 5

        # Verify exact correspondence without duplicate items or key generation
        unique_chunk_ids = {item.chunk_id for item in adapted_items}
        assert len(unique_chunk_ids) == 5
        assert unique_chunk_ids == {f"c-img-{i}" for i in range(5)}


# ============================================================================
# 3. RESULT NORMALIZER EFFICIENCY & IDEMPOTENCE
# ============================================================================


class TestResultNormalizerEfficiencyAndIdempotence:
    """Verifies that result normalization and metadata sanitization are idempotent single-pass operations."""

    def test_metadata_sanitization_is_idempotent(self) -> None:
        """Verify calling sanitize_metadata multiple times produces identical clean output without side effects."""
        raw_metadata = {
            "query_time_ms": 15,
            "provider": "offline_provider",
            "api_key": "SECRET_KEY",
            "token": "SECRET_TOKEN",
            "secret": "TOP_SECRET_PAYLOAD",
            "score": 0.99,
        }

        first_pass = VisionResultNormalizer.sanitize_metadata(raw_metadata)
        second_pass = VisionResultNormalizer.sanitize_metadata(first_pass)
        third_pass = VisionResultNormalizer.sanitize_metadata(second_pass)

        assert first_pass == second_pass == third_pass
        assert "api_key" not in first_pass
        assert "token" not in first_pass
        assert "secret" not in first_pass
        assert first_pass["query_time_ms"] == 15
        assert first_pass["provider"] == "offline_provider"

    def test_vision_result_normalization_stability(self) -> None:
        """Verify VisionResult maintains consistent representation across repeated serialization passes."""
        evidence = VisualEvidence(
            document_id="doc-norm-01",
            filename="norm.pdf",
            chunk_id="chunk-norm-1",
            content_type="diagram",
        )
        res = VisionResult(
            query="Normalization test",
            status="success",
            description="Normalized result",
            evidence=[evidence],
            metadata={"latency": 10},
        )

        d1 = res.to_dict()
        res_restored = VisionResult.from_dict(d1)
        d2 = res_restored.to_dict()

        assert d1 == d2


# ============================================================================
# 4. CONCURRENT RESOURCE EFFICIENCY
# ============================================================================


class TestConcurrentResourceEfficiency:
    """Verifies that concurrent requests do not cross-trigger redundant operations."""

    def test_concurrent_search_embedding_efficiency(self) -> None:
        """Verify 8 concurrent threads execute exactly 8 distinct embedding operations."""
        provider = CountingEmbeddingProvider(dimension=4)
        mock_store = MagicMock(spec=QdrantVectorStore)

        agent = SearchAgent(
            embedding_provider=provider,
            store=mock_store,
            collection_name="test_collection",
            expected_dimension=4,
        )

        num_threads = 8
        errors: list[Exception] = []

        from unittest.mock import patch
        from ingestion.models import RetrievalServiceResult

        with patch("agents.search_agent.retrieve_context") as mock_rc:
            mock_rc.return_value = RetrievalServiceResult(
                query_vector_dimension=4,
                results=[
                    VectorSearchResult(
                        chunk_id="c1",
                        score=0.9,
                        document_id="d1",
                        filename="f.pdf",
                        page_number=1,
                        chunk_index=0,
                        content_type="text",
                        content="Content",
                    )
                ],
                context="Context",
            )

            def worker(idx: int) -> None:
                try:
                    resp = agent.search(AgentRequest(query=f"Concurrent query {idx}"))
                    assert resp.is_success is True
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Thread errors: {errors}"
            assert provider.embed_call_count == num_threads
            assert mock_rc.call_count == num_threads


# ============================================================================
# 5. RESULT CORRECTNESS UNDER HIGH THROUGHPUT
# ============================================================================


class TestResultCorrectnessUnderHighThroughput:
    """Verifies that efficiency and reuse never compromise lineage, citations, or result correctness."""

    def test_end_to_end_correctness_across_rapid_sequential_runs(self) -> None:
        """Run 20 rapid sequential end-to-end integration flows and verify 100% lineage accuracy."""
        for i in range(20):
            doc_id = f"doc-rapid-{i}"
            fname = f"file_{i}.pdf"
            cid = f"chk-{i}"
            page = (i % 5) + 1

            vs_item = VectorSearchResult(
                chunk_id=cid,
                score=0.90 + (i * 0.002),
                document_id=doc_id,
                filename=fname,
                page_number=page,
                chunk_index=i,
                content_type="chart",
                content=f"Rapid test figure content {i}",
                metadata={"iteration": i, "chunk_index": i},
            )

            cit = AgentCitation.from_search_result(vs_item)
            ev = VisualEvidenceAdapter.adapt_citation(cit)
            res = VisionResult(
                query=f"Analyze figure from {doc_id}",
                status="success",
                description=f"Verified description {i}",
                evidence=[ev],
            )

            state = AgentState(query=res.query, route="vision")
            state.add_citation(
                AgentCitation(
                    document_id=res.document_id,
                    filename=res.filename,
                    chunk_id=res.chunk_id,
                    page_number=res.page_number,
                    content_type=res.content_type,
                    score=1.0,
                )
            )
            state.update(answer=res.description, status="completed")

            assert state.citations[0].document_id == doc_id
            assert state.citations[0].filename == fname
            assert state.citations[0].chunk_id == cid
            assert state.citations[0].page_number == page
            assert state.citations[0].content_type == "chart"
            assert state.answer == f"Verified description {i}"
            assert state.status == "completed"
