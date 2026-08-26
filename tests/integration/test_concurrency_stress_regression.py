"""
OmniBrain Member 4 — Day 39 Concurrency & Stress Regression Certification.

Verifies the existing OMNIBRAIN system behaves correctly when multiple
independent requests are executed concurrently using Python standard-library
ThreadPoolExecutor (test-only, no production concurrency code added).

Coverage:
  - Sequential baseline (5 requests) for correctness reference
  - Concurrent Level 1: 4 workers, 4 requests
  - Concurrent Level 2: 4 workers, 8 requests
  - Concurrent Level 3: 4 workers, 16 requests
  - Request marker isolation across concurrent executions
  - Document identity isolation across concurrent executions
  - Metadata isolation under concurrency
  - AgentState instance isolation under concurrent mutations
  - AgentCitation isolation per request
  - VisionRequest / VisualEvidence isolation per request
  - Order-independent result verification (keyed by request ID)
  - Repeated concurrent workloads (3 rounds, no cross-round contamination)
  - Same-document concurrent requests (different request IDs → isolated state)
  - Same-content different requests (identity NOT confused)
  - Failure isolation and exception visibility under concurrency
  - Result completeness (N in → N results out where contract guarantees this)
  - Caller-owned input mutation safety under concurrent execution
  - Concurrent serialization (to_dict / from_dict) isolation
  - tracemalloc memory observation
  - Duration timing observation across worker levels

Constraints:
  - 100% offline. Zero external APIs, network, LLM, or production credentials.
  - Zero production code modified.
  - ThreadPoolExecutor exists only inside tests, not production.
  - No locks, queues, retry logic, or async infrastructure added to production.
"""

from __future__ import annotations

import copy
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# ---------------------------------------------------------------------------
# Ingestion Subsystem (Member 1)
# ---------------------------------------------------------------------------
from ingestion.models import (
    DocumentChunk,
    DocumentMetadata,
    EmbeddingPreparationResult,
    PageData,
    ParsedDocument,
    VectorSearchResult,
)
from ingestion.chunk_validator import normalize_chunks
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.retrieval_processor import process_retrieval_results
from ingestion.ingestion_errors import IngestionValidationError

# ---------------------------------------------------------------------------
# Agents / Search Subsystem (Member 2)
# ---------------------------------------------------------------------------
from agents.models import (
    AgentCitation,
    AgentResponse,
    AgentState,
)
from agents.exceptions import AgentValidationError

# ---------------------------------------------------------------------------
# Vision Subsystem (Member 3)
# ---------------------------------------------------------------------------
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import VisionEvidenceError

# ---------------------------------------------------------------------------
# Helpers — Synthetic Request Factory (Section 5)
# ---------------------------------------------------------------------------

def _make_chunks(req_id: str, doc_id: str, page_count: int = 3) -> list[DocumentChunk]:
    """Build a deterministic list of DocumentChunks for one synthetic request."""
    return [
        DocumentChunk(
            chunk_id=f"CHK_{doc_id}_P{p:03d}",
            chunk_index=p - 1,
            document_id=doc_id,
            filename=f"{doc_id}.pdf",
            page_number=p,
            content=f"DAY39_{req_id} :: {doc_id} :: Page {p:03d}",
            content_type="image",
            metadata={
                "day39_request": req_id,
                "day39_document": doc_id,
            },
        )
        for p in range(1, page_count + 1)
    ]


def _process_request(req_id: str, doc_id: str, page_count: int = 3) -> dict[str, Any]:
    """Execute one full ingestion pipeline for a synthetic request. Thread-safe (no shared state)."""
    chunks = _make_chunks(req_id, doc_id, page_count)
    prep = prepare_for_embedding(chunks)

    vsrs = [
        VectorSearchResult(
            chunk_id=r.chunk_id, score=0.95, document_id=r.document_id,
            filename=r.filename, page_number=r.page_number, chunk_index=r.chunk_index,
            content_type=r.content_type, content=r.content, metadata=r.metadata,
        )
        for r in prep.items
    ]
    processed = process_retrieval_results(vsrs, min_score=0.5, max_results=page_count)
    citations = [AgentCitation.from_search_result(r) for r in processed]
    response = AgentResponse(answer=f"Answer for {req_id}", agent_name="Agent", citations=citations)

    evidence = VisualEvidenceAdapter.adapt_batch(citations)
    vision_req = VisionRequest(query=f"Examine {req_id}", evidence=evidence)

    return {
        "req_id": req_id,
        "doc_id": doc_id,
        "document_id": prep.document_id,
        "total_items": prep.total_items,
        "unique_documents": response.unique_documents,
        "citation_doc_ids": [c.document_id for c in citations],
        "vision_doc_ids": [e.document_id for e in vision_req.evidence],
        "first_content": prep.items[0].content,
        "metadata_sample": prep.items[0].metadata,
    }


def _build_requests(count: int) -> list[tuple[str, str]]:
    """Return deterministic (req_id, doc_id) tuples."""
    return [(f"REQ_{i:03d}", f"DOC_{i:03d}") for i in range(1, count + 1)]


# ===========================================================================
# 1. SEQUENTIAL BASELINE (Section 6)
# ===========================================================================

class TestSequentialBaseline:
    """Run 5 requests sequentially to establish the correctness reference."""

    def test_sequential_baseline_5_requests(self) -> None:
        requests = _build_requests(5)
        results: list[dict] = []

        for req_id, doc_id in requests:
            r = _process_request(req_id, doc_id)
            results.append(r)

        assert len(results) == 5
        for req_id, doc_id in requests:
            match = next(r for r in results if r["req_id"] == req_id)
            assert match["document_id"] == doc_id
            assert match["total_items"] == 3
            assert match["unique_documents"] == [doc_id]
            assert all(cid == doc_id for cid in match["citation_doc_ids"])
            assert f"DAY39_{req_id}" in match["first_content"]
            assert match["metadata_sample"]["day39_request"] == req_id


# ===========================================================================
# 2. CONCURRENT LEVELS (Sections 7, 8, 17)
# ===========================================================================

class TestConcurrentLevels:
    """Execute controlled concurrent workloads at 3 stress levels."""

    def _run_concurrent(self, count: int, max_workers: int) -> list[dict]:
        requests = _build_requests(count)
        results: list[dict] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_request, req_id, doc_id): req_id for req_id, doc_id in requests}
            for fut in as_completed(futures):
                results.append(fut.result())  # raises if worker raised

        return results

    def test_level1_4_concurrent_requests(self) -> None:
        results = self._run_concurrent(count=4, max_workers=4)
        assert len(results) == 4
        doc_ids = {r["document_id"] for r in results}
        assert doc_ids == {f"DOC_{i:03d}" for i in range(1, 5)}

    def test_level2_8_concurrent_requests(self) -> None:
        results = self._run_concurrent(count=8, max_workers=4)
        assert len(results) == 8
        doc_ids = {r["document_id"] for r in results}
        assert doc_ids == {f"DOC_{i:03d}" for i in range(1, 9)}

    def test_level3_16_concurrent_requests(self) -> None:
        results = self._run_concurrent(count=16, max_workers=4)
        assert len(results) == 16
        doc_ids = {r["document_id"] for r in results}
        assert doc_ids == {f"DOC_{i:03d}" for i in range(1, 17)}


# ===========================================================================
# 3. REQUEST & DOCUMENT ISOLATION (Sections 9, 10, 11)
# ===========================================================================

class TestConcurrentIsolation:
    """Verify marker, document, and metadata isolation under concurrent execution."""

    def test_request_marker_isolation_concurrent(self) -> None:
        count = 8
        requests = _build_requests(count)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_process_request, r, d): r for r, d in requests}
            results = {futures[f]: f.result() for f in as_completed(futures)}

        for req_id, result in results.items():
            assert f"DAY39_{req_id}" in result["first_content"]
            # Must NOT contain another request's marker
            for other_req, _ in requests:
                if other_req != req_id:
                    assert f"DAY39_{other_req}" not in result["first_content"]

    def test_document_isolation_concurrent(self) -> None:
        count = 8
        requests = _build_requests(count)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_process_request, r, d): (r, d) for r, d in requests}
            for fut in as_completed(futures):
                req_id, doc_id = futures[fut]
                result = fut.result()
                assert result["document_id"] == doc_id
                assert result["unique_documents"] == [doc_id]
                assert all(cid == doc_id for cid in result["citation_doc_ids"])
                assert all(vid == doc_id for vid in result["vision_doc_ids"])

    def test_metadata_isolation_concurrent(self) -> None:
        count = 8
        requests = _build_requests(count)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_process_request, r, d): (r, d) for r, d in requests}
            for fut in as_completed(futures):
                req_id, doc_id = futures[fut]
                result = fut.result()
                assert result["metadata_sample"]["day39_request"] == req_id
                assert result["metadata_sample"]["day39_document"] == doc_id


# ===========================================================================
# 4. STATE ISOLATION (Section 12)
# ===========================================================================

class TestAgentStateIsolationConcurrent:
    """AgentState objects mutated in separate workers must not bleed into each other."""

    def _mutate_state(self, req_id: str) -> dict:
        state = AgentState(query=f"Query for {req_id}")
        cit = AgentCitation(
            document_id=f"DOC_{req_id}", filename=f"{req_id}.pdf",
            chunk_id=f"CHK_{req_id}_001", content_type="image",
        )
        state.add_citation(cit)
        state.add_error(f"Non-critical warning for {req_id}")
        return {
            "req_id": req_id,
            "citation_count": len(state.citations),
            "error_count": len(state.errors),
            "citation_doc_id": state.citations[0].document_id,
        }

    def test_agent_state_isolation_across_threads(self) -> None:
        req_ids = [f"REQ_{i:03d}" for i in range(1, 9)]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(self._mutate_state, r): r for r in req_ids}
            for fut in as_completed(futures):
                req_id = futures[fut]
                result = fut.result()
                # Each state has exactly its own 1 citation and 1 error
                assert result["citation_count"] == 1
                assert result["error_count"] == 1
                assert result["citation_doc_id"] == f"DOC_{req_id}"


# ===========================================================================
# 5. ORDER INDEPENDENCE (Section 15)
# ===========================================================================

class TestConcurrentOrderIndependence:
    """Submit in shuffled order; verify results keyed by identity, not completion order."""

    def test_shuffled_submission_order_correct_results(self) -> None:
        # Deliberately shuffled order
        shuffled = [
            ("REQ_005", "DOC_005"),
            ("REQ_002", "DOC_002"),
            ("REQ_008", "DOC_008"),
            ("REQ_001", "DOC_001"),
            ("REQ_007", "DOC_007"),
            ("REQ_003", "DOC_003"),
            ("REQ_006", "DOC_006"),
            ("REQ_004", "DOC_004"),
        ]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_process_request, r, d): (r, d) for r, d in shuffled}
            for fut in as_completed(futures):
                req_id, doc_id = futures[fut]
                result = fut.result()
                assert result["document_id"] == doc_id
                assert f"DAY39_{req_id}" in result["first_content"]


# ===========================================================================
# 6. REPEATED CONCURRENCY (Section 16)
# ===========================================================================

class TestRepeatedConcurrency:
    """Run 3 rounds of identical concurrent workloads; verify no cross-round contamination."""

    def test_three_rounds_no_state_accumulation(self) -> None:
        for round_num in range(1, 4):
            requests = _build_requests(8)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_process_request, r, d): (r, d) for r, d in requests}
                for fut in as_completed(futures):
                    req_id, doc_id = futures[fut]
                    result = fut.result()
                    assert result["document_id"] == doc_id
                    assert result["total_items"] == 3
                    assert result["unique_documents"] == [doc_id]


# ===========================================================================
# 7. SAME-DOCUMENT & SAME-CONTENT CONCURRENCY (Sections 19, 20)
# ===========================================================================

class TestSameDocumentAndContentConcurrency:
    """Multiple requests share the same document ID or identical content but have distinct request IDs."""

    def _process_shared_doc(self, req_id: str, doc_id: str) -> dict:
        chunks = _make_chunks(req_id, doc_id)
        prep = prepare_for_embedding(chunks)
        return {
            "req_id": req_id,
            "doc_id": doc_id,
            "document_id": prep.document_id,
            "first_content": prep.items[0].content,
            "metadata_req": prep.items[0].metadata["day39_request"],
        }

    def test_same_document_concurrent_state_independence(self) -> None:
        shared_doc = "DOC_SHARED"
        req_ids = [f"REQ_SH_{i}" for i in range(1, 5)]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(self._process_shared_doc, r, shared_doc): r for r in req_ids}
            for fut in as_completed(futures):
                req_id = futures[fut]
                result = fut.result()
                assert result["document_id"] == shared_doc
                assert result["metadata_req"] == req_id
                assert f"DAY39_{req_id}" in result["first_content"]

    def test_same_content_different_requests_identity_preserved(self) -> None:
        shared_content = "DAY39 IDENTICAL PAYLOAD FOR MULTIPLE REQUESTS"

        def make_identical_chunk(req_id: str, doc_id: str) -> dict:
            chunk = DocumentChunk(
                chunk_id=f"CHK_{req_id}",
                chunk_index=0,
                document_id=doc_id,
                filename=f"{doc_id}.pdf",
                page_number=1,
                content=shared_content,
                content_type="image",
            )
            prep = prepare_for_embedding([chunk])
            return {"req_id": req_id, "document_id": prep.document_id}

        pairs = [(f"REQ_SAME_{i}", f"DOC_SAME_{i}") for i in range(1, 5)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(make_identical_chunk, r, d): d for r, d in pairs}
            for fut in as_completed(futures):
                expected_doc = futures[fut]
                result = fut.result()
                assert result["document_id"] == expected_doc


# ===========================================================================
# 8. FAILURE ISOLATION & EXCEPTION VISIBILITY (Sections 21, 22, 23)
# ===========================================================================

class TestConcurrentFailureIsolation:
    """Valid requests remain successful when one concurrent request fails."""

    def _maybe_failing_request(self, req_id: str, doc_id: str, fail: bool) -> dict:
        if fail:
            # AgentCitation with empty document_id → expected AgentValidationError
            AgentCitation("", "bad.pdf", "CHK_BAD")
        return _process_request(req_id, doc_id)

    def test_failure_isolated_valid_requests_succeed(self) -> None:
        tasks = [
            ("REQ_VALID_A", "DOC_VA", False),
            ("REQ_INVALID", "DOC_INV", True),   # will raise
            ("REQ_VALID_B", "DOC_VB", False),
        ]

        succeeded: list[dict] = []
        errors: list[Exception] = []

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(self._maybe_failing_request, r, d, f): r for r, d, f in tasks}
            for fut in as_completed(futures):
                try:
                    succeeded.append(fut.result())
                except (AgentValidationError, Exception) as exc:
                    errors.append(exc)

        # Exactly one error and two successes
        assert len(errors) == 1
        assert len(succeeded) == 2
        assert all(r["doc_id"] in ("DOC_VA", "DOC_VB") for r in succeeded)


# ===========================================================================
# 9. INPUT MUTATION SAFETY (Section 25)
# ===========================================================================

class TestConcurrentInputMutationSafety:
    """Caller-owned metadata dicts must remain unchanged after concurrent processing."""

    def _process_with_owned_meta(self, req_id: str, doc_id: str, caller_meta: dict) -> str:
        chunk = DocumentChunk(
            chunk_id=f"CHK_{req_id}", chunk_index=0, document_id=doc_id,
            filename=f"{doc_id}.pdf", page_number=1,
            content=f"DAY39_{req_id} payload", content_type="image",
            metadata=caller_meta,
        )
        prepare_for_embedding([chunk])
        return req_id

    def test_caller_metadata_not_mutated_concurrently(self) -> None:
        entries = [(f"REQ_{i:03d}", f"DOC_{i:03d}", {"owner": f"caller_{i}", "val": i}) for i in range(1, 9)]
        snapshots = {r: copy.deepcopy(m) for r, _, m in entries}

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self._process_with_owned_meta, r, d, m) for r, d, m in entries]
            for fut in as_completed(futures):
                fut.result()

        for req_id, _, meta in entries:
            assert meta == snapshots[req_id]


# ===========================================================================
# 10. CONCURRENT SERIALIZATION ISOLATION (Section 26)
# ===========================================================================

class TestConcurrentSerializationIsolation:
    """Concurrent to_dict/from_dict cycles must not merge source identities."""

    def _serialize_roundtrip(self, req_id: str, doc_id: str) -> dict:
        cit = AgentCitation(
            document_id=doc_id, filename=f"{doc_id}.pdf",
            chunk_id=f"CHK_{req_id}", content_type="image",
            metadata={"req": req_id},
        )
        d = cit.to_dict()
        restored = AgentCitation.from_dict(d)
        return {
            "req_id": req_id,
            "document_id": restored.document_id,
            "metadata_req": restored.metadata.get("req"),
        }

    def test_concurrent_serialization_isolation(self) -> None:
        pairs = _build_requests(8)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(self._serialize_roundtrip, r, d): (r, d) for r, d in pairs}
            for fut in as_completed(futures):
                req_id, doc_id = futures[fut]
                result = fut.result()
                assert result["document_id"] == doc_id
                assert result["metadata_req"] == req_id


# ===========================================================================
# 11. RESOURCE OBSERVATION & PERFORMANCE TIMING (Sections 29, 30)
# ===========================================================================

class TestConcurrentResourceAndPerformanceObservation:
    """Measure memory footprint and duration at each worker level."""

    def _run_timed(self, count: int, max_workers: int) -> tuple[float, int, float]:
        requests = _build_requests(count)
        tracemalloc.start()
        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_process_request, r, d) for r, d in requests]
            results = [f.result() for f in as_completed(futures)]

        elapsed = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return elapsed, len(results), peak_mem

    def test_performance_and_memory_across_levels(self) -> None:
        measurements = []
        for count, workers in ((4, 4), (8, 4), (16, 4)):
            elapsed, result_count, peak_mem_bytes = self._run_timed(count, workers)
            measurements.append({
                "count": count,
                "workers": workers,
                "elapsed_sec": round(elapsed, 4),
                "result_count": result_count,
                "peak_mem_kb": round(peak_mem_bytes / 1024, 2),
            })

        # Correctness: all requests returned
        assert measurements[0]["result_count"] == 4
        assert measurements[1]["result_count"] == 8
        assert measurements[2]["result_count"] == 16

        # Sanity: elapsed > 0, memory > 0
        for m in measurements:
            assert m["elapsed_sec"] > 0.0
            assert m["peak_mem_kb"] > 0.0
            # No obvious runaway memory (below 50 MB for these workloads)
            assert m["peak_mem_kb"] < 50 * 1024
