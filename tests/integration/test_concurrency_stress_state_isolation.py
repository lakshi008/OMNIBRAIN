"""
OmniBrain Member 4 — Day 28 Concurrency Stress & State Isolation Certification Tests.

Verifies that multi-threaded, concurrent execution of existing OMNIBRAIN workflows
guarantees complete state isolation without data races, shared mutable contamination,
or cross-request lineage corruption across Ingestion, Search, Vision, and Supervisor layers:

    Ingestion (Member 1)
         ↓
    Search / Retrieval (Member 2)
         ↓
    Vision (Member 3)
         ↓
    Downstream Supervisor / Agent Consumers

Focus areas:
 1. Basic concurrent execution across 4 independent requests (A, B, C, D).
 2. Per-request metadata isolation under multi-threaded concurrency.
 3. Document identity isolation under concurrent execution.
 4. Chunk identity isolation under concurrent execution.
 5. Citation isolation across concurrent requests.
 6. VisualEvidence isolation across concurrent requests.
 7. End-to-end lineage isolation (Doc -> Chunk -> Retrieval -> Citation -> Evidence -> Vision).
 8. Repeated concurrency rounds (Rounds 1 to 5) with zero state accumulation.
 9. Higher concurrency scaling (2, 4, 8 concurrent worker threads).
 10. Same-document / different-request concurrency (shared document, distinct requests).
 11. Different-document / same-metadata-shape concurrency.
 12. Serialization / deserialization round-trip under concurrency.
 13. Failure isolation under concurrency (1 failed request, 3 successful requests).
 14. Cancellation token isolation under concurrency (VisionCancellationToken).
 15. Caller-owned object mutation safety under multi-threaded execution.
 16. Resource safety and zero workspace pollution.

Constraints:
 - 100% Offline: Zero external APIs, network, real LLMs, or production secrets.
 - Zero production code modified.
 - Only observable behavior guaranteed by existing public contracts tested.
"""

from __future__ import annotations

import concurrent.futures
import copy
import sys
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# Ingestion Subsystem (Member 1)
from ingestion.models import (
    ChunkValidationResult,
    ChunkingResult,
    DocumentChunk,
    DocumentMetadata,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    VectorSearchResult,
)
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.retrieval_processor import build_retrieval_context, process_retrieval_results

# Search / Agents Subsystem (Member 2)
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    AgentState,
    SearchRequest,
    SearchResult,
)
from agents.exceptions import AgentValidationError

# Vision Subsystem (Member 3)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.result_normalizer import VisionResultNormalizer
from vision.exceptions import (
    VisionAgentError,
    VisionCancellationError,
    VisionEvidenceError,
    VisionInputValidationError,
)
from vision.lifecycle import (
    VisionCancellationToken,
    VisionExecutionLifecycle,
    VisionExecutionStage,
)

# ---------------------------------------------------------------------------
# Synthetic Concurrency Fixtures
# ---------------------------------------------------------------------------

DOC_A = "DAY28_DOC_A"
DOC_B = "DAY28_DOC_B"
DOC_C = "DAY28_DOC_C"
DOC_D = "DAY28_DOC_D"

FILE_A = "day28_doc_a.pdf"
FILE_B = "day28_doc_b.pdf"
FILE_C = "day28_doc_c.pdf"
FILE_D = "day28_doc_d.pdf"

REQUEST_CONFIGS = [
    ("REQ_A", DOC_A, FILE_A, "A"),
    ("REQ_B", DOC_B, FILE_B, "B"),
    ("REQ_C", DOC_C, FILE_C, "C"),
    ("REQ_D", DOC_D, FILE_D, "D"),
]


def _make_concurrent_vsr(doc_id: str, filename: str, req_id: str, count: int = 2) -> list[VectorSearchResult]:
    results = []
    modalities = ["text", "image"]
    for i in range(count):
        ct = modalities[i % len(modalities)]
        vsr = VectorSearchResult(
            chunk_id=f"chunk_{doc_id}_{i+1}",
            score=0.95 - (i * 0.05),
            document_id=doc_id,
            filename=filename,
            page_number=i + 1,
            chunk_index=i,
            content_type=ct,
            content=f"Content for {req_id} from {doc_id} chunk {i+1}",
            metadata={"req_id": req_id, "document_id": doc_id, "chunk_idx": i},
        )
        results.append(vsr)
    return results


def _execute_concurrent_request(
    req_id: str,
    doc_id: str,
    filename: str,
    marker: str,
) -> tuple[AgentResponse, VisionResult]:
    """Runs a complete Member 1 -> Member 2 -> Member 3 workflow for a single concurrent request."""
    # Step 1: Member 1 Retrieval Processing
    vsrs = _make_concurrent_vsr(doc_id, filename, req_id, count=2)
    processed = process_retrieval_results(vsrs, min_score=0.5, max_results=10)
    ctx = build_retrieval_context(processed)

    # Step 2: Member 2 Agent Citations & Response
    citations = [AgentCitation.from_search_result(v) for v in processed]
    agent_resp = AgentResponse(
        answer=f"Answer for {req_id}",
        agent_name="SearchAgent",
        status="success",
        citations=citations,
        metadata={"req_id": req_id, "marker": marker, "context": ctx},
    )

    # Step 3: Member 3 Vision Evidence Adaptation & Normalization
    img_citations = agent_resp.image_results
    evidence = VisualEvidenceAdapter.adapt_batch(img_citations)
    normalizer = VisionResultNormalizer()
    raw_res = VisionResult(
        query=f"Visual query for {req_id}",
        status="success",
        description=f"Visual summary for {req_id}",
        evidence=evidence,
        metadata={"req_id": req_id},
    )
    normalized_res = normalizer.normalize(raw_res)

    return agent_resp, normalized_res


# ===========================================================================
# 1. Basic Concurrent Request Execution (A, B, C, D)
# ===========================================================================

class TestBasicConcurrentExecution:
    """Verifies that 4 independent requests run concurrently with complete isolation."""

    def test_four_concurrent_requests_isolation(self) -> None:
        def worker(cfg: tuple[str, str, str, str]) -> tuple[str, AgentResponse, VisionResult]:
            req_id, doc_id, filename, marker = cfg
            resp, vis = _execute_concurrent_request(req_id, doc_id, filename, marker)
            return req_id, resp, vis

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, cfg) for cfg in REQUEST_CONFIGS]
            results = [f.result() for f in futures]

        assert len(results) == 4
        result_map = {req_id: (resp, vis) for req_id, resp, vis in results}

        for req_id, doc_id, filename, marker in REQUEST_CONFIGS:
            assert req_id in result_map
            resp, vis = result_map[req_id]

            # AgentResponse checks
            assert resp.is_success
            assert resp.metadata["req_id"] == req_id
            assert resp.metadata["marker"] == marker
            assert resp.unique_documents == [doc_id]
            assert all(c.document_id == doc_id for c in resp.citations)
            assert all(c.filename == filename for c in resp.citations)

            # VisionResult checks
            assert vis.is_success
            assert vis.document_id == doc_id
            assert vis.filename == filename
            assert len(vis.evidence) == 1
            assert vis.evidence[0].document_id == doc_id


# ===========================================================================
# 2. Metadata, Document & Chunk Isolation Under Concurrency
# ===========================================================================

class TestMetadataDocumentChunkIsolation:
    """Verifies metadata, document, and chunk boundaries are strictly preserved under concurrency."""

    def test_metadata_no_cross_thread_bleed(self) -> None:
        def worker(cfg: tuple[str, str, str, str]) -> tuple[str, dict[str, Any]]:
            req_id, doc_id, filename, marker = cfg
            resp, vis = _execute_concurrent_request(req_id, doc_id, filename, marker)
            return req_id, resp.metadata

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, cfg) for cfg in REQUEST_CONFIGS]
            results = dict([f.result() for f in futures])

        assert results["REQ_A"]["marker"] == "A"
        assert results["REQ_B"]["marker"] == "B"
        assert results["REQ_C"]["marker"] == "C"
        assert results["REQ_D"]["marker"] == "D"

    def test_chunk_ids_remain_strictly_per_request(self) -> None:
        def worker(cfg: tuple[str, str, str, str]) -> tuple[str, list[str]]:
            req_id, doc_id, filename, marker = cfg
            resp, vis = _execute_concurrent_request(req_id, doc_id, filename, marker)
            return req_id, [c.chunk_id for c in resp.citations]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, cfg) for cfg in REQUEST_CONFIGS]
            results = dict([f.result() for f in futures])

        for req_id, doc_id, _, _ in REQUEST_CONFIGS:
            chunk_ids = results[req_id]
            assert all(cid.startswith(f"chunk_{doc_id}_") for cid in chunk_ids)


# ===========================================================================
# 3. Citation, Evidence & Lineage Isolation
# ===========================================================================

class TestCitationEvidenceLineageIsolation:
    """Verifies citation, evidence, and lineage chains remain clean across concurrent executions."""

    def test_citations_and_evidence_isolation(self) -> None:
        def worker(cfg: tuple[str, str, str, str]) -> tuple[str, list[str], list[str]]:
            req_id, doc_id, filename, marker = cfg
            resp, vis = _execute_concurrent_request(req_id, doc_id, filename, marker)
            citation_docs = [c.document_id for c in resp.citations]
            evidence_docs = [e.document_id for e in vis.evidence]
            return req_id, citation_docs, evidence_docs

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, cfg) for cfg in REQUEST_CONFIGS]
            results = [f.result() for f in futures]

        for req_id, citation_docs, evidence_docs in results:
            cfg = next(c for c in REQUEST_CONFIGS if c[0] == req_id)
            expected_doc = cfg[1]
            assert all(d == expected_doc for d in citation_docs)
            assert all(d == expected_doc for d in evidence_docs)


# ===========================================================================
# 4. Repeated Concurrency Rounds (Rounds 1 to 5)
# ===========================================================================

class TestRepeatedConcurrencyRounds:
    """Verifies repeated multi-threaded execution causes zero state accumulation."""

    def test_five_concurrency_rounds_stability(self) -> None:
        for round_idx in range(5):
            def worker(cfg: tuple[str, str, str, str]) -> tuple[str, AgentResponse]:
                req_id, doc_id, filename, marker = cfg
                resp, _ = _execute_concurrent_request(req_id, doc_id, filename, marker)
                return req_id, resp

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(worker, cfg) for cfg in REQUEST_CONFIGS]
                results = dict([f.result() for f in futures])

            assert len(results) == 4
            for req_id, doc_id, _, marker in REQUEST_CONFIGS:
                resp = results[req_id]
                assert resp.is_success
                assert len(resp.citations) == 2
                assert resp.unique_documents == [doc_id]
                assert resp.metadata["marker"] == marker


# ===========================================================================
# 5. Higher Concurrency Scaling (2, 4, 8 Worker Threads)
# ===========================================================================

class TestHigherConcurrencyScaling:
    """Verifies scaling concurrent threads across 2, 4, and 8 workers."""

    @pytest.mark.parametrize("worker_count", [2, 4, 8])
    def test_scaling_concurrency_workers(self, worker_count: int) -> None:
        def worker(idx: int) -> tuple[int, str, AgentResponse]:
            cfg = REQUEST_CONFIGS[idx % len(REQUEST_CONFIGS)]
            req_id, doc_id, filename, marker = cfg
            scoped_req_id = f"{req_id}_{idx}"
            resp, _ = _execute_concurrent_request(scoped_req_id, doc_id, filename, marker)
            return idx, doc_id, resp

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(worker, i) for i in range(worker_count)]
            results = [f.result() for f in futures]

        assert len(results) == worker_count
        for idx, expected_doc, resp in results:
            assert resp.is_success
            assert resp.unique_documents == [expected_doc]


# ===========================================================================
# 6. Same Document / Different Requests
# ===========================================================================

class TestSameDocumentDifferentRequests:
    """Verifies multiple concurrent requests accessing the SAME document maintain independent state."""

    def test_shared_document_distinct_request_isolation(self) -> None:
        shared_doc = "DOC_SHARED_DAY28"
        shared_file = "shared_day28.pdf"

        def worker(req_tag: str) -> tuple[str, AgentResponse, VisionResult]:
            resp, vis = _execute_concurrent_request(f"REQ_{req_tag}", shared_doc, shared_file, req_tag)
            return req_tag, resp, vis

        tags = ["ALPHA", "BETA", "GAMMA", "DELTA"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, tag) for tag in tags]
            results = dict([(f.result()[0], (f.result()[1], f.result()[2])) for f in futures])

        for tag in tags:
            resp, vis = results[tag]
            assert resp.is_success
            assert resp.metadata["req_id"] == f"REQ_{tag}"
            assert resp.metadata["marker"] == tag
            assert resp.unique_documents == [shared_doc]

            assert vis.is_success
            assert vis.metadata["req_id"] == f"REQ_{tag}"
            assert vis.document_id == shared_doc


# ===========================================================================
# 7. Different Document / Same Metadata Shape
# ===========================================================================

class TestDifferentDocumentSameMetadataShape:
    """Verifies equivalent metadata dictionary shapes retain request-specific values."""

    def test_same_metadata_shape_different_values(self) -> None:
        def worker(cfg: tuple[str, str, str, str]) -> tuple[str, str]:
            req_id, doc_id, filename, marker = cfg
            resp, _ = _execute_concurrent_request(req_id, doc_id, filename, marker)
            return req_id, resp.metadata["marker"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, cfg) for cfg in REQUEST_CONFIGS]
            results = dict([f.result() for f in futures])

        assert results == {"REQ_A": "A", "REQ_B": "B", "REQ_C": "C", "REQ_D": "D"}


# ===========================================================================
# 8. Serialization Round-Trip Under Concurrency
# ===========================================================================

class TestSerializationUnderConcurrency:
    """Verifies to_dict() and from_dict() roundtrips executed concurrently across threads."""

    def test_concurrent_serialization_roundtrip(self) -> None:
        def worker(cfg: tuple[str, str, str, str]) -> tuple[str, bool]:
            req_id, doc_id, filename, marker = cfg
            resp, vis = _execute_concurrent_request(req_id, doc_id, filename, marker)

            # Roundtrip AgentResponse
            resp_dict = resp.to_dict()
            restored_resp = AgentResponse.from_dict(resp_dict)
            resp_ok = (
                restored_resp.answer == resp.answer
                and restored_resp.metadata["req_id"] == req_id
                and restored_resp.unique_documents == [doc_id]
            )

            # Roundtrip VisionResult
            vis_dict = vis.to_dict()
            restored_vis = VisionResult.from_dict(vis_dict)
            vis_ok = (
                restored_vis.document_id == doc_id
                and restored_vis.metadata["req_id"] == req_id
                and len(restored_vis.evidence) == 1
            )

            return req_id, (resp_ok and vis_ok)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, cfg) for cfg in REQUEST_CONFIGS]
            results = dict([f.result() for f in futures])

        assert all(results.values())


# ===========================================================================
# 9. Failure Isolation Under Concurrency
# ===========================================================================

class TestFailureIsolationUnderConcurrency:
    """Verifies a failure in one concurrent thread does not contaminate other concurrent threads."""

    def test_one_failure_three_successes_concurrency(self) -> None:
        def worker(cfg: tuple[str, str, str, str]) -> tuple[str, str, Any]:
            req_id, doc_id, filename, marker = cfg
            if req_id == "REQ_B":
                # Inject synthetic validation failure on Thread B
                try:
                    AgentCitation(document_id="", filename="f.pdf", chunk_id="ck")
                    return req_id, "unexpected_success", None
                except AgentValidationError as e:
                    return req_id, "caught_error", str(e)
            else:
                resp, vis = _execute_concurrent_request(req_id, doc_id, filename, marker)
                return req_id, "success", resp

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, cfg) for cfg in REQUEST_CONFIGS]
            results = dict([(r[0], (r[1], r[2])) for r in [f.result() for f in futures]])

        # Thread B failed cleanly with expected error
        status_b, err_b = results["REQ_B"]
        assert status_b == "caught_error"
        assert len(err_b) > 0

        # Threads A, C, D succeeded cleanly without contamination
        for req_id in ("REQ_A", "REQ_C", "REQ_D"):
            status, resp = results[req_id]
            assert status == "success"
            assert resp.is_success


# ===========================================================================
# 10. Cancellation Token Isolation Under Concurrency
# ===========================================================================

class TestCancellationIsolationUnderConcurrency:
    """Verifies VisionCancellationToken cancellation in one thread does not affect other tokens."""

    def test_cancellation_token_thread_isolation(self) -> None:
        tokens = {req_id: VisionCancellationToken() for req_id, _, _, _ in REQUEST_CONFIGS}

        def worker(req_id: str) -> tuple[str, bool, bool]:
            token = tokens[req_id]
            if req_id == "REQ_C":
                token.cancel("Cancelled thread C")

            # Check if raise_if_cancelled raises
            did_raise = False
            try:
                token.raise_if_cancelled()
            except VisionCancellationError:
                did_raise = True

            return req_id, token.is_cancelled, did_raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, req_id) for req_id in tokens]
            results = dict([(r[0], (r[1], r[2])) for r in [f.result() for f in futures]])

        # Only Thread C is cancelled
        for req_id in tokens:
            is_cancelled, did_raise = results[req_id]
            if req_id == "REQ_C":
                assert is_cancelled is True
                assert did_raise is True
            else:
                assert is_cancelled is False
                assert did_raise is False


# ===========================================================================
# 11. Caller-Owned Object Mutation Safety
# ===========================================================================

class TestCallerObjectMutationSafety:
    """Verifies caller-owned input objects are never mutated during concurrent processing."""

    def test_caller_objects_unmodified_after_concurrent_runs(self) -> None:
        original_citations = [
            AgentCitation(document_id=DOC_A, filename=FILE_A, chunk_id="ck_orig", page_number=1, content_type="image")
        ]
        citation_snapshot = copy.deepcopy(original_citations)

        def worker(idx: int) -> bool:
            # Pass caller-owned citation list to adaptation
            ev = VisualEvidenceAdapter.adapt_batch(original_citations)
            return len(ev) == 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            results = [f.result() for f in futures]

        assert all(results)
        # Verify original list and items were not mutated
        assert original_citations == citation_snapshot


# ===========================================================================
# 12. Resource Safety Under Concurrency
# ===========================================================================

class TestResourceSafetyUnderConcurrency:
    """Verifies concurrent execution generates zero temporary files or workspace pollution."""

    def test_zero_disk_artifacts_after_concurrency(self) -> None:
        def worker(cfg: tuple[str, str, str, str]) -> bool:
            resp, vis = _execute_concurrent_request(*cfg)
            return resp.is_success and vis.is_success

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, REQUEST_CONFIGS[i % 4]) for i in range(16)]
            results = [f.result() for f in futures]

        assert all(results)

        root_path = Path(REPO_ROOT)
        unexpected = [
            f.name for f in root_path.iterdir()
            if f.is_file() and f.name.endswith((".tmp", ".temp", ".dump", ".log", ".bak"))
        ]
        assert unexpected == []
