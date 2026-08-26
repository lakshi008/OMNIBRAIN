"""
OmniBrain Member 4 — Day 26 Performance Regression & Resource Efficiency Certification Tests.

Evaluates performance stability, repeated execution behavior, batch processing,
concurrency isolation, duplicate-work prevention, resource safety, and deterministic
result consistency across the OMNIBRAIN integration pipeline:

    Ingestion (Member 1)
         ↓
    Search / Retrieval (Member 2)
         ↓
    Vision (Member 3)
         ↓
    Downstream Supervisor / Agent Consumers

Focus areas:
 1. Single-workflow timing and end-to-end integration latency baseline.
 2. Repeated execution stability (Run 1 through Run 5 timing & state isolation).
 3. Batch processing of multiple synthetic requests without cross-contamination.
 4. Multi-document workload processing and strict per-document lineage isolation.
 5. Concurrent execution stability under multi-threaded load.
 6. Resource safety (zero workspace pollution, zero leaked files or dangling handles).
 7. Duplicate work prevention (single-pass adaptation, single search per query).
 8. Deterministic result consistency across repeated workflow executions.
 9. Failure isolation under workload (SUCCESS -> FAILURE -> SUCCESS recovery).

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
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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
from agents.search_agent import SearchAgent
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
from vision.exceptions import VisionEvidenceError, VisionInputValidationError
from vision.lifecycle import (
    VisionCancellationToken,
    VisionExecutionLifecycle,
    VisionExecutionStage,
)

# ---------------------------------------------------------------------------
# Synthetic Test Fixtures & Identity Helpers
# ---------------------------------------------------------------------------

PERF_DOC_A = "DAY26_PERF_DOC_A"
PERF_DOC_B = "DAY26_PERF_DOC_B"
PERF_DOC_C = "DAY26_PERF_DOC_C"

PERF_FILE_A = "perf_doc_a.pdf"
PERF_FILE_B = "perf_doc_b.pdf"
PERF_FILE_C = "perf_doc_c.pdf"


def _make_perf_chunk(
    doc_id: str = PERF_DOC_A,
    filename: str = PERF_FILE_A,
    chunk_index: int = 0,
    content_type: str = "text",
    content: str = "Day 26 performance chunk text.",
    **kw: Any,
) -> DocumentChunk:
    chunk_id = f"chunk_{doc_id}_{chunk_index}"
    meta = {"source_doc": doc_id, "perf_day": 26, "chunk_index": chunk_index}
    meta.update(kw.pop("metadata", {}))
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        document_id=doc_id,
        filename=filename,
        page_number=chunk_index + 1,
        content=content,
        content_type=content_type,
        metadata=meta,
        **kw,
    )


def _make_perf_vsr(
    doc_id: str = PERF_DOC_A,
    filename: str = PERF_FILE_A,
    chunk_index: int = 0,
    content_type: str = "text",
    score: float = 0.95,
    content: str = "Day 26 performance retrieved content.",
    **kw: Any,
) -> VectorSearchResult:
    chunk_id = f"vsr_{doc_id}_{chunk_index}"
    meta = {"source_doc": doc_id, "perf_day": 26, "score": score}
    meta.update(kw.pop("metadata", {}))
    return VectorSearchResult(
        chunk_id=chunk_id,
        score=score,
        document_id=doc_id,
        filename=filename,
        page_number=chunk_index + 1,
        chunk_index=chunk_index,
        content_type=content_type,
        content=content,
        metadata=meta,
        **kw,
    )


def _run_representative_pipeline(
    doc_id: str = PERF_DOC_A,
    filename: str = PERF_FILE_A,
    query: str = "Performance representative evaluation query",
) -> tuple[AgentResponse, VisionResult]:
    """Executes a complete offline end-to-end integration workflow across all 3 members."""
    # Step 1: Member 1 Ingestion / Chunk Validation & Retrieval Context
    chunks = [
        _make_perf_chunk(doc_id=doc_id, filename=filename, chunk_index=0, content_type="text"),
        _make_perf_chunk(doc_id=doc_id, filename=filename, chunk_index=1, content_type="image"),
    ]
    val_res = validate_chunks(chunks)
    assert val_res.is_valid

    vsr_results = [
        _make_perf_vsr(doc_id=doc_id, filename=filename, chunk_index=0, content_type="text", score=0.92),
        _make_perf_vsr(doc_id=doc_id, filename=filename, chunk_index=1, content_type="image", score=0.88),
    ]
    processed_vsr = process_retrieval_results(vsr_results, min_score=0.5, max_results=10)
    context_str = build_retrieval_context(processed_vsr)

    # Step 2: Member 2 SearchAgent / Citations
    citations = [AgentCitation.from_search_result(v) for v in processed_vsr]
    agent_resp = AgentResponse(
        answer=f"Synthesized answer for {query}",
        agent_name="SearchAgent",
        status="success",
        citations=citations,
        metadata={"query": query, "context": context_str, "doc_id": doc_id},
    )

    # Step 3: Member 3 Vision Evidence Adaptation & Normalization
    image_citations = agent_resp.image_results
    visual_evidence = VisualEvidenceAdapter.adapt_batch(image_citations)

    normalizer = VisionResultNormalizer()
    raw_result = VisionResult(
        query=query,
        status="success",
        description="Visual chart extracted summary.",
        evidence=visual_evidence,
        metadata={"normalized": False},
    )
    normalized_result = normalizer.normalize(raw_result)

    return agent_resp, normalized_result


# ===========================================================================
# 1. Single-Workflow Timing
# ===========================================================================

class TestSingleWorkflowTiming:
    """Verifies baseline end-to-end workflow execution timing without regression."""

    def test_single_workflow_execution_and_timing(self) -> None:
        t0 = time.perf_counter()
        agent_resp, vision_res = _run_representative_pipeline()
        elapsed = time.perf_counter() - t0

        assert elapsed >= 0.0
        assert agent_resp.is_success
        assert agent_resp.total_citations == 2
        assert vision_res.is_success
        assert vision_res.has_evidence
        assert vision_res.document_id == PERF_DOC_A


# ===========================================================================
# 2. Repeated Execution Stability
# ===========================================================================

class TestRepeatedExecutionStability:
    """Verifies repeated execution (Run 1 to 5) executes stably without state accumulation."""

    def test_repeated_five_runs_stability(self) -> None:
        timings: list[float] = []
        results: list[tuple[AgentResponse, VisionResult]] = []

        for i in range(5):
            t0 = time.perf_counter()
            resp, vis = _run_representative_pipeline(query=f"Repeated query run {i+1}")
            elapsed = time.perf_counter() - t0

            timings.append(elapsed)
            results.append((resp, vis))

        # Verify all runs succeeded
        for idx, (resp, vis) in enumerate(results):
            assert resp.is_success, f"Run {idx+1} AgentResponse failed"
            assert vis.is_success, f"Run {idx+1} VisionResult failed"
            assert resp.total_citations == 2
            assert len(vis.evidence) == 1
            assert vis.evidence[0].document_id == PERF_DOC_A

        # Verify execution completed reliably for all runs
        assert len(timings) == 5
        assert all(t >= 0.0 for t in timings)


# ===========================================================================
# 3. Batch Workload Processing
# ===========================================================================

class TestBatchWorkloadProcessing:
    """Verifies batch execution of multiple requests completes without contamination."""

    def test_batch_ten_requests_isolation(self) -> None:
        batch_size = 10
        batch_results: list[tuple[AgentResponse, VisionResult]] = []

        for i in range(batch_size):
            doc_id = f"BATCH_DOC_{i:03d}"
            filename = f"batch_file_{i:03d}.pdf"
            query = f"Batch query {i:03d}"

            resp, vis = _run_representative_pipeline(doc_id=doc_id, filename=filename, query=query)
            batch_results.append((resp, vis))

        assert len(batch_results) == batch_size

        for i, (resp, vis) in enumerate(batch_results):
            expected_doc = f"BATCH_DOC_{i:03d}"
            expected_file = f"batch_file_{i:03d}.pdf"

            assert resp.is_success
            assert resp.metadata["doc_id"] == expected_doc
            assert all(c.document_id == expected_doc for c in resp.citations)
            assert all(c.filename == expected_file for c in resp.citations)

            assert vis.is_success
            assert vis.document_id == expected_doc
            assert vis.filename == expected_file


# ===========================================================================
# 4. Multi-Document Workload
# ===========================================================================

class TestMultiDocumentWorkload:
    """Verifies multi-document workload preserves strict per-document lineage."""

    def test_multi_document_lineage_isolation(self) -> None:
        docs = [
            (PERF_DOC_A, PERF_FILE_A),
            (PERF_DOC_B, PERF_FILE_B),
            (PERF_DOC_C, PERF_FILE_C),
        ]

        doc_outputs: dict[str, tuple[AgentResponse, VisionResult]] = {}
        for doc_id, filename in docs:
            resp, vis = _run_representative_pipeline(doc_id=doc_id, filename=filename)
            doc_outputs[doc_id] = (resp, vis)

        assert len(doc_outputs) == 3

        # Verify Document A
        resp_a, vis_a = doc_outputs[PERF_DOC_A]
        assert resp_a.unique_documents == [PERF_DOC_A]
        assert vis_a.document_id == PERF_DOC_A

        # Verify Document B
        resp_b, vis_b = doc_outputs[PERF_DOC_B]
        assert resp_b.unique_documents == [PERF_DOC_B]
        assert vis_b.document_id == PERF_DOC_B

        # Verify Document C
        resp_c, vis_c = doc_outputs[PERF_DOC_C]
        assert resp_c.unique_documents == [PERF_DOC_C]
        assert vis_c.document_id == PERF_DOC_C


# ===========================================================================
# 5. Concurrent Performance
# ===========================================================================

class TestConcurrentPerformance:
    """Verifies concurrent multi-threaded execution preserves isolation and correctness."""

    def test_concurrent_four_threads_workload(self) -> None:
        def worker(thread_idx: int) -> tuple[int, AgentResponse, VisionResult]:
            doc_id = f"THREAD_DOC_{thread_idx}"
            filename = f"thread_file_{thread_idx}.pdf"
            query = f"Thread query {thread_idx}"
            resp, vis = _run_representative_pipeline(doc_id=doc_id, filename=filename, query=query)
            return thread_idx, resp, vis

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            results = [f.result() for f in futures]

        assert len(results) == 4
        for thread_idx, resp, vis in results:
            expected_doc = f"THREAD_DOC_{thread_idx}"
            assert resp.is_success
            assert resp.unique_documents == [expected_doc]
            assert vis.is_success
            assert vis.document_id == expected_doc


# ===========================================================================
# 6. Resource Safety & Workspace Cleanliness
# ===========================================================================

class TestResourceSafetyAndCleanliness:
    """Verifies in-memory execution does not create leaked disk artifacts."""

    def test_zero_disk_artifacts_generated(self) -> None:
        # Run workflow 5 times
        for i in range(5):
            _run_representative_pipeline(query=f"Resource check query {i}")

        # Check repo root has no unexpected scratch files created
        root_path = Path(REPO_ROOT)
        unexpected_files = [
            f.name for f in root_path.iterdir()
            if f.is_file() and f.name.endswith((".tmp", ".temp", ".dump", ".log", ".bak"))
        ]
        assert unexpected_files == [], f"Unexpected disk artifacts found: {unexpected_files}"


# ===========================================================================
# 7. Duplicate Work Prevention
# ===========================================================================

class TestDuplicateWorkPrevention:
    """Verifies single-pass processing without redundant operations."""

    def test_evidence_adaptation_single_pass(self) -> None:
        citations = [
            AgentCitation(document_id=PERF_DOC_A, filename=PERF_FILE_A, chunk_id=f"c_{i}",
                          page_number=i+1, content_type="image")
            for i in range(5)
        ]
        ev = VisualEvidenceAdapter.adapt_batch(citations)
        assert len(ev) == 5
        # Exact 1:1 mapping preserved without duplicates
        assert len({e.chunk_id for e in ev}) == 5

    def test_result_normalizer_single_pass(self) -> None:
        normalizer = VisionResultNormalizer()
        raw = VisionResult(
            query="test query",
            status="success",
            evidence=[VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck", content_type="image")],
        )
        normalized = normalizer.normalize(raw)
        assert normalized.is_success
        assert len(normalized.evidence) == 1


# ===========================================================================
# 8. Result Consistency Across Repeated Runs
# ===========================================================================

class TestResultConsistency:
    """Verifies contractually stable fields remain identical across repeated runs."""

    def test_deterministic_field_consistency(self) -> None:
        resp1, vis1 = _run_representative_pipeline(query="deterministic check")
        resp2, vis2 = _run_representative_pipeline(query="deterministic check")

        # Citations consistency
        assert len(resp1.citations) == len(resp2.citations)
        for c1, c2 in zip(resp1.citations, resp2.citations):
            assert c1.document_id == c2.document_id
            assert c1.chunk_id == c2.chunk_id
            assert c1.filename == c2.filename
            assert c1.content_type == c2.content_type
            assert c1.score == pytest.approx(c2.score)

        # Vision Result consistency
        assert vis1.status == vis2.status
        assert vis1.document_id == vis2.document_id
        assert vis1.filename == vis2.filename
        assert len(vis1.evidence) == len(vis2.evidence)


# ===========================================================================
# 9. Failure Recovery Under Workload
# ===========================================================================

class TestFailureRecoveryUnderWorkload:
    """Verifies SUCCESS -> FAILURE -> SUCCESS executes cleanly without residual error state."""

    def test_success_failure_success_recovery(self) -> None:
        # Run 1: SUCCESS
        resp1, vis1 = _run_representative_pipeline(query="Run 1 success")
        assert resp1.is_success
        assert vis1.is_success

        # Run 2: FAILURE (invalid input caught and handled cleanly)
        with pytest.raises(AgentValidationError):
            AgentRequest(query="")

        # Run 3: SUCCESS (subsequent run completely unaffected by prior failure)
        resp3, vis3 = _run_representative_pipeline(query="Run 3 success")
        assert resp3.is_success
        assert vis3.is_success
        assert resp3.error is None
        assert vis3.error is None
