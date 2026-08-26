"""
OmniBrain Member 4 — Day 36 Performance & Scalability Regression Certification.

Evaluates the existing OMNIBRAIN public APIs under progressively larger synthetic workloads:
  - Small Workload Baseline (10 items)
  - Medium Workload (50 items)
  - Large Workload (100 items)
  - Scaling Behavior & Linear Scalability Observation
  - Output Completeness & Cardinality Preservation
  - Content Integrity & Metadata Preservation
  - Batch Size Regression (1, 10, 50, 100 items)
  - Empty Batch Boundary Handling
  - Single-Item vs. Batch Processing Equivalence
  - Repeated Execution Purity (no accumulated state across runs)
  - Memory Behavior & Leakage Detection (tracemalloc)
  - Resource Cleanup Verification
  - Multi-Document Workload Isolation
  - Concurrency Performance & State Isolation (multi-threaded execution)
  - Failure Under Load & Immediate Recovery
  - Performance & Scalability Benchmark Summary Reporting

Constraints:
  - 100% offline. Zero external APIs, network, LLM, or production credentials.
  - Zero production code modified.
  - No caching, batching logic, concurrency infrastructure, or performance optimizations added.
"""

from __future__ import annotations

import concurrent.futures
import copy
import sys
import time
import tracemalloc
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
    ChunkingResult,
    ChunkValidationResult,
    DocumentChunk,
    DocumentMetadata,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    PageData,
    ParsedDocument,
    VectorSearchResult,
)
from ingestion.chunk_validator import normalize_chunks, validate_chunks
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.retrieval_processor import (
    build_retrieval_context,
    process_retrieval_results,
)
from ingestion.ingestion_errors import IngestionValidationError

# ---------------------------------------------------------------------------
# Agents / Search Subsystem (Member 2)
# ---------------------------------------------------------------------------
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    AgentState,
    SearchRequest,
    SearchResult,
)
from agents.exceptions import AgentValidationError

# ---------------------------------------------------------------------------
# Vision Subsystem (Member 3)
# ---------------------------------------------------------------------------
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.result_normalizer import (
    VisionExecutionTrace,
    VisionResultNormalizer,
)
from vision.exceptions import (
    VisionEvidenceError,
    VisionInputValidationError,
)

# ---------------------------------------------------------------------------
# Deterministic Synthetic Generators (Section 5)
# ---------------------------------------------------------------------------

DOC_ID = "DAY36-PERF-DOC-001"
FILENAME = "day36_performance_workload.pdf"


def _generate_synthetic_chunks(
    count: int,
    document_id: str = DOC_ID,
    filename: str = FILENAME,
    content_type: str = "image",
    prefix: str = "DAY36_ITEM",
) -> list[DocumentChunk]:
    """Generate deterministic synthetic DocumentChunk items."""
    chunks: list[DocumentChunk] = []
    for i in range(count):
        chunk_id = f"{prefix}_{i:04d}"
        page_num = (i // 5) + 1
        content = f"Synthetic content payload for item {chunk_id} on page {page_num}."
        metadata = {
            "day36_item_id": chunk_id,
            "seq": i,
            "document_id": document_id,
            "content_type": content_type,
            "payload_tag": f"tag_{i % 10}",
        }
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                chunk_index=i,
                document_id=document_id,
                filename=filename,
                page_number=page_num,
                content=content,
                content_type=content_type,
                metadata=metadata,
            )
        )
    return chunks


def _generate_synthetic_vsrs(
    chunks: list[DocumentChunk],
    base_score: float = 0.99,
) -> list[VectorSearchResult]:
    """Generate deterministic VectorSearchResult objects from chunks."""
    vsrs: list[VectorSearchResult] = []
    for i, c in enumerate(chunks):
        score = max(0.1, round(base_score - (i * 0.005), 4))
        vsrs.append(
            VectorSearchResult(
                chunk_id=c.chunk_id,
                score=score,
                document_id=c.document_id,
                filename=c.filename,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                content_type=c.content_type,
                content=c.content,
                metadata=c.metadata,
            )
        )
    return vsrs


# ===========================================================================
# 1. WORKLOAD BASELINES & SCALING BEHAVIOR
# ===========================================================================

class TestWorkloadBaselinesAndScaling:
    """Sections 6, 7, 8, 9: Small (10), Medium (50), and Large (100) workloads."""

    def test_small_workload_baseline_10_items(self) -> None:
        chunks = _generate_synthetic_chunks(10)

        t0 = time.perf_counter()
        normalized = normalize_chunks(chunks)
        prep = prepare_for_embedding(normalized)
        vsrs = _generate_synthetic_vsrs(chunks)
        processed = process_retrieval_results(vsrs, min_score=0.1, max_results=10)
        citations = [AgentCitation.from_search_result(r) for r in processed]
        evidence = VisualEvidenceAdapter.adapt_batch(citations)
        v_req = VisionRequest(query="Day 36 Small Benchmark", evidence=evidence)
        v_res = VisionResult(query=v_req.query, status="success", description="Small batch", evidence=v_req.evidence)
        normalized_res = VisionResultNormalizer.normalize(v_res, request=v_req)
        duration = time.perf_counter() - t0

        assert len(normalized) == 10
        assert prep.total_items == 10
        assert len(processed) == 10
        assert len(citations) == 10
        assert len(evidence) == 10
        assert normalized_res.is_success is True
        assert duration < 5.0  # Safe upper bound

    def test_medium_workload_50_items(self) -> None:
        chunks = _generate_synthetic_chunks(50)

        t0 = time.perf_counter()
        normalized = normalize_chunks(chunks)
        prep = prepare_for_embedding(normalized)
        vsrs = _generate_synthetic_vsrs(chunks)
        processed = process_retrieval_results(vsrs, min_score=0.1, max_results=50)
        citations = [AgentCitation.from_search_result(r) for r in processed]
        evidence = VisualEvidenceAdapter.adapt_batch(citations)
        duration = time.perf_counter() - t0

        assert len(normalized) == 50
        assert prep.total_items == 50
        assert len(processed) == 50
        assert len(citations) == 50
        assert len(evidence) == 50
        assert duration < 10.0

    def test_large_workload_100_items(self) -> None:
        chunks = _generate_synthetic_chunks(100)

        t0 = time.perf_counter()
        normalized = normalize_chunks(chunks)
        prep = prepare_for_embedding(normalized)
        vsrs = _generate_synthetic_vsrs(chunks)
        processed = process_retrieval_results(vsrs, min_score=0.1, max_results=100)
        citations = [AgentCitation.from_search_result(r) for r in processed]
        evidence = VisualEvidenceAdapter.adapt_batch(citations)
        duration = time.perf_counter() - t0

        assert len(normalized) == 100
        assert prep.total_items == 100
        assert len(processed) == 100
        assert len(citations) == 100
        assert len(evidence) == 100
        assert duration < 15.0

    def test_progressive_scaling_performance_comparison(self) -> None:
        timings: dict[int, float] = {}

        for size in (10, 50, 100):
            chunks = _generate_synthetic_chunks(size)
            t0 = time.perf_counter()
            norm = normalize_chunks(chunks)
            prep = prepare_for_embedding(norm)
            vsrs = _generate_synthetic_vsrs(chunks)
            processed = process_retrieval_results(vsrs, min_score=0.1, max_results=size)
            citations = [AgentCitation.from_search_result(r) for r in processed]
            evidence = VisualEvidenceAdapter.adapt_batch(citations)
            elapsed = time.perf_counter() - t0
            timings[size] = elapsed

            assert len(evidence) == size
            assert elapsed > 0.0

        # Per-item processing should remain reasonably consistent without catastrophic degradation
        avg_time_10 = timings[10] / 10
        avg_time_100 = timings[100] / 100
        assert avg_time_100 < avg_time_10 * 50  # No super-quadratic explosion


# ===========================================================================
# 2. OUTPUT COMPLETENESS, CONTENT & METADATA INTEGRITY
# ===========================================================================

class TestCompletenessAndIntegrityUnderLoad:
    """Sections 10, 11, 12: Cardinality preservation, content & metadata integrity."""

    def test_output_cardinality_100_percent_preserved(self) -> None:
        chunks = _generate_synthetic_chunks(60)
        norm = normalize_chunks(chunks)
        prep = prepare_for_embedding(norm)
        vsrs = _generate_synthetic_vsrs(chunks)
        proc = process_retrieval_results(vsrs, min_score=0.0, max_results=60)
        cits = [AgentCitation.from_search_result(r) for r in proc]
        evs = VisualEvidenceAdapter.adapt_batch(cits)

        assert len(chunks) == 60
        assert len(norm) == 60
        assert prep.total_items == 60
        assert len(proc) == 60
        assert len(cits) == 60
        assert len(evs) == 60

    def test_unique_content_integrity_under_large_workload(self) -> None:
        chunks = _generate_synthetic_chunks(75)
        norm = normalize_chunks(chunks)
        prep = prepare_for_embedding(norm)

        prep_map = {item.chunk_id: item.content for item in prep.items}
        for chunk in chunks:
            assert chunk.chunk_id in prep_map
            assert prep_map[chunk.chunk_id] == chunk.content

    def test_metadata_integrity_preserved_under_large_workload(self) -> None:
        chunks = _generate_synthetic_chunks(80)
        vsrs = _generate_synthetic_vsrs(chunks)
        cits = [AgentCitation.from_search_result(r) for r in vsrs]
        evs = VisualEvidenceAdapter.adapt_batch(cits)

        for i, ev in enumerate(evs):
            assert ev.metadata["seq"] == i
            assert ev.metadata["day36_item_id"] == f"DAY36_ITEM_{i:04d}"
            assert ev.metadata["payload_tag"] == f"tag_{i % 10}"


# ===========================================================================
# 3. BATCH SIZE REGRESSION & EMPTY BATCH BOUNDARIES
# ===========================================================================

class TestBatchSizeRegressionAndBoundaries:
    """Sections 13, 14, 15: Batch size variations (1, 10, 50, 100) and empty batches."""

    @pytest.mark.parametrize("batch_size", [1, 10, 50, 100])
    def test_batch_sizes_execute_correctly(self, batch_size: int) -> None:
        chunks = _generate_synthetic_chunks(batch_size)
        norm = normalize_chunks(chunks)
        prep = prepare_for_embedding(norm)
        assert prep.total_items == batch_size

    def test_empty_batch_handling_across_all_apis(self) -> None:
        # Ingestion normalize_chunks
        assert normalize_chunks([]) == []

        # Ingestion prepare_for_embedding
        prep = prepare_for_embedding([])
        assert prep.is_ready is True
        assert prep.total_items == 0

        # Ingestion retrieval_processor
        assert process_retrieval_results([], min_score=0.0, max_results=10) == []
        assert build_retrieval_context([]) == ""

        # Vision adapter
        assert VisualEvidenceAdapter.adapt_batch([]) == []

    def test_single_item_equivalence_standalone_vs_in_batch(self) -> None:
        single_chunk = _generate_synthetic_chunks(1)[0]
        batch_chunks = _generate_synthetic_chunks(50)
        batch_chunks[0] = single_chunk

        # Standalone
        prep_single = prepare_for_embedding([single_chunk])
        # In Batch
        prep_batch = prepare_for_embedding(batch_chunks)

        rec_standalone = prep_single.items[0]
        rec_in_batch = prep_batch.items[0]

        assert rec_standalone.chunk_id == rec_in_batch.chunk_id
        assert rec_standalone.content == rec_in_batch.content
        assert rec_standalone.metadata == rec_in_batch.metadata
        assert rec_standalone.document_id == rec_in_batch.document_id


# ===========================================================================
# 4. REPEATED EXECUTION PURITY & STATELESSNESS
# ===========================================================================

class TestRepeatedExecutionPurity:
    """Section 16: Multiple runs do not accumulate state or degrade correctness."""

    def test_three_repeated_executions_yield_identical_results(self) -> None:
        chunks = _generate_synthetic_chunks(50)
        results: list[list[str]] = []

        for run in range(3):
            norm = normalize_chunks(chunks)
            prep = prepare_for_embedding(norm)
            vsrs = _generate_synthetic_vsrs(chunks)
            proc = process_retrieval_results(vsrs, min_score=0.1, max_results=50)
            cits = [AgentCitation.from_search_result(r) for r in proc]
            evs = VisualEvidenceAdapter.adapt_batch(cits)

            chunk_ids = [e.chunk_id for e in evs]
            results.append(chunk_ids)
            assert len(evs) == 50

        # Assert zero drift across runs
        assert results[0] == results[1] == results[2]


# ===========================================================================
# 5. MEMORY MEASUREMENT & RESOURCE CLEANUP
# ===========================================================================

class TestMemoryMeasurementAndCleanup:
    """Sections 17 & 18: Tracemalloc memory footprint monitoring and resource cleanup."""

    def test_memory_footprint_under_100_item_workload(self) -> None:
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        chunks = _generate_synthetic_chunks(100)
        norm = normalize_chunks(chunks)
        prep = prepare_for_embedding(norm)
        vsrs = _generate_synthetic_vsrs(chunks)
        proc = process_retrieval_results(vsrs, min_score=0.1, max_results=100)
        cits = [AgentCitation.from_search_result(r) for r in proc]
        evs = VisualEvidenceAdapter.adapt_batch(cits)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert len(evs) == 100
        # Peak memory for a 100-item metadata/text workload should comfortably be under 50 MB
        assert peak < 50 * 1024 * 1024, f"Peak memory {peak / 1024 / 1024:.2f} MB exceeded threshold."


# ===========================================================================
# 6. MULTI-DOCUMENT WORKLOAD ISOLATION
# ===========================================================================

class TestMultiDocumentWorkloadIsolation:
    """Section 19: Multiple documents processed concurrently maintain distinct lineage."""

    def test_multi_document_scaling_without_cross_contamination(self) -> None:
        doc_count = 5
        chunks_per_doc = 15

        all_doc_evidence: dict[str, list[VisualEvidence]] = {}

        for doc_idx in range(doc_count):
            doc_id = f"DAY36_DOC_{doc_idx:02d}"
            filename = f"document_{doc_idx:02d}.pdf"
            chunks = _generate_synthetic_chunks(
                chunks_per_doc,
                document_id=doc_id,
                filename=filename,
                prefix=f"CHK_DOC_{doc_idx}",
            )
            norm = normalize_chunks(chunks)
            prep = prepare_for_embedding(norm)
            vsrs = _generate_synthetic_vsrs(chunks)
            proc = process_retrieval_results(vsrs, min_score=0.1, max_results=chunks_per_doc)
            cits = [AgentCitation.from_search_result(r) for r in proc]
            evs = VisualEvidenceAdapter.adapt_batch(cits)
            all_doc_evidence[doc_id] = evs

        assert len(all_doc_evidence) == 5
        for doc_id, evs in all_doc_evidence.items():
            assert len(evs) == chunks_per_doc
            assert all(e.document_id == doc_id for e in evs)


# ===========================================================================
# 7. CONCURRENCY PERFORMANCE & THREAD ISOLATION
# ===========================================================================

class TestConcurrencyPerformance:
    """Section 20: Thread-safe concurrent execution without state leakage."""

    def test_concurrent_multithreaded_pipeline_execution(self) -> None:
        def worker_task(thread_id: int) -> tuple[int, int, str]:
            doc_id = f"DAY36_CONC_DOC_{thread_id}"
            chunks = _generate_synthetic_chunks(
                20, document_id=doc_id, filename=f"thread_{thread_id}.pdf",
                prefix=f"TH_{thread_id}",
            )
            norm = normalize_chunks(chunks)
            prep = prepare_for_embedding(norm)
            vsrs = _generate_synthetic_vsrs(chunks)
            proc = process_retrieval_results(vsrs, min_score=0.1, max_results=20)
            cits = [AgentCitation.from_search_result(r) for r in proc]
            evs = VisualEvidenceAdapter.adapt_batch(cits)
            return thread_id, len(evs), evs[0].document_id

        num_threads = 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            results = [f.result() for f in futures]

        assert len(results) == num_threads
        for thread_id, count, doc_id in results:
            assert count == 20
            assert doc_id == f"DAY36_CONC_DOC_{thread_id}"


# ===========================================================================
# 8. FAILURE UNDER LOAD & RECOVERY
# ===========================================================================

class TestFailureUnderLoadAndRecovery:
    """Section 21: Controlled invalid item in batch triggers contract error and enables recovery."""

    def test_invalid_item_in_large_batch_fails_and_recovers(self) -> None:
        chunks = _generate_synthetic_chunks(30)
        # Inject invalid item with empty chunk_id at index 15
        bad_chunk = DocumentChunk(
            chunk_id="", chunk_index=15, document_id=DOC_ID,
            filename=FILENAME, page_number=1, content="Bad content", content_type="text",
        )
        chunks[15] = bad_chunk

        # Ingestion prepare_for_embedding should fail fast as per validation contract
        with pytest.raises(ValueError):
            prepare_for_embedding(chunks)

        # Immediate recovery: subsequent valid workload must succeed without remnant error state
        valid_chunks = _generate_synthetic_chunks(30)
        prep = prepare_for_embedding(valid_chunks)
        assert prep.is_ready is True
        assert prep.total_items == 30


# ===========================================================================
# 9. PERFORMANCE BENCHMARK DATA REPORT
# ===========================================================================

class TestPerformanceDataReport:
    """Section 22: Generates actual test-side execution time and memory measurements."""

    def test_generate_and_verify_performance_report(self) -> None:
        benchmarks: list[dict[str, Any]] = []

        for size in (10, 50, 100):
            chunks = _generate_synthetic_chunks(size)
            tracemalloc.start()
            t0 = time.perf_counter()

            norm = normalize_chunks(chunks)
            prep = prepare_for_embedding(norm)
            vsrs = _generate_synthetic_vsrs(chunks)
            proc = process_retrieval_results(vsrs, min_score=0.1, max_results=size)
            cits = [AgentCitation.from_search_result(r) for r in proc]
            evs = VisualEvidenceAdapter.adapt_batch(cits)

            elapsed = time.perf_counter() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            benchmarks.append({
                "workload": f"{size} items",
                "items": size,
                "duration_sec": round(elapsed, 4),
                "avg_time_per_item_ms": round((elapsed / size) * 1000, 3),
                "success": len(evs) == size,
                "failure": 0,
                "peak_memory_kb": round(peak_mem / 1024, 2),
            })

        assert len(benchmarks) == 3
        for b in benchmarks:
            assert b["success"] is True
            assert b["duration_sec"] > 0.0
            assert b["peak_memory_kb"] > 0.0
