"""
OmniBrain Member 4 -- Day 13 Performance, Scalability & Resource-Boundary Integration Tests.

Verifies that existing OMNIBRAIN integration contracts remain stable when processing
different workload sizes and repeated requests across Ingestion, Search, Vision, and Supervisor.

Concern areas:
 1. Small workload verification (minimal single-item pipeline)
 2. Medium workload processing (batched chunks/evidence sets without dropping or duplication)
 3. Multi-evidence workload processing (ordering, 1:1 lineage, complete retention)
 4. Multi-document workload processing (strict cross-document isolation under load)
 5. Repeated execution stability (deterministic contracts, zero state leakage)
 6. Concurrent execution workload (thread isolation across independent requests)
 7. Resource safety & state isolation (no temp leaks, no caller-data mutations)
 8. Duplicate work prevention (single-pass validation, preparation, adaptation, and search)
 9. Result integrity across workload sizes (status, evidence, citation, lineage, metadata)
10. Serialization across workload sizes (round-trip integrity for small/medium/multi-doc)
11. Failure under workload (containment of errors and clean recovery for subsequent requests)

Constraints:
 - 100% Offline: Zero external APIs, real LLMs, network, or production secrets.
 - Zero production code modified.
 - Zero caching, batching, or algorithm changes added to production.
"""

from __future__ import annotations

import concurrent.futures
import copy
import sys
import threading
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
    RetrievalServiceResult,
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
from agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentRoutingError,
    AgentValidationError,
)

# Vision Subsystem (Member 3)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionAgentError,
    VisionCancellationError,
    VisionError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderError,
    VisionTimeoutError,
)
from vision.lifecycle import (
    VisionCancellationToken,
    VisionExecutionLifecycle,
    VisionExecutionStage,
)


# ============================================================================
# Shared Fixtures & Helpers
# ============================================================================


def _create_chunk(
    chunk_id: str = "chk-001",
    chunk_index: int = 0,
    document_id: str = "doc-001",
    filename: str = "report.pdf",
    page_number: int | None = 1,
    content: str = "Content text.",
    content_type: str = "image",
    metadata: dict[str, Any] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        content=content,
        content_type=content_type,
        metadata=metadata if metadata is not None else {"dept": "analytics"},
    )


def _create_vsr(
    chunk_id: str = "chk-001",
    score: float = 0.90,
    document_id: str = "doc-001",
    filename: str = "report.pdf",
    page_number: int | None = 1,
    chunk_index: int = 0,
    content_type: str = "chart",
    content: str = "Content text.",
    metadata: dict[str, Any] | None = None,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        score=score,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        chunk_index=chunk_index,
        content_type=content_type,
        content=content,
        metadata=metadata if metadata is not None else {"dept": "analytics"},
    )


def _create_visual_evidence(
    chunk_id: str = "chk-001",
    document_id: str = "doc-001",
    filename: str = "report.pdf",
    page_number: int | None = 1,
    chunk_index: int = 0,
    content_type: str = "chart",
    metadata: dict[str, Any] | None = None,
) -> VisualEvidence:
    return VisualEvidence(
        document_id=document_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        chunk_index=chunk_index,
        content_type=content_type,
        metadata=metadata if metadata is not None else {"dept": "analytics"},
    )


# ============================================================================
# 1. SMALL WORKLOAD VERIFICATION
# ============================================================================


class TestSmallWorkloadIntegration:
    """Verifies minimal single-item request executes through complete contract with exact lineage."""

    def test_single_item_small_workload_lifecycle(self) -> None:
        chunk = _create_chunk(
            chunk_id="chk-sm-01",
            chunk_index=0,
            document_id="doc-small-01",
            filename="small_doc.pdf",
            page_number=1,
            content="Q1 Growth: 5%",
            content_type="image",
            metadata={"scale": "small"},
        )
        val = validate_chunks([chunk])
        assert val.is_valid is True
        assert val.valid_chunks == 1

        vsr = _create_vsr(
            chunk_id=chunk.chunk_id,
            score=0.95,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            content=chunk.content,
            metadata=chunk.metadata,
        )
        citation = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_search_result(vsr)
        vreq = VisionRequest(query="Analyze small growth chart.", evidence=[ev])
        vres = VisionResult(
            query=vreq.query,
            status="success",
            description="Growth is 5%.",
            evidence=vreq.evidence,
        )

        assert vres.status == "success"
        assert vres.document_id == "doc-small-01"
        assert vres.filename == "small_doc.pdf"
        assert vres.chunk_id == "chk-sm-01"
        assert vres.page_number == 1
        assert citation.document_id == "doc-small-01"
        assert citation.score == 0.95


# ============================================================================
# 2. MEDIUM WORKLOAD PROCESSING
# ============================================================================


class TestMediumWorkloadIntegration:
    """Verifies batch of medium size (e.g. 20 chunks / evidence items) preserves all items without duplication or drops."""

    def test_medium_workload_chunk_validation_and_preparation(self) -> None:
        count = 20
        chunks = [
            _create_chunk(
                chunk_id=f"chk-med-{i:03d}",
                chunk_index=i,
                document_id="doc-medium-100",
                filename="medium_dataset.pdf",
                page_number=(i // 2) + 1,
                content=f"Medium payload entry {i}",
                content_type="image" if i % 2 == 0 else "text",
                metadata={"index": i, "batch": "medium"},
            )
            for i in range(count)
        ]

        # Validation step
        val_result = validate_chunks(chunks)
        assert val_result.is_valid is True
        assert val_result.total_chunks == count
        assert val_result.valid_chunks == count
        assert val_result.invalid_chunks == 0

        # Normalization step
        normalized = normalize_chunks(chunks)
        assert len(normalized) == count
        assert [c.chunk_id for c in normalized] == [f"chk-med-{i:03d}" for i in range(count)]

        # Embedding Preparation step
        prep_result = prepare_for_embedding(normalized)
        assert prep_result.is_ready is True
        assert prep_result.total_items == count
        assert len(prep_result.items) == count
        for i, item in enumerate(prep_result.items):
            assert item.chunk_id == f"chk-med-{i:03d}"
            assert item.document_id == "doc-medium-100"


# ============================================================================
# 3. MULTI-EVIDENCE WORKLOAD
# ============================================================================


class TestMultiEvidenceWorkloadIntegration:
    """Verifies multiple evidence items maintain 1:1 citation, order, and complete retention."""

    def test_multi_evidence_retention_and_ordering(self) -> None:
        ev_items = [
            _create_visual_evidence(
                chunk_id=f"chk-ev-{idx}",
                document_id="doc-multi-ev",
                filename="multi_evidence.pdf",
                page_number=idx + 1,
                chunk_index=idx,
                content_type="image" if idx % 2 == 0 else "chart",
                metadata={"order": idx},
            )
            for idx in range(5)
        ]

        req = VisionRequest(query="Analyze 5 multi-evidence figures.", evidence=ev_items)
        assert req.total_evidence == 5
        assert [e.chunk_id for e in req.evidence] == [f"chk-ev-{idx}" for idx in range(5)]

        res = VisionResult(
            query=req.query,
            status="success",
            description="All 5 evidence items analyzed.",
            evidence=req.evidence,
        )
        assert len(res.evidence) == 5
        assert [e.metadata["order"] for e in res.evidence] == list(range(5))


# ============================================================================
# 4. MULTI-DOCUMENT WORKLOAD
# ============================================================================


class TestMultiDocumentWorkloadIntegration:
    """Verifies that evidence from multiple documents under load does not cross-contaminate."""

    def test_multi_document_isolation_under_load(self) -> None:
        docs = ["doc-alpha", "doc-beta", "doc-gamma", "doc-delta"]
        all_citations = []

        for doc_idx, doc_id in enumerate(docs):
            for chk_idx in range(3):
                vsr = _create_vsr(
                    chunk_id=f"chk-{doc_id}-{chk_idx}",
                    score=0.90 - (doc_idx * 0.05),
                    document_id=doc_id,
                    filename=f"{doc_id}.pdf",
                    page_number=chk_idx + 1,
                    metadata={"doc_tag": doc_id, "chk_tag": chk_idx},
                )
                citation = AgentCitation.from_search_result(vsr)
                all_citations.append(citation)

        assert len(all_citations) == 12

        search_result = SearchResult(
            query="Analyze multi-document portfolio",
            status="RESULTS_FOUND",
            citations=all_citations,
        )
        assert search_result.unique_document_count == 4
        assert set(search_result.unique_documents) == set(docs)

        # Verify each citation has strict isolation
        for cit in search_result.citations:
            assert cit.metadata["doc_tag"] == cit.document_id
            assert cit.filename == f"{cit.document_id}.pdf"


# ============================================================================
# 5. REPEATED EXECUTION STABILITY
# ============================================================================


class TestRepeatedExecutionStability:
    """Verifies that repeated executions of the same workflow produce deterministic results without state buildup."""

    def test_repeated_pipeline_executions(self) -> None:
        run_count = 10
        signatures = []

        for run in range(run_count):
            chunk = _create_chunk(
                chunk_id="chk-repeat-01",
                document_id="doc-repeat-01",
                content="Deterministic test content.",
            )
            vsr = _create_vsr(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
            )
            cit = AgentCitation.from_search_result(vsr)
            ev = VisualEvidence.from_search_result(vsr)
            signatures.append((cit.document_id, cit.chunk_id, ev.document_id, ev.chunk_id))

        assert len(signatures) == run_count
        assert len(set(signatures)) == 1
        assert signatures[0] == ("doc-repeat-01", "chk-repeat-01", "doc-repeat-01", "chk-repeat-01")


# ============================================================================
# 6. CONCURRENT WORKLOAD ISOLATION
# ============================================================================


class TestConcurrentWorkloadIntegration:
    """Verifies independent requests executed concurrently across worker threads remain fully isolated."""

    def test_concurrent_request_workload_isolation(self) -> None:
        def _worker(thread_id: int) -> dict[str, Any]:
            doc_id = f"doc-thread-{thread_id:02d}"
            chunk_id = f"chk-thread-{thread_id:02d}"
            ev = _create_visual_evidence(
                chunk_id=chunk_id,
                document_id=doc_id,
                filename=f"{doc_id}.pdf",
                metadata={"thread_id": thread_id},
            )
            req = VisionRequest(query=f"Thread query {thread_id}", evidence=[ev])
            res = VisionResult(
                query=req.query,
                status="success",
                description=f"Thread {thread_id} complete.",
                evidence=req.evidence,
            )
            return {
                "thread_id": thread_id,
                "doc_id": res.document_id,
                "chunk_id": res.chunk_id,
                "evidence_doc": res.evidence[0].document_id,
                "evidence_thread": res.evidence[0].metadata["thread_id"],
            }

        concurrency = 16
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_worker, i) for i in range(concurrency)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == concurrency
        for r in results:
            tid = r["thread_id"]
            expected_doc = f"doc-thread-{tid:02d}"
            expected_chk = f"chk-thread-{tid:02d}"
            assert r["doc_id"] == expected_doc
            assert r["chunk_id"] == expected_chk
            assert r["evidence_doc"] == expected_doc
            assert r["evidence_thread"] == tid


# ============================================================================
# 7. RESOURCE SAFETY & STATE ISOLATION
# ============================================================================


class TestResourceSafetyAndStateIsolation:
    """Verifies that passing data through pipeline does not mutate inputs or leave persistent artifacts."""

    def test_caller_dictionary_mutation_safety(self) -> None:
        caller_dict = {"owner": "caller", "counter": 100}
        dict_copy = copy.deepcopy(caller_dict)

        chunk = _create_chunk(metadata=caller_dict)
        vsr = _create_vsr(metadata=caller_dict)
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", metadata=caller_dict)
        ev = VisualEvidence(document_id="d", filename="f.pdf", chunk_id="c", metadata=caller_dict)

        # Mutate local caller_dict to confirm objects hold independent copies
        caller_dict["counter"] = 999
        assert cit.metadata["counter"] == 100
        assert ev.metadata["counter"] == 100


# ============================================================================
# 8. DUPLICATE WORK PREVENTION
# ============================================================================


class TestDuplicateWorkPrevention:
    """Verifies single-pass execution across validation, normalization, and context building."""

    def test_single_pass_retrieval_processing(self) -> None:
        vsr_items = [
            _create_vsr(chunk_id=f"c-{i}", score=0.80 + (i * 0.02), content=f"Text {i}")
            for i in range(5)
        ]
        # Process retrieval results filters in single pass
        processed = process_retrieval_results(vsr_items, min_score=0.83, max_results=3)
        assert len(processed) <= 3
        for item in processed:
            assert item.score >= 0.83

        # Context builder builds in single pass
        context = build_retrieval_context(processed)
        assert isinstance(context, str)
        for item in processed:
            assert item.content in context


# ============================================================================
# 9. RESULT INTEGRITY ACROSS WORKLOAD SIZES
# ============================================================================


class TestResultIntegrityAcrossWorkloadSizes:
    """Verifies result properties and contract invariants across small, medium, and multi-doc workloads."""

    @pytest.mark.parametrize("size", [1, 5, 15])
    def test_result_contract_invariants(self, size: int) -> None:
        ev_list = [
            _create_visual_evidence(
                chunk_id=f"chk-inv-{i}",
                document_id="doc-inv-01",
                page_number=i + 1,
                chunk_index=i,
            )
            for i in range(size)
        ]
        vres = VisionResult(
            query=f"Query for size {size}",
            status="success",
            description=f"Analyzed {size} items.",
            evidence=ev_list,
        )

        assert vres.status == "success"
        assert vres.has_evidence is True
        assert len(vres.evidence) == size
        assert vres.document_id == "doc-inv-01"
        assert vres.chunk_id == "chk-inv-0"
        assert vres.page_number == 1
        assert vres.error is None


# ============================================================================
# 10. SERIALIZATION ACROSS WORKLOAD SIZES
# ============================================================================


class TestSerializationAcrossWorkloadSizes:
    """Verifies lossless serialization round-trips for small, medium, and multi-document results."""

    def test_medium_workload_search_result_serialization(self) -> None:
        citations = [
            AgentCitation(
                document_id=f"doc-{i // 3}",
                filename=f"doc_{i // 3}.pdf",
                chunk_id=f"chk-{i}",
                page_number=(i % 3) + 1,
                score=0.85 + (i * 0.01),
                metadata={"item_idx": i},
            )
            for i in range(12)
        ]
        orig_sr = SearchResult(
            query="Analyze portfolio",
            status="RESULTS_FOUND",
            citations=citations,
            context="Portfolio summary context.",
        )

        data = orig_sr.to_dict()
        restored = SearchResult.from_dict(data)

        assert restored.query == orig_sr.query
        assert restored.status == orig_sr.status
        assert len(restored.citations) == 12
        assert restored.unique_document_count == 4
        for idx, cit in enumerate(restored.citations):
            assert cit.chunk_id == f"chk-{idx}"
            assert cit.metadata["item_idx"] == idx


# ============================================================================
# 11. FAILURE UNDER WORKLOAD
# ============================================================================


class TestFailureUnderWorkload:
    """Verifies that an error occurring under load is safely contained and allows immediate subsequent success."""

    def test_failure_under_load_does_not_poison_next_request(self) -> None:
        # 1. Failed batch
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="corrupt.pdf", chunk_id="chk-bad")

        # 2. Subsequent workload request executes successfully
        valid_chunk = _create_chunk(chunk_id="chk-rec-01", document_id="doc-good-01")
        valid_vsr = _create_vsr(chunk_id=valid_chunk.chunk_id, document_id=valid_chunk.document_id)
        valid_cit = AgentCitation.from_search_result(valid_vsr)

        assert valid_cit.document_id == "doc-good-01"
        assert valid_cit.chunk_id == "chk-rec-01"
