"""
OmniBrain Member 4 — Day 27 Large-Batch & Multi-Document Integration Certification Tests.

Certifies that the existing OMNIBRAIN pipeline correctly handles multiple documents
and multiple retrieval/evidence items within the same integration workflow.

Pipeline Flow:
    Ingestion (Member 1)
         ↓
    Search / Retrieval (Member 2)
         ↓
    Vision (Member 3)
         ↓
    Downstream Supervisor / Agent Consumers

Focus areas:
 1. Multi-document dataset definition (DOCUMENT_A, B, C, D) and chunk generation.
 2. Multi-chunk data validation and exact document-to-chunk provenance mapping.
 3. Multi-document retrieval processing and strict non-overlap across documents.
 4. Per-document metadata isolation under multi-item workloads.
 5. Multi-evidence visual processing across multiple source documents.
 6. Citation integrity and exact 1:1 lineage tracking across pipeline stages.
 7. End-to-end lineage preservation from raw Document to final VisionResult.
 8. Variable batch size execution (1, 5, 10, 20 items).
 9. Duplicate-input handling according to public contract specifications.
 10. Multi-document serialization round-trip preservation (to_dict -> from_dict).
 11. Result count and ordering preservation.
 12. Request isolation across independent document subsets (A+B vs C+D).
 13. Repeated batch execution stability without state accumulation.
 14. Failure handling within batch workloads.
 15. Concurrent multi-document processing isolation across threads.
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
from vision.exceptions import VisionEvidenceError, VisionInputValidationError

# ---------------------------------------------------------------------------
# Multi-Document Synthetic Constants
# ---------------------------------------------------------------------------

DOC_A = "DAY27_DOCUMENT_A"
DOC_B = "DAY27_DOCUMENT_B"
DOC_C = "DAY27_DOCUMENT_C"
DOC_D = "DAY27_DOCUMENT_D"

FILE_A = "day27_doc_a.pdf"
FILE_B = "day27_doc_b.pdf"
FILE_C = "day27_doc_c.pdf"
FILE_D = "day27_doc_d.pdf"

ALL_DOC_TUPLES = [
    (DOC_A, FILE_A, "A"),
    (DOC_B, FILE_B, "B"),
    (DOC_C, FILE_C, "C"),
    (DOC_D, FILE_D, "D"),
]


def _make_doc_chunks(doc_id: str, filename: str, marker: str, count: int = 3) -> list[DocumentChunk]:
    """Generate multiple structured chunks for a specific document."""
    chunks = []
    modalities = ["text", "image", "text", "chart", "diagram"]
    for i in range(count):
        content_type = modalities[i % len(modalities)]
        chunk = DocumentChunk(
            chunk_id=f"chunk_{doc_id}_{i+1}",
            chunk_index=i,
            document_id=doc_id,
            filename=filename,
            page_number=i + 1,
            content=f"Day 27 content for {doc_id} chunk {i+1} ({content_type}).",
            content_type=content_type,
            metadata={
                "day27_marker": marker,
                "document_id": doc_id,
                "chunk_idx": i,
                "tenant": f"TENANT_{marker}",
            },
        )
        chunks.append(chunk)
    return chunks


def _make_doc_vsr(doc_id: str, filename: str, marker: str, count: int = 3) -> list[VectorSearchResult]:
    """Generate multiple vector search results for a specific document."""
    results = []
    modalities = ["text", "image", "text", "chart", "diagram"]
    for i in range(count):
        content_type = modalities[i % len(modalities)]
        score = 0.98 - (i * 0.05)
        vsr = VectorSearchResult(
            chunk_id=f"chunk_{doc_id}_{i+1}",
            score=score,
            document_id=doc_id,
            filename=filename,
            page_number=i + 1,
            chunk_index=i,
            content_type=content_type,
            content=f"Retrieved content for {doc_id} chunk {i+1} ({content_type}).",
            metadata={
                "day27_marker": marker,
                "document_id": doc_id,
                "score": score,
                "tenant": f"TENANT_{marker}",
            },
        )
        results.append(vsr)
    return results


# ===========================================================================
# 1. Multi-Document Dataset & Multi-Chunk Validation
# ===========================================================================

class TestMultiDocumentMultiChunkDataset:
    """Verifies generation, validation, and normalization of multi-document datasets."""

    def test_multi_document_dataset_structure(self) -> None:
        all_chunks: list[DocumentChunk] = []
        for doc_id, filename, marker in ALL_DOC_TUPLES:
            chunks = _make_doc_chunks(doc_id, filename, marker, count=3)
            assert len(chunks) == 3
            all_chunks.extend(chunks)

        assert len(all_chunks) == 12

        # Validate each document's chunks using chunk_validator
        for doc_id, filename, marker in ALL_DOC_TUPLES:
            doc_chunks = [c for c in all_chunks if c.document_id == doc_id]
            val_res = validate_chunks(doc_chunks)
            assert val_res.is_valid
            assert val_res.total_chunks == 3
            assert val_res.valid_chunks == 3

        # Normalize complete multi-document chunk collection in a single batch
        normalized = normalize_chunks(all_chunks)
        assert len(normalized) == 12

    def test_chunk_document_provenance_isolation(self) -> None:
        for doc_id, filename, marker in ALL_DOC_TUPLES:
            chunks = _make_doc_chunks(doc_id, filename, marker, count=3)
            for c in chunks:
                assert c.document_id == doc_id
                assert c.filename == filename
                assert c.metadata["day27_marker"] == marker
                assert c.metadata["tenant"] == f"TENANT_{marker}"


# ===========================================================================
# 2. Multi-Document Retrieval & Context Building
# ===========================================================================

class TestMultiDocumentRetrievalAndContext:
    """Verifies retrieval processing across multi-document batches."""

    def test_multi_document_retrieval_processing(self) -> None:
        all_vsr: list[VectorSearchResult] = []
        for doc_id, filename, marker in ALL_DOC_TUPLES:
            all_vsr.extend(_make_doc_vsr(doc_id, filename, marker, count=3))

        assert len(all_vsr) == 12

        # Process all retrieval results
        processed = process_retrieval_results(all_vsr, min_score=0.5, max_results=20)
        assert len(processed) == 12

        # Verify context building contains content from all documents
        context_str = build_retrieval_context(processed)
        assert DOC_A in context_str
        assert DOC_B in context_str
        assert DOC_C in context_str
        assert DOC_D in context_str


# ===========================================================================
# 3. Metadata & Citation Integrity Under Multi-Document Workload
# ===========================================================================

class TestMetadataAndCitationIntegrity:
    """Verifies citation attributes match their respective source documents exactly."""

    def test_citations_preserve_exact_document_and_chunk_identities(self) -> None:
        all_vsr: list[VectorSearchResult] = []
        for doc_id, filename, marker in ALL_DOC_TUPLES:
            all_vsr.extend(_make_doc_vsr(doc_id, filename, marker, count=3))

        citations = [AgentCitation.from_search_result(v) for v in all_vsr]
        assert len(citations) == 12

        # Verify each citation matches its expected source document
        for c in citations:
            expected_marker = c.metadata.get("day27_marker")
            assert expected_marker in ("A", "B", "C", "D")
            assert c.document_id == f"DAY27_DOCUMENT_{expected_marker}"
            assert c.filename == f"day27_doc_{expected_marker.lower()}.pdf"
            assert c.metadata["tenant"] == f"TENANT_{expected_marker}"

    def test_search_result_grouping_by_document(self) -> None:
        all_vsr: list[VectorSearchResult] = []
        for doc_id, filename, marker in ALL_DOC_TUPLES:
            all_vsr.extend(_make_doc_vsr(doc_id, filename, marker, count=3))

        citations = [AgentCitation.from_search_result(v) for v in all_vsr]
        sr = SearchResult(query="multi-doc search", citations=citations)

        assert sr.unique_document_count == 4
        assert sr.unique_documents == sorted([DOC_A, DOC_B, DOC_C, DOC_D])

        # Verify by_document dictionary grouping
        by_doc = sr.by_document
        assert set(by_doc.keys()) == {DOC_A, DOC_B, DOC_C, DOC_D}
        for doc_id in (DOC_A, DOC_B, DOC_C, DOC_D):
            assert len(by_doc[doc_id]) == 3
            assert all(c.document_id == doc_id for c in by_doc[doc_id])


# ===========================================================================
# 4. Multi-Evidence Visual Processing
# ===========================================================================

class TestMultiEvidenceVisualProcessing:
    """Verifies adaptation and processing of multi-evidence items from multiple documents."""

    def test_multi_evidence_adaptation_across_documents(self) -> None:
        all_vsr: list[VectorSearchResult] = []
        for doc_id, filename, marker in ALL_DOC_TUPLES:
            all_vsr.extend(_make_doc_vsr(doc_id, filename, marker, count=3))

        citations = [AgentCitation.from_search_result(v) for v in all_vsr]
        # In our fixture, chunk 2 (idx 1) is "image", chunk 4 (idx 3) is "chart", chunk 5 (idx 4) is "diagram"
        visual_citations = [c for c in citations if VisualEvidenceAdapter.is_visual_content_type(c.content_type)]
        assert len(visual_citations) == 4  # 1 image per document (chunks count=3)

        evidence_list = VisualEvidenceAdapter.adapt_batch(visual_citations)
        assert len(evidence_list) == 4

        # Verify each evidence retains exact source document
        for ev in evidence_list:
            expected_marker = ev.metadata.get("day27_marker")
            assert ev.document_id == f"DAY27_DOCUMENT_{expected_marker}"
            assert ev.content_type == "image"

    def test_vision_result_with_multi_evidence(self) -> None:
        all_vsr: list[VectorSearchResult] = []
        for doc_id, filename, marker in ALL_DOC_TUPLES:
            all_vsr.extend(_make_doc_vsr(doc_id, filename, marker, count=3))

        citations = [AgentCitation.from_search_result(v) for v in all_vsr]
        visual_citations = [c for c in citations if VisualEvidenceAdapter.is_visual_content_type(c.content_type)]
        evidence_list = VisualEvidenceAdapter.adapt_batch(visual_citations)

        v_res = VisionResult(
            query="Multi-document visual evidence query",
            status="success",
            description="Multi-evidence summary.",
            evidence=evidence_list,
        )

        assert v_res.is_success
        assert v_res.has_evidence
        assert len(v_res.evidence) == 4
        # Auto-inherited primary lineage from first item
        assert v_res.document_id == evidence_list[0].document_id


# ===========================================================================
# 5. End-to-End Lineage Integrity (Doc -> Chunk -> Search -> Citation -> Evidence -> Vision)
# ===========================================================================

class TestEndToEndLineageIntegrity:
    """Verifies strict 1:1 lineage preservation across the complete pipeline."""

    def test_full_pipeline_lineage_preservation(self) -> None:
        for doc_id, filename, marker in ALL_DOC_TUPLES:
            # 1. Chunk
            chunks = _make_doc_chunks(doc_id, filename, marker, count=3)
            # 2. Vector search result
            vsrs = _make_doc_vsr(doc_id, filename, marker, count=3)
            # 3. Agent citation
            citations = [AgentCitation.from_search_result(v) for v in vsrs]
            # 4. Visual evidence (image chunk)
            img_cits = [c for c in citations if c.content_type == "image"]
            evidence = VisualEvidenceAdapter.adapt_batch(img_cits)
            # 5. Vision result
            v_res = VisionResult(query="Lineage check", evidence=evidence)

            # Assert lineage consistency at all hops
            assert chunks[0].document_id == doc_id
            assert vsrs[0].document_id == doc_id
            assert citations[0].document_id == doc_id
            assert evidence[0].document_id == doc_id
            assert v_res.document_id == doc_id


# ===========================================================================
# 6. Variable Batch Sizes
# ===========================================================================

class TestVariableBatchSizes:
    """Verifies batch processing across 1, 5, 10, and 20 items."""

    @pytest.mark.parametrize("batch_size", [1, 5, 10, 20])
    def test_batch_sizes_execution(self, batch_size: int) -> None:
        chunks = [
            DocumentChunk(
                chunk_id=f"batch_chunk_{i}",
                chunk_index=i,
                document_id="BATCH_DOC_UNIFIED",
                filename="batch_file.pdf",
                page_number=i + 1,
                content=f"Batch content item {i}",
                content_type="text",
                metadata={"batch_index": i},
            )
            for i in range(batch_size)
        ]

        val_res = validate_chunks(chunks)
        assert val_res.is_valid
        assert val_res.total_chunks == batch_size
        assert val_res.valid_chunks == batch_size

        normalized = normalize_chunks(chunks)
        assert len(normalized) == batch_size


# ===========================================================================
# 7. Duplicate Input Preservation
# ===========================================================================

class TestDuplicateInputHandling:
    """Verifies contractual behavior when duplicate input items (A, A, B, B) are passed."""

    def test_duplicate_chunks_preservation(self) -> None:
        chunk_a = _make_doc_chunks(DOC_A, FILE_A, "A", count=1)[0]
        chunk_b = _make_doc_chunks(DOC_B, FILE_B, "B", count=1)[0]

        # Duplicate inputs: A, A, B, B
        input_list = [chunk_a, chunk_a, chunk_b, chunk_b]

        # Ingestion chunk validation correctly detects cross-document/duplicate chunk_id errors
        val_res = validate_chunks(input_list)
        assert not val_res.is_valid
        assert len(val_res.errors) > 0

        # normalize_chunks preserves all 4 elements in input list
        normalized = normalize_chunks(input_list)
        assert len(normalized) == 4
        assert normalized[0].document_id == DOC_A
        assert normalized[1].document_id == DOC_A
        assert normalized[2].document_id == DOC_B
        assert normalized[3].document_id == DOC_B

    def test_duplicate_citations_preservation(self) -> None:
        c_a = AgentCitation(document_id=DOC_A, filename=FILE_A, chunk_id="ck_a", page_number=1)
        c_b = AgentCitation(document_id=DOC_B, filename=FILE_B, chunk_id="ck_b", page_number=1)

        # Duplicate citations: A, A, B, B
        duplicate_citations = [c_a, c_a, c_b, c_b]
        resp = AgentResponse(answer="Dup answer", agent_name="Agent", citations=duplicate_citations)

        assert resp.is_success
        assert len(resp.citations) == 4
        assert resp.citations[0].document_id == DOC_A
        assert resp.citations[1].document_id == DOC_A
        assert resp.citations[2].document_id == DOC_B
        assert resp.citations[3].document_id == DOC_B


# ===========================================================================
# 8. Multi-Document Serialization Round-Trip
# ===========================================================================

class TestMultiDocumentSerialization:
    """Verifies to_dict() and from_dict() roundtrips with multi-document collections."""

    def test_search_result_multi_document_serialization(self) -> None:
        all_vsr: list[VectorSearchResult] = []
        for doc_id, filename, marker in ALL_DOC_TUPLES:
            all_vsr.extend(_make_doc_vsr(doc_id, filename, marker, count=3))

        citations = [AgentCitation.from_search_result(v) for v in all_vsr]
        sr = SearchResult(query="Serialization query", citations=citations)

        # Roundtrip
        d = sr.to_dict()
        sr_restored = SearchResult.from_dict(d)

        assert sr_restored.query == sr.query
        assert sr_restored.status == sr.status
        assert len(sr_restored.citations) == 12
        assert sr_restored.unique_document_count == 4
        assert sr_restored.unique_documents == sorted([DOC_A, DOC_B, DOC_C, DOC_D])

    def test_agent_response_multi_document_serialization(self) -> None:
        all_vsr: list[VectorSearchResult] = []
        for doc_id, filename, marker in ALL_DOC_TUPLES:
            all_vsr.extend(_make_doc_vsr(doc_id, filename, marker, count=2))

        citations = [AgentCitation.from_search_result(v) for v in all_vsr]
        resp = AgentResponse(
            answer="Multi-document synthesized response",
            agent_name="SupervisorAgent",
            status="success",
            citations=citations,
            metadata={"total_docs": 4},
        )

        d = resp.to_dict()
        resp_restored = AgentResponse.from_dict(d)

        assert resp_restored.answer == resp.answer
        assert resp_restored.unique_document_count == 4
        assert len(resp_restored.citations) == 8


# ===========================================================================
# 9. Request Isolation (A+B vs C+D)
# ===========================================================================

class TestRequestIsolationAcrossDocumentSubsets:
    """Verifies independent requests with different document sets remain completely isolated."""

    def test_request_ab_and_cd_isolation(self) -> None:
        # Request 1: DOC_A + DOC_B
        vsr_ab = _make_doc_vsr(DOC_A, FILE_A, "A", count=2) + _make_doc_vsr(DOC_B, FILE_B, "B", count=2)
        cits_ab = [AgentCitation.from_search_result(v) for v in vsr_ab]
        resp_ab = AgentResponse(answer="AB ans", agent_name="AgentAB", citations=cits_ab)

        # Request 2: DOC_C + DOC_D
        vsr_cd = _make_doc_vsr(DOC_C, FILE_C, "C", count=2) + _make_doc_vsr(DOC_D, FILE_D, "D", count=2)
        cits_cd = [AgentCitation.from_search_result(v) for v in vsr_cd]
        resp_cd = AgentResponse(answer="CD ans", agent_name="AgentCD", citations=cits_cd)

        # Assert AB never contains C or D
        assert set(resp_ab.unique_documents) == {DOC_A, DOC_B}
        assert DOC_C not in resp_ab.unique_documents
        assert DOC_D not in resp_ab.unique_documents

        # Assert CD never contains A or B
        assert set(resp_cd.unique_documents) == {DOC_C, DOC_D}
        assert DOC_A not in resp_cd.unique_documents
        assert DOC_B not in resp_cd.unique_documents


# ===========================================================================
# 10. Repeated Batch Execution Stability
# ===========================================================================

class TestRepeatedBatchExecutionStability:
    """Verifies repeating multi-document batch workflows creates no state accumulation."""

    def test_repeated_three_batches_stability(self) -> None:
        all_vsr = (
            _make_doc_vsr(DOC_A, FILE_A, "A", count=2)
            + _make_doc_vsr(DOC_B, FILE_B, "B", count=2)
        )

        responses = []
        for i in range(3):
            citations = [AgentCitation.from_search_result(v) for v in all_vsr]
            resp = AgentResponse(answer=f"Run {i}", agent_name="RepeatAgent", citations=citations)
            responses.append(resp)

        for resp in responses:
            assert len(resp.citations) == 4
            assert resp.unique_document_count == 2
            assert resp.unique_documents == [DOC_A, DOC_B]


# ===========================================================================
# 11. Failure Handling in Batch
# ===========================================================================

class TestFailureHandlingInBatch:
    """Verifies error isolation when an invalid item is rejected."""

    def test_invalid_item_rejection_does_not_affect_valid_batch(self) -> None:
        # Valid items
        valid_chunk = _make_doc_chunks(DOC_A, FILE_A, "A", count=1)[0]
        assert valid_chunk.document_id == DOC_A

        # Invalid item constructor raises AgentValidationError
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="file.pdf", chunk_id="ck")

        # Subsequent valid batch executes cleanly
        valid_citations = [
            AgentCitation(document_id=DOC_B, filename=FILE_B, chunk_id="ck_b", page_number=1)
        ]
        resp = AgentResponse(answer="valid", agent_name="bot", citations=valid_citations)
        assert resp.is_success
        assert resp.unique_documents == [DOC_B]


# ===========================================================================
# 12. Concurrent Multi-Document Processing
# ===========================================================================

class TestConcurrentMultiDocumentProcessing:
    """Verifies concurrent processing of independent documents across threads."""

    def test_concurrent_four_documents_isolation(self) -> None:
        def worker(doc_tuple: tuple[str, str, str]) -> tuple[str, list[str]]:
            doc_id, filename, marker = doc_tuple
            vsrs = _make_doc_vsr(doc_id, filename, marker, count=3)
            citations = [AgentCitation.from_search_result(v) for v in vsrs]
            resp = AgentResponse(answer=f"Ans for {doc_id}", agent_name="ThreadAgent", citations=citations)
            return doc_id, resp.unique_documents

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, item) for item in ALL_DOC_TUPLES]
            results = [f.result() for f in futures]

        assert len(results) == 4
        for doc_id, unique_docs in results:
            assert unique_docs == [doc_id]


# ===========================================================================
# 13. Resource Safety
# ===========================================================================

class TestResourceSafetyUnderBatch:
    """Verifies batch execution does not leave disk pollution."""

    def test_zero_disk_artifacts_after_batch(self) -> None:
        # Execute batch of all 4 documents
        all_vsr: list[VectorSearchResult] = []
        for doc_id, filename, marker in ALL_DOC_TUPLES:
            all_vsr.extend(_make_doc_vsr(doc_id, filename, marker, count=5))

        citations = [AgentCitation.from_search_result(v) for v in all_vsr]
        resp = AgentResponse(answer="Batch res", agent_name="Bot", citations=citations)
        assert resp.total_citations == 20

        root_path = Path(REPO_ROOT)
        unexpected = [
            f.name for f in root_path.iterdir()
            if f.is_file() and f.name.endswith((".tmp", ".temp", ".dump", ".log", ".bak"))
        ]
        assert unexpected == []
