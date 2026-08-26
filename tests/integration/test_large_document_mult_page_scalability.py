"""
OmniBrain Member 4 — Day 37 Large-Document Processing & Multi-Page Scalability Certification.

Validates the ingestion and downstream contracts across multi-page synthetic documents:
  - Small (5 pages), Medium (25 pages), Large (50 pages), Stress (75 pages)
  - Deterministic unique page markers and metadata
  - Page count and sequential ordering integrity
  - Chunk completeness, content integrity, and identity stability
  - Multi-document isolation (e.g. DOC-A vs DOC-B with 50 pages each)
  - Identical content isolation across different parent documents
  - Large document retrieval and search compatibility
  - Large document agent citation and response contracts
  - Large document vision request and evidence adaptation
  - Serialization of large multi-page responses (to_dict / from_dict)
  - Batch document processing (multiple documents in one scenario)
  - Repeated large document processing (zero state accumulation)
  - Edge pages (very short text, long text, punctuation, numbers)
  - Empty page handling (has_content=False)
  - Substantial text payload resilience
  - Memory behavior monitoring (tracemalloc)
  - Performance duration scaling observation
  - Deterministic reproducibility

Constraints:
  - 100% offline. Zero external APIs, network, LLM, or production credentials.
  - Zero production code modified.
  - No streaming, caching, pagination, or performance optimizations added.
"""

from __future__ import annotations

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
# Synthetic Multi-Page Document Generator (Section 5 & 6)
# ---------------------------------------------------------------------------

def generate_synthetic_document(
    doc_id: str,
    filename: str,
    page_count: int,
    content_type: str = "image",
    custom_content_builder: Any = None,
) -> tuple[ParsedDocument, list[DocumentChunk]]:
    """Generate a deterministic synthetic ParsedDocument and corresponding DocumentChunk list."""
    pages: list[PageData] = []
    chunks: list[DocumentChunk] = []

    for p in range(1, page_count + 1):
        if custom_content_builder:
            page_text = custom_content_builder(p)
        else:
            page_text = f"DAY37 DOCUMENT {doc_id} PAGE {p:03d}: Comprehensive report section {p}."

        char_len = len(page_text)
        has_content = char_len > 0

        pages.append(
            PageData(
                page_number=p,
                text=page_text,
                char_count=char_len,
                has_content=has_content,
            )
        )

        chunk_id = f"CHK_{doc_id}_P{p:03d}_C0"
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                chunk_index=p - 1,
                document_id=doc_id,
                filename=filename,
                page_number=p,
                content=page_text,
                content_type=content_type,
                metadata={
                    "day37_document": doc_id,
                    "day37_page": p,
                    "section_id": f"sec_{p}",
                },
            )
        )

    meta = DocumentMetadata(
        document_id=doc_id,
        filename=filename,
        total_pages=page_count,
        content_type="application/pdf",
        created_at="2026-08-26T00:00:00Z",
        pages_with_content=sum(1 for p in pages if p.has_content),
        pages_without_content=sum(1 for p in pages if not p.has_content),
    )

    doc = ParsedDocument(metadata=meta, pages=pages)
    return doc, chunks


# ===========================================================================
# 1. MULTI-PAGE PROCESSING & SCALING (5, 25, 50, 75 PAGES)
# ===========================================================================

class TestMultiPageProcessingAndScaling:
    """Sections 5, 8, 9: Small, Medium, Large, and Stress document sizes."""

    def test_5_page_document_processing(self) -> None:
        doc, chunks = generate_synthetic_document("DAY37-DOC-5P", "doc_5p.pdf", 5)
        assert len(doc.pages) == 5
        assert doc.metadata.total_pages == 5

        norm = normalize_chunks(chunks)
        prep = prepare_for_embedding(norm)
        assert prep.total_items == 5
        assert prep.is_ready is True

    def test_25_page_document_processing(self) -> None:
        doc, chunks = generate_synthetic_document("DAY37-DOC-25P", "doc_25p.pdf", 25)
        assert len(doc.pages) == 25
        norm = normalize_chunks(chunks)
        prep = prepare_for_embedding(norm)
        assert prep.total_items == 25

    def test_50_page_document_processing(self) -> None:
        doc, chunks = generate_synthetic_document("DAY37-DOC-50P", "doc_50p.pdf", 50)
        assert len(doc.pages) == 50
        norm = normalize_chunks(chunks)
        prep = prepare_for_embedding(norm)
        assert prep.total_items == 50

    def test_75_page_stress_document_processing(self) -> None:
        doc, chunks = generate_synthetic_document("DAY37-DOC-75P", "doc_75p.pdf", 75)
        assert len(doc.pages) == 75
        norm = normalize_chunks(chunks)
        prep = prepare_for_embedding(norm)
        assert prep.total_items == 75

    def test_page_order_sequential_integrity(self) -> None:
        doc, chunks = generate_synthetic_document("DAY37-DOC-ORDER", "doc_order.pdf", 30)
        page_numbers = [p.page_number for p in doc.pages]
        assert page_numbers == list(range(1, 31))

        chunk_page_numbers = [c.page_number for c in chunks]
        assert chunk_page_numbers == list(range(1, 31))


# ===========================================================================
# 2. CHUNK COMPLETENESS, IDENTITY & METADATA INTEGRITY
# ===========================================================================

class TestChunkCompletenessAndMetadata:
    """Sections 10, 11, 12, 13: Unique markers, chunk identity, and metadata preservation."""

    def test_chunk_content_traces_back_to_page_markers(self) -> None:
        doc, chunks = generate_synthetic_document("DAY37-DOC-TRACE", "doc_trace.pdf", 20)
        for page in doc.pages:
            matched = [c for c in chunks if c.page_number == page.page_number]
            assert len(matched) == 1
            assert matched[0].content == page.text
            assert f"PAGE {page.page_number:03d}" in matched[0].content

    def test_chunk_identities_unique_and_stable(self) -> None:
        _, chunks = generate_synthetic_document("DAY37-DOC-ID", "doc_id.pdf", 40)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_metadata_preserved_across_large_embedding_prep(self) -> None:
        _, chunks = generate_synthetic_document("DAY37-DOC-META", "doc_meta.pdf", 35)
        prep = prepare_for_embedding(chunks)

        for rec in prep.items:
            p_num = rec.page_number
            assert rec.metadata["day37_page"] == p_num
            assert rec.metadata["section_id"] == f"sec_{p_num}"


# ===========================================================================
# 3. MULTI-DOCUMENT & IDENTICAL CONTENT ISOLATION
# ===========================================================================

class TestMultiDocumentAndIdenticalContentIsolation:
    """Sections 14 & 15: Separate documents and identical text payload isolation."""

    def test_multi_document_large_processing_isolation(self) -> None:
        doc_a, chunks_a = generate_synthetic_document("DOC-A", "doc_a.pdf", 30)
        doc_b, chunks_b = generate_synthetic_document("DOC-B", "doc_b.pdf", 30)

        prep_a = prepare_for_embedding(chunks_a)
        prep_b = prepare_for_embedding(chunks_b)

        assert prep_a.total_items == 30
        assert prep_b.total_items == 30

        assert all(r.document_id == "DOC-A" for r in prep_a.items)
        assert all(r.document_id == "DOC-B" for r in prep_b.items)

    def test_identical_page_content_retains_document_distinction(self) -> None:
        shared_text = "DAY37 IDENTICAL CONTENT ACROSS DOCUMENTS"
        doc_a, chunks_a = generate_synthetic_document("DOC-A", "doc_a.pdf", 5, custom_content_builder=lambda _: shared_text)
        doc_b, chunks_b = generate_synthetic_document("DOC-B", "doc_b.pdf", 5, custom_content_builder=lambda _: shared_text)

        # Chunks have identical content
        assert chunks_a[0].content == chunks_b[0].content
        # But distinct identities
        assert chunks_a[0].document_id != chunks_b[0].document_id
        assert chunks_a[0].chunk_id != chunks_b[0].chunk_id

        prep_a = prepare_for_embedding(chunks_a)
        prep_b = prepare_for_embedding(chunks_b)
        assert prep_a.items[0].document_id == "DOC-A"
        assert prep_b.items[0].document_id == "DOC-B"


# ===========================================================================
# 4. DOWNSTREAM CONTRACT COMPATIBILITY (RETRIEVAL, AGENTS, VISION)
# ===========================================================================

class TestDownstreamContractCompatibility:
    """Sections 16, 17, 18, 19: Full downstream handoff with large documents."""

    def test_large_document_retrieval_and_agent_citations(self) -> None:
        doc, chunks = generate_synthetic_document("DAY37-DOWNSTREAM", "doc_downstream.pdf", 40)
        prep = prepare_for_embedding(chunks)

        vsrs = [
            VectorSearchResult(
                chunk_id=r.chunk_id,
                score=round(0.99 - (i * 0.01), 3),
                document_id=r.document_id,
                filename=r.filename,
                page_number=r.page_number,
                chunk_index=r.chunk_index,
                content_type=r.content_type,
                content=r.content,
                metadata=r.metadata,
            )
            for i, r in enumerate(prep.items)
        ]

        processed = process_retrieval_results(vsrs, min_score=0.5, max_results=40)
        assert len(processed) == 40

        citations = [AgentCitation.from_search_result(r) for r in processed]
        assert len(citations) == 40
        assert citations[0].document_id == "DAY37-DOWNSTREAM"
        assert citations[0].page_number == 1
        assert citations[39].page_number == 40

        # AgentResponse
        resp = AgentResponse(
            answer="Multi-page summary",
            agent_name="SearchAgent",
            citations=citations,
            metadata={"query": "summary"},
        )
        assert resp.total_citations == 40
        assert resp.unique_documents == ["DAY37-DOWNSTREAM"]

        # Serialization round trip
        d = resp.to_dict()
        resp2 = AgentResponse.from_dict(d)
        assert resp2.total_citations == 40
        assert resp2.citations[25].page_number == 26

    def test_large_document_vision_adaptation(self) -> None:
        _, chunks = generate_synthetic_document("DAY37-VISION-DOC", "doc_vis.pdf", 30)
        vsrs = [
            VectorSearchResult(
                chunk_id=c.chunk_id, score=0.95, document_id=c.document_id,
                filename=c.filename, page_number=c.page_number, chunk_index=c.chunk_index,
                content_type="image", content=c.content, metadata=c.metadata,
            )
            for c in chunks
        ]
        citations = [AgentCitation.from_search_result(r) for r in vsrs]
        evidence = VisualEvidenceAdapter.adapt_batch(citations)

        assert len(evidence) == 30
        assert evidence[0].document_id == "DAY37-VISION-DOC"
        assert evidence[29].page_number == 30

        v_req = VisionRequest(query="Examine all pages", evidence=evidence)
        assert v_req.total_evidence == 30


# ===========================================================================
# 5. BATCH PROCESSING & REPEATED LARGE DOCUMENT RUNS
# ===========================================================================

class TestBatchAndRepeatedExecution:
    """Sections 20 & 21: Multi-document batches and repeated executions."""

    def test_batch_processing_of_multiple_synthetic_documents(self) -> None:
        docs = [
            generate_synthetic_document(f"BATCH_DOC_{d}", f"doc_{d}.pdf", 15)
            for d in range(3)
        ]

        all_prep_results: list[EmbeddingPreparationResult] = []
        for _, chunks in docs:
            all_prep_results.append(prepare_for_embedding(chunks))

        assert len(all_prep_results) == 3
        for idx, prep in enumerate(all_prep_results):
            assert prep.total_items == 15
            assert prep.document_id == f"BATCH_DOC_{idx}"

    def test_repeated_large_document_execution_purity(self) -> None:
        _, chunks = generate_synthetic_document("DAY37-REPEAT", "doc_repeat.pdf", 35)
        run_outputs: list[list[str]] = []

        for run in range(3):
            prep = prepare_for_embedding(chunks)
            item_ids = [item.chunk_id for item in prep.items]
            run_outputs.append(item_ids)

        assert run_outputs[0] == run_outputs[1] == run_outputs[2]


# ===========================================================================
# 6. EDGE PAGES & LARGE CHUNK PAYLOADS
# ===========================================================================

class TestEdgePagesAndLargePayloads:
    """Sections 22, 23, 24: Short text, empty pages, and substantial text payloads."""

    def test_edge_pages_short_text_numbers_and_punctuation(self) -> None:
        def edge_builder(p: int) -> str:
            if p == 1:
                return "X"  # Very short text
            elif p == 2:
                return "1234567890 9876543210 #@$%^&*()_+"  # Numbers & symbols
            elif p == 3:
                return "Normal text paragraph."
            return f"Standard page {p}"

        doc, chunks = generate_synthetic_document("DAY37-EDGE", "doc_edge.pdf", 4, custom_content_builder=edge_builder)
        prep = prepare_for_embedding(chunks)

        assert prep.total_items == 4
        assert prep.items[0].content == "X"
        assert "1234567890" in prep.items[1].content

    def test_empty_page_handling_in_parsed_document(self) -> None:
        pages = [
            PageData(page_number=1, text="Page 1 text", char_count=11, has_content=True),
            PageData(page_number=2, text="", char_count=0, has_content=False),
            PageData(page_number=3, text="Page 3 text", char_count=11, has_content=True),
        ]
        meta = DocumentMetadata(
            document_id="DAY37-EMPTY-P", filename="doc_empty.pdf", total_pages=3,
            content_type="application/pdf", created_at="2026-08-26T00:00:00Z",
            pages_with_content=2, pages_without_content=1,
        )
        doc = ParsedDocument(metadata=meta, pages=pages)

        assert doc.metadata.pages_without_content == 1
        assert "Page 1 text" in doc.get_all_text()
        assert "Page 3 text" in doc.get_all_text()

    def test_substantial_page_text_payload(self) -> None:
        large_text = ("Comprehensive enterprise audit report section. " * 100).strip()
        def large_builder(p: int) -> str:
            if p == 10:
                return large_text
            return f"Page {p} text."

        doc, chunks = generate_synthetic_document("DAY37-BIG-PAYLOAD", "doc_big.pdf", 15, custom_content_builder=large_builder)
        prep = prepare_for_embedding(chunks)

        assert prep.total_items == 15
        assert prep.items[9].content == large_text


# ===========================================================================
# 7. PERFORMANCE OBSERVATION & MEMORY BEHAVIOR
# ===========================================================================

class TestPerformanceObservationAndMemory:
    """Sections 25, 26, 28: Tracemalloc memory footprint and performance measurements."""

    def test_performance_scaling_measurement_across_document_sizes(self) -> None:
        measurements: list[dict[str, Any]] = []

        for p_count in (5, 25, 50, 75):
            _, chunks = generate_synthetic_document(f"PERF_{p_count}P", f"doc_{p_count}.pdf", p_count)
            tracemalloc.start()
            t0 = time.perf_counter()

            norm = normalize_chunks(chunks)
            prep = prepare_for_embedding(norm)

            elapsed = time.perf_counter() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            measurements.append({
                "page_count": p_count,
                "duration_sec": round(elapsed, 4),
                "avg_ms_per_page": round((elapsed / p_count) * 1000, 3),
                "peak_memory_kb": round(peak_mem / 1024, 2),
                "success": prep.total_items == p_count,
            })

        assert len(measurements) == 4
        for m in measurements:
            assert m["success"] is True
            assert m["duration_sec"] > 0.0
            assert m["peak_memory_kb"] > 0.0

        # Memory footprint for 75 pages should remain well below 50 MB
        assert measurements[-1]["peak_memory_kb"] < 50 * 1024
