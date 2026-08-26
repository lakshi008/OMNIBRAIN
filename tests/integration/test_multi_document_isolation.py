"""
OmniBrain Member 4 — Day 38 Multi-Document Isolation & Cross-Document Contamination Certification.

Verifies strict isolation across multiple documents processed within the same workflow, batch, retrieval operation, or repeated request:
  - Synthetic Document Set (DOC-A, DOC-B, DOC-C) with unique markers
  - Document → Page isolation
  - Page → Chunk isolation
  - Document marker contamination prevention (zero cross-bleeding)
  - Metadata isolation
  - Batch processing isolation
  - Identical content across different documents (distinguishable identity)
  - Different content on identical page numbers
  - Retrieval result isolation
  - SearchResult and AgentCitation isolation
  - VisionRequest and VisualEvidence isolation
  - VisionResult provenance isolation
  - Serialization & Batch serialization isolation (to_dict / from_dict)
  - Cross-request isolation & request-order invariance (A->B->C vs C->A->B)
  - Repeated request statelessness (no state accumulation across runs)
  - Caller-owned metadata mutation safety
  - Failure isolation & error contamination prevention
  - AgentState instance isolation
  - Large multi-document workload isolation (3 x 20 pages)

Constraints:
  - 100% offline. Zero external APIs, network, LLM, or production credentials.
  - Zero production code modified.
  - No isolation infrastructure, filtering logic, caching, or wrappers added.
"""

from __future__ import annotations

import copy
import sys
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
# Deterministic Synthetic Multi-Document Fixtures (Section 5 & 6)
# ---------------------------------------------------------------------------

DOC_A_ID = "DAY38-DOC-A"
DOC_A_FILENAME = "doc_a.pdf"
DOC_A_MARKER = "DAY38_DOCUMENT_A_UNIQUE_MARKER"

DOC_B_ID = "DAY38-DOC-B"
DOC_B_FILENAME = "doc_b.pdf"
DOC_B_MARKER = "DAY38_DOCUMENT_B_UNIQUE_MARKER"

DOC_C_ID = "DAY38-DOC-C"
DOC_C_FILENAME = "doc_c.pdf"
DOC_C_MARKER = "DAY38_DOCUMENT_C_UNIQUE_MARKER"


def build_triad_document(
    doc_id: str,
    filename: str,
    marker: str,
    page_count: int = 3,
    content_type: str = "image",
) -> tuple[ParsedDocument, list[DocumentChunk]]:
    """Build a deterministic synthetic document with unique markers and metadata."""
    pages: list[PageData] = []
    chunks: list[DocumentChunk] = []

    for p in range(1, page_count + 1):
        page_text = f"Content for {doc_id} Page {p:03d}. Unique payload: [{marker} :: PAGE_{p:03d}]."
        pages.append(
            PageData(
                page_number=p,
                text=page_text,
                char_count=len(page_text),
                has_content=True,
            )
        )
        chunk_id = f"CHK_{doc_id}_P{p:03d}"
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
                    "day38_document": doc_id,
                    "marker": marker,
                    "page": p,
                },
            )
        )

    meta = DocumentMetadata(
        document_id=doc_id,
        filename=filename,
        total_pages=page_count,
        content_type="application/pdf",
        created_at="2026-08-26T00:00:00Z",
        pages_with_content=page_count,
        pages_without_content=0,
    )
    return ParsedDocument(metadata=meta, pages=pages), chunks


# ===========================================================================
# 1. DOCUMENT → PAGE & PAGE → CHUNK ISOLATION
# ===========================================================================

class TestDocumentHierarchyIsolation:
    """Sections 7 & 8: Verify strict containment from Document to Page to Chunk."""

    def test_document_page_containment_triad(self) -> None:
        doc_a, _ = build_triad_document(DOC_A_ID, DOC_A_FILENAME, DOC_A_MARKER)
        doc_b, _ = build_triad_document(DOC_B_ID, DOC_B_FILENAME, DOC_B_MARKER)
        doc_c, _ = build_triad_document(DOC_C_ID, DOC_C_FILENAME, DOC_C_MARKER)

        assert all(DOC_A_MARKER in p.text for p in doc_a.pages)
        assert all(DOC_B_MARKER in p.text for p in doc_b.pages)
        assert all(DOC_C_MARKER in p.text for p in doc_c.pages)

        # Assert no cross-bleeding in text
        assert not any(DOC_B_MARKER in p.text or DOC_C_MARKER in p.text for p in doc_a.pages)
        assert not any(DOC_A_MARKER in p.text or DOC_C_MARKER in p.text for p in doc_b.pages)
        assert not any(DOC_A_MARKER in p.text or DOC_B_MARKER in p.text for p in doc_c.pages)

    def test_chunk_lineage_containment(self) -> None:
        _, chunks_a = build_triad_document(DOC_A_ID, DOC_A_FILENAME, DOC_A_MARKER)
        _, chunks_b = build_triad_document(DOC_B_ID, DOC_B_FILENAME, DOC_B_MARKER)

        assert all(c.document_id == DOC_A_ID for c in chunks_a)
        assert all(c.document_id == DOC_B_ID for c in chunks_b)
        assert {c.chunk_id for c in chunks_a}.isdisjoint({c.chunk_id for c in chunks_b})


# ===========================================================================
# 2. DOCUMENT MARKER & METADATA CONTAMINATION PREVENTION
# ===========================================================================

class TestMarkerAndMetadataIsolation:
    """Sections 9 & 10: Strict absence of foreign markers and metadata leakage."""

    def test_marker_isolation_across_embedding_preparations(self) -> None:
        _, chunks_a = build_triad_document(DOC_A_ID, DOC_A_FILENAME, DOC_A_MARKER)
        _, chunks_b = build_triad_document(DOC_B_ID, DOC_B_FILENAME, DOC_B_MARKER)
        _, chunks_c = build_triad_document(DOC_C_ID, DOC_C_FILENAME, DOC_C_MARKER)

        prep_a = prepare_for_embedding(chunks_a)
        prep_b = prepare_for_embedding(chunks_b)
        prep_c = prepare_for_embedding(chunks_c)

        for rec in prep_a.items:
            assert DOC_A_MARKER in rec.content
            assert DOC_B_MARKER not in rec.content
            assert DOC_C_MARKER not in rec.content
            assert rec.metadata["day38_document"] == DOC_A_ID

        for rec in prep_b.items:
            assert DOC_B_MARKER in rec.content
            assert DOC_A_MARKER not in rec.content
            assert DOC_C_MARKER not in rec.content
            assert rec.metadata["day38_document"] == DOC_B_ID

        for rec in prep_c.items:
            assert DOC_C_MARKER in rec.content
            assert DOC_A_MARKER not in rec.content
            assert DOC_B_MARKER not in rec.content
            assert rec.metadata["day38_document"] == DOC_C_ID


# ===========================================================================
# 3. IDENTICAL CONTENT & SAME PAGE NUMBER DIFFERENTIATION
# ===========================================================================

class TestIdenticalContentAndSamePageNumberDifferentiation:
    """Sections 12 & 13: Identical text or identical page numbers remain separated by document ID."""

    def test_identical_content_maintains_distinct_identities(self) -> None:
        identical_text = "DAY38 EXACT DUPLICATE PARAGRAPH"
        chunk_a = DocumentChunk(
            chunk_id="CHK_A_IDENTICAL", chunk_index=0, document_id=DOC_A_ID,
            filename=DOC_A_FILENAME, page_number=1, content=identical_text, content_type="text",
        )
        chunk_b = DocumentChunk(
            chunk_id="CHK_B_IDENTICAL", chunk_index=0, document_id=DOC_B_ID,
            filename=DOC_B_FILENAME, page_number=1, content=identical_text, content_type="text",
        )

        assert chunk_a.content == chunk_b.content
        assert chunk_a.document_id != chunk_b.document_id
        assert chunk_a.chunk_id != chunk_b.chunk_id

        prep_a = prepare_for_embedding([chunk_a])
        prep_b = prepare_for_embedding([chunk_b])

        assert prep_a.items[0].document_id == DOC_A_ID
        assert prep_b.items[0].document_id == DOC_B_ID

    def test_same_page_number_different_documents_distinct(self) -> None:
        # Page 1 across all 3 documents
        c_a1 = DocumentChunk("C_A1", 0, DOC_A_ID, DOC_A_FILENAME, 1, "Page 1 of Doc A", "text")
        c_b1 = DocumentChunk("C_B1", 0, DOC_B_ID, DOC_B_FILENAME, 1, "Page 1 of Doc B", "text")
        c_c1 = DocumentChunk("C_C1", 0, DOC_C_ID, DOC_C_FILENAME, 1, "Page 1 of Doc C", "text")

        cit_a = AgentCitation(document_id=DOC_A_ID, filename=DOC_A_FILENAME, chunk_id="C_A1", page_number=1)
        cit_b = AgentCitation(document_id=DOC_B_ID, filename=DOC_B_FILENAME, chunk_id="C_B1", page_number=1)
        cit_c = AgentCitation(document_id=DOC_C_ID, filename=DOC_C_FILENAME, chunk_id="C_C1", page_number=1)

        assert cit_a.page_number == cit_b.page_number == cit_c.page_number == 1
        assert cit_a.document_id == DOC_A_ID
        assert cit_b.document_id == DOC_B_ID
        assert cit_c.document_id == DOC_C_ID


# ===========================================================================
# 4. DOWNSTREAM ISOLATION (RETRIEVAL, CITATIONS, VISION)
# ===========================================================================

class TestDownstreamComponentIsolation:
    """Sections 14, 15, 16, 17, 18, 19: VectorSearchResult, Citations, Vision Request, Evidence."""

    def test_search_results_and_citations_isolation(self) -> None:
        _, chunks_a = build_triad_document(DOC_A_ID, DOC_A_FILENAME, DOC_A_MARKER)
        _, chunks_b = build_triad_document(DOC_B_ID, DOC_B_FILENAME, DOC_B_MARKER)

        vsrs_a = [
            VectorSearchResult(
                chunk_id=c.chunk_id, score=0.95, document_id=c.document_id,
                filename=c.filename, page_number=c.page_number, chunk_index=c.chunk_index,
                content_type=c.content_type, content=c.content, metadata=c.metadata,
            )
            for c in chunks_a
        ]
        vsrs_b = [
            VectorSearchResult(
                chunk_id=c.chunk_id, score=0.90, document_id=c.document_id,
                filename=c.filename, page_number=c.page_number, chunk_index=c.chunk_index,
                content_type=c.content_type, content=c.content, metadata=c.metadata,
            )
            for c in chunks_b
        ]

        cits_a = [AgentCitation.from_search_result(r) for r in vsrs_a]
        cits_b = [AgentCitation.from_search_result(r) for r in vsrs_b]

        resp_a = AgentResponse(answer="A answer", agent_name="Agent", citations=cits_a)
        resp_b = AgentResponse(answer="B answer", agent_name="Agent", citations=cits_b)

        assert resp_a.unique_documents == [DOC_A_ID]
        assert resp_b.unique_documents == [DOC_B_ID]
        assert not any(c.document_id == DOC_B_ID for c in resp_a.citations)
        assert not any(c.document_id == DOC_A_ID for c in resp_b.citations)

    def test_vision_evidence_and_results_isolation(self) -> None:
        ev_a = VisualEvidence(
            document_id=DOC_A_ID, filename=DOC_A_FILENAME, chunk_id="CHK_A_001",
            page_number=1, content_type="image", description="Doc A Image",
        )
        ev_b = VisualEvidence(
            document_id=DOC_B_ID, filename=DOC_B_FILENAME, chunk_id="CHK_B_001",
            page_number=1, content_type="image", description="Doc B Image",
        )

        req_a = VisionRequest(query="Examine A", evidence=[ev_a])
        req_b = VisionRequest(query="Examine B", evidence=[ev_b])

        res_a = VisionResult(query=req_a.query, status="success", description="Analyzed A", evidence=req_a.evidence)
        res_b = VisionResult(query=req_b.query, status="success", description="Analyzed B", evidence=req_b.evidence)

        assert res_a.document_id == DOC_A_ID
        assert res_b.document_id == DOC_B_ID
        assert res_a.filename == DOC_A_FILENAME
        assert res_b.filename == DOC_B_FILENAME


# ===========================================================================
# 5. SERIALIZATION & BATCH SERIALIZATION ISOLATION
# ===========================================================================

class TestSerializationIsolation:
    """Sections 20 & 21: Serialization does not overwrite or merge source identities."""

    def test_serialization_and_batch_deserialization_isolation(self) -> None:
        cits = [
            AgentCitation(document_id=DOC_A_ID, filename=DOC_A_FILENAME, chunk_id="CHK_A1", metadata={"origin": "A"}),
            AgentCitation(document_id=DOC_B_ID, filename=DOC_B_FILENAME, chunk_id="CHK_B1", metadata={"origin": "B"}),
            AgentCitation(document_id=DOC_C_ID, filename=DOC_C_FILENAME, chunk_id="CHK_C1", metadata={"origin": "C"}),
        ]

        serialized_batch = [c.to_dict() for c in cits]
        deserialized_batch = [AgentCitation.from_dict(d) for d in serialized_batch]

        assert len(deserialized_batch) == 3
        assert deserialized_batch[0].document_id == DOC_A_ID
        assert deserialized_batch[1].document_id == DOC_B_ID
        assert deserialized_batch[2].document_id == DOC_C_ID

        assert deserialized_batch[0].metadata["origin"] == "A"
        assert deserialized_batch[1].metadata["origin"] == "B"
        assert deserialized_batch[2].metadata["origin"] == "C"


# ===========================================================================
# 6. CROSS-REQUEST ISOLATION, ORDER INVARIANCE & REPETITION
# ===========================================================================

class TestCrossRequestIsolationAndOrderInvariance:
    """Sections 22, 23, 24: Request execution order permutations and repeated requests."""

    def test_request_order_invariance(self) -> None:
        def run_pipeline_for_doc(doc_id: str, filename: str, marker: str) -> str:
            _, chunks = build_triad_document(doc_id, filename, marker)
            prep = prepare_for_embedding(chunks)
            return prep.items[0].document_id

        # Permutation 1: A -> B -> C
        assert [run_pipeline_for_doc(DOC_A_ID, DOC_A_FILENAME, DOC_A_MARKER),
                run_pipeline_for_doc(DOC_B_ID, DOC_B_FILENAME, DOC_B_MARKER),
                run_pipeline_for_doc(DOC_C_ID, DOC_C_FILENAME, DOC_C_MARKER)] == [DOC_A_ID, DOC_B_ID, DOC_C_ID]

        # Permutation 2: C -> A -> B
        assert [run_pipeline_for_doc(DOC_C_ID, DOC_C_FILENAME, DOC_C_MARKER),
                run_pipeline_for_doc(DOC_A_ID, DOC_A_FILENAME, DOC_A_MARKER),
                run_pipeline_for_doc(DOC_B_ID, DOC_B_FILENAME, DOC_B_MARKER)] == [DOC_C_ID, DOC_A_ID, DOC_B_ID]

        # Permutation 3: B -> C -> A
        assert [run_pipeline_for_doc(DOC_B_ID, DOC_B_FILENAME, DOC_B_MARKER),
                run_pipeline_for_doc(DOC_C_ID, DOC_C_FILENAME, DOC_C_MARKER),
                run_pipeline_for_doc(DOC_A_ID, DOC_A_FILENAME, DOC_A_MARKER)] == [DOC_B_ID, DOC_C_ID, DOC_A_ID]

    def test_repeated_processing_statelessness(self) -> None:
        for _ in range(2):
            _, chunks_a = build_triad_document(DOC_A_ID, DOC_A_FILENAME, DOC_A_MARKER)
            _, chunks_b = build_triad_document(DOC_B_ID, DOC_B_FILENAME, DOC_B_MARKER)
            _, chunks_c = build_triad_document(DOC_C_ID, DOC_C_FILENAME, DOC_C_MARKER)

            prep_a = prepare_for_embedding(chunks_a)
            prep_b = prepare_for_embedding(chunks_b)
            prep_c = prepare_for_embedding(chunks_c)

            assert prep_a.total_items == 3 and prep_a.document_id == DOC_A_ID
            assert prep_b.total_items == 3 and prep_b.document_id == DOC_B_ID
            assert prep_c.total_items == 3 and prep_c.document_id == DOC_C_ID


# ===========================================================================
# 7. MUTATION SAFETY, FAILURE & STATE ISOLATION
# ===========================================================================

class TestMutationFailureAndStateIsolation:
    """Sections 26, 27, 28, 29: Caller metadata safety, failure isolation, and AgentState isolation."""

    def test_caller_metadata_unmutated_across_multi_doc_processing(self) -> None:
        meta_a = {"doc": "A", "key": "val_a"}
        meta_b = {"doc": "B", "key": "val_b"}
        snapshot_a = copy.deepcopy(meta_a)
        snapshot_b = copy.deepcopy(meta_b)

        cit_a = AgentCitation(DOC_A_ID, DOC_A_FILENAME, "C_A1", content_type="image", metadata=meta_a)
        cit_b = AgentCitation(DOC_B_ID, DOC_B_FILENAME, "C_B1", content_type="image", metadata=meta_b)

        VisualEvidenceAdapter.adapt_citation(cit_a)
        VisualEvidenceAdapter.adapt_citation(cit_b)

        assert meta_a == snapshot_a
        assert meta_b == snapshot_b

    def test_failure_isolation_and_error_containment(self) -> None:
        # Request A: valid
        cit_a = AgentCitation(DOC_A_ID, DOC_A_FILENAME, "C_A1")
        assert cit_a.document_id == DOC_A_ID

        # Request B: invalid (empty document_id)
        with pytest.raises(AgentValidationError):
            AgentCitation("", DOC_B_FILENAME, "C_B1")

        # Request C: valid (unaffected by Request B failure)
        cit_c = AgentCitation(DOC_C_ID, DOC_C_FILENAME, "C_C1")
        assert cit_c.document_id == DOC_C_ID

    def test_agent_state_instance_isolation(self) -> None:
        state_a = AgentState(query="Query A")
        state_b = AgentState(query="Query B")
        state_c = AgentState(query="Query C")

        state_a.add_error("Error in A")
        state_a.add_citation(AgentCitation(DOC_A_ID, DOC_A_FILENAME, "C_A1"))

        assert len(state_a.errors) == 1
        assert len(state_a.citations) == 1

        # States B and C remain pristine
        assert state_b.errors == []
        assert state_b.citations == []
        assert state_c.errors == []
        assert state_c.citations == []


# ===========================================================================
# 8. LARGE MULTI-DOCUMENT WORKLOAD ISOLATION (3 x 20 PAGES)
# ===========================================================================

class TestLargeMultiDocumentWorkload:
    """Section 30: 3 documents x 20 pages each processed simultaneously."""

    def test_3x20_page_multi_document_workload_isolation(self) -> None:
        doc_a, chunks_a = build_triad_document(DOC_A_ID, DOC_A_FILENAME, DOC_A_MARKER, page_count=20)
        doc_b, chunks_b = build_triad_document(DOC_B_ID, DOC_B_FILENAME, DOC_B_MARKER, page_count=20)
        doc_c, chunks_c = build_triad_document(DOC_C_ID, DOC_C_FILENAME, DOC_C_MARKER, page_count=20)

        prep_a = prepare_for_embedding(chunks_a)
        prep_b = prepare_for_embedding(chunks_b)
        prep_c = prepare_for_embedding(chunks_c)

        assert prep_a.total_items == 20
        assert prep_b.total_items == 20
        assert prep_c.total_items == 20

        # Assert full isolation across all items
        assert all(r.document_id == DOC_A_ID and DOC_A_MARKER in r.content for r in prep_a.items)
        assert all(r.document_id == DOC_B_ID and DOC_B_MARKER in r.content for r in prep_b.items)
        assert all(r.document_id == DOC_C_ID and DOC_C_MARKER in r.content for r in prep_c.items)
