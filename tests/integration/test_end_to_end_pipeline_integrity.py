"""
OmniBrain Member 4 — Day 40 End-to-End Pipeline Integrity & Full Workflow Certification.

Validates the complete existing OMNIBRAIN workflow from input document through final result
using only actual public APIs and models. Covers every stage that exists in the repository:

  DOCUMENT
     ↓  ParsedDocument / PageData
  INGESTION (chunking + validation)
     ↓  DocumentChunk / ChunkingResult / ChunkValidationResult
  EMBEDDING PREPARATION
     ↓  EmbeddingPreparationResult / EmbeddingRecord
  VECTOR / RETRIEVAL RESULT
     ↓  VectorSearchResult / RetrievalServiceResult
  SEARCH / AGENT
     ↓  AgentCitation / AgentResponse / SearchResult
  CONTEXT BUILDING
     ↓  build_retrieval_context
  VISION / VISUAL EVIDENCE
     ↓  VisualEvidence / VisionRequest / VisionResult
  FINAL RESULT (AgentResponse serialization + lineage)

Coverage:
  - Single-document end-to-end pipeline (3 pages, unique content markers)
  - Extraction integrity (markers survive every stage)
  - Chunking stage verification (content, lineage, metadata)
  - Embedding preparation integrity
  - Retrieval stage (VectorSearchResult construction and processing)
  - Search / Agent stage (AgentCitation, AgentResponse)
  - Context building (build_retrieval_context output)
  - Vision stage (VisualEvidence, VisionRequest, VisionResult)
  - Multi-modal lineage (text + visual evidence from same document)
  - Full end-to-end unique marker trace through all stages
  - Cross-document negative isolation test (DOC-A vs DOC-B)
  - Serialization round-trip for all supported public models
  - Error path (expected validation failures with no state leakage)
  - Repeated end-to-end execution (3 runs, no accumulated state)
  - Multi-document full-pipeline (DOC-A, DOC-B, DOC-C → correct final results)
  - Input mutation safety across pipeline calls

Constraints:
  - 100% offline. Zero real APIs, network, LLM, vision models, or production credentials.
  - Zero production code modified.
  - No orchestration, adapters, wrappers, caching, or optimization added.
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
    RetrievalServiceResult,
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
from vision.result_normalizer import VisionResultNormalizer, VisionExecutionTrace
from vision.exceptions import VisionEvidenceError, VisionInputValidationError

# ---------------------------------------------------------------------------
# Synthetic Pipeline Fixture — Section 5
# ---------------------------------------------------------------------------

DOC_001_ID = "DAY40-DOC-001"
DOC_001_FILE = "day40_doc_001.pdf"

DAY40_TITLE = "DAY40_TITLE"
DAY40_SECTION_A = "DAY40_SECTION_A"
DAY40_SECTION_B = "DAY40_SECTION_B"
DAY40_UNIQUE_MARKER = "DAY40_UNIQUE_MARKER"

_PAGES = [
    (1, f"{DOC_001_ID} {DAY40_TITLE} {DAY40_SECTION_A} — Section heading and introductory material."),
    (2, f"{DOC_001_ID} {DAY40_SECTION_B} — Detailed analysis with supporting evidence."),
    (3, f"{DOC_001_ID} {DAY40_UNIQUE_MARKER} — Conclusive findings and recommendations."),
]


def _make_parsed_document(doc_id: str = DOC_001_ID, filename: str = DOC_001_FILE, pages=_PAGES) -> ParsedDocument:
    page_data = [
        PageData(page_number=pn, text=txt, char_count=len(txt), has_content=True)
        for pn, txt in pages
    ]
    meta = DocumentMetadata(
        document_id=doc_id, filename=filename, total_pages=len(pages),
        content_type="application/pdf", created_at="2026-08-26T00:00:00Z",
        pages_with_content=len(pages), pages_without_content=0,
    )
    return ParsedDocument(metadata=meta, pages=page_data)


def _make_chunks(doc_id: str = DOC_001_ID, filename: str = DOC_001_FILE, pages=_PAGES) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"CHK_{doc_id}_P{pn:03d}",
            chunk_index=pn - 1,
            document_id=doc_id,
            filename=filename,
            page_number=pn,
            content=txt,
            content_type="image",
            metadata={"day40_document": doc_id, "day40_page": pn},
        )
        for pn, txt in pages
    ]


def _full_pipeline(doc_id: str, filename: str, pages) -> dict[str, Any]:
    """Execute the full supported pipeline for one synthetic document. Returns key lineage data."""
    # Stage 1: Ingestion
    doc = _make_parsed_document(doc_id, filename, pages)
    chunks = _make_chunks(doc_id, filename, pages)

    # Stage 2: Chunk validation
    validation = validate_chunks(chunks)

    # Stage 3: Normalization + Embedding preparation
    norm = normalize_chunks(chunks)
    prep = prepare_for_embedding(norm)

    # Stage 4: Retrieval (synthetic VectorSearchResults)
    vsrs = [
        VectorSearchResult(
            chunk_id=r.chunk_id, score=round(0.99 - i * 0.01, 3),
            document_id=r.document_id, filename=r.filename,
            page_number=r.page_number, chunk_index=r.chunk_index,
            content_type=r.content_type, content=r.content, metadata=r.metadata,
        )
        for i, r in enumerate(prep.items)
    ]
    processed = process_retrieval_results(vsrs, min_score=0.5, max_results=len(pages))

    # Stage 5: Context building
    ctx = build_retrieval_context(processed)

    # Stage 6: Agent citations + response
    citations = [AgentCitation.from_search_result(r) for r in processed]
    response = AgentResponse(
        answer=f"Synthesized answer for {doc_id}",
        agent_name="SearchAgent",
        citations=citations,
        metadata={"pipeline": "day40"},
    )

    # Stage 7: Vision evidence + request
    evidence = VisualEvidenceAdapter.adapt_batch(citations)
    v_req = VisionRequest(query=f"Examine {doc_id}", evidence=evidence)

    return {
        "doc_id": doc_id,
        "doc_pages": len(doc.pages),
        "validation_valid": validation.is_valid,
        "prep_total": prep.total_items,
        "prep_doc_id": prep.document_id,
        "processed_count": len(processed),
        "context": ctx,
        "response_citations": len(response.citations),
        "unique_docs": response.unique_documents,
        "evidence_count": len(evidence),
        "vision_doc_id": v_req.evidence[0].document_id if v_req.evidence else None,
        "page_contents": [r.content for r in prep.items],
        "citation_doc_ids": [c.document_id for c in citations],
        "evidence_doc_ids": [e.document_id for e in evidence],
        "response": response,
        "v_req": v_req,
    }


# ===========================================================================
# 1. INGESTION & EXTRACTION STAGE (Sections 6, 7)
# ===========================================================================

class TestIngestionAndExtractionStage:
    """Verify document ingestion, extraction integrity, and marker survival."""

    def test_parsed_document_construction_and_markers(self) -> None:
        doc = _make_parsed_document()
        assert doc.metadata.document_id == DOC_001_ID
        assert doc.metadata.total_pages == 3

        all_text = doc.get_all_text()
        for marker in (DAY40_TITLE, DAY40_SECTION_A, DAY40_SECTION_B, DAY40_UNIQUE_MARKER):
            assert marker in all_text

    def test_page_identity_and_order(self) -> None:
        doc = _make_parsed_document()
        page_numbers = [p.page_number for p in doc.pages]
        assert page_numbers == [1, 2, 3]
        assert doc.pages[0].text and DAY40_SECTION_A in doc.pages[0].text
        assert doc.pages[2].text and DAY40_UNIQUE_MARKER in doc.pages[2].text


# ===========================================================================
# 2. CHUNKING STAGE & LINEAGE (Sections 8, 9)
# ===========================================================================

class TestChunkingStageAndLineage:
    """Verify chunking output, content integrity, and document→page→chunk lineage."""

    def test_chunk_creation_and_content_integrity(self) -> None:
        chunks = _make_chunks()
        assert len(chunks) == 3
        assert all(c.document_id == DOC_001_ID for c in chunks)

        for pn, txt in _PAGES:
            chunk = next(c for c in chunks if c.page_number == pn)
            assert chunk.content == txt
            assert chunk.chunk_id == f"CHK_{DOC_001_ID}_P{pn:03d}"

    def test_chunk_validation_passes(self) -> None:
        chunks = _make_chunks()
        result = validate_chunks(chunks)
        assert result.is_valid
        assert result.total_chunks == 3

    def test_chunk_lineage_document_page_chunk(self) -> None:
        chunks = _make_chunks()
        for chunk in chunks:
            assert chunk.document_id == DOC_001_ID
            assert chunk.filename == DOC_001_FILE
            assert chunk.page_number in [1, 2, 3]
            assert chunk.metadata["day40_document"] == DOC_001_ID
            assert chunk.metadata["day40_page"] == chunk.page_number


# ===========================================================================
# 3. EMBEDDING PREPARATION STAGE (Section 10)
# ===========================================================================

class TestEmbeddingPreparationStage:
    """Verify embedding preparation: count, identity, content, metadata."""

    def test_embedding_preparation_integrity(self) -> None:
        chunks = _make_chunks()
        prep = prepare_for_embedding(chunks)

        assert prep.is_ready
        assert prep.total_items == 3
        assert prep.document_id == DOC_001_ID

        for rec in prep.items:
            assert rec.document_id == DOC_001_ID
            assert rec.metadata["day40_document"] == DOC_001_ID


# ===========================================================================
# 4. RETRIEVAL & CONTEXT BUILDING STAGES (Sections 11, 15)
# ===========================================================================

class TestRetrievalAndContextStage:
    """Verify VectorSearchResults and context building lineage."""

    def test_retrieval_results_source_integrity(self) -> None:
        chunks = _make_chunks()
        prep = prepare_for_embedding(chunks)
        vsrs = [
            VectorSearchResult(
                chunk_id=r.chunk_id, score=0.95, document_id=r.document_id,
                filename=r.filename, page_number=r.page_number, chunk_index=r.chunk_index,
                content_type=r.content_type, content=r.content, metadata=r.metadata,
            )
            for r in prep.items
        ]
        processed = process_retrieval_results(vsrs, min_score=0.5, max_results=3)
        assert len(processed) == 3
        assert all(r.document_id == DOC_001_ID for r in processed)

        # Context building
        ctx = build_retrieval_context(processed)
        assert isinstance(ctx, str)
        for marker in (DAY40_SECTION_A, DAY40_SECTION_B, DAY40_UNIQUE_MARKER):
            assert marker in ctx


# ===========================================================================
# 5. AGENT & CITATION LINEAGE STAGES (Sections 13, 14)
# ===========================================================================

class TestAgentAndCitationStage:
    """Verify AgentCitation and AgentResponse with correct source lineage."""

    def test_agent_citation_lineage(self) -> None:
        chunks = _make_chunks()
        prep = prepare_for_embedding(chunks)
        vsrs = [
            VectorSearchResult(
                chunk_id=r.chunk_id, score=0.92, document_id=r.document_id,
                filename=r.filename, page_number=r.page_number, chunk_index=r.chunk_index,
                content_type=r.content_type, content=r.content, metadata=r.metadata,
            )
            for r in prep.items
        ]
        processed = process_retrieval_results(vsrs, min_score=0.5, max_results=3)
        citations = [AgentCitation.from_search_result(r) for r in processed]

        assert len(citations) == 3
        assert all(c.document_id == DOC_001_ID for c in citations)
        assert {c.page_number for c in citations} == {1, 2, 3}

        # Verify marker's page is correctly associated (content lives in retrieval, not citation)
        pg3_cit = next(c for c in citations if c.page_number == 3)
        assert pg3_cit.document_id == DOC_001_ID
        assert pg3_cit.chunk_id == f"CHK_{DOC_001_ID}_P003"

    def test_agent_response_final_result(self) -> None:
        chunks = _make_chunks()
        prep = prepare_for_embedding(chunks)
        vsrs = [
            VectorSearchResult(
                chunk_id=r.chunk_id, score=0.88, document_id=r.document_id,
                filename=r.filename, page_number=r.page_number, chunk_index=r.chunk_index,
                content_type=r.content_type, content=r.content, metadata=r.metadata,
            )
            for r in prep.items
        ]
        processed = process_retrieval_results(vsrs, min_score=0.5, max_results=3)
        citations = [AgentCitation.from_search_result(r) for r in processed]
        response = AgentResponse(answer="Final answer", agent_name="Agent", citations=citations)

        assert response.total_citations == 3
        assert response.unique_documents == [DOC_001_ID]


# ===========================================================================
# 6. VISION & VISUAL EVIDENCE STAGES (Sections 16, 17, 18)
# ===========================================================================

class TestVisionAndVisualEvidenceStage:
    """Verify vision stage lineage: VisualEvidence, VisionRequest, VisionResult."""

    def test_visual_evidence_source_identity(self) -> None:
        chunks = _make_chunks()
        prep = prepare_for_embedding(chunks)
        vsrs = [
            VectorSearchResult(
                chunk_id=r.chunk_id, score=0.91, document_id=r.document_id,
                filename=r.filename, page_number=r.page_number, chunk_index=r.chunk_index,
                content_type=r.content_type, content=r.content, metadata=r.metadata,
            )
            for r in prep.items
        ]
        processed = process_retrieval_results(vsrs, min_score=0.5, max_results=3)
        citations = [AgentCitation.from_search_result(r) for r in processed]
        evidence = VisualEvidenceAdapter.adapt_batch(citations)

        assert len(evidence) == 3
        assert all(e.document_id == DOC_001_ID for e in evidence)
        assert all(e.filename == DOC_001_FILE for e in evidence)

        # Vision request creation
        v_req = VisionRequest(query="Analyze document", evidence=evidence)
        assert v_req.total_evidence == 3
        assert v_req.evidence[2].document_id == DOC_001_ID

        # VisionResult preserves provenance
        v_res = VisionResult(
            query=v_req.query, status="success",
            description="Document analyzed successfully.", evidence=v_req.evidence,
        )
        assert v_res.document_id == DOC_001_ID
        assert v_res.filename == DOC_001_FILE

    def test_multimodal_text_and_visual_from_same_document(self) -> None:
        chunks = _make_chunks()
        prep = prepare_for_embedding(chunks)
        # Text evidence: raw content
        text_contents = [r.content for r in prep.items]
        # Visual evidence: via adapter
        vsrs = [
            VectorSearchResult(
                chunk_id=r.chunk_id, score=0.93, document_id=r.document_id,
                filename=r.filename, page_number=r.page_number, chunk_index=r.chunk_index,
                content_type=r.content_type, content=r.content, metadata=r.metadata,
            )
            for r in prep.items
        ]
        processed = process_retrieval_results(vsrs, min_score=0.5, max_results=3)
        citations = [AgentCitation.from_search_result(r) for r in processed]
        visual_evidence = VisualEvidenceAdapter.adapt_batch(citations)

        # Both text and visual must originate from DOC_001_ID
        assert all(DOC_001_ID in t for t in text_contents)
        assert all(e.document_id == DOC_001_ID for e in visual_evidence)


# ===========================================================================
# 7. FULL END-TO-END MARKER TRACE (Section 19)
# ===========================================================================

class TestEndToEndMarkerTrace:
    """DAY40_UNIQUE_MARKER must remain traceable through every supported stage."""

    def test_unique_marker_survives_full_pipeline(self) -> None:
        result = _full_pipeline(DOC_001_ID, DOC_001_FILE, _PAGES)

        # Marker in extracted page content
        assert any(DAY40_UNIQUE_MARKER in c for c in result["page_contents"])

        # Marker in context string
        assert DAY40_UNIQUE_MARKER in result["context"]

        # Citation for page 3 (UNIQUE_MARKER page) must exist and reference correct doc+chunk
        pg3_citations = [c for c in result["response"].citations if c.page_number == 3]
        assert len(pg3_citations) >= 1
        assert all(c.document_id == DOC_001_ID for c in pg3_citations)

        # Vision evidence for page 3 must exist with correct document identity
        pg3_evidence = [e for e in result["v_req"].evidence if e.page_number == 3]
        assert len(pg3_evidence) >= 1
        assert all(e.document_id == DOC_001_ID for e in pg3_evidence)


# ===========================================================================
# 8. CROSS-DOCUMENT NEGATIVE ISOLATION (Section 20)
# ===========================================================================

class TestCrossDocumentNegativeIsolation:
    """DOC-A results must never contain DOC-B markers and vice versa."""

    def test_cross_document_marker_isolation(self) -> None:
        pages_a = [
            (1, "DAY40_DOCUMENT_A_MARKER — Section A content."),
            (2, "DAY40_DOCUMENT_A_MARKER — Section A page 2."),
        ]
        pages_b = [
            (1, "DAY40_DOCUMENT_B_MARKER — Section B content."),
            (2, "DAY40_DOCUMENT_B_MARKER — Section B page 2."),
        ]

        result_a = _full_pipeline("DAY40-DOC-A", "doc_a.pdf", pages_a)
        result_b = _full_pipeline("DAY40-DOC-B", "doc_b.pdf", pages_b)

        assert result_a["prep_doc_id"] == "DAY40-DOC-A"
        assert result_b["prep_doc_id"] == "DAY40-DOC-B"

        # No DOC-B marker in DOC-A results
        assert not any("DAY40_DOCUMENT_B_MARKER" in c for c in result_a["page_contents"])
        # No DOC-A marker in DOC-B results
        assert not any("DAY40_DOCUMENT_A_MARKER" in c for c in result_b["page_contents"])

        assert all(cid == "DAY40-DOC-A" for cid in result_a["citation_doc_ids"])
        assert all(cid == "DAY40-DOC-B" for cid in result_b["citation_doc_ids"])


# ===========================================================================
# 9. SERIALIZATION ROUND TRIPS (Section 21)
# ===========================================================================

class TestSerializationRoundTrips:
    """Verify that all supported models survive to_dict/from_dict with lineage intact."""

    def test_document_chunk_field_lineage(self) -> None:
        # DocumentChunk does not expose to_dict; verify lineage via direct field access
        chunk = _make_chunks()[2]  # Page 3 with UNIQUE_MARKER
        assert chunk.document_id == DOC_001_ID
        assert chunk.page_number == 3
        assert DAY40_UNIQUE_MARKER in chunk.content
        assert chunk.chunk_id == f"CHK_{DOC_001_ID}_P003"
        assert chunk.metadata["day40_page"] == 3

    def test_agent_citation_serialization(self) -> None:
        # AgentCitation has no 'content' field; lineage is via document_id, chunk_id, page_number
        cit = AgentCitation(
            document_id=DOC_001_ID, filename=DOC_001_FILE,
            chunk_id="CHK_DAY40_P003", page_number=3, content_type="image",
            metadata={"day40_page": 3},
        )
        d = cit.to_dict()
        restored = AgentCitation.from_dict(d)
        assert restored.document_id == DOC_001_ID
        assert restored.page_number == 3
        assert restored.chunk_id == "CHK_DAY40_P003"
        assert restored.metadata["day40_page"] == 3

    def test_agent_response_serialization(self) -> None:
        result = _full_pipeline(DOC_001_ID, DOC_001_FILE, _PAGES)
        resp = result["response"]
        d = resp.to_dict()
        restored = AgentResponse.from_dict(d)
        assert restored.total_citations == 3
        assert restored.unique_documents == [DOC_001_ID]
        # Page 3 citation (UNIQUE_MARKER page) must survive serialization
        pg3 = [c for c in restored.citations if c.page_number == 3]
        assert len(pg3) == 1
        assert pg3[0].document_id == DOC_001_ID

    def test_visual_evidence_serialization(self) -> None:
        # VisualEvidence has no 'content' field; lineage via document_id, chunk_id, page_number
        ev = VisualEvidence(
            document_id=DOC_001_ID, filename=DOC_001_FILE,
            chunk_id="CHK_DAY40_P003", page_number=3, content_type="image",
            description=f"Evidence for {DAY40_UNIQUE_MARKER} page",
        )
        d = ev.to_dict()
        restored = VisualEvidence.from_dict(d)
        assert restored.document_id == DOC_001_ID
        assert restored.page_number == 3


# ===========================================================================
# 10. ERROR PATH (Section 22)
# ===========================================================================

class TestErrorPath:
    """Controlled invalid inputs should raise correct exceptions without state leakage."""

    def test_invalid_chunk_document_id_raises_expected_exception(self) -> None:
        with pytest.raises((IngestionValidationError, ValueError, AgentValidationError)):
            AgentCitation("", DOC_001_FILE, "CHK_BAD")

    def test_vision_evidence_invalid_content_type_raises(self) -> None:
        cit = AgentCitation(DOC_001_ID, DOC_001_FILE, "CHK_TXT", content_type="text")
        with pytest.raises(VisionEvidenceError):
            VisualEvidenceAdapter.adapt_citation(cit)

    def test_valid_docs_unaffected_after_invalid_input(self) -> None:
        # Trigger error
        try:
            AgentCitation("", "bad.pdf", "CHK_BAD")
        except AgentValidationError:
            pass

        # Valid pipeline must still work
        result = _full_pipeline(DOC_001_ID, DOC_001_FILE, _PAGES)
        assert result["prep_total"] == 3
        assert result["prep_doc_id"] == DOC_001_ID


# ===========================================================================
# 11. REPEATED END-TO-END EXECUTION (Section 23)
# ===========================================================================

class TestRepeatedEndToEndExecution:
    """Three identical runs must produce the same stable lineage and no accumulated state."""

    def test_three_identical_runs_stable_output(self) -> None:
        run_data: list[dict] = []
        for _ in range(3):
            result = _full_pipeline(DOC_001_ID, DOC_001_FILE, _PAGES)
            run_data.append({
                "prep_total": result["prep_total"],
                "prep_doc_id": result["prep_doc_id"],
                "processed_count": result["processed_count"],
                "unique_docs": result["unique_docs"],
                "evidence_count": result["evidence_count"],
                "vision_doc_id": result["vision_doc_id"],
            })

        assert run_data[0] == run_data[1] == run_data[2]


# ===========================================================================
# 12. MULTI-DOCUMENT FULL PIPELINE (Section 24)
# ===========================================================================

class TestMultiDocumentFullPipeline:
    """Process DOC-A, DOC-B, DOC-C through full pipeline; verify correct final result per doc."""

    def test_multi_document_pipeline_isolation(self) -> None:
        docs = [
            ("DAY40-MULTI-A", "multi_a.pdf", [(1, "DAY40_MULTI_A PAGE 1"), (2, "DAY40_MULTI_A PAGE 2")]),
            ("DAY40-MULTI-B", "multi_b.pdf", [(1, "DAY40_MULTI_B PAGE 1"), (2, "DAY40_MULTI_B PAGE 2")]),
            ("DAY40-MULTI-C", "multi_c.pdf", [(1, "DAY40_MULTI_C PAGE 1"), (2, "DAY40_MULTI_C PAGE 2")]),
        ]

        for doc_id, filename, pages in docs:
            result = _full_pipeline(doc_id, filename, pages)
            assert result["prep_doc_id"] == doc_id
            assert result["unique_docs"] == [doc_id]
            assert result["vision_doc_id"] == doc_id
            assert all(cid == doc_id for cid in result["citation_doc_ids"])
            assert all(eid == doc_id for eid in result["evidence_doc_ids"])


# ===========================================================================
# 13. INPUT MUTATION SAFETY (Section 25)
# ===========================================================================

class TestInputMutationSafety:
    """Caller-owned objects (metadata, lists, chunks) must remain unchanged after pipeline calls."""

    def test_caller_metadata_unmutated_through_pipeline(self) -> None:
        caller_meta = {"owner": "caller", "important_flag": True, "count": 42}
        snapshot = copy.deepcopy(caller_meta)

        chunk = DocumentChunk(
            chunk_id="CHK_MUTATION_TEST", chunk_index=0,
            document_id=DOC_001_ID, filename=DOC_001_FILE,
            page_number=1, content=f"{DAY40_UNIQUE_MARKER} mutation safety test",
            content_type="image", metadata=caller_meta,
        )

        prepare_for_embedding([chunk])
        assert caller_meta == snapshot

    def test_caller_chunk_list_unmutated(self) -> None:
        chunks = _make_chunks()
        original_ids = [c.chunk_id for c in chunks]

        normalize_chunks(chunks)

        assert [c.chunk_id for c in chunks] == original_ids
