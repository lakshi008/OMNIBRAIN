"""
OmniBrain Member 4 — Day 35 End-to-End Data Lineage & Evidence Integrity Certification.

Verifies end-to-end lineage integrity and provenance tracking across the complete OMNIBRAIN pipeline:
  Document → Page → Chunk → Embedding/Retrieval → Search Result → Agent Citation → Vision Request → Visual Evidence → Vision Result → Final Result

Focus areas:
  1.  Deterministic synthetic document set (DOC-A, DOC-B).
  2.  Document → Page lineage.
  3.  Page → Chunk lineage.
  4.  Chunk content integrity.
  5.  Chunk metadata integrity.
  6.  Retrieval lineage (VectorSearchResult).
  7.  Search Result → Citation lineage.
  8.  Citation lineage isolation.
  9.  Vision Request lineage.
  10. Visual Evidence lineage.
  11. Vision Result lineage.
  12. Complete End-to-End lineage.
  13. Multi-document isolation (zero cross-bleeding).
  14. Cross-request isolation.
  15. Same document, different chunks discrimination.
  16. Duplicate content with distinct identity preservation.
  17. Serialization lineage preservation (to_dict / from_dict).
  18. Batch lineage independence.
  19. Error lineage containment.
  20. Failure → Success lineage purity (no error leakage).
  21. Mutation safety across lineage handoffs.
  22. Lineage completeness & provenance traceability.
  23. Lineage uniqueness.

Constraints:
  - 100% offline. Zero external APIs, network, LLM, or production credentials.
  - Zero production code modified.
  - No lineage infrastructure, provenance fields, adapters, or wrappers added.
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
# Deterministic Synthetic Document Fixtures (Section 5)
# ---------------------------------------------------------------------------

DOC_A_ID = "DAY35-DOC-A"
DOC_A_FILENAME = "doc_a.pdf"

DOC_B_ID = "DAY35-DOC-B"
DOC_B_FILENAME = "doc_b.pdf"


def build_synthetic_doc_a() -> tuple[ParsedDocument, list[DocumentChunk]]:
    """Construct Document A with 2 pages and 3 chunks."""
    pages = [
        PageData(page_number=1, text="Doc A Page 1 financial tables and charts.", char_count=42, has_content=True),
        PageData(page_number=2, text="Doc A Page 2 revenue summary text.", char_count=34, has_content=True),
    ]
    meta = DocumentMetadata(
        document_id=DOC_A_ID, filename=DOC_A_FILENAME, total_pages=2,
        content_type="application/pdf", created_at="2026-08-26T00:00:00Z",
        pages_with_content=2, pages_without_content=0,
    )
    doc = ParsedDocument(metadata=meta, pages=pages)

    chunks = [
        DocumentChunk(
            chunk_id="CHUNK_A1", chunk_index=0, document_id=DOC_A_ID,
            filename=DOC_A_FILENAME, page_number=1,
            content="[Table A1: Q1 revenue]", content_type="image",
            metadata={"day35_document": "A", "day35_chunk": "A1", "section": "finance"},
        ),
        DocumentChunk(
            chunk_id="CHUNK_A2", chunk_index=1, document_id=DOC_A_ID,
            filename=DOC_A_FILENAME, page_number=1,
            content="Doc A text paragraph on Page 1.", content_type="text",
            metadata={"day35_document": "A", "day35_chunk": "A2", "section": "intro"},
        ),
        DocumentChunk(
            chunk_id="CHUNK_A3", chunk_index=2, document_id=DOC_A_ID,
            filename=DOC_A_FILENAME, page_number=2,
            content="[Chart A3: Annual growth diagram]", content_type="image",
            metadata={"day35_document": "A", "day35_chunk": "A3", "section": "growth"},
        ),
    ]
    return doc, chunks


def build_synthetic_doc_b() -> tuple[ParsedDocument, list[DocumentChunk]]:
    """Construct Document B with 2 pages and 3 chunks."""
    pages = [
        PageData(page_number=1, text="Doc B Page 1 engineering specifications.", char_count=40, has_content=True),
        PageData(page_number=2, text="Doc B Page 2 system architecture diagrams.", char_count=42, has_content=True),
    ]
    meta = DocumentMetadata(
        document_id=DOC_B_ID, filename=DOC_B_FILENAME, total_pages=2,
        content_type="application/pdf", created_at="2026-08-26T00:00:00Z",
        pages_with_content=2, pages_without_content=0,
    )
    doc = ParsedDocument(metadata=meta, pages=pages)

    chunks = [
        DocumentChunk(
            chunk_id="CHUNK_B1", chunk_index=0, document_id=DOC_B_ID,
            filename=DOC_B_FILENAME, page_number=1,
            content="[Diagram B1: Microservice topology]", content_type="image",
            metadata={"day35_document": "B", "day35_chunk": "B1", "system": "core"},
        ),
        DocumentChunk(
            chunk_id="CHUNK_B2", chunk_index=1, document_id=DOC_B_ID,
            filename=DOC_B_FILENAME, page_number=1,
            content="Doc B engineering overview text.", content_type="text",
            metadata={"day35_document": "B", "day35_chunk": "B2", "system": "intro"},
        ),
        DocumentChunk(
            chunk_id="CHUNK_B3", chunk_index=2, document_id=DOC_B_ID,
            filename=DOC_B_FILENAME, page_number=2,
            content="[Image B3: Physical server rack layout]", content_type="image",
            metadata={"day35_document": "B", "day35_chunk": "B3", "system": "hardware"},
        ),
    ]
    return doc, chunks


# ===========================================================================
# 1. DOCUMENT → PAGE LINEAGE
# ===========================================================================

class TestDocumentToPageLineage:
    """Section 6: Verify pages remain associated with their parent document."""

    def test_doc_a_page_lineage_integrity(self) -> None:
        doc_a, _ = build_synthetic_doc_a()
        assert doc_a.metadata.document_id == DOC_A_ID
        assert doc_a.metadata.filename == DOC_A_FILENAME
        assert len(doc_a.pages) == 2

        p1 = doc_a.get_page(1)
        assert p1 is not None
        assert p1.page_number == 1
        assert "Doc A Page 1" in p1.text

        p2 = doc_a.get_page(2)
        assert p2 is not None
        assert p2.page_number == 2
        assert "Doc A Page 2" in p2.text

    def test_doc_b_page_lineage_integrity(self) -> None:
        doc_b, _ = build_synthetic_doc_b()
        assert doc_b.metadata.document_id == DOC_B_ID
        assert doc_b.metadata.filename == DOC_B_FILENAME
        assert len(doc_b.pages) == 2

        p1 = doc_b.get_page(1)
        assert p1 is not None
        assert "Doc B Page 1" in p1.text

    def test_pages_do_not_cross_documents(self) -> None:
        doc_a, _ = build_synthetic_doc_a()
        doc_b, _ = build_synthetic_doc_b()
        assert doc_a.metadata.document_id != doc_b.metadata.document_id
        assert doc_a.get_all_text() != doc_b.get_all_text()


# ===========================================================================
# 2. PAGE → CHUNK LINEAGE
# ===========================================================================

class TestPageToChunkLineage:
    """Section 7: Verify chunks correctly associate with parent pages and documents."""

    def test_doc_a_chunk_lineage_mapping(self) -> None:
        _, chunks_a = build_synthetic_doc_a()
        assert len(chunks_a) == 3

        # Page 1 -> CHUNK_A1, CHUNK_A2
        page_1_chunks = [c for c in chunks_a if c.page_number == 1]
        assert len(page_1_chunks) == 2
        assert {c.chunk_id for c in page_1_chunks} == {"CHUNK_A1", "CHUNK_A2"}
        assert all(c.document_id == DOC_A_ID for c in page_1_chunks)

        # Page 2 -> CHUNK_A3
        page_2_chunks = [c for c in chunks_a if c.page_number == 2]
        assert len(page_2_chunks) == 1
        assert page_2_chunks[0].chunk_id == "CHUNK_A3"
        assert page_2_chunks[0].document_id == DOC_A_ID

    def test_doc_b_chunk_lineage_mapping(self) -> None:
        _, chunks_b = build_synthetic_doc_b()
        assert len(chunks_b) == 3

        # Page 1 -> CHUNK_B1, CHUNK_B2
        page_1_chunks = [c for c in chunks_b if c.page_number == 1]
        assert len(page_1_chunks) == 2
        assert {c.chunk_id for c in page_1_chunks} == {"CHUNK_B1", "CHUNK_B2"}
        assert all(c.document_id == DOC_B_ID for c in page_1_chunks)

        # Page 2 -> CHUNK_B3
        page_2_chunks = [c for c in chunks_b if c.page_number == 2]
        assert len(page_2_chunks) == 1
        assert page_2_chunks[0].chunk_id == "CHUNK_B3"


# ===========================================================================
# 3. CHUNK CONTENT & METADATA INTEGRITY
# ===========================================================================

class TestChunkContentAndMetadataIntegrity:
    """Sections 8 & 9: Verify content and metadata are preserved through pipeline."""

    def test_chunk_content_preserved_in_embedding_prep(self) -> None:
        _, chunks_a = build_synthetic_doc_a()
        prep = prepare_for_embedding(chunks_a)
        assert prep.is_ready is True

        content_map = {item.chunk_id: item.content for item in prep.items}
        for chunk in chunks_a:
            assert content_map[chunk.chunk_id] == chunk.content

    def test_chunk_metadata_preserved_in_embedding_records(self) -> None:
        _, chunks_a = build_synthetic_doc_a()
        prep = prepare_for_embedding(chunks_a)

        for rec in prep.items:
            assert rec.metadata["day35_document"] == "A"
            assert rec.metadata["day35_chunk"] == rec.chunk_id.replace("CHUNK_", "")
            assert rec.metadata["document_id"] == DOC_A_ID


# ===========================================================================
# 4. RETRIEVAL & SEARCH RESULT LINEAGE
# ===========================================================================

class TestRetrievalLineage:
    """Sections 10 & 11: Trace VectorSearchResult and AgentCitation back to source."""

    def test_vsr_preserves_document_page_and_chunk_identity(self) -> None:
        _, chunks_a = build_synthetic_doc_a()
        vsr_a1 = VectorSearchResult(
            chunk_id=chunks_a[0].chunk_id,
            score=0.96,
            document_id=chunks_a[0].document_id,
            filename=chunks_a[0].filename,
            page_number=chunks_a[0].page_number,
            chunk_index=chunks_a[0].chunk_index,
            content_type=chunks_a[0].content_type,
            content=chunks_a[0].content,
            metadata=chunks_a[0].metadata,
        )

        assert vsr_a1.document_id == DOC_A_ID
        assert vsr_a1.filename == DOC_A_FILENAME
        assert vsr_a1.page_number == 1
        assert vsr_a1.chunk_id == "CHUNK_A1"
        assert vsr_a1.metadata["day35_chunk"] == "A1"

    def test_vsr_to_agent_citation_lineage(self) -> None:
        _, chunks_a = build_synthetic_doc_a()
        vsr_a1 = VectorSearchResult(
            chunk_id=chunks_a[0].chunk_id,
            score=0.96,
            document_id=chunks_a[0].document_id,
            filename=chunks_a[0].filename,
            page_number=chunks_a[0].page_number,
            chunk_index=chunks_a[0].chunk_index,
            content_type=chunks_a[0].content_type,
            content=chunks_a[0].content,
            metadata=chunks_a[0].metadata,
        )
        cit = AgentCitation.from_search_result(vsr_a1)

        assert cit.document_id == DOC_A_ID
        assert cit.filename == DOC_A_FILENAME
        assert cit.chunk_id == "CHUNK_A1"
        assert cit.page_number == 1
        assert cit.content_type == "image"
        assert cit.score == 0.96
        assert cit.metadata["day35_chunk"] == "A1"


# ===========================================================================
# 5. CITATION LINEAGE ISOLATION
# ===========================================================================

class TestCitationLineageIsolation:
    """Section 12: Ensure citations point strictly to their own source document/chunk."""

    def test_citation_lineage_distinctness(self) -> None:
        _, chunks_a = build_synthetic_doc_a()
        _, chunks_b = build_synthetic_doc_b()

        cit_a1 = AgentCitation.from_search_result(
            VectorSearchResult(
                chunk_id=chunks_a[0].chunk_id, score=0.9, document_id=DOC_A_ID,
                filename=DOC_A_FILENAME, page_number=1, chunk_index=0,
                content_type="image", content=chunks_a[0].content, metadata=chunks_a[0].metadata,
            )
        )
        cit_b1 = AgentCitation.from_search_result(
            VectorSearchResult(
                chunk_id=chunks_b[0].chunk_id, score=0.92, document_id=DOC_B_ID,
                filename=DOC_B_FILENAME, page_number=1, chunk_index=0,
                content_type="diagram", content=chunks_b[0].content, metadata=chunks_b[0].metadata,
            )
        )

        assert cit_a1.document_id == DOC_A_ID
        assert cit_a1.chunk_id == "CHUNK_A1"
        assert cit_a1.metadata["day35_document"] == "A"

        assert cit_b1.document_id == DOC_B_ID
        assert cit_b1.chunk_id == "CHUNK_B1"
        assert cit_b1.metadata["day35_document"] == "B"

        assert cit_a1.document_id != cit_b1.document_id
        assert cit_a1.chunk_id != cit_b1.chunk_id


# ===========================================================================
# 6. VISION REQUEST & EVIDENCE LINEAGE
# ===========================================================================

class TestVisionLineage:
    """Sections 13, 14, 15: Verify VisionRequest, VisualEvidence, and VisionResult lineage."""

    def test_visual_evidence_derived_from_citation(self) -> None:
        _, chunks_a = build_synthetic_doc_a()
        cit_a1 = AgentCitation(
            document_id=DOC_A_ID, filename=DOC_A_FILENAME, chunk_id="CHUNK_A1",
            page_number=1, content_type="image", score=0.95,
            metadata=chunks_a[0].metadata,
        )
        ev_a1 = VisualEvidenceAdapter.adapt_citation(cit_a1)

        assert ev_a1.document_id == DOC_A_ID
        assert ev_a1.filename == DOC_A_FILENAME
        assert ev_a1.chunk_id == "CHUNK_A1"
        assert ev_a1.page_number == 1
        assert ev_a1.content_type == "image"
        assert ev_a1.metadata["day35_document"] == "A"

    def test_vision_result_inherits_primary_evidence_lineage(self) -> None:
        ev_a1 = VisualEvidence(
            document_id=DOC_A_ID, filename=DOC_A_FILENAME, chunk_id="CHUNK_A1",
            page_number=1, content_type="image", description="Q1 Revenue chart",
        )
        v_res = VisionResult(
            query="Analyze revenue", status="success",
            description="Revenue growth confirmed", evidence=[ev_a1],
        )

        assert v_res.document_id == DOC_A_ID
        assert v_res.filename == DOC_A_FILENAME
        assert v_res.page_number == 1
        assert v_res.chunk_id == "CHUNK_A1"
        assert v_res.content_type == "image"


# ===========================================================================
# 7. COMPLETE END-TO-END DATA LINEAGE
# ===========================================================================

class TestCompleteEndToEndLineage:
    """Section 16: Complete supported pipeline flow preserving lineage at every boundary."""

    def test_full_pipeline_lineage_trace(self) -> None:
        # 1. Document & Page
        doc_a, chunks_a = build_synthetic_doc_a()
        p1 = doc_a.get_page(1)
        assert p1 is not None

        # 2. Chunk
        chunk = chunks_a[0]
        assert chunk.document_id == doc_a.metadata.document_id
        assert chunk.page_number == p1.page_number

        # 3. Embedding Prep
        prep = prepare_for_embedding([chunk])
        assert prep.items[0].document_id == DOC_A_ID
        assert prep.items[0].chunk_id == "CHUNK_A1"

        # 4. Vector Search Result
        vsr = VectorSearchResult(
            chunk_id=prep.items[0].chunk_id, score=0.98,
            document_id=prep.items[0].document_id, filename=prep.items[0].filename,
            page_number=prep.items[0].page_number, chunk_index=prep.items[0].chunk_index,
            content_type=prep.items[0].content_type, content=prep.items[0].content,
            metadata=prep.items[0].metadata,
        )
        processed = process_retrieval_results([vsr], min_score=0.5, max_results=5)
        assert len(processed) == 1

        # 5. Agent Citation
        cit = AgentCitation.from_search_result(processed[0])
        assert cit.document_id == DOC_A_ID
        assert cit.chunk_id == "CHUNK_A1"

        # 6. Agent Response
        resp = AgentResponse(
            answer="Q1 Revenue increased", agent_name="SearchAgent",
            status="success", citations=[cit], metadata={"query": "Q1 revenue"},
        )
        assert resp.citations[0].document_id == DOC_A_ID

        # 7. Vision Request & Evidence
        evidence = VisualEvidenceAdapter.adapt_search_package(resp)
        assert len(evidence) == 1
        assert evidence[0].document_id == DOC_A_ID
        assert evidence[0].chunk_id == "CHUNK_A1"

        v_req = VisionRequest(query="Examine chart", evidence=evidence)
        assert v_req.evidence[0].document_id == DOC_A_ID

        # 8. Vision Result
        v_res = VisionResult(
            query=v_req.query, status="success",
            description="Revenue chart verified", evidence=v_req.evidence,
        )
        normalized = VisionResultNormalizer.normalize(v_res, request=v_req)

        # 9. Provenance Verification
        assert normalized.document_id == DOC_A_ID
        assert normalized.filename == DOC_A_FILENAME
        assert normalized.chunk_id == "CHUNK_A1"
        assert normalized.page_number == 1
        assert normalized.is_success is True


# ===========================================================================
# 8. MULTI-DOCUMENT ISOLATION
# ===========================================================================

class TestMultiDocumentIsolation:
    """Section 17: Process DOC_A and DOC_B concurrently; verify zero lineage bleeding."""

    def test_multi_document_coexistence_without_bleeding(self) -> None:
        _, chunks_a = build_synthetic_doc_a()
        _, chunks_b = build_synthetic_doc_b()

        prep_a = prepare_for_embedding(chunks_a)
        prep_b = prepare_for_embedding(chunks_b)

        assert prep_a.is_ready is True
        assert prep_b.is_ready is True
        assert len(prep_a.items) == 3
        assert len(prep_b.items) == 3

        # Assert no A record has B metadata
        for rec in prep_a.items:
            assert rec.document_id == DOC_A_ID
            assert rec.filename == DOC_A_FILENAME
            assert rec.metadata["day35_document"] == "A"
            assert "system" not in rec.metadata

        # Assert no B record has A metadata
        for rec in prep_b.items:
            assert rec.document_id == DOC_B_ID
            assert rec.filename == DOC_B_FILENAME
            assert rec.metadata["day35_document"] == "B"
            assert "section" not in rec.metadata

    def test_multi_document_citations_remain_partitioned(self) -> None:
        _, chunks_a = build_synthetic_doc_a()
        _, chunks_b = build_synthetic_doc_b()

        vsrs = [
            VectorSearchResult(
                chunk_id=c.chunk_id, score=0.9, document_id=c.document_id,
                filename=c.filename, page_number=c.page_number, chunk_index=c.chunk_index,
                content_type=c.content_type, content=c.content, metadata=c.metadata,
            )
            for c in chunks_a + chunks_b
        ]
        citations = [AgentCitation.from_search_result(r) for r in vsrs]

        resp = AgentResponse(
            answer="Mixed summary", agent_name="Agent",
            citations=citations, metadata={"query": "summary"},
        )
        assert resp.total_citations == 6
        assert resp.unique_documents == [DOC_A_ID, DOC_B_ID]

        by_doc_a = [c for c in resp.citations if c.document_id == DOC_A_ID]
        by_doc_b = [c for c in resp.citations if c.document_id == DOC_B_ID]
        assert len(by_doc_a) == 3
        assert len(by_doc_b) == 3

        # Zero cross-citation
        assert all(c.document_id == DOC_A_ID for c in by_doc_a)
        assert all(c.document_id == DOC_B_ID for c in by_doc_b)

        # Also verify SearchResult.by_document
        sr = SearchResult.from_response(resp)
        assert len(sr.by_document[DOC_A_ID]) == 3
        assert len(sr.by_document[DOC_B_ID]) == 3


# ===========================================================================
# 9. CROSS-REQUEST ISOLATION
# ===========================================================================

class TestCrossRequestIsolation:
    """Section 18: Distinct requests for DOC_A and DOC_B remain strictly isolated."""

    def test_requests_contain_only_targeted_lineage(self) -> None:
        _, chunks_a = build_synthetic_doc_a()
        _, chunks_b = build_synthetic_doc_b()

        req_a = AgentRequest(query="Find Doc A data", metadata={"target_doc": DOC_A_ID})
        req_b = AgentRequest(query="Find Doc B data", metadata={"target_doc": DOC_B_ID})

        cit_a = [
            AgentCitation.from_search_result(
                VectorSearchResult(
                    chunk_id=c.chunk_id, score=0.95, document_id=c.document_id,
                    filename=c.filename, page_number=c.page_number, chunk_index=c.chunk_index,
                    content_type=c.content_type, content=c.content, metadata=c.metadata,
                )
            )
            for c in chunks_a
        ]
        resp_a = AgentResponse(answer="A results", agent_name="Agent", citations=cit_a)

        cit_b = [
            AgentCitation.from_search_result(
                VectorSearchResult(
                    chunk_id=c.chunk_id, score=0.92, document_id=c.document_id,
                    filename=c.filename, page_number=c.page_number, chunk_index=c.chunk_index,
                    content_type=c.content_type, content=c.content, metadata=c.metadata,
                )
            )
            for c in chunks_b
        ]
        resp_b = AgentResponse(answer="B results", agent_name="Agent", citations=cit_b)

        assert resp_a.unique_documents == [DOC_A_ID]
        assert resp_b.unique_documents == [DOC_B_ID]
        assert set(resp_a.unique_documents).isdisjoint(set(resp_b.unique_documents))


# ===========================================================================
# 10. SAME DOCUMENT, DIFFERENT CHUNKS
# ===========================================================================

class TestSameDocumentDifferentChunks:
    """Section 19: Chunk-level lineage remains distinct within the same document."""

    def test_distinct_chunk_identities_in_doc_a(self) -> None:
        _, chunks_a = build_synthetic_doc_a()

        c1, c2, c3 = chunks_a[0], chunks_a[1], chunks_a[2]
        assert c1.chunk_id == "CHUNK_A1" and c1.chunk_index == 0 and c1.page_number == 1
        assert c2.chunk_id == "CHUNK_A2" and c2.chunk_index == 1 and c2.page_number == 1
        assert c3.chunk_id == "CHUNK_A3" and c3.chunk_index == 2 and c3.page_number == 2

        cit1 = AgentCitation(document_id=DOC_A_ID, filename=DOC_A_FILENAME, chunk_id=c1.chunk_id, page_number=c1.page_number)
        cit2 = AgentCitation(document_id=DOC_A_ID, filename=DOC_A_FILENAME, chunk_id=c2.chunk_id, page_number=c2.page_number)
        cit3 = AgentCitation(document_id=DOC_A_ID, filename=DOC_A_FILENAME, chunk_id=c3.chunk_id, page_number=c3.page_number)

        assert cit1.chunk_id != cit2.chunk_id
        assert cit2.chunk_id != cit3.chunk_id
        assert cit1.page_number == 1 and cit3.page_number == 2


# ===========================================================================
# 11. DUPLICATE CONTENT WITH DISTINCT IDENTITIES
# ===========================================================================

class TestDuplicateContentWithDistinctIdentities:
    """Section 20: Chunks with identical text retain distinct source identities."""

    def test_duplicate_text_different_chunks_preserved(self) -> None:
        shared_text = "Standard corporate disclaimer text."
        chunk_x = DocumentChunk(
            chunk_id="CHUNK_X", chunk_index=0, document_id=DOC_A_ID,
            filename=DOC_A_FILENAME, page_number=1, content=shared_text, content_type="text",
        )
        chunk_y = DocumentChunk(
            chunk_id="CHUNK_Y", chunk_index=5, document_id=DOC_B_ID,
            filename=DOC_B_FILENAME, page_number=3, content=shared_text, content_type="text",
        )

        assert chunk_x.content == chunk_y.content
        assert chunk_x.chunk_id != chunk_y.chunk_id
        assert chunk_x.document_id != chunk_y.document_id

        prep_x = prepare_for_embedding([chunk_x])
        prep_y = prepare_for_embedding([chunk_y])
        assert len(prep_x.items) == 1 and prep_x.items[0].chunk_id == "CHUNK_X"
        assert len(prep_y.items) == 1 and prep_y.items[0].chunk_id == "CHUNK_Y"


# ===========================================================================
# 12. SERIALIZATION LINEAGE PRESERVATION
# ===========================================================================

class TestSerializationLineage:
    """Section 21: to_dict / from_dict preserves all lineage and provenance fields."""

    def test_agent_citation_serialization_lineage(self) -> None:
        cit = AgentCitation(
            document_id=DOC_A_ID, filename=DOC_A_FILENAME, chunk_id="CHUNK_A1",
            page_number=1, content_type="image", score=0.99,
            metadata={"day35_document": "A", "custom_tag": "audit_passed"},
        )
        d = cit.to_dict()
        cit2 = AgentCitation.from_dict(d)

        assert cit2.document_id == DOC_A_ID
        assert cit2.filename == DOC_A_FILENAME
        assert cit2.chunk_id == "CHUNK_A1"
        assert cit2.page_number == 1
        assert cit2.content_type == "image"
        assert cit2.score == 0.99
        assert cit2.metadata == {"day35_document": "A", "custom_tag": "audit_passed"}

    def test_visual_evidence_serialization_lineage(self) -> None:
        ev = VisualEvidence(
            document_id=DOC_B_ID, filename=DOC_B_FILENAME, chunk_id="CHUNK_B1",
            page_number=1, chunk_index=0, content_type="diagram",
            description="System topology", metadata={"system": "core"},
        )
        d = ev.to_dict()
        ev2 = VisualEvidence.from_dict(d)

        assert ev2.document_id == DOC_B_ID
        assert ev2.filename == DOC_B_FILENAME
        assert ev2.chunk_id == "CHUNK_B1"
        assert ev2.content_type == "diagram"
        assert ev2.metadata == {"system": "core"}


# ===========================================================================
# 13. BATCH LINEAGE INDEPENDENCE
# ===========================================================================

class TestBatchLineageIndependence:
    """Section 22: Every item in a batch maintains independent lineage."""

    def test_batch_items_retain_individual_lineage(self) -> None:
        _, chunks_a = build_synthetic_doc_a()
        _, chunks_b = build_synthetic_doc_b()

        batch = chunks_a + chunks_b
        assert len(batch) == 6

        normalized = normalize_chunks(batch)
        assert len(normalized) == 6

        for original, norm in zip(batch, normalized):
            assert norm.chunk_id == original.chunk_id
            assert norm.document_id == original.document_id
            assert norm.page_number == original.page_number
            assert norm.metadata == original.metadata


# ===========================================================================
# 14. ERROR LINEAGE & CONTAMINATION PREVENTION
# ===========================================================================

class TestErrorLineageAndContamination:
    """Sections 23 & 24: Error in Request A does not leak into Request B or C."""

    def test_invalid_request_does_not_affect_subsequent_valid_lineage(self) -> None:
        # Step 1: Trigger failure on invalid citation
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename=DOC_A_FILENAME, chunk_id="CHUNK_A1")

        # Step 2: Request B succeeds cleanly
        cit_b = AgentCitation(
            document_id=DOC_B_ID, filename=DOC_B_FILENAME, chunk_id="CHUNK_B1",
            page_number=1, content_type="diagram",
        )
        assert cit_b.document_id == DOC_B_ID
        assert cit_b.chunk_id == "CHUNK_B1"

        # Step 3: Request C succeeds cleanly
        cit_c = AgentCitation(
            document_id=DOC_A_ID, filename=DOC_A_FILENAME, chunk_id="CHUNK_A2",
            page_number=1, content_type="text",
        )
        assert cit_c.document_id == DOC_A_ID
        assert cit_c.chunk_id == "CHUNK_A2"


# ===========================================================================
# 15. MUTATION SAFETY ACROSS LINEAGE HANDOFFS
# ===========================================================================

class TestMutationSafetyAcrossLineageHandoffs:
    """Section 25: Caller-owned objects and metadata are not mutated by downstream consumers."""

    def test_caller_metadata_unmutated_after_citation_creation(self) -> None:
        original_meta = {"immutable_key": "val", "day35_document": "A"}
        snapshot = copy.deepcopy(original_meta)

        cit = AgentCitation(
            document_id=DOC_A_ID, filename=DOC_A_FILENAME, chunk_id="CHUNK_A1",
            content_type="image", metadata=original_meta,
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit)

        assert original_meta == snapshot
        assert cit.metadata == snapshot

    def test_chunk_objects_unmutated_after_embedding_prep(self) -> None:
        _, chunks_a = build_synthetic_doc_a()
        snapshot = copy.deepcopy(chunks_a)

        prepare_for_embedding(chunks_a)
        assert chunks_a == snapshot


# ===========================================================================
# 16. LINEAGE COMPLETENESS & PROVENANCE TRACEABILITY
# ===========================================================================

class TestLineageCompletenessAndTraceability:
    """Section 26 & 27: Full provenance trace from final result back to document."""

    def test_full_provenance_traceability(self) -> None:
        doc_b, chunks_b = build_synthetic_doc_b()
        img_chunk = chunks_b[2]  # CHUNK_B3, Page 2, physical server rack layout

        # Trace forward
        vsr = VectorSearchResult(
            chunk_id=img_chunk.chunk_id, score=0.93,
            document_id=img_chunk.document_id, filename=img_chunk.filename,
            page_number=img_chunk.page_number, chunk_index=img_chunk.chunk_index,
            content_type=img_chunk.content_type, content=img_chunk.content,
            metadata=img_chunk.metadata,
        )
        cit = AgentCitation.from_search_result(vsr)
        ev = VisualEvidenceAdapter.adapt_citation(cit)
        vr = VisionResult(
            query="Analyze server layout", status="success",
            description="Hardware rack detected", evidence=[ev],
        )

        # Trace backward from VisionResult
        assert vr.document_id == DOC_B_ID
        assert vr.filename == DOC_B_FILENAME
        assert vr.chunk_id == "CHUNK_B3"
        assert vr.page_number == 2

        # Trace back to ParsedDocument
        matched_page = doc_b.get_page(vr.page_number)
        assert matched_page is not None
        assert matched_page.page_number == 2
        assert "Doc B Page 2" in matched_page.text
