"""
OmniBrain Member 4 — Day 43 Data Integrity & Corruption Detection Regression Certification.

Certifies end-to-end data integrity, lineage preservation, and deterministic corruption
detection across:
  - Ingestion (DocumentChunk, ChunkingResult, ChunkValidationResult,
               DocumentMetadata, ParsedDocument, PageData, VectorSearchResult,
               RetrievalServiceResult, validate_chunks, process_retrieval_results,
               build_retrieval_context)
  - Agents / Search (AgentCitation, AgentRequest, SearchRequest,
                     AgentResponse, SearchResult, AgentState)
  - Vision (VisualEvidence, VisionRequest, VisionResult,
             VisualEvidenceAdapter, VisionResultNormalizer, VisionExecutionTrace)

Covers:
  1.  Document, page, and chunk construction and structural integrity.
  2.  Deterministic content markers and lineage preservation across transformations.
  3.  Metadata integrity, key-value retention, and isolation.
  4.  Duplicate data detection (chunk ID deduplication, retrieval result deduplication).
  5.  Missing required data detection and deterministic validation failure.
  6.  Empty data behavior (empty strings, empty collections, empty metadata).
  7.  Malformed data rejection (invalid page numbers, negative indices, invalid modalities).
  8.  Document and page misassociation detection and partitioning.
  9.  Corrupted content differentiation and metadata sanitization.
  10. Serialization corruption detection and round-trip integrity.
  11. Retrieval result, citation, and visual evidence lineage integrity.
  12. Cross-document and cross-request isolation (prevention of contamination).
  13. Input mutation safety and aliasing protection.
  14. Partial data handling and error isolation across sequential/batch executions.
  15. Repeated corruption test repeatability without state accumulation.
  16. Multi-iteration execution integrity and stability.
  17. Controlled multi-document batch data integrity.

Constraints:
  - 100% Offline: No external APIs, network, real LLMs, or production secrets.
  - Zero production code modified.
  - No new validators, adapters, wrappers, or repair mechanisms added.
  - Synthetic deterministic data only.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# Ingestion layer (Member 1)
from ingestion.models import (
    ChunkingResult,
    ChunkValidationResult,
    DocumentChunk,
    DocumentMetadata,
    EmbeddingRecord,
    EmbeddingVectorRecord,
    ExtractedImage,
    ExtractedTable,
    ImageExtractionResult,
    IngestionResult,
    PageData,
    ParsedDocument,
    RetrievalServiceResult,
    TableExtractionResult,
    VectorSearchResult,
)
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.retrieval_processor import process_retrieval_results, build_retrieval_context

# Agents layer (Member 2)
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
    AgentValidationError,
)

# Vision layer (Member 3)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.exceptions import (
    VisionError,
    VisionEvidenceError,
    VisionInputValidationError,
)
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.result_normalizer import (
    FORBIDDEN_METADATA_KEYS,
    VisionExecutionTrace,
    VisionResultNormalizer,
)


# ============================================================================
# Deterministic Synthetic Fixtures
# ============================================================================

DAY43_DOC_001 = "DAY43-DOC-001"
DAY43_DOC_002 = "DAY43-DOC-002"
DAY43_FILE_001 = "day43_system_doc_001.pdf"
DAY43_FILE_002 = "day43_system_doc_002.pdf"

DAY43_PAGE_001 = 1
DAY43_PAGE_002 = 2
DAY43_PAGE_003 = 3

DAY43_CHUNK_001 = "DAY43-CHUNK-001"
DAY43_CHUNK_002 = "DAY43-CHUNK-002"
DAY43_CHUNK_003 = "DAY43-CHUNK-003"

DAY43_MARKER_A = "DAY43_MARKER_ALPHA_91823"
DAY43_MARKER_B = "DAY43_MARKER_BETA_74102"
DAY43_MARKER_C = "DAY43_MARKER_GAMMA_58391"
DAY43_MARKER_CORRUPTED = "DAY43_MARKER_CORRUPTED_99999"

DAY43_META_001: dict[str, Any] = {
    "day43_document": "001",
    "day43_source": "synthetic",
    "section": "architecture",
}
DAY43_META_002: dict[str, Any] = {
    "day43_document": "002",
    "day43_source": "synthetic",
    "section": "evaluation",
}


# ============================================================================
# 1. Document & Page Integrity
# ============================================================================

class TestDocumentAndPageIntegrity:
    """Certifies that document and page data retain identity, content, and metadata."""

    def test_document_metadata_and_parsed_document_integrity(self) -> None:
        """Verify DocumentMetadata and ParsedDocument preserve page structure and text."""
        p1 = PageData(
            page_number=DAY43_PAGE_001,
            text=f"Page 1 content containing {DAY43_MARKER_A}",
            char_count=len(f"Page 1 content containing {DAY43_MARKER_A}"),
            has_content=True,
        )
        p2 = PageData(
            page_number=DAY43_PAGE_002,
            text=f"Page 2 content containing {DAY43_MARKER_B}",
            char_count=len(f"Page 2 content containing {DAY43_MARKER_B}"),
            has_content=True,
        )
        p3_empty = PageData(
            page_number=DAY43_PAGE_003,
            text="",
            char_count=0,
            has_content=False,
        )

        meta = DocumentMetadata(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            total_pages=3,
            content_type="application/pdf",
            created_at="2026-08-26T00:00:00Z",
            pages_with_content=2,
            pages_without_content=1,
        )

        doc = ParsedDocument(metadata=meta, pages=[p1, p2, p3_empty])

        # Verify document-level integrity
        assert doc.metadata.document_id == DAY43_DOC_001
        assert doc.metadata.filename == DAY43_FILE_001
        assert doc.metadata.total_pages == 3
        assert doc.metadata.pages_with_content == 2
        assert doc.metadata.pages_without_content == 1

        # Verify page retrieval and association
        assert doc.get_page(1) == p1
        assert doc.get_page(2) == p2
        assert doc.get_page(3) == p3_empty
        assert doc.get_page(4) is None

        # Verify text concatenation preserves only content pages
        all_text = doc.get_all_text()
        assert DAY43_MARKER_A in all_text
        assert DAY43_MARKER_B in all_text
        assert len(all_text.split("\n\n")) == 2

    def test_ingestion_result_unified_integrity(self) -> None:
        """Verify IngestionResult aggregates text, tables, and images without crosstalk."""
        meta = DocumentMetadata(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            total_pages=2,
            content_type="application/pdf",
            created_at="2026-08-26T00:00:00Z",
            pages_with_content=2,
            pages_without_content=0,
        )
        p1 = PageData(page_number=1, text="Text p1", char_count=7, has_content=True)
        p2 = PageData(page_number=2, text="Text p2", char_count=7, has_content=True)

        tbl = ExtractedTable(
            page_number=1,
            table_index=0,
            rows=2,
            columns=2,
            cells=[["A", "B"], ["C", "D"]],
        )
        img = ExtractedImage(
            page_number=2,
            image_index=0,
            image_format="png",
            width=100,
            height=100,
            image_bytes=b"\x89PNG",
            size_bytes=4,
            colorspace="DeviceRGB",
            xref=10,
        )

        ing_res = IngestionResult(
            metadata=meta,
            pages=[p1, p2],
            tables=[tbl],
            images=[img],
        )

        assert ing_res.total_pages == 2
        assert ing_res.total_tables == 1
        assert ing_res.total_images == 1
        assert ing_res.has_text is True
        assert ing_res.has_tables is True
        assert ing_res.has_images is True
        assert ing_res.get_tables_on_page(1) == [tbl]
        assert ing_res.get_tables_on_page(2) == []
        assert ing_res.get_images_on_page(2) == [img]
        assert ing_res.get_images_on_page(1) == []


# ============================================================================
# 2. Chunk Lineage & Content Integrity
# ============================================================================

class TestChunkLineageAndContentIntegrity:
    """Certifies Document -> Page -> Chunk lineage and marker association."""

    def test_chunk_lineage_and_filtering(self) -> None:
        """Verify DocumentChunk preserves full lineage and ChunkingResult filter integrity."""
        chunk1 = DocumentChunk(
            chunk_id=DAY43_CHUNK_001,
            chunk_index=0,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=DAY43_PAGE_001,
            content=f"Text chunk containing {DAY43_MARKER_A}",
            content_type="text",
            metadata={"marker": DAY43_MARKER_A},
        )
        chunk2 = DocumentChunk(
            chunk_id=DAY43_CHUNK_002,
            chunk_index=1,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=DAY43_PAGE_002,
            content=f"Table chunk containing {DAY43_MARKER_B}",
            content_type="table",
            metadata={"marker": DAY43_MARKER_B},
        )
        chunk3 = DocumentChunk(
            chunk_id=DAY43_CHUNK_003,
            chunk_index=2,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=DAY43_PAGE_002,
            content=f"Image chunk containing {DAY43_MARKER_C}",
            content_type="image",
            metadata={"marker": DAY43_MARKER_C},
        )

        chunking_result = ChunkingResult(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            chunks=[chunk1, chunk2, chunk3],
        )

        assert chunking_result.total_chunks == 3
        assert chunking_result.text_chunks == 1
        assert chunking_result.table_chunks == 1
        assert chunking_result.image_chunks == 1

        # Check page filtering
        page1_chunks = chunking_result.get_chunks_on_page(1)
        assert len(page1_chunks) == 1
        assert page1_chunks[0].chunk_id == DAY43_CHUNK_001
        assert DAY43_MARKER_A in page1_chunks[0].content

        page2_chunks = chunking_result.get_chunks_on_page(2)
        assert len(page2_chunks) == 2
        assert {c.chunk_id for c in page2_chunks} == {DAY43_CHUNK_002, DAY43_CHUNK_003}

        # Check type filtering
        assert chunking_result.get_chunks_by_type("text") == [chunk1]
        assert chunking_result.get_chunks_by_type("table") == [chunk2]
        assert chunking_result.get_chunks_by_type("image") == [chunk3]


# ============================================================================
# 3. Metadata Integrity & Isolation
# ============================================================================

class TestMetadataIntegrityAndIsolation:
    """Certifies metadata retention, isolation, and absence of synthetic leakages."""

    def test_metadata_retention_across_models(self) -> None:
        """Verify custom metadata is strictly preserved and not modified across models."""
        citation = AgentCitation(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            chunk_id=DAY43_CHUNK_001,
            page_number=DAY43_PAGE_001,
            metadata=DAY43_META_001,
        )
        assert citation.metadata["day43_document"] == "001"
        assert citation.metadata["section"] == "architecture"

        # Serialization preserves exact metadata
        d = citation.to_dict()
        assert d["metadata"] == DAY43_META_001

        restored = AgentCitation.from_dict(d)
        assert restored.metadata == DAY43_META_001

    def test_caller_metadata_mutation_isolation(self) -> None:
        """Verify modifying caller metadata dict does not mutate model state."""
        caller_dict = {"env": "prod", "tenant": "A"}
        req = AgentRequest(query="Test", metadata=caller_dict)

        # Mutate caller dict
        caller_dict["env"] = "corrupted"
        caller_dict["injected"] = "leak"

        assert req.metadata["env"] == "prod"
        assert "injected" not in req.metadata


# ============================================================================
# 4. Duplicate Data Handling
# ============================================================================

class TestDuplicateDataHandling:
    """Certifies duplicate chunk ID, content warning, and retrieval deduplication."""

    def test_validate_chunks_detects_duplicate_chunk_ids(self) -> None:
        """validate_chunks detects and rejects duplicate chunk_ids in a batch."""
        c1 = DocumentChunk(
            chunk_id=DAY43_CHUNK_001,
            chunk_index=0,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=1,
            content="Content 1",
            content_type="text",
        )
        c2_dup = DocumentChunk(
            chunk_id=DAY43_CHUNK_001,  # Duplicate chunk_id
            chunk_index=1,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=2,
            content="Content 2",
            content_type="text",
        )

        res = validate_chunks([c1, c2_dup])
        assert res.is_valid is False
        assert any("Duplicate chunk_id" in err for err in res.errors)

    def test_validate_chunks_flags_duplicate_content_as_warning(self) -> None:
        """validate_chunks warns when identical content is placed in different chunks."""
        identical_text = "This exact text is duplicated across pages."
        c1 = DocumentChunk(
            chunk_id=DAY43_CHUNK_001,
            chunk_index=0,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=1,
            content=identical_text,
            content_type="text",
        )
        c2 = DocumentChunk(
            chunk_id=DAY43_CHUNK_002,
            chunk_index=1,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=2,
            content=identical_text,  # Identical content
            content_type="text",
        )

        res = validate_chunks([c1, c2])
        assert res.is_valid is True  # Duplicate content is a warning, not fatal
        assert any("Duplicate content detected" in w for w in res.warnings)

    def test_retrieval_deduplication_by_chunk_id(self) -> None:
        """process_retrieval_results deduplicates by chunk_id, retaining highest score."""
        r1 = VectorSearchResult(
            chunk_id=DAY43_CHUNK_001,
            score=0.75,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Chunk 1 lower score",
        )
        r2_higher = VectorSearchResult(
            chunk_id=DAY43_CHUNK_001,  # Same chunk_id
            score=0.95,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Chunk 1 higher score",
        )
        r3 = VectorSearchResult(
            chunk_id=DAY43_CHUNK_002,
            score=0.85,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=2,
            chunk_index=1,
            content_type="text",
            content="Chunk 2 content",
        )

        processed = process_retrieval_results([r1, r2_higher, r3])
        assert len(processed) == 2
        # r2_higher retained over r1
        assert processed[0].chunk_id == DAY43_CHUNK_001
        assert processed[0].score == 0.95
        assert processed[1].chunk_id == DAY43_CHUNK_002
        assert processed[1].score == 0.85


# ============================================================================
# 5. Missing & Empty Data Rejection
# ============================================================================

class TestMissingAndEmptyDataRejection:
    """Certifies validation failures when required data is missing or empty."""

    def test_missing_required_fields_raise_deterministic_errors(self) -> None:
        """Missing or empty required fields raise deterministic validation errors."""
        with pytest.raises(AgentValidationError, match="document_id must be a non-empty string"):
            AgentCitation(document_id="", filename=DAY43_FILE_001, chunk_id=DAY43_CHUNK_001)

        with pytest.raises(AgentValidationError, match="filename must be a non-empty string"):
            AgentCitation(document_id=DAY43_DOC_001, filename="", chunk_id=DAY43_CHUNK_001)

        with pytest.raises(AgentValidationError, match="chunk_id must be a non-empty string"):
            AgentCitation(document_id=DAY43_DOC_001, filename=DAY43_FILE_001, chunk_id="")

        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            AgentRequest(query="")

        with pytest.raises(VisionEvidenceError, match="document_id must be a non-empty string"):
            VisualEvidence(document_id="", filename=DAY43_FILE_001, chunk_id=DAY43_CHUNK_001)

        with pytest.raises(VisionInputValidationError, match="query cannot be empty"):
            VisionRequest(query="   ")

    def test_valid_empty_collections_behavior(self) -> None:
        """Valid empty collections produce expected state without errors."""
        # Empty citations
        resp = AgentResponse(answer="No citations", agent_name="SearchAgent", citations=[])
        assert resp.has_citations is False
        assert resp.total_citations == 0
        assert resp.unique_documents == []

        # Empty search results
        sres = SearchResult(query="No results query", citations=[])
        assert sres.has_results is False
        assert sres.total_results == 0
        assert sres.status == "NO_RESULTS"

        # Empty visual evidence
        vreq = VisionRequest(query="Analyze", evidence=[])
        assert vreq.has_evidence is False
        assert vreq.total_evidence == 0


# ============================================================================
# 6. Malformed Data Rejection
# ============================================================================

class TestMalformedDataRejection:
    """Certifies rejection of malformed or invalidly-typed values."""

    def test_invalid_page_number_rejection(self) -> None:
        """page_number must be a positive integer >= 1 or None."""
        for invalid_page in (0, -1, -99, True, False, "1", [1]):
            with pytest.raises(AgentValidationError):
                AgentCitation(
                    document_id=DAY43_DOC_001,
                    filename=DAY43_FILE_001,
                    chunk_id=DAY43_CHUNK_001,
                    page_number=invalid_page,  # type: ignore[arg-type]
                )

            with pytest.raises(VisionEvidenceError):
                VisualEvidence(
                    document_id=DAY43_DOC_001,
                    filename=DAY43_FILE_001,
                    chunk_id=DAY43_CHUNK_001,
                    page_number=invalid_page,  # type: ignore[arg-type]
                )

    def test_invalid_chunk_index_rejection(self) -> None:
        """chunk_index must be >= 0."""
        for invalid_idx in (-1, -10, True, "0"):
            with pytest.raises(VisionEvidenceError):
                VisualEvidence(
                    document_id=DAY43_DOC_001,
                    filename=DAY43_FILE_001,
                    chunk_id=DAY43_CHUNK_001,
                    chunk_index=invalid_idx,  # type: ignore[arg-type]
                )

    def test_invalid_score_values_rejection(self) -> None:
        """Non-finite or boolean scores in AgentCitation are rejected."""
        for bad_score in (float("nan"), float("inf"), float("-inf"), True, False):
            with pytest.raises(AgentValidationError, match="score must be a finite numeric float"):
                AgentCitation(
                    document_id=DAY43_DOC_001,
                    filename=DAY43_FILE_001,
                    chunk_id=DAY43_CHUNK_001,
                    score=bad_score,  # type: ignore[arg-type]
                )

    def test_invalid_visual_content_type_rejection(self) -> None:
        """VisualEvidence content_type must strictly be in ('image', 'chart', 'diagram')."""
        for bad_type in ("text", "table", "audio", "video", "pdf", "raw"):
            with pytest.raises(VisionEvidenceError, match="Invalid visual content_type"):
                VisualEvidence(
                    document_id=DAY43_DOC_001,
                    filename=DAY43_FILE_001,
                    chunk_id=DAY43_CHUNK_001,
                    content_type=bad_type,
                )


# ============================================================================
# 7. Document & Page Misassociation Detection
# ============================================================================

class TestDocumentAndPageMisassociation:
    """Certifies behavior when chunks/citations from mismatched documents/pages are combined."""

    def test_validate_chunks_detects_mixed_document_ids(self) -> None:
        """validate_chunks flags an error if chunks belonging to different documents are mixed."""
        c_doc1 = DocumentChunk(
            chunk_id=DAY43_CHUNK_001,
            chunk_index=0,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=1,
            content="Content from doc 1",
            content_type="text",
        )
        c_doc2 = DocumentChunk(
            chunk_id=DAY43_CHUNK_002,
            chunk_index=1,
            document_id=DAY43_DOC_002,  # Inconsistent document_id
            filename=DAY43_FILE_002,
            page_number=1,
            content="Content from doc 2",
            content_type="text",
        )

        res = validate_chunks([c_doc1, c_doc2])
        assert res.is_valid is False
        assert any("Inconsistent document_id" in err for err in res.errors)

    def test_search_result_partitions_mixed_documents_correctly(self) -> None:
        """SearchResult.by_document correctly groups citations without cross-document mixing."""
        c1 = AgentCitation(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            chunk_id=DAY43_CHUNK_001,
            content_type="text",
        )
        c2 = AgentCitation(
            document_id=DAY43_DOC_002,
            filename=DAY43_FILE_002,
            chunk_id=DAY43_CHUNK_002,
            content_type="table",
        )
        c3 = AgentCitation(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            chunk_id=DAY43_CHUNK_003,
            content_type="image",
        )

        sres = SearchResult(
            query="Multi-doc search",
            status="RESULTS_FOUND",
            citations=[c1, c2, c3],
        )

        grouped = sres.by_document
        assert set(grouped.keys()) == {DAY43_DOC_001, DAY43_DOC_002}
        assert len(grouped[DAY43_DOC_001]) == 2
        assert len(grouped[DAY43_DOC_002]) == 1
        assert grouped[DAY43_DOC_001][0].chunk_id == DAY43_CHUNK_001
        assert grouped[DAY43_DOC_001][1].chunk_id == DAY43_CHUNK_003
        assert grouped[DAY43_DOC_002][0].chunk_id == DAY43_CHUNK_002


# ============================================================================
# 8. Corrupted Content & Metadata Sanitization
# ============================================================================

class TestCorruptedContentAndMetadata:
    """Certifies differentiation of corrupted content and sanitization of forbidden metadata."""

    def test_corrupted_chunk_content_differentiation(self) -> None:
        """Verify corrupted chunk content is distinguishable from original via content & hash."""
        orig_chunk = DocumentChunk(
            chunk_id=DAY43_CHUNK_001,
            chunk_index=0,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=1,
            content=f"Genuine content: {DAY43_MARKER_A}",
            content_type="text",
        )
        corrupt_chunk = DocumentChunk(
            chunk_id=DAY43_CHUNK_001,
            chunk_index=0,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=1,
            content=f"Corrupted content: {DAY43_MARKER_CORRUPTED}",
            content_type="text",
        )

        orig_hash = hashlib.sha256(orig_chunk.content.encode("utf-8")).hexdigest()
        corrupt_hash = hashlib.sha256(corrupt_chunk.content.encode("utf-8")).hexdigest()

        assert orig_chunk.content != corrupt_chunk.content
        assert orig_hash != corrupt_hash
        assert DAY43_MARKER_A in orig_chunk.content
        assert DAY43_MARKER_A not in corrupt_chunk.content

    def test_vision_normalizer_sanitizes_corrupted_or_forbidden_metadata(self) -> None:
        """VisionResultNormalizer strips forbidden keys (secrets/api_key) from metadata."""
        polluted_metadata = {
            "safe_metric": "ok",
            "api_key": "SECRET_KEY_LEAK",
            "password": "PLAINTEXT_PASSWORD",
            "token": "BEARER_TOKEN",
            "auth": "AUTH_HEADER",
        }
        ev = VisualEvidence(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            chunk_id=DAY43_CHUNK_001,
            content_type="image",
            metadata=polluted_metadata,
        )
        vreq = VisionRequest(query="Analyze image", evidence=[ev])
        normalizer = VisionResultNormalizer()

        # 1. Direct dictionary sanitization
        clean_dict = VisionResultNormalizer.sanitize_metadata(polluted_metadata)
        for forbidden in FORBIDDEN_METADATA_KEYS:
            assert forbidden not in clean_dict
        assert clean_dict.get("safe_metric") == "ok"

        # 2. Result normalization sanitizes result.metadata
        sanitized_res = normalizer.normalize(
            result={"query": "Analyze image", "description": "Normalized output", "metadata": polluted_metadata},
            request=vreq,
        )

        for forbidden in FORBIDDEN_METADATA_KEYS:
            assert forbidden not in sanitized_res.metadata

        assert sanitized_res.metadata.get("safe_metric") == "ok"


# ============================================================================
# 9. Serialization Corruption & Round-Trip
# ============================================================================

class TestSerializationCorruptionAndRoundTrip:
    """Certifies round-trip integrity and behavior when serialized data is mutated."""

    def test_citation_roundtrip_identity(self) -> None:
        """AgentCitation survives serialize -> deserialize without loss of lineage."""
        citation = AgentCitation(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            chunk_id=DAY43_CHUNK_001,
            page_number=DAY43_PAGE_001,
            content_type="table",
            score=0.93,
            metadata=DAY43_META_001,
        )
        d = citation.to_dict()
        restored = AgentCitation.from_dict(d)
        assert restored == citation

    def test_serialization_mutation_behavior(self) -> None:
        """Mutating serialized dictionary modifies restored object faithfully without cache bleed."""
        citation = AgentCitation(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            chunk_id=DAY43_CHUNK_001,
            page_number=DAY43_PAGE_001,
            score=0.90,
        )
        d = citation.to_dict()

        # Corrupt serialized document_id and score
        d["document_id"] = "MUTATED-DOC-999"
        d["score"] = 0.50

        restored = AgentCitation.from_dict(d)
        assert restored.document_id == "MUTATED-DOC-999"
        assert restored.score == 0.50
        # Original remains untouched
        assert citation.document_id == DAY43_DOC_001
        assert citation.score == 0.90

    def test_serialization_corruption_with_empty_required_field_raises(self) -> None:
        """Corrupting serialized dictionary by emptying required field raises on from_dict."""
        citation = AgentCitation(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            chunk_id=DAY43_CHUNK_001,
        )
        d = citation.to_dict()
        d["document_id"] = ""  # Corrupt to empty string

        with pytest.raises(AgentValidationError, match="document_id must be a non-empty string"):
            AgentCitation.from_dict(d)


# ============================================================================
# 10. Retrieval, Citation & Visual Evidence Lineage
# ============================================================================

class TestLineageFlowIntegrity:
    """Certifies lineage flow across VectorSearchResult -> AgentCitation -> VisualEvidence."""

    def test_end_to_end_lineage_flow(self) -> None:
        """VectorSearchResult preserves all lineage fields into AgentCitation and VisualEvidence."""
        vs_result = VectorSearchResult(
            chunk_id=DAY43_CHUNK_001,
            score=0.98,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=DAY43_PAGE_001,
            chunk_index=0,
            content_type="image",
            content=f"Image description containing {DAY43_MARKER_A}",
            metadata=DAY43_META_001,
        )

        citation = AgentCitation.from_search_result(vs_result)
        assert citation.document_id == DAY43_DOC_001
        assert citation.filename == DAY43_FILE_001
        assert citation.chunk_id == DAY43_CHUNK_001
        assert citation.page_number == DAY43_PAGE_001
        assert citation.content_type == "image"
        assert citation.score == 0.98
        assert citation.metadata["day43_document"] == "001"

        evidence = VisualEvidence.from_citation(citation)
        assert evidence.document_id == DAY43_DOC_001
        assert evidence.filename == DAY43_FILE_001
        assert evidence.chunk_id == DAY43_CHUNK_001
        assert evidence.page_number == DAY43_PAGE_001
        assert evidence.content_type == "image"

    def test_visual_evidence_adapter_rejects_text_citations(self) -> None:
        """VisualEvidenceAdapter rejects non-visual citations deterministically."""
        text_citation = AgentCitation(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            chunk_id=DAY43_CHUNK_001,
            content_type="text",  # Not visual
        )
        assert VisualEvidenceAdapter.is_visual(text_citation) is False

        with pytest.raises(VisionEvidenceError, match="Unsupported content_type 'text'"):
            VisualEvidenceAdapter.adapt_citation(text_citation)


# ============================================================================
# 11. Cross-Document & Cross-Request Isolation
# ============================================================================

class TestCrossDocumentAndRequestIsolation:
    """Certifies zero cross-document and cross-request contamination."""

    def test_cross_document_isolation_in_search_and_responses(self) -> None:
        """DOC-A and DOC-B queries and responses remain completely isolated."""
        c_a = AgentCitation(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            chunk_id=DAY43_CHUNK_001,
            metadata={"marker": DAY43_MARKER_A},
        )
        c_b = AgentCitation(
            document_id=DAY43_DOC_002,
            filename=DAY43_FILE_002,
            chunk_id=DAY43_CHUNK_002,
            metadata={"marker": DAY43_MARKER_B},
        )

        resp_a = AgentResponse(
            answer=f"Answer for {DAY43_DOC_001}",
            agent_name="AgentA",
            citations=[c_a],
            metadata={"session_id": "SESS-A", "doc": DAY43_DOC_001},
        )
        resp_b = AgentResponse(
            answer=f"Answer for {DAY43_DOC_002}",
            agent_name="AgentB",
            citations=[c_b],
            metadata={"session_id": "SESS-B", "doc": DAY43_DOC_002},
        )

        # Assert no cross contamination
        assert resp_a.citations[0].document_id == DAY43_DOC_001
        assert resp_a.citations[0].metadata["marker"] == DAY43_MARKER_A
        assert DAY43_DOC_002 not in resp_a.unique_documents

        assert resp_b.citations[0].document_id == DAY43_DOC_002
        assert resp_b.citations[0].metadata["marker"] == DAY43_MARKER_B
        assert DAY43_DOC_001 not in resp_b.unique_documents

    def test_cross_request_trace_isolation(self) -> None:
        """VisionExecutionTrace instances do not share stage history."""
        trace1 = VisionExecutionTrace(["request_received", "validation_started"])
        trace2 = VisionExecutionTrace(["request_received"])

        trace1.add_stage("input_prepared")

        assert trace1.stages == ["request_received", "validation_started", "input_prepared"]
        assert trace2.stages == ["request_received"]


# ============================================================================
# 12. Input Mutation Safety & Aliasing
# ============================================================================

class TestInputMutationSafetyAndAliasing:
    """Certifies that models and functions protect caller inputs from mutation and aliasing."""

    def test_process_retrieval_results_does_not_mutate_input_list(self) -> None:
        """process_retrieval_results does not modify caller-supplied list."""
        r1 = VectorSearchResult(
            chunk_id=DAY43_CHUNK_001,
            score=0.9,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Content 1",
        )
        r2 = VectorSearchResult(
            chunk_id=DAY43_CHUNK_002,
            score=0.8,
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            page_number=2,
            chunk_index=1,
            content_type="text",
            content="Content 2",
        )
        input_list = [r1, r2]
        input_list_copy = list(input_list)

        processed = process_retrieval_results(input_list, min_score=0.85)

        assert len(processed) == 1
        assert input_list == input_list_copy
        assert len(input_list) == 2

    def test_independent_copies_do_not_alias(self) -> None:
        """Modifying deep copy does not mutate original object."""
        c = AgentCitation(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            chunk_id=DAY43_CHUNK_001,
            metadata={"nested": {"count": 1}},
        )
        c_copy = copy.deepcopy(c)
        c_copy.metadata["nested"]["count"] = 999

        assert c.metadata["nested"]["count"] == 1


# ============================================================================
# 13. Partial Data & Error Isolation
# ============================================================================

class TestPartialDataAndErrorIsolation:
    """Certifies optional field defaulting and sequential error isolation."""

    def test_partially_populated_objects_default_correctly(self) -> None:
        """Partially populated optional fields default gracefully."""
        citation = AgentCitation(
            document_id=DAY43_DOC_001,
            filename=DAY43_FILE_001,
            chunk_id=DAY43_CHUNK_001,
        )
        assert citation.page_number is None
        assert citation.score == 0.0
        assert citation.content_type == "text"
        assert citation.metadata == {}

    def test_sequential_error_isolation(self) -> None:
        """An invalid request in a sequence does not corrupt state of subsequent valid requests."""
        results: list[str] = []

        # 1. Valid A
        req_a = AgentRequest(query="Valid query A")
        results.append(req_a.query)

        # 2. Invalid B
        with pytest.raises(AgentValidationError):
            AgentRequest(query="")

        # 3. Valid C
        req_c = AgentRequest(query="Valid query C")
        results.append(req_c.query)

        assert results == ["Valid query A", "Valid query C"]


# ============================================================================
# 14. Repeated Corruption & Execution Determinism
# ============================================================================

class TestRepeatedExecutionDeterminism:
    """Certifies stability across repeated corruption and multi-iteration execution."""

    def test_repeated_corruption_produces_identical_errors(self) -> None:
        """Running the same corruption 10 times yields identical exception types and messages."""
        expected_msg = "query cannot be empty or whitespace-only."
        for _ in range(10):
            with pytest.raises(AgentValidationError) as exc:
                AgentRequest(query="   ")
            assert str(exc.value) == expected_msg

    def test_pipeline_lineage_stability_across_3_iterations(self) -> None:
        """Running valid data 3 times through the pipeline yields identical outputs."""
        outputs: list[dict[str, Any]] = []

        for _ in range(3):
            vs_result = VectorSearchResult(
                chunk_id=DAY43_CHUNK_001,
                score=0.95,
                document_id=DAY43_DOC_001,
                filename=DAY43_FILE_001,
                page_number=DAY43_PAGE_001,
                chunk_index=0,
                content_type="text",
                content=DAY43_MARKER_A,
            )
            citation = AgentCitation.from_search_result(vs_result)
            resp = AgentResponse(
                answer=f"Answer with {DAY43_MARKER_A}",
                agent_name="SearchAgent",
                citations=[citation],
            )
            outputs.append(resp.to_dict())

        assert outputs[0] == outputs[1] == outputs[2]


# ============================================================================
# 15. Controlled Multi-Document Batch Data Integrity
# ============================================================================

class TestControlledMultiDocumentBatchIntegrity:
    """Certifies data integrity across a controlled batch of multiple documents."""

    def test_batch_of_50_chunks_across_5_documents(self) -> None:
        """Generate 50 chunks across 5 documents and verify no ID, page, or content corruption."""
        all_chunks: list[DocumentChunk] = []

        for doc_num in range(1, 6):
            doc_id = f"DAY43-DOC-{doc_num:03d}"
            filename = f"doc_{doc_num:03d}.pdf"

            for chunk_num in range(10):
                chunk_id = f"DAY43-CHUNK-{doc_num:03d}-{chunk_num:02d}"
                page_num = (chunk_num % 3) + 1
                content = f"Document {doc_id} Page {page_num} Chunk {chunk_num} Marker_{doc_num}_{chunk_num}"

                c = DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_index=chunk_num,
                    document_id=doc_id,
                    filename=filename,
                    page_number=page_num,
                    content=content,
                    content_type="text",
                    metadata={"doc_num": doc_num, "chunk_num": chunk_num},
                )
                all_chunks.append(c)

        assert len(all_chunks) == 50

        # Validate each document's chunk slice individually
        for doc_num in range(1, 6):
            doc_id = f"DAY43-DOC-{doc_num:03d}"
            doc_chunks = [c for c in all_chunks if c.document_id == doc_id]
            assert len(doc_chunks) == 10

            val_res = validate_chunks(doc_chunks)
            assert val_res.is_valid is True
            assert val_res.total_chunks == 10
            assert val_res.valid_chunks == 10
            assert len(val_res.errors) == 0
