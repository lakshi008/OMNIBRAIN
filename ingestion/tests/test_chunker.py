"""
Tests for the document chunker module.

Tests cover text splitting, table formatting, image metadata preservation,
overlap verification, edge cases, and error handling.
"""

from __future__ import annotations

import uuid
import pytest

from ingestion.chunker import chunk_document
from ingestion.models import (
    ChunkingResult,
    DocumentChunk,
    DocumentMetadata,
    ExtractedImage,
    ExtractedTable,
    IngestionResult,
    PageData,
    ParsedDocument,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    """Create a sample DocumentMetadata instance."""
    return DocumentMetadata(
        document_id=str(uuid.uuid4()),
        filename="report.pdf",
        total_pages=3,
        content_type="application/pdf",
        created_at="2026-08-15T10:00:00Z",
        pages_with_content=2,
        pages_without_content=1,
    )


@pytest.fixture
def short_text_doc(sample_metadata: DocumentMetadata) -> ParsedDocument:
    """Create a ParsedDocument with a single short page."""
    pages = [
        PageData(
            page_number=1,
            text="Short document summary paragraph.",
            char_count=33,
            has_content=True,
        )
    ]
    return ParsedDocument(metadata=sample_metadata, pages=pages)


@pytest.fixture
def long_text_doc(sample_metadata: DocumentMetadata) -> ParsedDocument:
    """Create a ParsedDocument with a long text page (1500 chars)."""
    # 1500 chars text
    text = "Sentence number " + " ".join(f"word{i}" for i in range(300))
    pages = [
        PageData(
            page_number=1,
            text=text,
            char_count=len(text),
            has_content=True,
        )
    ]
    return ParsedDocument(metadata=sample_metadata, pages=pages)


@pytest.fixture
def multi_page_text_doc(sample_metadata: DocumentMetadata) -> ParsedDocument:
    """Create a multi-page document including an empty page."""
    pages = [
        PageData(page_number=1, text="Page 1 text content.", char_count=20, has_content=True),
        PageData(page_number=2, text="", char_count=0, has_content=False),
        PageData(page_number=3, text="Page 3 text content.", char_count=20, has_content=True),
    ]
    return ParsedDocument(metadata=sample_metadata, pages=pages)


@pytest.fixture
def full_ingestion_result(sample_metadata: DocumentMetadata) -> IngestionResult:
    """Create an IngestionResult with text, tables, and images."""
    pages = [
        PageData(page_number=1, text="Executive Summary of OmniBrain.", char_count=31, has_content=True),
        PageData(page_number=2, text="Detailed analysis on Page 2.", char_count=28, has_content=True),
    ]
    tables = [
        ExtractedTable(
            page_number=1,
            table_index=0,
            rows=2,
            columns=2,
            cells=[["Metric", "Value"], ["Precision", "98.5%"]],
        )
    ]
    images = [
        ExtractedImage(
            page_number=2,
            image_index=0,
            image_format="png",
            width=200,
            height=150,
            image_bytes=b"\x89PNG\r\n\x1a\nfakebytes",
            size_bytes=16,
            colorspace="DeviceRGB",
            xref=12,
        )
    ]
    return IngestionResult(
        metadata=sample_metadata,
        pages=pages,
        tables=tables,
        images=images,
    )


@pytest.fixture
def empty_doc(sample_metadata: DocumentMetadata) -> IngestionResult:
    """Create a completely empty IngestionResult."""
    return IngestionResult(
        metadata=sample_metadata,
        pages=[],
        tables=[],
        images=[],
    )


# ── Text Chunking Tests ──────────────────────────────────────────────────


class TestTextChunking:
    """Tests for text chunking behavior."""

    def test_returns_chunking_result(self, short_text_doc: ParsedDocument) -> None:
        """chunk_document should return a ChunkingResult with DocumentChunks."""
        result = chunk_document(short_text_doc)
        assert isinstance(result, ChunkingResult)
        assert result.total_chunks == 1
        assert isinstance(result.chunks[0], DocumentChunk)

    def test_single_short_text_chunk(self, short_text_doc: ParsedDocument) -> None:
        """Text shorter than chunk_size should produce exactly one chunk."""
        result = chunk_document(short_text_doc, chunk_size=1000)
        assert result.total_chunks == 1
        chunk = result.chunks[0]
        assert chunk.content == "Short document summary paragraph."
        assert chunk.content_type == "text"
        assert chunk.page_number == 1
        assert chunk.chunk_index == 0

    def test_long_text_splitting(self, long_text_doc: ParsedDocument) -> None:
        """Long text should be split into multiple overlapping chunks."""
        chunk_size = 400
        chunk_overlap = 100
        result = chunk_document(long_text_doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        assert result.total_chunks > 1
        for chunk in result.chunks:
            assert len(chunk.content) <= chunk_size
            assert chunk.content_type == "text"

    def test_chunk_overlap_content(self, long_text_doc: ParsedDocument) -> None:
        """Consecutive chunks must share overlap characters."""
        chunk_size = 200
        chunk_overlap = 50
        result = chunk_document(long_text_doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        assert result.total_chunks >= 2
        # Check that the end of chunk 0 matches the start of chunk 1
        chunk0_end = result.chunks[0].content[-chunk_overlap:]
        chunk1_start = result.chunks[1].content[:chunk_overlap]
        assert chunk0_end == chunk1_start

    def test_chunk_indexes_are_sequential(self, long_text_doc: ParsedDocument) -> None:
        """Chunk indexes should be 0-indexed and sequential."""
        result = chunk_document(long_text_doc, chunk_size=300, chunk_overlap=50)
        indexes = [c.chunk_index for c in result.chunks]
        assert indexes == list(range(len(result.chunks)))

    def test_unique_chunk_ids(self, long_text_doc: ParsedDocument) -> None:
        """Every chunk must have a unique UUID4 chunk_id."""
        result = chunk_document(long_text_doc, chunk_size=300, chunk_overlap=50)
        chunk_ids = [c.chunk_id for c in result.chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
        for cid in chunk_ids:
            parsed_uuid = uuid.UUID(cid)
            assert parsed_uuid.version == 4

    def test_metadata_preservation(self, short_text_doc: ParsedDocument) -> None:
        """document_id and filename should be preserved in result and chunks."""
        result = chunk_document(short_text_doc)
        assert result.document_id == short_text_doc.metadata.document_id
        assert result.filename == short_text_doc.metadata.filename
        assert result.chunks[0].document_id == short_text_doc.metadata.document_id
        assert result.chunks[0].filename == short_text_doc.metadata.filename

    def test_multi_page_page_numbers(self, multi_page_text_doc: ParsedDocument) -> None:
        """Page numbers should be preserved and empty pages skipped."""
        result = chunk_document(multi_page_text_doc)
        assert result.total_chunks == 2
        assert result.chunks[0].page_number == 1
        assert result.chunks[1].page_number == 3


# ── Multi-Modal Chunking Tests ───────────────────────────────────────────


class TestMultiModalChunking:
    """Tests for table and image chunking in IngestionResult."""

    def test_mixed_modalities_created(self, full_ingestion_result: IngestionResult) -> None:
        """Should create text, table, and image chunks with proper content types."""
        result = chunk_document(full_ingestion_result)
        assert result.text_chunks == 2
        assert result.table_chunks == 1
        assert result.image_chunks == 1
        assert result.total_chunks == 4

    def test_table_chunk_content_and_metadata(self, full_ingestion_result: IngestionResult) -> None:
        """Table chunks should contain Markdown formatted tables and metadata."""
        result = chunk_document(full_ingestion_result)
        table_chunks = result.get_chunks_by_type("table")
        assert len(table_chunks) == 1
        tc = table_chunks[0]
        assert tc.page_number == 1
        assert "| Metric | Value |" in tc.content
        assert "| Precision | 98.5% |" in tc.content
        assert tc.metadata["table_index"] == 0
        assert tc.metadata["rows"] == 2
        assert tc.metadata["columns"] == 2

    def test_image_chunk_content_and_metadata(self, full_ingestion_result: IngestionResult) -> None:
        """Image chunks should contain text reference and full image metadata."""
        result = chunk_document(full_ingestion_result)
        image_chunks = result.get_chunks_by_type("image")
        assert len(image_chunks) == 1
        ic = image_chunks[0]
        assert ic.page_number == 2
        assert "[Image on Page 2" in ic.content
        assert "PNG" in ic.content
        assert ic.metadata["width"] == 200
        assert ic.metadata["height"] == 150
        assert ic.metadata["colorspace"] == "DeviceRGB"
        assert ic.metadata["xref"] == 12

    def test_get_chunks_on_page(self, full_ingestion_result: IngestionResult) -> None:
        """get_chunks_on_page should return all chunks originating from that page."""
        result = chunk_document(full_ingestion_result)
        p1_chunks = result.get_chunks_on_page(1)
        p2_chunks = result.get_chunks_on_page(2)
        # Page 1 has 1 text chunk + 1 table chunk
        assert len(p1_chunks) == 2
        # Page 2 has 1 text chunk + 1 image chunk
        assert len(p2_chunks) == 2
        # Page 99 has 0
        assert len(result.get_chunks_on_page(99)) == 0


# ── Edge Cases and Error Handling ────────────────────────────────────────


class TestChunkerEdgeCasesAndErrors:
    """Tests for edge cases and parameter validation."""

    def test_empty_document(self, empty_doc: IngestionResult) -> None:
        """Empty document should return ChunkingResult with 0 chunks."""
        result = chunk_document(empty_doc)
        assert result.total_chunks == 0
        assert result.has_chunks is False
        assert result.chunks == []

    def test_invalid_chunk_size_raises(self, short_text_doc: ParsedDocument) -> None:
        """chunk_size <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            chunk_document(short_text_doc, chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            chunk_document(short_text_doc, chunk_size=-10)

    def test_invalid_chunk_overlap_raises(self, short_text_doc: ParsedDocument) -> None:
        """chunk_overlap < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            chunk_document(short_text_doc, chunk_overlap=-1)

    def test_overlap_greater_or_equal_to_size_raises(self, short_text_doc: ParsedDocument) -> None:
        """chunk_overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="strictly less than chunk_size"):
            chunk_document(short_text_doc, chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="strictly less than chunk_size"):
            chunk_document(short_text_doc, chunk_size=100, chunk_overlap=150)
