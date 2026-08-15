"""
Tests for the embedding preparation module.

Tests cover conversion of validated DocumentChunks to EmbeddingRecords,
metadata and lineage preservation, deterministic ordering, counter helpers,
and error handling on invalid chunks.
"""

from __future__ import annotations

import uuid
import pytest

from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.models import (
    ChunkingResult,
    DocumentChunk,
    EmbeddingPreparationResult,
    EmbeddingRecord,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_doc_id() -> str:
    """Create a sample document UUID."""
    return str(uuid.uuid4())


@pytest.fixture
def valid_text_chunks(sample_doc_id: str) -> list[DocumentChunk]:
    """Create sample text chunks."""
    return [
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            chunk_index=0,
            document_id=sample_doc_id,
            filename="annual_report.pdf",
            page_number=1,
            content="Introduction to OmniBrain architecture and agents.",
            content_type="text",
            metadata={"char_count": 48},
        ),
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            chunk_index=1,
            document_id=sample_doc_id,
            filename="annual_report.pdf",
            page_number=1,
            content="Detailed explanation of the ingestion sub-modules.",
            content_type="text",
            metadata={"char_count": 51},
        ),
    ]


@pytest.fixture
def valid_table_chunk(sample_doc_id: str) -> DocumentChunk:
    """Create a sample table chunk."""
    return DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        chunk_index=2,
        document_id=sample_doc_id,
        filename="annual_report.pdf",
        page_number=2,
        content="| Metric | Q1 | Q2 |\n| --- | --- | --- |\n| Revenue | $10M | $12M |",
        content_type="table",
        metadata={"table_index": 0, "rows": 2, "columns": 3},
    )


@pytest.fixture
def valid_image_chunk(sample_doc_id: str) -> DocumentChunk:
    """Create a sample image chunk."""
    return DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        chunk_index=3,
        document_id=sample_doc_id,
        filename="annual_report.pdf",
        page_number=3,
        content="[Image on Page 3 (Index 0): format=PNG, width=800px, height=600px, size=45000 bytes]",
        content_type="image",
        metadata={"image_index": 0, "image_format": "png", "width": 800, "height": 600, "size_bytes": 45000},
    )


@pytest.fixture
def mixed_chunks(
    valid_text_chunks: list[DocumentChunk],
    valid_table_chunk: DocumentChunk,
    valid_image_chunk: DocumentChunk,
) -> list[DocumentChunk]:
    """Create a mixed list of text, table, and image chunks."""
    return list(valid_text_chunks) + [valid_table_chunk, valid_image_chunk]


@pytest.fixture
def chunking_result(mixed_chunks: list[DocumentChunk], sample_doc_id: str) -> ChunkingResult:
    """Create a ChunkingResult instance."""
    return ChunkingResult(
        document_id=sample_doc_id,
        filename="annual_report.pdf",
        chunks=mixed_chunks,
    )


# ── Success Tests ────────────────────────────────────────────────────────


class TestEmbeddingPreparationSuccess:
    """Tests for successful embedding preparation."""

    def test_returns_embedding_preparation_result(self, mixed_chunks: list[DocumentChunk]) -> None:
        """prepare_for_embedding should return an EmbeddingPreparationResult."""
        result = prepare_for_embedding(mixed_chunks)
        assert isinstance(result, EmbeddingPreparationResult)
        assert result.is_ready is True
        assert result.total_items == 4
        for item in result.items:
            assert isinstance(item, EmbeddingRecord)

    def test_accepts_chunking_result_input(self, chunking_result: ChunkingResult) -> None:
        """prepare_for_embedding should accept a ChunkingResult instance."""
        result = prepare_for_embedding(chunking_result)
        assert result.is_ready is True
        assert result.total_items == 4
        assert result.document_id == chunking_result.document_id
        assert result.filename == chunking_result.filename

    def test_empty_input_handled_safely(self) -> None:
        """Empty list returns an empty EmbeddingPreparationResult with is_ready=True."""
        result = prepare_for_embedding([])
        assert result.is_ready is True
        assert result.total_items == 0
        assert result.items == []

    def test_text_only_chunks(self, valid_text_chunks: list[DocumentChunk]) -> None:
        """Works correctly for text-only chunk collections."""
        result = prepare_for_embedding(valid_text_chunks)
        assert result.total_items == 2
        assert result.text_items == 2
        assert result.table_items == 0
        assert result.image_items == 0

    def test_table_chunks_preserved(self, sample_doc_id: str, valid_table_chunk: DocumentChunk) -> None:
        """Table chunks produce valid embedding records with table modality."""
        result = prepare_for_embedding([valid_table_chunk])
        assert result.table_items == 1
        record = result.items[0]
        assert record.content_type == "table"
        assert "| Revenue |" in record.content

    def test_image_chunks_preserved(self, sample_doc_id: str, valid_image_chunk: DocumentChunk) -> None:
        """Image chunks produce valid embedding records with image modality."""
        result = prepare_for_embedding([valid_image_chunk])
        assert result.image_items == 1
        record = result.items[0]
        assert record.content_type == "image"
        assert "[Image on Page 3" in record.content


# ── Metadata & Citation Lineage Tests ────────────────────────────────────


class TestMetadataAndLineage:
    """Tests verifying lineage and citation metadata preservation."""

    def test_preserves_document_and_chunk_ids(self, mixed_chunks: list[DocumentChunk]) -> None:
        """document_id and chunk_id must match the source chunks."""
        result = prepare_for_embedding(mixed_chunks)
        for orig, record in zip(mixed_chunks, result.items):
            assert record.chunk_id == orig.chunk_id
            assert record.document_id == orig.document_id
            assert record.filename == orig.filename

    def test_preserves_page_numbers_for_citations(self, mixed_chunks: list[DocumentChunk]) -> None:
        """page_number is preserved for citation generation."""
        result = prepare_for_embedding(mixed_chunks)
        assert result.items[0].page_number == 1
        assert result.items[1].page_number == 1
        assert result.items[2].page_number == 2
        assert result.items[3].page_number == 3

    def test_metadata_payload_separation(self, valid_table_chunk: DocumentChunk) -> None:
        """Metadata payload is separate from embeddable text content."""
        result = prepare_for_embedding([valid_table_chunk])
        record = result.items[0]
        assert isinstance(record.content, str)
        assert isinstance(record.metadata, dict)
        # Citation & lineage fields in metadata
        assert record.metadata["chunk_id"] == valid_table_chunk.chunk_id
        assert record.metadata["document_id"] == valid_table_chunk.document_id
        assert record.metadata["filename"] == valid_table_chunk.filename
        assert record.metadata["page_number"] == 2
        assert record.metadata["content_type"] == "table"
        assert record.metadata["rows"] == 2
        assert record.metadata["columns"] == 3


# ── Ordering & Determinism Tests ─────────────────────────────────────────


class TestOrderingAndDeterminism:
    """Tests for deterministic sorting by chunk_index."""

    def test_shuffled_chunks_sorted_by_chunk_index(self, mixed_chunks: list[DocumentChunk]) -> None:
        """Input chunks provided in reverse order are sorted deterministically."""
        shuffled = list(reversed(mixed_chunks))
        result = prepare_for_embedding(shuffled)
        indices = [item.chunk_index for item in result.items]
        assert indices == [0, 1, 2, 3]

    def test_deterministic_output(self, mixed_chunks: list[DocumentChunk]) -> None:
        """Multiple calls produce identical item sequences."""
        res1 = prepare_for_embedding(mixed_chunks)
        res2 = prepare_for_embedding(mixed_chunks)
        assert [r.chunk_id for r in res1.items] == [r.chunk_id for r in res2.items]


# ── Counters & Helper Methods Tests ──────────────────────────────────────


class TestCountersAndHelpers:
    """Tests for counter properties and filtering helpers."""

    def test_counter_properties(self, mixed_chunks: list[DocumentChunk]) -> None:
        """Check total, text, table, and image counters."""
        result = prepare_for_embedding(mixed_chunks)
        assert result.total_items == 4
        assert result.text_items == 2
        assert result.table_items == 1
        assert result.image_items == 1

    def test_get_items_by_type(self, mixed_chunks: list[DocumentChunk]) -> None:
        """Filter items by content type."""
        result = prepare_for_embedding(mixed_chunks)
        assert len(result.get_items_by_type("text")) == 2
        assert len(result.get_items_by_type("table")) == 1
        assert len(result.get_items_by_type("image")) == 1
        assert len(result.get_items_by_type("audio")) == 0

    def test_get_items_on_page(self, mixed_chunks: list[DocumentChunk]) -> None:
        """Filter items by page number."""
        result = prepare_for_embedding(mixed_chunks)
        assert len(result.get_items_on_page(1)) == 2
        assert len(result.get_items_on_page(2)) == 1
        assert len(result.get_items_on_page(3)) == 1
        assert len(result.get_items_on_page(99)) == 0


# ── Error Handling Tests ─────────────────────────────────────────────────


class TestErrorHandling:
    """Tests for rejection of invalid chunks."""

    def test_missing_content_raises_value_error(self, sample_doc_id: str) -> None:
        """Chunk with empty content raises ValueError."""
        bad_chunk = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id=sample_doc_id,
            filename="a.pdf",
            page_number=1,
            content="",
            content_type="text",
        )
        with pytest.raises(ValueError, match="Chunk validation failed"):
            prepare_for_embedding([bad_chunk])

    def test_whitespace_only_content_raises_value_error(self, sample_doc_id: str) -> None:
        """Chunk with whitespace content raises ValueError."""
        bad_chunk = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id=sample_doc_id,
            filename="a.pdf",
            page_number=1,
            content="   \n\t  ",
            content_type="text",
        )
        with pytest.raises(ValueError, match="Chunk validation failed"):
            prepare_for_embedding([bad_chunk])

    def test_duplicate_chunk_ids_raises_value_error(self, sample_doc_id: str) -> None:
        """Duplicate chunk IDs raise ValueError."""
        shared_id = str(uuid.uuid4())
        c1 = DocumentChunk(
            chunk_id=shared_id,
            chunk_index=0,
            document_id=sample_doc_id,
            filename="a.pdf",
            page_number=1,
            content="Text 1",
            content_type="text",
        )
        c2 = DocumentChunk(
            chunk_id=shared_id,
            chunk_index=1,
            document_id=sample_doc_id,
            filename="a.pdf",
            page_number=1,
            content="Text 2",
            content_type="text",
        )
        with pytest.raises(ValueError, match="Chunk validation failed"):
            prepare_for_embedding([c1, c2])

    def test_inconsistent_document_id_raises_value_error(self) -> None:
        """Inconsistent document_ids raise ValueError."""
        c1 = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id="doc-1",
            filename="a.pdf",
            page_number=1,
            content="Text 1",
            content_type="text",
        )
        c2 = DocumentChunk(
            chunk_id="c2",
            chunk_index=1,
            document_id="doc-2",
            filename="a.pdf",
            page_number=1,
            content="Text 2",
            content_type="text",
        )
        with pytest.raises(ValueError, match="Chunk validation failed"):
            prepare_for_embedding([c1, c2])

    def test_invalid_input_type_raises_type_error(self) -> None:
        """Non-list / non-ChunkingResult raises TypeError."""
        with pytest.raises(TypeError, match="Expected list"):
            prepare_for_embedding("not a valid input")  # type: ignore
