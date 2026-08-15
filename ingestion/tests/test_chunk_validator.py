"""
Tests for the chunk validator and normalizer module.

Comprehensive tests covering valid chunks, empty lists, missing/duplicate IDs,
index validation, content integrity, content types, page numbers, metadata,
mixed error collections, and normalization.
"""

from __future__ import annotations

import uuid
import pytest

from ingestion.chunk_validator import normalize_chunks, validate_chunks
from ingestion.models import (
    ChunkValidationResult,
    ChunkingResult,
    DocumentChunk,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def valid_chunks() -> list[DocumentChunk]:
    """Create a list of 3 valid DocumentChunk instances."""
    doc_id = str(uuid.uuid4())
    return [
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            chunk_index=0,
            document_id=doc_id,
            filename="report.pdf",
            page_number=1,
            content="This is the first valid text chunk.",
            content_type="text",
            metadata={"char_count": 35},
        ),
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            chunk_index=1,
            document_id=doc_id,
            filename="report.pdf",
            page_number=1,
            content="| Header | Value |\n| --- | --- |\n| A | 1 |",
            content_type="table",
            metadata={"table_index": 0, "rows": 2, "columns": 2},
        ),
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            chunk_index=2,
            document_id=doc_id,
            filename="report.pdf",
            page_number=2,
            content="[Image on Page 2: format=PNG, width=100px, height=100px]",
            content_type="image",
            metadata={"image_index": 0, "image_format": "png", "width": 100, "height": 100},
        ),
    ]


@pytest.fixture
def valid_chunking_result(valid_chunks: list[DocumentChunk]) -> ChunkingResult:
    """Create a valid ChunkingResult instance."""
    return ChunkingResult(
        document_id=valid_chunks[0].document_id,
        filename="report.pdf",
        chunks=valid_chunks,
    )


# ── Success Tests ────────────────────────────────────────────────────────


class TestValidateChunksSuccess:
    """Tests for successful chunk validation."""

    def test_valid_chunks_list(self, valid_chunks: list[DocumentChunk]) -> None:
        """Valid list of DocumentChunks passes with is_valid=True and 0 errors."""
        result = validate_chunks(valid_chunks)
        assert isinstance(result, ChunkValidationResult)
        assert result.is_valid is True
        assert result.total_chunks == 3
        assert result.valid_chunks == 3
        assert result.invalid_chunks == 0
        assert len(result.errors) == 0

    def test_valid_chunking_result_object(self, valid_chunking_result: ChunkingResult) -> None:
        """ChunkingResult object is accepted and passes validation."""
        result = validate_chunks(valid_chunking_result)
        assert result.is_valid is True
        assert result.total_chunks == 3
        assert result.valid_chunks == 3
        assert result.invalid_chunks == 0

    def test_empty_chunk_list(self) -> None:
        """Empty list is valid with 0 chunks and a warning."""
        result = validate_chunks([])
        assert result.is_valid is True
        assert result.total_chunks == 0
        assert result.valid_chunks == 0
        assert result.invalid_chunks == 0
        assert len(result.errors) == 0
        assert len(result.warnings) > 0

    def test_none_page_number_is_valid(self, valid_chunks: list[DocumentChunk]) -> None:
        """page_number=None is valid (e.g. for whole-document summary chunks)."""
        chunks = [
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                chunk_index=0,
                document_id=valid_chunks[0].document_id,
                filename="report.pdf",
                page_number=None,
                content="Document level summary.",
                content_type="text",
            )
        ]
        result = validate_chunks(chunks)
        assert result.is_valid is True
        assert result.valid_chunks == 1


# ── Identity and Index Validation Tests ───────────────────────────────────


class TestChunkIdentityAndIndexValidation:
    """Tests for chunk_id, document_id, and chunk_index validation."""

    def test_missing_or_empty_chunk_id(self, valid_chunks: list[DocumentChunk]) -> None:
        """Missing or empty chunk_id is flagged as invalid."""
        bad_chunk = DocumentChunk(
            chunk_id="",
            chunk_index=0,
            document_id=valid_chunks[0].document_id,
            filename="report.pdf",
            page_number=1,
            content="Some text",
            content_type="text",
        )
        result = validate_chunks([bad_chunk])
        assert result.is_valid is False
        assert result.invalid_chunks == 1
        assert any("chunk_id" in err for err in result.errors)

    def test_duplicate_chunk_id(self, valid_chunks: list[DocumentChunk]) -> None:
        """Duplicate chunk_ids across chunks are flagged as invalid."""
        shared_id = str(uuid.uuid4())
        c1 = DocumentChunk(
            chunk_id=shared_id,
            chunk_index=0,
            document_id="doc-1",
            filename="a.pdf",
            page_number=1,
            content="Text 1",
            content_type="text",
        )
        c2 = DocumentChunk(
            chunk_id=shared_id,
            chunk_index=1,
            document_id="doc-1",
            filename="a.pdf",
            page_number=1,
            content="Text 2",
            content_type="text",
        )
        result = validate_chunks([c1, c2])
        assert result.is_valid is False
        assert any("Duplicate chunk_id" in err for err in result.errors)

    def test_missing_or_empty_document_id(self) -> None:
        """Empty document_id is flagged as invalid."""
        chunk = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id="",
            filename="a.pdf",
            page_number=1,
            content="Text",
            content_type="text",
        )
        result = validate_chunks([chunk])
        assert result.is_valid is False
        assert any("document_id" in err for err in result.errors)

    def test_inconsistent_document_id(self) -> None:
        """Different document_ids across the same chunk batch are flagged."""
        c1 = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id="doc-A",
            filename="a.pdf",
            page_number=1,
            content="Text 1",
            content_type="text",
        )
        c2 = DocumentChunk(
            chunk_id="c2",
            chunk_index=1,
            document_id="doc-B",
            filename="a.pdf",
            page_number=1,
            content="Text 2",
            content_type="text",
        )
        result = validate_chunks([c1, c2])
        assert result.is_valid is False
        assert any("Inconsistent document_id" in err for err in result.errors)

    def test_invalid_chunk_index(self) -> None:
        """Negative chunk_index is flagged as invalid."""
        chunk = DocumentChunk(
            chunk_id="c1",
            chunk_index=-1,
            document_id="doc-1",
            filename="a.pdf",
            page_number=1,
            content="Text",
            content_type="text",
        )
        result = validate_chunks([chunk])
        assert result.is_valid is False
        assert any("chunk_index" in err for err in result.errors)

    def test_duplicate_chunk_index(self) -> None:
        """Duplicate chunk_index is flagged as invalid."""
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
            chunk_index=0,
            document_id="doc-1",
            filename="a.pdf",
            page_number=1,
            content="Text 2",
            content_type="text",
        )
        result = validate_chunks([c1, c2])
        assert result.is_valid is False
        assert any("Duplicate chunk_index" in err for err in result.errors)


# ── Content and Content-Type Validation Tests ────────────────────────────


class TestContentAndTypeValidation:
    """Tests for content validity, content types, and page numbers."""

    def test_empty_content(self) -> None:
        """Empty string content is flagged as invalid."""
        chunk = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id="doc-1",
            filename="a.pdf",
            page_number=1,
            content="",
            content_type="text",
        )
        result = validate_chunks([chunk])
        assert result.is_valid is False
        assert any("content is empty" in err.lower() for err in result.errors)

    def test_whitespace_only_content(self) -> None:
        """Whitespace-only content is flagged as invalid."""
        chunk = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id="doc-1",
            filename="a.pdf",
            page_number=1,
            content="   \n\t  \r\n  ",
            content_type="text",
        )
        result = validate_chunks([chunk])
        assert result.is_valid is False
        assert any("empty or whitespace" in err.lower() for err in result.errors)

    def test_invalid_content_type(self) -> None:
        """Unsupported content_type is flagged as invalid."""
        chunk = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id="doc-1",
            filename="a.pdf",
            page_number=1,
            content="Some audio transcript",
            content_type="audio",
        )
        result = validate_chunks([chunk])
        assert result.is_valid is False
        assert any("content_type" in err for err in result.errors)

    def test_invalid_page_number_zero_or_negative(self) -> None:
        """page_number <= 0 is flagged as invalid."""
        c1 = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id="doc-1",
            filename="a.pdf",
            page_number=0,
            content="Text",
            content_type="text",
        )
        result = validate_chunks([c1])
        assert result.is_valid is False
        assert any("page_number" in err for err in result.errors)

    def test_invalid_metadata_type(self) -> None:
        """Non-dict metadata is flagged as invalid."""
        # Using object.__setattr__ since dataclass is frozen
        chunk = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id="doc-1",
            filename="a.pdf",
            page_number=1,
            content="Text",
            content_type="text",
            metadata="not a dict",  # type: ignore
        )
        result = validate_chunks([chunk])
        assert result.is_valid is False
        assert any("metadata" in err for err in result.errors)


# ── Mixed Errors, Duplicate Content, and Edge Cases ──────────────────────


class TestMixedErrorsAndWarnings:
    """Tests for duplicate content warnings and mixed batch handling."""

    def test_duplicate_content_generates_warning(self) -> None:
        """Identical content across distinct chunks generates a warning but remains valid."""
        c1 = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id="doc-1",
            filename="a.pdf",
            page_number=1,
            content="Exact duplicate text",
            content_type="text",
        )
        c2 = DocumentChunk(
            chunk_id="c2",
            chunk_index=1,
            document_id="doc-1",
            filename="a.pdf",
            page_number=2,
            content="Exact duplicate text",
            content_type="text",
        )
        result = validate_chunks([c1, c2])
        assert result.is_valid is True
        assert result.total_chunks == 2
        assert len(result.warnings) > 0
        assert any("Duplicate content detected" in w for w in result.warnings)

    def test_mixed_valid_and_invalid_chunks(self, valid_chunks: list[DocumentChunk]) -> None:
        """Correctly counts valid vs invalid chunks in mixed collections."""
        bad_chunk = DocumentChunk(
            chunk_id="",
            chunk_index=99,
            document_id="doc-1",
            filename="a.pdf",
            page_number=1,
            content="",
            content_type="invalid_type",
        )
        mixed = list(valid_chunks) + [bad_chunk]
        result = validate_chunks(mixed)
        assert result.is_valid is False
        assert result.total_chunks == 4
        assert result.valid_chunks == 3
        assert result.invalid_chunks == 1
        assert len(result.errors) > 0

    def test_invalid_input_type(self) -> None:
        """Non-list/non-ChunkingResult input returns is_valid=False."""
        result = validate_chunks("invalid string input")  # type: ignore
        assert result.is_valid is False
        assert result.total_chunks == 0
        assert any("Invalid input type" in err for err in result.errors)


# ── Normalization Tests ──────────────────────────────────────────────────


class TestNormalizeChunks:
    """Tests for the normalize_chunks function."""

    def test_trims_leading_and_trailing_whitespace(self) -> None:
        """Leading and trailing whitespace is stripped from content."""
        chunk = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id="doc-1",
            filename="a.pdf",
            page_number=1,
            content="  \n\n  Text with spaces around it.   \n\n  ",
            content_type="text",
            metadata={"key": "val"},
        )
        normalized = normalize_chunks([chunk])
        assert len(normalized) == 1
        assert normalized[0].content == "Text with spaces around it."
        assert normalized[0].metadata == {"key": "val"}
        assert normalized[0].chunk_id == "c1"

    def test_normalizes_excessive_blank_lines(self) -> None:
        """Multiple blank lines (>2 newlines) are reduced to double newlines."""
        chunk = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id="doc-1",
            filename="a.pdf",
            page_number=1,
            content="Paragraph 1\n\n\n\n\nParagraph 2",
            content_type="text",
        )
        normalized = normalize_chunks([chunk])
        assert normalized[0].content == "Paragraph 1\n\nParagraph 2"

    def test_preserves_all_metadata_and_attributes(self, valid_chunks: list[DocumentChunk]) -> None:
        """Normalization preserves all IDs, indices, filenames, page numbers, and metadata."""
        normalized = normalize_chunks(valid_chunks)
        assert len(normalized) == len(valid_chunks)
        for orig, norm in zip(valid_chunks, normalized):
            assert norm.chunk_id == orig.chunk_id
            assert norm.chunk_index == orig.chunk_index
            assert norm.document_id == orig.document_id
            assert norm.filename == orig.filename
            assert norm.page_number == orig.page_number
            assert norm.content_type == orig.content_type
            assert norm.metadata == orig.metadata

    def test_accepts_chunking_result(self, valid_chunking_result: ChunkingResult) -> None:
        """normalize_chunks accepts a ChunkingResult instance."""
        normalized = normalize_chunks(valid_chunking_result)
        assert len(normalized) == 3

    def test_invalid_type_raises_type_error(self) -> None:
        """Invalid input type raises TypeError."""
        with pytest.raises(TypeError, match="Expected list"):
            normalize_chunks("not a list")  # type: ignore
