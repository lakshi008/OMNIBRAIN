"""
Tests for the PDF text extractor module.

All test PDFs are created in-memory using PyMuPDF — no external files needed.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pymupdf
import pytest

from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
)
from ingestion.models import DocumentMetadata, PageData, ParsedDocument
from ingestion.pdf_text_extractor import extract_text, validate_pdf


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a simple 3-page PDF with known text content."""
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()

    page_texts = [
        "Hello World. This is page one.",
        "Page two contains different content.",
        "Final page number three.",
    ]

    for text in page_texts:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), text, fontsize=12)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def single_page_pdf(tmp_path: Path) -> Path:
    """Create a single-page PDF."""
    pdf_path = tmp_path / "single.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Only one page here.", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def empty_page_pdf(tmp_path: Path) -> Path:
    """Create a PDF with one empty page and one page with text."""
    pdf_path = tmp_path / "empty_page.pdf"
    doc = pymupdf.open()

    # Page 1: empty (no text inserted)
    doc.new_page(width=612, height=792)

    # Page 2: has text
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "This page has text.", fontsize=12)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def all_empty_pdf(tmp_path: Path) -> Path:
    """Create a PDF where all pages are empty (no extractable text)."""
    pdf_path = tmp_path / "all_empty.pdf"
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.new_page(width=612, height=792)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def corrupted_pdf(tmp_path: Path) -> Path:
    """Create a file with .pdf extension but random garbage bytes."""
    pdf_path = tmp_path / "corrupted.pdf"
    pdf_path.write_bytes(b"\x00\x01\x02\xff\xfe\xfd garbage data not a pdf")
    return pdf_path


@pytest.fixture
def non_pdf_file(tmp_path: Path) -> Path:
    """Create a plain text file with .txt extension."""
    txt_path = tmp_path / "readme.txt"
    txt_path.write_text("This is not a PDF.", encoding="utf-8")
    return txt_path


# ── Validation Tests ─────────────────────────────────────────────────────


class TestValidatePDF:
    """Tests for the validate_pdf function."""

    def test_valid_pdf_passes(self, sample_pdf: Path) -> None:
        """A valid PDF should pass validation and return a resolved Path."""
        result = validate_pdf(sample_pdf)
        assert result.exists()
        assert result.suffix == ".pdf"

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """A path to a non-existent file should raise PDFNotFoundError."""
        fake_path = tmp_path / "does_not_exist.pdf"
        with pytest.raises(PDFNotFoundError) as exc_info:
            validate_pdf(fake_path)
        assert "does_not_exist.pdf" in str(exc_info.value)

    def test_wrong_extension_raises(self, non_pdf_file: Path) -> None:
        """A file with non-.pdf extension should raise InvalidFileTypeError."""
        with pytest.raises(InvalidFileTypeError) as exc_info:
            validate_pdf(non_pdf_file)
        assert ".txt" in str(exc_info.value)

    def test_corrupted_pdf_raises(self, corrupted_pdf: Path) -> None:
        """A corrupted file with .pdf extension should raise CorruptedPDFError."""
        with pytest.raises(CorruptedPDFError):
            validate_pdf(corrupted_pdf)

    def test_accepts_string_path(self, sample_pdf: Path) -> None:
        """validate_pdf should accept string paths as well as Path objects."""
        result = validate_pdf(str(sample_pdf))
        assert result.exists()


# ── Extraction Tests ─────────────────────────────────────────────────────


class TestExtractText:
    """Tests for the extract_text function."""

    def test_returns_parsed_document(self, sample_pdf: Path) -> None:
        """extract_text should return a ParsedDocument instance."""
        result = extract_text(sample_pdf)
        assert isinstance(result, ParsedDocument)

    def test_correct_page_count(self, sample_pdf: Path) -> None:
        """Should extract the correct number of pages."""
        result = extract_text(sample_pdf)
        assert len(result.pages) == 3
        assert result.metadata.total_pages == 3

    def test_page_numbers_are_sequential(self, sample_pdf: Path) -> None:
        """Page numbers should be 1-indexed and sequential."""
        result = extract_text(sample_pdf)
        page_numbers = [p.page_number for p in result.pages]
        assert page_numbers == [1, 2, 3]

    def test_text_content_extracted(self, sample_pdf: Path) -> None:
        """Extracted text should contain the content inserted into the PDF."""
        result = extract_text(sample_pdf)
        assert "Hello World" in result.pages[0].text
        assert "page one" in result.pages[0].text
        assert "Page two" in result.pages[1].text
        assert "Final page" in result.pages[2].text

    def test_char_count_matches_text_length(self, sample_pdf: Path) -> None:
        """char_count should equal len(text) for each page."""
        result = extract_text(sample_pdf)
        for page in result.pages:
            assert page.char_count == len(page.text)

    def test_has_content_flag(self, sample_pdf: Path) -> None:
        """Pages with text should have has_content=True."""
        result = extract_text(sample_pdf)
        for page in result.pages:
            assert page.has_content is True

    def test_single_page_pdf(self, single_page_pdf: Path) -> None:
        """Single-page PDF should work correctly."""
        result = extract_text(single_page_pdf)
        assert result.metadata.total_pages == 1
        assert len(result.pages) == 1
        assert "Only one page" in result.pages[0].text


# ── Empty Page Handling Tests ────────────────────────────────────────────


class TestEmptyPageHandling:
    """Tests for handling PDFs with empty or no-text pages."""

    def test_empty_page_has_no_content(self, empty_page_pdf: Path) -> None:
        """Empty pages should have has_content=False and empty text."""
        result = extract_text(empty_page_pdf)
        page1 = result.pages[0]
        assert page1.has_content is False
        assert page1.text == ""
        assert page1.char_count == 0

    def test_mixed_content_pages(self, empty_page_pdf: Path) -> None:
        """PDF with mixed empty/content pages should track both correctly."""
        result = extract_text(empty_page_pdf)
        assert result.metadata.pages_with_content == 1
        assert result.metadata.pages_without_content == 1

    def test_all_empty_pages(self, all_empty_pdf: Path) -> None:
        """PDF where all pages are empty should still parse successfully."""
        result = extract_text(all_empty_pdf)
        assert result.metadata.total_pages == 2
        assert result.metadata.pages_with_content == 0
        assert result.metadata.pages_without_content == 2
        for page in result.pages:
            assert page.has_content is False


# ── Metadata Tests ───────────────────────────────────────────────────────


class TestMetadata:
    """Tests for document metadata generation."""

    def test_document_id_is_valid_uuid(self, sample_pdf: Path) -> None:
        """document_id should be a valid UUID4 string."""
        result = extract_text(sample_pdf)
        parsed_uuid = uuid.UUID(result.metadata.document_id)
        assert parsed_uuid.version == 4

    def test_filename_matches(self, sample_pdf: Path) -> None:
        """Metadata filename should match the input file's name."""
        result = extract_text(sample_pdf)
        assert result.metadata.filename == "sample.pdf"

    def test_content_type_is_pdf(self, sample_pdf: Path) -> None:
        """content_type should be 'application/pdf'."""
        result = extract_text(sample_pdf)
        assert result.metadata.content_type == "application/pdf"

    def test_created_at_is_iso_format(self, sample_pdf: Path) -> None:
        """created_at should be a valid ISO 8601 timestamp."""
        result = extract_text(sample_pdf)
        from datetime import datetime

        # Should not raise ValueError
        datetime.fromisoformat(result.metadata.created_at)

    def test_unique_document_ids(self, sample_pdf: Path) -> None:
        """Each call to extract_text should generate a unique document_id."""
        result1 = extract_text(sample_pdf)
        result2 = extract_text(sample_pdf)
        assert result1.metadata.document_id != result2.metadata.document_id


# ── Error Handling Tests ─────────────────────────────────────────────────


class TestErrorHandling:
    """Tests for error handling in extract_text."""

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Should raise PDFNotFoundError for missing files."""
        with pytest.raises(PDFNotFoundError):
            extract_text(tmp_path / "ghost.pdf")

    def test_wrong_extension(self, non_pdf_file: Path) -> None:
        """Should raise InvalidFileTypeError for non-PDF files."""
        with pytest.raises(InvalidFileTypeError):
            extract_text(non_pdf_file)

    def test_corrupted_file(self, corrupted_pdf: Path) -> None:
        """Should raise CorruptedPDFError for corrupted files."""
        with pytest.raises(CorruptedPDFError):
            extract_text(corrupted_pdf)


# ── Helper Method Tests ──────────────────────────────────────────────────


class TestParsedDocumentHelpers:
    """Tests for ParsedDocument helper methods."""

    def test_get_page_returns_correct_page(self, sample_pdf: Path) -> None:
        """get_page should return the page with the matching number."""
        result = extract_text(sample_pdf)
        page2 = result.get_page(2)
        assert page2 is not None
        assert page2.page_number == 2

    def test_get_page_returns_none_for_invalid(self, sample_pdf: Path) -> None:
        """get_page should return None for a non-existent page number."""
        result = extract_text(sample_pdf)
        assert result.get_page(999) is None

    def test_get_all_text_combines_pages(self, sample_pdf: Path) -> None:
        """get_all_text should concatenate text from all content pages."""
        result = extract_text(sample_pdf)
        full_text = result.get_all_text()
        assert "Hello World" in full_text
        assert "Page two" in full_text
        assert "Final page" in full_text

    def test_get_all_text_skips_empty_pages(
        self, empty_page_pdf: Path
    ) -> None:
        """get_all_text should skip empty pages."""
        result = extract_text(empty_page_pdf)
        full_text = result.get_all_text()
        assert "This page has text" in full_text
        # Should not have double separators from empty pages
        assert "\n\n\n\n" not in full_text
