"""
Tests for the PDF image extractor module.

All test PDFs are created in-memory using PyMuPDF.
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
from ingestion.models import DocumentMetadata, ExtractedImage, ImageExtractionResult
from ingestion.pdf_image_extractor import extract_images


# ── Fixtures ─────────────────────────────────────────────────────────────


def make_dummy_image_bytes(width: int = 60, height: int = 40, color: int = 128) -> bytes:
    """Create raw PNG image bytes using PyMuPDF Pixmap."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), 0)
    pix.clear_with(color)
    return pix.tobytes("png")


@pytest.fixture
def pdf_with_single_image(tmp_path: Path) -> Path:
    """Create a PDF containing a single embedded PNG image on Page 1."""
    pdf_path = tmp_path / "single_image.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text((72, 50), "Document with Single Image", fontsize=14)

    img_bytes = make_dummy_image_bytes(width=100, height=80, color=200)
    page.insert_image(pymupdf.Rect(72, 80, 172, 160), stream=img_bytes)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_with_multiple_images_single_page(tmp_path: Path) -> Path:
    """Create a PDF with 2 embedded images on Page 1."""
    pdf_path = tmp_path / "multiple_images_single_page.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text((72, 40), "Two Images on One Page", fontsize=14)

    img1 = make_dummy_image_bytes(width=50, height=50, color=100)
    img2 = make_dummy_image_bytes(width=80, height=60, color=180)

    page.insert_image(pymupdf.Rect(72, 60, 122, 110), stream=img1)
    page.insert_image(pymupdf.Rect(72, 140, 152, 200), stream=img2)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_with_images_multi_page(tmp_path: Path) -> Path:
    """Create a 3-page PDF with images on Page 1 & Page 2, Page 3 is empty."""
    pdf_path = tmp_path / "multi_page_images.pdf"
    doc = pymupdf.open()

    # Page 1
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((72, 40), "Page 1 with image", fontsize=12)
    img1 = make_dummy_image_bytes(width=70, height=70, color=50)
    p1.insert_image(pymupdf.Rect(72, 60, 142, 130), stream=img1)

    # Page 2
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((72, 40), "Page 2 with image", fontsize=12)
    img2 = make_dummy_image_bytes(width=90, height=45, color=150)
    p2.insert_image(pymupdf.Rect(72, 60, 162, 105), stream=img2)

    # Page 3 (Empty)
    doc.new_page(width=612, height=792)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_no_images(tmp_path: Path) -> Path:
    """Create a PDF with text only (no images)."""
    pdf_path = tmp_path / "no_images.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Only plain text content here, no images.", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_empty(tmp_path: Path) -> Path:
    """Create a completely empty PDF."""
    pdf_path = tmp_path / "empty_doc.pdf"
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def corrupted_pdf(tmp_path: Path) -> Path:
    """Create a corrupted file with .pdf extension."""
    pdf_path = tmp_path / "bad.pdf"
    pdf_path.write_bytes(b"\x00\xff invalid pdf bytes")
    return pdf_path


@pytest.fixture
def non_pdf_file(tmp_path: Path) -> Path:
    """Create a .txt file."""
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("Not a PDF file.", encoding="utf-8")
    return txt_path


# ── Extraction Tests ─────────────────────────────────────────────────────


class TestExtractImages:
    """Tests for extracting images from PDFs."""

    def test_returns_image_extraction_result(self, pdf_with_single_image: Path) -> None:
        """extract_images should return ImageExtractionResult."""
        result = extract_images(pdf_with_single_image)
        assert isinstance(result, ImageExtractionResult)

    def test_single_image_detected(self, pdf_with_single_image: Path) -> None:
        """Should detect exactly 1 image on Page 1."""
        result = extract_images(pdf_with_single_image)
        assert result.total_images == 1
        assert result.has_images is True
        img = result.images[0]
        assert isinstance(img, ExtractedImage)
        assert img.page_number == 1
        assert img.image_index == 0

    def test_image_dimensions(self, pdf_with_single_image: Path) -> None:
        """Should extract correct image width and height."""
        result = extract_images(pdf_with_single_image)
        img = result.images[0]
        assert img.width == 100
        assert img.height == 80

    def test_image_format_and_bytes(self, pdf_with_single_image: Path) -> None:
        """Extracted image should have format, valid bytes, and size_bytes matching bytes length."""
        result = extract_images(pdf_with_single_image)
        img = result.images[0]
        assert img.image_format in ("png", "jpeg", "jpg")
        assert isinstance(img.image_bytes, bytes)
        assert len(img.image_bytes) > 0
        assert img.size_bytes == len(img.image_bytes)
        assert img.xref > 0


class TestMultipleImages:
    """Tests for multiple images in a PDF."""

    def test_multiple_images_same_page(self, pdf_with_multiple_images_single_page: Path) -> None:
        """Should detect 2 images on the same page with sequential indices."""
        result = extract_images(pdf_with_multiple_images_single_page)
        assert result.total_images == 2
        p1_images = result.get_images_on_page(1)
        assert len(p1_images) == 2
        assert p1_images[0].image_index == 0
        assert p1_images[1].image_index == 1

    def test_images_across_multiple_pages(self, pdf_with_images_multi_page: Path) -> None:
        """Should detect images across different pages and track page numbers."""
        result = extract_images(pdf_with_images_multi_page)
        assert result.total_images == 2
        p1_images = result.get_images_on_page(1)
        p2_images = result.get_images_on_page(2)
        p3_images = result.get_images_on_page(3)

        assert len(p1_images) == 1
        assert p1_images[0].page_number == 1
        assert len(p2_images) == 1
        assert p2_images[0].page_number == 2
        assert len(p3_images) == 0


class TestNoImagesAndEmptyPages:
    """Tests for PDFs without images or with empty pages."""

    def test_pdf_with_no_images(self, pdf_no_images: Path) -> None:
        """PDF with text and no images returns empty list gracefully."""
        result = extract_images(pdf_no_images)
        assert result.total_images == 0
        assert result.has_images is False
        assert result.images == []
        assert result.metadata.total_pages == 1

    def test_empty_pdf(self, pdf_empty: Path) -> None:
        """Completely empty PDF returns 0 images without error."""
        result = extract_images(pdf_empty)
        assert result.total_images == 0
        assert result.has_images is False
        assert result.metadata.total_pages == 1
        assert result.metadata.pages_with_content == 0
        assert result.metadata.pages_without_content == 1


class TestImageMetadata:
    """Tests for metadata generation in image extraction."""

    def test_metadata_fields(self, pdf_with_single_image: Path) -> None:
        """Metadata should contain valid document ID, total pages, and content type."""
        result = extract_images(pdf_with_single_image)
        assert isinstance(result.metadata, DocumentMetadata)
        parsed_uuid = uuid.UUID(result.metadata.document_id)
        assert parsed_uuid.version == 4
        assert result.metadata.filename == "single_image.pdf"
        assert result.metadata.total_pages == 1
        assert result.metadata.content_type == "application/pdf"

    def test_accepts_string_path(self, pdf_with_single_image: Path) -> None:
        """extract_images should accept string paths as well as Path objects."""
        result = extract_images(str(pdf_with_single_image))
        assert result.total_images == 1


class TestImageHelperMethods:
    """Tests for ImageExtractionResult helper methods."""

    def test_get_images_on_page_empty_if_none(self, pdf_with_single_image: Path) -> None:
        """get_images_on_page should return empty list for non-existent page numbers."""
        result = extract_images(pdf_with_single_image)
        assert result.get_images_on_page(999) == []


class TestErrorHandling:
    """Tests for error handling in image extraction."""

    def test_missing_file_raises_not_found(self, tmp_path: Path) -> None:
        """Missing file raises PDFNotFoundError."""
        with pytest.raises(PDFNotFoundError):
            extract_images(tmp_path / "non_existent.pdf")

    def test_invalid_extension_raises(self, non_pdf_file: Path) -> None:
        """Non-pdf extension raises InvalidFileTypeError."""
        with pytest.raises(InvalidFileTypeError):
            extract_images(non_pdf_file)

    def test_corrupted_file_raises(self, corrupted_pdf: Path) -> None:
        """Corrupted PDF file raises CorruptedPDFError."""
        with pytest.raises(CorruptedPDFError):
            extract_images(corrupted_pdf)
