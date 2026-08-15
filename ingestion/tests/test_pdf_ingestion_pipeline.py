"""
Tests for the unified PDF ingestion pipeline.

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
from ingestion.models import (
    DocumentMetadata,
    ExtractedImage,
    ExtractedTable,
    IngestionResult,
    PageData,
)
from ingestion.pdf_ingestion_pipeline import ingest_pdf


# ── Fixtures ─────────────────────────────────────────────────────────────


def make_dummy_image_bytes(width: int = 60, height: int = 40, color: int = 128) -> bytes:
    """Create raw PNG image bytes using PyMuPDF Pixmap."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), 0)
    pix.clear_with(color)
    return pix.tobytes("png")


def draw_table(
    page: pymupdf.Page,
    start_x: float,
    start_y: float,
    col_widths: list[float],
    row_heights: list[float],
    data: list[list[str]],
) -> None:
    """Helper to draw a grid table with lines and text in a PyMuPDF page."""
    total_width = sum(col_widths)
    total_height = sum(row_heights)

    # Draw outer rectangle
    rect = pymupdf.Rect(start_x, start_y, start_x + total_width, start_y + total_height)
    page.draw_rect(rect, color=(0, 0, 0), width=1)

    # Draw horizontal grid lines
    curr_y = start_y
    for h in row_heights[:-1]:
        curr_y += h
        page.draw_line(
            pymupdf.Point(start_x, curr_y),
            pymupdf.Point(start_x + total_width, curr_y),
            color=(0, 0, 0),
            width=1,
        )

    # Draw vertical grid lines
    curr_x = start_x
    for w in col_widths[:-1]:
        curr_x += w
        page.draw_line(
            pymupdf.Point(curr_x, start_y),
            pymupdf.Point(curr_x, start_y + total_height),
            color=(0, 0, 0),
            width=1,
        )

    # Insert cell texts
    curr_y = start_y
    for r_idx, row in enumerate(data):
        curr_x = start_x
        for c_idx, val in enumerate(row):
            if val:
                page.insert_text(
                    pymupdf.Point(curr_x + 5, curr_y + row_heights[r_idx] - 6),
                    val,
                    fontsize=10,
                )
            curr_x += col_widths[c_idx]
        curr_y += row_heights[r_idx]


@pytest.fixture
def multi_modal_pdf(tmp_path: Path) -> Path:
    """Create a multi-page PDF containing text, tables, and images across pages.

    - Page 1: Heading text, table, and embedded image
    - Page 2: Text summary and second table
    - Page 3: Empty page (no text, tables, or images)
    """
    pdf_path = tmp_path / "multi_modal.pdf"
    doc = pymupdf.open()

    # ── Page 1: Text + Table + Image ──
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((72, 50), "OmniBrain Multi-Modal Report", fontsize=16)
    p1.insert_text((72, 75), "Section 1: Performance metrics and overview.", fontsize=11)

    # Table on Page 1
    table_data = [
        ["Model", "Accuracy", "Latency"],
        ["VLM-Alpha", "94.5%", "120ms"],
        ["VLM-Beta", "96.2%", "145ms"],
    ]
    draw_table(
        p1,
        start_x=72,
        start_y=95,
        col_widths=[100, 80, 80],
        row_heights=[22, 22, 22],
        data=table_data,
    )

    # Image on Page 1
    img1 = make_dummy_image_bytes(width=120, height=80, color=90)
    p1.insert_image(pymupdf.Rect(72, 180, 192, 260), stream=img1)

    # ── Page 2: Text + Table ──
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((72, 50), "Section 2: Database Summary", fontsize=14)

    table2_data = [
        ["Region", "Queries"],
        ["US-East", "12,400"],
        ["EU-West", "8,900"],
    ]
    draw_table(
        p2,
        start_x=72,
        start_y=80,
        col_widths=[100, 100],
        row_heights=[22, 22, 22],
        data=table2_data,
    )

    # ── Page 3: Empty page ──
    doc.new_page(width=612, height=792)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def text_only_pdf(tmp_path: Path) -> Path:
    """Create a PDF with text only (no tables, no images)."""
    pdf_path = tmp_path / "text_only.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "This is plain text with no tables and no images.", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def image_only_pdf(tmp_path: Path) -> Path:
    """Create a PDF with an image only (no text, no tables)."""
    pdf_path = tmp_path / "image_only.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    img = make_dummy_image_bytes(width=150, height=100, color=200)
    page.insert_image(pymupdf.Rect(72, 72, 222, 172), stream=img)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
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
    pdf_path = tmp_path / "corrupted.pdf"
    pdf_path.write_bytes(b"\x00\xff invalid binary")
    return pdf_path


@pytest.fixture
def non_pdf_file(tmp_path: Path) -> Path:
    """Create a non-pdf file."""
    txt_path = tmp_path / "readme.txt"
    txt_path.write_text("Hello world", encoding="utf-8")
    return txt_path


# ── Pipeline Tests ───────────────────────────────────────────────────────


class TestUnifiedPipelineCombined:
    """Tests for unified ingestion of multi-modal PDF containing text, tables, and images."""

    def test_returns_ingestion_result(self, multi_modal_pdf: Path) -> None:
        """ingest_pdf should return an IngestionResult instance."""
        result = ingest_pdf(multi_modal_pdf)
        assert isinstance(result, IngestionResult)

    def test_metadata_fields(self, multi_modal_pdf: Path) -> None:
        """Metadata should have valid document_id, filename, total_pages, content_type."""
        result = ingest_pdf(multi_modal_pdf)
        assert isinstance(result.metadata, DocumentMetadata)
        parsed_uuid = uuid.UUID(result.metadata.document_id)
        assert parsed_uuid.version == 4
        assert result.metadata.filename == "multi_modal.pdf"
        assert result.metadata.total_pages == 3
        assert result.metadata.content_type == "application/pdf"

    def test_text_integration(self, multi_modal_pdf: Path) -> None:
        """Text extraction should be integrated and preserve page numbers."""
        result = ingest_pdf(multi_modal_pdf)
        assert len(result.pages) == 3
        assert result.total_pages == 3
        assert result.pages_with_text == 2
        assert result.pages_without_text == 1
        assert result.has_text is True

        p1 = result.get_page(1)
        assert p1 is not None
        assert "OmniBrain Multi-Modal Report" in p1.text
        assert p1.has_content is True

        p3 = result.get_page(3)
        assert p3 is not None
        assert p3.has_content is False
        assert p3.text == ""

    def test_table_integration(self, multi_modal_pdf: Path) -> None:
        """Table extraction should be integrated with correct counts and pages."""
        result = ingest_pdf(multi_modal_pdf)
        assert result.total_tables == 2
        assert result.has_tables is True

        p1_tables = result.get_tables_on_page(1)
        assert len(p1_tables) == 1
        assert p1_tables[0].page_number == 1
        assert p1_tables[0].rows == 3

        p2_tables = result.get_tables_on_page(2)
        assert len(p2_tables) == 1
        assert p2_tables[0].page_number == 2

        p3_tables = result.get_tables_on_page(3)
        assert len(p3_tables) == 0

    def test_image_integration(self, multi_modal_pdf: Path) -> None:
        """Image extraction should be integrated with correct counts and bytes."""
        result = ingest_pdf(multi_modal_pdf)
        assert result.total_images == 1
        assert result.has_images is True

        p1_images = result.get_images_on_page(1)
        assert len(p1_images) == 1
        img = p1_images[0]
        assert img.page_number == 1
        assert img.image_index == 0
        assert img.width == 120
        assert img.height == 80
        assert len(img.image_bytes) > 0
        assert img.size_bytes == len(img.image_bytes)

        p2_images = result.get_images_on_page(2)
        assert len(p2_images) == 0

    def test_get_all_text_helper(self, multi_modal_pdf: Path) -> None:
        """get_all_text should concatenate text across all content pages."""
        result = ingest_pdf(multi_modal_pdf)
        full_text = result.get_all_text()
        assert "OmniBrain Multi-Modal Report" in full_text
        assert "Database Summary" in full_text

    def test_accepts_string_path(self, multi_modal_pdf: Path) -> None:
        """ingest_pdf should accept string paths."""
        result = ingest_pdf(str(multi_modal_pdf))
        assert result.total_pages == 3
        assert result.has_text is True


class TestUnifiedPipelineSpecificModalities:
    """Tests for single-modality PDFs."""

    def test_text_only_pdf(self, text_only_pdf: Path) -> None:
        """Text-only PDF has text but no tables and no images."""
        result = ingest_pdf(text_only_pdf)
        assert result.has_text is True
        assert result.has_tables is False
        assert result.has_images is False
        assert result.total_tables == 0
        assert result.total_images == 0
        assert len(result.tables) == 0
        assert len(result.images) == 0

    def test_image_only_pdf(self, image_only_pdf: Path) -> None:
        """Image-only PDF has images but no text and no tables."""
        result = ingest_pdf(image_only_pdf)
        assert result.has_images is True
        assert result.total_images == 1
        assert result.has_tables is False
        assert result.total_tables == 0
        assert result.has_text is False


class TestUnifiedPipelineEmptyAndNoContent:
    """Tests for empty and no-content documents."""

    def test_empty_pdf(self, empty_pdf: Path) -> None:
        """Completely empty PDF is handled gracefully with 0 counts."""
        result = ingest_pdf(empty_pdf)
        assert result.total_pages == 1
        assert result.pages_with_text == 0
        assert result.pages_without_text == 1
        assert result.has_text is False
        assert result.has_tables is False
        assert result.has_images is False
        assert result.total_tables == 0
        assert result.total_images == 0

    def test_get_page_nonexistent_returns_none(self, empty_pdf: Path) -> None:
        """get_page returns None for invalid page number."""
        result = ingest_pdf(empty_pdf)
        assert result.get_page(999) is None


class TestUnifiedPipelineErrorHandling:
    """Tests for error handling in unified pipeline."""

    def test_missing_file_raises_not_found(self, tmp_path: Path) -> None:
        """Non-existent file raises PDFNotFoundError."""
        with pytest.raises(PDFNotFoundError):
            ingest_pdf(tmp_path / "does_not_exist.pdf")

    def test_invalid_extension_raises_invalid_type(self, non_pdf_file: Path) -> None:
        """Non-pdf extension raises InvalidFileTypeError."""
        with pytest.raises(InvalidFileTypeError):
            ingest_pdf(non_pdf_file)

    def test_corrupted_file_raises_corrupted_error(self, corrupted_pdf: Path) -> None:
        """Corrupted PDF raises CorruptedPDFError."""
        with pytest.raises(CorruptedPDFError):
            ingest_pdf(corrupted_pdf)
