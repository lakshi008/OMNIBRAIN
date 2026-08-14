"""
Tests for the PDF table extractor module.

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
from ingestion.models import DocumentMetadata, ExtractedTable, TableExtractionResult
from ingestion.pdf_table_extractor import extract_tables


# ── Fixtures ─────────────────────────────────────────────────────────────


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
                # insert text slightly offset into the cell
                page.insert_text(
                    pymupdf.Point(curr_x + 5, curr_y + row_heights[r_idx] - 6),
                    val,
                    fontsize=10,
                )
            curr_x += col_widths[c_idx]
        curr_y += row_heights[r_idx]


@pytest.fixture
def pdf_with_table(tmp_path: Path) -> Path:
    """Create a PDF containing a single 3x3 table on Page 1."""
    pdf_path = tmp_path / "table_single.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text((72, 50), "Document with Table", fontsize=14)

    table_data = [
        ["ID", "Name", "Department"],
        ["101", "Alice", "Engineering"],
        ["102", "Bob", "Design"],
    ]
    draw_table(
        page,
        start_x=72,
        start_y=80,
        col_widths=[60, 120, 140],
        row_heights=[25, 25, 25],
        data=table_data,
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_with_multiple_tables(tmp_path: Path) -> Path:
    """Create a PDF with 2 tables on Page 1."""
    pdf_path = tmp_path / "multiple_tables.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text((72, 40), "Two Tables Page", fontsize=14)

    # Table 1
    table1 = [
        ["Product", "Price"],
        ["Laptop", "$1000"],
    ]
    draw_table(
        page,
        start_x=72,
        start_y=60,
        col_widths=[150, 100],
        row_heights=[25, 25],
        data=table1,
    )

    # Table 2
    table2 = [
        ["City", "Country"],
        ["Paris", "France"],
        ["Tokyo", "Japan"],
    ]
    draw_table(
        page,
        start_x=72,
        start_y=160,
        col_widths=[120, 130],
        row_heights=[25, 25, 25],
        data=table2,
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_multi_page_tables(tmp_path: Path) -> Path:
    """Create a multi-page PDF with tables on page 1 and page 2, and empty page 3."""
    pdf_path = tmp_path / "multi_page_tables.pdf"
    doc = pymupdf.open()

    # Page 1 - Table
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((72, 40), "Page 1", fontsize=12)
    draw_table(
        p1,
        start_x=72,
        start_y=60,
        col_widths=[100, 100],
        row_heights=[25, 25],
        data=[["A", "B"], ["1", "2"]],
    )

    # Page 2 - Table
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((72, 40), "Page 2", fontsize=12)
    draw_table(
        p2,
        start_x=72,
        start_y=60,
        col_widths=[80, 80, 80],
        row_heights=[25, 25],
        data=[["X", "Y", "Z"], ["7", "8", "9"]],
    )

    # Page 3 - Empty page
    doc.new_page(width=612, height=792)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_no_tables(tmp_path: Path) -> Path:
    """Create a PDF with text only and no table lines/structure."""
    pdf_path = tmp_path / "no_tables.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Just a standard paragraph with no tables.", fontsize=12)
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
    pdf_path.write_bytes(b"\x00\xff random binary data")
    return pdf_path


@pytest.fixture
def non_pdf_file(tmp_path: Path) -> Path:
    """Create a .csv file."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b,c\n1,2,3", encoding="utf-8")
    return csv_path


# ── Extraction Tests ─────────────────────────────────────────────────────


class TestExtractTables:
    """Tests for extracting tables from PDFs."""

    def test_returns_table_extraction_result(self, pdf_with_table: Path) -> None:
        """extract_tables should return TableExtractionResult."""
        result = extract_tables(pdf_with_table)
        assert isinstance(result, TableExtractionResult)

    def test_single_table_detected(self, pdf_with_table: Path) -> None:
        """Should detect exactly 1 table on the page."""
        result = extract_tables(pdf_with_table)
        assert result.total_tables == 1
        assert result.has_tables is True
        table = result.tables[0]
        assert isinstance(table, ExtractedTable)
        assert table.page_number == 1
        assert table.table_index == 0

    def test_table_dimensions(self, pdf_with_table: Path) -> None:
        """Should detect correct number of rows and columns."""
        result = extract_tables(pdf_with_table)
        table = result.tables[0]
        assert table.rows == 3
        assert table.columns == 3
        assert len(table.cells) == 3
        assert len(table.cells[0]) == 3

    def test_table_cell_values(self, pdf_with_table: Path) -> None:
        """Extracted cell values should contain the expected texts."""
        result = extract_tables(pdf_with_table)
        cells = result.tables[0].cells
        # Verify header
        assert any("ID" in str(c) for c in cells[0])
        assert any("Name" in str(c) for c in cells[0])
        assert any("Department" in str(c) for c in cells[0])
        # Verify rows
        flat_cells = " ".join(str(c) for row in cells for c in row if c)
        assert "Alice" in flat_cells
        assert "Engineering" in flat_cells
        assert "Bob" in flat_cells
        assert "Design" in flat_cells


class TestMultipleTables:
    """Tests for multiple tables in a document."""

    def test_multiple_tables_same_page(self, pdf_with_multiple_tables: Path) -> None:
        """Should detect multiple tables on the same page with correct indices."""
        result = extract_tables(pdf_with_multiple_tables)
        assert result.total_tables == 2
        p1_tables = result.get_tables_on_page(1)
        assert len(p1_tables) == 2
        assert p1_tables[0].table_index == 0
        assert p1_tables[1].table_index == 1

    def test_tables_across_multiple_pages(self, pdf_multi_page_tables: Path) -> None:
        """Should detect tables across different pages."""
        result = extract_tables(pdf_multi_page_tables)
        assert result.total_tables == 2
        p1_tables = result.get_tables_on_page(1)
        p2_tables = result.get_tables_on_page(2)
        p3_tables = result.get_tables_on_page(3)

        assert len(p1_tables) == 1
        assert p1_tables[0].page_number == 1
        assert len(p2_tables) == 1
        assert p2_tables[0].page_number == 2
        assert len(p3_tables) == 0


class TestNoTablesAndEmptyPages:
    """Tests for PDFs with no tables or empty pages."""

    def test_pdf_with_no_tables(self, pdf_no_tables: Path) -> None:
        """PDF with regular text and no tables should return empty list gracefully."""
        result = extract_tables(pdf_no_tables)
        assert result.total_tables == 0
        assert result.has_tables is False
        assert result.tables == []
        assert result.metadata.total_pages == 1

    def test_empty_pdf(self, pdf_empty: Path) -> None:
        """Completely empty PDF should return 0 tables without error."""
        result = extract_tables(pdf_empty)
        assert result.total_tables == 0
        assert result.has_tables is False
        assert result.metadata.total_pages == 1
        assert result.metadata.pages_with_content == 0
        assert result.metadata.pages_without_content == 1


class TestTableMetadata:
    """Tests for metadata generation in table extraction."""

    def test_metadata_fields(self, pdf_with_table: Path) -> None:
        """Metadata should contain valid document ID, total pages, and content type."""
        result = extract_tables(pdf_with_table)
        assert isinstance(result.metadata, DocumentMetadata)
        parsed_uuid = uuid.UUID(result.metadata.document_id)
        assert parsed_uuid.version == 4
        assert result.metadata.filename == "table_single.pdf"
        assert result.metadata.total_pages == 1
        assert result.metadata.content_type == "application/pdf"

    def test_accepts_string_path(self, pdf_with_table: Path) -> None:
        """extract_tables should accept string paths as well as Path objects."""
        result = extract_tables(str(pdf_with_table))
        assert result.total_tables == 1


class TestTableHelperMethods:
    """Tests for TableExtractionResult helper methods."""

    def test_get_tables_on_page_empty_if_none(self, pdf_with_table: Path) -> None:
        """get_tables_on_page should return empty list for pages without tables."""
        result = extract_tables(pdf_with_table)
        assert result.get_tables_on_page(999) == []


class TestErrorHandling:
    """Tests for error handling in table extraction."""

    def test_missing_file_raises_not_found(self, tmp_path: Path) -> None:
        """Missing file raises PDFNotFoundError."""
        with pytest.raises(PDFNotFoundError):
            extract_tables(tmp_path / "non_existent.pdf")

    def test_invalid_extension_raises(self, non_pdf_file: Path) -> None:
        """Non-pdf extension raises InvalidFileTypeError."""
        with pytest.raises(InvalidFileTypeError):
            extract_tables(non_pdf_file)

    def test_corrupted_file_raises(self, corrupted_pdf: Path) -> None:
        """Corrupted PDF file raises CorruptedPDFError."""
        with pytest.raises(CorruptedPDFError):
            extract_tables(corrupted_pdf)
