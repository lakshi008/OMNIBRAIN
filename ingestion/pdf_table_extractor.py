"""
PDF table extractor for the OmniBrain ingestion pipeline.

Provides validation and page-by-page table extraction from PDF files,
returning structured TableExtractionResult objects with full metadata.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from ingestion.models import DocumentMetadata, ExtractedTable, TableExtractionResult
from ingestion.pdf_text_extractor import validate_pdf


def extract_tables(filepath: str | Path) -> TableExtractionResult:
    """Extract tables from every page of a PDF and return structured data.

    This function validates the PDF, opens it, inspects each page for tables
    using PyMuPDF's table finder, and builds a TableExtractionResult with
    document metadata and structured table definitions.

    Args:
        filepath: Path to the PDF file to parse for tables.

    Returns:
        TableExtractionResult containing document metadata and extracted tables.

    Raises:
        PDFNotFoundError: If the file does not exist.
        InvalidFileTypeError: If the file extension is not `.pdf`.
        CorruptedPDFError: If PyMuPDF cannot open/parse the file.
    """
    path = validate_pdf(filepath)
    document = pymupdf.open(str(path))

    try:
        tables: list[ExtractedTable] = []
        pages_with_content = 0
        total_pages = len(document)

        for page_index in range(total_pages):
            page = document[page_index]
            page_number = page_index + 1

            # Check if page has text content
            page_text = page.get_text("text").strip()
            if page_text:
                pages_with_content += 1

            # Detect tables on the page
            tabs = page.find_tables()
            for table_idx, tab in enumerate(tabs):
                raw_cells = tab.extract()
                row_count = getattr(tab, "row_count", len(raw_cells))
                col_count = getattr(tab, "col_count", len(raw_cells[0]) if raw_cells else 0)

                tables.append(
                    ExtractedTable(
                        page_number=page_number,
                        table_index=table_idx,
                        rows=row_count,
                        columns=col_count,
                        cells=raw_cells,
                    )
                )

        metadata = DocumentMetadata(
            document_id=str(uuid.uuid4()),
            filename=path.name,
            total_pages=total_pages,
            content_type="application/pdf",
            created_at=datetime.now(timezone.utc).isoformat(),
            pages_with_content=pages_with_content,
            pages_without_content=total_pages - pages_with_content,
        )

        return TableExtractionResult(metadata=metadata, tables=tables)

    finally:
        document.close()
