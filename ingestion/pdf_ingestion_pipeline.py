"""
Unified PDF ingestion pipeline for the OmniBrain system.

Orchestrates validation, text extraction, table extraction, and image extraction
into a single structured IngestionResult.
"""

from __future__ import annotations

from pathlib import Path

from ingestion.models import IngestionResult
from ingestion.pdf_image_extractor import extract_images
from ingestion.pdf_table_extractor import extract_tables
from ingestion.pdf_text_extractor import extract_text, validate_pdf


def ingest_pdf(filepath: str | Path) -> IngestionResult:
    """Run the unified PDF ingestion pipeline on a document.

    Executes the following pipeline stages:
    1. Validation: Verifies file existence, `.pdf` extension, and document integrity.
    2. Text Extraction: Extracts page-by-page text, character counts, and page data.
    3. Table Extraction: Detects and extracts structured tables per page.
    4. Image Extraction: Detects and extracts embedded images, formats, and bytes per page.

    Args:
        filepath: Path to the target PDF file.

    Returns:
        IngestionResult containing document metadata, pages, tables, and images.

    Raises:
        PDFNotFoundError: If the file does not exist.
        InvalidFileTypeError: If the file extension is not `.pdf`.
        CorruptedPDFError: If PyMuPDF cannot open/parse the file.
    """
    # 1. Validate file (raises typed exceptions if invalid)
    path = validate_pdf(filepath)

    # 2. Text extraction
    text_result = extract_text(path)

    # 3. Table extraction
    table_result = extract_tables(path)

    # 4. Image extraction
    image_result = extract_images(path)

    # 5. Combine into unified result
    return IngestionResult(
        metadata=text_result.metadata,
        pages=text_result.pages,
        tables=table_result.tables,
        images=image_result.images,
    )
