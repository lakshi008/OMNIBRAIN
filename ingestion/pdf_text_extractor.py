"""
PDF text extractor for the OmniBrain ingestion pipeline.

Provides validation and page-by-page text extraction from PDF files,
returning structured ParsedDocument objects with full metadata.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
)
from ingestion.models import DocumentMetadata, PageData, ParsedDocument


def validate_pdf(filepath: str | Path) -> Path:
    """Validate that the given path points to a readable PDF file.

    Performs three checks in order:
    1. File exists on disk.
    2. File extension is `.pdf` (case-insensitive).
    3. File can be opened by PyMuPDF (not corrupted).

    Args:
        filepath: Path to the PDF file to validate.

    Returns:
        Resolved Path object for the validated file.

    Raises:
        PDFNotFoundError: If the file does not exist.
        InvalidFileTypeError: If the file extension is not `.pdf`.
        CorruptedPDFError: If PyMuPDF cannot open the file.
    """
    path = Path(filepath).resolve()

    # 1. Check existence
    if not path.exists():
        raise PDFNotFoundError(str(path))

    # 2. Check extension
    if path.suffix.lower() != ".pdf":
        raise InvalidFileTypeError(str(path), path.suffix)

    # 3. Try opening with PyMuPDF to verify it's a valid PDF
    try:
        doc = pymupdf.open(str(path))
        doc.close()
    except Exception as exc:
        raise CorruptedPDFError(str(path), reason=str(exc)) from exc

    return path


def extract_text(filepath: str | Path) -> ParsedDocument:
    """Extract text from every page of a PDF and return structured data.

    This function validates the PDF, opens it, iterates over each page,
    extracts the text content, and builds a ParsedDocument with full
    metadata including a unique document ID.

    Args:
        filepath: Path to the PDF file to parse.

    Returns:
        ParsedDocument containing metadata and per-page text data.

    Raises:
        PDFNotFoundError: If the file does not exist.
        InvalidFileTypeError: If the file extension is not `.pdf`.
        CorruptedPDFError: If PyMuPDF cannot open/parse the file.
    """
    # Validate first
    path = validate_pdf(filepath)

    # Open and extract
    document = pymupdf.open(str(path))

    try:
        pages: list[PageData] = []

        for page_index in range(len(document)):
            page = document[page_index]
            raw_text = page.get_text("text")
            text = raw_text.strip()

            pages.append(
                PageData(
                    page_number=page_index + 1,
                    text=text,
                    char_count=len(text),
                    has_content=len(text) > 0,
                )
            )

        total_pages = len(pages)
        pages_with = sum(1 for p in pages if p.has_content)

        metadata = DocumentMetadata(
            document_id=str(uuid.uuid4()),
            filename=path.name,
            total_pages=total_pages,
            content_type="application/pdf",
            created_at=datetime.now(timezone.utc).isoformat(),
            pages_with_content=pages_with,
            pages_without_content=total_pages - pages_with,
        )

        return ParsedDocument(metadata=metadata, pages=pages)

    finally:
        document.close()
