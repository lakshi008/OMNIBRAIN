"""
OmniBrain Ingestion Pipeline.

Provides PDF parsing, text extraction, and document metadata generation.
"""

from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
)
from ingestion.models import DocumentMetadata, PageData, ParsedDocument
from ingestion.pdf_text_extractor import extract_text, validate_pdf

__all__ = [
    # Core functions
    "extract_text",
    "validate_pdf",
    # Data models
    "ParsedDocument",
    "PageData",
    "DocumentMetadata",
    # Exceptions
    "PDFNotFoundError",
    "InvalidFileTypeError",
    "CorruptedPDFError",
]
