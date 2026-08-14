"""
OmniBrain Ingestion Pipeline.

Provides PDF parsing, text extraction, table extraction, and document metadata generation.
"""

from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
)
from ingestion.models import (
    DocumentMetadata,
    ExtractedTable,
    PageData,
    ParsedDocument,
    TableExtractionResult,
)
from ingestion.pdf_table_extractor import extract_tables
from ingestion.pdf_text_extractor import extract_text, validate_pdf

__all__ = [
    # Core functions
    "extract_text",
    "extract_tables",
    "validate_pdf",
    # Data models
    "ParsedDocument",
    "PageData",
    "DocumentMetadata",
    "ExtractedTable",
    "TableExtractionResult",
    # Exceptions
    "PDFNotFoundError",
    "InvalidFileTypeError",
    "CorruptedPDFError",
]
