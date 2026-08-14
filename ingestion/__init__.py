"""
OmniBrain Ingestion Pipeline.

Provides PDF parsing, text extraction, table extraction, image extraction,
and document metadata generation.
"""

from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
)
from ingestion.models import (
    DocumentMetadata,
    ExtractedImage,
    ExtractedTable,
    ImageExtractionResult,
    PageData,
    ParsedDocument,
    TableExtractionResult,
)
from ingestion.pdf_image_extractor import extract_images
from ingestion.pdf_table_extractor import extract_tables
from ingestion.pdf_text_extractor import extract_text, validate_pdf

__all__ = [
    # Core functions
    "extract_text",
    "extract_tables",
    "extract_images",
    "validate_pdf",
    # Data models
    "ParsedDocument",
    "PageData",
    "DocumentMetadata",
    "ExtractedTable",
    "TableExtractionResult",
    "ExtractedImage",
    "ImageExtractionResult",
    # Exceptions
    "PDFNotFoundError",
    "InvalidFileTypeError",
    "CorruptedPDFError",
]
