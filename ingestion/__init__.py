"""
OmniBrain Ingestion Pipeline.

Provides PDF parsing, text extraction, table extraction, image extraction,
unified document ingestion pipeline, document chunking, and document metadata generation.
"""

from ingestion.chunker import chunk_document
from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
)
from ingestion.models import (
    ChunkingResult,
    DocumentChunk,
    DocumentMetadata,
    ExtractedImage,
    ExtractedTable,
    ImageExtractionResult,
    IngestionResult,
    PageData,
    ParsedDocument,
    TableExtractionResult,
)
from ingestion.pdf_image_extractor import extract_images
from ingestion.pdf_ingestion_pipeline import ingest_pdf
from ingestion.pdf_table_extractor import extract_tables
from ingestion.pdf_text_extractor import extract_text, validate_pdf

__all__ = [
    # Core pipeline, extractors & chunker
    "ingest_pdf",
    "extract_text",
    "extract_tables",
    "extract_images",
    "chunk_document",
    "validate_pdf",
    # Data models
    "IngestionResult",
    "ChunkingResult",
    "DocumentChunk",
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
