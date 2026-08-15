"""
OmniBrain Ingestion Pipeline.

Provides PDF parsing, text extraction, table extraction, image extraction,
unified document ingestion pipeline, document chunking, chunk validation,
normalization, embedding preparation, embedding generation, Qdrant vector store integration,
similarity retrieval, retrieval result processing, retrieval service layer, and document metadata generation.
"""

from ingestion.chunk_validator import normalize_chunks, validate_chunks
from ingestion.chunker import chunk_document
from ingestion.embedding_generator import EmbeddingProvider, generate_embeddings
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
)
from ingestion.models import (
    ChunkValidationResult,
    ChunkingResult,
    DocumentChunk,
    DocumentMetadata,
    EmbeddingGenerationResult,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    EmbeddingVectorRecord,
    ExtractedImage,
    ExtractedTable,
    ImageExtractionResult,
    IngestionResult,
    PageData,
    ParsedDocument,
    RetrievalServiceResult,
    TableExtractionResult,
    VectorSearchResult,
)
from ingestion.pdf_image_extractor import extract_images
from ingestion.pdf_ingestion_pipeline import ingest_pdf
from ingestion.pdf_table_extractor import extract_tables
from ingestion.pdf_text_extractor import extract_text, validate_pdf
from ingestion.qdrant_config import QdrantConfig
from ingestion.qdrant_store import QdrantVectorStore
from ingestion.retrieval import retrieve
from ingestion.retrieval_processor import (
    build_retrieval_context,
    process_retrieval_results,
)
from ingestion.retrieval_service import retrieve_context

__all__ = [
    # Core pipeline, extractors, chunker, validator, preparation, generator, store, retrieval, processor & service
    "ingest_pdf",
    "extract_text",
    "extract_tables",
    "extract_images",
    "chunk_document",
    "validate_chunks",
    "normalize_chunks",
    "prepare_for_embedding",
    "generate_embeddings",
    "EmbeddingProvider",
    "QdrantVectorStore",
    "QdrantConfig",
    "retrieve",
    "process_retrieval_results",
    "build_retrieval_context",
    "retrieve_context",
    "validate_pdf",
    # Data models
    "IngestionResult",
    "ChunkingResult",
    "ChunkValidationResult",
    "EmbeddingRecord",
    "EmbeddingPreparationResult",
    "EmbeddingVectorRecord",
    "EmbeddingGenerationResult",
    "VectorSearchResult",
    "RetrievalServiceResult",
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
