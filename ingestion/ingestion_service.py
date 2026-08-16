"""
End-to-end ingestion service for the OmniBrain ingestion pipeline.

Orchestrates the entire document ingestion workflow from raw PDF to embedding generation:
PDF Validation -> Multi-Modal Extraction -> Document Chunking -> Normalization & Validation ->
Embedding Preparation -> Embedding Generation.
"""

from __future__ import annotations

from pathlib import Path

from ingestion.chunk_validator import normalize_chunks, validate_chunks
from ingestion.chunker import chunk_document
from ingestion.embedding_generator import EmbeddingProvider, generate_embeddings
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.models import EmbeddingGenerationResult
from ingestion.pdf_ingestion_pipeline import ingest_pdf
from ingestion.pdf_text_extractor import validate_pdf


def run_ingestion(
    pdf_path: str | Path,
    embedding_provider: EmbeddingProvider,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> EmbeddingGenerationResult:
    """Execute the complete end-to-end document ingestion pipeline.

    Workflow stages:
    1. Validates the PDF file existence, format, and structure.
    2. Runs unified multi-modal extraction (text, tables, images).
    3. Chunks extracted content into structured DocumentChunk objects.
    4. Normalizes chunk whitespace and validates chunk integrity.
    5. Prepares chunks for embedding while preserving citation & document lineage.
    6. Generates dense vector embeddings using the provided EmbeddingProvider.

    Args:
        pdf_path: Path to the target PDF document.
        embedding_provider: Instance implementing the EmbeddingProvider interface.
        chunk_size: Target size for text chunking in characters (> 0).
        chunk_overlap: Overlap between consecutive text chunks (>= 0 and < chunk_size).

    Returns:
        EmbeddingGenerationResult containing all generated vector records, dimension, and lineage.

    Raises:
        PDFNotFoundError: If the PDF file does not exist.
        InvalidFileTypeError: If the file is not a PDF.
        CorruptedPDFError: If the PDF cannot be opened or parsed.
        TypeError: If embedding_provider is invalid or parameters are wrong types.
        ValueError: If chunk_size or chunk_overlap are invalid, or if chunk validation fails.
    """
    # 1. Validate chunk configuration
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer > 0, got {chunk_size!r}.")

    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool) or chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be a non-negative integer >= 0, got {chunk_overlap!r}.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})."
        )

    # 2. Validate embedding provider interface
    if embedding_provider is None or not (
        (hasattr(embedding_provider, "embed") and callable(getattr(embedding_provider, "embed")))
        or (hasattr(embedding_provider, "embed_batch") and callable(getattr(embedding_provider, "embed_batch")))
    ):
        raise TypeError(
            "Invalid embedding provider: must implement 'embed' or 'embed_batch' method."
        )

    # 3. Validate PDF file (raises PDFNotFoundError, InvalidFileTypeError, CorruptedPDFError)
    validated_path = validate_pdf(pdf_path)

    # 4. Run unified multi-modal extraction pipeline
    ingestion_result = ingest_pdf(validated_path)

    # 5. Chunk document content
    chunking_result = chunk_document(
        document=ingestion_result,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 6. Normalize chunks
    normalized_chunks = normalize_chunks(chunking_result.chunks)

    # 7. Validate normalized chunks
    validation_result = validate_chunks(normalized_chunks)
    if not validation_result.is_valid:
        raise ValueError(
            f"Chunk validation failed with {len(validation_result.errors)} error(s): "
            f"{'; '.join(validation_result.errors)}"
        )

    # 8. Prepare validated chunks for embedding (preserves document lineage & citations)
    prepared_result = prepare_for_embedding(normalized_chunks)

    # 9. Generate dense embeddings via provider
    generation_result = generate_embeddings(
        items=prepared_result,
        provider=embedding_provider,
    )

    return generation_result
