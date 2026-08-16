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
from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
)
from ingestion.ingestion_errors import (
    IngestionChunkingError,
    IngestionEmbeddingError,
    IngestionExtractionError,
    IngestionPipelineError,
    IngestionValidationError,
)
from ingestion.ingestion_status import (
    IngestionStatus,
    PipelineStage,
    PipelineStatus,
)
from ingestion.models import EmbeddingGenerationResult
from ingestion.pdf_ingestion_pipeline import ingest_pdf
from ingestion.pdf_text_extractor import validate_pdf


def run_ingestion(
    pdf_path: str | Path,
    embedding_provider: EmbeddingProvider,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    status_tracker: IngestionStatus | None = None,
) -> EmbeddingGenerationResult:
    """Execute the complete end-to-end document ingestion pipeline with stage tracking.

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
        status_tracker: Optional IngestionStatus instance to track progression.

    Returns:
        EmbeddingGenerationResult containing all generated vector records, dimension, and lineage.

    Raises:
        IngestionValidationError / ValueError: If chunk configuration is invalid.
        IngestionPipelineError / TypeError: If embedding provider is invalid.
        IngestionExtractionError / PDFNotFoundError / InvalidFileTypeError / CorruptedPDFError:
            If PDF validation or extraction fails.
        IngestionChunkingError: If chunking document content fails.
        IngestionValidationError: If chunk validation fails.
        IngestionEmbeddingError: If embedding preparation or generation fails.
    """
    tracker = status_tracker or IngestionStatus()
    if tracker.status == PipelineStatus.PENDING:
        tracker.start(stage=PipelineStage.EXTRACTION)

    try:
        # 1. Validate chunk configuration
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            err = IngestionValidationError(
                f"chunk_size must be a positive integer > 0, got {chunk_size!r}.",
                stage="VALIDATION",
            )
            tracker.fail(err.message, original_error=err)
            raise err

        if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool) or chunk_overlap < 0:
            err = IngestionValidationError(
                f"chunk_overlap must be a non-negative integer >= 0, got {chunk_overlap!r}.",
                stage="VALIDATION",
            )
            tracker.fail(err.message, original_error=err)
            raise err

        if chunk_overlap >= chunk_size:
            err = IngestionValidationError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size}).",
                stage="VALIDATION",
            )
            tracker.fail(err.message, original_error=err)
            raise err

        # 2. Validate embedding provider interface
        if embedding_provider is None or not (
            (hasattr(embedding_provider, "embed") and callable(getattr(embedding_provider, "embed")))
            or (hasattr(embedding_provider, "embed_batch") and callable(getattr(embedding_provider, "embed_batch")))
        ):
            err_msg = "Invalid embedding provider: must implement 'embed' or 'embed_batch' method."
            tracker.fail(err_msg)
            raise TypeError(err_msg)

        # 3. Validate & Extract PDF
        try:
            validated_path = validate_pdf(pdf_path)
            tracker.filename = validated_path.name
            ingestion_result = ingest_pdf(validated_path)
            tracker.document_id = ingestion_result.metadata.document_id
        except (PDFNotFoundError, InvalidFileTypeError, CorruptedPDFError) as e:
            tracker.fail(str(e), original_error=e)
            raise e
        except Exception as e:
            err = IngestionExtractionError(
                f"PDF extraction failed: {e}",
                stage="EXTRACTION",
                original_error=e,
            )
            tracker.fail(err.message, original_error=e)
            raise err from e

        # 4. Chunk document content
        tracker.advance_stage(PipelineStage.CHUNKING)
        try:
            chunking_result = chunk_document(
                document=ingestion_result,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as e:
            err = IngestionChunkingError(
                f"Document chunking failed: {e}",
                stage="CHUNKING",
                original_error=e,
            )
            tracker.fail(err.message, original_error=e)
            raise err from e

        # 5. Normalization & Validation
        tracker.advance_stage(PipelineStage.NORMALIZATION)
        normalized_chunks = normalize_chunks(chunking_result.chunks)

        tracker.advance_stage(PipelineStage.VALIDATION)
        validation_result = validate_chunks(normalized_chunks)
        if not validation_result.is_valid:
            err = IngestionValidationError(
                f"Chunk validation failed with {len(validation_result.errors)} error(s): "
                f"{'; '.join(validation_result.errors)}",
                stage="VALIDATION",
            )
            tracker.fail(err.message, original_error=err)
            raise err

        # 6. Prepare validated chunks for embedding
        tracker.advance_stage(PipelineStage.EMBEDDING_PREPARATION)
        try:
            prepared_result = prepare_for_embedding(normalized_chunks)
        except Exception as e:
            err = IngestionEmbeddingError(
                f"Embedding preparation failed: {e}",
                stage="EMBEDDING_PREPARATION",
                original_error=e,
            )
            tracker.fail(err.message, original_error=e)
            raise err from e

        # 7. Generate dense embeddings via provider
        tracker.advance_stage(PipelineStage.EMBEDDING_GENERATION)
        try:
            generation_result = generate_embeddings(
                items=prepared_result,
                provider=embedding_provider,
            )
        except Exception as e:
            err = IngestionEmbeddingError(
                f"Embedding generation failed: {e}",
                stage="EMBEDDING_GENERATION",
                original_error=e,
            )
            tracker.fail(err.message, original_error=e)
            raise err from e

        # 8. Mark pipeline complete
        tracker.complete()
        return generation_result

    except Exception:
        if tracker.status != PipelineStatus.FAILED:
            tracker.status = PipelineStatus.FAILED
        raise
