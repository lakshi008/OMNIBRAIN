"""
End-to-end ingestion service for the OmniBrain ingestion pipeline.

Orchestrates the complete document ingestion workflow from raw PDF to embedding generation:
PDF Validation -> Multi-Modal Extraction -> Document Chunking -> Normalization & Validation ->
Embedding Preparation -> Embedding Generation.

Integrates:
- Day 14: Structured error handling and pipeline status tracking.
- Day 15: Reusable IngestionConfig configuration.
- Day 16: Per-stage metrics and execution statistics.
- Day 17: Structured ingestion logging.
"""

from __future__ import annotations

import logging as _logging
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
from ingestion.ingestion_config import IngestionConfig
from ingestion.ingestion_errors import (
    IngestionChunkingError,
    IngestionEmbeddingError,
    IngestionExtractionError,
    IngestionValidationError,
)
from ingestion.ingestion_logging import IngestionLogger
from ingestion.ingestion_metrics import IngestionMetrics
from ingestion.ingestion_status import (
    IngestionStatus,
    PipelineStage,
    PipelineStatus,
)
from ingestion.models import EmbeddingGenerationResult
from ingestion.pdf_ingestion_pipeline import ingest_pdf
from ingestion.pdf_text_extractor import validate_pdf

_UNSET = object()  # sentinel for detecting when chunk_size/overlap are not passed


def run_ingestion(
    pdf_path: str | Path,
    embedding_provider: EmbeddingProvider,
    chunk_size: int | object = _UNSET,
    chunk_overlap: int | object = _UNSET,
    config: IngestionConfig | None = None,
    status_tracker: IngestionStatus | None = None,
    metrics: IngestionMetrics | None = None,
    logger: IngestionLogger | _logging.Logger | None = None,
) -> EmbeddingGenerationResult:
    """Execute the complete end-to-end document ingestion pipeline.

    Integrates Day 14 structured error handling, Day 15 configuration,
    Day 16 per-stage metrics tracking, and Day 17 structured logging.

    Configuration precedence (highest to lowest):
    1. Explicit ``chunk_size`` / ``chunk_overlap`` keyword arguments.
    2. Values from the provided ``config`` (IngestionConfig).
    3. IngestionConfig defaults (chunk_size=1000, chunk_overlap=200).

    Workflow stages:
    1. Validates the PDF file existence, format, and structure.
    2. Runs unified multi-modal extraction (text, tables, images).
    3. Chunks extracted content into structured DocumentChunk objects.
    4. Normalises chunk whitespace and validates chunk integrity.
    5. Prepares chunks for embedding while preserving citation & document lineage.
    6. Generates dense vector embeddings using the provided EmbeddingProvider.

    Args:
        pdf_path: Path to the target PDF document.
        embedding_provider: Instance implementing the EmbeddingProvider interface.
        chunk_size: Target size for text chunking in characters (> 0).
            Overrides config.chunk_size when provided.
        chunk_overlap: Overlap between consecutive text chunks (>= 0 and < chunk_size).
            Overrides config.chunk_overlap when provided.
        config: Optional IngestionConfig providing chunking and pipeline settings.
            Defaults to IngestionConfig() when not provided.
        status_tracker: Optional IngestionStatus for stage progression tracking.
        metrics: Optional IngestionMetrics for timing and counter recording.
        logger: Optional IngestionLogger (or standard logging.Logger) for structured
            event logging. When None, logging is silently skipped.

    Returns:
        EmbeddingGenerationResult with all generated vector records, dimension, and lineage.

    Raises:
        IngestionValidationError / ValueError: If chunk configuration is invalid.
        TypeError: If embedding provider is invalid.
        IngestionExtractionError / PDFNotFoundError / InvalidFileTypeError / CorruptedPDFError:
            If PDF validation or extraction fails.
        IngestionChunkingError: If chunking document content fails.
        IngestionValidationError: If chunk validation fails.
        IngestionEmbeddingError: If embedding preparation or generation fails.
    """
    # Resolve configuration: explicit args > config > defaults
    effective_config = config if config is not None else IngestionConfig()
    effective_chunk_size: int = (
        chunk_size if chunk_size is not _UNSET else effective_config.chunk_size  # type: ignore[assignment]
    )
    effective_chunk_overlap: int = (
        chunk_overlap if chunk_overlap is not _UNSET else effective_config.chunk_overlap  # type: ignore[assignment]
    )

    # Normalise logger argument into an IngestionLogger (Day 17)
    _log: IngestionLogger | None
    if logger is None:
        _log = None
    elif isinstance(logger, IngestionLogger):
        _log = logger
    else:
        # Wrap a plain logging.Logger in IngestionLogger so callers can pass either
        _log = IngestionLogger(logger=logger)

    # Status tracker (Day 14)
    tracker = status_tracker or IngestionStatus()
    if tracker.status == PipelineStatus.PENDING:
        tracker.start(stage=PipelineStage.EXTRACTION)

    # Metrics tracker (Day 16)
    m = metrics
    if m is not None:
        m.start_pipeline()

    # Resolve filename for early logging before extraction completes
    _early_filename = Path(pdf_path).name if pdf_path else ""
    if _log is not None:
        _log.log_ingestion_start(_early_filename)

    try:
        # 1. Validate chunk configuration
        if (
            not isinstance(effective_chunk_size, int)
            or isinstance(effective_chunk_size, bool)
            or effective_chunk_size <= 0
        ):
            err = IngestionValidationError(
                f"chunk_size must be a positive integer > 0, got {effective_chunk_size!r}.",
                stage="VALIDATION",
            )
            tracker.fail(err.message, original_error=err)
            if m is not None:
                m.finish_pipeline(success=False, error=err.message)
            if _log is not None:
                _log.log_ingestion_failed(_early_filename, stage="VALIDATION", error=err.message)
            raise err

        if (
            not isinstance(effective_chunk_overlap, int)
            or isinstance(effective_chunk_overlap, bool)
            or effective_chunk_overlap < 0
        ):
            err = IngestionValidationError(
                f"chunk_overlap must be a non-negative integer >= 0, got {effective_chunk_overlap!r}.",
                stage="VALIDATION",
            )
            tracker.fail(err.message, original_error=err)
            if m is not None:
                m.finish_pipeline(success=False, error=err.message)
            if _log is not None:
                _log.log_ingestion_failed(_early_filename, stage="VALIDATION", error=err.message)
            raise err

        if effective_chunk_overlap >= effective_chunk_size:
            err = IngestionValidationError(
                f"chunk_overlap ({effective_chunk_overlap}) must be strictly less than "
                f"chunk_size ({effective_chunk_size}).",
                stage="VALIDATION",
            )
            tracker.fail(err.message, original_error=err)
            if m is not None:
                m.finish_pipeline(success=False, error=err.message)
            if _log is not None:
                _log.log_ingestion_failed(_early_filename, stage="VALIDATION", error=err.message)
            raise err

        # 2. Validate embedding provider interface
        if embedding_provider is None or not (
            (hasattr(embedding_provider, "embed") and callable(getattr(embedding_provider, "embed")))
            or (hasattr(embedding_provider, "embed_batch") and callable(getattr(embedding_provider, "embed_batch")))
        ):
            err_msg = "Invalid embedding provider: must implement 'embed' or 'embed_batch' method."
            tracker.fail(err_msg)
            if m is not None:
                m.finish_pipeline(success=False, error=err_msg)
            if _log is not None:
                _log.log_ingestion_failed(_early_filename, stage="VALIDATION", error=err_msg)
            raise TypeError(err_msg)

        # ── Stage helpers ──────────────────────────────────────────────────

        def _stage_start(stage_name: str) -> None:
            if _log is not None:
                _log.log_stage_start(stage_name)

        def _stage_done(stage_name: str, **extra: object) -> None:
            if m is not None:
                sm = m.get_stage(stage_name)
                dur = sm.duration_seconds if sm else 0.0
            else:
                dur = 0.0
            if _log is not None:
                _log.log_stage_complete(
                    stage_name,
                    duration_seconds=dur,
                    document_id=tracker.document_id or "",
                    **extra,
                )

        def _stage_fail(stage_name: str, error: str) -> None:
            if m is not None:
                sm = m.get_stage(stage_name)
                dur = sm.duration_seconds if sm else 0.0
            else:
                dur = 0.0
            if _log is not None:
                _log.log_stage_failed(stage_name, duration_seconds=dur, error=error)

        # 3. EXTRACTION
        _stage_start("EXTRACTION")
        try:
            if m is not None:
                with m.track_stage("EXTRACTION"):
                    validated_path = validate_pdf(pdf_path)
                    tracker.filename = validated_path.name
                    ingestion_result = ingest_pdf(validated_path)
                    tracker.document_id = ingestion_result.metadata.document_id
                    m.document_id = ingestion_result.metadata.document_id
                    m.filename = validated_path.name
                    if _log is not None:
                        _log.document_id = ingestion_result.metadata.document_id
                        _log.filename = validated_path.name
            else:
                validated_path = validate_pdf(pdf_path)
                tracker.filename = validated_path.name
                ingestion_result = ingest_pdf(validated_path)
                tracker.document_id = ingestion_result.metadata.document_id
                if _log is not None:
                    _log.document_id = ingestion_result.metadata.document_id
                    _log.filename = validated_path.name
            _stage_done("EXTRACTION")
        except (PDFNotFoundError, InvalidFileTypeError, CorruptedPDFError) as e:
            _stage_fail("EXTRACTION", str(e))
            tracker.fail(str(e), original_error=e)
            if m is not None and m.status != "FAILED":
                m.finish_pipeline(success=False, error=str(e))
            if _log is not None:
                _log.log_ingestion_failed(_early_filename, stage="EXTRACTION", error=str(e))
            raise e
        except IngestionValidationError:
            raise
        except Exception as e:
            err = IngestionExtractionError(
                f"PDF extraction failed: {e}",
                stage="EXTRACTION",
                original_error=e,
            )
            _stage_fail("EXTRACTION", err.message)
            tracker.fail(err.message, original_error=e)
            if m is not None and m.status != "FAILED":
                m.finish_pipeline(success=False, error=err.message)
            if _log is not None:
                _log.log_ingestion_failed(_early_filename, stage="EXTRACTION", error=err.message)
            raise err from e

        # 4. CHUNKING
        tracker.advance_stage(PipelineStage.CHUNKING)
        _stage_start("CHUNKING")
        try:
            if m is not None:
                with m.track_stage("CHUNKING"):
                    chunking_result = chunk_document(
                        document=ingestion_result,
                        chunk_size=effective_chunk_size,
                        chunk_overlap=effective_chunk_overlap,
                    )
                    m.record_chunks(chunking_result)
            else:
                chunking_result = chunk_document(
                    document=ingestion_result,
                    chunk_size=effective_chunk_size,
                    chunk_overlap=effective_chunk_overlap,
                )
            _stage_done(
                "CHUNKING",
                total_chunks=chunking_result.total_chunks,
                text_chunks=chunking_result.text_chunks,
                table_chunks=chunking_result.table_chunks,
                image_chunks=chunking_result.image_chunks,
            )
        except IngestionChunkingError:
            raise
        except Exception as e:
            err = IngestionChunkingError(
                f"Document chunking failed: {e}",
                stage="CHUNKING",
                original_error=e,
            )
            _stage_fail("CHUNKING", err.message)
            tracker.fail(err.message, original_error=e)
            if m is not None and m.status != "FAILED":
                m.finish_pipeline(success=False, error=err.message)
            if _log is not None:
                _log.log_ingestion_failed(_early_filename, stage="CHUNKING", error=err.message)
            raise err from e

        # 5. NORMALIZATION
        tracker.advance_stage(PipelineStage.NORMALIZATION)
        _stage_start("NORMALIZATION")
        if m is not None:
            with m.track_stage("NORMALIZATION"):
                normalized_chunks = normalize_chunks(chunking_result.chunks)
        else:
            normalized_chunks = normalize_chunks(chunking_result.chunks)
        _stage_done("NORMALIZATION")

        # 6. VALIDATION
        tracker.advance_stage(PipelineStage.VALIDATION)
        _stage_start("VALIDATION")
        if m is not None:
            with m.track_stage("VALIDATION"):
                validation_result = validate_chunks(normalized_chunks)
        else:
            validation_result = validate_chunks(normalized_chunks)

        if not validation_result.is_valid:
            err = IngestionValidationError(
                f"Chunk validation failed with {len(validation_result.errors)} error(s): "
                f"{'; '.join(validation_result.errors)}",
                stage="VALIDATION",
            )
            _stage_fail("VALIDATION", err.message)
            tracker.fail(err.message, original_error=err)
            if m is not None:
                m.finish_pipeline(success=False, error=err.message)
            if _log is not None:
                _log.log_ingestion_failed(_early_filename, stage="VALIDATION", error=err.message)
            raise err
        _stage_done("VALIDATION")

        # 7. EMBEDDING PREPARATION
        tracker.advance_stage(PipelineStage.EMBEDDING_PREPARATION)
        _stage_start("EMBEDDING_PREPARATION")
        try:
            if m is not None:
                with m.track_stage("EMBEDDING_PREPARATION"):
                    prepared_result = prepare_for_embedding(normalized_chunks)
            else:
                prepared_result = prepare_for_embedding(normalized_chunks)
            _stage_done("EMBEDDING_PREPARATION", total_embedding_items=prepared_result.total_items)
        except IngestionEmbeddingError:
            raise
        except Exception as e:
            err = IngestionEmbeddingError(
                f"Embedding preparation failed: {e}",
                stage="EMBEDDING_PREPARATION",
                original_error=e,
            )
            _stage_fail("EMBEDDING_PREPARATION", err.message)
            tracker.fail(err.message, original_error=e)
            if m is not None and m.status != "FAILED":
                m.finish_pipeline(success=False, error=err.message)
            if _log is not None:
                _log.log_ingestion_failed(
                    _early_filename, stage="EMBEDDING_PREPARATION", error=err.message
                )
            raise err from e

        # 8. EMBEDDING GENERATION
        tracker.advance_stage(PipelineStage.EMBEDDING_GENERATION)
        _stage_start("EMBEDDING_GENERATION")
        try:
            if m is not None:
                with m.track_stage("EMBEDDING_GENERATION"):
                    generation_result = generate_embeddings(
                        items=prepared_result,
                        provider=embedding_provider,
                    )
                    m.record_embeddings(generation_result)
            else:
                generation_result = generate_embeddings(
                    items=prepared_result,
                    provider=embedding_provider,
                )
            _stage_done(
                "EMBEDDING_GENERATION",
                total_vectors=generation_result.total_items,
            )
        except IngestionEmbeddingError:
            raise
        except Exception as e:
            err = IngestionEmbeddingError(
                f"Embedding generation failed: {e}",
                stage="EMBEDDING_GENERATION",
                original_error=e,
            )
            _stage_fail("EMBEDDING_GENERATION", err.message)
            tracker.fail(err.message, original_error=e)
            if m is not None and m.status != "FAILED":
                m.finish_pipeline(success=False, error=err.message)
            if _log is not None:
                _log.log_ingestion_failed(
                    _early_filename, stage="EMBEDDING_GENERATION", error=err.message
                )
            raise err from e

        # 9. Mark pipeline complete
        tracker.complete()
        if m is not None:
            m.finish_pipeline(success=True)
        if _log is not None:
            _dur = m.total_duration_seconds if m is not None else 0.0
            _log.log_ingestion_complete(
                document_id=generation_result.document_id,
                filename=generation_result.filename,
                total_duration_seconds=_dur,
                total_chunks=m.total_chunks if m is not None else 0,
                text_chunks=m.text_chunks if m is not None else 0,
                table_chunks=m.table_chunks if m is not None else 0,
                image_chunks=m.image_chunks if m is not None else 0,
                total_embedding_items=m.total_embedding_items if m is not None else 0,
                total_vectors=m.total_vectors if m is not None else 0,
            )
            if m is not None:
                _log.log_from_metrics(m)

        return generation_result

    except Exception:
        if tracker.status != PipelineStatus.FAILED:
            tracker.status = PipelineStatus.FAILED
        if m is not None and m.status not in ("COMPLETED", "FAILED"):
            m.finish_pipeline(success=False, error="Unexpected pipeline failure.")
        raise
