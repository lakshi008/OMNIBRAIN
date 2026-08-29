"""
Ingestion orchestration service.

Bridges FastAPI routes with the existing ingestion domain layer.
Runs ingestion in a background thread (asyncio.to_thread) so uploads
are non-blocking, and maintains per-document IngestionStatus /
IngestionMetrics instances in memory for status polling.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ingestion.ingestion_metrics import IngestionMetrics
from ingestion.ingestion_service import run_ingestion
from ingestion.ingestion_status import IngestionStatus, PipelineStage, PipelineStatus
from ingestion.qdrant_store import QdrantVectorStore

from backend.schemas.ingestion import IngestionStatusResponse, StageMetricSchema

logger = logging.getLogger(__name__)

# In-memory per-document state: document_id → (status, metrics)
_status_registry: dict[str, IngestionStatus] = {}
_metrics_registry: dict[str, IngestionMetrics] = {}
_lock = asyncio.Lock()

# Ordered stage list for progress calculation
_STAGE_ORDER = [
    PipelineStage.EXTRACTION,
    PipelineStage.CHUNKING,
    PipelineStage.NORMALIZATION,
    PipelineStage.VALIDATION,
    PipelineStage.EMBEDDING_PREPARATION,
    PipelineStage.EMBEDDING_GENERATION,
    PipelineStage.COMPLETED,
]


def _stage_to_progress(status: IngestionStatus) -> int:
    """Estimate progress 0-100 based on completed stages."""
    if status.status == PipelineStatus.COMPLETED:
        return 100
    if status.status == PipelineStatus.FAILED:
        completed = len(status.completed_stages)
        return min(int(completed / (len(_STAGE_ORDER) - 1) * 100), 95)
    completed = len(status.completed_stages)
    total = len(_STAGE_ORDER) - 1  # exclude COMPLETED sentinel
    return min(int(completed / total * 100), 95)


async def trigger_ingestion(
    document_id: str,
    file_path: str,
    embedding_provider: Any,
    store: QdrantVectorStore,
    collection_name: str,
    vector_dimension: int,
) -> None:
    """Trigger ingestion for the given document in a background thread.

    Creates fresh IngestionStatus + IngestionMetrics instances, stores them
    in the registry, then runs the pipeline asynchronously via
    asyncio.to_thread so the API response is not blocked.

    After successful ingestion the embedding vectors are upserted into Qdrant.

    Args:
        document_id: Unique document identifier.
        file_path: Absolute path to the stored PDF.
        embedding_provider: EmbeddingProvider implementation.
        store: QdrantVectorStore instance.
        collection_name: Target Qdrant collection name.
        vector_dimension: Dimension expected by the collection.
    """
    tracker = IngestionStatus()
    metrics = IngestionMetrics(document_id=document_id, filename=Path(file_path).name)

    async with _lock:
        _status_registry[document_id] = tracker
        _metrics_registry[document_id] = metrics

    asyncio.create_task(
        _run_ingestion_task(
            document_id=document_id,
            file_path=file_path,
            embedding_provider=embedding_provider,
            store=store,
            collection_name=collection_name,
            vector_dimension=vector_dimension,
            tracker=tracker,
            metrics=metrics,
        )
    )


async def _run_ingestion_task(
    document_id: str,
    file_path: str,
    embedding_provider: Any,
    store: QdrantVectorStore,
    collection_name: str,
    vector_dimension: int,
    tracker: IngestionStatus,
    metrics: IngestionMetrics,
) -> None:
    """Background task: run ingestion pipeline then upsert to Qdrant."""
    logger.info("Ingestion started for document_id=%s file=%s", document_id, file_path)
    try:
        # Run blocking pipeline in thread pool
        result = await asyncio.to_thread(
            run_ingestion,
            pdf_path=file_path,
            embedding_provider=embedding_provider,
            status_tracker=tracker,
            metrics=metrics,
        )

        # Upsert vectors to Qdrant
        if result and result.items:
            try:
                if not store.collection_exists(collection_name):
                    store.create_collection(collection_name, vector_dimension)
                await asyncio.to_thread(
                    store.upsert_embeddings,
                    collection_name=collection_name,
                    result=result,
                )
                logger.info(
                    "Upserted %d vectors for document_id=%s",
                    result.total_items,
                    document_id,
                )
            except Exception as qdrant_err:
                logger.warning(
                    "Qdrant upsert failed for document_id=%s: %s", document_id, qdrant_err
                )

        logger.info("Ingestion completed for document_id=%s", document_id)

    except Exception as exc:
        logger.error("Ingestion failed for document_id=%s: %s", document_id, exc)
        if tracker.status != PipelineStatus.FAILED:
            tracker.fail(str(exc), original_error=exc)


async def get_ingestion_status(document_id: str) -> IngestionStatusResponse | None:
    """Return current IngestionStatusResponse for a document, or None if not found."""
    async with _lock:
        tracker = _status_registry.get(document_id)
        metrics = _metrics_registry.get(document_id)

    if tracker is None:
        return None

    # Build stage metric list from metrics if available
    stage_metrics: list[StageMetricSchema] = []
    if metrics is not None:
        for sm in metrics.stage_metrics:
            stage_metrics.append(
                StageMetricSchema(
                    stage=sm.stage,
                    duration_seconds=sm.duration_seconds,
                )
            )

    errors: list[str] = []
    if tracker.error:
        errors.append(tracker.error)

    return IngestionStatusResponse(
        document_id=tracker.document_id or document_id,
        filename=tracker.filename,
        status=tracker.status.value,
        current_stage=tracker.current_stage.value,
        progress=_stage_to_progress(tracker),
        completed_stages=[s.value for s in tracker.completed_stages],
        chunks=metrics.total_chunks if metrics else 0,
        text_chunks=metrics.text_chunks if metrics else 0,
        table_chunks=metrics.table_chunks if metrics else 0,
        image_chunks=metrics.image_chunks if metrics else 0,
        vectors=metrics.total_vectors if metrics else 0,
        duration_seconds=metrics.total_duration_seconds if metrics else 0.0,
        stage_metrics=stage_metrics,
        errors=errors,
    )


async def list_ingestion_statuses() -> dict[str, IngestionStatusResponse]:
    """Return all known ingestion statuses keyed by document_id."""
    async with _lock:
        doc_ids = list(_status_registry.keys())

    result = {}
    for doc_id in doc_ids:
        status = await get_ingestion_status(doc_id)
        if status:
            result[doc_id] = status
    return result
