"""
Health check routes.

GET /health  — aggregated system health including ingestion subsystem,
               Qdrant config, embedding provider, and document counts.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ingestion.ingestion_health import check_ingestion_health

from backend.dependencies import (
    APP_START_TIME,
    COLLECTION_NAME,
    get_embedding_provider,
    get_qdrant_store,
)
from backend.schemas.health import ComponentHealth, HealthResponse
from backend.services import document_service

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def get_health() -> HealthResponse:
    """
    Aggregated system health check.

    Checks:
    - Ingestion pipeline modules (via check_ingestion_health)
    - Qdrant vector store connectivity
    - Embedding provider availability
    - API server liveness
    """
    components: list[ComponentHealth] = []
    overall_healthy = True

    # 1. Ingestion subsystem health (pure Python, no external calls)
    try:
        health = check_ingestion_health()
        components.append(
            ComponentHealth(
                name="ingestion_pipeline",
                status="healthy" if health.healthy else "unhealthy",
                message=health.status,
            )
        )
        if not health.healthy:
            overall_healthy = False
    except Exception as exc:
        components.append(
            ComponentHealth(
                name="ingestion_pipeline",
                status="unhealthy",
                message=str(exc),
            )
        )
        overall_healthy = False

    # 2. Qdrant store
    try:
        store = get_qdrant_store()
        # A simple non-destructive check: try to list collections
        _ = store.client.get_collections()
        components.append(
            ComponentHealth(
                name="qdrant_vector_store",
                status="healthy",
                message="Qdrant client operational",
            )
        )
    except Exception as exc:
        components.append(
            ComponentHealth(
                name="qdrant_vector_store",
                status="degraded",
                message=f"Qdrant may be unavailable: {exc}",
            )
        )
        # Qdrant being in-memory is normal; don't fail overall health

    # 3. Embedding provider
    try:
        provider = get_embedding_provider()
        vec = provider.embed("health check probe")
        status = "healthy" if isinstance(vec, list) and len(vec) > 0 else "degraded"
        components.append(
            ComponentHealth(
                name="embedding_provider",
                status=status,
                message=f"Provider functional — dim={len(vec)}",
            )
        )
    except Exception as exc:
        components.append(
            ComponentHealth(
                name="embedding_provider",
                status="unhealthy",
                message=str(exc),
            )
        )
        overall_healthy = False

    # 4. API server (always healthy if we reach here)
    components.append(
        ComponentHealth(
            name="api_server",
            status="healthy",
            message="FastAPI running",
        )
    )

    # Aggregate counts
    docs = await document_service.list_documents()
    total_docs = len(docs)

    uptime = time.time() - APP_START_TIME
    agg_status = "healthy" if overall_healthy else "degraded"

    return HealthResponse(
        status=agg_status,
        components=components,
        uptime_seconds=round(uptime, 2),
        total_documents=total_docs,
        total_vectors=0,  # Would need Qdrant query for exact count
    )
