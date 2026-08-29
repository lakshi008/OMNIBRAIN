"""
Search / RAG routes with LLM Answer Synthesis.

POST /api/search         — Execute a RAG search query via SearchAgent & synthesize grounded answer
GET  /api/search/health  — Health check for the search & LLM subsystem
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import (
    COLLECTION_NAME,
    get_answer_synthesizer,
    get_embedding_provider,
    get_llm_provider,
    get_qdrant_store,
    get_search_agent,
)
from backend.schemas.search import SearchHealthResponse, SearchRequest, SearchResponse
from backend.services.search_service import run_search

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["Search"])


@router.post("", response_model=SearchResponse, summary="Execute a RAG search query with grounded answer synthesis")
async def search_documents(
    request: SearchRequest,
    agent=Depends(get_search_agent),
    synthesizer=Depends(get_answer_synthesizer),
) -> SearchResponse:
    """
    Search indexed documents using dense vector similarity and synthesize a grounded answer.

    Workflow:
    1. Query embedding & vector search in Qdrant via `SearchAgent`.
    2. Exact lineage citation packaging (document ID, page, chunk ID, similarity score).
    3. Grounded LLM answer synthesis using only retrieved context via `AnswerSynthesizer`.
    """
    response = await run_search(
        query=request.query,
        search_agent=agent,
        synthesizer=synthesizer,
        top_k=request.top_k,
        min_score=request.min_score,
        max_results=request.top_k,
        collection_name=request.collection_name,
    )

    return response


@router.get(
    "/health",
    response_model=SearchHealthResponse,
    summary="Health check for the search subsystem",
)
async def search_health() -> SearchHealthResponse:
    """Check whether the search and embedding subsystem is operational."""
    try:
        store = get_qdrant_store()
        provider = get_embedding_provider()
        llm = get_llm_provider()

        # Probe the embedding provider
        test_vec = provider.embed("health check")
        embedding_ok = isinstance(test_vec, list) and len(test_vec) > 0

        # Check collection existence
        collection_exists = store.collection_exists(COLLECTION_NAME)

        llm_status = "configured" if llm is not None and getattr(llm, "api_key", None) else "unconfigured_or_optional"

        return SearchHealthResponse(
            status="healthy" if embedding_ok else "degraded",
            vector_store="healthy",
            embedding_provider="healthy" if embedding_ok else "unhealthy",
            collection_exists=collection_exists,
            collection_name=COLLECTION_NAME,
            message=(
                f"Collection '{COLLECTION_NAME}' {'exists' if collection_exists else 'not yet created'}. "
                f"LLM synthesis status: {llm_status}."
            ),
        )
    except Exception as exc:
        logger.error("Search health check failed: %s", exc)
        return SearchHealthResponse(
            status="unhealthy",
            vector_store="unknown",
            embedding_provider="unknown",
            collection_exists=False,
            collection_name=COLLECTION_NAME,
            message=str(exc),
        )
