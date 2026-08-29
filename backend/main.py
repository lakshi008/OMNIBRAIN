"""
OMNIBRAIN FastAPI application entry point.

Exposes the complete API surface for document management, ingestion,
and RAG search built on top of the existing ingestion/agents domain layer.

Endpoints:
  GET  /                               → redirect to /docs
  GET  /health                         → system health check
  POST /api/documents/upload           → upload PDF + trigger ingestion
  GET  /api/documents                  → list all documents
  GET  /api/documents/{id}             → document details + ingestion status
  POST /api/documents/{id}/ingest      → re-trigger ingestion
  DELETE /api/documents/{id}           → remove document
  GET  /api/ingestion/{id}/status      → ingestion pipeline status
  POST /api/search                     → RAG search query
  GET  /api/search/health              → search subsystem health

Swagger UI:  http://127.0.0.1:8000/docs
ReDoc:       http://127.0.0.1:8000/redoc
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# ── Load .env FIRST — before any backend.* imports read os.getenv() ───────────
# dependencies.py captures env vars at module-import time, so dotenv must be
# loaded here, before the 'from backend.routes import ...' line below.
try:
    from dotenv import load_dotenv
    _dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(_dotenv_path, override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from backend.routes import documents, health, ingestion, search

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown lifecycle."""
    logger.info("OMNIBRAIN backend starting up...")

    # ── LLM API key startup validation ───────────────────────────────────────
    # Report key presence/absence without ever logging the actual value.
    _llm_provider_name = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    if _llm_provider_name in ("groq", "default"):
        _groq_key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
        if _groq_key:
            logger.info("LLM startup: GROQ_API_KEY is configured (length=%d).", len(_groq_key))
        else:
            logger.warning(
                "LLM startup: GROQ_API_KEY is missing. "
                "Set GROQ_API_KEY in your .env file or shell environment. "
                "Groq synthesis will fail with HTTP 401 until this is set."
            )
    elif _llm_provider_name in ("openai", "openai_compatible"):
        _openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if _openai_key:
            logger.info("LLM startup: OPENAI_API_KEY is configured (length=%d).", len(_openai_key))
        else:
            logger.warning(
                "LLM startup: OPENAI_API_KEY is missing. "
                "Set OPENAI_API_KEY in your .env file or shell environment."
            )

    # Eagerly initialise singletons so the first request isn't slow
    from backend.dependencies import get_embedding_provider, get_qdrant_store, get_search_agent
    try:
        store = get_qdrant_store()
        provider = get_embedding_provider()
        agent = get_search_agent()
        logger.info("Singletons initialised successfully.")
    except Exception as exc:
        logger.warning("Singleton initialisation warning: %s", exc)

    yield

    logger.info("OMNIBRAIN backend shutting down.")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="OMNIBRAIN API",
    description=(
        "Enterprise-grade multi-modal RAG system API.\n\n"
        "Upload PDF documents, track ingestion progress, and query indexed content "
        "using dense vector similarity search backed by Qdrant.\n\n"
        "**Note**: The search agent returns citations and retrieval context. "
        "LLM answer synthesis will be integrated in a future phase."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────────────────────────

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_ORIGIN,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a structured JSON error for any unhandled exception."""
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
            "path": str(request.url.path),
        },
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(ingestion.router)
app.include_router(search.router)


# ── Root redirect ─────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to Swagger UI."""
    return RedirectResponse(url="/docs")
