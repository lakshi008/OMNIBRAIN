"""
Pydantic schemas for the search / RAG API endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request body for POST /api/search."""

    query: str = Field(..., min_length=1, description="User question or search query")
    document_id: str | None = Field(None, description="Optional: restrict search to one document")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to retrieve")
    min_score: float = Field(0.0, ge=-1.0, le=1.0, description="Minimum similarity score threshold")
    collection_name: str | None = Field(None, description="Override target Qdrant collection name")


class CitationItem(BaseModel):
    """A single source citation returned from the search agent."""

    chunk_id: str
    document_id: str
    filename: str
    page: int | None = None
    content_type: str = "text"
    score: float
    citation: str = Field(..., description="Human-readable citation string, e.g. 'sample.pdf — Page 3'")
    content: str = Field(default="", description="Chunk text content excerpt")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response body for POST /api/search."""

    query: str
    answer: str = Field(
        default="",
        description="LLM-synthesised answer (empty — answer synthesis not yet integrated)",
    )
    status: str = Field(..., description="RESULTS_FOUND | NO_RESULTS | ERROR")
    total_results: int
    results: list[CitationItem] = Field(default_factory=list)
    context: str = Field(default="", description="Structured retrieval context passed to LLM")
    collection_name: str = Field(default="")
    error: str | None = None


class SearchHealthResponse(BaseModel):
    """Health status of the search subsystem."""

    status: str
    vector_store: str
    embedding_provider: str
    collection_exists: bool
    collection_name: str
    message: str = ""
