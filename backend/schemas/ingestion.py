"""
Pydantic schemas for ingestion status API endpoints.

Maps the existing IngestionStatus / IngestionMetrics domain objects
into clean JSON-serialisable response models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StageMetricSchema(BaseModel):
    """Per-stage timing metrics."""

    stage: str
    duration_seconds: float | None = None
    started_at: str | None = None
    finished_at: str | None = None


class IngestionStatusResponse(BaseModel):
    """
    Structured ingestion status response.

    Example::

        {
            "document_id": "...",
            "filename": "sample.pdf",
            "status": "PROCESSING",
            "current_stage": "EMBEDDING_GENERATION",
            "progress": 80,
            "completed_stages": ["EXTRACTION", "CHUNKING", ...],
            "chunks": 120,
            "vectors": 120,
            "errors": []
        }
    """

    document_id: str | None = Field(None)
    filename: str | None = Field(None)
    status: str = Field(..., description="PENDING | RUNNING | COMPLETED | FAILED")
    current_stage: str = Field(..., description="Active or last pipeline stage")
    progress: int = Field(0, ge=0, le=100, description="Estimated progress 0-100")
    completed_stages: list[str] = Field(default_factory=list)
    chunks: int = Field(0, description="Total chunks produced")
    text_chunks: int = Field(0)
    table_chunks: int = Field(0)
    image_chunks: int = Field(0)
    vectors: int = Field(0, description="Total vectors stored")
    duration_seconds: float = Field(0.0, description="Pipeline wall-clock duration")
    stage_metrics: list[StageMetricSchema] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
