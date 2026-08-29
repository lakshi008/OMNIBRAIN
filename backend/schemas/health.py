"""
Pydantic schemas for health check API endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ComponentHealth(BaseModel):
    """Health status of a single system component."""

    name: str
    status: str = Field(..., description="healthy | degraded | unhealthy | unknown")
    message: str = ""


class HealthResponse(BaseModel):
    """Aggregated system health response for GET /health."""

    status: str = Field(..., description="healthy | degraded | unhealthy")
    version: str = "1.0.0"
    components: list[ComponentHealth] = Field(default_factory=list)
    uptime_seconds: float = 0.0
    total_documents: int = 0
    total_vectors: int = 0
