"""
Pydantic schemas for document-related API endpoints.

Covers upload responses, document records, and document list responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response returned after a successful PDF upload."""

    document_id: str = Field(..., description="Unique document identifier (UUID4)")
    filename: str = Field(..., description="Original filename of the uploaded PDF")
    status: str = Field(..., description="Current document status (e.g. 'uploaded')")
    message: str = Field(..., description="Human-readable status message")
    file_size_bytes: int = Field(..., description="Size of the uploaded file in bytes")


class DocumentRecord(BaseModel):
    """Represents a document registered in the backend registry."""

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Absolute path to stored PDF file")
    file_size_bytes: int = Field(..., description="File size in bytes")
    status: str = Field(default="uploaded", description="Current processing status")
    upload_time: str = Field(..., description="ISO 8601 upload timestamp")
    ingestion_triggered: bool = Field(default=False)
    total_chunks: int = Field(default=0)
    total_vectors: int = Field(default=0)
    error: str | None = Field(default=None)


class DocumentListResponse(BaseModel):
    """Response for listing all documents."""

    documents: list[DocumentRecord] = Field(default_factory=list)
    total: int = Field(..., description="Total number of documents")


class DocumentDetailResponse(BaseModel):
    """Detailed response for a single document including ingestion status."""

    document_id: str
    filename: str
    file_size_bytes: int
    status: str
    upload_time: str
    ingestion_triggered: bool
    total_chunks: int
    total_vectors: int
    error: str | None = None
    ingestion_status: dict[str, Any] | None = None
