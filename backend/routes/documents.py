"""
Document management routes.

POST   /api/documents/upload           — Upload a PDF and trigger ingestion
GET    /api/documents                  — List all documents
GET    /api/documents/{document_id}    — Get document details + ingestion status
DELETE /api/documents/{document_id}    — Remove document from registry
POST   /api/documents/{document_id}/ingest  — Re-trigger ingestion
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.dependencies import (
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    get_embedding_provider,
    get_qdrant_store,
)
from backend.schemas.documents import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentRecord,
    DocumentUploadResponse,
)
from backend.services import document_service, ingestion_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["Documents"])

MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB


def _validate_pdf_upload(file: UploadFile) -> None:
    """Validate content type and extension of an uploaded file."""
    allowed_types = {"application/pdf", "application/octet-stream"}
    content_type = (file.content_type or "").lower()
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Only PDF files are accepted. Got filename: '{filename}'",
        )
    if content_type and content_type not in allowed_types:
        # Some browsers send 'application/pdf', accept both
        if "pdf" not in content_type:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid content type '{content_type}'. Expected application/pdf.",
            )


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF document and trigger ingestion",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload"),
) -> DocumentUploadResponse:
    """
    Accept a PDF file, store it, and trigger the ingestion pipeline asynchronously.

    - Validates file type and size.
    - Saves the file to `backend/uploads/`.
    - Registers the document in the in-memory registry.
    - Triggers the existing ingestion pipeline in a background task.
    - Returns document_id immediately; poll `/api/ingestion/{document_id}/status` for progress.
    """
    _validate_pdf_upload(file)

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
        )

    filename = file.filename or "upload.pdf"

    # Register document and save file
    record = await document_service.register_document(
        filename=filename,
        file_bytes=file_bytes,
    )

    # Update status to 'processing' and mark ingestion triggered
    await document_service.update_document(
        record.document_id,
        status="processing",
        ingestion_triggered=True,
    )

    # Trigger async ingestion
    try:
        await ingestion_service.trigger_ingestion(
            document_id=record.document_id,
            file_path=record.file_path,
            embedding_provider=get_embedding_provider(),
            store=get_qdrant_store(),
            collection_name=COLLECTION_NAME,
            vector_dimension=EMBEDDING_DIMENSION,
        )
    except Exception as exc:
        logger.error("Failed to trigger ingestion for %s: %s", record.document_id, exc)
        await document_service.update_document(
            record.document_id,
            status="error",
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document saved but ingestion failed to start: {exc}",
        )

    return DocumentUploadResponse(
        document_id=record.document_id,
        filename=filename,
        status="processing",
        message="Document uploaded and ingestion started. Poll /api/ingestion/{document_id}/status for progress.",
        file_size_bytes=len(file_bytes),
    )


@router.get("", response_model=DocumentListResponse, summary="List all documents")
async def list_documents() -> DocumentListResponse:
    """Return all registered documents with their current status."""
    docs = await document_service.list_documents()

    # Enrich status from ingestion tracker
    enriched: list[DocumentRecord] = []
    for doc in docs:
        ing_status = await ingestion_service.get_ingestion_status(doc.document_id)
        if ing_status:
            status_str = ing_status.status.lower()
            if status_str == "completed":
                status_str = "completed"
            elif status_str == "failed":
                status_str = "error"
            elif status_str == "running":
                status_str = "processing"
            updated = await document_service.update_document(
                doc.document_id,
                status=status_str,
                total_chunks=ing_status.chunks,
                total_vectors=ing_status.vectors,
            )
            enriched.append(updated or doc)
        else:
            enriched.append(doc)

    return DocumentListResponse(documents=enriched, total=len(enriched))


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details and ingestion status",
)
async def get_document(document_id: str) -> DocumentDetailResponse:
    """Return full document details including current ingestion status."""
    record = await document_service.get_document(document_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    ing_status = await ingestion_service.get_ingestion_status(document_id)
    ing_dict = ing_status.model_dump() if ing_status else None

    return DocumentDetailResponse(
        document_id=record.document_id,
        filename=record.filename,
        file_size_bytes=record.file_size_bytes,
        status=record.status,
        upload_time=record.upload_time,
        ingestion_triggered=record.ingestion_triggered,
        total_chunks=record.total_chunks,
        total_vectors=record.total_vectors,
        error=record.error,
        ingestion_status=ing_dict,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Remove a document from the registry",
)
async def delete_document(document_id: str) -> None:
    """Remove a document from the in-memory registry (does not delete vectors from Qdrant)."""
    deleted = await document_service.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )


@router.post(
    "/{document_id}/ingest",
    response_model=DocumentUploadResponse,
    summary="Re-trigger ingestion for an existing document",
)
async def reingest_document(document_id: str) -> DocumentUploadResponse:
    """Re-trigger the ingestion pipeline for a previously uploaded document."""
    record = await document_service.get_document(document_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    await document_service.update_document(
        document_id,
        status="processing",
        ingestion_triggered=True,
        error=None,
    )

    await ingestion_service.trigger_ingestion(
        document_id=document_id,
        file_path=record.file_path,
        embedding_provider=get_embedding_provider(),
        store=get_qdrant_store(),
        collection_name=COLLECTION_NAME,
        vector_dimension=EMBEDDING_DIMENSION,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=record.filename,
        status="processing",
        message="Re-ingestion triggered. Poll /api/ingestion/{document_id}/status for progress.",
        file_size_bytes=record.file_size_bytes,
    )
