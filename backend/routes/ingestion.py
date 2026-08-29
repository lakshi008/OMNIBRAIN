"""
Ingestion status routes.

GET /api/ingestion/{document_id}/status — Poll ingestion status for a document.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.schemas.ingestion import IngestionStatusResponse
from backend.services import ingestion_service

router = APIRouter(prefix="/api/ingestion", tags=["Ingestion"])


@router.get(
    "/{document_id}/status",
    response_model=IngestionStatusResponse,
    summary="Get ingestion status for a document",
)
async def get_ingestion_status(document_id: str) -> IngestionStatusResponse:
    """
    Return the current ingestion pipeline status for the given document.

    Fields:
    - **status**: PENDING | RUNNING (PROCESSING) | COMPLETED | FAILED
    - **current_stage**: Active pipeline stage
    - **progress**: Estimated progress 0-100
    - **completed_stages**: Stages already finished
    - **chunks**: Total chunks produced
    - **vectors**: Total vectors stored
    - **errors**: Any error messages
    """
    result = await ingestion_service.get_ingestion_status(document_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No ingestion record found for document '{document_id}'. "
                "The document may not have been ingested yet."
            ),
        )
    return result
