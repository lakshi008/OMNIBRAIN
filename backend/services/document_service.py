"""
Document registry service.

Provides an in-memory, thread-safe registry mapping document_id → DocumentRecord,
with persistent file storage for uploaded PDFs under backend/uploads/.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.schemas.documents import DocumentRecord

# Upload directory (auto-created on first use)
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_registry: dict[str, DocumentRecord] = {}
_lock = asyncio.Lock()


async def register_document(
    filename: str,
    file_bytes: bytes,
) -> DocumentRecord:
    """Save the uploaded PDF to disk and register it in the in-memory registry.

    Args:
        filename: Original filename of the PDF.
        file_bytes: Raw PDF bytes.

    Returns:
        The newly created DocumentRecord.
    """
    document_id = str(uuid.uuid4())
    safe_name = f"{document_id}_{Path(filename).name}"
    file_path = UPLOAD_DIR / safe_name
    file_path.write_bytes(file_bytes)

    record = DocumentRecord(
        document_id=document_id,
        filename=filename,
        file_path=str(file_path),
        file_size_bytes=len(file_bytes),
        status="uploaded",
        upload_time=datetime.now(timezone.utc).isoformat(),
        ingestion_triggered=False,
        total_chunks=0,
        total_vectors=0,
        error=None,
    )

    async with _lock:
        _registry[document_id] = record

    return record


async def get_document(document_id: str) -> DocumentRecord | None:
    """Retrieve a DocumentRecord by its ID, or None if not found."""
    async with _lock:
        return _registry.get(document_id)


async def list_documents() -> list[DocumentRecord]:
    """Return all registered documents ordered by upload time (newest first)."""
    async with _lock:
        docs = list(_registry.values())
    docs.sort(key=lambda d: d.upload_time, reverse=True)
    return docs


async def update_document(document_id: str, **updates: Any) -> DocumentRecord | None:
    """Apply keyword updates to an existing DocumentRecord.

    Args:
        document_id: Target document.
        **updates: Fields to update (e.g. status='processing', total_chunks=42).

    Returns:
        Updated DocumentRecord, or None if not found.
    """
    async with _lock:
        record = _registry.get(document_id)
        if record is None:
            return None
        updated = record.model_copy(update=updates)
        _registry[document_id] = updated
        return updated


async def delete_document(document_id: str) -> bool:
    """Remove a document from the registry (does NOT delete the file on disk).

    Returns:
        True if the document existed and was removed, False otherwise.
    """
    async with _lock:
        if document_id in _registry:
            del _registry[document_id]
            return True
        return False
