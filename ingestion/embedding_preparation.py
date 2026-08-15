"""
Embedding preparation module for the OmniBrain RAG pipeline.

Converts validated DocumentChunk objects into structured, embedding-ready records
with clear separation between embeddable text content and citation/lineage metadata.
"""

from __future__ import annotations

from typing import Any

from ingestion.chunk_validator import validate_chunks
from ingestion.models import (
    ChunkingResult,
    DocumentChunk,
    EmbeddingPreparationResult,
    EmbeddingRecord,
)


def prepare_for_embedding(
    chunks: list[DocumentChunk] | ChunkingResult,
) -> EmbeddingPreparationResult:
    """Transform validated DocumentChunk objects into embedding-ready records.

    Performs the following steps:
    1. Extracts and inspects chunk collections.
    2. Validates chunk integrity using `validate_chunks()`.
    3. Rejects invalid chunk batches with clear ValueError messages.
    4. Deterministically orders records by `chunk_index`.
    5. Preserves document lineage and citation metadata (document_id, filename, page_number).
    6. Separates embeddable content from metadata payload without calling external embedding APIs.

    Args:
        chunks: List of DocumentChunk instances or a ChunkingResult object.

    Returns:
        EmbeddingPreparationResult containing sorted EmbeddingRecord objects and counters.

    Raises:
        TypeError: If input is neither a list nor ChunkingResult.
        ValueError: If any chunk fails validation (missing fields, duplicate IDs, etc.).
    """
    if isinstance(chunks, ChunkingResult):
        doc_id = chunks.document_id
        filename = chunks.filename
        chunk_list = chunks.chunks
    elif isinstance(chunks, list):
        chunk_list = chunks
        doc_id = chunks[0].document_id if chunks and hasattr(chunks[0], "document_id") else ""
        filename = chunks[0].filename if chunks and hasattr(chunks[0], "filename") else ""
    else:
        raise TypeError(
            f"Expected list[DocumentChunk] or ChunkingResult, got {type(chunks).__name__}"
        )

    # Handle empty input safely
    if not chunk_list:
        return EmbeddingPreparationResult(
            document_id=doc_id,
            filename=filename,
            items=[],
            is_ready=True,
        )

    # Validate chunks
    validation = validate_chunks(chunk_list)
    if not validation.is_valid:
        raise ValueError(
            f"Chunk validation failed with {len(validation.errors)} error(s): {'; '.join(validation.errors)}"
        )

    # Deterministic sorting by chunk_index, then chunk_id
    sorted_chunks = sorted(chunk_list, key=lambda c: (c.chunk_index, c.chunk_id))

    # Update doc_id / filename from first validated chunk if needed
    if not doc_id and sorted_chunks:
        doc_id = sorted_chunks[0].document_id
    if not filename and sorted_chunks:
        filename = sorted_chunks[0].filename

    items: list[EmbeddingRecord] = []
    for chunk in sorted_chunks:
        # Build metadata payload preserving all citation & lineage info
        metadata_payload = dict(chunk.metadata)
        metadata_payload["chunk_id"] = chunk.chunk_id
        metadata_payload["document_id"] = chunk.document_id
        metadata_payload["filename"] = chunk.filename
        metadata_payload["chunk_index"] = chunk.chunk_index
        metadata_payload["page_number"] = chunk.page_number
        metadata_payload["content_type"] = chunk.content_type

        items.append(
            EmbeddingRecord(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                content=chunk.content,
                content_type=chunk.content_type,
                metadata=metadata_payload,
            )
        )

    return EmbeddingPreparationResult(
        document_id=doc_id,
        filename=filename,
        items=items,
        is_ready=True,
    )
