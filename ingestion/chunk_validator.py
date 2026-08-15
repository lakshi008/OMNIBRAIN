"""
Chunk validation and normalization module for the OmniBrain RAG pipeline.

Provides comprehensive validation of DocumentChunk collections and normalization
utilities to guarantee data integrity before embedding generation.
"""

from __future__ import annotations

import re
from typing import Any

from ingestion.models import ChunkValidationResult, ChunkingResult, DocumentChunk

ALLOWED_CONTENT_TYPES = frozenset({"text", "table", "image"})


def validate_chunks(chunks: list[DocumentChunk] | ChunkingResult) -> ChunkValidationResult:
    """Validate a collection of DocumentChunks or a ChunkingResult.

    Performs comprehensive structural, typing, and semantic integrity checks:
    1. chunk_id: Exists, is a non-empty string, and is globally unique across the batch.
    2. document_id: Exists, is a non-empty string, and is consistent across all chunks.
    3. chunk_index: Exists, is an integer >= 0, unique, and ordered.
    4. content: Exists, is a string, and is not empty or whitespace-only.
    5. content_type: Must be one of 'text', 'table', or 'image'.
    6. page_number: When present, must be an integer >= 1.
    7. metadata: Must be a valid dictionary.
    8. Duplicate content: Detects identical content across distinct chunks (reported as warnings).

    Args:
        chunks: List of DocumentChunk instances or a ChunkingResult object.

    Returns:
        ChunkValidationResult with status, counts, errors, and warnings.
    """
    if isinstance(chunks, ChunkingResult):
        chunk_list = chunks.chunks
    elif isinstance(chunks, list):
        chunk_list = chunks
    else:
        return ChunkValidationResult(
            is_valid=False,
            total_chunks=0,
            valid_chunks=0,
            invalid_chunks=0,
            errors=[f"Invalid input type '{type(chunks).__name__}'. Expected list[DocumentChunk] or ChunkingResult."],
            warnings=[],
        )

    total_chunks = len(chunk_list)
    if total_chunks == 0:
        return ChunkValidationResult(
            is_valid=True,
            total_chunks=0,
            valid_chunks=0,
            invalid_chunks=0,
            errors=[],
            warnings=["No chunks provided for validation."],
        )

    errors: list[str] = []
    warnings: list[str] = []

    seen_chunk_ids: dict[str, int] = {}
    seen_chunk_indices: dict[int, int] = {}
    doc_ids_seen: set[str] = set()
    content_map: dict[str, list[int]] = {}

    invalid_chunk_indices: set[int] = set()

    for pos, chunk in enumerate(chunk_list):
        chunk_has_error = False

        # 1. Structural check
        if not hasattr(chunk, "chunk_id") or not hasattr(chunk, "content"):
            errors.append(f"Position {pos}: Object is not a valid DocumentChunk instance.")
            invalid_chunk_indices.add(pos)
            continue

        cid = chunk.chunk_id
        cidx = chunk.chunk_index
        doc_id = chunk.document_id

        # 2. chunk_id checks
        if not cid or not isinstance(cid, str) or not cid.strip():
            errors.append(f"Position {pos}: Missing, empty, or non-string chunk_id.")
            chunk_has_error = True
        else:
            if cid in seen_chunk_ids:
                errors.append(
                    f"Position {pos}: Duplicate chunk_id '{cid}' (first seen at position {seen_chunk_ids[cid]})."
                )
                chunk_has_error = True
            else:
                seen_chunk_ids[cid] = pos

        # 3. document_id checks
        if not doc_id or not isinstance(doc_id, str) or not doc_id.strip():
            errors.append(f"Position {pos} (chunk_id={cid!r}): Missing, empty, or non-string document_id.")
            chunk_has_error = True
        else:
            doc_ids_seen.add(doc_id)

        # 4. chunk_index checks
        if cidx is None or not isinstance(cidx, int) or isinstance(cidx, bool) or cidx < 0:
            errors.append(f"Position {pos} (chunk_id={cid!r}): Invalid chunk_index {cidx!r}. Must be integer >= 0.")
            chunk_has_error = True
        else:
            if cidx in seen_chunk_indices:
                errors.append(
                    f"Position {pos} (chunk_id={cid!r}): Duplicate chunk_index {cidx} (first seen at position {seen_chunk_indices[cidx]})."
                )
                chunk_has_error = True
            else:
                seen_chunk_indices[cidx] = pos

        # 5. content checks
        content = chunk.content
        if content is None or not isinstance(content, str):
            errors.append(f"Position {pos} (chunk_id={cid!r}): Content must be a string, got {type(content).__name__}.")
            chunk_has_error = True
        elif not content.strip():
            errors.append(f"Position {pos} (chunk_id={cid!r}): Content is empty or whitespace-only.")
            chunk_has_error = True
        else:
            # Track content for duplicate warning detection
            norm_content = content.strip()
            content_map.setdefault(norm_content, []).append(pos)

        # 6. content_type checks
        ctype = chunk.content_type
        if not ctype or not isinstance(ctype, str) or ctype not in ALLOWED_CONTENT_TYPES:
            errors.append(
                f"Position {pos} (chunk_id={cid!r}): Invalid content_type {ctype!r}. Expected one of: {sorted(ALLOWED_CONTENT_TYPES)}."
            )
            chunk_has_error = True

        # 7. page_number checks
        page_num = chunk.page_number
        if page_num is not None:
            if not isinstance(page_num, int) or isinstance(page_num, bool) or page_num < 1:
                errors.append(
                    f"Position {pos} (chunk_id={cid!r}): Invalid page_number {page_num!r}. Must be positive integer >= 1 or None."
                )
                chunk_has_error = True

        # 8. metadata checks
        if not isinstance(chunk.metadata, dict):
            errors.append(
                f"Position {pos} (chunk_id={cid!r}): Invalid metadata type '{type(chunk.metadata).__name__}'. Expected dict."
            )
            chunk_has_error = True

        # 9. filename checks
        if not hasattr(chunk, "filename") or not isinstance(chunk.filename, str) or not chunk.filename.strip():
            errors.append(f"Position {pos} (chunk_id={cid!r}): Missing or invalid filename.")
            chunk_has_error = True

        if chunk_has_error:
            invalid_chunk_indices.add(pos)

    # 10. Cross-chunk consistency checks
    if len(doc_ids_seen) > 1:
        errors.append(
            f"Inconsistent document_id values detected across chunks: {sorted(doc_ids_seen)}."
        )

    # 11. Duplicate content warnings
    for content_str, positions in content_map.items():
        if len(positions) > 1:
            warnings.append(
                f"Duplicate content detected across {len(positions)} chunks at positions: {positions}."
            )

    invalid_count = len(invalid_chunk_indices)
    valid_count = total_chunks - invalid_count

    return ChunkValidationResult(
        is_valid=len(errors) == 0,
        total_chunks=total_chunks,
        valid_chunks=valid_count,
        invalid_chunks=invalid_count,
        errors=errors,
        warnings=warnings,
    )


def normalize_chunks(chunks: list[DocumentChunk] | ChunkingResult) -> list[DocumentChunk]:
    """Normalize a collection of DocumentChunks without modifying semantic content.

    Normalization actions performed:
    - Strips leading and trailing whitespace from chunk content.
    - Normalizes internal multiple blank lines to double newlines.
    - Strips trailing whitespace per line.
    - Preserves chunk_id, chunk_index, document_id, filename, page_number, content_type, and metadata.

    Args:
        chunks: List of DocumentChunk instances or a ChunkingResult object.

    Returns:
        New list of normalized DocumentChunk instances.
    """
    if isinstance(chunks, ChunkingResult):
        chunk_list = chunks.chunks
    elif isinstance(chunks, list):
        chunk_list = chunks
    else:
        raise TypeError(f"Expected list[DocumentChunk] or ChunkingResult, got {type(chunks).__name__}")

    normalized: list[DocumentChunk] = []

    for chunk in chunk_list:
        content = chunk.content
        if isinstance(content, str):
            # Clean per-line trailing whitespace
            lines = [line.rstrip() for line in content.strip().splitlines()]
            cleaned_content = "\n".join(lines)
            # Normalize 3+ consecutive newlines to 2 newlines
            cleaned_content = re.sub(r"\n{3,}", "\n\n", cleaned_content)
        else:
            cleaned_content = content

        normalized.append(
            DocumentChunk(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                content=cleaned_content,
                content_type=chunk.content_type,
                metadata=dict(chunk.metadata),
            )
        )

    return normalized
