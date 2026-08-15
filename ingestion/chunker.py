"""
Document chunker for the OmniBrain RAG ingestion pipeline.

Transforms parsed documents and unified ingestion results into structured,
typed DocumentChunks preserving page context, tables, and image references.
"""

from __future__ import annotations

import uuid
from typing import Any

from ingestion.models import (
    ChunkingResult,
    DocumentChunk,
    ExtractedImage,
    ExtractedTable,
    IngestionResult,
    ParsedDocument,
)


def _format_table_as_markdown(table: ExtractedTable) -> str:
    """Format an ExtractedTable's cells as a clean Markdown table string.

    Args:
        table: ExtractedTable object containing rows and columns.

    Returns:
        Markdown table string suitable for embedding and retrieval.
    """
    if not table.cells or table.rows == 0 or table.columns == 0:
        return f"[Table on Page {table.page_number} (Index {table.table_index}): Empty table {table.rows}x{table.columns}]"

    lines: list[str] = []
    # Header row
    header_cells = [str(c or "").strip() for c in table.cells[0]]
    lines.append("| " + " | ".join(header_cells) + " |")
    # Separator
    lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")

    # Data rows
    for row in table.cells[1:]:
        row_cells = [str(c or "").strip() for c in row]
        # Pad with empty strings if row has fewer cells than header
        if len(row_cells) < len(header_cells):
            row_cells.extend([""] * (len(header_cells) - len(row_cells)))
        lines.append("| " + " | ".join(row_cells[:len(header_cells)]) + " |")

    return "\n".join(lines)


def _format_image_as_text(image: ExtractedImage) -> str:
    """Format an ExtractedImage's metadata into a textual reference for chunking.

    Args:
        image: ExtractedImage object.

    Returns:
        Informative textual placeholder string.
    """
    return (
        f"[Image on Page {image.page_number} (Index {image.image_index}): "
        f"format={image.image_format.upper()}, width={image.width}px, "
        f"height={image.height}px, size={image.size_bytes} bytes]"
    )


def chunk_document(
    document: IngestionResult | ParsedDocument,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> ChunkingResult:
    """Chunk an ingested document into manageable units for vector embeddings.

    Supports text, table, and image modalities:
    - Text: Split per page using a configurable sliding window (`chunk_size`, `chunk_overlap`).
    - Tables: Formatted as structured Markdown tables preserving structural semantics.
    - Images: Structured references preserving image dimensions, format, and metadata.

    Args:
        document: IngestionResult or ParsedDocument to chunk.
        chunk_size: Maximum character count per text chunk (must be > 0).
        chunk_overlap: Number of overlapping characters between consecutive text chunks (must be >= 0 and < chunk_size).

    Returns:
        ChunkingResult containing the list of created DocumentChunks.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    doc_id = document.metadata.document_id
    filename = document.metadata.filename
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    # ── 1. Text Chunking (Per Page) ───────────────────────────────────────
    step = chunk_size - chunk_overlap

    for page in document.pages:
        if not page.has_content or not page.text.strip():
            continue

        text = page.text.strip()
        text_len = len(text)

        if text_len <= chunk_size:
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    chunk_index=chunk_index,
                    document_id=doc_id,
                    filename=filename,
                    page_number=page.page_number,
                    content=text,
                    content_type="text",
                    metadata={"char_count": text_len, "start_char": 0, "end_char": text_len},
                )
            )
            chunk_index += 1
        else:
            start = 0
            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunk_text = text[start:end]

                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        chunk_index=chunk_index,
                        document_id=doc_id,
                        filename=filename,
                        page_number=page.page_number,
                        content=chunk_text,
                        content_type="text",
                        metadata={
                            "char_count": len(chunk_text),
                            "start_char": start,
                            "end_char": end,
                        },
                    )
                )
                chunk_index += 1

                if end >= text_len:
                    break
                start += step

    # ── 2. Table Chunking ────────────────────────────────────────────────
    if hasattr(document, "tables") and document.tables:
        for table in document.tables:
            table_content = _format_table_as_markdown(table)
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    chunk_index=chunk_index,
                    document_id=doc_id,
                    filename=filename,
                    page_number=table.page_number,
                    content=table_content,
                    content_type="table",
                    metadata={
                        "table_index": table.table_index,
                        "rows": table.rows,
                        "columns": table.columns,
                    },
                )
            )
            chunk_index += 1

    # ── 3. Image Reference Chunking ──────────────────────────────────────
    if hasattr(document, "images") and document.images:
        for img in document.images:
            img_content = _format_image_as_text(img)
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    chunk_index=chunk_index,
                    document_id=doc_id,
                    filename=filename,
                    page_number=img.page_number,
                    content=img_content,
                    content_type="image",
                    metadata={
                        "image_index": img.image_index,
                        "image_format": img.image_format,
                        "width": img.width,
                        "height": img.height,
                        "size_bytes": img.size_bytes,
                        "colorspace": img.colorspace,
                        "xref": img.xref,
                    },
                )
            )
            chunk_index += 1

    return ChunkingResult(
        document_id=doc_id,
        filename=filename,
        chunks=chunks,
    )
