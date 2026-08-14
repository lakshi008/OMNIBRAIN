"""
Data models for the OmniBrain ingestion pipeline.

Provides structured, typed representations for parsed PDF documents,
individual pages, and document metadata.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class PageData:
    """Represents extracted data from a single PDF page.

    Attributes:
        page_number: 1-indexed page number.
        text: Extracted text content (stripped of leading/trailing whitespace).
        char_count: Number of characters in the extracted text.
        has_content: Whether the page contains any extractable text.
    """

    page_number: int
    text: str
    char_count: int
    has_content: bool


@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata describing a parsed PDF document.

    Attributes:
        document_id: Unique identifier (UUID4) for this document parse.
        filename: Original filename of the PDF.
        total_pages: Total number of pages in the document.
        content_type: MIME type of the source file.
        created_at: ISO 8601 timestamp of when the document was parsed.
        pages_with_content: Number of pages that contain extractable text.
        pages_without_content: Number of pages with no extractable text.
    """

    document_id: str
    filename: str
    total_pages: int
    content_type: str
    created_at: str
    pages_with_content: int
    pages_without_content: int


@dataclass
class ParsedDocument:
    """Complete result of parsing a PDF document.

    Attributes:
        metadata: Document-level metadata.
        pages: List of PageData objects, one per page.
    """

    metadata: DocumentMetadata
    pages: list[PageData]

    def get_page(self, page_number: int) -> PageData | None:
        """Return a specific page by its 1-indexed page number.

        Args:
            page_number: The 1-indexed page number to retrieve.

        Returns:
            The PageData for the requested page, or None if not found.
        """
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def get_all_text(self, separator: str = "\n\n") -> str:
        """Concatenate text from all pages.

        Args:
            separator: String to place between page texts.

        Returns:
            Combined text from all pages.
        """
        return separator.join(
            page.text for page in self.pages if page.has_content
        )
