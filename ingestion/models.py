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


@dataclass(frozen=True)
class ExtractedTable:
    """Represents an extracted table from a PDF page.

    Attributes:
        page_number: 1-indexed page number where table was found.
        table_index: 0-indexed index of table on this page.
        rows: Number of rows in table.
        columns: Number of columns in table.
        cells: 2D list of cell values (rows x columns).
    """

    page_number: int
    table_index: int
    rows: int
    columns: int
    cells: list[list[str | None]]


@dataclass
class TableExtractionResult:
    """Complete result of extracting tables from a PDF document.

    Attributes:
        metadata: Document-level metadata.
        tables: List of ExtractedTable objects found across all pages.
    """

    metadata: DocumentMetadata
    tables: list[ExtractedTable]

    @property
    def total_tables(self) -> int:
        """Total number of tables found across the document."""
        return len(self.tables)

    @property
    def has_tables(self) -> bool:
        """Whether any tables were found in the document."""
        return len(self.tables) > 0

    def get_tables_on_page(self, page_number: int) -> list[ExtractedTable]:
        """Return all tables extracted from a specific 1-indexed page.

        Args:
            page_number: The 1-indexed page number to look up.

        Returns:
            List of ExtractedTable objects for that page.
        """
        return [table for table in self.tables if table.page_number == page_number]


@dataclass(frozen=True)
class ExtractedImage:
    """Represents an extracted image from a PDF page.

    Attributes:
        page_number: 1-indexed page number where image was found.
        image_index: 0-indexed index of image on this page.
        image_format: Format/extension of the image (e.g. 'png', 'jpeg').
        width: Width of image in pixels.
        height: Height of image in pixels.
        image_bytes: Raw binary bytes of the image.
        size_bytes: Size of the image in bytes.
        colorspace: Colorspace name or component count (e.g. 'DeviceRGB').
        xref: Internal PDF cross-reference ID of the image object.
    """

    page_number: int
    image_index: int
    image_format: str
    width: int
    height: int
    image_bytes: bytes
    size_bytes: int
    colorspace: str
    xref: int


@dataclass
class ImageExtractionResult:
    """Complete result of extracting images from a PDF document.

    Attributes:
        metadata: Document-level metadata.
        images: List of ExtractedImage objects found across all pages.
    """

    metadata: DocumentMetadata
    images: list[ExtractedImage]

    @property
    def total_images(self) -> int:
        """Total number of images found across the document."""
        return len(self.images)

    @property
    def has_images(self) -> bool:
        """Whether any images were found in the document."""
        return len(self.images) > 0

    def get_images_on_page(self, page_number: int) -> list[ExtractedImage]:
        """Return all images extracted from a specific 1-indexed page.

        Args:
            page_number: The 1-indexed page number to look up.

        Returns:
            List of ExtractedImage objects for that page.
        """
        return [img for img in self.images if img.page_number == page_number]


@dataclass
class IngestionResult:
    """Unified result of the complete PDF ingestion pipeline.

    Combines document metadata, page text data, extracted tables, and extracted images.

    Attributes:
        metadata: DocumentMetadata describing the parsed document.
        pages: List of PageData objects for each page.
        tables: List of ExtractedTable objects found across the document.
        images: List of ExtractedImage objects found across the document.
    """

    metadata: DocumentMetadata
    pages: list[PageData]
    tables: list[ExtractedTable]
    images: list[ExtractedImage]

    @property
    def total_pages(self) -> int:
        """Total number of pages in the document."""
        return self.metadata.total_pages

    @property
    def pages_with_text(self) -> int:
        """Number of pages with extractable text."""
        return self.metadata.pages_with_content

    @property
    def pages_without_text(self) -> int:
        """Number of pages without extractable text."""
        return self.metadata.pages_without_content

    @property
    def total_tables(self) -> int:
        """Total number of tables extracted across all pages."""
        return len(self.tables)

    @property
    def total_images(self) -> int:
        """Total number of images extracted across all pages."""
        return len(self.images)

    @property
    def has_text(self) -> bool:
        """Whether the document contains any extractable text."""
        return self.pages_with_text > 0

    @property
    def has_tables(self) -> bool:
        """Whether any tables were extracted from the document."""
        return len(self.tables) > 0

    @property
    def has_images(self) -> bool:
        """Whether any images were extracted from the document."""
        return len(self.images) > 0

    def get_page(self, page_number: int) -> PageData | None:
        """Return PageData for a specific 1-indexed page number."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def get_tables_on_page(self, page_number: int) -> list[ExtractedTable]:
        """Return all tables extracted from a specific 1-indexed page."""
        return [table for table in self.tables if table.page_number == page_number]

    def get_images_on_page(self, page_number: int) -> list[ExtractedImage]:
        """Return all images extracted from a specific 1-indexed page."""
        return [img for img in self.images if img.page_number == page_number]

    def get_all_text(self, separator: str = "\n\n") -> str:
        """Concatenate text from all content pages."""
        return separator.join(
            page.text for page in self.pages if page.has_content
        )



