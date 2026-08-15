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


@dataclass(frozen=True)
class DocumentChunk:
    """Represents an individual chunk of document content for RAG pipelines.

    Attributes:
        chunk_id: Unique UUID identifier for this chunk.
        chunk_index: 0-indexed sequential position of this chunk within the document.
        document_id: Unique identifier of the parent document.
        filename: Name of the source file.
        page_number: 1-indexed page number from which content originated.
        content: Textual content of the chunk (raw text, serialized table, or image reference).
        content_type: Type of content ('text', 'table', 'image').
        metadata: Additional contextual metadata (e.g. dimensions, table indices, etc.).
    """

    chunk_id: str
    chunk_index: int
    document_id: str
    filename: str
    page_number: int | None
    content: str
    content_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingResult:
    """Structured result of chunking an ingested document.

    Attributes:
        document_id: Unique identifier of the parent document.
        filename: Name of the source file.
        chunks: List of DocumentChunk objects generated from the document.
    """

    document_id: str
    filename: str
    chunks: list[DocumentChunk]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks created."""
        return len(self.chunks)

    @property
    def text_chunks(self) -> int:
        """Number of text chunks."""
        return sum(1 for c in self.chunks if c.content_type == "text")

    @property
    def table_chunks(self) -> int:
        """Number of table chunks."""
        return sum(1 for c in self.chunks if c.content_type == "table")

    @property
    def image_chunks(self) -> int:
        """Number of image chunks."""
        return sum(1 for c in self.chunks if c.content_type == "image")

    @property
    def has_chunks(self) -> bool:
        """Whether any chunks were generated."""
        return len(self.chunks) > 0

    def get_chunks_by_type(self, content_type: str) -> list[DocumentChunk]:
        """Filter and return chunks matching a specific content type.

        Args:
            content_type: One of 'text', 'table', 'image'.

        Returns:
            List of matching DocumentChunk objects.
        """
        return [c for c in self.chunks if c.content_type == content_type]

    def get_chunks_on_page(self, page_number: int) -> list[DocumentChunk]:
        """Filter and return chunks associated with a specific 1-indexed page number.

        Args:
            page_number: The 1-indexed page number.

        Returns:
            List of matching DocumentChunk objects.
        """
        return [c for c in self.chunks if c.page_number == page_number]


@dataclass
class ChunkValidationResult:
    """Result of validating a collection of DocumentChunk objects.

    Attributes:
        is_valid: True if no errors were encountered.
        total_chunks: Total number of chunks evaluated.
        valid_chunks: Number of valid chunks.
        invalid_chunks: Number of chunks with errors.
        errors: Detailed list of validation error messages.
        warnings: List of non-fatal warning messages (e.g., duplicate content).
    """

    is_valid: bool
    total_chunks: int
    valid_chunks: int
    invalid_chunks: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EmbeddingRecord:
    """Represents an embedding-ready record prepared from a DocumentChunk.

    Separates the text content to be embedded from the structured metadata required
    for vector storage, filtering, and RAG citation generation.

    Attributes:
        chunk_id: Unique UUID identifier of the source chunk.
        document_id: Unique identifier of the parent document.
        filename: Name of the source file.
        chunk_index: 0-indexed sequential position within the document.
        page_number: 1-indexed page number from which content originated (or None).
        content: The text content to be embedded.
        content_type: Content modality ('text', 'table', 'image').
        metadata: Separate metadata dictionary for vector payload/filtering/citations.
    """

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None
    content: str
    content_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingPreparationResult:
    """Structured result of preparing validated chunks for embedding generation.

    Attributes:
        document_id: Unique identifier of the parent document.
        filename: Name of the source file.
        items: Deterministically ordered list of EmbeddingRecord objects.
        is_ready: True if records were successfully validated and prepared.
    """

    document_id: str
    filename: str
    items: list[EmbeddingRecord]
    is_ready: bool

    @property
    def total_items(self) -> int:
        """Total number of embedding-ready records."""
        return len(self.items)

    @property
    def text_items(self) -> int:
        """Number of text records."""
        return sum(1 for item in self.items if item.content_type == "text")

    @property
    def table_items(self) -> int:
        """Number of table records."""
        return sum(1 for item in self.items if item.content_type == "table")

    @property
    def image_items(self) -> int:
        """Number of image records."""
        return sum(1 for item in self.items if item.content_type == "image")

    def get_items_by_type(self, content_type: str) -> list[EmbeddingRecord]:
        """Filter embedding records by content type ('text', 'table', 'image')."""
        return [item for item in self.items if item.content_type == content_type]

    def get_items_on_page(self, page_number: int) -> list[EmbeddingRecord]:
        """Filter embedding records by 1-indexed page number."""
        return [item for item in self.items if item.page_number == page_number]


@dataclass(frozen=True)
class EmbeddingVectorRecord:
    """Represents an embedding vector associated with its source chunk and metadata.

    Attributes:
        chunk_id: Unique UUID identifier of the source chunk.
        document_id: Unique identifier of the parent document.
        filename: Name of the source file.
        chunk_index: 0-indexed sequential position within the document.
        page_number: 1-indexed page number from which content originated (or None).
        content_type: Content modality ('text', 'table', 'image').
        vector: Dense embedding vector as a list of floats.
        metadata: Full metadata payload for vector store indexing, filtering, and citations.
    """

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None
    content_type: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingGenerationResult:
    """Structured result of generating embeddings for a document's chunks.

    Attributes:
        document_id: Unique identifier of the parent document.
        filename: Name of the source file.
        items: Deterministically ordered list of EmbeddingVectorRecord objects.
        dimension: Dimensionality of the generated embedding vectors.
        is_ready: True if vectors were successfully generated and validated.
    """

    document_id: str
    filename: str
    items: list[EmbeddingVectorRecord]
    dimension: int
    is_ready: bool

    @property
    def total_items(self) -> int:
        """Total number of generated vector records."""
        return len(self.items)

    @property
    def text_items(self) -> int:
        """Number of text vector records."""
        return sum(1 for item in self.items if item.content_type == "text")

    @property
    def table_items(self) -> int:
        """Number of table vector records."""
        return sum(1 for item in self.items if item.content_type == "table")

    @property
    def image_items(self) -> int:
        """Number of image vector records."""
        return sum(1 for item in self.items if item.content_type == "image")

    def get_vectors_by_type(self, content_type: str) -> list[EmbeddingVectorRecord]:
        """Filter embedding vector records by content type ('text', 'table', 'image')."""
        return [item for item in self.items if item.content_type == content_type]

    def get_vectors_on_page(self, page_number: int) -> list[EmbeddingVectorRecord]:
        """Filter embedding vector records by 1-indexed page number."""
        return [item for item in self.items if item.page_number == page_number]


@dataclass(frozen=True)
class VectorSearchResult:
    """Represents a single search result retrieved from the Qdrant vector store.

    Attributes:
        chunk_id: Unique UUID identifier of the matched chunk.
        score: Similarity score (e.g. Cosine similarity).
        document_id: Unique identifier of the parent document.
        filename: Name of the source file.
        page_number: 1-indexed page number from which content originated (or None).
        chunk_index: 0-indexed sequential position within the document.
        content_type: Content modality ('text', 'table', 'image').
        content: Text content associated with the vector.
        metadata: Full metadata payload for filtering, citations, and lineage.
    """

    chunk_id: str
    score: float
    document_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    content_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)








