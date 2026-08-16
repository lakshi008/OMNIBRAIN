"""
Custom exceptions for the OmniBrain ingestion pipeline.

These exceptions provide clear, typed error handling for PDF validation
and parsing failures, inheriting from IngestionExtractionError.
"""

from __future__ import annotations

from ingestion.ingestion_errors import IngestionExtractionError


class PDFNotFoundError(IngestionExtractionError):
    """Raised when the specified PDF file does not exist."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        super().__init__(
            message=f"PDF file not found: {filepath}",
            stage="EXTRACTION",
        )


class InvalidFileTypeError(IngestionExtractionError):
    """Raised when the file is not a PDF (wrong extension)."""

    def __init__(self, filepath: str, extension: str) -> None:
        self.filepath = filepath
        self.extension = extension
        super().__init__(
            message=f"Invalid file type '{extension}'. Expected '.pdf': {filepath}",
            stage="EXTRACTION",
        )


class CorruptedPDFError(IngestionExtractionError):
    """Raised when the PDF file cannot be opened or parsed (corrupted data)."""

    def __init__(self, filepath: str, reason: str = "") -> None:
        self.filepath = filepath
        self.reason = reason
        detail = f" ({reason})" if reason else ""
        super().__init__(
            message=f"Corrupted or unreadable PDF{detail}: {filepath}",
            stage="EXTRACTION",
        )
