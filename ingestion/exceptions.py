"""
Custom exceptions for the OmniBrain ingestion pipeline.

These exceptions provide clear, typed error handling for PDF validation
and parsing failures.
"""


class PDFNotFoundError(Exception):
    """Raised when the specified PDF file does not exist."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        super().__init__(f"PDF file not found: {filepath}")


class InvalidFileTypeError(Exception):
    """Raised when the file is not a PDF (wrong extension)."""

    def __init__(self, filepath: str, extension: str) -> None:
        self.filepath = filepath
        self.extension = extension
        super().__init__(
            f"Invalid file type '{extension}'. Expected '.pdf': {filepath}"
        )


class CorruptedPDFError(Exception):
    """Raised when the PDF file cannot be opened or parsed (corrupted data)."""

    def __init__(self, filepath: str, reason: str = "") -> None:
        self.filepath = filepath
        self.reason = reason
        detail = f" ({reason})" if reason else ""
        super().__init__(f"Corrupted or unreadable PDF{detail}: {filepath}")
