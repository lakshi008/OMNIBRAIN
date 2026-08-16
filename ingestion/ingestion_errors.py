"""
Structured exception hierarchy for the OmniBrain ingestion pipeline.

Provides typed exceptions with pipeline stage attribution, descriptive messages,
and optional original exception tracking.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base exception for all ingestion pipeline errors.

    Attributes:
        message: Descriptive explanation of what failed.
        stage: Pipeline stage where the failure occurred.
        original_error: The underlying caught exception (if any).
    """

    def __init__(
        self,
        message: str,
        stage: str = "PIPELINE",
        original_error: Exception | None = None,
    ) -> None:
        self.message = message
        self.stage = stage
        self.original_error = original_error
        super().__init__(f"[{self.stage}] {self.message}")


class IngestionValidationError(IngestionError, ValueError):
    """Raised when validation of input documents, chunks, or parameters fails."""

    def __init__(
        self,
        message: str,
        stage: str = "VALIDATION",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message=message, stage=stage, original_error=original_error)


class IngestionExtractionError(IngestionError):
    """Raised when extracting text, tables, or images from a PDF fails."""

    def __init__(
        self,
        message: str,
        stage: str = "EXTRACTION",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message=message, stage=stage, original_error=original_error)


class IngestionChunkingError(IngestionError, ValueError):
    """Raised when chunking document content fails."""

    def __init__(
        self,
        message: str,
        stage: str = "CHUNKING",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message=message, stage=stage, original_error=original_error)


class IngestionEmbeddingError(IngestionError, ValueError):
    """Raised when preparing or generating embedding vectors fails."""

    def __init__(
        self,
        message: str,
        stage: str = "EMBEDDING_GENERATION",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message=message, stage=stage, original_error=original_error)


class IngestionPipelineError(IngestionError):
    """Raised for high-level pipeline orchestration or configuration failures."""

    def __init__(
        self,
        message: str,
        stage: str = "PIPELINE",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message=message, stage=stage, original_error=original_error)
