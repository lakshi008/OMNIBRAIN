"""
Tests for the IngestionError hierarchy.

Verifies base exception creation, inheritance, stage preservation,
message formatting, and original exception retention across all custom error types.
"""

from __future__ import annotations

import pytest

from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
)
from ingestion.ingestion_errors import (
    IngestionChunkingError,
    IngestionEmbeddingError,
    IngestionError,
    IngestionExtractionError,
    IngestionPipelineError,
    IngestionValidationError,
)


class TestIngestionErrorHierarchy:
    """Tests for base IngestionError class and subclasses."""

    def test_base_error_creation(self) -> None:
        """Base IngestionError correctly formats stage and message."""
        err = IngestionError("Base failure message", stage="CUSTOM_STAGE")
        assert isinstance(err, Exception)
        assert err.message == "Base failure message"
        assert err.stage == "CUSTOM_STAGE"
        assert err.original_error is None
        assert str(err) == "[CUSTOM_STAGE] Base failure message"

    def test_base_error_with_original_error(self) -> None:
        """Original exception is preserved on the IngestionError instance."""
        original = ValueError("Original root cause")
        err = IngestionError("Wrapper error", stage="EXTRACTION", original_error=original)
        assert err.original_error is original
        assert isinstance(err.original_error, ValueError)

    def test_validation_error_is_value_error_and_ingestion_error(self) -> None:
        """IngestionValidationError inherits from both IngestionError and ValueError."""
        err = IngestionValidationError("Invalid chunk configuration", stage="VALIDATION")
        assert isinstance(err, IngestionError)
        assert isinstance(err, IngestionValidationError)
        assert isinstance(err, ValueError)
        assert err.stage == "VALIDATION"
        assert "[VALIDATION]" in str(err)

    def test_extraction_error_hierarchy(self) -> None:
        """IngestionExtractionError is an IngestionError with EXTRACTION stage."""
        err = IngestionExtractionError("Could not extract PDF", stage="EXTRACTION")
        assert isinstance(err, IngestionError)
        assert isinstance(err, IngestionExtractionError)
        assert err.stage == "EXTRACTION"

    def test_chunking_error_hierarchy(self) -> None:
        """IngestionChunkingError is an IngestionError and ValueError."""
        err = IngestionChunkingError("Chunking split failed", stage="CHUNKING")
        assert isinstance(err, IngestionError)
        assert isinstance(err, IngestionChunkingError)
        assert isinstance(err, ValueError)
        assert err.stage == "CHUNKING"

    def test_embedding_error_hierarchy(self) -> None:
        """IngestionEmbeddingError is an IngestionError and ValueError."""
        err = IngestionEmbeddingError("Provider dimension mismatch", stage="EMBEDDING_GENERATION")
        assert isinstance(err, IngestionError)
        assert isinstance(err, IngestionEmbeddingError)
        assert isinstance(err, ValueError)
        assert err.stage == "EMBEDDING_GENERATION"

    def test_pipeline_error_hierarchy(self) -> None:
        """IngestionPipelineError is an IngestionError."""
        err = IngestionPipelineError("Pipeline orchestration failed", stage="PIPELINE")
        assert isinstance(err, IngestionError)
        assert isinstance(err, IngestionPipelineError)
        assert err.stage == "PIPELINE"

    def test_pdf_exceptions_inherit_from_ingestion_extraction_error(self) -> None:
        """PDFNotFoundError, InvalidFileTypeError, and CorruptedPDFError inherit from IngestionExtractionError."""
        err_not_found = PDFNotFoundError("missing.pdf")
        assert isinstance(err_not_found, IngestionExtractionError)
        assert isinstance(err_not_found, IngestionError)
        assert err_not_found.stage == "EXTRACTION"
        assert err_not_found.filepath == "missing.pdf"

        err_invalid_type = InvalidFileTypeError("doc.txt", ".txt")
        assert isinstance(err_invalid_type, IngestionExtractionError)
        assert isinstance(err_invalid_type, IngestionError)
        assert err_invalid_type.extension == ".txt"

        err_corrupted = CorruptedPDFError("bad.pdf", reason="syntax error")
        assert isinstance(err_corrupted, IngestionExtractionError)
        assert isinstance(err_corrupted, IngestionError)
        assert err_corrupted.reason == "syntax error"
