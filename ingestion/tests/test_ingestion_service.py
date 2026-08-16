"""
Tests for the end-to-end ingestion service.

Verifies complete execution from PDF parsing, table and image extraction,
chunking, normalization, validation, embedding preparation to embedding generation.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
)
from ingestion.ingestion_service import run_ingestion
from ingestion.models import (
    EmbeddingGenerationResult,
    EmbeddingVectorRecord,
)


# ── Deterministic Test Provider ──────────────────────────────────────────


class DeterministicTestEmbeddingProvider:
    """Deterministic embedding provider for testing without external calls."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector: list[float] = []
        for i in range(self.dimension):
            val = (sum((ord(c) * (i + 1)) for c in text) % 1000) / 1000.0
            vector.append(round(val, 4))
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class BrokenEmbeddingProvider:
    """Provider returning broken vectors to test pipeline error propagation."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0] if i == 0 else [1.0] for i in range(len(texts))]


# ── Fixtures & PDF Helpers ───────────────────────────────────────────────


def make_dummy_image_bytes(width: int = 60, height: int = 40, color: int = 128) -> bytes:
    """Create raw PNG image bytes using PyMuPDF Pixmap."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), 0)
    pix.clear_with(color)
    return pix.tobytes("png")


def draw_table(
    page: pymupdf.Page,
    start_x: float,
    start_y: float,
    col_widths: list[float],
    row_heights: list[float],
    data: list[list[str]],
) -> None:
    """Draw a grid table with lines and text in a PyMuPDF page."""
    total_width = sum(col_widths)
    total_height = sum(row_heights)

    rect = pymupdf.Rect(start_x, start_y, start_x + total_width, start_y + total_height)
    page.draw_rect(rect, color=(0, 0, 0), width=1)

    curr_y = start_y
    for h in row_heights[:-1]:
        curr_y += h
        page.draw_line(
            pymupdf.Point(start_x, curr_y),
            pymupdf.Point(start_x + total_width, curr_y),
            color=(0, 0, 0),
            width=0.5,
        )

    curr_x = start_x
    for w in col_widths[:-1]:
        curr_x += w
        page.draw_line(
            pymupdf.Point(curr_x, start_y),
            pymupdf.Point(curr_x, start_y + total_height),
            color=(0, 0, 0),
            width=0.5,
        )

    y = start_y
    for r_idx, row in enumerate(data):
        x = start_x
        for c_idx, cell_text in enumerate(row):
            page.insert_text(
                pymupdf.Point(x + 5, y + row_heights[r_idx] - 5),
                cell_text,
                fontsize=9,
            )
            x += col_widths[c_idx]
        y += row_heights[r_idx]


@pytest.fixture
def provider() -> DeterministicTestEmbeddingProvider:
    """Create a standard 8-dimensional test provider."""
    return DeterministicTestEmbeddingProvider(dimension=8)


@pytest.fixture
def text_only_pdf(tmp_path: Path) -> Path:
    """Create a 2-page text-only PDF."""
    pdf_path = tmp_path / "text_only.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 100), "OmniBrain architecture overview and supervisor design.")
    p2 = doc.new_page()
    p2.insert_text((50, 100), "Multi-modal RAG retrieval and citation engine details.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def table_pdf(tmp_path: Path) -> Path:
    """Create a PDF with a structured table."""
    pdf_path = tmp_path / "table_doc.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Financial Performance Table:")
    draw_table(
        p1,
        start_x=50,
        start_y=80,
        col_widths=[100, 100],
        row_heights=[20, 20],
        data=[["Quarter", "Revenue"], ["Q1", "$10M"]],
    )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def image_pdf(tmp_path: Path) -> Path:
    """Create a PDF with an embedded image."""
    pdf_path = tmp_path / "image_doc.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Figure 1 Architecture:")
    img_bytes = make_dummy_image_bytes(80, 50, color=180)
    p1.insert_image(pymupdf.Rect(50, 80, 150, 140), stream=img_bytes)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def mixed_pdf(tmp_path: Path) -> Path:
    """Create a multi-modal PDF with text, tables, and images."""
    pdf_path = tmp_path / "mixed_doc.pdf"
    doc = pymupdf.open()
    
    # Page 1: Text + Table
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Executive Summary and Key Metrics:")
    draw_table(
        p1,
        start_x=50,
        start_y=80,
        col_widths=[100, 100],
        row_heights=[20, 20],
        data=[["Metric", "Value"], ["Uptime", "99.9%"]],
    )
    
    # Page 2: Text + Image
    p2 = doc.new_page()
    p2.insert_text((50, 50), "System Architecture Diagram:")
    img_bytes = make_dummy_image_bytes(100, 60, color=200)
    p2.insert_image(pymupdf.Rect(50, 80, 180, 160), stream=img_bytes)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """Create an empty 2-page PDF."""
    pdf_path = tmp_path / "empty_doc.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ── Success Tests ────────────────────────────────────────────────────────


class TestIngestionServiceSuccess:
    """Tests for successful end-to-end execution of run_ingestion."""

    def test_text_only_ingestion(
        self, text_only_pdf: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Runs end-to-end on text-only PDF and produces embedding vectors."""
        result = run_ingestion(text_only_pdf, provider)
        assert isinstance(result, EmbeddingGenerationResult)
        assert result.is_ready is True
        assert result.dimension == 8
        assert result.total_items == 2
        assert result.text_items == 2
        assert result.table_items == 0
        assert result.image_items == 0
        for item in result.items:
            assert isinstance(item, EmbeddingVectorRecord)
            assert len(item.vector) == 8

    def test_table_ingestion(
        self, table_pdf: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Runs end-to-end on PDF with table and extracts text + table chunks."""
        result = run_ingestion(table_pdf, provider)
        assert result.is_ready is True
        assert result.total_items >= 2
        assert result.table_items == 1
        table_rec = result.get_vectors_by_type("table")[0]
        assert table_rec.content_type == "table"
        assert table_rec.metadata["rows"] == 2
        assert table_rec.metadata["columns"] == 2

    def test_image_ingestion(
        self, image_pdf: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Runs end-to-end on PDF with image and extracts text + image reference chunks."""
        result = run_ingestion(image_pdf, provider)
        assert result.is_ready is True
        assert result.image_items == 1
        img_rec = result.get_vectors_by_type("image")[0]
        assert img_rec.content_type == "image"

    def test_mixed_multi_modal_ingestion(
        self, mixed_pdf: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Runs end-to-end on multi-modal PDF containing text, tables, and images."""
        result = run_ingestion(mixed_pdf, provider)
        assert result.is_ready is True
        assert result.text_items >= 2
        assert result.table_items == 1
        assert result.image_items == 1
        assert result.total_items == result.text_items + result.table_items + result.image_items

    def test_empty_pdf_ingestion(
        self, empty_pdf: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Empty PDF returns safely with 0 items and dimension 0."""
        result = run_ingestion(empty_pdf, provider)
        assert result.is_ready is True
        assert result.total_items == 0
        assert result.dimension == 0
        assert result.items == []

    def test_custom_chunking_parameters(
        self, text_only_pdf: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Custom chunk_size and chunk_overlap are respected."""
        result = run_ingestion(
            text_only_pdf,
            provider,
            chunk_size=20,
            chunk_overlap=5,
        )
        assert result.is_ready is True
        assert result.total_items > 2


# ── Lineage & Attribute Preservation Tests ───────────────────────────────


class TestLineageAndAttributes:
    """Tests verifying complete preservation of citation and document lineage."""

    def test_metadata_and_lineage_preserved(
        self, mixed_pdf: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Preserves document_id, filename, page_number, chunk_index, and metadata."""
        result = run_ingestion(mixed_pdf, provider)
        assert len(result.document_id) > 0
        assert result.filename == "mixed_doc.pdf"

        for idx, item in enumerate(result.items):
            assert item.document_id == result.document_id
            assert item.filename == "mixed_doc.pdf"
            assert item.chunk_index == idx
            assert item.page_number in (1, 2)
            assert isinstance(item.metadata, dict)
            assert "chunk_id" in item.metadata
            assert "content_type" in item.metadata

    def test_deterministic_ordering(
        self, mixed_pdf: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Multiple runs on the same PDF produce identical chunk indices and vectors."""
        res1 = run_ingestion(mixed_pdf, provider)
        res2 = run_ingestion(mixed_pdf, provider)

        indices1 = [r.chunk_index for r in res1.items]
        indices2 = [r.chunk_index for r in res2.items]
        assert indices1 == indices2 == list(range(len(res1.items)))

        types1 = [r.content_type for r in res1.items]
        types2 = [r.content_type for r in res2.items]
        assert types1 == types2


# ── Validation & Error Handling Tests ────────────────────────────────────


class TestValidationAndErrorHandling:
    """Tests for strict input validation and pipeline error propagation."""

    def test_nonexistent_pdf_raises_not_found(
        self, tmp_path: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Missing PDF raises PDFNotFoundError."""
        with pytest.raises(PDFNotFoundError):
            run_ingestion(tmp_path / "missing.pdf", provider)

    def test_invalid_extension_raises_invalid_type(
        self, tmp_path: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Non-PDF file raises InvalidFileTypeError."""
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("Hello")
        with pytest.raises(InvalidFileTypeError):
            run_ingestion(txt_file, provider)

    def test_corrupted_pdf_raises_corrupted_error(
        self, tmp_path: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Corrupted PDF raises CorruptedPDFError."""
        corrupted = tmp_path / "corrupted.pdf"
        corrupted.write_bytes(b"not a valid pdf data")
        with pytest.raises(CorruptedPDFError):
            run_ingestion(corrupted, provider)

    def test_invalid_chunk_size_zero_raises(
        self, text_only_pdf: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """chunk_size <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            run_ingestion(text_only_pdf, provider, chunk_size=0)

    def test_invalid_chunk_size_negative_raises(
        self, text_only_pdf: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Negative chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            run_ingestion(text_only_pdf, provider, chunk_size=-100)

    def test_invalid_chunk_overlap_negative_raises(
        self, text_only_pdf: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Negative chunk_overlap raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer"):
            run_ingestion(text_only_pdf, provider, chunk_overlap=-10)

    def test_chunk_overlap_greater_than_chunk_size_raises(
        self, text_only_pdf: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """chunk_overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="strictly less than chunk_size"):
            run_ingestion(text_only_pdf, provider, chunk_size=100, chunk_overlap=100)

    def test_invalid_embedding_provider_raises(
        self, text_only_pdf: Path
    ) -> None:
        """Invalid embedding provider raises TypeError."""
        with pytest.raises(TypeError, match="Invalid embedding provider"):
            run_ingestion(text_only_pdf, "invalid_provider")  # type: ignore

    def test_broken_embedding_provider_propagates_error(
        self, text_only_pdf: Path
    ) -> None:
        """Inconsistent dimensions from provider propagate descriptive ValueError."""
        broken = BrokenEmbeddingProvider()
        with pytest.raises(ValueError, match="Inconsistent vector dimension"):
            run_ingestion(text_only_pdf, broken)


# ── Status Tracking Integration Tests ────────────────────────────────────


class TestServiceStatusTracking:
    """Tests verifying IngestionStatus tracking integration with run_ingestion."""

    def test_successful_ingestion_updates_status_tracker(
        self, text_only_pdf: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """status_tracker records stages and completes in COMPLETED status."""
        from ingestion.ingestion_status import IngestionStatus, PipelineStage, PipelineStatus

        tracker = IngestionStatus()
        result = run_ingestion(text_only_pdf, provider, status_tracker=tracker)

        assert tracker.status == PipelineStatus.COMPLETED
        assert tracker.current_stage == PipelineStage.COMPLETED
        assert tracker.error is None
        assert tracker.filename == "text_only.pdf"
        assert tracker.document_id == result.document_id
        assert PipelineStage.EXTRACTION in tracker.completed_stages
        assert PipelineStage.CHUNKING in tracker.completed_stages
        assert PipelineStage.NORMALIZATION in tracker.completed_stages
        assert PipelineStage.VALIDATION in tracker.completed_stages
        assert PipelineStage.EMBEDDING_PREPARATION in tracker.completed_stages
        assert PipelineStage.EMBEDDING_GENERATION in tracker.completed_stages

    def test_failed_extraction_marks_tracker_failed(
        self, tmp_path: Path, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """When extraction fails, status_tracker transitions to FAILED."""
        from ingestion.ingestion_errors import IngestionExtractionError
        from ingestion.ingestion_status import IngestionStatus, PipelineStatus

        tracker = IngestionStatus()
        with pytest.raises(IngestionExtractionError):
            run_ingestion(tmp_path / "missing.pdf", provider, status_tracker=tracker)

        assert tracker.status == PipelineStatus.FAILED
        assert tracker.error is not None

    def test_failed_embedding_marks_tracker_failed(
        self, text_only_pdf: Path
    ) -> None:
        """When embedding provider fails, status_tracker transitions to FAILED."""
        from ingestion.ingestion_errors import IngestionEmbeddingError
        from ingestion.ingestion_status import IngestionStatus, PipelineStage, PipelineStatus

        broken = BrokenEmbeddingProvider()
        tracker = IngestionStatus()
        with pytest.raises(IngestionEmbeddingError):
            run_ingestion(text_only_pdf, broken, status_tracker=tracker)

        assert tracker.status == PipelineStatus.FAILED
        assert tracker.current_stage == PipelineStage.EMBEDDING_GENERATION
        assert tracker.error is not None

