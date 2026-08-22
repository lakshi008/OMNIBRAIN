"""
Unit tests for Member 3 Vision Agent domain models.
"""

from __future__ import annotations

from typing import Any
import pytest

from agents.models import AgentCitation
from ingestion.models import VectorSearchResult
from vision.exceptions import VisionEvidenceError, VisionInputValidationError
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)


class TestVisualEvidenceModel:
    """Test suite for VisualEvidence data model and lineage validation."""

    def test_01_valid_visual_evidence_creation(self) -> None:
        """VisualEvidence constructs with valid required and optional fields."""
        ev = VisualEvidence(
            document_id="doc-100",
            filename="financial_chart.pdf",
            chunk_id="chk-img-01",
            page_number=3,
            chunk_index=2,
            content_type="chart",
            image_path="/data/images/chart_p3.png",
            image_format="png",
            width=800,
            height=600,
            description="Q3 Revenue bar chart",
            metadata={"chart_type": "bar"},
        )
        assert ev.document_id == "doc-100"
        assert ev.filename == "financial_chart.pdf"
        assert ev.chunk_id == "chk-img-01"
        assert ev.page_number == 3
        assert ev.chunk_index == 2
        assert ev.content_type == "chart"
        assert ev.image_path == "/data/images/chart_p3.png"
        assert ev.width == 800
        assert ev.height == 600
        assert ev.metadata == {"chart_type": "bar"}

    @pytest.mark.parametrize("bad_doc_id", ["", "   ", None, 123])
    def test_02_invalid_document_id_raises_error(self, bad_doc_id: Any) -> None:
        """Empty or non-string document_id raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id=bad_doc_id, filename="f.pdf", chunk_id="c1")

    @pytest.mark.parametrize("bad_filename", ["", "   ", None, 123])
    def test_03_invalid_filename_raises_error(self, bad_filename: Any) -> None:
        """Empty or non-string filename raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d1", filename=bad_filename, chunk_id="c1")

    @pytest.mark.parametrize("bad_chunk_id", ["", "   ", None, 123])
    def test_04_invalid_chunk_id_raises_error(self, bad_chunk_id: Any) -> None:
        """Empty or non-string chunk_id raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d1", filename="f.pdf", chunk_id=bad_chunk_id)

    @pytest.mark.parametrize("bad_page", [0, -1, -5, "one", 1.5, True])
    def test_05_invalid_page_number_raises_error(self, bad_page: Any) -> None:
        """Non-positive integer page number raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", page_number=bad_page)

    @pytest.mark.parametrize("bad_type", ["text", "table", "audio", "video", "", "   "])
    def test_06_unsupported_content_type_raises_error(self, bad_type: str) -> None:
        """Unsupported or empty visual content type raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", content_type=bad_type)

    @pytest.mark.parametrize("valid_type", ["image", "chart", "diagram", "IMAGE", "Chart", "DIAGRAM"])
    def test_07_valid_content_types_normalized(self, valid_type: str) -> None:
        """Valid visual content types are accepted and normalized to lower case."""
        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", content_type=valid_type)
        assert ev.content_type == valid_type.strip().lower()

    def test_08_from_citation_factory(self) -> None:
        """from_citation correctly creates VisualEvidence preserving all lineage fields."""
        cit = AgentCitation(
            document_id="doc-cit-01",
            filename="report.pdf",
            chunk_id="chk-cit-99",
            page_number=5,
            content_type="image",
            score=0.92,
            metadata={"source_dpi": 300, "chunk_index": 3},
        )
        ev = VisualEvidence.from_citation(
            citation=cit,
            image_path="/tmp/extracted_img.png",
            image_format="png",
        )
        assert ev.document_id == "doc-cit-01"
        assert ev.filename == "report.pdf"
        assert ev.chunk_id == "chk-cit-99"
        assert ev.page_number == 5
        assert ev.chunk_index == 3
        assert ev.content_type == "image"
        assert ev.image_path == "/tmp/extracted_img.png"
        assert ev.image_format == "png"
        assert ev.metadata == {"source_dpi": 300, "chunk_index": 3}

    def test_09_from_search_result_factory(self) -> None:
        """from_search_result creates VisualEvidence preserving VectorSearchResult fields."""
        res = VectorSearchResult(
            chunk_id="chk-sr-01",
            score=0.88,
            document_id="doc-sr-01",
            filename="diagrams.pdf",
            page_number=2,
            chunk_index=1,
            content_type="diagram",
            content="Circuit diagram schematic",
            metadata={"version": "v1.2"},
        )
        ev = VisualEvidence.from_search_result(res)
        assert ev.document_id == "doc-sr-01"
        assert ev.filename == "diagrams.pdf"
        assert ev.chunk_id == "chk-sr-01"
        assert ev.page_number == 2
        assert ev.content_type == "diagram"
        assert ev.description == "Circuit diagram schematic"
        assert ev.metadata == {"version": "v1.2"}

    def test_10_serialization_roundtrip(self) -> None:
        """VisualEvidence to_dict and from_dict roundtrip accurately."""
        ev = VisualEvidence(
            document_id="d1",
            filename="f1.pdf",
            chunk_id="c1",
            page_number=1,
            chunk_index=0,
            content_type="chart",
            image_path="/path/to/chart.png",
            width=640,
            height=480,
            metadata={"origin": "scanner"},
        )
        d = ev.to_dict()
        restored = VisualEvidence.from_dict(d)
        assert restored.document_id == ev.document_id
        assert restored.filename == ev.filename
        assert restored.chunk_id == ev.chunk_id
        assert restored.page_number == ev.page_number
        assert restored.content_type == ev.content_type
        assert restored.width == ev.width
        assert restored.height == ev.height
        assert restored.metadata == ev.metadata


class TestVisionRequestModel:
    """Test suite for VisionRequest model validation and operations."""

    def test_11_valid_vision_request_creation(self) -> None:
        """VisionRequest constructs with valid query, evidence list, and metadata."""
        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1")
        req = VisionRequest(
            query="Analyze this bar chart",
            evidence=[ev],
            metadata={"caller": "supervisor"},
            session_id="sess-123",
        )
        assert req.query == "Analyze this bar chart"
        assert req.total_evidence == 1
        assert req.has_evidence is True
        assert req.session_id == "sess-123"

    @pytest.mark.parametrize("bad_query", ["", "   ", "\t\n", None, 123, []])
    def test_12_invalid_query_raises_validation_error(self, bad_query: Any) -> None:
        """Empty, whitespace, or non-string query raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query=bad_query)

    def test_13_invalid_evidence_list_raises_validation_error(self) -> None:
        """Non-list or non-VisualEvidence elements in evidence raise VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="valid query", evidence="not_a_list")  # type: ignore[arg-type]

        with pytest.raises(VisionInputValidationError, match="not a VisualEvidence instance"):
            VisionRequest(query="valid query", evidence=["not_an_evidence"])  # type: ignore[list-item]

    def test_14_request_serialization_roundtrip(self) -> None:
        """VisionRequest to_dict and from_dict roundtrip accurately."""
        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1")
        req = VisionRequest(query="Describe image", evidence=[ev], session_id="s1")
        d = req.to_dict()
        restored = VisionRequest.from_dict(d)
        assert restored.query == req.query
        assert restored.total_evidence == 1
        assert restored.evidence[0].chunk_id == "c1"
        assert restored.session_id == "s1"


class TestVisionResultModel:
    """Test suite for VisionResult model and lineage preservation."""

    def test_15_valid_vision_result_creation_and_properties(self) -> None:
        """VisionResult constructs and calculates status properties correctly."""
        ev = VisualEvidence(document_id="doc-res-01", filename="chart.pdf", chunk_id="chk-01", page_number=2)
        res = VisionResult(
            query="Describe chart",
            status="success",
            description="Bar chart showing upward growth.",
            evidence=[ev],
        )
        assert res.is_success is True
        assert res.is_error is False
        assert res.has_evidence is True
        assert res.document_id == "doc-res-01"
        assert res.filename == "chart.pdf"
        assert res.page_number == 2
        assert res.chunk_id == "chk-01"

    def test_16_error_status_properties(self) -> None:
        """VisionResult with error indicates is_error=True and is_success=False."""
        res = VisionResult(query="Failed query", status="error", error="Image corrupt")
        assert res.is_error is True
        assert res.is_success is False

    def test_17_result_serialization_roundtrip(self) -> None:
        """VisionResult to_dict and from_dict roundtrip accurately."""
        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", content_type="diagram")
        res = VisionResult(
            query="Audit query",
            status="success",
            description="Diagram schematic",
            evidence=[ev],
            metadata={"latency_ms": 45.2},
        )
        d = res.to_dict()
        restored = VisionResult.from_dict(d)
        assert restored.query == res.query
        assert restored.status == res.status
        assert restored.description == res.description
        assert restored.document_id == "d1"
        assert restored.content_type == "diagram"
        assert restored.metadata["latency_ms"] == 45.2
