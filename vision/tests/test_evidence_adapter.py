"""
Unit and integration tests for Member 3 VisualEvidenceAdapter.
"""

from __future__ import annotations

from typing import Any
import pytest

from agents.models import AgentCitation, AgentResponse, SearchResult
from ingestion.models import DocumentChunk, VectorSearchResult
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import VisionEvidenceError, VisionInputValidationError
from vision.models import VisualEvidence


class TestVisualEvidenceAdapter:
    """Test suite for VisualEvidenceAdapter validation, conversion, and lineage preservation."""

    # ------------------------------------------------------------------
    # 1. Valid visual evidence adaptation
    # ------------------------------------------------------------------

    def test_01_valid_image_citation_adaptation(self) -> None:
        """AgentCitation with content_type='image' adapts cleanly to VisualEvidence."""
        cit = AgentCitation(
            document_id="doc-img-01",
            filename="quarterly_report.pdf",
            chunk_id="chk-img-101",
            page_number=4,
            content_type="image",
            score=0.94,
            metadata={"resolution": "300dpi", "chunk_index": 2},
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_path="/images/p4_img.png")

        assert isinstance(ev, VisualEvidence)
        assert ev.document_id == "doc-img-01"
        assert ev.filename == "quarterly_report.pdf"
        assert ev.chunk_id == "chk-img-101"
        assert ev.page_number == 4
        assert ev.chunk_index == 2
        assert ev.content_type == "image"
        assert ev.image_path == "/images/p4_img.png"
        assert ev.metadata["resolution"] == "300dpi"

    def test_02_valid_chart_citation_adaptation(self) -> None:
        """AgentCitation with content_type='chart' adapts cleanly."""
        cit = AgentCitation(
            document_id="doc-chart-01",
            filename="sales_deck.pdf",
            chunk_id="chk-chart-202",
            page_number=8,
            content_type="chart",
            score=0.88,
            metadata={"chart_type": "bar"},
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit)
        assert ev.content_type == "chart"
        assert ev.document_id == "doc-chart-01"
        assert ev.metadata["chart_type"] == "bar"

    def test_03_valid_diagram_citation_adaptation(self) -> None:
        """AgentCitation with content_type='diagram' adapts cleanly."""
        cit = AgentCitation(
            document_id="doc-diag-01",
            filename="system_arch.pdf",
            chunk_id="chk-diag-303",
            page_number=12,
            content_type="diagram",
            score=0.91,
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit)
        assert ev.content_type == "diagram"
        assert ev.chunk_id == "chk-diag-303"

    # ------------------------------------------------------------------
    # 2. Member 1 VectorSearchResult and Chunk adaptation
    # ------------------------------------------------------------------

    def test_04_valid_vector_search_result_adaptation(self) -> None:
        """VectorSearchResult adapts into VisualEvidence preserving content and lineage."""
        sr = VectorSearchResult(
            chunk_id="chk-vsr-01",
            score=0.89,
            document_id="doc-vsr-01",
            filename="product_manual.pdf",
            page_number=15,
            chunk_index=3,
            content_type="image",
            content="Schematic diagram of wiring assembly.",
            metadata={"camera": "optical"},
        )
        ev = VisualEvidenceAdapter.adapt_search_result(sr)

        assert ev.document_id == "doc-vsr-01"
        assert ev.filename == "product_manual.pdf"
        assert ev.chunk_id == "chk-vsr-01"
        assert ev.page_number == 15
        assert ev.chunk_index == 3
        assert ev.content_type == "image"
        assert ev.description == "Schematic diagram of wiring assembly."
        assert ev.metadata["camera"] == "optical"

    def test_05_valid_chunk_adaptation(self) -> None:
        """Member 1 DocumentChunk adapts into VisualEvidence accurately."""
        chk = DocumentChunk(
            chunk_id="chk-raw-01",
            document_id="doc-raw-01",
            filename="spec.pdf",
            page_number=1,
            chunk_index=0,
            content_type="chart",
            content="Plot of temperature vs time",
            metadata={"axis_x": "time", "axis_y": "temp"},
        )
        ev = VisualEvidenceAdapter.adapt_chunk(chk)

        assert ev.document_id == "doc-raw-01"
        assert ev.filename == "spec.pdf"
        assert ev.chunk_id == "chk-raw-01"
        assert ev.content_type == "chart"
        assert ev.description == "Plot of temperature vs time"
        assert ev.metadata == {"axis_x": "time", "axis_y": "temp"}

    # ------------------------------------------------------------------
    # 3. Lineage preservation verification
    # ------------------------------------------------------------------

    def test_06_lineage_preservation_strictness(self) -> None:
        """All 7 provenance attributes remain exact and untransformed."""
        cit = AgentCitation(
            document_id="doc-exact-lineage",
            filename="critical_blueprint.pdf",
            chunk_id="chk-exact-999",
            page_number=42,
            content_type="diagram",
            score=0.995,
            metadata={"author": "Engineering Lead", "chunk_index": 7},
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit)

        assert ev.document_id == cit.document_id
        assert ev.filename == cit.filename
        assert ev.chunk_id == cit.chunk_id
        assert ev.page_number == cit.page_number
        assert ev.chunk_index == 7
        assert ev.content_type == cit.content_type
        assert ev.metadata == cit.metadata

    # ------------------------------------------------------------------
    # 4. Modality rejection rules
    # ------------------------------------------------------------------

    def test_07_unsupported_text_modality_rejected(self) -> None:
        """Text citations raise VisionEvidenceError."""
        cit = AgentCitation(
            document_id="d1",
            filename="f.pdf",
            chunk_id="c1",
            content_type="text",
        )
        with pytest.raises(VisionEvidenceError, match="Unsupported content_type 'text'"):
            VisualEvidenceAdapter.adapt_citation(cit)

    def test_08_unsupported_table_modality_rejected(self) -> None:
        """Table citations raise VisionEvidenceError."""
        cit = AgentCitation(
            document_id="d1",
            filename="f.pdf",
            chunk_id="c1",
            content_type="table",
        )
        with pytest.raises(VisionEvidenceError, match="Unsupported content_type 'table'"):
            VisualEvidenceAdapter.adapt_citation(cit)

    @pytest.mark.parametrize("bad_modality", ["audio", "video", "json", "unknown", ""])
    def test_09_arbitrary_unsupported_modality_rejected(self, bad_modality: str) -> None:
        """Non-visual modality strings are rejected."""
        assert VisualEvidenceAdapter.is_visual_content_type(bad_modality) is False

    # ------------------------------------------------------------------
    # 5. Invalid inputs and validation errors
    # ------------------------------------------------------------------

    def test_10_none_citation_raises_validation_error(self) -> None:
        """None passed to adapt_citation raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="Citation cannot be None"):
            VisualEvidenceAdapter.adapt_citation(None)  # type: ignore[arg-type]

    def test_11_invalid_type_to_adapt_citation(self) -> None:
        """Non-AgentCitation passed to adapt_citation raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="Expected AgentCitation"):
            VisualEvidenceAdapter.adapt_citation("not_a_citation")  # type: ignore[arg-type]

    def test_12_none_search_result_raises_validation_error(self) -> None:
        """None passed to adapt_search_result raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="VectorSearchResult cannot be None"):
            VisualEvidenceAdapter.adapt_search_result(None)  # type: ignore[arg-type]

    def test_13_none_chunk_raises_validation_error(self) -> None:
        """None passed to adapt_chunk raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="Chunk cannot be None"):
            VisualEvidenceAdapter.adapt_chunk(None)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # 6. Universal adapt and is_visual methods
    # ------------------------------------------------------------------

    def test_14_universal_adapt_various_types(self) -> None:
        """Universal adapt() handles VisualEvidence, AgentCitation, VectorSearchResult, Chunk, and dict."""
        # 1. VisualEvidence (pass-through)
        ve_orig = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", content_type="image")
        assert VisualEvidenceAdapter.adapt(ve_orig) is ve_orig

        # 2. AgentCitation
        cit = AgentCitation(document_id="d2", filename="f.pdf", chunk_id="c2", content_type="chart")
        ev2 = VisualEvidenceAdapter.adapt(cit)
        assert ev2.chunk_id == "c2"

        # 3. VectorSearchResult
        vsr = VectorSearchResult(
            chunk_id="c3",
            score=0.8,
            document_id="d3",
            filename="f.pdf",
            page_number=1,
            chunk_index=0,
            content_type="diagram",
            content="diagram info",
        )
        ev3 = VisualEvidenceAdapter.adapt(vsr)
        assert ev3.chunk_id == "c3"

        # 4. Dict
        d = {"document_id": "d4", "filename": "f.pdf", "chunk_id": "c4", "content_type": "image"}
        ev4 = VisualEvidenceAdapter.adapt(d)
        assert ev4.chunk_id == "c4"

    def test_15_is_visual_detection(self) -> None:
        """is_visual correctly identifies visual and non-visual items."""
        cit_img = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", content_type="image")
        cit_txt = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", content_type="text")
        vsr_chart = VectorSearchResult(
            chunk_id="c",
            score=0.9,
            document_id="d",
            filename="f.pdf",
            page_number=1,
            chunk_index=0,
            content_type="chart",
            content="chart info",
        )
        vsr_tbl = VectorSearchResult(
            chunk_id="c",
            score=0.9,
            document_id="d",
            filename="f.pdf",
            page_number=1,
            chunk_index=0,
            content_type="table",
            content="table info",
        )

        assert VisualEvidenceAdapter.is_visual(cit_img) is True
        assert VisualEvidenceAdapter.is_visual(cit_txt) is False
        assert VisualEvidenceAdapter.is_visual(vsr_chart) is True
        assert VisualEvidenceAdapter.is_visual(vsr_tbl) is False
        assert VisualEvidenceAdapter.is_visual(None) is False
        assert VisualEvidenceAdapter.is_visual(12345) is False

    # ------------------------------------------------------------------
    # 7. Search package adaptation (SearchResult / AgentResponse)
    # ------------------------------------------------------------------

    def test_16_adapt_search_package_non_strict(self) -> None:
        """Non-strict package adaptation filters out text and table citations, returning only visual ones."""
        cits = [
            AgentCitation(document_id="d1", filename="f.pdf", chunk_id="txt-1", content_type="text"),
            AgentCitation(document_id="d1", filename="f.pdf", chunk_id="img-1", content_type="image"),
            AgentCitation(document_id="d1", filename="f.pdf", chunk_id="tbl-1", content_type="table"),
            AgentCitation(document_id="d1", filename="f.pdf", chunk_id="chart-1", content_type="chart"),
        ]
        pkg = SearchResult(query="Test multimodal", citations=cits)
        visual_list = VisualEvidenceAdapter.adapt_search_package(pkg, strict=False)

        assert len(visual_list) == 2
        assert [e.chunk_id for e in visual_list] == ["img-1", "chart-1"]

    def test_17_adapt_search_package_strict_mode(self) -> None:
        """Strict package adaptation raises VisionEvidenceError if non-visual citations are present."""
        cits = [
            AgentCitation(document_id="d1", filename="f.pdf", chunk_id="txt-1", content_type="text"),
            AgentCitation(document_id="d1", filename="f.pdf", chunk_id="img-1", content_type="image"),
        ]
        pkg = SearchResult(query="Test", citations=cits)

        with pytest.raises(VisionEvidenceError, match="has non-visual content_type 'text' in strict adaptation mode"):
            VisualEvidenceAdapter.adapt_search_package(pkg, strict=True)

    def test_18_adapt_agent_response_package(self) -> None:
        """AgentResponse citations adapt cleanly."""
        cits = [AgentCitation(document_id="d1", filename="f.pdf", chunk_id="img-1", content_type="image")]
        resp = AgentResponse(answer="", agent_name="SearchAgent", citations=cits)
        visual_list = VisualEvidenceAdapter.adapt_search_package(resp)

        assert len(visual_list) == 1
        assert visual_list[0].chunk_id == "img-1"

    # ------------------------------------------------------------------
    # 8. Batch adaptation
    # ------------------------------------------------------------------

    def test_19_adapt_batch_filtering_and_order(self) -> None:
        """adapt_batch processes multiple items preserving rank order and filtering."""
        items = [
            AgentCitation(document_id="d1", filename="f.pdf", chunk_id="img-A", content_type="image"),
            AgentCitation(document_id="d1", filename="f.pdf", chunk_id="txt-B", content_type="text"),
            VectorSearchResult(
                chunk_id="diag-C",
                score=0.8,
                document_id="d1",
                filename="f.pdf",
                page_number=1,
                chunk_index=0,
                content_type="diagram",
                content="diagram info",
            ),
        ]
        res = VisualEvidenceAdapter.adapt_batch(items, strict=False)

        assert len(res) == 2
        assert [e.chunk_id for e in res] == ["img-A", "diag-C"]

    # ------------------------------------------------------------------
    # 9. Determinism and repeated execution
    # ------------------------------------------------------------------

    def test_20_deterministic_adaptation(self) -> None:
        """Repeated adaptation of same citation produces equivalent VisualEvidence dictionaries."""
        cit = AgentCitation(
            document_id="doc-det-01",
            filename="graph.pdf",
            chunk_id="chk-det-01",
            page_number=3,
            content_type="chart",
            score=0.91,
            metadata={"series": [1, 2, 3]},
        )
        ev1 = VisualEvidenceAdapter.adapt_citation(cit)
        ev2 = VisualEvidenceAdapter.adapt_citation(cit)

        assert ev1.to_dict() == ev2.to_dict()

    # ------------------------------------------------------------------
    # 10. No fake metadata or ID synthesis
    # ------------------------------------------------------------------

    def test_21_no_fabricated_lineage(self) -> None:
        """Adapter never synthesizes IDs, filenames, or metadata not in source."""
        cit = AgentCitation(
            document_id="doc-real-id",
            filename="real_file.pdf",
            chunk_id="chunk-real-id",
            page_number=None,
            content_type="image",
            metadata={},
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit)

        assert ev.document_id == "doc-real-id"
        assert ev.filename == "real_file.pdf"
        assert ev.chunk_id == "chunk-real-id"
        assert ev.page_number is None
        assert ev.metadata == {}
