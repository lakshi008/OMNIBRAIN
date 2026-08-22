"""
Unit and integration tests for Member 3 VisionAgent interface.
"""

from __future__ import annotations

from typing import Any
import pytest

from vision.exceptions import (
    VisionAgentError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
)
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.vision_agent import VisionAgent


class TestVisionAgentInterface:
    """Test suite for VisionAgent initialization, input validation, and execution contracts."""

    def test_01_valid_initialization(self) -> None:
        """VisionAgent initializes with default and custom arguments."""
        agent1 = VisionAgent()
        assert agent1.agent_name == "VisionAgent"
        assert agent1.model_name == "default-vision-model"

        agent2 = VisionAgent(agent_name="CustomVision", model_name="gemini-flash-vision", metadata={"mode": "ocr"})
        assert agent2.agent_name == "CustomVision"
        assert agent2.model_name == "gemini-flash-vision"
        assert agent2.metadata == {"mode": "ocr"}

    @pytest.mark.parametrize("bad_name", ["", "   ", None, 123])
    def test_02_invalid_agent_name_raises_error(self, bad_name: Any) -> None:
        """Empty or non-string agent_name raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="agent_name must be a non-empty string"):
            VisionAgent(agent_name=bad_name)

    @pytest.mark.parametrize("bad_model", ["", "   ", None, 123])
    def test_03_invalid_model_name_raises_error(self, bad_model: Any) -> None:
        """Empty or non-string model_name raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="model_name must be a non-empty string"):
            VisionAgent(model_name=bad_model)

    def test_04_prepare_request_from_string(self) -> None:
        """_prepare_request converts string query and evidence list into VisionRequest."""
        agent = VisionAgent()
        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1")
        req = agent._prepare_request("Analyze this chart", evidence=[ev])

        assert isinstance(req, VisionRequest)
        assert req.query == "Analyze this chart"
        assert req.total_evidence == 1
        assert req.evidence[0].chunk_id == "c1"

    def test_05_prepare_request_from_vision_request(self) -> None:
        """_prepare_request accepts existing VisionRequest and optionally merges evidence."""
        agent = VisionAgent()
        ev1 = VisualEvidence(document_id="d1", filename="f1.pdf", chunk_id="c1")
        ev2 = VisualEvidence(document_id="d2", filename="f2.pdf", chunk_id="c2")

        orig_req = VisionRequest(query="Original query", evidence=[ev1])
        req = agent._prepare_request(orig_req, evidence=[ev2])

        assert req.query == "Original query"
        assert req.total_evidence == 2
        assert [e.chunk_id for e in req.evidence] == ["c1", "c2"]

    @pytest.mark.parametrize("bad_req", [None, "", "   ", 123, []])
    def test_06_prepare_request_invalid_inputs(self, bad_req: Any) -> None:
        """Invalid request inputs raise VisionInputValidationError."""
        agent = VisionAgent()
        with pytest.raises(VisionInputValidationError):
            agent._prepare_request(bad_req)

    def test_07_controlled_not_implemented_inference(self) -> None:
        """Day 32 analyze() raises VisionProcessingError signaling inference is not yet implemented."""
        agent = VisionAgent(model_name="vision-core-v1")
        ev = VisualEvidence(document_id="d1", filename="chart.pdf", chunk_id="c1")

        with pytest.raises(VisionProcessingError, match="not implemented for Day 32 foundation"):
            agent.analyze("Describe this diagram", evidence=[ev])

    def test_08_process_and_call_aliases(self) -> None:
        """process() and __call__() route to analyze() and raise controlled VisionProcessingError."""
        agent = VisionAgent()
        ev = VisualEvidence(document_id="d1", filename="chart.pdf", chunk_id="c1")

        with pytest.raises(VisionProcessingError):
            agent.process("Query", evidence=[ev])

        with pytest.raises(VisionProcessingError):
            agent("Query", evidence=[ev])

    def test_09_no_fake_production_results(self) -> None:
        """VisionAgent never returns fabricated text descriptions like 'sample image'."""
        agent = VisionAgent()
        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1")

        # Must not silently return fake mock descriptions in production analyze()
        with pytest.raises(VisionProcessingError):
            agent.analyze("Analyze chart", evidence=[ev])

    def test_10_lineage_preservation_in_request(self) -> None:
        """Lineage fields (document_id, filename, chunk_id, page_number) are strictly maintained."""
        ev = VisualEvidence(
            document_id="doc-lineage-123",
            filename="financials.pdf",
            chunk_id="chk-img-456",
            page_number=10,
            content_type="chart",
        )
        req = VisionRequest(query="Lineage check", evidence=[ev])
        assert req.evidence[0].document_id == "doc-lineage-123"
        assert req.evidence[0].filename == "financials.pdf"
        assert req.evidence[0].chunk_id == "chk-img-456"
        assert req.evidence[0].page_number == 10
        assert req.evidence[0].content_type == "chart"
