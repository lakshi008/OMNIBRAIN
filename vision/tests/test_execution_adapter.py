"""
Comprehensive unit and integration tests for Day 37: Vision Execution Adapter & Provider Integration.

Tests cover:
  1.  VisionAgent accepts provider via dependency injection.
  2.  VisionExecutionAdapter accepts provider and rejects None / invalid provider types.
  3.  Complete end-to-end execution pipeline from raw request to VisionResult.
  4.  Provider receives a validated Day 35 VisionModelInput.
  5.  Query reaches provider unchanged and stripped.
  6.  Document_id lineage preserved across full pipeline.
  7.  Filename lineage preserved across full pipeline.
  8.  Page_number lineage preserved across full pipeline.
  9.  Chunk_id lineage preserved across full pipeline.
  10. Chunk_index lineage preserved across full pipeline.
  11. Content_type (modality) lineage preserved across full pipeline.
  12. Metadata preserved across full pipeline.
  13. Provider result returned correctly to VisionAgent caller.
  14. Provider is called exactly once per request.
  15. Invalid request (None, empty, non-string/non-VisionRequest) rejected with VisionInputValidationError.
  16. Invalid evidence (non-visual modality) rejected with VisionEvidenceError.
  17. Provider failure (VisionProviderExecutionError) propagated cleanly without swallowing.
  18. No fake or fabricated VisionResult returned on provider failure.
  19. Unsupported modality rejected via provider capabilities check.
  20. Multiple evidence items handled gracefully with primary evidence selected.
  21. Empty evidence request handled cleanly returning status='no_evidence'.
  22. Deterministic execution: identical inputs produce identical VisionModelInput to provider.
  23. Raw Member 2 AgentCitation adapted via Day 33 adapter and executed.
  24. Raw Member 1 VectorSearchResult adapted via Day 33 adapter and executed.
  25. Raw Member 1 DocumentChunk adapted via Day 33 adapter and executed.
  26. Raw dictionary evidence adapted via Day 33 adapter and executed.
  27. Module-level execute_vision_request convenience function.
  28. VisionAgent execute(), analyze(), process(), and __call__() aliases all function identically.
  29. VisionAgent without provider retains backward-compatible Day 32 VisionProcessingError.
  30. No network imports or external API calls in execution adapter code.
"""

from __future__ import annotations

import inspect
import io
from typing import Any

import pytest
from PIL import Image

from agents.models import AgentCitation
from ingestion.models import DocumentChunk, VectorSearchResult
from vision.exceptions import (
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderExecutionError,
    VisionUnsupportedCapabilityError,
)
from vision.execution_adapter import (
    VisionExecutionAdapter,
    execute_vision_request,
)
from vision.input_builder import VisionModelInput
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.provider import VisionModelProvider
from vision.provider_config import (
    VisionProviderCapabilities,
    VisionProviderConfig,
)
from vision.vision_agent import VisionAgent


# ===========================================================================
# Test Helpers & Test Doubles (Conformed strictly to test scope only)
# ===========================================================================


def _make_test_png(width: int = 64, height: int = 64) -> bytes:
    """Generate minimal PNG bytes for test evidence."""
    img = Image.new("RGB", (width, height), color=(120, 180, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class RecordingTestProvider(VisionModelProvider):
    """Test double that records received VisionModelInputs and invocation counts."""

    def __init__(
        self,
        config: VisionProviderConfig,
        capabilities: VisionProviderCapabilities | None = None,
        return_description: str = "Test visual analysis result.",
    ) -> None:
        super().__init__(config, capabilities)
        self.recorded_inputs: list[VisionModelInput] = []
        self.call_count: int = 0
        self.return_description = return_description

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        self.call_count += 1
        self.recorded_inputs.append(model_input)
        return VisionResult(
            query=model_input.query,
            status="success",
            description=self.return_description,
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={"call_count": self.call_count, **model_input.evidence_metadata},
        )


class FailingExecutionProvider(VisionModelProvider):
    """Test double that raises a controlled execution error."""

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        raise VisionProviderExecutionError("Backend model execution failed unexpectedly.")


# ===========================================================================
# 1. Test class: VisionExecutionAdapter Initialization & Basic Validation
# ===========================================================================


class TestVisionExecutionAdapterInitialization:
    """Tests 1-2: Adapter initialization, provider validation, and rejection of invalid types."""

    def test_01_valid_initialization(self) -> None:
        """VisionExecutionAdapter initializes with a valid VisionModelProvider."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        assert adapter.provider is provider

    @pytest.mark.parametrize("bad_provider", [None, "invalid", {"provider": 1}, 123])
    def test_02_invalid_provider_raises_error(self, bad_provider: Any) -> None:
        """Passing None or non-VisionModelProvider raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="provider"):
            VisionExecutionAdapter(provider=bad_provider)  # type: ignore[arg-type]


# ===========================================================================
# 2. Test class: End-to-End Execution Pipeline & Lineage Preservation
# ===========================================================================


class TestExecutionPipelineAndLineage:
    """Tests 3-14: Multi-stage pipeline execution, lineage preservation, and provider receipt."""

    def test_03_complete_pipeline_execution(self) -> None:
        """VisionExecutionAdapter executes the full pipeline returning a structured VisionResult."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config, return_description="Chart indicates upward trend.")
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(
            document_id="doc-pipeline-99",
            filename="quarterly_report.pdf",
            chunk_id="chk-img-77",
            page_number=5,
            chunk_index=2,
            content_type="chart",
            image_bytes=_make_test_png(64, 48),
            metadata={"chart_type": "line"},
        )
        result = adapter.execute("What is the quarterly trend?", evidence=[ev])

        assert isinstance(result, VisionResult)
        assert result.is_success is True
        assert result.description == "Chart indicates upward trend."
        assert result.query == "What is the quarterly trend?"
        assert result.document_id == "doc-pipeline-99"
        assert result.filename == "quarterly_report.pdf"
        assert result.page_number == 5
        assert result.chunk_id == "chk-img-77"
        assert result.content_type == "chart"

    def test_04_provider_receives_exact_vision_model_input(self) -> None:
        """Provider receives a validated Day 35 VisionModelInput with exact lineage."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(
            document_id="doc-rec-01",
            filename="stats.pdf",
            chunk_id="chk-rec-01",
            page_number=12,
            chunk_index=3,
            content_type="diagram",
            image_bytes=_make_test_png(32, 32),
            metadata={"system": "cooling"},
        )
        adapter.execute("  Explain the diagram.  ", evidence=[ev])

        assert provider.call_count == 1
        recorded = provider.recorded_inputs[0]

        assert isinstance(recorded, VisionModelInput)
        assert recorded.query == "Explain the diagram."  # stripped
        assert recorded.document_id == "doc-rec-01"
        assert recorded.filename == "stats.pdf"
        assert recorded.page_number == 12
        assert recorded.chunk_id == "chk-rec-01"
        assert recorded.chunk_index == 3
        assert recorded.content_type == "diagram"
        assert recorded.image_format == "png"
        assert recorded.width == 32
        assert recorded.height == 32
        assert recorded.evidence_metadata == {"system": "cooling"}

    def test_05_provider_called_exactly_once_per_execute(self) -> None:
        """Provider is invoked exactly once per execution call."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        assert provider.call_count == 0
        adapter.execute("Query 1", evidence=[ev])
        assert provider.call_count == 1
        adapter.execute("Query 2", evidence=[ev])
        assert provider.call_count == 2


# ===========================================================================
# 3. Test class: Input Validation and Request Normalization
# ===========================================================================


class TestInputValidationAndNormalization:
    """Tests 15-18: Invalid requests, invalid evidence, and error conditions."""

    @pytest.mark.parametrize("bad_query", [None, "", "   ", 123, []])
    def test_06_invalid_request_rejected(self, bad_query: Any) -> None:
        """Invalid query string or request raises VisionInputValidationError."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        with pytest.raises(VisionInputValidationError):
            adapter.execute(bad_query)

    def test_07_invalid_evidence_type_raises_error(self) -> None:
        """Passing non-visual evidence raises VisionEvidenceError."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        # String is not a supported evidence type
        with pytest.raises(VisionEvidenceError):
            adapter.execute("Query", evidence=["not-visual-evidence"])

    def test_08_empty_evidence_returns_no_evidence_status(self) -> None:
        """Request with no visual evidence returns status='no_evidence' without calling provider."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        result = adapter.execute("Query with no images", evidence=[])

        assert isinstance(result, VisionResult)
        assert result.status == "no_evidence"
        assert result.has_evidence is False
        assert provider.call_count == 0

    def test_09_multiple_evidence_items_selects_primary(self) -> None:
        """Multiple visual evidence items are accepted with the primary item prepared."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev1 = VisualEvidence(document_id="d1", filename="f1.pdf", chunk_id="c1", image_bytes=_make_test_png(16, 16))
        ev2 = VisualEvidence(document_id="d2", filename="f2.pdf", chunk_id="c2", image_bytes=_make_test_png(32, 32))

        result = adapter.execute("Analyze primary image", evidence=[ev1, ev2])

        assert result.is_success is True
        assert provider.recorded_inputs[0].chunk_id == "c1"


# ===========================================================================
# 4. Test class: Evidence Adaptation from Member 1 & Member 2 Types
# ===========================================================================


class TestEvidenceAdaptationInPipeline:
    """Tests 23-26: Day 33 VisualEvidenceAdapter integration inside execution adapter."""

    def test_10_adapt_agent_citation_in_pipeline(self) -> None:
        """Member 2 AgentCitation is automatically adapted and executed through pipeline."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        citation = AgentCitation(
            document_id="doc-cit-01",
            filename="report.pdf",
            chunk_id="chk-cit-01",
            page_number=3,
            content_type="chart",
            metadata={"image_bytes": _make_test_png()},
        )
        result = adapter.execute("Explain chart citation", evidence=[citation])

        assert result.is_success is True
        assert result.document_id == "doc-cit-01"
        assert result.chunk_id == "chk-cit-01"
        assert result.content_type == "chart"

    def test_11_adapt_vector_search_result_in_pipeline(self) -> None:
        """Member 1 VectorSearchResult is automatically adapted and executed through pipeline."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        vsr = VectorSearchResult(
            chunk_id="chk-vsr-01",
            score=0.92,
            document_id="doc-vsr-01",
            filename="diagrams.pdf",
            page_number=7,
            chunk_index=1,
            content_type="diagram",
            content="Architecture diagram",
            metadata={"image_bytes": _make_test_png()},
        )
        result = adapter.execute("Explain architecture", evidence=[vsr])

        assert result.is_success is True
        assert result.document_id == "doc-vsr-01"
        assert result.chunk_id == "chk-vsr-01"

    def test_12_adapt_document_chunk_in_pipeline(self) -> None:
        """Member 1 DocumentChunk is automatically adapted and executed through pipeline."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        chunk = DocumentChunk(
            chunk_id="chk-doc-01",
            chunk_index=0,
            document_id="doc-chunk-01",
            filename="flow.pdf",
            page_number=2,
            content="Flow diagram",
            content_type="diagram",
            metadata={"image_bytes": _make_test_png()},
        )
        result = adapter.execute("Explain flow", evidence=[chunk])

        assert result.is_success is True
        assert result.document_id == "doc-chunk-01"

    def test_13_adapt_dict_evidence_in_pipeline(self) -> None:
        """Raw dictionary evidence is automatically adapted and executed through pipeline."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        dict_ev = {
            "document_id": "doc-dict-01",
            "filename": "chart.pdf",
            "chunk_id": "chk-dict-01",
            "page_number": 1,
            "chunk_index": 0,
            "content_type": "chart",
            "image_bytes": _make_test_png(),
        }
        result = adapter.execute("Explain chart dict", evidence=[dict_ev])

        assert result.is_success is True
        assert result.document_id == "doc-dict-01"


# ===========================================================================
# 5. Test class: Error Propagation & Failure Isolation
# ===========================================================================


class TestErrorPropagationAndFailureIsolation:
    """Tests 17-21: Controlled error propagation without swallowing or fake results."""

    def test_14_provider_execution_failure_propagates(self) -> None:
        """When provider fails, VisionProviderExecutionError is raised without fake output."""
        config = VisionProviderConfig(provider_name="fail-prov", model_name="vlm-1")
        provider = FailingExecutionProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        with pytest.raises(VisionProviderExecutionError, match="Backend model execution failed"):
            adapter.execute("Query", evidence=[ev])

    def test_15_unsupported_modality_raises_capability_error(self) -> None:
        """When evidence modality is not supported by provider, capability error is raised."""
        config = VisionProviderConfig(provider_name="chart-only", model_name="vlm-1")
        # Provider only supports chart
        caps = VisionProviderCapabilities(supported_modalities=frozenset({"chart"}))
        provider = RecordingTestProvider(config, capabilities=caps)
        adapter = VisionExecutionAdapter(provider=provider)

        diagram_ev = VisualEvidence(
            document_id="d1", filename="f.pdf", chunk_id="c1",
            content_type="diagram", image_bytes=_make_test_png(),
        )

        with pytest.raises(VisionUnsupportedCapabilityError, match="modality"):
            adapter.execute("Explain diagram", evidence=[diagram_ev])


# ===========================================================================
# 6. Test class: VisionAgent Integration
# ===========================================================================


class TestVisionAgentIntegration:
    """Tests 28-29: VisionAgent execute(), analyze(), process(), __call__() with provider."""

    def test_16_vision_agent_dependency_injection(self) -> None:
        """VisionAgent receives provider via dependency injection and executes successfully."""
        config = VisionProviderConfig(provider_name="agent-prov", model_name="vlm-v1")
        provider = RecordingTestProvider(config, return_description="Agent analysis complete.")
        agent = VisionAgent(agent_name="TestAgent", provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        # Test execute()
        res_exec = agent.execute("Analyze this chart", evidence=[ev])
        assert res_exec.is_success is True
        assert res_exec.description == "Agent analysis complete."

        # Test analyze()
        res_ana = agent.analyze("Analyze this chart", evidence=[ev])
        assert res_ana.is_success is True

        # Test process()
        res_proc = agent.process("Analyze this chart", evidence=[ev])
        assert res_proc.is_success is True

        # Test __call__()
        res_call = agent("Analyze this chart", evidence=[ev])
        assert res_call.is_success is True

    def test_17_vision_agent_without_provider_raises_processing_error(self) -> None:
        """VisionAgent initialized without provider continues raising Day 32 VisionProcessingError."""
        agent = VisionAgent(model_name="unconfigured-model")
        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        with pytest.raises(VisionProcessingError, match="not implemented for Day 32 foundation"):
            agent.execute("Analyze", evidence=[ev])


# ===========================================================================
# 7. Test class: Convenience Function & Determinism
# ===========================================================================


class TestConvenienceFunctionAndDeterminism:
    """Tests 27, 22: execute_vision_request module-level function and deterministic execution."""

    def test_18_convenience_function_execution(self) -> None:
        """execute_vision_request executes full pipeline correctly."""
        config = VisionProviderConfig(provider_name="conv-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config, return_description="Convenience function result.")

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())
        result = execute_vision_request(provider=provider, request="Run query", evidence=[ev])

        assert isinstance(result, VisionResult)
        assert result.description == "Convenience function result."

    def test_19_deterministic_provider_input_generation(self) -> None:
        """Executing with identical inputs twice produces identical VisionModelInputs."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-1")
        provider = RecordingTestProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(
            document_id="doc-det-01",
            filename="det.pdf",
            chunk_id="chk-det-01",
            page_number=1,
            chunk_index=0,
            content_type="image",
            image_bytes=_make_test_png(),
            metadata={"seed": 42},
        )

        adapter.execute("Deterministic query", evidence=[ev])
        adapter.execute("Deterministic query", evidence=[ev])

        assert len(provider.recorded_inputs) == 2
        assert provider.recorded_inputs[0].to_dict() == provider.recorded_inputs[1].to_dict()


# ===========================================================================
# 8. Test class: Offline and Security Verification
# ===========================================================================


class TestOfflineAndSecurityVerification:
    """Tests 30: Zero network calls, no HTTP libraries, no credentials in execution adapter."""

    def test_20_no_network_libraries_in_execution_adapter(self) -> None:
        """execution_adapter.py imports no socket, HTTP, or vendor SDK libraries."""
        from vision import execution_adapter

        source = inspect.getsource(execution_adapter)
        forbidden = [
            "import requests",
            "import httpx",
            "import aiohttp",
            "import socket",
            "import urllib.request",
            "import openai",
            "import anthropic",
            "import google.generativeai",
            "import langchain",
            "import langgraph",
            "import fastapi",
            "import streamlit",
        ]
        for pattern in forbidden:
            assert pattern not in source, f"execution_adapter.py contains forbidden pattern '{pattern}'"
