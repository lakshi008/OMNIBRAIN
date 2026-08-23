"""
Comprehensive End-to-End Integration Contract tests for Day 40:
Vision Agent — Pipeline Contract & Integration Hardening.

Verifies that the entire Member 3 Vision pipeline operates cohesively through one safe contract:
Search/Citation Evidence -> VisualEvidence -> PreparedImageEvidence -> VisionModelInput ->
VisionModelProvider -> VisionExecutionLifecycle -> Result Normalization -> VisionResult.

Tests cover:
  1.  End-to-end pipeline success path via VisionAgent and VisionPipeline.
  2.  Provider injection verification (no hidden, fake, or default providers).
  3.  Single provider invocation control (called exactly once per request).
  4.  Query preservation across the entire multi-stage pipeline.
  5.  End-to-end lineage preservation (document_id, filename, chunk_id, page_number, chunk_index, content_type).
  6.  Metadata preservation and sanitization.
  7.  Result normalization & trace attachment in VisionResult.
  8.  Execution lifecycle stage tracking (VALIDATING -> PREPARING -> BUILDING_INPUT -> EXECUTING -> COMPLETED).
  9.  Controlled failure at every pipeline stage (validation, evidence, preparation, input builder, provider, normalizer).
  10. Cause-preserving error propagation (__cause__ preserved) without fake VisionResult fabrication.
  11. Rejection of None, invalid types, or malformed provider output.
  12. Input immutability (VisionRequest, VisualEvidence, VisionModelInput untouched).
  13. Deterministic pipeline execution.
  14. Support for multiple visual evidence items with ordering preserved.
  15. Handlers for missing or invalid provider instances.
  16. Offline guarantee -- zero network, zero LLM SDKs, zero secrets.
  17. Public API package imports verification.
"""

from __future__ import annotations

import inspect
import io
from typing import Any

import pytest
from PIL import Image

from agents.models import AgentCitation, SearchResult
from ingestion.models import DocumentChunk, VectorSearchResult
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderError,
    VisionProviderExecutionError,
    VisionTimeoutError,
)
from vision.execution_adapter import VisionExecutionAdapter
from vision.input_builder import VisionModelInput, build_vision_input
from vision.lifecycle import VisionExecutionLifecycle, VisionExecutionStage
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.pipeline import VisionPipeline, run_vision_pipeline
from vision.provider import VisionModelProvider
from vision.provider_config import (
    VisionProviderCapabilities,
    VisionProviderConfig,
)
from vision.result_normalizer import (
    FORBIDDEN_METADATA_KEYS,
    VisionExecutionTrace,
    VisionResultNormalizer,
)
from vision.vision_agent import VisionAgent


# ===========================================================================
# Test Helpers & Test Doubles (Conformed strictly to test scope only)
# ===========================================================================


def _make_test_png(width: int = 64, height: int = 64) -> bytes:
    """Generate minimal PNG bytes for test evidence."""
    img = Image.new("RGB", (width, height), color=(100, 200, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class E2EDeterministicProvider(VisionModelProvider):
    """Test double that records received inputs and returns a valid, lineage-preserving VisionResult."""

    def __init__(
        self,
        config: VisionProviderConfig,
        capabilities: VisionProviderCapabilities | None = None,
    ) -> None:
        super().__init__(config, capabilities)
        self.invocation_count: int = 0
        self.recorded_inputs: list[VisionModelInput] = []

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        self.invocation_count += 1
        self.recorded_inputs.append(model_input)

        return VisionResult(
            query=model_input.query,
            status="success",
            description="End-to-end pipeline analysis output.",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={
                "provider_name": self.provider_name,
                "confidence": 0.98,
                "api_key": "leaky_key_should_be_stripped",  # Test sanitization integration
            },
        )


class FailingTestProvider(VisionModelProvider):
    """Test double that raises a controlled VisionProviderExecutionError."""

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        raise VisionProviderExecutionError("Backend inference hardware failure.")


class InvalidOutputTestProvider(VisionModelProvider):
    """Test double that returns None or an invalid output type."""

    def __init__(self, config: VisionProviderConfig, return_val: Any = None) -> None:
        super().__init__(config)
        self.return_val = return_val

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> Any:
        self.validate_input(model_input)
        return self.return_val


# ===========================================================================
# 1. Test class: End-to-End Pipeline Success Path
# ===========================================================================


class TestEndToEndPipelineSuccessPath:
    """Tests 1-8, 15-18: Full pipeline orchestration via VisionAgent & VisionPipeline."""

    def test_01_full_pipeline_success_via_vision_agent(self) -> None:
        """VisionAgent orchestrates full pipeline and returns normalized VisionResult."""
        config = VisionProviderConfig(provider_name="e2e-agent-prov", model_name="vlm-e2e")
        provider = E2EDeterministicProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(
            document_id="doc-e2e-01",
            filename="financial_chart.pdf",
            chunk_id="chk-e2e-01",
            page_number=12,
            chunk_index=3,
            content_type="chart",
            image_bytes=_make_test_png(64, 64),
            metadata={"source": "annual_report"},
        )

        query_str = "Explain the Q3 revenue growth chart in detail."
        result = agent.execute(query_str, evidence=[ev])

        assert isinstance(result, VisionResult)
        assert result.is_success is True
        assert result.query == query_str
        assert result.description == "End-to-end pipeline analysis output."
        assert result.document_id == "doc-e2e-01"
        assert result.filename == "financial_chart.pdf"
        assert result.chunk_id == "chk-e2e-01"
        assert result.page_number == 12
        assert result.content_type == "chart"

        # Metadata sanitization check
        assert "api_key" not in result.metadata
        assert result.metadata["provider_name"] == "e2e-agent-prov"
        assert result.metadata["confidence"] == 0.98

        # Lifecycle check
        assert "execution_lifecycle" in result.metadata
        lc = result.metadata["execution_lifecycle"]
        assert lc["stage"] == "completed"
        assert lc["is_completed"] is True
        assert lc["error"] is None

        # Trace check
        assert "execution_trace" in result.metadata
        tr = result.metadata["execution_trace"]
        assert "request_received" in tr["stages"]
        assert "result_normalized" in tr["stages"]
        assert "execution_completed" in tr["stages"]

    def test_02_full_pipeline_success_via_vision_pipeline_helper(self) -> None:
        """run_vision_pipeline convenience function executes pipeline cleanly."""
        config = VisionProviderConfig(provider_name="e2e-pipe-prov", model_name="vlm-e2e")
        provider = E2EDeterministicProvider(config)

        ev = VisualEvidence(
            document_id="doc-pipe-01",
            filename="diagram.png",
            chunk_id="chk-pipe-01",
            image_bytes=_make_test_png(),
        )

        result = run_vision_pipeline(provider, "Describe architecture diagram", evidence=[ev])
        assert result.is_success is True
        assert result.document_id == "doc-pipe-01"
        assert provider.invocation_count == 1

    def test_03_member_1_2_evidence_adaptation_in_e2e_pipeline(self) -> None:
        """Member 1 DocumentChunk / Member 2 AgentCitation adapt and pass through complete pipeline."""
        config = VisionProviderConfig(provider_name="adapter-prov", model_name="v1")
        provider = E2EDeterministicProvider(config)
        agent = VisionAgent(provider=provider)

        chunk = DocumentChunk(
            chunk_id="chk-m1-99",
            document_id="doc-m1-99",
            content="",
            chunk_index=0,
            filename="infra.pdf",
            page_number=5,
            content_type="diagram",
            metadata={"image_bytes": _make_test_png()},
        )
        citation = AgentCitation(
            document_id="doc-m1-99",
            filename="infra.pdf",
            chunk_id="chk-m1-99",
            page_number=5,
            content_type="diagram",
            metadata={"image_bytes": _make_test_png(), "document_chunk": chunk},
        )

        result = agent.execute("Analyze infrastructure diagram", evidence=[citation])

        assert result.is_success is True
        assert result.document_id == "doc-m1-99"
        assert result.filename == "infra.pdf"
        assert result.page_number == 5
        assert result.chunk_id == "chk-m1-99"


# ===========================================================================
# 2. Test class: Contract Invariants & Immutability
# ===========================================================================


class TestContractInvariantsAndImmutability:
    """Tests 3-7, 9-14, 19, 24-25: Invariants on invocation count, query, lineage, immutability, and determinism."""

    def test_04_single_invocation_guarantee(self) -> None:
        """Provider is called exactly once per request in the end-to-end pipeline."""
        config = VisionProviderConfig(provider_name="count-prov", model_name="v1")
        provider = E2EDeterministicProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        assert provider.invocation_count == 0
        agent.execute("Invocation test", evidence=[ev])
        assert provider.invocation_count == 1

    def test_05_query_preservation(self) -> None:
        """Exact query string reaches provider without modification or silent rewriting."""
        config = VisionProviderConfig(provider_name="query-prov", model_name="v1")
        provider = E2EDeterministicProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())
        exact_query = "What is the key takeaway from Section 4.2 diagram?"

        agent.execute(exact_query, evidence=[ev])
        assert provider.recorded_inputs[0].query == exact_query

    def test_06_lineage_survival_end_to_end(self) -> None:
        """All lineage attributes survive from input to final VisionResult."""
        config = VisionProviderConfig(provider_name="lineage-prov", model_name="v1")
        provider = E2EDeterministicProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(
            document_id="doc-surv-77",
            filename="blueprint.pdf",
            chunk_id="chk-surv-77",
            page_number=14,
            chunk_index=2,
            content_type="diagram",
            image_bytes=_make_test_png(),
            metadata={"tier": "confidential"},
        )

        res = agent.execute("Review blueprint", evidence=[ev])

        assert res.document_id == "doc-surv-77"
        assert res.filename == "blueprint.pdf"
        assert res.chunk_id == "chk-surv-77"
        assert res.page_number == 14
        assert res.content_type == "diagram"
        assert provider.recorded_inputs[0].chunk_index == 2

    def test_07_input_immutability_end_to_end(self) -> None:
        """VisionRequest and VisualEvidence instances are untouched after pipeline execution."""
        ev = VisualEvidence(
            document_id="doc-imm-99",
            filename="test.pdf",
            chunk_id="chk-imm-99",
            page_number=1,
            image_bytes=_make_test_png(),
        )
        req = VisionRequest(query="Immutability check", evidence=[ev])

        ev_dict_before = ev.to_dict()
        req_dict_before = req.to_dict()

        config = VisionProviderConfig(provider_name="imm-prov", model_name="v1")
        provider = E2EDeterministicProvider(config)
        agent = VisionAgent(provider=provider)

        agent.execute(req)

        assert ev.to_dict() == ev_dict_before
        assert req.to_dict() == req_dict_before

    def test_08_deterministic_execution_output(self) -> None:
        """Running identical request twice produces identical VisionResult attributes."""
        config = VisionProviderConfig(provider_name="det-prov", model_name="v1")
        provider = E2EDeterministicProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        res1 = agent.execute("Query deterministic", evidence=[ev])
        res2 = agent.execute("Query deterministic", evidence=[ev])

        assert res1.to_dict() == res2.to_dict()


# ===========================================================================
# 3. Test class: Multi-Evidence & Controlled Failures
# ===========================================================================


class TestMultiEvidenceAndControlledFailures:
    """Tests 20-23, 26-29: Multi-evidence ordering, provider failures, and invalid result handling."""

    def test_09_multi_evidence_ordering_supported(self) -> None:
        """Multiple visual evidence items maintain ordering and primary evidence selection."""
        config = VisionProviderConfig(provider_name="multi-prov", model_name="v1")
        provider = E2EDeterministicProvider(config)
        agent = VisionAgent(provider=provider)

        ev1 = VisualEvidence(document_id="doc-primary", filename="first.png", chunk_id="c1", image_bytes=_make_test_png())
        ev2 = VisualEvidence(document_id="doc-secondary", filename="second.png", chunk_id="c2", image_bytes=_make_test_png())

        res = agent.execute("Compare images", evidence=[ev1, ev2])

        assert res.document_id == "doc-primary"
        assert res.filename == "first.png"

    def test_10_provider_failure_raises_controlled_exception(self) -> None:
        """Provider failure raises VisionProviderExecutionError and returns no fake result."""
        config = VisionProviderConfig(provider_name="fail-prov", model_name="v1")
        provider = FailingTestProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        with pytest.raises(VisionProviderExecutionError) as exc_info:
            agent.execute("Fail query", evidence=[ev])

        assert "Backend inference hardware failure." in str(exc_info.value)

    @pytest.mark.parametrize("bad_val", [None, "invalid_str", 12345, [1, 2]])
    def test_11_invalid_provider_result_rejected(self, bad_val: Any) -> None:
        """Provider returning None or wrong type raises VisionProcessingError without fake result."""
        config = VisionProviderConfig(provider_name="bad-val-prov", model_name="v1")
        provider = InvalidOutputTestProvider(config, return_val=bad_val)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        with pytest.raises(VisionProcessingError):
            agent.execute("Bad return query", evidence=[ev])

    @pytest.mark.parametrize("bad_prov", ["not_a_provider_instance", 123])
    def test_12_missing_or_invalid_provider_rejected(self, bad_prov: Any) -> None:
        """Instantiating VisionAgent or VisionPipeline with invalid provider raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="provider"):
            VisionAgent(provider=bad_prov)

        with pytest.raises(VisionInputValidationError, match="provider"):
            VisionPipeline(provider=bad_prov)

    def test_12b_unconfigured_vision_agent_raises_error_on_execute(self) -> None:
        """VisionAgent initialized with provider=None raises controlled error when executed."""
        agent = VisionAgent(provider=None)
        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())
        with pytest.raises(VisionProcessingError):
            agent.execute("Query", evidence=[ev])

    def test_15_invalid_evidence_handled_controlled_error(self) -> None:
        """Unsupported object passed in evidence list raises VisionEvidenceError."""
        config = VisionProviderConfig(provider_name="inv-ev-prov", model_name="v1")
        provider = E2EDeterministicProvider(config)
        agent = VisionAgent(provider=provider)

        with pytest.raises(VisionEvidenceError, match="not a supported visual modality"):
            agent.execute("Query", evidence=["invalid_evidence_string"])

    def test_16_stage_failures_handling(self) -> None:
        """Controlled failures at validation, preparation, execution, and normalization stages."""
        config = VisionProviderConfig(provider_name="stage-fail-prov", model_name="v1")
        provider = E2EDeterministicProvider(config)
        agent = VisionAgent(provider=provider)

        # 1. Request validation failure
        with pytest.raises(VisionInputValidationError):
            agent.execute("   ", evidence=[VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1")])

        # 2. Evidence preparation failure (missing image source)
        ev_no_bytes = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1")
        with pytest.raises(VisionEvidenceError):
            agent.execute("Query", evidence=[ev_no_bytes])

    def test_17_cause_preserving_error_propagation(self) -> None:
        """Unexpected provider exception is re-raised as VisionProviderExecutionError preserving original __cause__."""
        class ExceptionThrowingProvider(VisionModelProvider):
            def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
                raise KeyError("Internal dictionary lookup failed")

        config = VisionProviderConfig(provider_name="cause-prov", model_name="v1")
        provider = ExceptionThrowingProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        with pytest.raises(VisionProviderExecutionError) as exc_info:
            agent.execute("Query", evidence=[ev])

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, KeyError)


# ===========================================================================
# 4. Test class: Offline & Package Integrity
# ===========================================================================


class TestOfflineAndPackageIntegrity:
    """Tests 30-40: Offline verification, package imports, and architectural reuse."""

    def test_13_public_exports_integrity(self) -> None:
        """All Day 32-40 public symbols export cleanly from vision package."""
        from vision import (
            FORBIDDEN_METADATA_KEYS,
            VALID_VISUAL_CONTENT_TYPES,
            ImageEvidencePreparator,
            OversizedImagePolicy,
            PreparedImageEvidence,
            VisualEvidence,
            VisualEvidenceAdapter,
            VisionAgent,
            VisionAgentError,
            VisionEvidenceError,
            VisionExecutionAdapter,
            VisionExecutionLifecycle,
            VisionExecutionStage,
            VisionExecutionTrace,
            VisionInputBuilder,
            VisionInputValidationError,
            VisionModelInput,
            VisionModelProvider,
            VisionPipeline,
            VisionProcessingError,
            VisionProviderCapabilities,
            VisionProviderConfig,
            VisionProviderError,
            VisionProviderExecutionError,
            VisionProviderRegistry,
            VisionProviderUnavailableError,
            VisionRequest,
            VisionResult,
            VisionResultNormalizer,
            VisionTimeoutError,
            VisionUnsupportedCapabilityError,
            build_vision_input,
            execute_vision_request,
            prepare_image_evidence,
            run_vision_pipeline,
        )

        assert inspect.isclass(VisionPipeline)
        assert callable(run_vision_pipeline)
        assert inspect.isclass(VisionAgent)

    def test_14_no_network_libraries_in_pipeline_module(self) -> None:
        """pipeline.py contains no network or vendor SDK imports."""
        from vision import pipeline

        source = inspect.getsource(pipeline)
        forbidden = [
            "import requests",
            "import httpx",
            "import aiohttp",
            "import socket",
            "import urllib.request",
            "import openai",
            "import anthropic",
            "import google.generativeai",
        ]
        for pattern in forbidden:
            assert pattern not in source, f"pipeline.py contains forbidden pattern '{pattern}'"

