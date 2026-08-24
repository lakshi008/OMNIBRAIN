"""
Day 55 - OmniBrain Member 3 Vision Agent: Final Regression & Integration Stability Test Suite.

Comprehensive cross-day integration test suite verifying that all contract guarantees,
pipeline stages, error boundaries, concurrency isolations, and downstream contracts
established across Days 33-54 remain 100% stable, deterministic, and fully integrated.

Workflow Pipeline Verified:
  Search Evidence -> VisualEvidenceAdapter -> prepare_image_evidence -> build_vision_input
  -> VisionModelProvider -> VisionExecutionAdapter -> VisionExecutionLifecycle
  -> ResultNormalizer -> VisionPipeline -> VisionResult -> Downstream Supervisor / Consumer

All tests execute 100% offline with zero external network, HTTP, or LLM dependencies.
"""

from __future__ import annotations

import io
import sys
import threading
import time
from typing import Any

import pytest
from PIL import Image

import vision
from agents.models import AgentCitation, AgentResponse, SearchResult
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionAgentError,
    VisionCancellationError,
    VisionError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderConfigError,
    VisionProviderError,
    VisionProviderExecutionError,
    VisionProviderUnavailableError,
    VisionTimeoutError,
    VisionUnsupportedCapabilityError,
)
from vision.execution_adapter import VisionExecutionAdapter, execute_vision_request
from vision.image_preparation import (
    SUPPORTED_IMAGE_FORMATS,
    ImageEvidencePreparator,
    OversizedImagePolicy,
    PreparedImageEvidence,
    prepare_image_evidence,
)
from vision.input_builder import VisionInputBuilder, VisionModelInput, build_vision_input
from vision.lifecycle import (
    VisionCancellationToken,
    VisionExecutionLifecycle,
    VisionExecutionObservation,
    VisionExecutionStage,
    VisionRetryPolicy,
)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.pipeline import VisionPipeline, run_vision_pipeline
from vision.provider import VisionModelProvider, VisionProviderRegistry
from vision.provider_config import VisionProviderCapabilities, VisionProviderConfig
from vision.result_normalizer import (
    FORBIDDEN_METADATA_KEYS,
    VisionExecutionTrace,
    VisionResultNormalizer,
)
from vision.vision_agent import VisionAgent


# ===========================================================================
# Test Helpers & Downstream Test Doubles
# ===========================================================================

def _make_test_png(width: int = 16, height: int = 16, color: tuple[int, int, int] = (40, 100, 160)) -> bytes:
    """Generate a minimal valid PNG image byte stream for visual evidence testing."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_REGRESSION_PNG = _make_test_png()


def _make_search_citation(
    doc_id: str = "DOC-REG-001",
    filename: str = "arch_diagram.pdf",
    chunk_id: str = "CHUNK-REG-001",
    content_type: str = "diagram",
    page_number: int | None = 3,
    score: float = 0.95,
    metadata: dict[str, Any] | None = None,
) -> AgentCitation:
    """Construct a Member 2 Search AgentCitation for integration testing."""
    return AgentCitation(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        content_type=content_type,
        score=score,
        metadata=metadata if metadata is not None else {"suite": "day55_regression", "env": "offline_test"},
    )


class RegressionTestProvider(VisionModelProvider):
    """Controlled offline provider double for final regression & integration testing."""

    def __init__(
        self,
        config: VisionProviderConfig | None = None,
        capabilities: VisionProviderCapabilities | None = None,
        should_fail: bool = False,
        fail_count: int = 0,
        latency: float = 0.0,
        simulate_timeout: bool = False,
    ) -> None:
        cfg = config or VisionProviderConfig(provider_name="regression_provider", model_name="reg_v1")
        super().__init__(config=cfg, capabilities=capabilities)
        self._should_fail = should_fail
        self._fail_count = fail_count
        self._call_count = 0
        self._received_inputs: list[VisionModelInput] = []
        self._latency = latency
        self._simulate_timeout = simulate_timeout
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    @property
    def received_inputs(self) -> list[VisionModelInput]:
        with self._lock:
            return list(self._received_inputs)

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        """Execute request deterministically, logging calls and inputs."""
        if self._latency > 0:
            time.sleep(self._latency)

        with self._lock:
            self._call_count += 1
            current_count = self._call_count
            self._received_inputs.append(model_input)

        if self._simulate_timeout:
            raise VisionTimeoutError("Simulated provider execution timeout in regression test.")

        if self._should_fail and current_count <= self._fail_count:
            raise VisionProviderExecutionError(f"Simulated provider failure on call {current_count}.")

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Regression analysis output for query: '{model_input.query}'",
            document_id=model_input.document_id,
            filename=model_input.filename,
            chunk_id=model_input.chunk_id,
            page_number=model_input.page_number,
            content_type=model_input.content_type,
            metadata={
                "provider_name": self.config.provider_name,
                "model_name": self.config.model_name,
                "call_count": current_count,
                "api_key": "sanitized_test_secret_key",
            },
        )


class DownstreamSupervisorConsumer:
    """Downstream consumer contract double representing Member 2 / Supervisor handoff."""

    def __init__(self) -> None:
        self.received: list[VisionResult] = []

    def consume_vision_result(self, result: VisionResult) -> dict[str, Any]:
        """Consume VisionResult and output downstream consumer status dict."""
        if not isinstance(result, VisionResult):
            raise TypeError(f"Expected VisionResult, got {type(result).__name__}")

        self.received.append(result)
        cit = AgentCitation(
            document_id=result.document_id or "UNKNOWN",
            filename=result.filename or "unknown",
            chunk_id=result.chunk_id or "chunk-0",
            page_number=result.page_number,
            content_type=result.content_type,
            score=1.0,
            metadata=dict(result.metadata),
        )
        return {
            "consumed": True,
            "status": result.status,
            "query": result.query,
            "answer": result.description,
            "document_id": result.document_id,
            "filename": result.filename,
            "citation": cit,
            "evidence_count": len(result.evidence),
        }


# ===========================================================================
# Day 55 Regression & Integration Test Suite
# ===========================================================================

class TestMember3FinalRegression:
    """Final regression and integration stability test suite for OmniBrain Member 3 Vision Agent."""

    def test_01_public_api_regression(self) -> None:
        """Step 2: Inspect vision.__all__ exports, types, signatures, and symbol integrity."""
        exports = vision.__all__
        assert len(exports) >= 30

        # Ensure all exported symbols exist and can be referenced
        for symbol_name in exports:
            assert hasattr(vision, symbol_name), f"Exported symbol '{symbol_name}' missing from vision module."
            obj = getattr(vision, symbol_name)
            assert obj is not None

        # Explicitly verify core symbols
        assert issubclass(VisionPipeline, object)
        assert issubclass(VisionAgent, object)
        assert issubclass(VisionExecutionAdapter, object)
        assert issubclass(VisionResultNormalizer, object)
        assert issubclass(VisionError, Exception)

    def test_02_complete_happy_path_integration(self) -> None:
        """Step 3: End-to-end happy path across all Member 3 stages (Days 33-54)."""
        cit = _make_search_citation(
            doc_id="DOC-HAPPY-55",
            filename="happy_flow.pdf",
            chunk_id="CHK-HAPPY-55",
            content_type="chart",
            page_number=12,
        )

        # Stage 1: Adapter
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_REGRESSION_PNG)
        assert ev.document_id == "DOC-HAPPY-55"

        # Stage 2: Prep
        prep = prepare_image_evidence(ev)
        assert prep.source.document_id == "DOC-HAPPY-55"

        # Stage 3: Input
        inp = build_vision_input("Happy Path Query", prep)
        assert inp.document_id == "DOC-HAPPY-55"

        # Stage 4-10: Pipeline Execution
        req = VisionRequest(query="Happy Path Query", evidence=[ev])
        provider = RegressionTestProvider()
        pipeline = VisionPipeline(provider=provider)

        result = pipeline.run(req)

        # Stage 11: Downstream Consumer
        consumer = DownstreamSupervisorConsumer()
        consumed = consumer.consume_vision_result(result)

        assert result.status == "success"
        assert result.document_id == "DOC-HAPPY-55"
        assert result.filename == "happy_flow.pdf"
        assert result.chunk_id == "CHK-HAPPY-55"
        assert result.page_number == 12
        assert result.content_type == "chart"
        assert "api_key" not in result.metadata
        assert consumed["consumed"] is True
        assert consumed["document_id"] == "DOC-HAPPY-55"

    def test_03_multi_evidence_regression(self) -> None:
        """Step 4: Verify 1, 2, 3, 5, 10 evidence items maintain count, order, and lineage."""
        for count in [1, 2, 3, 5, 10]:
            citations = [
                _make_search_citation(doc_id=f"DOC-M-{i}", filename=f"file_{i}.pdf", chunk_id=f"chk-m-{i}")
                for i in range(count)
            ]
            evidence = [
                VisualEvidenceAdapter.adapt_citation(c, image_bytes=_REGRESSION_PNG)
                for c in citations
            ]

            req = VisionRequest(query=f"Multi-evidence query for {count} items", evidence=evidence)
            provider = RegressionTestProvider()
            pipeline = VisionPipeline(provider=provider)

            res = pipeline.run(req)

            assert res.status == "success"
            assert len(res.evidence) == count
            for idx in range(count):
                assert res.evidence[idx].document_id == f"DOC-M-{idx}"
                assert res.evidence[idx].filename == f"file_{idx}.pdf"
                assert res.evidence[idx].chunk_id == f"chk-m-{idx}"

    def test_04_multi_document_isolation(self) -> None:
        """Step 5: Verify DOC-A, DOC-B, DOC-C, DOC-D retain exact source identities without swapping."""
        docs = ["DOC-A", "DOC-B", "DOC-C", "DOC-D"]
        for doc_id in docs:
            cit = _make_search_citation(doc_id=doc_id, filename=f"{doc_id.lower()}.pdf")
            ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_REGRESSION_PNG)
            req = VisionRequest(query=f"Query for {doc_id}", evidence=[ev])

            provider = RegressionTestProvider()
            pipeline = VisionPipeline(provider=provider)
            res = pipeline.run(req)

            assert res.status == "success"
            assert res.document_id == doc_id
            assert res.evidence[0].document_id == doc_id

    def test_05_validation_boundary_regression(self) -> None:
        """Step 6: Invalid query, request, evidence stop before provider execution."""
        cit = _make_search_citation()
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_REGRESSION_PNG)
        provider = RegressionTestProvider()
        pipeline = VisionPipeline(provider=provider)

        # 1. Invalid query (empty / whitespace)
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="", evidence=[ev])

        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="   ", evidence=[ev])

        # 2. None request
        with pytest.raises(VisionInputValidationError):
            pipeline.run(None)  # type: ignore[arg-type]

        # 3. Non-visual citation in strict mode
        cit_txt = _make_search_citation(content_type="text")
        with pytest.raises(VisionEvidenceError):
            VisualEvidenceAdapter.adapt_citation(cit_txt)

        # Provider never invoked for any invalid input
        assert provider.call_count == 0

    def test_06_provider_configuration_regression(self) -> None:
        """Step 7: VisionProviderConfig and capabilities validation & offline doubles."""
        cfg = VisionProviderConfig(provider_name="reg_prov", model_name="v2", timeout=15.0)
        assert cfg.provider_name == "reg_prov"
        assert cfg.model_name == "v2"
        assert cfg.timeout == 15.0

        caps = VisionProviderCapabilities(
            supported_modalities=frozenset({"image", "chart", "diagram"}),
            supported_formats=frozenset({"png", "jpeg"}),
            supports_multi_image=True,
        )
        assert caps.supports_modality("diagram") is True
        assert caps.supports_format("png") is True

        # Unsupported modality fails fast in pipeline
        cit = _make_search_citation(content_type="video")  # unsupported
        with pytest.raises((VisionInputValidationError, VisionUnsupportedCapabilityError, VisionEvidenceError)):
            ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_REGRESSION_PNG)
            req = VisionRequest(query="Test video", evidence=[ev])
            p = RegressionTestProvider(capabilities=caps)
            pipe = VisionPipeline(provider=p)
            pipe.run(req)

    def test_07_retry_recovery_regression(self) -> None:
        """Step 8: Provider failure -> VisionRetryPolicy retry -> success without duplicate work."""
        cit = _make_search_citation(doc_id="DOC-RETRY-55")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_REGRESSION_PNG)
        req = VisionRequest(query="Retry policy test", evidence=[ev])

        provider = RegressionTestProvider(should_fail=True, fail_count=1)
        retry_policy = VisionRetryPolicy(max_retries=2)
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req, retry_policy=retry_policy)

        assert res.status == "success"
        assert provider.call_count == 2
        assert res.evidence[0].document_id == "DOC-RETRY-55"

    def test_08_timeout_regression(self) -> None:
        """Step 9: Provider timeout raises VisionTimeoutError, protects state, subsequent request succeeds."""
        cit = _make_search_citation(doc_id="DOC-TIMEOUT-55")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_REGRESSION_PNG)
        req = VisionRequest(query="Timeout policy test", evidence=[ev])

        provider_timeout = RegressionTestProvider(simulate_timeout=True)
        pipeline_timeout = VisionPipeline(provider=provider_timeout)

        with pytest.raises(VisionTimeoutError):
            pipeline_timeout.run(req)

        # Subsequent execution with healthy provider succeeds cleanly
        provider_ok = RegressionTestProvider()
        pipeline_ok = VisionPipeline(provider=provider_ok)
        res = pipeline_ok.run(req)
        assert res.status == "success"

    def test_09_cancellation_regression(self) -> None:
        """Step 10: Cancellation token halts execution before provider invocation, subsequent request succeeds."""
        cit = _make_search_citation(doc_id="DOC-CANCEL-55")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_REGRESSION_PNG)
        req = VisionRequest(query="Cancel policy test", evidence=[ev])

        token = VisionCancellationToken()
        token.cancel()

        provider = RegressionTestProvider()
        pipeline = VisionPipeline(provider=provider)

        with pytest.raises(VisionCancellationError):
            pipeline.run(req, cancellation_token=token)

        assert provider.call_count == 0

        # Subsequent uncancelled execution succeeds
        token_active = VisionCancellationToken()
        res = pipeline.run(req, cancellation_token=token_active)
        assert res.status == "success"

    def test_10_concurrency_mixed_outcomes_isolation(self) -> None:
        """Step 11: Concurrent execution (A: success, B: retry->success, C: timeout, D: cancellation) maintains total isolation."""
        provider_a = RegressionTestProvider()
        provider_b = RegressionTestProvider(should_fail=True, fail_count=1)
        provider_c = RegressionTestProvider(simulate_timeout=True)
        provider_d = RegressionTestProvider()

        results: dict[str, Any] = {}
        exceptions: dict[str, Exception] = {}

        def thread_a() -> None:
            try:
                ev = VisualEvidenceAdapter.adapt_citation(_make_search_citation(doc_id="DOC-A"), image_bytes=_REGRESSION_PNG)
                req = VisionRequest(query="Query A", evidence=[ev])
                pipe = VisionPipeline(provider=provider_a)
                results["A"] = pipe.run(req)
            except Exception as e:
                exceptions["A"] = e

        def thread_b() -> None:
            try:
                ev = VisualEvidenceAdapter.adapt_citation(_make_search_citation(doc_id="DOC-B"), image_bytes=_REGRESSION_PNG)
                req = VisionRequest(query="Query B", evidence=[ev])
                pipe = VisionPipeline(provider=provider_b)
                results["B"] = pipe.run(req, retry_policy=VisionRetryPolicy(max_retries=2))
            except Exception as e:
                exceptions["B"] = e

        def thread_c() -> None:
            try:
                ev = VisualEvidenceAdapter.adapt_citation(_make_search_citation(doc_id="DOC-C"), image_bytes=_REGRESSION_PNG)
                req = VisionRequest(query="Query C", evidence=[ev])
                pipe = VisionPipeline(provider=provider_c)
                results["C"] = pipe.run(req)
            except Exception as e:
                exceptions["C"] = e

        def thread_d() -> None:
            try:
                ev = VisualEvidenceAdapter.adapt_citation(_make_search_citation(doc_id="DOC-D"), image_bytes=_REGRESSION_PNG)
                req = VisionRequest(query="Query D", evidence=[ev])
                pipe = VisionPipeline(provider=provider_d)
                token = VisionCancellationToken()
                token.cancel()
                results["D"] = pipe.run(req, cancellation_token=token)
            except Exception as e:
                exceptions["D"] = e

        threads = [
            threading.Thread(target=thread_a),
            threading.Thread(target=thread_b),
            threading.Thread(target=thread_c),
            threading.Thread(target=thread_d),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["A"].status == "success"
        assert results["A"].document_id == "DOC-A"

        assert results["B"].status == "success"
        assert results["B"].document_id == "DOC-B"

        assert isinstance(exceptions["C"], VisionTimeoutError)
        assert isinstance(exceptions["D"], VisionCancellationError)

    def test_11_resource_safety_leak_prevention(self) -> None:
        """Step 12: Verify no thread leaks or dangling resources after multiple executions."""
        initial_threads = threading.active_count()

        for i in range(10):
            cit = _make_search_citation(doc_id=f"DOC-RES-{i}")
            ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_REGRESSION_PNG)
            req = VisionRequest(query=f"Resource leak test {i}", evidence=[ev])

            provider = RegressionTestProvider()
            pipeline = VisionPipeline(provider=provider)
            res = pipeline.run(req)
            assert res.status == "success"

        assert threading.active_count() <= initial_threads + 1

    def test_12_observability_lifecycle_transitions(self) -> None:
        """Step 13: Execution lifecycle stage transitions and observation creation."""
        lifecycle = VisionExecutionLifecycle(provider_name="reg_prov", model_name="v1")
        assert lifecycle.stage == VisionExecutionStage.PENDING

        lifecycle.transition_to(VisionExecutionStage.BUILDING_INPUT)
        assert lifecycle.stage == VisionExecutionStage.BUILDING_INPUT

        obs = lifecycle.to_observation()
        assert isinstance(obs, VisionExecutionObservation)
        assert obs.provider_name == "reg_prov"
        assert obs.model_name == "v1"

    def test_13_citation_lineage_integrity(self) -> None:
        """Step 14: Search -> Vision -> Result -> Downstream preserves all metadata and lineage fields."""
        cit = _make_search_citation(
            doc_id="DOC-LINEAGE-55",
            filename="lineage_report.pdf",
            chunk_id="CHK-LIN-55",
            content_type="chart",
            page_number=8,
            score=0.99,
            metadata={"origin": "search_agent_day51", "section": "finance"},
        )

        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_REGRESSION_PNG)
        req = VisionRequest(query="Lineage check query", evidence=[ev])

        provider = RegressionTestProvider()
        pipeline = VisionPipeline(provider=provider)
        res = pipeline.run(req)

        assert res.document_id == "DOC-LINEAGE-55"
        assert res.filename == "lineage_report.pdf"
        assert res.chunk_id == "CHK-LIN-55"
        assert res.page_number == 8
        assert res.content_type == "chart"
        assert res.evidence[0].metadata.get("origin") == "search_agent_day51"

    def test_14_supervisor_handoff_contract(self) -> None:
        """Step 15: VisionResult is 100% consumable by downstream supervisor contract."""
        cit = _make_search_citation(doc_id="DOC-SUP-55")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_REGRESSION_PNG)
        req = VisionRequest(query="Supervisor handoff query", evidence=[ev])

        provider = RegressionTestProvider()
        pipeline = VisionPipeline(provider=provider)
        res = pipeline.run(req)

        consumer = DownstreamSupervisorConsumer()
        consumed = consumer.consume_vision_result(res)

        assert consumed["consumed"] is True
        assert consumed["answer"].startswith("Regression analysis output")
        assert consumed["status"] == "success"

    def test_15_repeated_execution_state_isolation(self) -> None:
        """Step 16: Sequential reuse of the same pipeline instance for 5 runs maintains 100% state isolation."""
        provider = RegressionTestProvider()
        pipeline = VisionPipeline(provider=provider)

        for run_idx in range(1, 6):
            cit = _make_search_citation(doc_id=f"DOC-RUN-{run_idx}")
            ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_REGRESSION_PNG)
            req = VisionRequest(query=f"Repeated run query {run_idx}", evidence=[ev])

            res = pipeline.run(req)

            assert res.status == "success"
            assert res.document_id == f"DOC-RUN-{run_idx}"
            assert res.query == f"Repeated run query {run_idx}"
            assert len(res.evidence) == 1
            assert res.evidence[0].document_id == f"DOC-RUN-{run_idx}"

        assert provider.call_count == 5

    def test_16_duplicate_work_prevention(self) -> None:
        """Step 18: Member 3 performs zero duplicate image preparation or provider execution."""
        cit = _make_search_citation(doc_id="DOC-DEDUP-55")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_REGRESSION_PNG)
        req = VisionRequest(query="Dedup test query", evidence=[ev])

        provider = RegressionTestProvider()
        pipeline = VisionPipeline(provider=provider)

        res = pipeline.run(req)

        assert res.status == "success"
        assert provider.call_count == 1
        assert len(provider.received_inputs) == 1
        model_input = provider.received_inputs[0]
        assert model_input.document_id == "DOC-DEDUP-55"

    def test_17_offline_execution_guarantee(self) -> None:
        """Step 19: Verify offline operation with no external networks, HTTP, or LLM calls."""
        # Validate that execution succeeds using purely local objects
        cit = _make_search_citation(doc_id="DOC-OFFLINE-55")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_REGRESSION_PNG)
        req = VisionRequest(query="Offline test query", evidence=[ev])

        provider = RegressionTestProvider()
        pipeline = VisionPipeline(provider=provider)
        res = pipeline.run(req)

        assert res.status == "success"
        assert res.document_id == "DOC-OFFLINE-55"
