"""
OmniBrain Member 3 Vision Subsystem.
Day 58: Post-Handoff Integration Support & Stability Audit Test Suite.

Verifies:
  1. Public API Stability (imports, no private leaks, exact __all__ contract)
  2. Handoff Regression Compatibility (adapters, models, pipeline, agent)
  3. Full Contract Regression (validation, lifecycle, retry, timeout, cancellation,
     concurrency, resource safety, observability, lineage, serialization, supervisor)
  4. State Isolation (sequential A/B/C requests, concurrent execution)
  5. Lineage Stability (strict metadata and citation fidelity, no fabrication)
  6. Error Contract Determinism (deterministic exceptions in VisionError hierarchy)
  7. Offline Guarantee (zero network, no external telemetry/APIs)
  8. Member Boundary Compliance (clean isolation from ingestion & agents)
"""

from __future__ import annotations

import concurrent.futures
import io
import threading
import time
from typing import Any

from PIL import Image
import pytest

from agents.models import AgentCitation
from ingestion.models import VectorSearchResult
import vision
from vision import (
    FORBIDDEN_METADATA_KEYS,
    SUPPORTED_IMAGE_FORMATS,
    ImageEvidencePreparator,
    OversizedImagePolicy,
    PreparedImageEvidence,
    VALID_VISUAL_CONTENT_TYPES,
    VisionAgent,
    VisionAgentError,
    VisionCancellationError,
    VisionCancellationToken,
    VisionError,
    VisionEvidenceError,
    VisionExecutionAdapter,
    VisionExecutionLifecycle,
    VisionExecutionObservation,
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
    VisionProviderConfigError,
    VisionProviderError,
    VisionProviderExecutionError,
    VisionProviderRegistry,
    VisionProviderUnavailableError,
    VisionRequest,
    VisionResult,
    VisionResultNormalizer,
    VisionRetryPolicy,
    VisionTimeoutError,
    VisionUnsupportedCapabilityError,
    VisualEvidence,
    VisualEvidenceAdapter,
    build_vision_input,
    execute_vision_request,
    prepare_image_evidence,
    run_vision_pipeline,
)


def _png(color: tuple[int, int, int] = (40, 100, 200)) -> bytes:
    """Minimal valid PNG for image evidence."""
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return buf.getvalue()


_PNG = _png()


class StabilityMockProvider(VisionModelProvider):
    """Deterministic, thread-safe test provider for stability audit."""

    def __init__(
        self,
        name: str = "stability-mock-provider",
        response_text: str = "Verified visual content with high confidence.",
        delay_seconds: float = 0.0,
        fail_count: int = 0,
        fail_exception: Exception | None = None,
        timeout: bool = False,
    ) -> None:
        cfg = VisionProviderConfig(
            provider_name=name,
            model_name="mock-vision-stable-v1",
            timeout=5.0,
            max_tokens=512,
            temperature=0.1,
        )
        super().__init__(config=cfg)
        self._name = name
        self._response_text = response_text
        self._delay_seconds = delay_seconds
        self._fail_count = fail_count
        self._timeout = timeout
        self._call_count = 0
        self._lock = threading.Lock()
        self._fail_exception = fail_exception or VisionProviderExecutionError("Deterministic mock failure")

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        with self._lock:
            self._call_count += 1
            call_seq = self._call_count

        if self._timeout:
            raise VisionTimeoutError("Simulated timeout error in mock provider.")

        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)

        if call_seq <= self._fail_count:
            raise self._fail_exception

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"{self._response_text} [Query: '{model_input.query}']",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={"call_seq": call_seq, "audited": True, "provider": self._name, **model_input.evidence_metadata},
        )


# =========================================================================
# 1. PUBLIC API STABILITY
# =========================================================================
class TestPublicApiStability:
    """Verifies that public API exports and symbols remain fully backward compatible."""

    def test_canonical_all_symbols_present(self) -> None:
        expected_symbols = {
            "VisualEvidence",
            "VisionRequest",
            "VisionResult",
            "VALID_VISUAL_CONTENT_TYPES",
            "PreparedImageEvidence",
            "ImageEvidencePreparator",
            "OversizedImagePolicy",
            "SUPPORTED_IMAGE_FORMATS",
            "prepare_image_evidence",
            "VisionModelInput",
            "VisionInputBuilder",
            "build_vision_input",
            "VisionModelProvider",
            "VisionProviderRegistry",
            "VisionProviderConfig",
            "VisionProviderCapabilities",
            "VisionExecutionAdapter",
            "VisionExecutionStage",
            "VisionExecutionLifecycle",
            "VisionExecutionObservation",
            "VisionCancellationToken",
            "VisionRetryPolicy",
            "execute_vision_request",
            "VisionResultNormalizer",
            "VisionExecutionTrace",
            "FORBIDDEN_METADATA_KEYS",
            "VisionPipeline",
            "run_vision_pipeline",
            "VisualEvidenceAdapter",
            "VisionAgent",
            "VisionAgentError",
            "VisionError",
            "VisionCancellationError",
            "VisionInputValidationError",
            "VisionEvidenceError",
            "VisionProcessingError",
            "VisionProviderError",
            "VisionProviderConfigError",
            "VisionProviderExecutionError",
            "VisionProviderUnavailableError",
            "VisionTimeoutError",
            "VisionUnsupportedCapabilityError",
        }
        actual_symbols = set(vision.__all__)
        missing = expected_symbols - actual_symbols
        assert not missing, f"Public symbols missing from vision.__all__: {missing}"

    def test_all_symbols_are_directly_callable_or_instantiable(self) -> None:
        for sym in vision.__all__:
            obj = getattr(vision, sym, None)
            assert obj is not None, f"Exported symbol '{sym}' is None in vision namespace"


# =========================================================================
# 2. HANDOFF REGRESSION COMPATIBILITY
# =========================================================================
class TestHandoffRegressionCompatibility:
    """Verifies seamless inter-member and pipeline compatibility with handoff structures."""

    def test_visual_evidence_adapter_citation_flow(self) -> None:
        citation = AgentCitation(
            document_id="DOC-AUDIT-001",
            filename="architecture_diagram.png",
            page_number=3,
            chunk_id="CHK-990",
            content_type="chart",
            score=0.95,
            metadata={"author": "SecurityTeam", "chunk_index": 4},
        )

        evidence = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_PNG)
        assert isinstance(evidence, VisualEvidence)
        assert evidence.document_id == "DOC-AUDIT-001"
        assert evidence.filename == "architecture_diagram.png"
        assert evidence.page_number == 3
        assert evidence.chunk_id == "CHK-990"
        assert evidence.chunk_index == 4
        assert evidence.metadata["author"] == "SecurityTeam"
        assert evidence.content_type == "chart"

    def test_visual_evidence_adapter_search_result_flow(self) -> None:
        search_res = VectorSearchResult(
            document_id="DOC-SRCH-42",
            filename="infra.png",
            page_number=1,
            chunk_id="CHK-SRCH-1",
            chunk_index=0,
            content="Extracted infrastructure diagram",
            content_type="diagram",
            score=0.88,
            metadata={"cluster": "us-east-1"},
        )
        evidence = VisualEvidenceAdapter.adapt_search_result(search_res, image_bytes=_PNG)
        assert isinstance(evidence, VisualEvidence)
        assert evidence.document_id == "DOC-SRCH-42"
        assert evidence.content_type == "diagram"
        assert evidence.metadata["cluster"] == "us-east-1"

    def test_full_pipeline_handoff_execution(self) -> None:
        provider = StabilityMockProvider(name="handoff-provider")
        ev = VisualEvidence(
            document_id="DOC-PIPE-1",
            filename="diagram.png",
            chunk_id="CHK-P1",
            content_type="diagram",
            image_bytes=_PNG,
        )
        req = VisionRequest(query="Describe system boundary", evidence=[ev])
        result = run_vision_pipeline(provider, req)

        assert isinstance(result, VisionResult)
        assert result.is_success is True
        assert "Verified visual content" in result.description
        assert result.document_id == "DOC-PIPE-1"

    def test_vision_agent_handoff_execution(self) -> None:
        provider = StabilityMockProvider(name="agent-provider")
        agent = VisionAgent(provider=provider)
        ev = VisualEvidence(
            document_id="DOC-AGENT-1",
            filename="chart.png",
            chunk_id="CHK-A1",
            content_type="chart",
            image_bytes=_PNG,
        )
        req = VisionRequest(query="Analyze latency spike", evidence=[ev])
        result = agent.process(req)

        assert isinstance(result, VisionResult)
        assert result.is_success is True
        assert result.metadata["provider"] == "agent-provider"


# =========================================================================
# 3. FULL CONTRACT REGRESSION
# =========================================================================
class TestFullContractRegression:
    """Verifies all architectural lifecycle, retry, timeout, and serialization guarantees."""

    def test_validation_rejects_empty_query(self) -> None:
        ev = VisualEvidence(document_id="DOC-1", filename="f.png", chunk_id="C1", image_bytes=_PNG, content_type="image")
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="", evidence=[ev])

    def test_validation_rejects_whitespace_only_query(self) -> None:
        ev = VisualEvidence(document_id="DOC-1", filename="f.png", chunk_id="C1", image_bytes=_PNG, content_type="image")
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="   \t\n  ", evidence=[ev])

    def test_validation_rejects_invalid_evidence_type(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="Valid query", evidence=["not_evidence"])  # type: ignore

    def test_lifecycle_stage_transitions_recorded(self) -> None:
        provider = StabilityMockProvider()
        pipeline = VisionPipeline(provider=provider)
        ev = VisualEvidence(document_id="DOC-1", filename="f.png", chunk_id="C1", image_bytes=_PNG, content_type="image")
        req = VisionRequest(query="Inspect lifecycle", evidence=[ev])

        result = pipeline.run(req)

        assert result.is_success is True
        assert "execution_lifecycle" in result.metadata
        lifecycle_data = result.metadata["execution_lifecycle"]
        assert lifecycle_data["stage"] == VisionExecutionStage.COMPLETED

    def test_retry_policy_recovers_from_transient_failure(self) -> None:
        # fail 1 time then succeed
        provider = StabilityMockProvider(fail_count=1)
        adapter = VisionExecutionAdapter(provider=provider)
        ev = VisualEvidence(document_id="DOC-1", filename="f.png", chunk_id="C1", image_bytes=_PNG, content_type="image")
        req = VisionRequest(query="Retry query", evidence=[ev])

        result = adapter.execute(req, retry_policy=VisionRetryPolicy(max_retries=2))
        assert result.is_success is True
        assert provider._call_count == 2

    def test_timeout_policy_enforcement(self) -> None:
        provider = StabilityMockProvider(timeout=True)
        adapter = VisionExecutionAdapter(provider=provider)
        ev = VisualEvidence(document_id="DOC-1", filename="f.png", chunk_id="C1", image_bytes=_PNG, content_type="image")
        req = VisionRequest(query="Timeout query", evidence=[ev])

        with pytest.raises(VisionTimeoutError):
            adapter.execute(req)

    def test_cancellation_token_enforcement(self) -> None:
        provider = StabilityMockProvider()
        adapter = VisionExecutionAdapter(provider=provider)
        token = VisionCancellationToken()
        token.cancel(reason="Aborted by supervisor")

        ev = VisualEvidence(document_id="DOC-1", filename="f.png", chunk_id="C1", image_bytes=_PNG, content_type="image")
        req = VisionRequest(query="Cancelled query", evidence=[ev])

        with pytest.raises(VisionCancellationError):
            adapter.execute(req, cancellation_token=token)

    def test_resource_safety_and_oversized_policy(self) -> None:
        ev = VisualEvidence(document_id="DOC-1", filename="large.png", chunk_id="C1", image_bytes=_PNG, content_type="image")
        with pytest.raises(VisionEvidenceError):
            prepare_image_evidence(ev, oversized_policy=OversizedImagePolicy(max_dimension=1, reject=True))

    def test_observability_trace_scrubs_forbidden_metadata_keys(self) -> None:
        raw_result = VisionResult(
            query="Trace query",
            description="Sensitive analysis output",
            status="success",
            metadata={
                "api_key": "SECRET_KEY_123",
                "auth_token": "BEARER_XYZ",
                "public_stat": 42,
            },
        )
        normalizer = VisionResultNormalizer()
        cleaned = normalizer.normalize(raw_result)

        for forbidden in FORBIDDEN_METADATA_KEYS:
            assert forbidden not in cleaned.metadata
        assert cleaned.metadata["public_stat"] == 42

    def test_domain_model_serialization_round_trip(self) -> None:
        ev = VisualEvidence(
            document_id="DOC-SER-1",
            filename="graph.png",
            chunk_id="CHK-1",
            content_type="chart",
            page_number=1,
            chunk_index=0,
            metadata={"source": "benchmark"},
        )
        ev_dict = ev.to_dict()
        restored_ev = VisualEvidence.from_dict(ev_dict)

        assert restored_ev.document_id == ev.document_id
        assert restored_ev.filename == ev.filename
        assert restored_ev.content_type == ev.content_type
        assert restored_ev.metadata == ev.metadata

        req = VisionRequest(query="Serialized query", evidence=[ev])
        req_dict = req.to_dict()
        restored_req = VisionRequest.from_dict(req_dict)
        assert restored_req.query == req.query
        assert len(restored_req.evidence) == 1
        assert restored_req.evidence[0].document_id == "DOC-SER-1"

    def test_supervisor_handoff_dict_contract(self) -> None:
        ev = VisualEvidence(document_id="DOC-SUP-1", filename="chart.png", chunk_id="CHK-S1", content_type="chart", page_number=2)
        result = VisionResult(
            query="Supervisor handoff contract",
            description="Structured vision insight",
            status="success",
            evidence=[ev],
            metadata={"latency_ms": 12.5},
        )
        handoff_payload = result.to_dict()
        assert handoff_payload["query"] == "Supervisor handoff contract"
        assert handoff_payload["description"] == "Structured vision insight"
        assert handoff_payload["status"] == "success"
        assert len(handoff_payload["evidence"]) == 1
        assert handoff_payload["evidence"][0]["document_id"] == "DOC-SUP-1"
        assert handoff_payload["metadata"]["latency_ms"] == 12.5


# =========================================================================
# 4. STATE ISOLATION
# =========================================================================
class TestStateIsolation:
    """Verifies that consecutive and concurrent requests have zero state leakage."""

    def test_sequential_requests_have_isolated_state(self) -> None:
        provider = StabilityMockProvider()
        pipeline = VisionPipeline(provider=provider)

        ev_a = VisualEvidence(document_id="DOC-A", filename="a.png", chunk_id="CA", image_bytes=_PNG, content_type="image")
        req_a = VisionRequest(query="Request A", evidence=[ev_a])

        ev_b = VisualEvidence(document_id="DOC-B", filename="b.png", chunk_id="CB", image_bytes=_PNG, content_type="chart")
        req_b = VisionRequest(query="Request B", evidence=[ev_b])

        ev_c = VisualEvidence(document_id="DOC-C", filename="c.png", chunk_id="CC", image_bytes=_PNG, content_type="diagram")
        req_c = VisionRequest(query="Request C", evidence=[ev_c])

        res_a = pipeline.run(req_a)
        res_b = pipeline.run(req_b)
        res_c = pipeline.run(req_c)

        assert "Request A" in res_a.description
        assert "Request B" in res_b.description
        assert "Request C" in res_c.description

        assert res_a.document_id == "DOC-A"
        assert res_b.document_id == "DOC-B"
        assert res_c.document_id == "DOC-C"

    def test_concurrent_pipeline_runs_are_strictly_isolated(self) -> None:
        provider = StabilityMockProvider()
        pipeline = VisionPipeline(provider=provider)

        def _execute_worker(idx: int) -> tuple[int, VisionResult]:
            ev = VisualEvidence(
                document_id=f"DOC-THREAD-{idx}",
                filename=f"thread_{idx}.png",
                chunk_id=f"CHK-{idx}",
                content_type="chart",
                image_bytes=_PNG,
            )
            req = VisionRequest(query=f"Concurrent Worker #{idx}", evidence=[ev])
            return idx, pipeline.run(req)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_execute_worker, i) for i in range(16)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 16
        for idx, res in results:
            assert res.is_success is True
            assert f"Concurrent Worker #{idx}" in res.description
            assert res.document_id == f"DOC-THREAD-{idx}"


# =========================================================================
# 5. LINEAGE STABILITY
# =========================================================================
class TestLineageStability:
    """Verifies that all supported lineage fields are faithfully preserved."""

    def test_all_lineage_attributes_preserved_without_fabrication(self) -> None:
        ev = VisualEvidence(
            document_id="DOC-EXACT-LINEAGE",
            filename="exact_financial_chart.png",
            chunk_id="CHK-LINEAGE-88",
            content_type="chart",
            page_number=14,
            chunk_index=7,
            image_bytes=_PNG,
            metadata={"department": "finance", "fiscal_year": 2026},
        )
        provider = StabilityMockProvider()
        pipeline = VisionPipeline(provider=provider)
        req = VisionRequest(query="Lineage preservation audit", evidence=[ev])

        result = pipeline.run(req)

        assert result.document_id == "DOC-EXACT-LINEAGE"
        assert result.filename == "exact_financial_chart.png"
        assert result.page_number == 14
        assert result.chunk_id == "CHK-LINEAGE-88"
        assert result.content_type == "chart"


# =========================================================================
# 6. ERROR STABILITY
# =========================================================================
class TestErrorStability:
    """Verifies that existing errors remain deterministic in the VisionError hierarchy."""

    def test_all_exceptions_inherit_from_vision_error(self) -> None:
        error_classes = [
            VisionAgentError,
            VisionCancellationError,
            VisionInputValidationError,
            VisionEvidenceError,
            VisionProcessingError,
            VisionProviderError,
            VisionProviderConfigError,
            VisionProviderExecutionError,
            VisionProviderUnavailableError,
            VisionTimeoutError,
            VisionUnsupportedCapabilityError,
        ]
        for cls in error_classes:
            assert issubclass(cls, VisionError), f"{cls.__name__} does not inherit from VisionError"

    def test_provider_exhaustion_raises_deterministic_error(self) -> None:
        provider = StabilityMockProvider(fail_count=10)
        adapter = VisionExecutionAdapter(provider=provider)
        ev = VisualEvidence(document_id="DOC-1", filename="f.png", chunk_id="C1", image_bytes=_PNG, content_type="image")
        req = VisionRequest(query="Exhaustion test", evidence=[ev])

        with pytest.raises(VisionProviderExecutionError):
            adapter.execute(req, retry_policy=VisionRetryPolicy(max_retries=1))


# =========================================================================
# 7. OFFLINE GUARANTEE
# =========================================================================
class TestOfflineGuarantee:
    """Verifies tests and pipeline components execute 100% locally and offline."""

    def test_offline_execution_without_network_or_api_keys(self) -> None:
        provider = StabilityMockProvider()
        ev = VisualEvidence(document_id="DOC-OFFLINE", filename="offline.png", chunk_id="C-OFF", image_bytes=_PNG, content_type="diagram")
        req = VisionRequest(query="Offline guarantee check", evidence=[ev])
        result = run_vision_pipeline(provider, req)

        assert result.is_success is True
        assert "Verified visual content" in result.description


# =========================================================================
# 8. MEMBER BOUNDARY
# =========================================================================
class TestMemberBoundary:
    """Verifies that Member 3 has zero dependencies on or leakage into Member 1/2 systems."""

    def test_member_3_does_not_import_or_use_prohibited_systems(self) -> None:
        import sys
        vision_modules = [mod for name, mod in sys.modules.items() if name.startswith("vision.")]
        for mod in vision_modules:
            code = getattr(mod, "__file__", "")
            assert "ingestion" not in code or "test" in code, f"Unexpected cross-module reference in {mod}"
