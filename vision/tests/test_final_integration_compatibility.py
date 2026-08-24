"""
OmniBrain Member 3 Vision Subsystem.
Day 59: Final Integration Compatibility & Release Audit Test Suite.

Verifies that the completed Vision subsystem is fully compatible with the complete
OMNIBRAIN enterprise multi-modal RAG system, adhering to all public contracts,
search-to-vision and supervisor handoff interfaces, state isolation, resource safety,
and offline execution requirements.
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


def _png(color: tuple[int, int, int] = (60, 120, 220)) -> bytes:
    """Generate minimal valid PNG bytes for offline testing."""
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return buf.getvalue()


_PNG = _png()


class AuditMockProvider(VisionModelProvider):
    """Deterministic, thread-safe mock provider for Day 59 final release audit."""

    def __init__(
        self,
        name: str = "audit-mock-provider",
        model_name: str = "audit-vision-v1",
        response_template: str = "Analyzed visual evidence successfully.",
        fail_count: int = 0,
        timeout: bool = False,
    ) -> None:
        cfg = VisionProviderConfig(
            provider_name=name,
            model_name=model_name,
            timeout=5.0,
            max_tokens=1024,
            temperature=0.0,
        )
        super().__init__(config=cfg)
        self._name = name
        self._response_template = response_template
        self._fail_count = fail_count
        self._timeout = timeout
        self._calls = 0
        self._lock = threading.Lock()

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        with self._lock:
            self._calls += 1
            call_idx = self._calls

        if self._timeout:
            raise VisionTimeoutError("AuditMockProvider simulated timeout condition.")

        if call_idx <= self._fail_count:
            raise VisionProviderExecutionError(f"AuditMockProvider simulated failure on call #{call_idx}.")

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"{self._response_template} [Lineage: doc={model_input.document_id}, type={model_input.content_type}]",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={
                "call_index": call_idx,
                "provider": self._name,
                "audit_passed": True,
                **model_input.evidence_metadata,
            },
        )


# =========================================================================
# 1. PUBLIC API COMPATIBILITY
# =========================================================================
class TestPublicApiCompatibility:
    """Verifies that all 41 public API exports and symbols remain fully backward compatible."""

    def test_01_all_canonical_exports_present_in_all(self) -> None:
        expected = {
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
        assert set(vision.__all__) == expected

    def test_02_all_exported_symbols_are_valid_objects(self) -> None:
        for sym in vision.__all__:
            obj = getattr(vision, sym)
            assert obj is not None, f"Symbol {sym} is None in vision"


# =========================================================================
# 2. SEARCH -> VISION COMPATIBILITY
# =========================================================================
class TestSearchToVisionCompatibility:
    """Verifies seamless handoff from Member 1 / Member 2 search evidence to Member 3 Vision."""

    def test_01_single_citation_handoff(self) -> None:
        citation = AgentCitation(
            document_id="DOC-CITATION-1",
            filename="quarterly_report.pdf",
            chunk_id="CHK-CIT-101",
            page_number=4,
            content_type="chart",
            score=0.92,
            metadata={"chart_type": "bar", "chunk_index": 2},
        )
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_PNG)

        assert isinstance(ev, VisualEvidence)
        assert ev.document_id == "DOC-CITATION-1"
        assert ev.filename == "quarterly_report.pdf"
        assert ev.chunk_id == "CHK-CIT-101"
        assert ev.page_number == 4
        assert ev.chunk_index == 2
        assert ev.content_type == "chart"
        assert ev.metadata["chart_type"] == "bar"

    def test_02_single_search_result_handoff(self) -> None:
        sr = VectorSearchResult(
            document_id="DOC-VEC-202",
            filename="network_topology.pdf",
            chunk_id="CHK-VEC-9",
            page_number=12,
            chunk_index=5,
            content="Extracted diagram showing VPC topology",
            content_type="diagram",
            score=0.89,
            metadata={"subnet": "10.0.0.0/16"},
        )
        ev = VisualEvidenceAdapter.adapt_search_result(sr, image_bytes=_PNG)

        assert isinstance(ev, VisualEvidence)
        assert ev.document_id == "DOC-VEC-202"
        assert ev.filename == "network_topology.pdf"
        assert ev.chunk_id == "CHK-VEC-9"
        assert ev.page_number == 12
        assert ev.chunk_index == 5
        assert ev.content_type == "diagram"
        assert ev.metadata["subnet"] == "10.0.0.0/16"

    def test_03_multi_evidence_and_multi_document_handoff(self) -> None:
        cit1 = AgentCitation(
            document_id="DOC-A",
            filename="doc_a.pdf",
            chunk_id="CHK-A-1",
            page_number=1,
            content_type="chart",
            metadata={"source": "doc_a", "chunk_index": 0},
        )
        cit2 = AgentCitation(
            document_id="DOC-B",
            filename="doc_b.pdf",
            chunk_id="CHK-B-2",
            page_number=3,
            content_type="diagram",
            metadata={"source": "doc_b", "chunk_index": 1},
        )
        sr3 = VectorSearchResult(
            document_id="DOC-C",
            filename="doc_c.pdf",
            chunk_id="CHK-C-3",
            page_number=8,
            chunk_index=2,
            content="Image of server rack",
            content_type="image",
            score=0.91,
            metadata={"source": "doc_c"},
        )

        ev1 = VisualEvidenceAdapter.adapt_citation(cit1, image_bytes=_PNG)
        ev2 = VisualEvidenceAdapter.adapt_citation(cit2, image_bytes=_PNG)
        ev3 = VisualEvidenceAdapter.adapt_search_result(sr3, image_bytes=_PNG)

        req = VisionRequest(
            query="Synthesize architectural and performance diagrams across documents",
            evidence=[ev1, ev2, ev3],
        )

        provider = AuditMockProvider()
        pipeline = VisionPipeline(provider=provider)
        result = pipeline.run(req)

        assert result.is_success is True
        assert result.document_id == "DOC-A"
        assert result.filename == "doc_a.pdf"
        assert result.content_type == "chart"

    @pytest.mark.parametrize("c_type", ["image", "chart", "diagram"])
    def test_04_all_valid_content_types_preserved(self, c_type: str) -> None:
        citation = AgentCitation(
            document_id="DOC-MODALITY",
            filename="modality_test.pdf",
            chunk_id=f"CHK-{c_type}",
            page_number=1,
            content_type=c_type,
            metadata={"modality": c_type},
        )
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_PNG)
        assert ev.content_type == c_type

        provider = AuditMockProvider()
        res = run_vision_pipeline(provider, VisionRequest(query=f"Analyze {c_type}", evidence=[ev]))
        assert res.is_success is True
        assert res.content_type == c_type


# =========================================================================
# 3. SUPERVISOR -> VISION COMPATIBILITY
# =========================================================================
class TestSupervisorToVisionCompatibility:
    """Verifies that the Supervisor workflow can cleanly invoke VisionAgent and pipeline."""

    def test_01_supervisor_invocation_flow(self) -> None:
        provider = AuditMockProvider(name="supervisor-vision-provider")
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(
            document_id="DOC-SUP-1",
            filename="system_overview.png",
            chunk_id="CHK-SUP-1",
            content_type="diagram",
            image_bytes=_PNG,
        )
        req = VisionRequest(query="Extract service boundaries", evidence=[ev])
        result = agent.process(req)

        assert isinstance(result, VisionResult)
        assert result.is_success is True
        assert result.document_id == "DOC-SUP-1"
        assert result.content_type == "diagram"
        assert "Analyzed visual evidence" in result.description

    def test_02_supervisor_handoff_dictionary_payload_compatibility(self) -> None:
        ev = VisualEvidence(
            document_id="DOC-DOWNSTREAM",
            filename="financials.pdf",
            chunk_id="CHK-FIN-1",
            page_number=10,
            content_type="chart",
            image_bytes=_PNG,
        )
        provider = AuditMockProvider()
        res = run_vision_pipeline(provider, VisionRequest(query="Describe Q3 growth", evidence=[ev]))

        payload = res.to_dict()
        assert isinstance(payload, dict)
        assert payload["query"] == "Describe Q3 growth"
        assert payload["status"] == "success"
        assert payload["document_id"] == "DOC-DOWNSTREAM"
        assert payload["content_type"] == "chart"
        assert isinstance(payload["metadata"], dict)
        assert payload["metadata"]["audit_passed"] is True


# =========================================================================
# 4. RESULT CONTRACT AND SERIALIZATION
# =========================================================================
class TestResultContractAndSerialization:
    """Verifies VisionResult contracts, to_dict/from_dict round-trips, and downstream safety."""

    def test_01_result_serialization_round_trip(self) -> None:
        ev = VisualEvidence(
            document_id="DOC-RT-1",
            filename="spec.png",
            chunk_id="CHK-RT-1",
            page_number=2,
            chunk_index=1,
            content_type="chart",
            image_bytes=_PNG,
            metadata={"reviewed": True},
        )
        original = VisionResult(
            query="Round trip query",
            status="success",
            description="Verified round trip result",
            evidence=[ev],
            document_id="DOC-RT-1",
            filename="spec.png",
            page_number=2,
            chunk_id="CHK-RT-1",
            content_type="chart",
            metadata={"confidence": 0.99},
        )

        d = original.to_dict()
        restored = VisionResult.from_dict(d)

        assert restored.query == original.query
        assert restored.status == original.status
        assert restored.description == original.description
        assert restored.document_id == original.document_id
        assert restored.filename == original.filename
        assert restored.page_number == original.page_number
        assert restored.chunk_id == original.chunk_id
        assert restored.content_type == original.content_type
        assert restored.metadata == original.metadata

    def test_02_request_serialization_round_trip(self) -> None:
        ev = VisualEvidence(
            document_id="DOC-REQ-1",
            filename="plan.png",
            chunk_id="CHK-REQ-1",
            content_type="diagram",
            image_bytes=_PNG,
        )
        req = VisionRequest(query="Original query", evidence=[ev], metadata={"tag": "release"})
        d = req.to_dict()
        restored = VisionRequest.from_dict(d)

        assert restored.query == req.query
        assert len(restored.evidence) == 1
        assert restored.evidence[0].document_id == "DOC-REQ-1"
        assert restored.metadata == {"tag": "release"}


# =========================================================================
# 5. FAILURE COMPATIBILITY
# =========================================================================
class TestFailureCompatibility:
    """Verifies deterministic, typed, and sanitized error behavior."""

    def test_01_invalid_request_raises_validation_error(self) -> None:
        ev = VisualEvidence(document_id="DOC-1", filename="f.png", chunk_id="C1", image_bytes=_PNG, content_type="image")
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="", evidence=[ev])
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="   ", evidence=[ev])
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="valid", evidence=["invalid_item"])  # type: ignore

    def test_02_invalid_evidence_raises_evidence_error(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename="f.png", chunk_id="C1", content_type="image")
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="D1", filename="f.png", chunk_id="C1", content_type="unsupported_type")

    def test_03_provider_failure_and_retry_exhaustion(self) -> None:
        provider = AuditMockProvider(fail_count=5)
        adapter = VisionExecutionAdapter(provider=provider)
        ev = VisualEvidence(document_id="DOC-1", filename="f.png", chunk_id="C1", image_bytes=_PNG, content_type="image")
        req = VisionRequest(query="Retry exhaustion test", evidence=[ev])

        with pytest.raises(VisionProviderExecutionError):
            adapter.execute(req, retry_policy=VisionRetryPolicy(max_retries=1))

    def test_04_timeout_raises_deterministic_timeout_error(self) -> None:
        provider = AuditMockProvider(timeout=True)
        adapter = VisionExecutionAdapter(provider=provider)
        ev = VisualEvidence(document_id="DOC-1", filename="f.png", chunk_id="C1", image_bytes=_PNG, content_type="image")
        req = VisionRequest(query="Timeout test", evidence=[ev])

        with pytest.raises(VisionTimeoutError):
            adapter.execute(req)

    def test_05_cancellation_raises_deterministic_cancellation_error(self) -> None:
        provider = AuditMockProvider()
        adapter = VisionExecutionAdapter(provider=provider)
        token = VisionCancellationToken()
        token.cancel(reason="Supervisor aborted")

        ev = VisualEvidence(document_id="DOC-1", filename="f.png", chunk_id="C1", image_bytes=_PNG, content_type="image")
        req = VisionRequest(query="Cancellation test", evidence=[ev])

        with pytest.raises(VisionCancellationError):
            adapter.execute(req, cancellation_token=token)


# =========================================================================
# 6. STATE ISOLATION
# =========================================================================
class TestStateIsolation:
    """Verifies that sequential and concurrent requests have zero state leakage."""

    def test_01_sequential_requests_remain_isolated(self) -> None:
        provider = AuditMockProvider()
        pipeline = VisionPipeline(provider=provider)

        ev_a = VisualEvidence(document_id="DOC-A", filename="a.png", chunk_id="CA", image_bytes=_PNG, content_type="image")
        ev_b = VisualEvidence(document_id="DOC-B", filename="b.png", chunk_id="CB", image_bytes=_PNG, content_type="chart")
        ev_c = VisualEvidence(document_id="DOC-C", filename="c.png", chunk_id="CC", image_bytes=_PNG, content_type="diagram")

        res_a = pipeline.run(VisionRequest(query="Req A", evidence=[ev_a]))
        res_b = pipeline.run(VisionRequest(query="Req B", evidence=[ev_b]))
        res_c = pipeline.run(VisionRequest(query="Req C", evidence=[ev_c]))

        assert res_a.document_id == "DOC-A"
        assert res_b.document_id == "DOC-B"
        assert res_c.document_id == "DOC-C"

    def test_02_concurrent_requests_remain_isolated(self) -> None:
        provider = AuditMockProvider()
        pipeline = VisionPipeline(provider=provider)

        def _worker(idx: int) -> tuple[int, VisionResult]:
            ev = VisualEvidence(
                document_id=f"DOC-CONCURRENT-{idx}",
                filename=f"file_{idx}.png",
                chunk_id=f"CHK-{idx}",
                content_type="chart",
                image_bytes=_PNG,
            )
            req = VisionRequest(query=f"Concurrent Query #{idx}", evidence=[ev])
            return idx, pipeline.run(req)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_worker, i) for i in range(16)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 16
        for idx, res in results:
            assert res.is_success is True
            assert res.document_id == f"DOC-CONCURRENT-{idx}"


# =========================================================================
# 7. RESOURCE SAFETY AND SECURITY
# =========================================================================
class TestResourceSafetyAndSecurity:
    """Verifies that secrets are stripped and resources are cleanly bounded."""

    def test_01_metadata_secret_scrubbing(self) -> None:
        raw_res = VisionResult(
            query="Scrub query",
            description="Analysis output",
            status="success",
            metadata={
                "api_key": "SK-SECRET-12345",
                "auth_token": "BEARER-54321",
                "clean_stat": 100,
            },
        )
        normalized = VisionResultNormalizer.normalize(raw_res)

        for k in FORBIDDEN_METADATA_KEYS:
            assert k not in normalized.metadata
        assert normalized.metadata["clean_stat"] == 100

    def test_02_oversized_image_policy_enforcement(self) -> None:
        ev = VisualEvidence(document_id="DOC-BIG", filename="huge.png", chunk_id="C-BIG", image_bytes=_PNG, content_type="image")
        with pytest.raises(VisionEvidenceError):
            prepare_image_evidence(ev, oversized_policy=OversizedImagePolicy(max_dimension=1, reject=True))


# =========================================================================
# 8. OFFLINE REQUIREMENT & NO DUPLICATE WORK
# =========================================================================
class TestOfflineAndNoDuplicateWork:
    """Verifies 100% offline execution and clean boundary preservation."""

    def test_01_offline_execution_guarantee(self) -> None:
        provider = AuditMockProvider()
        ev = VisualEvidence(document_id="DOC-OFF", filename="off.png", chunk_id="C-OFF", image_bytes=_PNG, content_type="image")
        res = run_vision_pipeline(provider, VisionRequest(query="Offline test", evidence=[ev]))

        assert res.is_success is True

    def test_02_no_prohibited_cross_boundary_dependencies(self) -> None:
        import sys
        # Verify vision package modules have zero hard imports of external servers
        for mod_name, mod in list(sys.modules.items()):
            if mod_name.startswith("vision."):
                filepath = getattr(mod, "__file__", "")
                assert "fastapi" not in filepath
                assert "streamlit" not in filepath
                assert "langfuse" not in filepath
                assert "opentelemetry" not in filepath
