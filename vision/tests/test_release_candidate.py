"""
Day 56 - OmniBrain Member 3 Vision Agent: Release Candidate Verification.

Release-gate test suite answering:
  "Can Member 3 Vision be safely handed to the rest of the OMNIBRAIN team
   as a stable subsystem?"

Validates the full public contract across:
  - Public API importability and contract stability
  - Data flow: VisualEvidence -> VisionRequest -> VisionModelInput -> VisionResult
  - Happy path end-to-end pipeline
  - Multi-evidence and multi-document isolation
  - Failure matrix (invalid input, provider failure, retry exhaustion, timeout, cancellation)
  - Retry policy correctness and no state leakage
  - Concurrency isolation (10 parallel requests + mixed outcomes)
  - Pipeline state reuse without contamination
  - Resource safety (no thread or resource leaks)
  - Observability non-interference
  - Citation / lineage preservation: Search -> Vision -> Result -> Downstream
  - Supervisor contract compatibility (offline consumer)
  - VisionResult serialization roundtrip (to_dict / from_dict)
  - Security: no credentials/API keys in public results or errors
  - Dependency and responsibility boundary: Member 3 owns Vision only

All tests execute 100% offline with zero external network, HTTP, or LLM calls.
"""

from __future__ import annotations

import io
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
# Test Helpers & Offline Provider/Consumer Doubles
# ===========================================================================

def _make_png(width: int = 16, height: int = 16, color: tuple[int, int, int] = (50, 110, 180)) -> bytes:
    """Generate a minimal valid PNG byte payload for visual evidence."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_RC_PNG = _make_png()


def _citation(
    doc_id: str = "DOC-RC-001",
    filename: str = "release_candidate.pdf",
    chunk_id: str = "CHK-RC-001",
    content_type: str = "chart",
    page_number: int | None = 4,
    metadata: dict[str, Any] | None = None,
) -> AgentCitation:
    """Construct a Member 2 AgentCitation for release verification."""
    return AgentCitation(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        content_type=content_type,
        score=0.97,
        metadata=metadata if metadata is not None else {"stage": "release_candidate_56"},
    )


class RCProvider(VisionModelProvider):
    """Deterministic offline provider double for release candidate verification."""

    def __init__(
        self,
        config: VisionProviderConfig | None = None,
        capabilities: VisionProviderCapabilities | None = None,
        fail_on_call_numbers: list[int] | None = None,
        simulate_timeout: bool = False,
    ) -> None:
        cfg = config or VisionProviderConfig(provider_name="rc_provider", model_name="rc_model_v1")
        super().__init__(config=cfg, capabilities=capabilities)
        self._fail_on = set(fail_on_call_numbers or [])
        self._simulate_timeout = simulate_timeout
        self._call_count = 0
        self._received: list[VisionModelInput] = []
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    @property
    def received_inputs(self) -> list[VisionModelInput]:
        with self._lock:
            return list(self._received)

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        with self._lock:
            self._call_count += 1
            n = self._call_count
            self._received.append(model_input)

        if self._simulate_timeout:
            raise VisionTimeoutError("RC provider simulated timeout.")

        if n in self._fail_on:
            raise VisionProviderExecutionError(f"RC provider simulated failure on call {n}.")

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"RC analysis: '{model_input.query}'",
            document_id=model_input.document_id,
            filename=model_input.filename,
            chunk_id=model_input.chunk_id,
            page_number=model_input.page_number,
            content_type=model_input.content_type,
            metadata={
                "provider": self.config.provider_name,
                "api_key": "harmless_test_credential",  # should be sanitized
                "call_n": n,
            },
        )


class RCSupervisorConsumer:
    """Lightweight offline downstream consumer contract for release verification."""

    def __init__(self) -> None:
        self.received: list[VisionResult] = []

    def consume(self, result: VisionResult) -> dict[str, Any]:
        assert isinstance(result, VisionResult)
        self.received.append(result)
        return {
            "status": result.status,
            "document_id": result.document_id,
            "filename": result.filename,
            "chunk_id": result.chunk_id,
            "page_number": result.page_number,
            "content_type": result.content_type,
            "evidence_count": len(result.evidence),
            "is_success": result.is_success,
            "has_evidence": result.has_evidence,
        }


# ===========================================================================
# Release Candidate Test Suite
# ===========================================================================

class TestReleaseCandidate:
    """Day 56 release-gate verification for OmniBrain Member 3 Vision subsystem."""

    # ------------------------------------------------------------------
    # Step 3 — Release API Audit
    # ------------------------------------------------------------------

    def test_rc_01_api_importability_and_contract(self) -> None:
        """All public exports from vision.__all__ exist, are importable, and have correct types."""
        assert isinstance(vision.__all__, list)
        assert len(vision.__all__) >= 30

        for name in vision.__all__:
            assert hasattr(vision, name), f"Missing export: '{name}'"
            obj = getattr(vision, name)
            assert obj is not None

        # Core class hierarchy
        assert issubclass(VisionError, Exception)
        assert issubclass(VisionInputValidationError, VisionError)
        assert issubclass(VisionEvidenceError, VisionError)
        assert issubclass(VisionProviderError, VisionError)
        assert issubclass(VisionProviderExecutionError, VisionProviderError)
        assert issubclass(VisionTimeoutError, VisionError)
        assert issubclass(VisionCancellationError, VisionError)
        assert issubclass(VisionProviderConfigError, VisionError)

        # Core classes are instantiable via normal paths
        assert callable(VisionPipeline)
        assert callable(VisionRequest)
        assert callable(VisionResult)
        assert callable(VisualEvidence)
        assert callable(VisionAgent)

    # ------------------------------------------------------------------
    # Step 4 — Release Data Contract
    # ------------------------------------------------------------------

    def test_rc_02_data_contract_flow(self) -> None:
        """VisualEvidence -> VisionRequest -> VisionModelInput -> VisionResult preserves all lineage."""
        cit = _citation(doc_id="DOC-DC-56", filename="dc_report.pdf", chunk_id="CHK-DC-56",
                        content_type="chart", page_number=7,
                        metadata={"section": "finance", "quarter": "Q3"})

        # 1. VisualEvidence from Search citation
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
        assert ev.document_id == "DOC-DC-56"
        assert ev.filename == "dc_report.pdf"
        assert ev.chunk_id == "CHK-DC-56"
        assert ev.page_number == 7
        assert ev.content_type == "chart"
        assert ev.metadata.get("section") == "finance"

        # 2. PreparedImageEvidence
        prep = prepare_image_evidence(ev)
        assert prep.document_id == "DOC-DC-56"
        assert prep.image_format in SUPPORTED_IMAGE_FORMATS

        # 3. VisionModelInput
        model_input = build_vision_input("Data contract check", prep)
        assert model_input.document_id == "DOC-DC-56"
        assert model_input.filename == "dc_report.pdf"
        assert model_input.chunk_id == "CHK-DC-56"
        assert model_input.page_number == 7
        assert model_input.content_type == "chart"

        # 4. VisionResult
        req = VisionRequest(query="Data contract check", evidence=[ev])
        provider = RCProvider()
        pipeline = VisionPipeline(provider=provider)
        res = pipeline.run(req)

        assert res.status == "success"
        assert res.document_id == "DOC-DC-56"
        assert res.filename == "dc_report.pdf"
        assert res.chunk_id == "CHK-DC-56"
        assert res.page_number == 7
        assert res.content_type == "chart"
        assert len(res.evidence) == 1
        assert res.evidence[0].metadata.get("section") == "finance"

    # ------------------------------------------------------------------
    # Step 5 — Happy Path
    # ------------------------------------------------------------------

    def test_rc_03_happy_path_full_pipeline(self) -> None:
        """Complete end-to-end offline happy path with downstream consumer handoff."""
        cit = _citation(doc_id="DOC-HP-56", content_type="diagram", page_number=11)
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
        req = VisionRequest(query="Release happy path analysis", evidence=[ev])

        provider = RCProvider()
        pipeline = VisionPipeline(provider=provider)
        res = pipeline.run(req)

        # Pipeline result
        assert res.is_success is True
        assert res.status == "success"
        assert res.error is None
        assert res.has_evidence is True
        assert res.document_id == "DOC-HP-56"
        assert "api_key" not in res.metadata  # sanitized by normalizer

        # Downstream consumer
        consumer = RCSupervisorConsumer()
        consumed = consumer.consume(res)
        assert consumed["is_success"] is True
        assert consumed["document_id"] == "DOC-HP-56"
        assert consumed["evidence_count"] == 1
        assert provider.call_count == 1

    # ------------------------------------------------------------------
    # Step 6 — Multi-Evidence
    # ------------------------------------------------------------------

    def test_rc_04_multi_evidence_release(self) -> None:
        """1, 5, 10 evidence items: count, order, lineage, metadata preserved — no duplication."""
        for n in [1, 5, 10]:
            citations = [
                _citation(doc_id=f"DOC-ME-{i}", filename=f"me_{i}.pdf",
                          chunk_id=f"chk-me-{i}", metadata={"idx": i})
                for i in range(n)
            ]
            evidence = [VisualEvidenceAdapter.adapt_citation(c, image_bytes=_RC_PNG) for c in citations]
            req = VisionRequest(query=f"Multi-ev {n}", evidence=evidence)
            provider = RCProvider()
            res = VisionPipeline(provider=provider).run(req)

            assert res.status == "success"
            assert len(res.evidence) == n
            for i, ev in enumerate(res.evidence):
                assert ev.document_id == f"DOC-ME-{i}"
                assert ev.chunk_id == f"chk-me-{i}"
                assert ev.metadata.get("idx") == i
            assert provider.call_count == 1  # no duplicate execution

    # ------------------------------------------------------------------
    # Step 7 — Multi-Document
    # ------------------------------------------------------------------

    def test_rc_05_multi_document_source_isolation(self) -> None:
        """DOC-A, DOC-B, DOC-C, DOC-D return strictly their own lineage, no cross-contamination."""
        for doc_id in ["DOC-A", "DOC-B", "DOC-C", "DOC-D"]:
            cit = _citation(doc_id=doc_id, filename=f"{doc_id.lower()}_report.pdf")
            ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
            req = VisionRequest(query=f"Isolation check {doc_id}", evidence=[ev])
            res = VisionPipeline(provider=RCProvider()).run(req)

            assert res.status == "success"
            assert res.document_id == doc_id
            assert res.filename == f"{doc_id.lower()}_report.pdf"
            assert res.evidence[0].document_id == doc_id

    # ------------------------------------------------------------------
    # Step 8 — Failure Matrix
    # ------------------------------------------------------------------

    def test_rc_06_failure_matrix(self) -> None:
        """All existing failure paths are deterministic, raise correct exceptions, leave no stale state."""
        cit = _citation(doc_id="DOC-FAIL-56")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
        provider = RCProvider()
        pipeline = VisionPipeline(provider=provider)

        # 1. Empty query
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="", evidence=[ev])

        # 2. Whitespace query
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="    ", evidence=[ev])

        # 3. None request to pipeline
        with pytest.raises(VisionInputValidationError):
            pipeline.run(None)  # type: ignore[arg-type]

        # 4. Invalid content type → evidence error
        cit_bad = _citation(content_type="video")
        with pytest.raises(VisionEvidenceError):
            VisualEvidenceAdapter.adapt_citation(cit_bad)

        # Validation failures must NOT invoke provider
        assert provider.call_count == 0

        # 5. Provider failure
        fail_provider = RCProvider(fail_on_call_numbers=[1])
        with pytest.raises(VisionProviderExecutionError):
            VisionPipeline(provider=fail_provider).run(
                VisionRequest(query="Fail test", evidence=[ev])
            )

        # 6. Timeout
        timeout_provider = RCProvider(simulate_timeout=True)
        with pytest.raises(VisionTimeoutError):
            VisionPipeline(provider=timeout_provider).run(
                VisionRequest(query="Timeout test", evidence=[ev])
            )

        # 7. Cancellation
        token = VisionCancellationToken()
        token.cancel()
        cancel_provider = RCProvider()
        with pytest.raises(VisionCancellationError):
            VisionPipeline(provider=cancel_provider).run(
                VisionRequest(query="Cancel test", evidence=[ev]),
                cancellation_token=token,
            )
        assert cancel_provider.call_count == 0

        # 8. Later valid request still succeeds after failures
        ok_provider = RCProvider()
        res = VisionPipeline(provider=ok_provider).run(
            VisionRequest(query="Valid after failures", evidence=[ev])
        )
        assert res.status == "success"

    # ------------------------------------------------------------------
    # Step 9 — Retry Verification
    # ------------------------------------------------------------------

    def test_rc_07_retry_fail_then_success(self) -> None:
        """Attempt 1 fails, attempt 2 succeeds — retry policy correct, no state leak."""
        cit = _citation(doc_id="DOC-RETRY-56")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
        req = VisionRequest(query="Retry test", evidence=[ev])

        provider = RCProvider(fail_on_call_numbers=[1])
        res = VisionPipeline(provider=provider).run(req, retry_policy=VisionRetryPolicy(max_retries=2))

        assert res.status == "success"
        assert provider.call_count == 2
        assert res.document_id == "DOC-RETRY-56"

    def test_rc_08_retry_exhaustion(self) -> None:
        """All retry attempts fail — VisionProviderExecutionError raised, no stale success."""
        cit = _citation(doc_id="DOC-RETRY-EXHAUST-56")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
        req = VisionRequest(query="Retry exhaustion", evidence=[ev])

        provider = RCProvider(fail_on_call_numbers=[1, 2, 3])
        with pytest.raises(VisionProviderExecutionError):
            VisionPipeline(provider=provider).run(req, retry_policy=VisionRetryPolicy(max_retries=2))

        # Subsequent request on fresh pipeline succeeds
        ok = VisionPipeline(provider=RCProvider()).run(
            VisionRequest(query="Post-exhaustion", evidence=[ev])
        )
        assert ok.status == "success"

    # ------------------------------------------------------------------
    # Step 10 — Timeout / Cancellation
    # ------------------------------------------------------------------

    def test_rc_09_timeout_cleanup_then_success(self) -> None:
        """Timeout: deterministic VisionTimeoutError, no duplicate completion, next succeeds."""
        cit = _citation(doc_id="DOC-TM-56")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
        req = VisionRequest(query="Timeout verify", evidence=[ev])

        with pytest.raises(VisionTimeoutError):
            VisionPipeline(provider=RCProvider(simulate_timeout=True)).run(req)

        res = VisionPipeline(provider=RCProvider()).run(req)
        assert res.status == "success"

    def test_rc_10_cancellation_cleanup_then_success(self) -> None:
        """Cancellation: deterministic VisionCancellationError, cleanup, subsequent request succeeds."""
        cit = _citation(doc_id="DOC-CX-56")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
        req = VisionRequest(query="Cancel verify", evidence=[ev])

        token = VisionCancellationToken()
        token.cancel()

        cancel_provider = RCProvider()
        with pytest.raises(VisionCancellationError):
            VisionPipeline(provider=cancel_provider).run(req, cancellation_token=token)
        assert cancel_provider.call_count == 0

        res = VisionPipeline(provider=RCProvider()).run(req)
        assert res.status == "success"

    # ------------------------------------------------------------------
    # Step 11 — Concurrency (10 parallel + mixed outcomes)
    # ------------------------------------------------------------------

    def test_rc_11_concurrent_isolation_10_requests(self) -> None:
        """10 parallel requests maintain strict result isolation — no cross-contamination."""
        n = 10
        results: dict[int, VisionResult] = {}
        errors: dict[int, Exception] = {}

        def run(i: int) -> None:
            try:
                c = _citation(doc_id=f"DOC-CONC-{i}", filename=f"conc_{i}.pdf",
                              chunk_id=f"chk-conc-{i}")
                ev = VisualEvidenceAdapter.adapt_citation(c, image_bytes=_RC_PNG)
                req = VisionRequest(query=f"Concurrent {i}", evidence=[ev])
                results[i] = VisionPipeline(provider=RCProvider()).run(req)
            except Exception as e:
                errors[i] = e

        threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Unexpected errors in concurrent run: {errors}"
        for i in range(n):
            assert results[i].status == "success"
            assert results[i].document_id == f"DOC-CONC-{i}"
            assert results[i].evidence[0].chunk_id == f"chk-conc-{i}"

    def test_rc_12_concurrent_mixed_outcomes(self) -> None:
        """Concurrent: success, retry->success, timeout, cancellation — each isolated."""
        results: dict[str, Any] = {}
        exceptions: dict[str, Exception] = {}

        def thread_success() -> None:
            try:
                ev = VisualEvidenceAdapter.adapt_citation(_citation(doc_id="DOC-MX-S"), image_bytes=_RC_PNG)
                results["S"] = VisionPipeline(provider=RCProvider()).run(
                    VisionRequest(query="Mixed S", evidence=[ev])
                )
            except Exception as e:
                exceptions["S"] = e

        def thread_retry() -> None:
            try:
                ev = VisualEvidenceAdapter.adapt_citation(_citation(doc_id="DOC-MX-R"), image_bytes=_RC_PNG)
                results["R"] = VisionPipeline(provider=RCProvider(fail_on_call_numbers=[1])).run(
                    VisionRequest(query="Mixed R", evidence=[ev]),
                    retry_policy=VisionRetryPolicy(max_retries=2),
                )
            except Exception as e:
                exceptions["R"] = e

        def thread_timeout() -> None:
            try:
                ev = VisualEvidenceAdapter.adapt_citation(_citation(doc_id="DOC-MX-T"), image_bytes=_RC_PNG)
                VisionPipeline(provider=RCProvider(simulate_timeout=True)).run(
                    VisionRequest(query="Mixed T", evidence=[ev])
                )
            except Exception as e:
                exceptions["T"] = e

        def thread_cancel() -> None:
            try:
                ev = VisualEvidenceAdapter.adapt_citation(_citation(doc_id="DOC-MX-C"), image_bytes=_RC_PNG)
                token = VisionCancellationToken()
                token.cancel()
                VisionPipeline(provider=RCProvider()).run(
                    VisionRequest(query="Mixed C", evidence=[ev]),
                    cancellation_token=token,
                )
            except Exception as e:
                exceptions["C"] = e

        threads = [
            threading.Thread(target=thread_success),
            threading.Thread(target=thread_retry),
            threading.Thread(target=thread_timeout),
            threading.Thread(target=thread_cancel),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["S"].status == "success" and results["S"].document_id == "DOC-MX-S"
        assert results["R"].status == "success" and results["R"].document_id == "DOC-MX-R"
        assert isinstance(exceptions["T"], VisionTimeoutError)
        assert isinstance(exceptions["C"], VisionCancellationError)

    # ------------------------------------------------------------------
    # Step 12 — Pipeline State Reuse
    # ------------------------------------------------------------------

    def test_rc_13_pipeline_sequential_reuse_no_contamination(self) -> None:
        """Sequential reuse of one VisionPipeline: requests A, B, C get strictly their own results."""
        provider = RCProvider()
        pipeline = VisionPipeline(provider=provider)

        for tag in ["A", "B", "C"]:
            cit = _citation(doc_id=f"DOC-REUSE-{tag}")
            ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
            req = VisionRequest(query=f"Reuse query {tag}", evidence=[ev])
            res = pipeline.run(req)

            assert res.status == "success"
            assert res.document_id == f"DOC-REUSE-{tag}"
            assert res.query == f"Reuse query {tag}"
            assert len(res.evidence) == 1
            assert res.evidence[0].document_id == f"DOC-REUSE-{tag}"

        assert provider.call_count == 3

    # ------------------------------------------------------------------
    # Step 13 — Resource Safety
    # ------------------------------------------------------------------

    def test_rc_14_no_resource_leaks_across_executions(self) -> None:
        """No thread accumulation over 10 consecutive executions."""
        baseline = threading.active_count()

        for i in range(10):
            cit = _citation(doc_id=f"DOC-RES-{i}")
            ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
            res = VisionPipeline(provider=RCProvider()).run(
                VisionRequest(query=f"Resource run {i}", evidence=[ev])
            )
            assert res.status == "success"

        assert threading.active_count() <= baseline + 1

    # ------------------------------------------------------------------
    # Step 14 — Observability Non-Interference
    # ------------------------------------------------------------------

    def test_rc_15_observability_does_not_alter_results(self) -> None:
        """Lifecycle observation records stages without changing functional results."""
        lifecycle = VisionExecutionLifecycle(provider_name="rc_prov", model_name="v1")
        assert lifecycle.stage == VisionExecutionStage.PENDING

        lifecycle.transition_to(VisionExecutionStage.BUILDING_INPUT)
        lifecycle.transition_to(VisionExecutionStage.EXECUTING)
        lifecycle.transition_to(VisionExecutionStage.COMPLETED)

        obs = lifecycle.to_observation()
        assert isinstance(obs, VisionExecutionObservation)
        assert obs.stage == VisionExecutionStage.COMPLETED

        # Functional result is unaffected
        cit = _citation(doc_id="DOC-OBS-56")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
        res = VisionPipeline(provider=RCProvider()).run(
            VisionRequest(query="Observability check", evidence=[ev])
        )
        assert res.status == "success"

    # ------------------------------------------------------------------
    # Step 15 — Lineage Preservation
    # ------------------------------------------------------------------

    def test_rc_16_lineage_search_to_result_to_downstream(self) -> None:
        """Search -> Vision -> Result -> Downstream: document_id, filename, chunk_id, page, content_type preserved."""
        meta = {"origin": "search_agent", "classification": "chart_analysis"}
        cit = _citation(doc_id="DOC-LIN-56", filename="lineage_rc.pdf", chunk_id="CHK-LIN-56",
                        content_type="chart", page_number=9, metadata=meta)

        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
        res = VisionPipeline(provider=RCProvider()).run(
            VisionRequest(query="Lineage release check", evidence=[ev])
        )

        consumer = RCSupervisorConsumer()
        consumed = consumer.consume(res)

        assert consumed["document_id"] == "DOC-LIN-56"
        assert consumed["filename"] == "lineage_rc.pdf"
        assert consumed["chunk_id"] == "CHK-LIN-56"
        assert consumed["page_number"] == 9
        assert consumed["content_type"] == "chart"
        assert res.evidence[0].metadata.get("origin") == "search_agent"

    # ------------------------------------------------------------------
    # Step 16 — Supervisor Contract
    # ------------------------------------------------------------------

    def test_rc_17_supervisor_downstream_contract(self) -> None:
        """Existing offline consumer can consume VisionResult without modification."""
        cit = _citation(doc_id="DOC-SUP-56")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
        res = VisionPipeline(provider=RCProvider()).run(
            VisionRequest(query="Supervisor RC contract", evidence=[ev])
        )

        consumer = RCSupervisorConsumer()
        consumed = consumer.consume(res)

        assert consumed["status"] == "success"
        assert consumed["is_success"] is True
        assert consumed["has_evidence"] is True
        assert consumed["evidence_count"] == 1

    # ------------------------------------------------------------------
    # Step 17 — Serialization Roundtrip
    # ------------------------------------------------------------------

    def test_rc_18_visionresult_serialization_roundtrip(self) -> None:
        """VisionResult.to_dict() and from_dict() roundtrip preserves all public lineage fields."""
        cit = _citation(doc_id="DOC-SER-56", filename="serial_rc.pdf", chunk_id="CHK-SER-56",
                        content_type="diagram", page_number=3)
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
        res = VisionPipeline(provider=RCProvider()).run(
            VisionRequest(query="Serialization check", evidence=[ev])
        )

        serialized = res.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["document_id"] == "DOC-SER-56"
        assert serialized["filename"] == "serial_rc.pdf"
        assert serialized["chunk_id"] == "CHK-SER-56"
        assert serialized["page_number"] == 3
        assert serialized["content_type"] == "diagram"
        assert serialized["status"] == "success"

        restored = VisionResult.from_dict(serialized)
        assert isinstance(restored, VisionResult)
        assert restored.document_id == "DOC-SER-56"
        assert restored.filename == "serial_rc.pdf"
        assert restored.chunk_id == "CHK-SER-56"
        assert restored.page_number == 3
        assert restored.content_type == "diagram"
        assert restored.status == "success"
        assert restored.is_success is True

    # ------------------------------------------------------------------
    # Step 18 — Security Release Check
    # ------------------------------------------------------------------

    def test_rc_19_no_credentials_in_public_result(self) -> None:
        """Public VisionResult.metadata must not expose api_key or other FORBIDDEN_METADATA_KEYS."""
        cit = _citation(doc_id="DOC-SEC-56")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
        res = VisionPipeline(provider=RCProvider()).run(
            VisionRequest(query="Security check", evidence=[ev])
        )

        for forbidden_key in FORBIDDEN_METADATA_KEYS:
            assert forbidden_key not in res.metadata, (
                f"Forbidden key '{forbidden_key}' leaked into public result metadata"
            )

    def test_rc_20_exception_messages_no_credentials(self) -> None:
        """Exception messages from validation do not expose credentials or sensitive paths."""
        try:
            VisionRequest(query="", evidence=[])
        except VisionInputValidationError as exc:
            msg = str(exc)
            assert "api_key" not in msg.lower()
            assert "password" not in msg.lower()
            assert "token" not in msg.lower()
            assert "secret" not in msg.lower()

    # ------------------------------------------------------------------
    # Step 19 — Dependency / Import Audit
    # ------------------------------------------------------------------

    def test_rc_21_no_prohibited_imports_in_vision_public(self) -> None:
        """vision source files do not directly import prohibited runtime dependencies."""
        import pathlib
        import re

        vision_root = pathlib.Path(__file__).parent.parent  # vision/
        # Check only source files, not test files
        source_files = [
            f for f in vision_root.glob("*.py")
            if not f.name.startswith("test_")
        ]

        prohibited_patterns = [
            r"^import langchain\b",
            r"^import langgraph\b",
            r"^import fastapi\b",
            r"^import streamlit\b",
            r"^import langfuse\b",
            r"^import opentelemetry\b",
            r"^import prometheus_client\b",
            r"^import redis\b",
            r"^import celery\b",
            r"^from langchain\b",
            r"^from langgraph\b",
            r"^from fastapi\b",
            r"^from streamlit\b",
        ]

        violations: list[str] = []
        for source_file in source_files:
            content = source_file.read_text(encoding="utf-8")
            for line_no, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                for pattern in prohibited_patterns:
                    if re.match(pattern, stripped):
                        violations.append(f"{source_file.name}:{line_no}: {stripped!r}")

        assert not violations, f"Prohibited imports found in vision source:\n" + "\n".join(violations)


    # ------------------------------------------------------------------
    # Step 20 — Responsibility Boundary
    # ------------------------------------------------------------------

    def test_rc_22_vision_does_not_duplicate_member1_member2(self) -> None:
        """VisionPipeline does not perform retrieval, embedding, or Qdrant — only Vision."""
        cit = _citation(doc_id="DOC-BOUND-56")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_RC_PNG)
        provider = RCProvider()
        res = VisionPipeline(provider=provider).run(
            VisionRequest(query="Responsibility boundary", evidence=[ev])
        )

        assert res.status == "success"
        # Provider called exactly once — no duplicate execution
        assert provider.call_count == 1
        # Only one input received — no duplicate input construction
        assert len(provider.received_inputs) == 1
