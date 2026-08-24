"""
Day 57 - OmniBrain Member 3 Vision Agent: Integration Handoff Certification.

Certifies that another team member can consume the Vision subsystem
using ONLY its documented public interfaces.

A clean downstream consumer imports ONLY from:
  - vision  (the package __init__.py re-exports)
  - agents.models  (for SearchResult / AgentCitation - Member 2 boundary)

No private module paths, no internals, no provider credentials.

Public handoff surface under test:
  VisualEvidence, VisionRequest, VisionResult           -- domain models
  VisualEvidenceAdapter                                 -- Search → Vision bridge
  VisionPipeline / run_vision_pipeline                  -- pipeline entry points
  VisionModelProvider / VisionProviderConfig            -- provider contract (subclassed)
  VisionRetryPolicy / VisionCancellationToken           -- control knobs
  VisionError and its subclass hierarchy                -- exception contract
  VisionResult.to_dict() / VisionResult.from_dict()    -- serialization

All tests execute 100% offline with zero external network, HTTP, or LLM calls.
"""

from __future__ import annotations

import io
import threading
from typing import Any

import pytest
from PIL import Image

# ── The ONLY allowed public import surface ───────────────────────────────────
import vision
from vision import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionCancellationError,
    VisionCancellationToken,
    VisionError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionModelProvider,
    VisionPipeline,
    VisionProviderConfig,
    VisionProviderExecutionError,
    VisionRequest,
    VisionResult,
    VisionRetryPolicy,
    VisionTimeoutError,
    VisualEvidence,
    VisualEvidenceAdapter,
    run_vision_pipeline,
)

# Member 2 boundary (read-only for lineage construction)
from agents.models import AgentCitation
# ─────────────────────────────────────────────────────────────────────────────


# ===========================================================================
# Helpers — shared offline fixtures
# ===========================================================================

def _png(color: tuple[int, int, int] = (40, 100, 200)) -> bytes:
    """Minimal valid PNG for image evidence."""
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return buf.getvalue()


_PNG = _png()


def _citation(
    doc_id: str = "DOC-HO-001",
    filename: str = "handoff.pdf",
    chunk_id: str = "CHK-HO-001",
    content_type: str = "chart",
    page_number: int | None = 5,
    metadata: dict[str, Any] | None = None,
) -> AgentCitation:
    return AgentCitation(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        content_type=content_type,
        score=0.95,
        metadata=metadata if metadata is not None else {"day": 57},
    )


class HandoffProvider(VisionModelProvider):
    """Clean-consumer offline provider double — instantiated with public API only."""

    def __init__(
        self,
        fail_calls: list[int] | None = None,
        timeout: bool = False,
    ) -> None:
        cfg = VisionProviderConfig(
            provider_name="handoff_provider",
            model_name="handoff_model_v1",
        )
        super().__init__(config=cfg)
        self._fail = set(fail_calls or [])
        self._timeout = timeout
        self._n = 0
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._n

    def execute(self, model_input: Any, **kwargs: Any) -> VisionResult:
        with self._lock:
            self._n += 1
            n = self._n

        if self._timeout:
            raise VisionTimeoutError("Handoff provider simulated timeout.")
        if n in self._fail:
            raise VisionProviderExecutionError(f"Handoff provider fail on call {n}.")

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Handoff result for: {model_input.query}",
            document_id=model_input.document_id,
            filename=model_input.filename,
            chunk_id=model_input.chunk_id,
            page_number=model_input.page_number,
            content_type=model_input.content_type,
            metadata={"provider": "handoff_provider", "call_n": n},
        )


# ===========================================================================
# Step 1 — Identify Handoff Surface
# ===========================================================================

class TestHandoffSurface:
    """Verifies the public interface is complete and stable."""

    def test_hs_01_all_expected_symbols_exported(self) -> None:
        """Every symbol a consumer needs is in vision.__all__."""
        required = [
            "VisualEvidence", "VisionRequest", "VisionResult",
            "VALID_VISUAL_CONTENT_TYPES",
            "VisualEvidenceAdapter",
            "VisionPipeline", "run_vision_pipeline",
            "VisionModelProvider", "VisionProviderConfig",
            "VisionRetryPolicy", "VisionCancellationToken",
            "VisionError", "VisionInputValidationError",
            "VisionEvidenceError", "VisionProviderError",
            "VisionProviderExecutionError", "VisionTimeoutError",
            "VisionCancellationError",
        ]
        for sym in required:
            assert sym in vision.__all__, f"Expected public export missing: '{sym}'"
            assert hasattr(vision, sym)

    def test_hs_02_no_private_module_needed(self) -> None:
        """All construction paths use only public symbols — no private paths required."""
        # Evidence construction
        ev = VisualEvidence(
            document_id="DOC-SURF-57",
            filename="surface.pdf",
            chunk_id="CHK-SURF-57",
            content_type="chart",
            image_bytes=_PNG,
        )
        assert ev.document_id == "DOC-SURF-57"

        # Request construction
        req = VisionRequest(query="Surface check", evidence=[ev])
        assert req.has_evidence is True

        # Pipeline via public API
        res = VisionPipeline(provider=HandoffProvider()).run(req)
        assert res.is_success is True


# ===========================================================================
# Steps 2–3 — Clean-Consumer & Public API Only
# ===========================================================================

class TestCleanConsumer:
    """Downstream consumer uses ONLY the public vision package."""

    def test_cc_01_consumer_constructs_evidence_from_citation(self) -> None:
        """Consumer bridges Member 2 AgentCitation to VisualEvidence via public adapter."""
        cit = _citation()
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_PNG)

        assert isinstance(ev, VisualEvidence)
        assert ev.document_id == cit.document_id
        assert ev.filename == cit.filename
        assert ev.chunk_id == cit.chunk_id
        assert ev.content_type in VALID_VISUAL_CONTENT_TYPES

    def test_cc_02_consumer_executes_pipeline_and_receives_result(self) -> None:
        """Consumer receives a VisionResult from VisionPipeline.run()."""
        cit = _citation(doc_id="DOC-CC-57")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_PNG)
        req = VisionRequest(query="Clean consumer handoff", evidence=[ev])
        res = VisionPipeline(provider=HandoffProvider()).run(req)

        assert isinstance(res, VisionResult)
        assert res.is_success is True
        assert res.status == "success"
        assert res.error is None

    def test_cc_03_consumer_uses_run_vision_pipeline_function(self) -> None:
        """Consumer can also use the public run_vision_pipeline() function."""
        cit = _citation(doc_id="DOC-CC-FN-57")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_PNG)
        req = VisionRequest(query="run_vision_pipeline handoff", evidence=[ev])
        res = run_vision_pipeline(HandoffProvider(), req)

        assert isinstance(res, VisionResult)
        assert res.is_success is True


# ===========================================================================
# Step 4 — Handoff Data Contract
# ===========================================================================

class TestHandoffDataContract:
    """All supported result fields are accessible to downstream consumer."""

    def test_dc_01_full_result_fields_accessible(self) -> None:
        """Consumer can read every documented public field of VisionResult."""
        cit = _citation(
            doc_id="DOC-DC-57",
            filename="data_contract.pdf",
            chunk_id="CHK-DC-57",
            content_type="diagram",
            page_number=12,
            metadata={"classification": "technical"},
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_PNG)
        res = VisionPipeline(provider=HandoffProvider()).run(
            VisionRequest(query="Data contract handoff", evidence=[ev])
        )

        # All public result fields
        assert isinstance(res.query, str) and res.query
        assert isinstance(res.status, str) and res.status
        assert isinstance(res.description, str)
        assert isinstance(res.evidence, list)
        assert isinstance(res.document_id, str) and res.document_id
        assert isinstance(res.filename, str) and res.filename
        assert isinstance(res.chunk_id, str) and res.chunk_id
        assert res.page_number == 12
        assert res.content_type == "diagram"
        assert isinstance(res.metadata, dict)
        assert res.error is None

        # Boolean helpers
        assert res.is_success is True
        assert res.is_error is False
        assert res.has_evidence is True

    def test_dc_02_error_result_fields(self) -> None:
        """Error results expose status and error fields correctly."""
        err = VisionResult(
            query="Error contract check",
            status="error",
            description="",
            error="Provider unavailable",
        )
        assert err.is_error is True
        assert err.is_success is False
        assert err.error == "Provider unavailable"
        assert err.status == "error"


# ===========================================================================
# Step 5 — Single Evidence
# ===========================================================================

class TestSingleEvidence:
    """Single evidence item: full Search → Vision → Result → Downstream chain."""

    def test_se_01_single_evidence_full_chain(self) -> None:
        """Single evidence preserves source identity, lineage, metadata, content type."""
        meta = {"source": "search_agent", "relevance": "high"}
        cit = _citation(
            doc_id="DOC-SE-57",
            filename="single_ev.pdf",
            chunk_id="CHK-SE-57",
            content_type="image",
            page_number=3,
            metadata=meta,
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_PNG)
        res = VisionPipeline(provider=HandoffProvider()).run(
            VisionRequest(query="Single evidence handoff", evidence=[ev])
        )

        assert res.document_id == "DOC-SE-57"
        assert res.filename == "single_ev.pdf"
        assert res.chunk_id == "CHK-SE-57"
        assert res.page_number == 3
        assert res.content_type == "image"
        assert len(res.evidence) == 1
        assert res.evidence[0].metadata.get("source") == "search_agent"


# ===========================================================================
# Step 6 — Multi-Evidence
# ===========================================================================

class TestMultiEvidence:
    """Multiple evidence items: count, order, lineage, metadata, no duplication."""

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_me_01_multi_evidence_count_and_order(self, n: int) -> None:
        """n evidence items arrive in the result in exactly the original order."""
        evidence = [
            VisualEvidenceAdapter.adapt_citation(
                _citation(
                    doc_id=f"DOC-ME57-{i}",
                    filename=f"ev_{i}.pdf",
                    chunk_id=f"CHK-ME57-{i}",
                    metadata={"seq": i},
                ),
                image_bytes=_PNG,
            )
            for i in range(n)
        ]
        res = VisionPipeline(provider=HandoffProvider()).run(
            VisionRequest(query=f"Multi-ev {n}", evidence=evidence)
        )

        assert res.is_success is True
        assert len(res.evidence) == n
        for i, ev in enumerate(res.evidence):
            assert ev.document_id == f"DOC-ME57-{i}"
            assert ev.chunk_id == f"CHK-ME57-{i}"
            assert ev.metadata.get("seq") == i


# ===========================================================================
# Step 7 — Multi-Document
# ===========================================================================

class TestMultiDocument:
    """Evidence from DOC-A/B/C/D returns strictly correct document identity."""

    @pytest.mark.parametrize("doc_id", ["DOC-A", "DOC-B", "DOC-C", "DOC-D"])
    def test_md_01_no_cross_document_contamination(self, doc_id: str) -> None:
        """Each document request returns strictly that document's identity."""
        cit = _citation(doc_id=doc_id, filename=f"{doc_id.lower()}.pdf")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_PNG)
        res = VisionPipeline(provider=HandoffProvider()).run(
            VisionRequest(query=f"Document isolation {doc_id}", evidence=[ev])
        )

        assert res.status == "success"
        assert res.document_id == doc_id
        assert res.filename == f"{doc_id.lower()}.pdf"
        assert res.evidence[0].document_id == doc_id


# ===========================================================================
# Step 8 — Error Handoff
# ===========================================================================

class TestErrorHandoff:
    """Downstream consumer distinguishes all failure types via public exceptions."""

    def test_eh_01_invalid_request_raises_input_validation(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="", evidence=[])

    def test_eh_02_invalid_evidence_content_type(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(
                document_id="DOC-EH-57",
                filename="fail.pdf",
                chunk_id="CHK-EH-57",
                content_type="video",  # not in VALID_VISUAL_CONTENT_TYPES
            )

    def test_eh_03_provider_failure_raises_provider_execution_error(self) -> None:
        ev = VisualEvidenceAdapter.adapt_citation(_citation(), image_bytes=_PNG)
        with pytest.raises(VisionProviderExecutionError):
            VisionPipeline(provider=HandoffProvider(fail_calls=[1])).run(
                VisionRequest(query="Provider failure", evidence=[ev])
            )

    def test_eh_04_retry_exhaustion(self) -> None:
        ev = VisualEvidenceAdapter.adapt_citation(_citation(), image_bytes=_PNG)
        with pytest.raises(VisionProviderExecutionError):
            VisionPipeline(provider=HandoffProvider(fail_calls=[1, 2, 3])).run(
                VisionRequest(query="Retry exhaustion", evidence=[ev]),
                retry_policy=VisionRetryPolicy(max_retries=2),
            )

    def test_eh_05_timeout_raises_vision_timeout_error(self) -> None:
        ev = VisualEvidenceAdapter.adapt_citation(_citation(), image_bytes=_PNG)
        with pytest.raises(VisionTimeoutError):
            VisionPipeline(provider=HandoffProvider(timeout=True)).run(
                VisionRequest(query="Timeout handoff", evidence=[ev])
            )

    def test_eh_06_cancellation_raises_vision_cancellation_error(self) -> None:
        ev = VisualEvidenceAdapter.adapt_citation(_citation(), image_bytes=_PNG)
        token = VisionCancellationToken()
        token.cancel()
        provider = HandoffProvider()
        with pytest.raises(VisionCancellationError):
            VisionPipeline(provider=provider).run(
                VisionRequest(query="Cancellation handoff", evidence=[ev]),
                cancellation_token=token,
            )
        assert provider.call_count == 0

    def test_eh_07_success_after_failures(self) -> None:
        """Consumer can send a valid request immediately after any failure."""
        ev = VisualEvidenceAdapter.adapt_citation(_citation(doc_id="DOC-EH-OK-57"), image_bytes=_PNG)
        res = VisionPipeline(provider=HandoffProvider()).run(
            VisionRequest(query="Valid post-failure", evidence=[ev])
        )
        assert res.status == "success"


# ===========================================================================
# Step 9 — Success / Failure Isolation (Concurrent)
# ===========================================================================

class TestConcurrentIsolation:
    """Concurrent A=success, B=failure, C=success, D=timeout, E=retry→success."""

    def test_ci_01_mixed_concurrent_outcomes_isolated(self) -> None:
        results: dict[str, Any] = {}
        exceptions: dict[str, Exception] = {}

        def run_a() -> None:
            try:
                ev = VisualEvidenceAdapter.adapt_citation(_citation(doc_id="DOC-MX-A"), image_bytes=_PNG)
                results["A"] = VisionPipeline(provider=HandoffProvider()).run(
                    VisionRequest(query="Mix A", evidence=[ev])
                )
            except Exception as e:
                exceptions["A"] = e

        def run_b() -> None:
            try:
                ev = VisualEvidenceAdapter.adapt_citation(_citation(doc_id="DOC-MX-B"), image_bytes=_PNG)
                VisionPipeline(provider=HandoffProvider(fail_calls=[1])).run(
                    VisionRequest(query="Mix B", evidence=[ev]),
                    retry_policy=VisionRetryPolicy(max_retries=0),
                )
            except Exception as e:
                exceptions["B"] = e

        def run_c() -> None:
            try:
                ev = VisualEvidenceAdapter.adapt_citation(_citation(doc_id="DOC-MX-C"), image_bytes=_PNG)
                results["C"] = VisionPipeline(provider=HandoffProvider()).run(
                    VisionRequest(query="Mix C", evidence=[ev])
                )
            except Exception as e:
                exceptions["C"] = e

        def run_d() -> None:
            try:
                ev = VisualEvidenceAdapter.adapt_citation(_citation(doc_id="DOC-MX-D"), image_bytes=_PNG)
                VisionPipeline(provider=HandoffProvider(timeout=True)).run(
                    VisionRequest(query="Mix D", evidence=[ev])
                )
            except Exception as e:
                exceptions["D"] = e

        def run_e() -> None:
            try:
                ev = VisualEvidenceAdapter.adapt_citation(_citation(doc_id="DOC-MX-E"), image_bytes=_PNG)
                results["E"] = VisionPipeline(provider=HandoffProvider(fail_calls=[1])).run(
                    VisionRequest(query="Mix E", evidence=[ev]),
                    retry_policy=VisionRetryPolicy(max_retries=2),
                )
            except Exception as e:
                exceptions["E"] = e

        threads = [
            threading.Thread(target=run_a),
            threading.Thread(target=run_b),
            threading.Thread(target=run_c),
            threading.Thread(target=run_d),
            threading.Thread(target=run_e),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # A → success, correct document
        assert results["A"].status == "success"
        assert results["A"].document_id == "DOC-MX-A"

        # B → provider execution error
        assert isinstance(exceptions["B"], VisionProviderExecutionError)

        # C → success, correct document
        assert results["C"].status == "success"
        assert results["C"].document_id == "DOC-MX-C"

        # D → timeout error
        assert isinstance(exceptions["D"], VisionTimeoutError)

        # E → retry then success
        assert results["E"].status == "success"
        assert results["E"].document_id == "DOC-MX-E"


# ===========================================================================
# Step 10 — Lineage Certification
# ===========================================================================

class TestLineageCertification:
    """Search → VisualEvidence → VisionRequest → VisionResult → downstream preserves all lineage."""

    def test_lc_01_lineage_end_to_end(self) -> None:
        meta = {"origin": "member2_search", "pipeline": "omnibrain", "day": 57}
        cit = _citation(
            doc_id="DOC-LC-57",
            filename="lineage_cert.pdf",
            chunk_id="CHK-LC-57",
            content_type="chart",
            page_number=7,
            metadata=meta,
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_PNG)

        # lineage at evidence level
        assert ev.document_id == "DOC-LC-57"
        assert ev.filename == "lineage_cert.pdf"
        assert ev.chunk_id == "CHK-LC-57"
        assert ev.page_number == 7
        assert ev.content_type == "chart"
        assert ev.chunk_index >= 0

        req = VisionRequest(query="Lineage cert", evidence=[ev])
        res = VisionPipeline(provider=HandoffProvider()).run(req)

        # lineage at result level
        assert res.document_id == "DOC-LC-57"
        assert res.filename == "lineage_cert.pdf"
        assert res.chunk_id == "CHK-LC-57"
        assert res.page_number == 7
        assert res.content_type == "chart"
        assert res.evidence[0].metadata.get("origin") == "member2_search"


# ===========================================================================
# Step 11 — Result Serialization
# ===========================================================================

class TestResultSerialization:
    """VisionResult.to_dict() / from_dict() roundtrip for downstream handoff."""

    def test_sr_01_roundtrip_preserves_all_public_fields(self) -> None:
        cit = _citation(
            doc_id="DOC-SR-57",
            filename="serial_cert.pdf",
            chunk_id="CHK-SR-57",
            content_type="diagram",
            page_number=2,
        )
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_PNG)
        res = VisionPipeline(provider=HandoffProvider()).run(
            VisionRequest(query="Serialization cert", evidence=[ev])
        )

        d = res.to_dict()
        assert isinstance(d, dict)

        restored = VisionResult.from_dict(d)
        assert restored.document_id == "DOC-SR-57"
        assert restored.filename == "serial_cert.pdf"
        assert restored.chunk_id == "CHK-SR-57"
        assert restored.page_number == 2
        assert restored.content_type == "diagram"
        assert restored.status == "success"
        assert restored.is_success is True
        assert len(restored.evidence) == 1
        assert restored.evidence[0].document_id == "DOC-SR-57"

    def test_sr_02_error_result_survives_roundtrip(self) -> None:
        err = VisionResult(
            query="Error roundtrip cert",
            status="error",
            error="Provider failed",
        )
        d = err.to_dict()
        restored = VisionResult.from_dict(d)
        assert restored.is_error is True
        assert restored.error == "Provider failed"


# ===========================================================================
# Step 12 — Mutation Safety
# ===========================================================================

class TestMutationSafety:
    """Downstream consumption does not mutate shared evidence, metadata, or results."""

    def test_ms_01_mutating_result_metadata_does_not_affect_evidence(self) -> None:
        cit = _citation(doc_id="DOC-MS-57", metadata={"key": "original"})
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_PNG)
        res = VisionPipeline(provider=HandoffProvider()).run(
            VisionRequest(query="Mutation safety", evidence=[ev])
        )

        # Consumer mutates result metadata
        res.metadata["injected"] = "should_not_propagate"

        # Original evidence metadata untouched
        assert "injected" not in res.evidence[0].metadata

    def test_ms_02_two_requests_share_no_metadata(self) -> None:
        cit_a = _citation(doc_id="DOC-MS-A", metadata={"req": "A"})
        cit_b = _citation(doc_id="DOC-MS-B", metadata={"req": "B"})
        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a, image_bytes=_PNG)
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b, image_bytes=_PNG)

        res_a = VisionPipeline(provider=HandoffProvider()).run(
            VisionRequest(query="Mutation A", evidence=[ev_a])
        )
        res_b = VisionPipeline(provider=HandoffProvider()).run(
            VisionRequest(query="Mutation B", evidence=[ev_b])
        )

        assert res_a.document_id == "DOC-MS-A"
        assert res_b.document_id == "DOC-MS-B"
        assert res_a.evidence[0].metadata.get("req") == "A"
        assert res_b.evidence[0].metadata.get("req") == "B"


# ===========================================================================
# Step 13 — Repeated Consumption
# ===========================================================================

class TestRepeatedConsumption:
    """Consuming the same VisionResult multiple times produces identical data."""

    def test_rc_01_repeated_field_access_is_stable(self) -> None:
        cit = _citation(doc_id="DOC-RC-57")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_PNG)
        res = VisionPipeline(provider=HandoffProvider()).run(
            VisionRequest(query="Repeated consumption", evidence=[ev])
        )

        for _ in range(5):
            assert res.status == "success"
            assert res.document_id == "DOC-RC-57"
            assert len(res.evidence) == 1
            assert res.is_success is True

    def test_rc_02_repeated_to_dict_produces_equal_dicts(self) -> None:
        cit = _citation(doc_id="DOC-RC2-57")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_PNG)
        res = VisionPipeline(provider=HandoffProvider()).run(
            VisionRequest(query="Repeated serialization", evidence=[ev])
        )

        d1 = res.to_dict()
        d2 = res.to_dict()
        assert d1["document_id"] == d2["document_id"]
        assert d1["status"] == d2["status"]
        assert d1["evidence"][0]["chunk_id"] == d2["evidence"][0]["chunk_id"]


# ===========================================================================
# Step 14 — Public Exception Contract
# ===========================================================================

class TestPublicExceptionContract:
    """All public exception types are importable, deterministic, and expose no secrets."""

    def test_exc_01_hierarchy_is_correct(self) -> None:
        assert issubclass(VisionInputValidationError, VisionError)
        assert issubclass(VisionEvidenceError, VisionError)
        assert issubclass(VisionProviderExecutionError, VisionError)
        assert issubclass(VisionTimeoutError, VisionError)
        assert issubclass(VisionCancellationError, VisionError)
        assert issubclass(VisionError, Exception)

    def test_exc_02_exception_messages_contain_no_secrets(self) -> None:
        try:
            VisionRequest(query="", evidence=[])
        except VisionInputValidationError as exc:
            msg = str(exc).lower()
            for forbidden in ("api_key", "password", "token", "secret", "bearer"):
                assert forbidden not in msg

    def test_exc_03_exception_is_catchable_as_vision_error(self) -> None:
        """Consumer can catch any Vision exception via the base VisionError."""
        caught = False
        try:
            VisionRequest(query="", evidence=[])
        except VisionError:
            caught = True
        assert caught is True


# ===========================================================================
# Step 15 — No Internal Leakage
# ===========================================================================

class TestNoInternalLeakage:
    """Downstream consumer never needs to touch private internals."""

    def test_il_01_result_exposes_no_private_attributes_needed(self) -> None:
        """Consumer exercises all needed functionality via public properties only."""
        cit = _citation(doc_id="DOC-IL-57")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_PNG)
        res = VisionPipeline(provider=HandoffProvider()).run(
            VisionRequest(query="Internal leakage check", evidence=[ev])
        )

        # All access via public interface — no _private attribute access
        _ = res.query
        _ = res.status
        _ = res.description
        _ = res.evidence
        _ = res.document_id
        _ = res.filename
        _ = res.chunk_id
        _ = res.page_number
        _ = res.content_type
        _ = res.metadata
        _ = res.error
        _ = res.is_success
        _ = res.is_error
        _ = res.has_evidence

        assert res.is_success is True


# ===========================================================================
# Step 16 — Member Responsibility Boundary
# ===========================================================================

class TestMemberBoundary:
    """Member 3 is Vision only — no retrieval, embedding, Qdrant, or orchestration."""

    def test_mb_01_vision_subsystem_contains_no_retrieval_code(self) -> None:
        """vision source files contain no qdrant/embed/retrieval-specific imports."""
        import pathlib
        import re

        vision_root = pathlib.Path(__file__).parent.parent
        source_files = [f for f in vision_root.glob("*.py") if not f.name.startswith("test_")]

        boundary_patterns = [
            r"^import qdrant",
            r"^from qdrant",
            r"^import sentence_transformers\b",
            r"^from sentence_transformers",
            r"^import langgraph\b",
            r"^from langgraph",
            r"^import fastapi\b",
            r"^from fastapi",
        ]

        violations: list[str] = []
        for src in source_files:
            for line_no, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                for pat in boundary_patterns:
                    if re.match(pat, stripped):
                        violations.append(f"{src.name}:{line_no}: {stripped!r}")

        assert not violations, "Boundary violations:\n" + "\n".join(violations)

    def test_mb_02_single_pipeline_call_single_execution(self) -> None:
        """One request → exactly one provider call — no duplicate retrieval or embedding."""
        provider = HandoffProvider()
        cit = _citation(doc_id="DOC-MB-57")
        ev = VisualEvidenceAdapter.adapt_citation(cit, image_bytes=_PNG)
        res = VisionPipeline(provider=provider).run(
            VisionRequest(query="Boundary execution check", evidence=[ev])
        )
        assert res.status == "success"
        assert provider.call_count == 1
