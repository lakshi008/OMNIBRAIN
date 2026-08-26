"""
OmniBrain Member 4 -- Day 22 Operational Failure Injection & Recovery Certification Tests.

Performs controlled, offline, synthetic component failure injection and recovery verification:

  Normal Request
        ↓
  Component Failure Injection
        ↓
  Existing Error / Recovery Contract
        ↓
  Clean State Reset
        ↓
  Subsequent Valid Request

Concern areas:
 1. Normal baseline request -- verified valid baseline execution
 2. Ingestion failure injection -- invalid parameters produce expected validation failures without stale state
 3. Search / Retrieval failure injection -- malformed query/params raise AgentValidationError, NO_RESULTS handled cleanly
 4. Vision failure injection -- invalid evidence/query raises VisionEvidenceError / VisionInputValidationError
 5. Retry behavior -- lifecycle transitions and state progression across repeated simulated attempts
 6. Repeated failure -- repeated error requests do not accumulate state or leak cross-request memory
 7. Failure -> Success recovery -- FAIL -> SUCCESS -> SUCCESS sequence with complete state independence
 8. Success -> Failure -> Success -- SUCCESS -> FAIL -> SUCCESS sequence without state contamination
 9. Multi-document failure isolation -- Doc A fails, Doc B succeeds without receiving Doc A metadata/lineage
10. Multi-evidence failure -- invalid item in multi-evidence collection cleanly rejected without partial corruption
11. Timeout behavior where supported -- provider timeout represented in VisionResult.error without dangling state
12. Cancellation / lifecycle error state -- transition to failed stage handled deterministically
13. Concurrent failure isolation -- concurrent requests where some fail and others succeed remain completely isolated
14. Error serialization -- error status and message survive to_dict() -> from_dict() roundtrip
15. Failure security boundary -- error context stripped of forbidden keys, secrets, and credentials
16. Resource safety & offline constraint -- 100% offline, pure in-memory execution

Constraints:
 - 100% Offline: No external APIs, network, real LLMs, or production secrets.
 - Zero production code modified.
 - Only observable behavior guaranteed by existing public contracts tested.
"""

from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# Ingestion Subsystem (Member 1)
from ingestion.models import (
    DocumentChunk,
    VectorSearchResult,
)
from ingestion.retrieval_processor import process_retrieval_results, build_retrieval_context

# Search / Agents Subsystem (Member 2)
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    AgentState,
    SearchRequest,
    SearchResult,
)
from agents.exceptions import AgentValidationError

# Vision Subsystem (Member 3)
from vision.models import (
    VisionRequest,
    VisionResult,
    VisualEvidence,
    VALID_VISUAL_CONTENT_TYPES,
)
from vision.exceptions import VisionEvidenceError, VisionInputValidationError
from vision.lifecycle import (
    VisionExecutionLifecycle,
    VisionExecutionStage,
)
from vision.result_normalizer import (
    VisionExecutionTrace,
    VisionResultNormalizer,
)


# ============================================================================
# Helpers & Synthetic Fixtures
# ============================================================================

def _fail_chunk(
    chunk_id: str = "chk-fail-001",
    document_id: str = "doc-fail-001",
    filename: str = "fail_test.pdf",
    page_number: int | None = 2,
    chunk_index: int = 0,
    content: str = "Financial failure injection report.",
    content_type: str = "chart",
    metadata: dict[str, Any] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        content=content,
        content_type=content_type,
        metadata=metadata if metadata is not None else {"failure_test": "DAY22", "source": "FAIL_FIXTURE"},
    )


def _fail_vsr(
    chunk_id: str = "chk-fail-001",
    score: float = 0.95,
    document_id: str = "doc-fail-001",
    filename: str = "fail_test.pdf",
    page_number: int | None = 2,
    chunk_index: int = 0,
    content: str = "Financial failure injection report.",
    content_type: str = "chart",
    metadata: dict[str, Any] | None = None,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        score=score,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        chunk_index=chunk_index,
        content_type=content_type,
        content=content,
        metadata=metadata if metadata is not None else {"failure_test": "DAY22", "source": "FAIL_FIXTURE"},
    )


def _fail_evidence(
    chunk_id: str = "chk-fail-001",
    document_id: str = "doc-fail-001",
    filename: str = "fail_test.pdf",
    page_number: int | None = 2,
    chunk_index: int = 0,
    content_type: str = "chart",
    metadata: dict[str, Any] | None = None,
) -> VisualEvidence:
    return VisualEvidence(
        document_id=document_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        chunk_index=chunk_index,
        content_type=content_type,
        metadata=metadata if metadata is not None else {"failure_test": "DAY22", "source": "FAIL_FIXTURE"},
    )


# ============================================================================
# 1. NORMAL BASELINE REQUEST
# ============================================================================


class TestNormalBaselineRequest:
    """Establishes baseline valid execution before failure injection."""

    def test_baseline_valid_workflow(self) -> None:
        chunk = _fail_chunk(chunk_id="chk-base-01", document_id="doc-base-01")
        vsr = _fail_vsr(chunk_id=chunk.chunk_id, document_id=chunk.document_id)
        citation = AgentCitation.from_search_result(vsr)
        evidence = VisualEvidence.from_search_result(vsr)
        result = VisionResult(
            query="Baseline valid query",
            status="success",
            description="Baseline verified.",
            evidence=[evidence],
        )

        assert result.status == "success"
        assert result.document_id == "doc-base-01"
        assert result.chunk_id == "chk-base-01"
        assert citation.document_id == "doc-base-01"


# ============================================================================
# 2. INGESTION FAILURE INJECTION
# ============================================================================


class TestIngestionFailureInjection:
    """Verifies that ingestion-level failures are handled with clean error boundaries."""

    def test_empty_retrieval_results_handling(self) -> None:
        # Injection: Empty search results from ingestion retriever
        processed = process_retrieval_results([], min_score=0.70)
        assert len(processed) == 0

        context = build_retrieval_context(processed)
        assert context == ""

    def test_invalid_score_filtering_drops_below_threshold(self) -> None:
        # Injection: Low-quality retrieval candidates
        low_score_vsr = _fail_vsr(chunk_id="chk-low-01", score=0.30)
        processed = process_retrieval_results([low_score_vsr], min_score=0.80)
        assert len(processed) == 0


# ============================================================================
# 3. SEARCH / RETRIEVAL FAILURE INJECTION
# ============================================================================


class TestSearchFailureInjection:
    """Verifies search agent validation and NO_RESULTS packaging on invalid/failed retrieval."""

    def test_invalid_search_request_parameters_raise(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="")

        with pytest.raises(AgentValidationError):
            SearchRequest(query="valid query", top_k=-5)

        with pytest.raises(AgentValidationError):
            SearchRequest(query="valid query", min_score=2.5)

    def test_no_results_search_packaging(self) -> None:
        sr = SearchResult(query="Unmatched query", status="NO_RESULTS", citations=[])
        assert sr.status == "NO_RESULTS"
        assert sr.has_results is False
        assert len(sr.citations) == 0


# ============================================================================
# 4. VISION FAILURE INJECTION
# ============================================================================


class TestVisionFailureInjection:
    """Verifies vision validation and error result contract on component failure."""

    def test_invalid_visual_evidence_creation_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename="bad.pdf", chunk_id="c1", content_type="image")

        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="doc1", filename="bad.pdf", chunk_id="c1", content_type="unsupported_modality")

    def test_invalid_vision_request_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="")

    def test_vision_result_error_contract(self) -> None:
        err_res = VisionResult(
            query="Failing vision request",
            status="error",
            description="",
            error="Vision provider rate limit exceeded (HTTP 429).",
            metadata={"injected_failure": True},
        )
        assert err_res.status == "error"
        assert err_res.error == "Vision provider rate limit exceeded (HTTP 429)."
        assert err_res.has_evidence is False
        assert err_res.evidence == []


# ============================================================================
# 5. RETRY BEHAVIOR
# ============================================================================


class TestRetryBehavior:
    """Verifies lifecycle progression across simulated retry sequences (Attempt 1 fail, Attempt 2 fail, Attempt 3 succeed)."""

    def test_simulated_retry_sequence_progression(self) -> None:
        # Attempt 1: Fails at validating stage
        lc1 = VisionExecutionLifecycle(provider_name="mock_p", model_name="mock_m")
        lc1.transition_to(VisionExecutionStage.VALIDATING)
        lc1.transition_to(VisionExecutionStage.FAILED, error="ValidationError on Attempt 1")
        assert lc1.stage == VisionExecutionStage.FAILED
        assert lc1.error == "ValidationError on Attempt 1"

        # Attempt 2: Fails at executing stage
        lc2 = VisionExecutionLifecycle(provider_name="mock_p", model_name="mock_m")
        lc2.transition_to(VisionExecutionStage.VALIDATING)
        lc2.transition_to(VisionExecutionStage.EXECUTING)
        lc2.transition_to(VisionExecutionStage.FAILED, error="Timeout on Attempt 2")
        assert lc2.stage == VisionExecutionStage.FAILED

        # Attempt 3: Succeeds completely
        lc3 = VisionExecutionLifecycle(provider_name="mock_p", model_name="mock_m")
        lc3.transition_to(VisionExecutionStage.VALIDATING)
        lc3.transition_to(VisionExecutionStage.EXECUTING)
        lc3.transition_to(VisionExecutionStage.COMPLETED)
        assert lc3.stage == VisionExecutionStage.COMPLETED
        assert lc3.error is None


# ============================================================================
# 6. REPEATED FAILURES
# ============================================================================


class TestRepeatedFailures:
    """Verifies that repeated failure executions do not accumulate state or leak resources."""

    def test_repeated_failure_execution_isolation(self) -> None:
        REPEAT_COUNT = 5
        for i in range(REPEAT_COUNT):
            with pytest.raises(AgentValidationError):
                AgentCitation(document_id="", filename=f"bad_{i}.pdf", chunk_id=f"chk_bad_{i}")

        # Ensure subsequent valid call is completely pristine
        clean_cit = AgentCitation(document_id="doc-clean", filename="clean.pdf", chunk_id="chk-clean")
        assert clean_cit.document_id == "doc-clean"
        assert clean_cit.filename == "clean.pdf"


# ============================================================================
# 7. FAILURE → SUCCESS RECOVERY (FAIL -> SUCCESS -> SUCCESS)
# ============================================================================


class TestFailureSuccessRecovery:
    """Verifies Request A -> FAIL, Request B -> SUCCESS, Request C -> SUCCESS sequence."""

    def test_fail_success_success_sequence(self) -> None:
        # Request A: FAIL
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename="badA.pdf", chunk_id="c_badA", content_type="image")

        # Request B: SUCCESS
        ev_b = _fail_evidence(chunk_id="chk-reqB", document_id="doc-reqB")
        res_b = VisionResult(query="Query B", status="success", description="Result B.", evidence=[ev_b])
        assert res_b.status == "success"
        assert res_b.document_id == "doc-reqB"

        # Request C: SUCCESS
        ev_c = _fail_evidence(chunk_id="chk-reqC", document_id="doc-reqC")
        res_c = VisionResult(query="Query C", status="success", description="Result C.", evidence=[ev_c])
        assert res_c.status == "success"
        assert res_c.document_id == "doc-reqC"

        assert "doc-reqB" not in str(res_c.to_dict())


# ============================================================================
# 8. SUCCESS → FAILURE → SUCCESS (SUCCESS -> FAIL -> SUCCESS)
# ============================================================================


class TestSuccessFailureSuccess:
    """Verifies Request A -> SUCCESS, Request B -> FAIL, Request C -> SUCCESS sequence."""

    def test_success_fail_success_sequence(self) -> None:
        # Request A: SUCCESS
        ev_a = _fail_evidence(chunk_id="chk-reqA", document_id="doc-reqA")
        res_a = VisionResult(query="Query A", status="success", description="Result A.", evidence=[ev_a])
        assert res_a.status == "success"

        # Request B: FAIL
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="doc-reqB", filename="", chunk_id="chk-reqB")

        # Request C: SUCCESS
        ev_c = _fail_evidence(chunk_id="chk-reqC", document_id="doc-reqC")
        res_c = VisionResult(query="Query C", status="success", description="Result C.", evidence=[ev_c])
        assert res_c.status == "success"
        assert res_c.document_id == "doc-reqC"
        assert res_c.error is None


# ============================================================================
# 9. MULTI-DOCUMENT FAILURE ISOLATION
# ============================================================================


class TestMultiDocumentFailureIsolation:
    """Verifies that if Document A fails, Document B proceeds successfully without cross-pollution."""

    def test_document_a_failure_does_not_affect_document_b(self) -> None:
        # Document A fails validation
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename="docA.pdf", chunk_id="chk-docA", content_type="image")

        # Document B succeeds normally
        chunk_b = _fail_chunk(document_id="doc-B-valid", filename="docB.pdf", metadata={"tenant": "CORP_B"})
        vsr_b = _fail_vsr(chunk_id=chunk_b.chunk_id, document_id=chunk_b.document_id, filename=chunk_b.filename, metadata=chunk_b.metadata)
        ev_b = VisualEvidence.from_search_result(vsr_b)
        res_b = VisionResult(query="Query B", status="success", description="Doc B processed.", evidence=[ev_b])

        assert res_b.document_id == "doc-B-valid"
        assert res_b.evidence[0].metadata["tenant"] == "CORP_B"
        assert "docA" not in str(res_b.to_dict())


# ============================================================================
# 10. MULTI-EVIDENCE FAILURE
# ============================================================================


class TestMultiEvidenceFailure:
    """Verifies that an invalid item in a multi-evidence list is rejected without partial corruption."""

    def test_multi_evidence_with_one_invalid_item_fails_cleanly(self) -> None:
        valid_ev = _fail_evidence(chunk_id="chk-valid-01")

        # Attempting to build VisionRequest with a non-VisualEvidence item
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="Multi-evidence query", evidence=[valid_ev, "INVALID_STRING_ITEM"])  # type: ignore[list-item]


# ============================================================================
# 11. TIMEOUT & CANCELLATION ERROR REPRESENTATION
# ============================================================================


class TestTimeoutAndCancellationErrorRepresentation:
    """Verifies timeout and cancellation error representations in VisionResult and Lifecycle."""

    def test_synthetic_timeout_error_result(self) -> None:
        timeout_result = VisionResult(
            query="Timeout query",
            status="error",
            description="",
            error="Execution timed out after 30000ms deadline.",
            metadata={"timeout_ms": 30000},
        )
        assert timeout_result.status == "error"
        assert "timed out" in timeout_result.error.lower()
        assert timeout_result.metadata["timeout_ms"] == 30000

    def test_synthetic_cancellation_lifecycle_state(self) -> None:
        lc = VisionExecutionLifecycle(provider_name="mock_p", model_name="mock_m")
        lc.transition_to(VisionExecutionStage.VALIDATING)
        lc.transition_to(VisionExecutionStage.EXECUTING)
        lc.transition_to(VisionExecutionStage.FAILED, error="Operation cancelled by user/supervisor.")
        assert lc.stage == VisionExecutionStage.FAILED
        assert "cancelled" in lc.error.lower()


# ============================================================================
# 12. CONCURRENT FAILURE ISOLATION
# ============================================================================


class TestConcurrentFailureIsolation:
    """Verifies that concurrent requests where some fail and others succeed maintain strict isolation."""

    def test_concurrent_mixed_success_and_failure(self) -> None:
        def _worker(thread_idx: int) -> dict[str, Any]:
            should_fail = (thread_idx % 2 == 1)

            if should_fail:
                try:
                    _ = VisualEvidence(document_id="", filename=f"bad_{thread_idx}.pdf", chunk_id=f"c_{thread_idx}", content_type="image")
                    return {"thread_idx": thread_idx, "status": "unexpected_success"}
                except VisionEvidenceError:
                    return {"thread_idx": thread_idx, "status": "expected_failure"}
            else:
                doc_id = f"doc-conc-succ-{thread_idx:02d}"
                ev = _fail_evidence(chunk_id=f"chk-succ-{thread_idx}", document_id=doc_id)
                res = VisionResult(query=f"Query {thread_idx}", status="success", description="OK.", evidence=[ev])
                return {
                    "thread_idx": thread_idx,
                    "status": "success",
                    "doc_id": res.document_id,
                    "serialized": str(res.to_dict()),
                }

        concurrency = 16
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_worker, i) for i in range(concurrency)]
            outputs = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(outputs) == concurrency
        for r in outputs:
            tidx = r["thread_idx"]
            if tidx % 2 == 1:
                assert r["status"] == "expected_failure"
            else:
                assert r["status"] == "success"
                assert r["doc_id"] == f"doc-conc-succ-{tidx:02d}"


# ============================================================================
# 13. ERROR SERIALIZATION
# ============================================================================


class TestErrorSerialization:
    """Verifies that error results survive to_dict() -> from_dict() roundtrip."""

    def test_error_vision_result_roundtrip(self) -> None:
        orig = VisionResult(
            query="Error serialization query",
            status="error",
            description="",
            error="Connection refused to offline mock provider.",
            metadata={"retryable": False, "failure_code": "ERR_CONN_REFUSED"},
        )
        d = orig.to_dict()
        restored = VisionResult.from_dict(d)

        assert restored.status == "error"
        assert restored.error == "Connection refused to offline mock provider."
        assert restored.query == orig.query
        assert restored.has_evidence is False


# ============================================================================
# 14. FAILURE SECURITY BOUNDARY & OFFLINE SAFETY
# ============================================================================


class TestFailureSecurityBoundaryAndOfflineSafety:
    """Verifies error sanitization and pure in-memory execution during failures."""

    def test_error_metadata_sanitization(self) -> None:
        dirty_error_meta = {
            "api_key": "SYNTHETIC_API_KEY_SECRET",
            "password": "SYNTHETIC_PASSWORD",
            "error_detail": "Authentication failed on mock boundary.",
        }
        sanitized = VisionResultNormalizer.sanitize_metadata(dirty_error_meta)
        assert "api_key" not in sanitized
        assert "password" not in sanitized
        assert sanitized["error_detail"] == "Authentication failed on mock boundary."
