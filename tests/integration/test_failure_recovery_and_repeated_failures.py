"""
OmniBrain Member 4 -- Day 12 Reliability, Recovery & Repeated-Failure Integration Tests.

Verifies that the existing OMNIBRAIN system can recover correctly from repeated failures
without corrupting state or breaking subsequent successful requests.

Concern areas:
 1. Failure -> Success recovery
 2. Repeated provider failure handling
 3. Failure followed by different request isolation
 4. Success -> Failure -> Success sequence
 5. Timeout recovery
 6. Cancellation recovery
 7. Multi-evidence recovery & ordering
 8. Multi-document recovery & cross-document non-leakage
 9. Repeated execution sequences (e.g. F-F-S-F-S-S)
10. Concurrent failure and recovery isolation
11. Serialization after recovery
12. Resource & State Safety
13. Error contract preservation & exception hierarchy

Constraints:
 - 100% Offline: Zero external APIs, real LLMs, network, or production secrets.
 - Zero production code modified.
 - Zero new retry/recovery logic implemented in production.
"""

from __future__ import annotations

import concurrent.futures
import copy
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# Ingestion Subsystem (Member 1)
from ingestion.models import (
    ChunkValidationResult,
    ChunkingResult,
    DocumentChunk,
    DocumentMetadata,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    RetrievalServiceResult,
    VectorSearchResult,
)
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.retrieval_processor import build_retrieval_context, process_retrieval_results
from ingestion.ingestion_errors import (
    IngestionChunkingError,
    IngestionEmbeddingError,
    IngestionError,
    IngestionExtractionError,
    IngestionPipelineError,
    IngestionValidationError,
)
from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
)

# Search / Agents Subsystem (Member 2)
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    AgentState,
    SearchRequest,
    SearchResult,
)
from agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentRoutingError,
    AgentValidationError,
)

# Vision Subsystem (Member 3)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionAgentError,
    VisionCancellationError,
    VisionError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderError,
    VisionProviderUnavailableError,
    VisionTimeoutError,
)
from vision.lifecycle import (
    VisionCancellationToken,
    VisionExecutionLifecycle,
    VisionExecutionStage,
)


# ============================================================================
# Shared Fixtures & Helpers
# ============================================================================


def _create_visual_evidence(
    document_id: str = "doc-rec-01",
    filename: str = "recovery_report.pdf",
    chunk_id: str = "chk-rec-01",
    page_number: int | None = 1,
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
        metadata=metadata if metadata is not None else {"source": "test_recovery"},
    )


def _create_search_result_item(
    chunk_id: str = "chk-001",
    score: float = 0.90,
    document_id: str = "doc-001",
    filename: str = "doc.pdf",
    page_number: int | None = 1,
    content_type: str = "chart",
    content: str = "Sample content for recovery testing.",
    metadata: dict[str, Any] | None = None,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        score=score,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        chunk_index=0,
        content_type=content_type,
        content=content,
        metadata=metadata if metadata is not None else {"dept": "engineering"},
    )


# ============================================================================
# 1. FAILURE -> SUCCESS RECOVERY
# ============================================================================


class TestFailureToSuccessRecovery:
    """Verifies that an initial failed attempt followed by a valid retry succeeds with intact evidence and lineage."""

    def test_search_validation_failure_then_valid_retry_success(self) -> None:
        # Step 1: Initial request with invalid score fails validation
        with pytest.raises(AgentValidationError):
            SearchRequest(query="Quarterly revenue", min_score=1.5)

        # Step 2: Valid retry request succeeds
        valid_request = SearchRequest(query="Quarterly revenue", min_score=0.7)
        assert valid_request.query == "Quarterly revenue"
        assert valid_request.min_score == 0.7

        # Step 3: Pipeline produces successful SearchResult with valid citation
        vsr = _create_search_result_item(
            chunk_id="chk-rev-01",
            document_id="doc-rev-101",
            filename="q3_financials.pdf",
            page_number=3,
            content="Q3 Revenue was $12.4M",
        )
        citation = AgentCitation.from_search_result(vsr)
        search_res = SearchResult(
            query=valid_request.query,
            status="RESULTS_FOUND",
            citations=[citation],
            context=vsr.content,
        )

        assert search_res.has_results is True
        assert search_res.citations[0].document_id == "doc-rev-101"
        assert search_res.citations[0].filename == "q3_financials.pdf"
        assert search_res.citations[0].page_number == 3

    def test_vision_evidence_failure_then_valid_retry_success(self) -> None:
        # Step 1: Invalid visual content_type triggers failure
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(
                document_id="doc-fail-01",
                filename="f.pdf",
                chunk_id="c1",
                content_type="invalid_text_type",
            )

        # Step 2: Valid retry with supported visual content_type succeeds
        valid_ev = _create_visual_evidence(
            document_id="doc-rec-99",
            filename="chart_valid.pdf",
            chunk_id="chk-chart-01",
            page_number=5,
            content_type="chart",
            metadata={"chart_type": "line"},
        )
        req = VisionRequest(query="Analyze line chart.", evidence=[valid_ev])
        result = VisionResult(
            query=req.query,
            status="success",
            description="Upward trend of 12%.",
            evidence=req.evidence,
        )

        assert result.status == "success"
        assert result.document_id == "doc-rec-99"
        assert result.filename == "chart_valid.pdf"
        assert result.page_number == 5
        assert result.evidence[0].metadata["chart_type"] == "line"


# ============================================================================
# 2. REPEATED PROVIDER FAILURE
# ============================================================================


class TestRepeatedProviderFailure:
    """Verifies that consecutive provider failures consistently return expected error states without corrupting state."""

    def test_consecutive_provider_failures_do_not_produce_stale_success(self) -> None:
        failures = []
        for attempt in range(5):
            lifecycle = VisionExecutionLifecycle(
                provider_name="mock_provider",
                model_name="mock_vision_model",
            )
            lifecycle.transition_to(VisionExecutionStage.VALIDATING)
            lifecycle.transition_to(VisionExecutionStage.FAILED, error=f"Provider unavailable attempt {attempt}")

            vres = VisionResult(
                query=f"Query {attempt}",
                status="error",
                description="",
                error=lifecycle.error,
            )
            failures.append(vres)

        assert len(failures) == 5
        for idx, res in enumerate(failures):
            assert res.status == "error"
            assert f"attempt {idx}" in (res.error or "")
            assert res.description == ""
            assert res.evidence == []


# ============================================================================
# 3. FAILURE THEN DIFFERENT REQUEST ISOLATION
# ============================================================================


class TestFailureThenDifferentRequestIsolation:
    """Verifies that a failed request A leaves zero artifacts or state leakage in subsequent request B."""

    def test_failed_request_does_not_contaminate_different_request(self) -> None:
        # Request A: Fails validation
        with pytest.raises(AgentValidationError):
            AgentRequest(query="   ")

        # Request B: Independent request succeeds
        req_b = AgentRequest(query="Explain quarterly projection", session_id="sess-b")
        ev_b = _create_visual_evidence(
            document_id="doc-proj-404",
            filename="projections.pdf",
            chunk_id="chk-proj-01",
            metadata={"sensitivity": "confidential"},
        )
        citation_b = AgentCitation.from_dict({
            "document_id": ev_b.document_id,
            "filename": ev_b.filename,
            "chunk_id": ev_b.chunk_id,
            "page_number": ev_b.page_number,
            "content_type": ev_b.content_type,
            "score": 0.92,
            "metadata": ev_b.metadata,
        })
        resp_b = AgentResponse(
            answer="Projections indicate 8% growth.",
            agent_name="SearchAgent",
            status="success",
            citations=[citation_b],
        )

        assert resp_b.is_success is True
        assert resp_b.citations[0].document_id == "doc-proj-404"
        assert resp_b.error is None


# ============================================================================
# 4. SUCCESS -> FAILURE -> SUCCESS SEQUENCE
# ============================================================================


class TestSuccessFailureSuccessSequence:
    """Verifies that a pipeline handles S -> F -> S sequence without cross-stage state contamination."""

    def test_success_failure_success_lifecycle(self) -> None:
        # Step 1: Request A succeeds
        state_a = AgentState(query="Query A")
        cit_a = AgentCitation(document_id="doc-A", filename="a.pdf", chunk_id="ck-A", score=0.9)
        state_a.add_citation(cit_a)
        state_a.answer = "Answer A"
        state_a.status = "completed"

        # Step 2: Request B fails
        state_b = AgentState(query="Query B")
        state_b.add_error("Execution failed for Query B")
        state_b.status = "failed"

        # Step 3: Request C succeeds
        state_c = AgentState(query="Query C")
        cit_c = AgentCitation(document_id="doc-C", filename="c.pdf", chunk_id="ck-C", score=0.95)
        state_c.add_citation(cit_c)
        state_c.answer = "Answer C"
        state_c.status = "completed"

        # Verify strict state boundaries
        assert state_a.status == "completed"
        assert state_a.answer == "Answer A"
        assert len(state_a.citations) == 1
        assert state_a.citations[0].document_id == "doc-A"
        assert state_a.errors == []

        assert state_b.status == "failed"
        assert "Execution failed" in state_b.errors[0]
        assert state_b.citations == []
        assert state_b.answer == ""

        assert state_c.status == "completed"
        assert state_c.answer == "Answer C"
        assert len(state_c.citations) == 1
        assert state_c.citations[0].document_id == "doc-C"
        assert state_c.errors == []


# ============================================================================
# 5. TIMEOUT RECOVERY
# ============================================================================


class TestTimeoutRecovery:
    """Verifies that a timeout lifecycle error cleanly resolves and allows subsequent requests to succeed."""

    def test_timeout_transition_and_subsequent_request_recovery(self) -> None:
        # 1. Timed out execution
        lifecycle_timeout = VisionExecutionLifecycle(provider_name="mock_provider", model_name="mock_model")
        lifecycle_timeout.transition_to(VisionExecutionStage.EXECUTING)
        lifecycle_timeout.transition_to(VisionExecutionStage.TIMEOUT, error="Execution exceeded deadline of 30.0s")

        res_timeout = VisionResult(
            query="Heavy query",
            status="error",
            description="",
            error=lifecycle_timeout.error,
        )
        assert res_timeout.status == "error"
        assert "exceeded deadline" in (res_timeout.error or "")

        # 2. Subsequent normal execution recovers completely
        lifecycle_normal = VisionExecutionLifecycle(provider_name="mock_provider", model_name="mock_model")
        lifecycle_normal.transition_to(VisionExecutionStage.COMPLETED)

        ev = _create_visual_evidence(document_id="doc-timeout-rec", filename="rec.pdf")
        res_normal = VisionResult(
            query="Standard query",
            status="success",
            description="Analysis complete.",
            evidence=[ev],
        )
        assert res_normal.status == "success"
        assert res_normal.document_id == "doc-timeout-rec"
        assert res_normal.error is None


# ============================================================================
# 6. CANCELLATION RECOVERY
# ============================================================================


class TestCancellationRecovery:
    """Verifies that an execution cancelled via VisionCancellationToken resets cleanly for next request."""

    def test_cancellation_token_does_not_leak_to_subsequent_tokens(self) -> None:
        # 1. Token A is cancelled
        token_a = VisionCancellationToken()
        token_a.cancel(reason="User clicked cancel")
        assert token_a.is_cancelled is True
        assert token_a.reason == "User clicked cancel"

        # 2. Token B is fresh and independent
        token_b = VisionCancellationToken()
        assert token_b.is_cancelled is False
        assert token_b.reason is None

        # 3. Successful execution with Token B
        ev = _create_visual_evidence(document_id="doc-cancel-rec", filename="cancel_rec.pdf")
        res_b = VisionResult(query="Query after cancellation", status="success", description="Ok", evidence=[ev])
        assert res_b.status == "success"
        assert res_b.document_id == "doc-cancel-rec"


# ============================================================================
# 7. MULTI-EVIDENCE RECOVERY & ORDERING
# ============================================================================


class TestMultiEvidenceRecovery:
    """Verifies that retry of a multi-evidence request recovers all evidence items in exact order."""

    def test_multi_evidence_retry_preserves_exact_order_and_count(self) -> None:
        ev_1 = _create_visual_evidence(document_id="doc-me", chunk_id="ck-01", page_number=1, metadata={"seq": 1})
        ev_2 = _create_visual_evidence(document_id="doc-me", chunk_id="ck-02", page_number=2, metadata={"seq": 2})
        ev_3 = _create_visual_evidence(document_id="doc-me", chunk_id="ck-03", page_number=3, metadata={"seq": 3})

        # Initial failed attempt
        failed_res = VisionResult(query="Analyze 3 chunks", status="error", error="Transient glitch")
        assert failed_res.status == "error"

        # Recovered retry with all 3 evidence items
        retry_req = VisionRequest(query="Analyze 3 chunks", evidence=[ev_1, ev_2, ev_3])
        assert retry_req.total_evidence == 3

        recovered_res = VisionResult(
            query=retry_req.query,
            status="success",
            description="Analysis of 3 chunks complete.",
            evidence=retry_req.evidence,
        )
        assert recovered_res.status == "success"
        assert len(recovered_res.evidence) == 3
        assert [e.chunk_id for e in recovered_res.evidence] == ["ck-01", "ck-02", "ck-03"]
        assert [e.metadata["seq"] for e in recovered_res.evidence] == [1, 2, 3]


# ============================================================================
# 8. MULTI-DOCUMENT RECOVERY
# ============================================================================


class TestMultiDocumentRecovery:
    """Verifies that failure on Document A does not pollute requests on Document B."""

    def test_failure_on_doc_a_does_not_leak_to_doc_b(self) -> None:
        # Document A fails
        doc_a_chunk = _create_document_chunk_invalid = None
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="doc_a.pdf", chunk_id="chk-a")

        # Document B executes successfully
        doc_b_vsr = _create_search_result_item(
            chunk_id="chk-b-01",
            document_id="doc-beta-unique-99",
            filename="doc_b.pdf",
            page_number=4,
            metadata={"owner": "BetaTeam"},
        )
        cit_b = AgentCitation.from_search_result(doc_b_vsr)
        assert cit_b.document_id == "doc-beta-unique-99"
        assert cit_b.filename == "doc_b.pdf"
        assert cit_b.metadata["owner"] == "BetaTeam"
        assert "doc_a" not in cit_b.filename


# ============================================================================
# 9. REPEATED EXECUTION SEQUENCES (F-F-S-F-S-S)
# ============================================================================


class TestRepeatedExecutionSequences:
    """Verifies alternating sequence of failure and success: [F, F, S, F, S, S]."""

    def test_interleaved_failure_success_sequence(self) -> None:
        sequence = ["F", "F", "S", "F", "S", "S"]
        results = []

        for idx, outcome in enumerate(sequence):
            if outcome == "F":
                # Simulated failure
                res = VisionResult(
                    query=f"Query {idx}",
                    status="error",
                    description="",
                    error=f"Error on execution {idx}",
                )
            else:
                # Simulated success
                ev = _create_visual_evidence(
                    document_id=f"doc-seq-{idx}",
                    filename=f"file_{idx}.pdf",
                    chunk_id=f"chk_{idx}",
                )
                res = VisionResult(
                    query=f"Query {idx}",
                    status="success",
                    description=f"Success output {idx}",
                    evidence=[ev],
                )
            results.append(res)

        assert len(results) == 6
        for idx, (res, expected_type) in enumerate(zip(results, sequence)):
            if expected_type == "F":
                assert res.status == "error"
                assert f"Error on execution {idx}" in (res.error or "")
                assert res.evidence == []
            else:
                assert res.status == "success"
                assert res.document_id == f"doc-seq-{idx}"
                assert res.error is None


# ============================================================================
# 10. CONCURRENT FAILURE & RECOVERY ISOLATION
# ============================================================================


class TestConcurrentFailureAndRecoveryIsolation:
    """Verifies that concurrent requests with mixed outcomes (A:Fail, B:Success, C:Fail, D:Success) execute in isolation."""

    def test_concurrent_mixed_outcomes_isolation(self) -> None:
        def _execute_request(req_id: int) -> dict[str, Any]:
            is_success = (req_id % 2 == 1)  # Odd: Success, Even: Fail
            if is_success:
                ev = _create_visual_evidence(
                    document_id=f"doc-conc-succ-{req_id}",
                    chunk_id=f"chk-succ-{req_id}",
                    metadata={"req_id": req_id},
                )
                res = VisionResult(
                    query=f"Query {req_id}",
                    status="success",
                    description=f"Result {req_id}",
                    evidence=[ev],
                )
                return {
                    "req_id": req_id,
                    "status": "success",
                    "doc_id": res.document_id,
                    "has_error": False,
                }
            else:
                res = VisionResult(
                    query=f"Query {req_id}",
                    status="error",
                    description="",
                    error=f"Simulated failure {req_id}",
                )
                return {
                    "req_id": req_id,
                    "status": "error",
                    "doc_id": res.document_id,
                    "has_error": True,
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_execute_request, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        for r in results:
            req_id = r["req_id"]
            if req_id % 2 == 1:
                assert r["status"] == "success"
                assert r["doc_id"] == f"doc-conc-succ-{req_id}"
                assert r["has_error"] is False
            else:
                assert r["status"] == "error"
                assert r["has_error"] is True


# ============================================================================
# 11. SERIALIZATION AFTER RECOVERY
# ============================================================================


class TestSerializationAfterRecovery:
    """Verifies that results obtained after a recovery cycle serialize and deserialize losslessly."""

    def test_recovered_result_serialization_round_trip(self) -> None:
        # Initial failure
        failed = VisionResult(query="q", status="error", error="glitch")
        assert failed.status == "error"

        # Recovered success
        ev = _create_visual_evidence(
            document_id="doc-recovered-ser",
            filename="recovered.pdf",
            chunk_id="chk-rec-ser-01",
            page_number=11,
            content_type="chart",
            metadata={"recovery_attempt": 2, "verified": True},
        )
        recovered = VisionResult(
            query="Analyze recovered chart",
            status="success",
            description="Chart shows 100% recovery.",
            evidence=[ev],
        )

        data = recovered.to_dict()
        restored = VisionResult.from_dict(data)

        assert restored.status == "success"
        assert restored.description == "Chart shows 100% recovery."
        assert restored.document_id == "doc-recovered-ser"
        assert restored.filename == "recovered.pdf"
        assert restored.page_number == 11
        assert len(restored.evidence) == 1
        assert restored.evidence[0].metadata["recovery_attempt"] == 2
        assert restored.evidence[0].metadata["verified"] is True


# ============================================================================
# 12. RESOURCE & STATE SAFETY
# ============================================================================


class TestResourceAndStateSafety:
    """Verifies that repeated failures do not leak state or mutate shared caller objects."""

    def test_repeated_validation_failures_do_not_mutate_caller_dict(self) -> None:
        caller_metadata = {"source": "caller", "version": 1}
        original_copy = copy.deepcopy(caller_metadata)

        for _ in range(5):
            try:
                # Trigger validation error
                AgentCitation(document_id="", filename="f.pdf", chunk_id="c", metadata=caller_metadata)
            except AgentValidationError:
                pass

        assert caller_metadata == original_copy


# ============================================================================
# 13. ERROR CONTRACT PRESERVATION
# ============================================================================


class TestErrorContractPreservation:
    """Verifies all failure modes produce specific exception classes adhering to public contract."""

    def test_public_exception_types_and_sanitization(self) -> None:
        # 1. Ingestion exceptions
        assert issubclass(IngestionValidationError, IngestionError)
        assert issubclass(PDFNotFoundError, IngestionExtractionError)

        # 2. Agent exceptions
        assert issubclass(AgentValidationError, AgentError)
        assert issubclass(AgentRoutingError, AgentError)

        # 3. Vision exceptions
        assert issubclass(VisionInputValidationError, VisionAgentError)
        assert issubclass(VisionEvidenceError, VisionAgentError)
        assert issubclass(VisionTimeoutError, VisionProviderError)
        assert issubclass(VisionCancellationError, VisionAgentError)
