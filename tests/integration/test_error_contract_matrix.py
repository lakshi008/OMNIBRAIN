"""
OmniBrain Member 4 — Day 29 Error Contract Matrix & Failure Isolation Certification Tests.

Validates the complete error contract matrix across all OmniBrain layers:
    Ingestion (Member 1)
         ↓
    Search / Retrieval (Member 2)
         ↓
    Vision (Member 3)
         ↓
    Downstream Supervisor / Agent Consumers

Focus areas:
 1. Public exception class hierarchy (IngestionError, AgentError, VisionAgentError).
 2. Invalid input validation matrix (empty strings, invalid types, out-of-range values).
 3. Not-found and empty result contract behavior (empty retrieval, NO_RESULTS status).
 4. Serialization error handling (deserialization rejection of non-dict inputs).
 5. Error information preservation and structured error reporting.
 6. Error serialization round-trip (AgentResponse and VisionResult in error form).
 7. Recovery workflows:
      - FAILURE → SUCCESS
      - SUCCESS → FAILURE → SUCCESS
      - FAILURE → FAILURE → SUCCESS
 8. Sequential diverse failure modes followed by clean success.
 9. Concurrent failure isolation (Thread A/C = SUCCESS, Thread B/D = FAILURE).
 10. Error state isolation (zero state leakage from failed requests into later successes).
 11. Error message safety (no credentials or secrets exposed in error structures).
 12. Mutation safety during validation errors on caller-owned inputs.

Constraints:
 - 100% Offline: Zero external APIs, network, real LLMs, or production secrets.
 - Zero production code modified.
 - Only observable behavior guaranteed by existing public contracts tested.
"""

from __future__ import annotations

import concurrent.futures
import copy
import sys
from pathlib import Path
from typing import Any

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
    VectorSearchResult,
)
from ingestion.ingestion_errors import (
    IngestionChunkingError,
    IngestionEmbeddingError,
    IngestionError,
    IngestionExtractionError,
    IngestionPipelineError,
    IngestionValidationError,
)
from ingestion.exceptions import CorruptedPDFError, InvalidFileTypeError, PDFNotFoundError
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.retrieval_processor import build_retrieval_context, process_retrieval_results

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
from vision.result_normalizer import VisionResultNormalizer
from vision.exceptions import (
    VisionAgentError,
    VisionCancellationError,
    VisionError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderError,
    VisionTimeoutError,
)
from vision.lifecycle import (
    VisionCancellationToken,
    VisionExecutionLifecycle,
    VisionExecutionStage,
)

# ---------------------------------------------------------------------------
# Synthetic Test Helpers
# ---------------------------------------------------------------------------

DOC_VALID = "DAY29_VALID_DOC"
FILE_VALID = "day29_valid.pdf"


def _make_valid_citation(chunk_id: str = "ck_01", content_type: str = "text") -> AgentCitation:
    return AgentCitation(
        document_id=DOC_VALID,
        filename=FILE_VALID,
        chunk_id=chunk_id,
        page_number=1,
        content_type=content_type,
        score=0.95,
        metadata={"day29_marker": "VALID"},
    )


def _make_valid_evidence(chunk_id: str = "ck_vis_01") -> VisualEvidence:
    return VisualEvidence(
        document_id=DOC_VALID,
        filename=FILE_VALID,
        chunk_id=chunk_id,
        page_number=1,
        content_type="image",
        metadata={"day29_marker": "VALID"},
    )


def _execute_clean_pipeline(query: str = "Clean valid query") -> tuple[AgentResponse, VisionResult]:
    """Executes a healthy end-to-end integration workflow."""
    vsr = VectorSearchResult(
        chunk_id="ck_01",
        score=0.92,
        document_id=DOC_VALID,
        filename=FILE_VALID,
        page_number=1,
        chunk_index=0,
        content_type="image",
        content="Valid image content.",
        metadata={"day29_marker": "CLEAN"},
    )
    processed = process_retrieval_results([vsr], min_score=0.5, max_results=10)
    ctx = build_retrieval_context(processed)

    citations = [AgentCitation.from_search_result(v) for v in processed]
    agent_resp = AgentResponse(
        answer=f"Answer for {query}",
        agent_name="SearchAgent",
        status="success",
        citations=citations,
        metadata={"query": query, "context": ctx},
    )

    evidence = VisualEvidenceAdapter.adapt_batch(agent_resp.image_results)
    normalizer = VisionResultNormalizer()
    raw_res = VisionResult(
        query=query,
        status="success",
        description="Visual chart summary.",
        evidence=evidence,
    )
    normalized_res = normalizer.normalize(raw_res)

    return agent_resp, normalized_res


# ===========================================================================
# 1. Public Exception Hierarchy Stability
# ===========================================================================

class TestPublicExceptionHierarchyStability:
    """Verifies documented inheritance chains for all public exception classes."""

    # Ingestion Exceptions
    def test_ingestion_error_hierarchy(self) -> None:
        assert issubclass(IngestionError, Exception)
        for exc in (
            IngestionChunkingError,
            IngestionEmbeddingError,
            IngestionExtractionError,
            IngestionPipelineError,
            IngestionValidationError,
        ):
            assert issubclass(exc, IngestionError)
            with pytest.raises(IngestionError):
                raise exc("Ingestion error test")

    def test_pdf_parsing_exceptions(self) -> None:
        for exc in (CorruptedPDFError, InvalidFileTypeError, PDFNotFoundError):
            assert issubclass(exc, Exception)

    # Agents Exceptions
    def test_agent_error_hierarchy(self) -> None:
        assert issubclass(AgentError, Exception)
        for exc in (AgentValidationError, AgentExecutionError, AgentRoutingError):
            assert issubclass(exc, AgentError)
            with pytest.raises(AgentError):
                raise exc("Agent error test")

    # Vision Exceptions
    def test_vision_error_hierarchy(self) -> None:
        assert issubclass(VisionAgentError, Exception)
        assert VisionError is VisionAgentError
        for exc in (
            VisionEvidenceError,
            VisionInputValidationError,
            VisionProcessingError,
            VisionProviderError,
            VisionTimeoutError,
            VisionCancellationError,
        ):
            assert issubclass(exc, VisionAgentError)
            with pytest.raises(VisionAgentError):
                raise exc("Vision error test")


# ===========================================================================
# 2. Invalid Input Validation Matrix
# ===========================================================================

class TestInvalidInputValidationMatrix:
    """Verifies typed validation exception triggers on malformed / out-of-range inputs."""

    # Ingestion validation
    def test_ingestion_validate_chunks_invalid_input_type(self) -> None:
        res = validate_chunks("not a list or chunking result")  # type: ignore[arg-type]
        assert isinstance(res, ChunkValidationResult)
        assert not res.is_valid
        assert len(res.errors) > 0

    # AgentCitation validations
    def test_agent_citation_empty_document_id(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="f.pdf", chunk_id="ck")

    def test_agent_citation_empty_filename(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="doc", filename="", chunk_id="ck")

    def test_agent_citation_empty_chunk_id(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="doc", filename="f.pdf", chunk_id="")

    def test_agent_citation_invalid_page_number_zero(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="doc", filename="f.pdf", chunk_id="ck", page_number=0)

    def test_agent_citation_invalid_score_infinite(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="doc", filename="f.pdf", chunk_id="ck", score=float("nan"))

    # AgentRequest validations
    def test_agent_request_empty_query(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentRequest(query="")

    def test_agent_request_invalid_session_id(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentRequest(query="q", session_id="")

    # SearchRequest validations
    def test_search_request_invalid_top_k_zero(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="q", top_k=0)

    def test_search_request_invalid_min_score_out_of_range(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="q", min_score=1.5)

    # AgentResponse validations
    def test_agent_response_empty_agent_name(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentResponse(answer="a", agent_name="")

    # VisualEvidence validations
    def test_visual_evidence_empty_document_id(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename="f.pdf", chunk_id="ck", content_type="image")

    def test_visual_evidence_invalid_content_type(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck", content_type="audio")

    # VisionRequest & VisionResult validations
    def test_vision_request_empty_query(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="")

    def test_vision_result_empty_query(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionResult(query="")


# ===========================================================================
# 3. Not-Found & Empty Result Contracts
# ===========================================================================

class TestNotFoundAndEmptyResultContracts:
    """Verifies that empty retrieval/evidence results return well-formed empty contracts."""

    def test_empty_retrieval_processing_returns_empty_list(self) -> None:
        res = process_retrieval_results([])
        assert res == []

    def test_empty_retrieval_context_returns_empty_string(self) -> None:
        ctx = build_retrieval_context([])
        assert ctx == ""

    def test_search_result_empty_citations_status_no_results(self) -> None:
        sr = SearchResult(query="empty query", citations=[])
        assert sr.status == "NO_RESULTS"
        assert sr.has_results is False
        assert sr.total_results == 0
        assert sr.unique_document_count == 0

    def test_agent_response_empty_citations(self) -> None:
        resp = AgentResponse(answer="no answer", agent_name="Agent", citations=[])
        assert resp.has_citations is False
        assert resp.total_citations == 0
        assert resp.text_results == []
        assert resp.image_results == []

    def test_vision_request_empty_evidence(self) -> None:
        vr = VisionRequest(query="query without evidence", evidence=[])
        assert vr.has_evidence is False
        assert vr.total_evidence == 0

    def test_vision_result_no_evidence_status(self) -> None:
        v_res = VisionResult(query="query", status="no_evidence", evidence=[])
        assert v_res.status == "no_evidence"
        assert v_res.has_evidence is False
        assert v_res.document_id == ""


# ===========================================================================
# 4. Serialization Error Handling
# ===========================================================================

class TestSerializationErrorHandling:
    """Verifies that public deserializers strictly reject invalid (non-dict) payloads."""

    def test_agent_citation_from_dict_non_dict_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation.from_dict("invalid")  # type: ignore[arg-type]

    def test_agent_request_from_dict_non_dict_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentRequest.from_dict(["query", "test"])  # type: ignore[arg-type]

    def test_search_request_from_dict_non_dict_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest.from_dict(12345)  # type: ignore[arg-type]

    def test_agent_response_from_dict_non_dict_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentResponse.from_dict(None)  # type: ignore[arg-type]

    def test_search_result_from_dict_non_dict_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchResult.from_dict("not_a_dict")  # type: ignore[arg-type]

    def test_visual_evidence_from_dict_non_dict_rejected(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence.from_dict([1, 2, 3])  # type: ignore[arg-type]

    def test_vision_request_from_dict_non_dict_rejected(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest.from_dict(True)  # type: ignore[arg-type]

    def test_vision_result_from_dict_non_dict_rejected(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionResult.from_dict("error")  # type: ignore[arg-type]


# ===========================================================================
# 5. Error Information & Structured Reporting
# ===========================================================================

class TestErrorInformationAndStructuredReporting:
    """Verifies that error models preserve structured error messages and status codes."""

    def test_agent_response_error_contract(self) -> None:
        resp = AgentResponse(
            answer="",
            agent_name="SearchAgent",
            status="error",
            error="Downstream retrieval timeout",
            citations=[],
        )
        assert resp.is_error is True
        assert resp.is_success is False
        assert resp.status == "error"
        assert resp.error == "Downstream retrieval timeout"

    def test_vision_result_error_contract(self) -> None:
        v_res = VisionResult(
            query="visual query",
            status="error",
            error="Image decoding failed",
            evidence=[],
        )
        assert v_res.is_error is True
        assert v_res.is_success is False
        assert v_res.status == "error"
        assert v_res.error == "Image decoding failed"


# ===========================================================================
# 6. Error Serialization Roundtrip
# ===========================================================================

class TestErrorSerializationRoundtrip:
    """Verifies that structured error information survives serialization roundtrips."""

    def test_agent_response_error_roundtrip(self) -> None:
        resp = AgentResponse(
            answer="",
            agent_name="SearchAgent",
            status="error",
            error="Critical provider failure",
            citations=[],
            metadata={"err_code": 500},
        )
        d = resp.to_dict()
        restored = AgentResponse.from_dict(d)

        assert restored.is_error is True
        assert restored.status == "error"
        assert restored.error == "Critical provider failure"
        assert restored.metadata["err_code"] == 500

    def test_vision_result_error_roundtrip(self) -> None:
        v_res = VisionResult(
            query="visual query",
            status="error",
            error="Vision provider timeout",
            evidence=[],
            metadata={"timeout_s": 10},
        )
        d = v_res.to_dict()
        restored = VisionResult.from_dict(d)

        assert restored.is_error is True
        assert restored.status == "error"
        assert restored.error == "Vision provider timeout"
        assert restored.metadata["timeout_s"] == 10


# ===========================================================================
# 7. Recovery Workflows (FAILURE -> SUCCESS, etc.)
# ===========================================================================

class TestFailureRecoveryWorkflows:
    """Verifies that preceding errors do not contaminate subsequent healthy requests."""

    def test_failure_then_success(self) -> None:
        # Step 1: Trigger failure
        with pytest.raises(AgentValidationError):
            AgentRequest(query="")

        # Step 2: Clean success
        resp, vis = _execute_clean_pipeline("Query after failure")
        assert resp.is_success
        assert vis.is_success
        assert resp.error is None
        assert vis.error is None

    def test_success_failure_success_sequence(self) -> None:
        # Step 1: Success 1
        resp1, vis1 = _execute_clean_pipeline("First success")
        assert resp1.is_success
        assert vis1.is_success

        # Step 2: Failure
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck", content_type="invalid_type")

        # Step 3: Success 2
        resp2, vis2 = _execute_clean_pipeline("Second success")
        assert resp2.is_success
        assert vis2.is_success
        assert resp2.error is None

    def test_repeated_failure_then_success(self) -> None:
        # Failure 1
        with pytest.raises(AgentValidationError):
            SearchRequest(query="q", top_k=-1)

        # Failure 2
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="")

        # Clean Success
        resp, vis = _execute_clean_pipeline("Recovery after double failure")
        assert resp.is_success
        assert vis.is_success


# ===========================================================================
# 8. Sequential Diverse Failure Modes
# ===========================================================================

class TestSequentialDiverseFailureModes:
    """Verifies executing a sequence of different failure types leaves the system in a clean state."""

    def test_diverse_failures_sequence(self) -> None:
        # 1. Validation failure (Agent)
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="f.pdf", chunk_id="ck")

        # 2. Deserialization failure (Vision)
        with pytest.raises(VisionEvidenceError):
            VisualEvidence.from_dict("not_a_dict")  # type: ignore[arg-type]

        # 3. Validation failure (Vision)
        with pytest.raises(VisionInputValidationError):
            VisionResult(query="")

        # 4. Ingestion chunk validation failure
        res = validate_chunks("not_a_list")  # type: ignore[arg-type]
        assert not res.is_valid

        # 5. Clean success
        resp, vis = _execute_clean_pipeline("Clean query after multiple failures")
        assert resp.is_success
        assert vis.is_success


# ===========================================================================
# 9. Concurrent Failure Isolation
# ===========================================================================

class TestConcurrentFailureIsolation:
    """Verifies failed requests running concurrently do not contaminate successful threads."""

    def test_concurrent_mixed_success_and_failure(self) -> None:
        def worker(thread_idx: int) -> tuple[int, str, Any]:
            if thread_idx in (1, 3):
                # Simulated failure on Threads 1 and 3
                try:
                    AgentCitation(document_id="", filename="f.pdf", chunk_id="ck")
                    return thread_idx, "unexpected_success", None
                except AgentValidationError as e:
                    return thread_idx, "caught_failure", str(e)
            else:
                # Clean success on Threads 0 and 2
                resp, vis = _execute_clean_pipeline(f"Concurrent thread {thread_idx}")
                return thread_idx, "success", (resp, vis)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            results = dict([(r[0], (r[1], r[2])) for r in [f.result() for f in futures]])

        assert len(results) == 4

        # Threads 1 and 3 failed with expected typed validation error
        assert results[1][0] == "caught_failure"
        assert results[3][0] == "caught_failure"

        # Threads 0 and 2 succeeded cleanly with full results
        status_0, (resp_0, vis_0) = results[0]
        status_2, (resp_2, vis_2) = results[2]
        assert status_0 == "success" and resp_0.is_success and vis_0.is_success
        assert status_2 == "success" and resp_2.is_success and vis_2.is_success


# ===========================================================================
# 10. Error Message Safety & Mutation Safety
# ===========================================================================

class TestErrorMessageAndMutationSafety:
    """Verifies errors do not leak secrets and caller-owned objects remain unmutated on error."""

    def test_error_message_no_secrets_leakage(self) -> None:
        secret_marker = "SK_LIVE_SECRET_API_KEY_12345"
        try:
            # Query containing synthetic secret
            AgentRequest(query="", metadata={"secret": secret_marker})
        except AgentValidationError as e:
            # The exception message for empty query is generic and does not reflect metadata
            assert secret_marker not in str(e)

    def test_caller_object_unmutated_on_error(self) -> None:
        caller_dict = {"document_id": "", "filename": "f.pdf", "chunk_id": "ck"}
        snapshot = copy.deepcopy(caller_dict)

        with pytest.raises(AgentValidationError):
            AgentCitation.from_dict(caller_dict)

        # Confirm caller dictionary was not altered
        assert caller_dict == snapshot
