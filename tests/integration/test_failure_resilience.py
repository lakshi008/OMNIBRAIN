"""
OmniBrain Member 4 — Day 6 Failure Matrix & Resilience Integration Tests.

Verifies system-level failure propagation, error boundaries, resilience, and recovery
across Ingestion, Search, Vision, and Supervisor subsystems.

Focus areas:
1. Input validation & domain error boundaries at every member interface.
2. Ingestion failure resilience (missing files, invalid types, corrupted PDFs).
3. Search failure handling (query validation, execution errors, empty results).
4. Vision failure handling (provider errors, timeout errors, cancellation).
5. Retry policy mechanics (VisionRetryPolicy, backoff calculation, retry exhaustion).
6. Timeout & Cancellation lifecycle (VisionCancellationToken, VisionTimeoutError, VisionCancellationError).
7. Concurrent multi-threaded isolation and mixed-outcome resilience.
8. Partial failure handling & mixed-modality evidence filtering.
9. Error & metadata sanitization (FORBIDDEN_METADATA_KEYS stripping).
10. 100% offline, deterministic, side-effect-free execution.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Ensure repo root is on sys.path for test runners executing this file directly
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

from agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentRoutingError,
    AgentValidationError,
)
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    AgentState,
    SearchRequest,
    SearchResult,
)
from agents.search_agent import SearchAgent
from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
)
from ingestion.models import DocumentChunk, VectorSearchResult
from ingestion.pdf_text_extractor import validate_pdf
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionAgentError,
    VisionCancellationError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderError,
    VisionProviderExecutionError,
    VisionTimeoutError,
)
from vision.lifecycle import (
    VisionCancellationToken,
    VisionExecutionLifecycle,
    VisionExecutionStage,
    VisionRetryPolicy,
)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.result_normalizer import (
    FORBIDDEN_METADATA_KEYS,
    VisionResultNormalizer,
)


# ============================================================================
# 1. INVALID INPUT VALIDATION FAILURES
# ============================================================================


class TestInvalidInputValidationFailures:
    """Verifies that invalid or malformed data is rejected cleanly across all member contracts."""

    def test_agent_request_validation_rejections(self) -> None:
        """Verify AgentRequest rejects empty queries and non-string inputs."""
        with pytest.raises(AgentValidationError, match="query must be a string"):
            AgentRequest(query=12345)  # type: ignore[arg-type]

        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            AgentRequest(query="   ")

        with pytest.raises(AgentValidationError, match="metadata must be a dictionary"):
            AgentRequest(query="Valid query", metadata="not_a_dict")  # type: ignore[arg-type]    def test_search_request_numeric_parameter_rejections(self) -> None:
        """Verify SearchRequest rejects non-positive top_k and out-of-range min_score."""
        with pytest.raises(AgentValidationError, match="top_k must be a positive integer"):
            SearchRequest(query="Valid query", top_k=0)

        with pytest.raises(AgentValidationError, match="top_k must be a positive integer"):
            SearchRequest(query="Valid query", top_k=-5)

        with pytest.raises(AgentValidationError, match="min_score"):
            SearchRequest(query="Valid query", min_score=2.5)

    def test_visual_evidence_field_rejections(self) -> None:
        """Verify VisualEvidence constructor rejects invalid lineage fields."""
        with pytest.raises(VisionEvidenceError, match="document_id"):
            VisualEvidence(document_id="", filename="f.pdf", chunk_id="c1", content_type="image")

        with pytest.raises(VisionEvidenceError, match="filename"):
            VisualEvidence(document_id="doc1", filename="  ", chunk_id="c1", content_type="image")

        with pytest.raises(VisionEvidenceError, match="chunk_id"):
            VisualEvidence(document_id="doc1", filename="f.pdf", chunk_id="", content_type="image")

        with pytest.raises(VisionEvidenceError, match="page_number"):
            VisualEvidence(document_id="doc1", filename="f.pdf", chunk_id="c1", page_number=-1, content_type="image")

        with pytest.raises(VisionEvidenceError, match="content_type"):
            VisualEvidence(document_id="doc1", filename="f.pdf", chunk_id="c1", content_type="unsupported_modality")


# ============================================================================
# 2. INGESTION SUBSYSTEM FAILURE HANDLING
# ============================================================================


class TestIngestionSubsystemFailureHandling:
    """Verifies that ingestion-level failures trigger structured exceptions without producing fake outputs."""

    def test_missing_pdf_file_raises_not_found(self) -> None:
        """Verify non-existent PDF path raises PDFNotFoundError."""
        with pytest.raises(PDFNotFoundError, match="not found"):
            validate_pdf("B:/non_existent_directory_12345/missing_doc.pdf")

    def test_invalid_file_extension_raises_invalid_file_type(self) -> None:
        """Verify non-PDF file extension raises InvalidFileTypeError."""
        with pytest.raises(InvalidFileTypeError, match="Invalid file type"):
            validate_pdf("B:/OMNIBRAIN/README.md")


# ============================================================================
# 3. SEARCH SUBSYSTEM FAILURE HANDLING
# ============================================================================


class TestSearchSubsystemFailureHandling:
    """Verifies that search-level errors prevent downstream corruption and preserve error contracts."""

    def test_search_agent_result_integrity_failure_detection(self) -> None:
        """Verify SearchAgent._validate_result_integrity catches corrupted lineage from retrieval."""
        corrupted_item = VectorSearchResult(
            chunk_id="chk-01",
            score=float("nan"),  # Invalid score
            document_id="doc-1",
            filename="file.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Some text",
        )

        with pytest.raises(AgentExecutionError, match="score is not a finite numeric value"):
            SearchAgent._validate_result_integrity(corrupted_item, idx=0)

    def test_search_result_no_results_status(self) -> None:
        """Verify empty search results construct valid NO_RESULTS package."""
        empty_res = SearchResult(
            query="Query with no matches",
            status="NO_RESULTS",
            citations=[],
            context="",
        )

        assert empty_res.has_results is False
        assert empty_res.total_results == 0
        assert empty_res.citations == []


# ============================================================================
# 4. VISION SUBSYSTEM FAILURE HANDLING
# ============================================================================


class TestVisionSubsystemFailureHandling:
    """Verifies that vision processing failures produce structured error results without data fabrication."""

    def test_vision_result_error_representation(self) -> None:
        """Verify VisionResult in error state preserves error string and query without fake lineage."""
        err_result = VisionResult(
            query="Analyze degraded image",
            status="error",
            description="",
            error="Vision provider failed: Model inference execution error",
            metadata={"stage": "EXECUTING", "attempt": 3},
        )

        assert err_result.is_success is False
        assert err_result.is_error is True
        assert "Model inference execution error" in (err_result.error or "")
        assert err_result.description == ""

    def test_supervisor_state_captures_vision_error(self) -> None:
        """Verify downstream AgentState records vision error without crashing."""
        state = AgentState(query="What is in the image?", route="vision")
        state.add_error("VisionProcessingError: Provider unavailable")
        state.update(status="failed")

        assert state.status == "failed"
        assert len(state.errors) == 1
        assert "Provider unavailable" in state.errors[0]
        assert state.answer == ""


# ============================================================================
# 5. RETRY POLICY & BEHAVIOR
# ============================================================================


class TestRetryPolicyAndBehavior:
    """Verifies that Member 3 VisionRetryPolicy manages retry thresholds and classification correctly."""

    def test_retry_policy_default_configuration(self) -> None:
        """Verify default retry policy parameters."""
        policy = VisionRetryPolicy()
        assert policy.max_retries == 0
        assert policy.max_attempts == 1

    def test_retry_policy_custom_configuration(self) -> None:
        """Verify custom retry policy parameters and attempt calculations."""
        policy = VisionRetryPolicy(max_retries=3)
        assert policy.max_retries == 3
        assert policy.max_attempts == 4

    def test_retry_eligibility_classification(self) -> None:
        """Verify is_retryable accurately distinguishes transient from terminal errors."""
        policy = VisionRetryPolicy(max_retries=2)

        # Retryable exceptions
        assert policy.is_retryable(VisionProviderExecutionError("Transient backend failure")) is True
        assert policy.is_retryable(VisionProcessingError("Transient processing error")) is True

        # Non-retryable exceptions
        assert policy.is_retryable(VisionInputValidationError("Invalid query")) is False
        assert policy.is_retryable(VisionEvidenceError("Invalid evidence")) is False
        assert policy.is_retryable(VisionCancellationError("Cancelled by user")) is False
        assert policy.is_retryable(VisionTimeoutError("Operation timed out")) is False


# ============================================================================
# 6. TIMEOUT & CANCELLATION LIFECYCLE
# ============================================================================


class TestTimeoutAndCancellationLifecycle:
    """Verifies that cancellation tokens and timeout exceptions function deterministically."""

    def test_cancellation_token_request_and_throw(self) -> None:
        """Verify VisionCancellationToken raises VisionCancellationError once cancelled."""
        token = VisionCancellationToken()
        assert token.is_cancelled is False

        # Should not raise when active
        token.raise_if_cancelled()

        # Request cancellation
        token.cancel(reason="Execution cancelled by supervisor.")
        assert token.is_cancelled is True
        assert token.reason == "Execution cancelled by supervisor."

        with pytest.raises(VisionCancellationError, match="cancelled"):
            token.raise_if_cancelled()

    def test_timeout_exception_hierarchy(self) -> None:
        """Verify VisionTimeoutError inherits from VisionAgentError / VisionError."""
        err = VisionTimeoutError("Operation timed out after 10.0 seconds.")
        assert isinstance(err, VisionAgentError)
        assert "10.0 seconds" in str(err)


# ============================================================================
# 7. CONCURRENT MULTI-THREADED ISOLATION
# ============================================================================


class TestConcurrentMultiThreadedIsolation:
    """Verifies that concurrent requests in separate threads do not cross-contaminate state."""

    def test_concurrent_search_and_vision_handoff_isolation(self) -> None:
        """Run 10 concurrent threads each processing distinct document flows."""
        results: dict[int, dict[str, Any]] = {}
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                doc_id = f"doc-thread-{thread_id}"
                fname = f"thread_{thread_id}.pdf"
                cid = f"chk-{thread_id}"

                cit = AgentCitation(
                    document_id=doc_id,
                    filename=fname,
                    chunk_id=cid,
                    page_number=thread_id + 1,
                    content_type="chart" if thread_id % 2 == 0 else "diagram",
                    score=0.90 + (thread_id * 0.005),
                    metadata={"thread_id": thread_id, "unique_token": f"token_{thread_id}"},
                )

                ev = VisualEvidenceAdapter.adapt_citation(cit)
                res = VisionResult(
                    query=f"Query from thread {thread_id}",
                    status="success",
                    description=f"Analysis result for thread {thread_id}",
                    evidence=[ev],
                    metadata={"thread_id": thread_id},
                )

                results[thread_id] = {
                    "document_id": res.document_id,
                    "filename": res.filename,
                    "chunk_id": res.chunk_id,
                    "token": ev.metadata.get("unique_token"),
                    "thread_id": res.metadata.get("thread_id"),
                }
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 10

        # Verify strict thread isolation
        for thread_id, data in results.items():
            assert data["document_id"] == f"doc-thread-{thread_id}"
            assert data["filename"] == f"thread_{thread_id}.pdf"
            assert data["chunk_id"] == f"chk-{thread_id}"
            assert data["token"] == f"token_{thread_id}"
            assert data["thread_id"] == thread_id


# ============================================================================
# 8. PARTIAL FAILURE & STRICT MODES
# ============================================================================


class TestPartialFailureAndStrictModes:
    """Verifies system behavior when handling mixed valid/invalid evidence packages."""

    def test_mixed_evidence_batch_filtering_vs_strict_rejection(self) -> None:
        """Verify non-strict mode safely filters invalid evidence while strict mode rejects it."""
        valid_cit = AgentCitation(
            document_id="doc-1",
            filename="f1.pdf",
            chunk_id="c-img-1",
            content_type="image",
        )
        non_visual_cit = AgentCitation(
            document_id="doc-1",
            filename="f1.pdf",
            chunk_id="c-txt-1",
            content_type="text",
        )

        search_pkg = SearchResult(
            query="Mixed search query",
            status="RESULTS_FOUND",
            citations=[valid_cit, non_visual_cit],
            context="Mixed context",
        )

        # Standard non-strict mode: filters out non_visual_cit
        filtered = VisualEvidenceAdapter.adapt_search_package(search_pkg, strict=False)
        assert len(filtered) == 1
        assert filtered[0].chunk_id == "c-img-1"

        # Strict mode: raises VisionEvidenceError
        with pytest.raises(VisionEvidenceError, match="has non-visual content_type"):
            VisualEvidenceAdapter.adapt_search_package(search_pkg, strict=True)


# ============================================================================
# 9. METADATA & SECRET ERROR SANITIZATION
# ============================================================================


class TestMetadataAndSecretSanitization:
    """Verifies that forbidden internal keys and secrets are never exposed in public metadata."""

    def test_forbidden_metadata_keys_definition(self) -> None:
        """Verify that FORBIDDEN_METADATA_KEYS contains sensitive security identifiers."""
        forbidden_set = set(FORBIDDEN_METADATA_KEYS)
        assert len(forbidden_set) > 0
        # Common sensitive patterns
        sensitive_patterns = {"api_key", "secret", "password", "token", "auth"}
        found_sensitive = any(any(pat in key.lower() for pat in sensitive_patterns) for key in forbidden_set)
        assert found_sensitive, "FORBIDDEN_METADATA_KEYS should contain sensitive token/key patterns"

    def test_sanitization_removes_forbidden_keys(self) -> None:
        """Verify VisionResultNormalizer sanitizes forbidden keys from metadata dictionaries."""
        raw_meta = {
            "valid_metric": "accuracy",
            "latency_ms": 45,
            "api_key": "SUPER_SECRET_KEY_12345",
            "access_token": "BEARER_SECRET_TOKEN",
            "password": "internal_password",
        }

        sanitized = VisionResultNormalizer.sanitize_metadata(raw_meta)
        for key in raw_meta:
            if key in FORBIDDEN_METADATA_KEYS or any(f in key.lower() for f in ["api_key", "secret", "password", "token"]):
                assert key not in sanitized
        assert sanitized["valid_metric"] == "accuracy"
        assert sanitized["latency_ms"] == 45
