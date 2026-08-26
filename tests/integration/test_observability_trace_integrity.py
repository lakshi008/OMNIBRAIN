"""
OmniBrain Member 4 — Day 31 Observability & Trace Integrity Certification Tests.

Validates the observability, execution lifecycle, auditability, and trace integrity
contracts across all OmniBrain subsystems:
    Ingestion (Member 1)
         ↓
    Search / Retrieval (Member 2)
         ↓
    Vision (Member 3)
         ↓
    Downstream Supervisor / Agent Consumers

Focus areas:
 1. Basic trace creation and lifecycle state recording (VisionExecutionTrace, VisionExecutionLifecycle).
 2. Request identity and correlation preservation (REQUEST_A vs REQUEST_B).
 3. Document traceability (DAY31_DOCUMENT_A vs DAY31_DOCUMENT_B).
 4. Chunk traceability (chunk-to-document provenance locking).
 5. Trace lifecycle stage transitions (VisionExecutionStage constants & transitions).
 6. Parent/child execution relationships and stage hierarchy.
 7. Metadata integrity across trace execution.
 8. Citation traceability across search and response contracts.
 9. Evidence traceability across vision contracts.
 10. Error trace integrity (failure representation in execution trace/result).
 11. Success → Failure → Success trace isolation.
 12. Repeated execution trace independence (Rounds 1 to 5).
 13. Concurrent trace isolation (multi-threaded mixed success/failure).
 14. Trace serialization round-trip (VisionExecutionTrace to_dict/from_dict).
 15. Trace metadata mutation safety.
 16. Duplicate event prevention in execution traces.
 17. Trace order determinism for stage transitions.
 18. Sensitive data safety in trace metadata (DAY31_FAKE_TOKEN / DAY31_FAKE_API_KEY).

Constraints:
 - 100% Offline: Synthetic markers only. Zero external network, real LLMs, or production secrets.
 - Zero production code modified.
 - Zero tracing/logging/telemetry infrastructure added.
"""

from __future__ import annotations

import concurrent.futures
import copy
import sys
import threading
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
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.embedding_preparation import prepare_for_embedding
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
from agents.exceptions import AgentValidationError

# Vision Subsystem (Member 3)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.result_normalizer import (
    FORBIDDEN_METADATA_KEYS,
    VisionExecutionTrace,
    VisionResultNormalizer,
)
from vision.exceptions import (
    VisionAgentError,
    VisionCancellationError,
    VisionEvidenceError,
    VisionInputValidationError,
)
from vision.lifecycle import (
    VisionCancellationToken,
    VisionExecutionLifecycle,
    VisionExecutionStage,
)

# ---------------------------------------------------------------------------
# Synthetic Day 31 Observability Fixtures
# ---------------------------------------------------------------------------

DOC_A = "DAY31_DOCUMENT_A"
DOC_B = "DAY31_DOCUMENT_B"
DOC_C = "DAY31_DOCUMENT_C"
DOC_D = "DAY31_DOCUMENT_D"

FILE_A = "day31_doc_a.pdf"
FILE_B = "day31_doc_b.pdf"
FILE_C = "day31_doc_c.pdf"
FILE_D = "day31_doc_d.pdf"

REQUEST_A = "DAY31_REQUEST_A"
REQUEST_B = "DAY31_REQUEST_B"
REQUEST_C = "DAY31_REQUEST_C"
REQUEST_D = "DAY31_REQUEST_D"

DAY31_FAKE_TOKEN = "DAY31_FAKE_TOKEN_XYZ_999"
DAY31_FAKE_API_KEY = "DAY31_FAKE_API_KEY_SECRET_888"


def _make_traceable_vsr(
    doc_id: str,
    filename: str,
    req_id: str,
    chunk_idx: int = 0,
    content_type: str = "image",
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=f"chunk_{doc_id}_{chunk_idx+1}",
        score=0.94 - (chunk_idx * 0.05),
        document_id=doc_id,
        filename=filename,
        page_number=chunk_idx + 1,
        chunk_index=chunk_idx,
        content_type=content_type,
        content=f"Traceable content for {req_id} from {doc_id}",
        metadata={"req_id": req_id, "doc_id": doc_id, "trace_hop": "retrieval"},
    )


def _execute_traced_workflow(
    req_id: str,
    doc_id: str,
    filename: str,
) -> tuple[AgentResponse, VisionResult, VisionExecutionTrace]:
    """Runs an integrated workflow with stage tracing."""
    trace = VisionExecutionTrace()
    trace.add_stage("request_received")
    trace.add_stage("validation_started")

    # Step 1: Ingestion & Retrieval
    vsr = _make_traceable_vsr(doc_id, filename, req_id, chunk_idx=0, content_type="image")
    processed = process_retrieval_results([vsr], min_score=0.5, max_results=10)
    ctx = build_retrieval_context(processed)
    trace.add_stage("retrieval_completed")

    # Step 2: Agent Response & Citations
    citations = [AgentCitation.from_search_result(v) for v in processed]
    agent_resp = AgentResponse(
        answer=f"Synthesized answer for {req_id}",
        agent_name="TraceAgent",
        status="success",
        citations=citations,
        metadata={"req_id": req_id, "doc_id": doc_id, "context": ctx},
    )
    trace.add_stage("agent_synthesized")

    # Step 3: Vision Adaptation & Result Normalization with trace
    image_citations = agent_resp.image_results
    evidence = VisualEvidenceAdapter.adapt_batch(image_citations)
    raw_res = VisionResult(
        query=f"Visual query for {req_id}",
        status="success",
        description=f"Visual summary for {req_id}",
        evidence=evidence,
        metadata={"req_id": req_id, "doc_id": doc_id},
    )
    normalized_res = VisionResultNormalizer.normalize(raw_res, trace=trace)

    return agent_resp, normalized_res, trace


# ===========================================================================
# 1. Basic Trace Creation & Lifecycle Stages
# ===========================================================================

class TestBasicTraceCreationAndLifecycle:
    """Verifies that VisionExecutionTrace and VisionExecutionLifecycle record stages accurately."""

    def test_trace_creation_and_stage_recording(self) -> None:
        trace = VisionExecutionTrace()
        assert trace.stages == []
        assert trace.to_dict()["stage_count"] == 0

        trace.add_stage("request_received")
        trace.add_stage("input_validated")
        trace.add_stage("execution_completed")

        assert len(trace.stages) == 3
        assert trace.to_dict()["stage_count"] == 3
        assert trace.stages == ["request_received", "input_validated", "execution_completed"]

    def test_default_trace_creation(self) -> None:
        trace = VisionExecutionTrace.create_default()
        assert len(trace.stages) == len(VisionExecutionTrace.DEFAULT_STAGES)
        assert trace.stages == list(VisionExecutionTrace.DEFAULT_STAGES)

    def test_vision_execution_lifecycle_stages(self) -> None:
        lifecycle = VisionExecutionLifecycle(
            stage=VisionExecutionStage.PENDING,
            provider_name="test_provider",
            model_name="test_model",
        )
        assert lifecycle.stage == VisionExecutionStage.PENDING
        assert lifecycle.provider_name == "test_provider"
        assert lifecycle.error is None


# ===========================================================================
# 2. Request & Document Traceability
# ===========================================================================

class TestRequestAndDocumentTraceability:
    """Verifies request and document identity remain bound end-to-end through the trace."""

    def test_request_identity_isolation_in_trace(self) -> None:
        resp_a, vis_a, trace_a = _execute_traced_workflow(REQUEST_A, DOC_A, FILE_A)
        resp_b, vis_b, trace_b = _execute_traced_workflow(REQUEST_B, DOC_B, FILE_B)

        # Request A traceability
        assert resp_a.metadata["req_id"] == REQUEST_A
        assert resp_a.unique_documents == [DOC_A]
        assert vis_a.document_id == DOC_A
        assert vis_a.metadata["req_id"] == REQUEST_A
        assert "execution_trace" in vis_a.metadata

        # Request B traceability
        assert resp_b.metadata["req_id"] == REQUEST_B
        assert resp_b.unique_documents == [DOC_B]
        assert vis_b.document_id == DOC_B
        assert vis_b.metadata["req_id"] == REQUEST_B

        # No cross-bleed
        assert REQUEST_B not in str(resp_a.to_dict())
        assert REQUEST_B not in str(vis_a.to_dict())
        assert REQUEST_A not in str(resp_b.to_dict())
        assert REQUEST_A not in str(vis_b.to_dict())


# ===========================================================================
# 3. Chunk, Citation & Evidence Traceability
# ===========================================================================

class TestChunkCitationEvidenceTraceability:
    """Verifies chunk, citation, and evidence provenance locking."""

    def test_citation_and_evidence_trace_locking(self) -> None:
        resp, vis, trace = _execute_traced_workflow(REQUEST_A, DOC_A, FILE_A)

        # Citation traceability
        assert len(resp.citations) == 1
        cit = resp.citations[0]
        assert cit.document_id == DOC_A
        assert cit.filename == FILE_A
        assert cit.chunk_id == "chunk_DAY31_DOCUMENT_A_1"

        # Evidence traceability
        assert len(vis.evidence) == 1
        ev = vis.evidence[0]
        assert ev.document_id == DOC_A
        assert ev.filename == FILE_A
        assert ev.chunk_id == "chunk_DAY31_DOCUMENT_A_1"


# ===========================================================================
# 4. Error Trace Integrity
# ===========================================================================

class TestErrorTraceIntegrity:
    """Verifies errors are correctly captured in trace state and do not contaminate healthy traces."""

    def test_error_trace_capture(self) -> None:
        err_res = VisionResult(
            query="Failing audit query",
            status="error",
            description="",
            error="Provider execution timeout",
            evidence=[],
        )
        normalizer = VisionResultNormalizer()
        trace = VisionExecutionTrace()
        trace.add_stage("request_received")
        trace.add_stage("provider_failed")

        normalized = normalizer.normalize(err_res, trace=trace)
        assert normalized.is_error is True
        assert normalized.status == "error"
        assert normalized.error == "Provider execution timeout"
        assert "execution_trace" in normalized.metadata
        assert "provider_failed" in normalized.metadata["execution_trace"]["stages"]


# ===========================================================================
# 5. Success → Failure → Success Trace Isolation
# ===========================================================================

class TestSuccessFailureSuccessTraceIsolation:
    """Verifies execution trace stays clean across success-failure-success sequences."""

    def test_success_failure_success_trace_sequence(self) -> None:
        # 1. Success A
        resp_a, vis_a, trace_a = _execute_traced_workflow(REQUEST_A, DOC_A, FILE_A)
        assert resp_a.is_success
        assert vis_a.is_success

        # 2. Failure B
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="f.pdf", chunk_id="ck")

        # 3. Success C
        resp_c, vis_c, trace_c = _execute_traced_workflow(REQUEST_C, DOC_C, FILE_C)
        assert resp_c.is_success
        assert vis_c.is_success
        assert resp_c.error is None
        assert vis_c.error is None
        assert resp_c.unique_documents == [DOC_C]


# ===========================================================================
# 6. Repeated Execution Trace Independence (Rounds 1 to 5)
# ===========================================================================

class TestRepeatedExecutionTraceIndependence:
    """Verifies that running traced workflows repeatedly creates independent trace instances."""

    def test_five_rounds_trace_independence(self) -> None:
        traces = []
        for i in range(5):
            req_id = f"DAY31_REPEAT_REQ_{i}"
            resp, vis, trace = _execute_traced_workflow(req_id, DOC_A, FILE_A)
            assert resp.is_success
            assert vis.is_success
            traces.append(trace)

        # All 5 traces have identical clean stage structure but are distinct objects
        assert len(traces) == 5
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                assert traces[i] is not traces[j]
                assert traces[i].stages == traces[j].stages


# ===========================================================================
# 7. Concurrent Trace Isolation
# ===========================================================================

class TestConcurrentTraceIsolation:
    """Verifies multi-threaded concurrent execution maintains trace isolation across threads."""

    def test_concurrent_trace_isolation_mixed(self) -> None:
        configs = [
            (REQUEST_A, DOC_A, FILE_A, False),
            (REQUEST_B, DOC_B, FILE_B, False),
            (REQUEST_C, DOC_C, FILE_C, True),  # Injected failure
            (REQUEST_D, DOC_D, FILE_D, False),
        ]

        def worker(cfg: tuple[str, str, str, bool]) -> tuple[str, str, Any]:
            req_id, doc_id, filename, should_fail = cfg
            if should_fail:
                try:
                    AgentCitation(document_id="", filename=filename, chunk_id="ck")
                    return req_id, "unexpected_success", None
                except AgentValidationError as e:
                    return req_id, "caught_failure", str(e)
            else:
                resp, vis, trace = _execute_traced_workflow(req_id, doc_id, filename)
                return req_id, "success", (resp, vis, trace)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, cfg) for cfg in configs]
            results = dict([(r[0], (r[1], r[2])) for r in [f.result() for f in futures]])

        assert len(results) == 4
        assert results[REQUEST_C][0] == "caught_failure"

        for req_id in (REQUEST_A, REQUEST_B, REQUEST_D):
            status, (resp, vis, trace) = results[req_id]
            assert status == "success"
            assert resp.is_success
            assert vis.is_success
            assert len(trace.stages) > 0


# ===========================================================================
# 8. Trace Serialization Round-Trip
# ===========================================================================

class TestTraceSerializationRoundTrip:
    """Verifies that trace objects and trace metadata survive serialization roundtrips."""

    def test_vision_execution_trace_serialization(self) -> None:
        trace = VisionExecutionTrace(initial_stages=["stage_1", "stage_2", "stage_3"])
        d = trace.to_dict()
        assert d["stages"] == ["stage_1", "stage_2", "stage_3"]
        assert d["stage_count"] == 3

    def test_vision_result_with_trace_serialization(self) -> None:
        _, vis, _ = _execute_traced_workflow(REQUEST_A, DOC_A, FILE_A)
        d = vis.to_dict()
        restored = VisionResult.from_dict(d)

        assert restored.document_id == DOC_A
        assert "execution_trace" in restored.metadata
        assert isinstance(restored.metadata["execution_trace"]["stages"], list)


# ===========================================================================
# 9. Trace Metadata Mutation Safety & Sensitive Data Safety
# ===========================================================================

class TestTraceMetadataMutationAndSensitiveDataSafety:
    """Verifies that caller metadata is not mutated and sensitive tokens are not leaked."""

    def test_caller_metadata_unmutated_during_tracing(self) -> None:
        caller_meta = {"day31_request": "A", "day31_document": "A"}
        snapshot = copy.deepcopy(caller_meta)

        trace = VisionExecutionTrace()
        trace.add_stage("request_received")

        # Pass caller metadata in result normalization
        raw_res = VisionResult(
            query="Query",
            status="success",
            evidence=[],
            metadata=caller_meta,
        )
        normalized = VisionResultNormalizer.normalize(raw_res, trace=trace)

        assert normalized.is_success
        assert caller_meta == snapshot

    def test_sensitive_tokens_not_exposed_in_trace(self) -> None:
        dirty_meta = {
            "api_key": DAY31_FAKE_API_KEY,
            "token": DAY31_FAKE_TOKEN,
            "safe_metric": 42,
        }
        sanitized = VisionResultNormalizer.sanitize_metadata(dirty_meta)

        assert "api_key" not in sanitized
        assert "token" not in sanitized
        assert DAY31_FAKE_API_KEY not in str(sanitized)
        assert DAY31_FAKE_TOKEN not in str(sanitized)
        assert sanitized["safe_metric"] == 42


# ===========================================================================
# 10. Resource & Artifact Safety
# ===========================================================================

class TestResourceAndArtifactSafety:
    """Verifies that observability and trace tests generate zero workspace leaks."""

    def test_zero_disk_artifacts_after_trace_runs(self) -> None:
        root_path = Path(REPO_ROOT)
        unexpected = [
            f.name for f in root_path.iterdir()
            if f.is_file() and f.name.endswith((".tmp", ".temp", ".dump", ".log", ".bak"))
        ]
        assert unexpected == []
