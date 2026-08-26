"""
OmniBrain Member 4 -- Day 15 Observability, Auditability & Trace-Integrity Integration Tests.

Verifies that the existing OMNIBRAIN integration contracts preserve correct observability,
traceability, evidence lineage, and request-level audit information across the integration pipeline.

Concern areas:
 1. Request traceability -- query identity preserved end-to-end
 2. Evidence traceability -- source document/chunk locked to evidence
 3. Citation lineage -- Document -> Chunk -> Citation -> Vision Result chain
 4. Metadata preservation -- synthetic trace markers survive integration boundaries
 5. Success trace -- consistent observable state on successful flow
 6. Failure trace -- error associated with correct request, no contamination
 7. Retry trace -- lifecycle stage transitions consistent (public retry counter: NOT APPLICABLE)
 8. Multi-document trace -- A->A, B->B with zero cross-contamination
 9. Multi-evidence trace -- each evidence item independently traceable
10. Repeated execution isolation -- deterministic, no stale trace
11. Concurrent trace isolation -- independent threads hold independent results
12. Serialization trace integrity -- trace-related info survives to_dict/from_dict
13. Error trace safety -- error messages contain no unrelated request/doc information
14. VisionExecutionTrace -- stage recording, serialization, isolation, normalizer integration
15. Observability interface -- NOT APPLICABLE beyond VisionExecutionTrace (no public logging/telemetry API)

Constraints:
 - 100% Offline: No external APIs, network, real LLMs, or production secrets.
 - Zero production code modified.
 - Zero tracing/logging/telemetry infrastructure added.
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

# Search / Agents Subsystem (Member 2)
from agents.models import (
    AgentCitation,
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
)
from vision.exceptions import VisionInputValidationError
from vision.lifecycle import (
    VisionExecutionLifecycle,
    VisionExecutionStage,
)
from vision.result_normalizer import (
    VisionExecutionTrace,
    VisionResultNormalizer,
)


# ============================================================================
# Shared Fixtures & Helpers
# ============================================================================


def _chunk(
    chunk_id: str = "chk-001",
    document_id: str = "doc-001",
    filename: str = "report.pdf",
    page_number: int | None = 1,
    chunk_index: int = 0,
    content: str = "Trace content.",
    content_type: str = "image",
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
        metadata=metadata if metadata is not None else {"trace_test": "DEFAULT"},
    )


def _vsr(
    chunk_id: str = "chk-001",
    score: float = 0.90,
    document_id: str = "doc-001",
    filename: str = "report.pdf",
    page_number: int | None = 1,
    chunk_index: int = 0,
    content: str = "Trace content.",
    content_type: str = "image",
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
        metadata=metadata if metadata is not None else {"trace_test": "DEFAULT"},
    )


def _ev(
    chunk_id: str = "chk-001",
    document_id: str = "doc-001",
    filename: str = "report.pdf",
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
        metadata=metadata if metadata is not None else {"trace_test": "DEFAULT"},
    )


# ============================================================================
# 1. REQUEST TRACEABILITY
# ============================================================================


class TestRequestTraceability:
    """Verifies that query/request identity is preserved across the integration pipeline."""

    def test_query_identity_preserved_through_pipeline(self) -> None:
        query = "DAY15_TRACE_QUERY_ALPHA_12345"

        search_req = SearchRequest(query=query)
        assert search_req.query == query

        evidence = _ev(chunk_id="chk-trace-01", document_id="doc-trace-01")
        vision_req = VisionRequest(query=query, evidence=[evidence])
        assert vision_req.query == query

        result = VisionResult(
            query=vision_req.query,
            status="success",
            description="Traced result.",
            evidence=vision_req.evidence,
        )
        assert result.query == query


# ============================================================================
# 2. EVIDENCE TRACEABILITY
# ============================================================================


class TestEvidenceTraceability:
    """Verifies that each evidence item can be traced back to its originating source."""

    def test_evidence_lineage_from_chunk_to_result(self) -> None:
        chunk = _chunk(
            chunk_id="chk-ev-trace-01",
            document_id="doc-ev-trace-50",
            filename="evidence_doc.pdf",
            page_number=7,
            metadata={"trace_test": "DAY15_EV_A", "source": "TEST_DOCUMENT_A"},
        )
        vsr = _vsr(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            content=chunk.content,
            metadata=chunk.metadata,
        )
        ev = VisualEvidence.from_search_result(vsr)

        assert ev.document_id == "doc-ev-trace-50"
        assert ev.filename == "evidence_doc.pdf"
        assert ev.chunk_id == "chk-ev-trace-01"
        assert ev.page_number == 7
        assert ev.metadata["trace_test"] == "DAY15_EV_A"
        assert ev.metadata["source"] == "TEST_DOCUMENT_A"


# ============================================================================
# 3. CITATION LINEAGE
# ============================================================================


class TestCitationLineage:
    """Verifies Document -> Chunk -> Citation -> Vision Result chain integrity."""

    def test_full_citation_lineage_chain(self) -> None:
        chunk = _chunk(
            chunk_id="chk-lin-01",
            document_id="doc-lin-100",
            filename="lineage_doc.pdf",
            page_number=4,
            chunk_index=2,
            metadata={"trace_test": "DAY15_LIN"},
        )
        vsr = _vsr(
            chunk_id=chunk.chunk_id,
            score=0.92,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            metadata=chunk.metadata,
        )
        citation = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_search_result(vsr)
        v_result = VisionResult(
            query="Analyze lineage chain",
            status="success",
            description="Lineage complete.",
            evidence=[ev],
        )

        assert citation.document_id == chunk.document_id
        assert citation.chunk_id == chunk.chunk_id
        assert citation.page_number == chunk.page_number
        assert citation.score == 0.92

        assert ev.document_id == chunk.document_id
        assert ev.chunk_id == chunk.chunk_id

        assert v_result.document_id == chunk.document_id
        assert v_result.filename == chunk.filename
        assert v_result.chunk_id == chunk.chunk_id
        assert v_result.page_number == chunk.page_number


# ============================================================================
# 4. METADATA PRESERVATION
# ============================================================================


class TestMetadataPreservation:
    """Verifies synthetic trace markers remain associated with the correct evidence/result."""

    def test_trace_metadata_preserved_across_boundaries(self) -> None:
        trace_meta = {
            "trace_test": "DAY15_META_ALPHA",
            "source": "TEST_DOCUMENT_ALPHA",
            "run_id": "RUN_DAY15_001",
        }
        ev = _ev(
            chunk_id="chk-meta-01",
            document_id="doc-meta-01",
            filename="meta_doc.pdf",
            metadata=trace_meta,
        )
        req = VisionRequest(query="Metadata trace test.", evidence=[ev])
        result = VisionResult(
            query=req.query,
            status="success",
            description="Metadata preserved.",
            evidence=req.evidence,
        )

        assert result.evidence[0].metadata["trace_test"] == "DAY15_META_ALPHA"
        assert result.evidence[0].metadata["source"] == "TEST_DOCUMENT_ALPHA"
        assert result.evidence[0].metadata["run_id"] == "RUN_DAY15_001"


# ============================================================================
# 5. SUCCESS TRACE
# ============================================================================


class TestSuccessTrace:
    """Verifies complete observable state consistency on a successful flow."""

    def test_success_trace_observable_state(self) -> None:
        chunk = _chunk(
            chunk_id="chk-suc-01",
            document_id="doc-suc-01",
            filename="success_report.pdf",
            page_number=2,
            metadata={"trace_test": "DAY15_SUCCESS"},
        )
        vsr = _vsr(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            metadata=chunk.metadata,
        )
        citation = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_search_result(vsr)
        result = VisionResult(
            query="Success trace query",
            status="success",
            description="Success trace verified.",
            evidence=[ev],
        )

        assert result.query == "Success trace query"
        assert result.document_id == "doc-suc-01"
        assert result.filename == "success_report.pdf"
        assert result.has_evidence is True
        assert result.evidence[0].chunk_id == "chk-suc-01"
        assert citation.document_id == result.document_id
        assert citation.chunk_id == result.evidence[0].chunk_id
        assert result.status == "success"
        assert result.error is None
        assert result.evidence[0].metadata["trace_test"] == "DAY15_SUCCESS"


# ============================================================================
# 6. FAILURE TRACE
# ============================================================================


class TestFailureTrace:
    """Verifies failures are associated with correct request and do not contaminate later requests."""

    def test_failure_associated_with_correct_request(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="bad.pdf", chunk_id="chk-bad")

        vsr = _vsr(chunk_id="chk-good-01", document_id="doc-good-01", filename="good.pdf")
        cit = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_search_result(vsr)
        result = VisionResult(
            query="Good query after failure",
            status="success",
            description="Clean.",
            evidence=[ev],
        )

        assert result.status == "success"
        assert result.error is None
        assert result.document_id == "doc-good-01"
        assert cit.document_id == "doc-good-01"


# ============================================================================
# 7. RETRY TRACE
# ============================================================================


class TestRetryTrace:
    """
    Public retry counter: NOT APPLICABLE -- no public retry_sequence_id in existing contract.
    VisionExecutionLifecycle tracks stage transitions per lifecycle instance.
    This test verifies stage sequence determinism across multiple independent lifecycle instances.
    """

    def test_lifecycle_stage_sequence_deterministic_across_instances(self) -> None:
        for attempt in range(3):
            lc = VisionExecutionLifecycle(
                provider_name="mock_provider",
                model_name="mock_model",
            )
            assert lc.stage == VisionExecutionStage.PENDING
            lc.transition_to(VisionExecutionStage.VALIDATING)
            lc.transition_to(VisionExecutionStage.COMPLETED)
            assert lc.stage == VisionExecutionStage.COMPLETED
            assert lc.error is None


# ============================================================================
# 8. MULTI-DOCUMENT TRACE
# ============================================================================


class TestMultiDocumentTrace:
    """Verifies A->A, B->B with zero cross-document trace contamination."""

    def test_multi_document_trace_isolation(self) -> None:
        docs = {
            "DOCUMENT_A": {
                "chunk_id": "chk-A-trace",
                "document_id": "doc-trace-A",
                "filename": "doc_trace_a.pdf",
                "meta": {"trace_test": "DAY15_DOC_A", "source": "DOC_A_SOURCE"},
            },
            "DOCUMENT_B": {
                "chunk_id": "chk-B-trace",
                "document_id": "doc-trace-B",
                "filename": "doc_trace_b.pdf",
                "meta": {"trace_test": "DAY15_DOC_B", "source": "DOC_B_SOURCE"},
            },
        }

        results: dict[str, VisionResult] = {}
        for label, cfg in docs.items():
            ev = _ev(
                chunk_id=cfg["chunk_id"],
                document_id=cfg["document_id"],
                filename=cfg["filename"],
                metadata=cfg["meta"],
            )
            results[label] = VisionResult(
                query=f"Trace query for {label}",
                status="success",
                description=f"Trace verified for {label}.",
                evidence=[ev],
            )

        res_a = results["DOCUMENT_A"]
        assert res_a.document_id == "doc-trace-A"
        assert res_a.evidence[0].metadata["trace_test"] == "DAY15_DOC_A"
        assert "DOC_B" not in str(res_a.to_dict())
        assert "DAY15_DOC_B" not in str(res_a.to_dict())

        res_b = results["DOCUMENT_B"]
        assert res_b.document_id == "doc-trace-B"
        assert res_b.evidence[0].metadata["trace_test"] == "DAY15_DOC_B"
        assert "DOC_A" not in str(res_b.to_dict())
        assert "DAY15_DOC_A" not in str(res_b.to_dict())


# ============================================================================
# 9. MULTI-EVIDENCE TRACE
# ============================================================================


class TestMultiEvidenceTrace:
    """Verifies each evidence item is independently traceable to its source."""

    def test_multi_evidence_independent_traceability(self) -> None:
        evidences = [
            _ev(
                chunk_id=f"chk-mev-{idx}",
                document_id=f"doc-mev-{idx}",
                filename=f"mev_doc_{idx}.pdf",
                page_number=idx + 1,
                chunk_index=idx,
                metadata={"trace_test": f"DAY15_MEV_{idx}", "idx": idx},
            )
            for idx in range(4)
        ]
        req = VisionRequest(query="Multi-evidence trace query", evidence=evidences)
        result = VisionResult(
            query=req.query,
            status="success",
            description="All evidence traced.",
            evidence=req.evidence,
        )

        assert len(result.evidence) == 4
        for idx, ev_item in enumerate(result.evidence):
            assert ev_item.chunk_id == f"chk-mev-{idx}"
            assert ev_item.document_id == f"doc-mev-{idx}"
            assert ev_item.page_number == idx + 1
            assert ev_item.metadata["trace_test"] == f"DAY15_MEV_{idx}"
            assert ev_item.metadata["idx"] == idx


# ============================================================================
# 10. REPEATED EXECUTION ISOLATION
# ============================================================================


class TestRepeatedExecutionIsolation:
    """Verifies repeated executions produce clean, independent results without stale trace data."""

    def test_repeated_trace_executions_have_no_stale_state(self) -> None:
        run_signatures: list[tuple[str, str, str]] = []

        for run_idx in range(6):
            run_marker = f"DAY15_RUN_{run_idx}"
            ev = _ev(
                chunk_id="chk-repeat-trace",
                document_id="doc-repeat-trace",
                filename="repeat_trace.pdf",
                metadata={"trace_test": run_marker},
            )
            result = VisionResult(
                query=f"Repeated trace query {run_idx}",
                status="success",
                description=f"Run {run_idx} trace complete.",
                evidence=[ev],
            )
            run_signatures.append((result.document_id, result.evidence[0].metadata["trace_test"], result.query))

        assert all(sig[0] == "doc-repeat-trace" for sig in run_signatures)
        markers = [sig[1] for sig in run_signatures]
        assert markers == [f"DAY15_RUN_{i}" for i in range(6)]
        queries = [sig[2] for sig in run_signatures]
        assert len(set(queries)) == 6


# ============================================================================
# 11. CONCURRENT TRACE ISOLATION
# ============================================================================


class TestConcurrentTraceIsolation:
    """Verifies that concurrent threads maintain fully isolated trace state."""

    def test_concurrent_requests_have_independent_trace_state(self) -> None:
        def _worker(thread_idx: int) -> dict[str, Any]:
            trace_marker = f"DAY15_THREAD_{thread_idx:02d}"
            ev = _ev(
                chunk_id=f"chk-conc-{thread_idx:02d}",
                document_id=f"doc-conc-{thread_idx:02d}",
                filename=f"conc_{thread_idx:02d}.pdf",
                page_number=thread_idx + 1,
                metadata={"trace_test": trace_marker, "thread_idx": thread_idx},
            )
            result = VisionResult(
                query=f"Concurrent trace query {thread_idx}",
                status="success",
                description=f"Thread {thread_idx} trace.",
                evidence=[ev],
            )
            return {
                "thread_idx": thread_idx,
                "document_id": result.document_id,
                "trace_marker": result.evidence[0].metadata["trace_test"],
                "serialized": str(result.to_dict()),
            }

        concurrency = 14
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            futures = [executor.submit(_worker, i) for i in range(concurrency)]
            outputs = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(outputs) == concurrency
        for r in outputs:
            tidx = r["thread_idx"]
            expected_doc = f"doc-conc-{tidx:02d}"
            expected_marker = f"DAY15_THREAD_{tidx:02d}"
            assert r["document_id"] == expected_doc
            assert r["trace_marker"] == expected_marker
            for other_idx in range(concurrency):
                if other_idx != tidx:
                    other_marker = f"DAY15_THREAD_{other_idx:02d}"
                    assert other_marker not in r["serialized"]


# ============================================================================
# 12. SERIALIZATION TRACE INTEGRITY
# ============================================================================


class TestSerializationTraceIntegrity:
    """Verifies trace-related information survives serialization round-trips."""

    def test_vision_result_trace_round_trip(self) -> None:
        trace_meta = {
            "trace_test": "DAY15_SERIAL",
            "source": "TEST_DOCUMENT_SERIAL",
            "run_id": "RUN_SERIAL_001",
        }
        ev = _ev(
            chunk_id="chk-ser-trace",
            document_id="doc-ser-trace",
            filename="serial_trace.pdf",
            page_number=5,
            metadata=trace_meta,
        )
        orig = VisionResult(
            query="Serialization trace query",
            status="success",
            description="Serial trace verified.",
            evidence=[ev],
        )

        data = orig.to_dict()
        restored = VisionResult.from_dict(data)

        assert restored.query == orig.query
        assert restored.document_id == "doc-ser-trace"
        assert restored.filename == "serial_trace.pdf"
        assert restored.page_number == 5
        assert len(restored.evidence) == 1
        assert restored.evidence[0].metadata["trace_test"] == "DAY15_SERIAL"
        assert restored.evidence[0].metadata["run_id"] == "RUN_SERIAL_001"

    def test_search_result_citation_trace_round_trip(self) -> None:
        citations = [
            AgentCitation(
                document_id=f"doc-trace-{i}",
                filename=f"trace_{i}.pdf",
                chunk_id=f"chk-trace-{i}",
                page_number=i + 1,
                score=0.80 + (i * 0.02),
                metadata={"trace_test": f"DAY15_CITE_{i}"},
            )
            for i in range(4)
        ]
        orig_sr = SearchResult(
            query="Citation trace serialization query",
            status="RESULTS_FOUND",
            citations=citations,
        )
        data = orig_sr.to_dict()
        restored_sr = SearchResult.from_dict(data)

        assert len(restored_sr.citations) == 4
        for idx, cit in enumerate(restored_sr.citations):
            assert cit.document_id == f"doc-trace-{idx}"
            assert cit.chunk_id == f"chk-trace-{idx}"
            assert cit.metadata["trace_test"] == f"DAY15_CITE_{idx}"


# ============================================================================
# 13. ERROR TRACE SAFETY
# ============================================================================


class TestErrorTraceSafety:
    """Verifies error messages do not expose unrelated request/document information."""

    def test_error_result_contains_only_own_error_context(self) -> None:
        error_result = VisionResult(
            query="Error query A",
            status="error",
            description="",
            error="Execution failed: model unavailable.",
        )

        ev_b = _ev(
            chunk_id="chk-suc-B",
            document_id="doc-suc-B",
            filename="success_doc_B.pdf",
            metadata={"trace_test": "DAY15_SUC_B"},
        )
        success_result = VisionResult(
            query="Success query B",
            status="success",
            description="Query B complete.",
            evidence=[ev_b],
        )

        assert "doc-suc-B" not in (error_result.error or "")
        assert "DAY15_SUC_B" not in str(error_result.to_dict())
        assert "Error query A" not in str(success_result.to_dict())
        assert success_result.status == "success"
        assert success_result.error is None


# ============================================================================
# 14. VISION EXECUTION TRACE
# ============================================================================


class TestVisionExecutionTrace:
    """Verifies VisionExecutionTrace records, isolates, and serializes stage information correctly."""

    def test_default_trace_has_all_expected_stages(self) -> None:
        trace = VisionExecutionTrace.create_default()
        assert trace.stages == list(VisionExecutionTrace.DEFAULT_STAGES)
        assert len(trace.stages) == len(VisionExecutionTrace.DEFAULT_STAGES)

    def test_custom_trace_stage_recording(self) -> None:
        trace = VisionExecutionTrace()
        assert trace.stages == []
        trace.add_stage("request_received")
        trace.add_stage("validation_started")
        trace.add_stage("provider_completed")
        assert trace.stages == ["request_received", "validation_started", "provider_completed"]

    def test_trace_serialization_round_trip(self) -> None:
        trace = VisionExecutionTrace(initial_stages=["stage_alpha", "stage_beta", "stage_gamma"])
        data = trace.to_dict()
        assert data["stages"] == ["stage_alpha", "stage_beta", "stage_gamma"]
        assert data["stage_count"] == 3

    def test_trace_isolation_between_instances(self) -> None:
        trace_a = VisionExecutionTrace()
        trace_b = VisionExecutionTrace()
        trace_a.add_stage("execution_completed")
        assert "execution_completed" in trace_a.stages
        assert trace_b.stages == []

    def test_invalid_stage_raises_validation_error(self) -> None:
        trace = VisionExecutionTrace()
        with pytest.raises(VisionInputValidationError):
            trace.add_stage("")

    def test_normalizer_attaches_trace_to_result_metadata(self) -> None:
        trace = VisionExecutionTrace()
        trace.add_stage("request_received")
        trace.add_stage("validation_started")
        trace.add_stage("provider_completed")

        ev = _ev(chunk_id="chk-norm-trace", document_id="doc-norm-trace", filename="norm.pdf")
        raw_result = VisionResult(
            query="Normalize trace test",
            status="success",
            description="Normalizer trace test.",
            evidence=[ev],
        )
        normalized = VisionResultNormalizer.normalize(raw_result, trace=trace)

        assert "execution_trace" in normalized.metadata
        trace_data = normalized.metadata["execution_trace"]
        assert isinstance(trace_data["stages"], list)
        assert "result_normalized" in trace_data["stages"]
        assert "execution_completed" in trace_data["stages"]


# ============================================================================
# 15. OBSERVABILITY INTERFACE CHECK
# ============================================================================


class TestObservabilityInterface:
    """
    Checks what observability interface exists.

    VisionExecutionTrace is the only public observability interface exposed by the
    existing contract. No external logging, telemetry service (Langfuse, OpenTelemetry),
    or monitoring hooks are exposed by the current repository.

    Verdict: NOT APPLICABLE for external observability -- VisionExecutionTrace is the sole
    observable interface and is verified in section 14 above.
    """

    def test_vision_execution_trace_is_the_public_observability_interface(self) -> None:
        trace = VisionExecutionTrace()
        assert hasattr(trace, "stages")
        assert hasattr(trace, "add_stage")
        assert hasattr(trace, "to_dict")
        assert callable(trace.add_stage)
        assert callable(trace.to_dict)
