"""
OmniBrain Member 4 -- Day 16 Deterministic Behavior & Reproducibility Integration Tests.

Verifies that existing OMNIBRAIN integration behavior is deterministic and reproducible
across repeated executions using the same inputs.

Concern areas:
 1. Identical input -> same observable result structure (type, status, evidence count)
 2. Result structure determinism (to_dict() stability)
 3. Evidence order reproducibility where contract defines ordering
 4. Citation determinism -- same chunk/source/page for same input
 5. Lineage determinism -- Document -> Chunk -> Evidence -> Citation -> Result chain
 6. Multi-document reproducibility -- A->A, B->B across runs
 7. Multi-evidence reproducibility -- no evidence lost or replaced across runs
 8. Serialization determinism -- to_dict()->from_dict()->to_dict() stability
 9. Failure reproducibility -- same exception category, same public error structure
10. Retry reproducibility -- lifecycle stage sequence deterministic (public retry counter: NOT APPLICABLE)
11. Request isolation -- A, B, A again: second A unaffected by B
12. Concurrent reproducibility -- independent threads produce correct isolated results
13. State reset -- no retained state between executions

Constraints:
 - 100% Offline: No external APIs, network, real LLMs, or production secrets.
 - Zero production code modified.
 - Variable fields excluded from strict equality (no timestamps, UUIDs, runtime durations).
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
from ingestion.models import DocumentChunk, VectorSearchResult
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.retrieval_processor import build_retrieval_context, process_retrieval_results

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
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.exceptions import VisionInputValidationError
from vision.lifecycle import VisionExecutionLifecycle, VisionExecutionStage
from vision.result_normalizer import VisionExecutionTrace, VisionResultNormalizer


# ============================================================================
# Shared Fixtures & Helpers
# ============================================================================

def _chunk(
    chunk_id: str = "chk-det-001",
    document_id: str = "doc-det-001",
    filename: str = "det_report.pdf",
    page_number: int | None = 1,
    chunk_index: int = 0,
    content: str = "Deterministic test content.",
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
        metadata=metadata if metadata is not None else {"run_marker": "DET_DEFAULT"},
    )


def _vsr(
    chunk_id: str = "chk-det-001",
    score: float = 0.90,
    document_id: str = "doc-det-001",
    filename: str = "det_report.pdf",
    page_number: int | None = 1,
    chunk_index: int = 0,
    content: str = "Deterministic test content.",
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
        metadata=metadata if metadata is not None else {"run_marker": "DET_DEFAULT"},
    )


def _ev(
    chunk_id: str = "chk-det-001",
    document_id: str = "doc-det-001",
    filename: str = "det_report.pdf",
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
        metadata=metadata if metadata is not None else {"run_marker": "DET_DEFAULT"},
    )


# ============================================================================
# 1. IDENTICAL INPUT -- REPEATED EXECUTION
# ============================================================================


class TestIdenticalInputRepeatedExecution:
    """Verifies that identical inputs produce identical observable result structures across runs."""

    def test_same_input_same_result_structure_across_runs(self) -> None:
        RUN_COUNT = 8
        snapshots: list[dict[str, Any]] = []

        for _ in range(RUN_COUNT):
            ev = _ev(
                chunk_id="chk-idr-01",
                document_id="doc-idr-01",
                filename="idr_doc.pdf",
                page_number=3,
                content_type="chart",
                metadata={"run_marker": "DAY16_IDR"},
            )
            result = VisionResult(
                query="Identical deterministic query",
                status="success",
                description="IDR verified.",
                evidence=[ev],
            )
            snapshots.append({
                "status": result.status,
                "document_id": result.document_id,
                "filename": result.filename,
                "chunk_id": result.chunk_id,
                "page_number": result.page_number,
                "evidence_count": len(result.evidence),
                "error": result.error,
                "run_marker": result.evidence[0].metadata["run_marker"],
            })

        first = snapshots[0]
        for snap in snapshots[1:]:
            assert snap == first


# ============================================================================
# 2. RESULT STRUCTURE DETERMINISM
# ============================================================================


class TestResultStructureDeterminism:
    """Verifies to_dict() output is stable across repeated calls for the same object."""

    def test_to_dict_output_stable_across_calls(self) -> None:
        ev = _ev(
            chunk_id="chk-rsd-01",
            document_id="doc-rsd-01",
            filename="rsd_doc.pdf",
            page_number=2,
            metadata={"run_marker": "DAY16_RSD"},
        )
        result = VisionResult(
            query="RSD query",
            status="success",
            description="RSD verified.",
            evidence=[ev],
        )

        first_dict = result.to_dict()
        for _ in range(5):
            assert result.to_dict() == first_dict

    def test_citation_to_dict_output_stable(self) -> None:
        vsr = _vsr(chunk_id="chk-cit-det", document_id="doc-cit-det", score=0.88)
        cit = AgentCitation.from_search_result(vsr)
        first_dict = cit.to_dict()
        for _ in range(5):
            assert cit.to_dict() == first_dict


# ============================================================================
# 3. EVIDENCE ORDER REPRODUCIBILITY
# ============================================================================


class TestEvidenceOrderReproducibility:
    """
    Verifies evidence list order stability.
    The VisionRequest and VisionResult preserve insertion order (Python list semantics).
    """

    def test_evidence_order_preserved_across_runs(self) -> None:
        ev_specs = [
            ("chk-ord-A", "doc-ord-A", "ord_a.pdf", 1),
            ("chk-ord-B", "doc-ord-B", "ord_b.pdf", 2),
            ("chk-ord-C", "doc-ord-C", "ord_c.pdf", 3),
        ]

        for _ in range(6):
            evidences = [
                _ev(chunk_id=chunk_id, document_id=doc_id, filename=fn, page_number=pg)
                for chunk_id, doc_id, fn, pg in ev_specs
            ]
            req = VisionRequest(query="Order test query", evidence=evidences)
            result = VisionResult(
                query=req.query,
                status="success",
                description="Order verified.",
                evidence=req.evidence,
            )
            for idx, (chunk_id, doc_id, fn, pg) in enumerate(ev_specs):
                assert result.evidence[idx].chunk_id == chunk_id
                assert result.evidence[idx].document_id == doc_id
                assert result.evidence[idx].page_number == pg


# ============================================================================
# 4. CITATION DETERMINISM
# ============================================================================


class TestCitationDeterminism:
    """Verifies citation fields remain identical across repeated runs with same input."""

    def test_citations_same_across_repeated_runs(self) -> None:
        RUN_COUNT = 8
        citation_snapshots: list[dict[str, Any]] = []

        for _ in range(RUN_COUNT):
            vsr = _vsr(
                chunk_id="chk-cit-rep",
                score=0.91,
                document_id="doc-cit-rep",
                filename="cit_rep.pdf",
                page_number=5,
                metadata={"run_marker": "DAY16_CIT"},
            )
            cit = AgentCitation.from_search_result(vsr)
            citation_snapshots.append(cit.to_dict())

        first = citation_snapshots[0]
        for snap in citation_snapshots[1:]:
            assert snap == first


# ============================================================================
# 5. LINEAGE DETERMINISM
# ============================================================================


class TestLineageDeterminism:
    """Verifies Document -> Chunk -> Evidence -> Citation -> VisionResult chain is deterministic."""

    def test_lineage_chain_deterministic_across_runs(self) -> None:
        RUN_COUNT = 6

        def _build_chain() -> dict[str, Any]:
            chunk = _chunk(
                chunk_id="chk-lin-det",
                document_id="doc-lin-det",
                filename="lin_det.pdf",
                page_number=4,
                chunk_index=1,
                metadata={"run_marker": "DAY16_LIN"},
            )
            vsr = _vsr(
                chunk_id=chunk.chunk_id,
                score=0.87,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                metadata=chunk.metadata,
            )
            cit = AgentCitation.from_search_result(vsr)
            ev = VisualEvidence.from_search_result(vsr)
            vres = VisionResult(
                query="Lineage determinism query",
                status="success",
                description="Lineage verified.",
                evidence=[ev],
            )
            return {
                "cit_doc": cit.document_id,
                "cit_chunk": cit.chunk_id,
                "cit_page": cit.page_number,
                "ev_doc": ev.document_id,
                "ev_chunk": ev.chunk_id,
                "res_doc": vres.document_id,
                "res_chunk": vres.chunk_id,
                "res_page": vres.page_number,
            }

        first = _build_chain()
        for _ in range(RUN_COUNT - 1):
            assert _build_chain() == first


# ============================================================================
# 6. MULTI-DOCUMENT REPRODUCIBILITY
# ============================================================================


class TestMultiDocumentReproducibility:
    """Verifies that A->A, B->B evidence associations remain stable across repeated runs."""

    def test_multi_document_result_stability(self) -> None:
        RUN_COUNT = 5
        for _ in range(RUN_COUNT):
            ev_a = _ev(
                chunk_id="chk-mdr-A", document_id="doc-mdr-A", filename="mdr_a.pdf",
                page_number=1, metadata={"run_marker": "DAY16_MDR_A"},
            )
            ev_b = _ev(
                chunk_id="chk-mdr-B", document_id="doc-mdr-B", filename="mdr_b.pdf",
                page_number=2, metadata={"run_marker": "DAY16_MDR_B"},
            )

            res_a = VisionResult(query="MDR query A", status="success", description="A ok.", evidence=[ev_a])
            res_b = VisionResult(query="MDR query B", status="success", description="B ok.", evidence=[ev_b])

            assert res_a.document_id == "doc-mdr-A"
            assert res_a.evidence[0].metadata["run_marker"] == "DAY16_MDR_A"
            assert "MDR_B" not in str(res_a.to_dict())

            assert res_b.document_id == "doc-mdr-B"
            assert res_b.evidence[0].metadata["run_marker"] == "DAY16_MDR_B"
            assert "MDR_A" not in str(res_b.to_dict())


# ============================================================================
# 7. MULTI-EVIDENCE REPRODUCIBILITY
# ============================================================================


class TestMultiEvidenceReproducibility:
    """Verifies evidence identity is stable across repeated runs with the same multi-evidence input."""

    def test_multi_evidence_identity_stable_across_runs(self) -> None:
        ev_specs = [
            ("chk-mer-0", "doc-mer-0", "mer_0.pdf", 1, "DAY16_MER_0"),
            ("chk-mer-1", "doc-mer-1", "mer_1.pdf", 2, "DAY16_MER_1"),
            ("chk-mer-2", "doc-mer-2", "mer_2.pdf", 3, "DAY16_MER_2"),
        ]

        RUN_COUNT = 6
        for _ in range(RUN_COUNT):
            evidences = [
                _ev(chunk_id=cid, document_id=did, filename=fn, page_number=pg,
                    chunk_index=idx, metadata={"run_marker": marker})
                for idx, (cid, did, fn, pg, marker) in enumerate(ev_specs)
            ]
            result = VisionResult(
                query="MER query",
                status="success",
                description="MER verified.",
                evidence=evidences,
            )
            assert len(result.evidence) == 3
            for idx, (cid, did, fn, pg, marker) in enumerate(ev_specs):
                assert result.evidence[idx].chunk_id == cid
                assert result.evidence[idx].document_id == did
                assert result.evidence[idx].page_number == pg
                assert result.evidence[idx].metadata["run_marker"] == marker


# ============================================================================
# 8. SERIALIZATION DETERMINISM
# ============================================================================


class TestSerializationDeterminism:
    """Verifies to_dict()->from_dict()->to_dict() round-trip produces stable output."""

    def test_vision_result_double_serialization_stable(self) -> None:
        ev = _ev(
            chunk_id="chk-sd-01",
            document_id="doc-sd-01",
            filename="sd_doc.pdf",
            page_number=7,
            metadata={"run_marker": "DAY16_SD", "quality": "verified"},
        )
        orig = VisionResult(
            query="SD query",
            status="success",
            description="SD verified.",
            evidence=[ev],
        )

        dict1 = orig.to_dict()
        restored = VisionResult.from_dict(dict1)
        dict2 = restored.to_dict()

        # Keys and values must be stable (excluding fields intentionally absent from contract)
        assert dict1["document_id"] == dict2["document_id"]
        assert dict1["filename"] == dict2["filename"]
        assert dict1["chunk_id"] == dict2["chunk_id"]
        assert dict1["page_number"] == dict2["page_number"]
        assert dict1["status"] == dict2["status"]
        assert dict1["description"] == dict2["description"]

    def test_search_result_double_serialization_stable(self) -> None:
        citations = [
            AgentCitation(
                document_id=f"doc-sr-{i}",
                filename=f"sr_{i}.pdf",
                chunk_id=f"chk-sr-{i}",
                page_number=i + 1,
                score=0.85 + (i * 0.02),
                metadata={"run_marker": f"DAY16_SR_{i}"},
            )
            for i in range(4)
        ]
        orig_sr = SearchResult(
            query="SR serialization query",
            status="RESULTS_FOUND",
            citations=citations,
        )
        dict1 = orig_sr.to_dict()
        restored_sr = SearchResult.from_dict(dict1)
        dict2 = restored_sr.to_dict()

        assert dict1["query"] == dict2["query"]
        assert dict1["status"] == dict2["status"]
        assert len(dict1["citations"]) == len(dict2["citations"])
        for c1, c2 in zip(dict1["citations"], dict2["citations"]):
            assert c1["document_id"] == c2["document_id"]
            assert c1["chunk_id"] == c2["chunk_id"]
            assert c1["metadata"] == c2["metadata"]


# ============================================================================
# 9. FAILURE REPRODUCIBILITY
# ============================================================================


class TestFailureReproducibility:
    """Verifies same failure scenario raises same exception category repeatedly."""

    def test_agent_validation_error_is_reproducible(self) -> None:
        for _ in range(5):
            with pytest.raises(AgentValidationError):
                AgentCitation(document_id="", filename="bad.pdf", chunk_id="chk-bad")

    def test_vision_input_validation_error_is_reproducible(self) -> None:
        for _ in range(5):
            trace = VisionExecutionTrace()
            with pytest.raises(VisionInputValidationError):
                trace.add_stage("")

    def test_error_vision_result_structure_stable(self) -> None:
        RUN_COUNT = 5
        error_snapshots: list[dict[str, Any]] = []

        for _ in range(RUN_COUNT):
            result = VisionResult(
                query="Failure repro query",
                status="error",
                description="",
                error="Provider timeout after 30s.",
            )
            error_snapshots.append({
                "status": result.status,
                "error": result.error,
                "has_evidence": result.has_evidence,
                "evidence_count": len(result.evidence),
            })

        first = error_snapshots[0]
        for snap in error_snapshots[1:]:
            assert snap == first


# ============================================================================
# 10. RETRY REPRODUCIBILITY
# ============================================================================


class TestRetryReproducibility:
    """
    Public retry counter: NOT APPLICABLE -- no public retry_sequence_id in existing contract.
    Verifies VisionExecutionLifecycle stage transitions are deterministic across lifecycle instances.
    """

    def test_lifecycle_stage_transitions_deterministic(self) -> None:
        RUN_COUNT = 5
        stage_sequences: list[str] = []

        for _ in range(RUN_COUNT):
            lc = VisionExecutionLifecycle(provider_name="mock_p", model_name="mock_m")
            lc.transition_to(VisionExecutionStage.VALIDATING)
            lc.transition_to(VisionExecutionStage.EXECUTING)
            lc.transition_to(VisionExecutionStage.COMPLETED)
            stage_sequences.append(lc.stage)

        assert all(s == VisionExecutionStage.COMPLETED for s in stage_sequences)


# ============================================================================
# 11. REQUEST ISOLATION (A, B, A again)
# ============================================================================


class TestRequestIsolation:
    """Verifies that second execution of Request A is unaffected by Request B."""

    def test_a_b_a_sequence_isolation(self) -> None:
        def _run_a() -> dict[str, Any]:
            ev = _ev(
                chunk_id="chk-iso-A",
                document_id="doc-iso-A",
                filename="iso_a.pdf",
                page_number=2,
                metadata={"run_marker": "DAY16_ISO_A"},
            )
            result = VisionResult(
                query="Isolation query A",
                status="success",
                description="A isolated.",
                evidence=[ev],
            )
            return {
                "document_id": result.document_id,
                "chunk_id": result.chunk_id,
                "run_marker": result.evidence[0].metadata["run_marker"],
                "status": result.status,
            }

        def _run_b() -> dict[str, Any]:
            ev = _ev(
                chunk_id="chk-iso-B",
                document_id="doc-iso-B",
                filename="iso_b.pdf",
                page_number=3,
                metadata={"run_marker": "DAY16_ISO_B"},
            )
            result = VisionResult(
                query="Isolation query B",
                status="success",
                description="B isolated.",
                evidence=[ev],
            )
            return {
                "document_id": result.document_id,
                "chunk_id": result.chunk_id,
                "run_marker": result.evidence[0].metadata["run_marker"],
                "status": result.status,
            }

        a1 = _run_a()
        _b = _run_b()  # noqa: F841 -- intentional: B must not affect A
        a2 = _run_a()

        assert a1 == a2
        assert "ISO_B" not in str(a2)


# ============================================================================
# 12. CONCURRENT REPRODUCIBILITY
# ============================================================================


class TestConcurrentReproducibility:
    """Verifies concurrent independent requests each produce correct, isolated, deterministic results."""

    def test_concurrent_requests_produce_deterministic_results(self) -> None:
        def _worker(thread_idx: int) -> dict[str, Any]:
            ev = _ev(
                chunk_id=f"chk-cr-{thread_idx:02d}",
                document_id=f"doc-cr-{thread_idx:02d}",
                filename=f"cr_{thread_idx:02d}.pdf",
                page_number=thread_idx + 1,
                metadata={"run_marker": f"DAY16_CR_{thread_idx:02d}"},
            )
            result = VisionResult(
                query=f"Concurrent repro query {thread_idx}",
                status="success",
                description=f"CR {thread_idx} verified.",
                evidence=[ev],
            )
            return {
                "thread_idx": thread_idx,
                "document_id": result.document_id,
                "chunk_id": result.chunk_id,
                "run_marker": result.evidence[0].metadata["run_marker"],
                "status": result.status,
                "serialized": str(result.to_dict()),
            }

        concurrency = 16
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_worker, i) for i in range(concurrency)]
            outputs = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(outputs) == concurrency
        for r in outputs:
            tidx = r["thread_idx"]
            assert r["document_id"] == f"doc-cr-{tidx:02d}"
            assert r["chunk_id"] == f"chk-cr-{tidx:02d}"
            assert r["run_marker"] == f"DAY16_CR_{tidx:02d}"
            assert r["status"] == "success"
            for other_idx in range(concurrency):
                if other_idx != tidx:
                    assert f"DAY16_CR_{other_idx:02d}" not in r["serialized"]


# ============================================================================
# 13. STATE RESET VERIFICATION
# ============================================================================


class TestStateReset:
    """Verifies no state leaks between executions -- each execution starts clean."""

    def test_no_retained_state_between_executions(self) -> None:
        previous_result: VisionResult | None = None
        for run_idx in range(6):
            ev = _ev(
                chunk_id=f"chk-sr-{run_idx}",
                document_id=f"doc-sr-{run_idx}",
                filename=f"sr_{run_idx}.pdf",
                page_number=run_idx + 1,
                metadata={"run_marker": f"DAY16_SR_{run_idx}"},
            )
            current_result = VisionResult(
                query=f"State reset query {run_idx}",
                status="success",
                description=f"SR run {run_idx}.",
                evidence=[ev],
            )
            if previous_result is not None:
                # Evidence from previous run must not appear in current run
                prev_doc = previous_result.document_id
                assert current_result.document_id != prev_doc
                assert prev_doc not in str(current_result.to_dict())
            previous_result = current_result

    def test_agent_state_instances_are_independent(self) -> None:
        """Verifies AgentState instances do not share citations list."""
        state_a = AgentState(query="State A query")
        state_b = AgentState(query="State B query")

        cit_a = AgentCitation(
            document_id="doc-sa-A", filename="sa_a.pdf", chunk_id="chk-sa-A",
            metadata={"run_marker": "DAY16_SA_A"},
        )
        state_a.add_citation(cit_a)

        # state_b must have zero citations -- no shared mutable default
        assert len(state_b.citations) == 0
        assert len(state_a.citations) == 1
        assert state_a.citations[0].document_id == "doc-sa-A"
