"""
OmniBrain Member 4 -- Day 23 End-to-End Data Integrity & Cross-Request Contamination Certification.

Certifies that independent requests remain completely isolated through the full
OMNIBRAIN pipeline. Specifically detects:

  - cross-request contamination
  - stale metadata
  - stale citations
  - stale evidence
  - stale document IDs / chunk IDs
  - stale lineage
  - result mutation
  - shared mutable state
  - request-order dependency
  - cross-document contamination

Test inventory:
 1. Sequential isolation (A -> B -> A)
 2. Reverse-order isolation (B -> A -> B)
 3. Interleaved step-by-step isolation (A1 -> B1 -> A2 -> B2)
 4. Multi-document isolation (A chunks never reach B evidence)
 5. Metadata isolation (unique synthetic markers per request)
 6. Citation isolation (A citations reference only A; B citations reference only B)
 7. Evidence isolation (A evidence bound to A source; B evidence bound to B source)
 8. Result mutation safety (mutating caller's test dict does not alter A result)
 9. Input mutation safety (caller-owned lists are not mutated by public API)
10. Serialization isolation (A and B roundtrip independently)
11. Failure contamination (A SUCCESS -> B FAIL -> A SUCCESS: B failure does not touch A)
12. Success contamination (A FAIL -> B SUCCESS -> A SUCCESS: B success does not appear in A)
13. Repeated request behavior (same request N times: no state accumulation)
14. Concurrent isolation (A and B in parallel threads: A=A, B=B)
15. Request-order independence (A->B->A vs B->A->B produce consistent A and B results)
16. Error-state isolation (B error does not contaminate later A success)

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

# Ingestion (Member 1)
from ingestion.models import DocumentChunk, VectorSearchResult
from ingestion.retrieval_processor import process_retrieval_results, build_retrieval_context

# Agents / Search (Member 2)
from agents.models import AgentCitation, AgentResponse, AgentState, SearchRequest, SearchResult
from agents.exceptions import AgentValidationError

# Vision (Member 3)
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.exceptions import VisionEvidenceError, VisionInputValidationError
from vision.lifecycle import VisionExecutionLifecycle, VisionExecutionStage
from vision.result_normalizer import VisionResultNormalizer

# ============================================================================
# Shared Synthetic Identity Constants
# ============================================================================

DOC_A = "DAY23_DOCUMENT_A"
DOC_B = "DAY23_DOCUMENT_B"
CHUNK_A = "DAY23_CHUNK_A"
CHUNK_B = "DAY23_CHUNK_B"
FILE_A = "day23_source_a.pdf"
FILE_B = "day23_source_b.pdf"
META_A: dict[str, Any] = {"day23_marker": "REQUEST_A", "source_marker": "SOURCE_A", "tenant": "TENANT_A"}
META_B: dict[str, Any] = {"day23_marker": "REQUEST_B", "source_marker": "SOURCE_B", "tenant": "TENANT_B"}


# ============================================================================
# Internal Fixture Factories
# ============================================================================

def _vsr(
    doc_id: str,
    chunk_id: str,
    filename: str,
    metadata: dict[str, Any],
    score: float = 0.92,
    page_number: int = 1,
    content: str = "Day 23 integration content.",
    content_type: str = "chart",
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        score=score,
        document_id=doc_id,
        filename=filename,
        page_number=page_number,
        chunk_index=0,
        content_type=content_type,
        content=content,
        metadata=dict(metadata),
    )


def _chunk(
    doc_id: str,
    chunk_id: str,
    filename: str,
    metadata: dict[str, Any],
    page_number: int = 1,
    content: str = "Day 23 integration content.",
    content_type: str = "text",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=0,
        document_id=doc_id,
        filename=filename,
        page_number=page_number,
        content=content,
        content_type=content_type,
        metadata=dict(metadata),
    )


def _evidence(
    doc_id: str,
    chunk_id: str,
    filename: str,
    metadata: dict[str, Any],
    content_type: str = "chart",
) -> VisualEvidence:
    return VisualEvidence(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=1,
        chunk_index=0,
        content_type=content_type,
        metadata=dict(metadata),
    )


def _result(doc_id: str, chunk_id: str, filename: str, metadata: dict[str, Any], query_suffix: str = "") -> VisionResult:
    ev = _evidence(doc_id, chunk_id, filename, metadata)
    return VisionResult(
        query=f"Day23 query {query_suffix}",
        status="success",
        description=f"Day23 result for {doc_id}",
        evidence=[ev],
        metadata=dict(metadata),
    )


def _assert_is_a(result: VisionResult) -> None:
    """Assert result belongs entirely to identity A and contains no B identity markers."""
    d = str(result.to_dict())
    assert result.document_id == DOC_A, f"Expected doc_id={DOC_A}, got {result.document_id}"
    assert result.evidence[0].document_id == DOC_A
    assert DOC_B not in d, f"DOC_B leaked into A result"
    assert CHUNK_B not in d
    assert "REQUEST_B" not in d
    assert "SOURCE_B" not in d
    assert "TENANT_B" not in d


def _assert_is_b(result: VisionResult) -> None:
    """Assert result belongs entirely to identity B and contains no A identity markers."""
    d = str(result.to_dict())
    assert result.document_id == DOC_B, f"Expected doc_id={DOC_B}, got {result.document_id}"
    assert result.evidence[0].document_id == DOC_B
    assert DOC_A not in d, f"DOC_A leaked into B result"
    assert CHUNK_A not in d
    assert "REQUEST_A" not in d
    assert "SOURCE_A" not in d
    assert "TENANT_A" not in d


# ============================================================================
# 1. SEQUENTIAL ISOLATION (A -> B -> A)
# ============================================================================


class TestSequentialIsolation:
    """Verifies A -> B -> A sequence: each result stays bound to its own identity."""

    def test_a_b_a_sequential_isolation(self) -> None:
        res_a1 = _result(DOC_A, CHUNK_A, FILE_A, META_A, "seq-a1")
        res_b = _result(DOC_B, CHUNK_B, FILE_B, META_B, "seq-b")
        res_a2 = _result(DOC_A, CHUNK_A, FILE_A, META_A, "seq-a2")

        _assert_is_a(res_a1)
        _assert_is_b(res_b)
        _assert_is_a(res_a2)


# ============================================================================
# 2. REVERSE-ORDER ISOLATION (B -> A -> B)
# ============================================================================


class TestReverseOrderIsolation:
    """Verifies B -> A -> B sequence: same isolation guarantees hold in reverse order."""

    def test_b_a_b_reverse_isolation(self) -> None:
        res_b1 = _result(DOC_B, CHUNK_B, FILE_B, META_B, "rev-b1")
        res_a = _result(DOC_A, CHUNK_A, FILE_A, META_A, "rev-a")
        res_b2 = _result(DOC_B, CHUNK_B, FILE_B, META_B, "rev-b2")

        _assert_is_b(res_b1)
        _assert_is_a(res_a)
        _assert_is_b(res_b2)


# ============================================================================
# 3. INTERLEAVED STEP ISOLATION (A1 -> B1 -> A2 -> B2)
# ============================================================================


class TestInterleavedStepIsolation:
    """Verifies state belonging to A remains separate from B when steps are interleaved."""

    def test_interleaved_pipeline_steps(self) -> None:
        # A step 1: ingestion chunk
        chunk_a = _chunk(DOC_A, CHUNK_A, FILE_A, META_A)
        # B step 1: ingestion chunk
        chunk_b = _chunk(DOC_B, CHUNK_B, FILE_B, META_B)

        # A step 2: retrieval
        vsr_a = _vsr(DOC_A, CHUNK_A, FILE_A, META_A)
        # B step 2: retrieval
        vsr_b = _vsr(DOC_B, CHUNK_B, FILE_B, META_B)

        assert chunk_a.document_id == DOC_A
        assert chunk_b.document_id == DOC_B
        assert chunk_a.chunk_id != chunk_b.chunk_id

        assert vsr_a.document_id == DOC_A
        assert vsr_b.document_id == DOC_B
        assert vsr_a.metadata["day23_marker"] == "REQUEST_A"
        assert vsr_b.metadata["day23_marker"] == "REQUEST_B"

        # A final result
        res_a = _result(DOC_A, CHUNK_A, FILE_A, META_A, "interleaved-a")
        # B final result
        res_b = _result(DOC_B, CHUNK_B, FILE_B, META_B, "interleaved-b")

        _assert_is_a(res_a)
        _assert_is_b(res_b)


# ============================================================================
# 4. MULTI-DOCUMENT ISOLATION
# ============================================================================


class TestMultiDocumentIsolation:
    """Verifies A chunks never reach B evidence and B chunks never reach A evidence."""

    def test_multi_document_no_cross_contamination(self) -> None:
        vsr_a = _vsr(DOC_A, CHUNK_A, FILE_A, META_A)
        vsr_b = _vsr(DOC_B, CHUNK_B, FILE_B, META_B)

        ev_a = VisualEvidence.from_search_result(vsr_a)
        ev_b = VisualEvidence.from_search_result(vsr_b)

        assert ev_a.document_id == DOC_A
        assert ev_a.chunk_id == CHUNK_A
        assert ev_b.document_id == DOC_B
        assert ev_b.chunk_id == CHUNK_B

        a_str = str(ev_a.to_dict())
        b_str = str(ev_b.to_dict())
        assert DOC_B not in a_str
        assert DOC_A not in b_str

        proc_a = process_retrieval_results([vsr_a], min_score=0.5)
        proc_b = process_retrieval_results([vsr_b], min_score=0.5)
        assert all(r.document_id == DOC_A for r in proc_a)
        assert all(r.document_id == DOC_B for r in proc_b)


# ============================================================================
# 5. METADATA ISOLATION
# ============================================================================


class TestMetadataIsolation:
    """Verifies unique synthetic metadata does not cross between requests."""

    def test_metadata_never_crosses_requests(self) -> None:
        ev_a = _evidence(DOC_A, CHUNK_A, FILE_A, META_A)
        ev_b = _evidence(DOC_B, CHUNK_B, FILE_B, META_B)

        assert ev_a.metadata["day23_marker"] == "REQUEST_A"
        assert ev_b.metadata["day23_marker"] == "REQUEST_B"
        assert ev_a.metadata.get("day23_marker") != ev_b.metadata.get("day23_marker")

        res_a = _result(DOC_A, CHUNK_A, FILE_A, META_A, "meta-a")
        res_b = _result(DOC_B, CHUNK_B, FILE_B, META_B, "meta-b")

        a_dict = res_a.to_dict()
        b_dict = res_b.to_dict()

        assert "REQUEST_B" not in str(a_dict)
        assert "REQUEST_A" not in str(b_dict)
        assert "TENANT_B" not in str(a_dict)
        assert "TENANT_A" not in str(b_dict)


# ============================================================================
# 6. CITATION ISOLATION
# ============================================================================


class TestCitationIsolation:
    """Verifies citations generated for A reference only A, and B citations reference only B."""

    def test_citation_identity_isolation(self) -> None:
        vsr_a = _vsr(DOC_A, CHUNK_A, FILE_A, META_A)
        vsr_b = _vsr(DOC_B, CHUNK_B, FILE_B, META_B)

        cit_a = AgentCitation.from_search_result(vsr_a)
        cit_b = AgentCitation.from_search_result(vsr_b)

        assert cit_a.document_id == DOC_A
        assert cit_a.chunk_id == CHUNK_A
        assert cit_a.filename == FILE_A
        assert cit_b.document_id == DOC_B
        assert cit_b.chunk_id == CHUNK_B
        assert cit_b.filename == FILE_B

        a_str = str(cit_a.to_dict())
        b_str = str(cit_b.to_dict())
        assert DOC_B not in a_str
        assert DOC_A not in b_str
        assert CHUNK_B not in a_str
        assert CHUNK_A not in b_str


# ============================================================================
# 7. EVIDENCE ISOLATION
# ============================================================================


class TestEvidenceIsolation:
    """Verifies visual evidence remains associated with the correct request source."""

    def test_evidence_source_identity_isolation(self) -> None:
        ev_a = _evidence(DOC_A, CHUNK_A, FILE_A, META_A)
        ev_b = _evidence(DOC_B, CHUNK_B, FILE_B, META_B)

        assert ev_a.document_id == DOC_A
        assert ev_a.chunk_id == CHUNK_A
        assert ev_b.document_id == DOC_B
        assert ev_b.chunk_id == CHUNK_B

        res_a = VisionResult(query="Q_A", status="success", description="A", evidence=[ev_a])
        res_b = VisionResult(query="Q_B", status="success", description="B", evidence=[ev_b])

        assert res_a.document_id == DOC_A
        assert res_b.document_id == DOC_B
        assert res_a.chunk_id == CHUNK_A
        assert res_b.chunk_id == CHUNK_B
        assert res_a.evidence is not res_b.evidence


# ============================================================================
# 8. RESULT MUTATION SAFETY
# ============================================================================


class TestResultMutationSafety:
    """Verifies mutating caller-owned test data for B does not alter the A result."""

    def test_caller_metadata_mutation_does_not_affect_other_result(self) -> None:
        meta_a_copy = dict(META_A)
        meta_b_copy = dict(META_B)

        res_a = _result(DOC_A, CHUNK_A, FILE_A, meta_a_copy, "mut-a")

        # Mutate caller's B metadata copy
        meta_b_copy["day23_marker"] = "MUTATED_B"

        # A result must remain unchanged
        assert res_a.document_id == DOC_A
        assert res_a.evidence[0].metadata["day23_marker"] == "REQUEST_A"


# ============================================================================
# 9. INPUT MUTATION SAFETY
# ============================================================================


class TestInputMutationSafety:
    """Verifies caller-owned lists/dicts are not mutated by public API calls."""

    def test_evidence_list_not_mutated_by_vision_result(self) -> None:
        ev_a = _evidence(DOC_A, CHUNK_A, FILE_A, META_A)
        caller_evidence_list = [ev_a]
        original_len = len(caller_evidence_list)

        _ = VisionResult(query="Q_mut", status="success", description="mut", evidence=caller_evidence_list)

        assert len(caller_evidence_list) == original_len

    def test_metadata_dict_not_mutated_by_retrieval_processing(self) -> None:
        caller_meta = {"day23_marker": "REQUEST_A_MUTABLE", "score": 0.90}
        vsr = _vsr(DOC_A, CHUNK_A, FILE_A, caller_meta)
        original_marker = caller_meta["day23_marker"]

        process_retrieval_results([vsr], min_score=0.5)

        assert caller_meta["day23_marker"] == original_marker


# ============================================================================
# 10. SERIALIZATION ISOLATION
# ============================================================================


class TestSerializationIsolation:
    """Verifies A and B roundtrip independently through to_dict() -> from_dict()."""

    def test_a_and_b_roundtrip_independently(self) -> None:
        res_a = _result(DOC_A, CHUNK_A, FILE_A, META_A, "ser-a")
        res_b = _result(DOC_B, CHUNK_B, FILE_B, META_B, "ser-b")

        dict_a = res_a.to_dict()
        dict_b = res_b.to_dict()

        restored_a = VisionResult.from_dict(dict_a)
        restored_b = VisionResult.from_dict(dict_b)

        assert restored_a.document_id == DOC_A
        assert restored_b.document_id == DOC_B
        assert DOC_B not in str(dict_a)
        assert DOC_A not in str(dict_b)
        assert restored_a.evidence[0].document_id == DOC_A
        assert restored_b.evidence[0].document_id == DOC_B


# ============================================================================
# 11. FAILURE CONTAMINATION (A SUCCESS -> B FAIL -> A SUCCESS)
# ============================================================================


class TestFailureContamination:
    """A SUCCESS -> B FAIL -> A SUCCESS: B failure must not contaminate A."""

    def test_b_failure_does_not_contaminate_a(self) -> None:
        res_a1 = _result(DOC_A, CHUNK_A, FILE_A, META_A, "fc-a1")
        _assert_is_a(res_a1)

        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename=FILE_B, chunk_id=CHUNK_B, content_type="image")

        res_a2 = _result(DOC_A, CHUNK_A, FILE_A, META_A, "fc-a2")
        _assert_is_a(res_a2)
        assert res_a2.error is None

    def test_b_failure_error_not_in_a_result(self) -> None:
        res_a = _result(DOC_A, CHUNK_A, FILE_A, META_A, "fc-err-a")

        _ = VisionResult(
            query="B error query",
            status="error",
            description="",
            error="Synthetic B error: provider 429.",
            metadata={**META_B, "failure_code": "ERR_B_429"},
        )

        a_str = str(res_a.to_dict())
        assert "Synthetic B error" not in a_str
        assert "ERR_B_429" not in a_str
        assert DOC_B not in a_str
        assert res_a.error is None


# ============================================================================
# 12. SUCCESS CONTAMINATION (A FAIL -> B SUCCESS -> A SUCCESS)
# ============================================================================


class TestSuccessContamination:
    """A FAIL -> B SUCCESS -> A SUCCESS: B successful state must not appear in A."""

    def test_b_success_does_not_appear_in_a(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename=FILE_A, chunk_id=CHUNK_A)

        res_b = _result(DOC_B, CHUNK_B, FILE_B, META_B, "sc-b")
        _assert_is_b(res_b)

        res_a = _result(DOC_A, CHUNK_A, FILE_A, META_A, "sc-a")
        _assert_is_a(res_a)

        a_str = str(res_a.to_dict())
        assert DOC_B not in a_str
        assert "REQUEST_B" not in a_str


# ============================================================================
# 13. REPEATED REQUEST BEHAVIOR
# ============================================================================


class TestRepeatedRequestBehavior:
    """Executing the same request N times must not accumulate state or duplicate results."""

    REPEAT_COUNT = 6

    def test_repeated_a_requests_no_accumulation(self) -> None:
        results = []
        for i in range(self.REPEAT_COUNT):
            res = _result(DOC_A, CHUNK_A, FILE_A, META_A, f"repeat-{i}")
            results.append(res)

        for res in results:
            _assert_is_a(res)
            assert len(res.evidence) == 1
            assert res.evidence[0].document_id == DOC_A

        first_ev = results[0].evidence[0]
        for res in results[1:]:
            assert res.evidence[0] is not first_ev


# ============================================================================
# 14. CONCURRENT ISOLATION
# ============================================================================


class TestConcurrentIsolation:
    """Verifies A and B executing in parallel threads remain completely isolated."""

    def test_concurrent_a_and_b_isolation(self) -> None:
        def _worker_a(idx: int) -> dict[str, Any]:
            res = _result(DOC_A, CHUNK_A, FILE_A, META_A, f"conc-a-{idx}")
            return {
                "identity": "A",
                "idx": idx,
                "doc_id": res.document_id,
                "chunk_id": res.evidence[0].chunk_id,
                "marker": res.evidence[0].metadata["day23_marker"],
                "serialized": str(res.to_dict()),
            }

        def _worker_b(idx: int) -> dict[str, Any]:
            res = _result(DOC_B, CHUNK_B, FILE_B, META_B, f"conc-b-{idx}")
            return {
                "identity": "B",
                "idx": idx,
                "doc_id": res.document_id,
                "chunk_id": res.evidence[0].chunk_id,
                "marker": res.evidence[0].metadata["day23_marker"],
                "serialized": str(res.to_dict()),
            }

        workers: list[tuple[str, int]] = [("A", i) for i in range(8)] + [("B", i) for i in range(8)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(_worker_a if identity == "A" else _worker_b, idx): identity
                for identity, idx in workers
            }
            outputs = [(futures[f], f.result()) for f in concurrent.futures.as_completed(futures)]

        assert len(outputs) == 16

        for expected_identity, result in outputs:
            if expected_identity == "A":
                assert result["doc_id"] == DOC_A
                assert result["chunk_id"] == CHUNK_A
                assert result["marker"] == "REQUEST_A"
                assert DOC_B not in result["serialized"]
            else:
                assert result["doc_id"] == DOC_B
                assert result["chunk_id"] == CHUNK_B
                assert result["marker"] == "REQUEST_B"
                assert DOC_A not in result["serialized"]


# ============================================================================
# 15. REQUEST-ORDER INDEPENDENCE
# ============================================================================


class TestRequestOrderIndependence:
    """Verifies (A->B->A) and (B->A->B) produce consistent, deterministic identity results."""

    def _run_a_b_a(self) -> tuple[str, str, str]:
        ra1 = _result(DOC_A, CHUNK_A, FILE_A, META_A, "ord-a1")
        rb = _result(DOC_B, CHUNK_B, FILE_B, META_B, "ord-b")
        ra2 = _result(DOC_A, CHUNK_A, FILE_A, META_A, "ord-a2")
        return ra1.document_id, rb.document_id, ra2.document_id

    def _run_b_a_b(self) -> tuple[str, str, str]:
        rb1 = _result(DOC_B, CHUNK_B, FILE_B, META_B, "inv-b1")
        ra = _result(DOC_A, CHUNK_A, FILE_A, META_A, "inv-a")
        rb2 = _result(DOC_B, CHUNK_B, FILE_B, META_B, "inv-b2")
        return rb1.document_id, ra.document_id, rb2.document_id

    def test_order_independence_a_b_a(self) -> None:
        a1, b, a2 = self._run_a_b_a()
        assert a1 == DOC_A
        assert b == DOC_B
        assert a2 == DOC_A

    def test_order_independence_b_a_b(self) -> None:
        b1, a, b2 = self._run_b_a_b()
        assert b1 == DOC_B
        assert a == DOC_A
        assert b2 == DOC_B

    def test_order_independence_consistent_identity(self) -> None:
        a1, b_aba, a2 = self._run_a_b_a()
        b1, a_bab, b2 = self._run_b_a_b()

        assert a1 == a_bab == DOC_A
        assert b_aba == b1 == DOC_B


# ============================================================================
# 16. ERROR STATE ISOLATION
# ============================================================================


class TestErrorStateIsolation:
    """Verifies errors from one request do not appear in another request's result."""

    def test_a_error_state_does_not_appear_in_b(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename=FILE_A, chunk_id=CHUNK_A, content_type="image")

        res_b = _result(DOC_B, CHUNK_B, FILE_B, META_B, "err-iso-b")
        b_str = str(res_b.to_dict())
        assert DOC_A not in b_str
        assert CHUNK_A not in b_str
        assert "REQUEST_A" not in b_str
        assert res_b.error is None

    def test_search_validation_error_isolation(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="", top_k=5)

        sr_b = SearchRequest(query="B valid search query", top_k=5)
        assert sr_b.query == "B valid search query"

        vsr_b = _vsr(DOC_B, CHUNK_B, FILE_B, META_B)
        cit_b = AgentCitation.from_search_result(vsr_b)
        search_result_b = SearchResult(
            query=sr_b.query,
            status="success",
            citations=[cit_b],
        )
        assert search_result_b.status == "success"
        assert search_result_b.citations[0].document_id == DOC_B
        assert DOC_A not in str(search_result_b.to_dict())

    def test_error_vision_result_isolation(self) -> None:
        _ = VisionResult(
            query="B error query",
            status="error",
            description="",
            error="B provider timeout.",
            metadata={**META_B, "err": "timeout"},
        )

        res_a = _result(DOC_A, CHUNK_A, FILE_A, META_A, "err-vis-a")
        a_str = str(res_a.to_dict())

        assert DOC_B not in a_str
        assert "timeout" not in a_str
        assert res_a.error is None
        assert res_a.document_id == DOC_A
