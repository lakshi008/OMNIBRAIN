"""
OmniBrain Member 4 -- Day 19 End-to-End Contract Integrity & Evidence Lineage Certification Tests.

Performs complete end-to-end certification of evidence integrity and lineage across
the existing OMNIBRAIN integration pipeline:

  DocumentChunk (Member 1)
       ↓
  VectorSearchResult (Member 1)
       ↓
  AgentCitation (Member 2)
       ↓
  VisualEvidence (Member 3)
       ↓
  VisionResult (Member 3)
       ↓
  AgentResponse / AgentState (Downstream)

Concern areas:
 1. Single-document lineage -- document_id, chunk_id, filename traceable across every hop
 2. Single-evidence flow -- citation and visual evidence locked to identical source
 3. Multi-evidence flow -- Evidence A->Source A, B->Source B, C->Source C without loss or substitution
 4. Multi-document flow -- Document A chain and Document B chain strictly isolated
 5. Page / location preservation -- 1-indexed page numbers preserved across handoffs
 6. Content type preservation -- text, table, image, chart, diagram modalities preserved
 7. Metadata lineage -- synthetic marker {"lineage_test": "DAY19", "source_marker": "..."} preserved
 8. Citation lineage -- AgentCitation fields accurately reflect underlying chunk provenance
 9. Serialization lineage -- to_dict -> from_dict -> to_dict roundtrip retains full lineage
10. Repeated execution -- deterministic lineage across repeated runs
11. Request isolation -- Request A, Request B, Request A again: second A unaffected by B
12. Failure lineage -- failed executions do not leak stale citations, evidence, or cross-request state
13. Retry lineage -- deterministic lifecycle stage progression across simulated retry attempts
14. Concurrent lineage -- concurrent thread executions maintain isolated document lineage
15. No duplicate evidence -- evidence list retains exact element count without unexpected duplication
16. Cross-component ID consistency -- Ingestion ID == Search ID == Citation ID == Vision Evidence ID

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
# Helpers & Fixtures
# ============================================================================

def _chunk(
    chunk_id: str = "chk-lin-001",
    document_id: str = "doc-lin-001",
    filename: str = "lineage_report.pdf",
    page_number: int | None = 3,
    chunk_index: int = 0,
    content: str = "Operating income chart for fiscal year 2026.",
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
        metadata=metadata if metadata is not None else {"lineage_test": "DAY19", "source_marker": "DOC_A"},
    )


def _vsr(
    chunk_id: str = "chk-lin-001",
    score: float = 0.92,
    document_id: str = "doc-lin-001",
    filename: str = "lineage_report.pdf",
    page_number: int | None = 3,
    chunk_index: int = 0,
    content: str = "Operating income chart for fiscal year 2026.",
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
        metadata=metadata if metadata is not None else {"lineage_test": "DAY19", "source_marker": "DOC_A"},
    )


def _ev(
    chunk_id: str = "chk-lin-001",
    document_id: str = "doc-lin-001",
    filename: str = "lineage_report.pdf",
    page_number: int | None = 3,
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
        metadata=metadata if metadata is not None else {"lineage_test": "DAY19", "source_marker": "DOC_A"},
    )


# ============================================================================
# 1. SINGLE-DOCUMENT LINEAGE
# ============================================================================


class TestSingleDocumentLineage:
    """Verifies that original document identity is fully traceable across all components."""

    def test_full_chain_lineage_traceability(self) -> None:
        DOC_ID = "DOC-ANNUAL-2026-99"
        CHUNK_ID = "CHK-CHUNK-0042"
        FILENAME = "annual_report_2026.pdf"

        # 1. Ingestion Chunk
        chunk = _chunk(chunk_id=CHUNK_ID, document_id=DOC_ID, filename=FILENAME, page_number=5)

        # 2. Search Result
        vsr = VectorSearchResult(
            chunk_id=chunk.chunk_id,
            score=0.96,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content_type=chunk.content_type,
            content=chunk.content,
            metadata=chunk.metadata,
        )

        # 3. Citation
        citation = AgentCitation.from_search_result(vsr)

        # 4. Visual Evidence
        evidence = VisualEvidence.from_search_result(vsr)

        # 5. Vision Result
        vision_result = VisionResult(
            query="Analyze operating income chart",
            status="success",
            description="Operating income grew 18%.",
            evidence=[evidence],
        )

        # 6. Downstream AgentState and AgentResponse
        state = AgentState(query=vision_result.query)
        state.add_citation(citation)
        response = AgentResponse(
            answer=vision_result.description,
            agent_name="SupervisorAgent",
            citations=state.citations,
            status="success",
            metadata={"query": state.query},
        )

        # Certify Lineage across entire chain
        assert chunk.document_id == DOC_ID
        assert vsr.document_id == DOC_ID
        assert citation.document_id == DOC_ID
        assert evidence.document_id == DOC_ID
        assert vision_result.document_id == DOC_ID
        assert response.citations[0].document_id == DOC_ID

        assert chunk.chunk_id == CHUNK_ID
        assert vsr.chunk_id == CHUNK_ID
        assert citation.chunk_id == CHUNK_ID
        assert evidence.chunk_id == CHUNK_ID
        assert vision_result.chunk_id == CHUNK_ID
        assert response.citations[0].chunk_id == CHUNK_ID

        assert chunk.filename == FILENAME
        assert vsr.filename == FILENAME
        assert citation.filename == FILENAME
        assert evidence.filename == FILENAME
        assert vision_result.filename == FILENAME
        assert response.citations[0].filename == FILENAME


# ============================================================================
# 2. SINGLE-EVIDENCE FLOW
# ============================================================================


class TestSingleEvidenceFlow:
    """Verifies that single-evidence items correctly preserve provenance and metadata."""

    def test_single_evidence_provenance_preservation(self) -> None:
        ev = _ev(
            chunk_id="chk-se-01",
            document_id="doc-se-01",
            filename="financials.pdf",
            page_number=8,
            content_type="chart",
            metadata={"lineage_test": "DAY19", "source_marker": "DOC_SINGLE"},
        )
        req = VisionRequest(query="Single evidence query", evidence=[ev])
        result = VisionResult(
            query=req.query,
            status="success",
            description="Chart analyzed.",
            evidence=req.evidence,
        )

        assert result.has_evidence is True
        assert len(result.evidence) == 1
        assert result.evidence[0].document_id == "doc-se-01"
        assert result.evidence[0].chunk_id == "chk-se-01"
        assert result.evidence[0].page_number == 8
        assert result.evidence[0].content_type == "chart"
        assert result.evidence[0].metadata["source_marker"] == "DOC_SINGLE"


# ============================================================================
# 3. MULTI-EVIDENCE FLOW
# ============================================================================


class TestMultiEvidenceFlow:
    """Verifies Evidence A->Source A, B->Source B, C->Source C without loss or substitution."""

    def test_multi_evidence_distinct_source_association(self) -> None:
        sources = [
            ("chk-me-A", "doc-me-A", "doc_a.pdf", 1, "chart", "SOURCE_A"),
            ("chk-me-B", "doc-me-B", "doc_b.pdf", 2, "diagram", "SOURCE_B"),
            ("chk-me-C", "doc-me-C", "doc_c.pdf", 3, "image", "SOURCE_C"),
        ]

        evidences = [
            VisualEvidence(
                chunk_id=cid,
                document_id=did,
                filename=fn,
                page_number=pg,
                chunk_index=idx,
                content_type=ct,
                metadata={"lineage_test": "DAY19", "source_marker": sm},
            )
            for idx, (cid, did, fn, pg, ct, sm) in enumerate(sources)
        ]

        req = VisionRequest(query="Multi evidence analysis", evidence=evidences)
        result = VisionResult(
            query=req.query,
            status="success",
            description="All multi-evidence processed.",
            evidence=req.evidence,
        )

        assert len(result.evidence) == 3
        for idx, (cid, did, fn, pg, ct, sm) in enumerate(sources):
            ev_item = result.evidence[idx]
            assert ev_item.chunk_id == cid
            assert ev_item.document_id == did
            assert ev_item.filename == fn
            assert ev_item.page_number == pg
            assert ev_item.content_type == ct
            assert ev_item.metadata["source_marker"] == sm


# ============================================================================
# 4. MULTI-DOCUMENT FLOW & ISOLATION
# ============================================================================


class TestMultiDocumentFlow:
    """Verifies that separate document workflows never mix citations, chunks, or metadata."""

    def test_multi_document_lineage_isolation(self) -> None:
        # Document A Workflow
        chunk_a = _chunk(
            chunk_id="chk-docA-01",
            document_id="doc-A-alpha",
            filename="document_alpha.pdf",
            page_number=2,
            metadata={"lineage_test": "DAY19", "source_marker": "ALPHA_CORP"},
        )
        vsr_a = _vsr(
            chunk_id=chunk_a.chunk_id,
            document_id=chunk_a.document_id,
            filename=chunk_a.filename,
            page_number=chunk_a.page_number,
            metadata=chunk_a.metadata,
        )
        cit_a = AgentCitation.from_search_result(vsr_a)
        ev_a = VisualEvidence.from_search_result(vsr_a)
        res_a = VisionResult(query="Alpha query", status="success", description="Alpha verified.", evidence=[ev_a])

        # Document B Workflow
        chunk_b = _chunk(
            chunk_id="chk-docB-01",
            document_id="doc-B-beta",
            filename="document_beta.pdf",
            page_number=4,
            metadata={"lineage_test": "DAY19", "source_marker": "BETA_CORP"},
        )
        vsr_b = _vsr(
            chunk_id=chunk_b.chunk_id,
            document_id=chunk_b.document_id,
            filename=chunk_b.filename,
            page_number=chunk_b.page_number,
            metadata=chunk_b.metadata,
        )
        cit_b = AgentCitation.from_search_result(vsr_b)
        ev_b = VisualEvidence.from_search_result(vsr_b)
        res_b = VisionResult(query="Beta query", status="success", description="Beta verified.", evidence=[ev_b])

        # Verify Document A lineage
        assert cit_a.document_id == "doc-A-alpha"
        assert res_a.document_id == "doc-A-alpha"
        assert res_a.evidence[0].metadata["source_marker"] == "ALPHA_CORP"
        assert "BETA" not in str(res_a.to_dict())

        # Verify Document B lineage
        assert cit_b.document_id == "doc-B-beta"
        assert res_b.document_id == "doc-B-beta"
        assert res_b.evidence[0].metadata["source_marker"] == "BETA_CORP"
        assert "ALPHA" not in str(res_b.to_dict())


# ============================================================================
# 5. PAGE / LOCATION PRESERVATION
# ============================================================================


class TestPageLocationPreservation:
    """Verifies that 1-indexed page numbers are preserved across all integration points."""

    def test_page_number_preserved_through_entire_chain(self) -> None:
        PAGE = 9

        chunk = _chunk(page_number=PAGE)
        vsr = _vsr(page_number=chunk.page_number)
        cit = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_search_result(vsr)
        vres = VisionResult(query="Page test", status="success", description="Page ok.", evidence=[ev])

        assert chunk.page_number == PAGE
        assert vsr.page_number == PAGE
        assert cit.page_number == PAGE
        assert ev.page_number == PAGE
        assert vres.page_number == PAGE

    def test_none_page_number_handled_safely(self) -> None:
        chunk = _chunk(page_number=None)
        vsr = _vsr(page_number=None)
        cit = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_search_result(vsr)
        vres = VisionResult(query="None page test", status="success", description="Page none ok.", evidence=[ev])

        assert chunk.page_number is None
        assert vsr.page_number is None
        assert cit.page_number is None
        assert ev.page_number is None
        assert vres.page_number is None


# ============================================================================
# 6. CONTENT TYPE PRESERVATION
# ============================================================================


class TestContentTypePreservation:
    """Verifies that supported modalities (text, table, image, chart, diagram) survive handoffs."""

    def test_all_visual_content_types_preserved_in_lineage(self) -> None:
        for ct in sorted(VALID_VISUAL_CONTENT_TYPES):
            chunk = _chunk(content_type=ct)
            vsr = _vsr(content_type=ct)
            cit = AgentCitation.from_search_result(vsr)
            ev = VisualEvidence.from_search_result(vsr)
            vres = VisionResult(query=f"Analyze {ct}", status="success", description=f"{ct} ok.", evidence=[ev])

            assert chunk.content_type == ct
            assert vsr.content_type == ct
            assert cit.content_type == ct
            assert ev.content_type == ct
            assert vres.content_type == ct


# ============================================================================
# 7. METADATA LINEAGE
# ============================================================================


class TestMetadataLineage:
    """Verifies that metadata dictionaries remain attached to their originating source."""

    def test_synthetic_metadata_lineage_preservation(self) -> None:
        test_meta = {
            "lineage_test": "DAY19",
            "source_marker": "DOCUMENT_A",
            "origin_system": "ERP_INGEST",
            "nested_provenance": {"batch": 42, "checksum": "abc123xyz"},
        }

        chunk = _chunk(metadata=test_meta)
        vsr = _vsr(metadata=chunk.metadata)
        cit = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_search_result(vsr)
        vres = VisionResult(query="Meta lineage test", status="success", description="Lineage ok.", evidence=[ev])

        assert cit.metadata["lineage_test"] == "DAY19"
        assert cit.metadata["source_marker"] == "DOCUMENT_A"
        assert cit.metadata["nested_provenance"]["batch"] == 42

        assert ev.metadata["lineage_test"] == "DAY19"
        assert ev.metadata["source_marker"] == "DOCUMENT_A"

        assert vres.evidence[0].metadata["lineage_test"] == "DAY19"
        assert vres.evidence[0].metadata["source_marker"] == "DOCUMENT_A"


# ============================================================================
# 8. CITATION LINEAGE
# ============================================================================


class TestCitationLineage:
    """Verifies that AgentCitation strictly mirrors source chunk provenance."""

    def test_citation_matches_chunk_attributes(self) -> None:
        vsr = _vsr(
            chunk_id="chk-cite-01",
            score=0.93,
            document_id="doc-cite-01",
            filename="citation_doc.pdf",
            page_number=6,
            chunk_index=2,
            content_type="table",
            metadata={"lineage_test": "DAY19", "source_marker": "DOC_CITE"},
        )
        cit = AgentCitation.from_search_result(vsr)

        assert cit.document_id == vsr.document_id
        assert cit.filename == vsr.filename
        assert cit.chunk_id == vsr.chunk_id
        assert cit.page_number == vsr.page_number
        assert cit.content_type == vsr.content_type
        assert cit.score == vsr.score
        assert cit.metadata["source_marker"] == "DOC_CITE"


# ============================================================================
# 9. SERIALIZATION LINEAGE
# ============================================================================


class TestSerializationLineage:
    """Verifies lineage survives serialization and deserialization intact."""

    def test_vision_result_serialization_lineage_retention(self) -> None:
        ev = _ev(
            chunk_id="chk-ser-lin-01",
            document_id="doc-ser-lin-01",
            filename="ser_lin.pdf",
            page_number=4,
            content_type="diagram",
            metadata={"lineage_test": "DAY19", "source_marker": "SERIAL_DOC"},
        )
        orig = VisionResult(
            query="Serialization lineage test",
            status="success",
            description="Serialization lineage verified.",
            evidence=[ev],
        )

        data = orig.to_dict()
        restored = VisionResult.from_dict(data)

        assert restored.document_id == orig.document_id
        assert restored.chunk_id == orig.chunk_id
        assert restored.filename == orig.filename
        assert restored.page_number == orig.page_number
        assert restored.content_type == orig.content_type
        assert restored.evidence[0].metadata["source_marker"] == "SERIAL_DOC"

    def test_search_result_serialization_lineage_retention(self) -> None:
        citations = [
            AgentCitation(
                document_id=f"doc-sr-{i}",
                filename=f"sr_{i}.pdf",
                chunk_id=f"chk-sr-{i}",
                page_number=i + 1,
                score=0.85 + (i * 0.02),
                metadata={"lineage_test": "DAY19", "source_marker": f"SR_DOC_{i}"},
            )
            for i in range(3)
        ]
        orig_sr = SearchResult(query="SR lineage query", status="RESULTS_FOUND", citations=citations)
        data = orig_sr.to_dict()
        restored_sr = SearchResult.from_dict(data)

        assert len(restored_sr.citations) == 3
        for idx, cit in enumerate(restored_sr.citations):
            assert cit.document_id == f"doc-sr-{idx}"
            assert cit.chunk_id == f"chk-sr-{idx}"
            assert cit.metadata["source_marker"] == f"SR_DOC_{idx}"


# ============================================================================
# 10. REPEATED EXECUTION
# ============================================================================


class TestRepeatedExecution:
    """Verifies that repeated execution of the lineage flow produces identical provenance."""

    def test_repeated_lineage_pipeline_stability(self) -> None:
        RUN_COUNT = 6
        lineage_snapshots: list[dict[str, Any]] = []

        for _ in range(RUN_COUNT):
            chunk = _chunk(
                chunk_id="chk-rep-lin",
                document_id="doc-rep-lin",
                filename="rep_lin.pdf",
                page_number=7,
                metadata={"lineage_test": "DAY19", "source_marker": "REPEAT_DOC"},
            )
            vsr = _vsr(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                metadata=chunk.metadata,
            )
            cit = AgentCitation.from_search_result(vsr)
            ev = VisualEvidence.from_search_result(vsr)
            vres = VisionResult(
                query="Repeated lineage query",
                status="success",
                description="Repeated lineage verified.",
                evidence=[ev],
            )
            lineage_snapshots.append({
                "cit_doc": cit.document_id,
                "cit_chunk": cit.chunk_id,
                "cit_page": cit.page_number,
                "ev_doc": ev.document_id,
                "ev_chunk": ev.chunk_id,
                "vres_doc": vres.document_id,
                "vres_chunk": vres.chunk_id,
                "marker": vres.evidence[0].metadata["source_marker"],
            })

        first = lineage_snapshots[0]
        for snap in lineage_snapshots[1:]:
            assert snap == first


# ============================================================================
# 11. REQUEST ISOLATION (A, B, A again)
# ============================================================================


class TestRequestIsolation:
    """Verifies running Request A, Request B, Request A leaves the second A unaffected by B."""

    def test_request_a_b_a_isolation(self) -> None:
        def _build_result(doc_id: str, marker: str) -> dict[str, Any]:
            ev = _ev(
                chunk_id=f"chk-{doc_id}",
                document_id=doc_id,
                filename=f"{doc_id}.pdf",
                page_number=1,
                metadata={"lineage_test": "DAY19", "source_marker": marker},
            )
            res = VisionResult(
                query=f"Query for {doc_id}",
                status="success",
                description=f"Result for {doc_id}.",
                evidence=[ev],
            )
            return res.to_dict()

        res_a1 = _build_result("doc-iso-A", "MARKER_A")
        _res_b = _build_result("doc-iso-B", "MARKER_B")  # noqa: F841
        res_a2 = _build_result("doc-iso-A", "MARKER_A")

        assert res_a1 == res_a2
        assert "MARKER_B" not in str(res_a2)
        assert "doc-iso-B" not in str(res_a2)


# ============================================================================
# 12. FAILURE LINEAGE
# ============================================================================


class TestFailureLineage:
    """Verifies failed executions do not retain stale citations or corrupt lineage."""

    def test_failed_execution_does_not_contaminate_subsequent_lineage(self) -> None:
        # Invalid citation attempt
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="bad.pdf", chunk_id="chk-bad")

        # Subsequent valid execution
        chunk = _chunk(
            chunk_id="chk-clean-01",
            document_id="doc-clean-01",
            filename="clean.pdf",
            metadata={"lineage_test": "DAY19", "source_marker": "CLEAN_DOC"},
        )
        vsr = _vsr(chunk_id=chunk.chunk_id, document_id=chunk.document_id, filename=chunk.filename, metadata=chunk.metadata)
        cit = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_search_result(vsr)
        vres = VisionResult(query="Clean query after failure", status="success", description="Clean ok.", evidence=[ev])

        assert vres.status == "success"
        assert vres.error is None
        assert vres.document_id == "doc-clean-01"
        assert cit.document_id == "doc-clean-01"


# ============================================================================
# 13. RETRY LINEAGE
# ============================================================================


class TestRetryLineage:
    """
    Public retry counter: NOT APPLICABLE -- no public retry_sequence_id in existing contract.
    Verifies that VisionExecutionLifecycle stage transitions remain deterministic across lifecycle instances.
    """

    def test_lifecycle_stage_transitions_deterministic_across_retries(self) -> None:
        for attempt in range(4):
            lc = VisionExecutionLifecycle(provider_name="mock_p", model_name="mock_m")
            assert lc.stage == VisionExecutionStage.PENDING
            lc.transition_to(VisionExecutionStage.VALIDATING)
            lc.transition_to(VisionExecutionStage.EXECUTING)
            lc.transition_to(VisionExecutionStage.COMPLETED)
            assert lc.stage == VisionExecutionStage.COMPLETED
            assert lc.error is None


# ============================================================================
# 14. CONCURRENT LINEAGE
# ============================================================================


class TestConcurrentLineage:
    """Verifies that concurrent threads running independent document workflows maintain isolated lineage."""

    def test_concurrent_document_lineage_isolation(self) -> None:
        def _worker(thread_idx: int) -> dict[str, Any]:
            doc_id = f"doc-conc-{thread_idx:02d}"
            chunk_id = f"chk-conc-{thread_idx:02d}"
            filename = f"conc_doc_{thread_idx:02d}.pdf"
            marker = f"THREAD_MARKER_{thread_idx:02d}"

            ev = _ev(
                chunk_id=chunk_id,
                document_id=doc_id,
                filename=filename,
                page_number=thread_idx + 1,
                metadata={"lineage_test": "DAY19", "source_marker": marker},
            )
            result = VisionResult(
                query=f"Concurrent query {thread_idx}",
                status="success",
                description=f"Thread {thread_idx} verified.",
                evidence=[ev],
            )
            return {
                "thread_idx": thread_idx,
                "document_id": result.document_id,
                "chunk_id": result.chunk_id,
                "page_number": result.page_number,
                "marker": result.evidence[0].metadata["source_marker"],
                "serialized": str(result.to_dict()),
            }

        concurrency = 16
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_worker, i) for i in range(concurrency)]
            outputs = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(outputs) == concurrency
        for r in outputs:
            tidx = r["thread_idx"]
            assert r["document_id"] == f"doc-conc-{tidx:02d}"
            assert r["chunk_id"] == f"chk-conc-{tidx:02d}"
            assert r["page_number"] == tidx + 1
            assert r["marker"] == f"THREAD_MARKER_{tidx:02d}"

            for other_idx in range(concurrency):
                if other_idx != tidx:
                    assert f"THREAD_MARKER_{other_idx:02d}" not in r["serialized"]


# ============================================================================
# 15. NO DUPLICATE EVIDENCE
# ============================================================================


class TestNoDuplicateEvidence:
    """Verifies that evidence list maintains exact item count without synthetic duplication."""

    def test_evidence_list_count_matches_input_strictly(self) -> None:
        ev_list = [
            _ev(chunk_id=f"chk-dup-{i}", document_id=f"doc-dup-{i}", filename=f"dup_{i}.pdf")
            for i in range(5)
        ]
        req = VisionRequest(query="Duplicate check query", evidence=ev_list)
        result = VisionResult(query=req.query, status="success", description="Count verified.", evidence=req.evidence)

        assert len(result.evidence) == 5
        unique_chunk_ids = {e.chunk_id for e in result.evidence}
        assert len(unique_chunk_ids) == 5


# ============================================================================
# 16. CROSS-COMPONENT ID CONSISTENCY
# ============================================================================


class TestCrossComponentIDConsistency:
    """Verifies Ingestion ID == Search ID == Citation ID == Vision Evidence ID."""

    def test_identity_preservation_across_all_subsystems(self) -> None:
        ID_PAIRS = [
            ("doc-audit-alpha", "chk-audit-01", "report_alpha.pdf"),
            ("doc-audit-beta", "chk-audit-02", "report_beta.pdf"),
            ("doc-audit-gamma", "chk-audit-03", "report_gamma.pdf"),
        ]

        for doc_id, chunk_id, filename in ID_PAIRS:
            chunk = _chunk(chunk_id=chunk_id, document_id=doc_id, filename=filename)
            vsr = _vsr(chunk_id=chunk.chunk_id, document_id=chunk.document_id, filename=chunk.filename)
            cit = AgentCitation.from_search_result(vsr)
            ev = VisualEvidence.from_search_result(vsr)
            vres = VisionResult(query="ID audit", status="success", description="Audit ok.", evidence=[ev])

            assert chunk.document_id == vsr.document_id == cit.document_id == ev.document_id == vres.document_id == doc_id
            assert chunk.chunk_id == vsr.chunk_id == cit.chunk_id == ev.chunk_id == vres.chunk_id == chunk_id
            assert chunk.filename == vsr.filename == cit.filename == ev.filename == vres.filename == filename
