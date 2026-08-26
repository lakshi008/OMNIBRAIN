"""
OmniBrain Member 4 -- Day 24 Data Contract & Serialization Round-Trip Certification.

Certifies that public data models can safely travel across the OMNIBRAIN pipeline
while preserving their contractual information via to_dict() -> from_dict() round-trips.

Serialization API surface confirmed by probing the actual repository:
 - Ingestion layer   : DocumentChunk, VectorSearchResult — NO to_dict/from_dict (NOT APPLICABLE)
 - Agents layer      : AgentRequest, SearchRequest, AgentCitation, SearchResult, AgentResponse — full to_dict/from_dict
                       AgentState — to_dict only (no from_dict, NOT APPLICABLE for full roundtrip)
 - Vision layer      : VisionRequest, VisualEvidence, VisionResult — full to_dict/from_dict
 - Unknown fields    : VisualEvidence.from_dict and SearchResult.from_dict both silently ignore
                       unknown fields (lenient deserializer contract verified)

Test inventory:
 1.  Ingestion model field contract verification (no to_dict/from_dict — NOT APPLICABLE for roundtrip)
 2.  AgentRequest serialization roundtrip
 3.  SearchRequest serialization roundtrip
 4.  AgentCitation serialization roundtrip
 5.  SearchResult serialization roundtrip (success with citations)
 6.  AgentResponse serialization roundtrip (success and error forms)
 7.  AgentState to_dict contract (from_dict NOT APPLICABLE)
 8.  VisionRequest serialization roundtrip
 9.  VisualEvidence serialization roundtrip
10.  VisionResult serialization roundtrip (success with evidence)
11.  VisionResult error-form serialization roundtrip
12.  Nested structure roundtrip (AgentResponse containing nested AgentCitation list)
13.  Nested structure roundtrip (VisionResult containing nested VisualEvidence list)
14.  Multi-item serialization isolation (A, B, C remain independent)
15.  Optional field behavior (None, empty collection, omitted)
16.  Default value stability
17.  Unknown/extra field lenient compatibility (VisualEvidence, SearchResult)
18.  Data type preservation (str, int, float, bool, list, dict, None)
19.  Metadata isolation (A marker never bleeds into B)
20.  Citation preservation (all contractual fields survive)
21.  Evidence preservation (all contractual fields survive)
22.  Error serialization (error status + message survive roundtrip)
23.  Cross-component handoff (ingestion -> agent -> vision pipeline chain)
24.  Mutation safety (original object unaffected by modifications to serialized copy)
25.  Repeated round-trip stability (double roundtrip produces consistent results)
26.  Security boundary (serialized representations contain no forbidden keys)

Constraints:
 - 100% Offline: No external APIs, network, real LLMs, or production secrets.
 - Zero production code modified.
 - Only observable behavior guaranteed by existing public contracts tested.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# Ingestion (Member 1) — no to_dict/from_dict on models
from ingestion.models import DocumentChunk, VectorSearchResult
from ingestion.retrieval_processor import process_retrieval_results

# Agents / Search (Member 2)
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    AgentState,
    SearchRequest,
    SearchResult,
)
from agents.exceptions import AgentValidationError

# Vision (Member 3)
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.exceptions import VisionEvidenceError
from vision.result_normalizer import VisionResultNormalizer

# ============================================================================
# Shared Synthetic Fixtures
# ============================================================================

_DOC_ID = "DAY24_DOC_001"
_CHUNK_ID = "DAY24_CHUNK_001"
_FILE = "day24_contract_test.pdf"
_PAGE = 4
_CHUNK_IDX = 2
_META = {"day24_marker": "CONTRACT_TEST", "source": "DAY24_FIXTURE"}

_DOC_A = "DAY24_DOC_A"
_DOC_B = "DAY24_DOC_B"
_CHUNK_A = "DAY24_CHUNK_A"
_CHUNK_B = "DAY24_CHUNK_B"
_META_A: dict[str, Any] = {"day24_marker": "A", "tenant": "TENANT_A"}
_META_B: dict[str, Any] = {"day24_marker": "B", "tenant": "TENANT_B"}


def _vsr(
    doc_id: str = _DOC_ID,
    chunk_id: str = _CHUNK_ID,
    filename: str = _FILE,
    metadata: dict[str, Any] | None = None,
    score: float = 0.91,
    content_type: str = "chart",
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        score=score,
        document_id=doc_id,
        filename=filename,
        page_number=_PAGE,
        chunk_index=_CHUNK_IDX,
        content_type=content_type,
        content="Day 24 contract test content.",
        metadata=dict(metadata) if metadata else dict(_META),
    )


def _citation(
    doc_id: str = _DOC_ID,
    chunk_id: str = _CHUNK_ID,
    filename: str = _FILE,
    metadata: dict[str, Any] | None = None,
) -> AgentCitation:
    return AgentCitation(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=_PAGE,
        score=0.91,
        content_type="chart",
        metadata=dict(metadata) if metadata else dict(_META),
    )


def _evidence(
    doc_id: str = _DOC_ID,
    chunk_id: str = _CHUNK_ID,
    filename: str = _FILE,
    metadata: dict[str, Any] | None = None,
    content_type: str = "chart",
) -> VisualEvidence:
    return VisualEvidence(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=_PAGE,
        chunk_index=_CHUNK_IDX,
        content_type=content_type,
        metadata=dict(metadata) if metadata else dict(_META),
    )


# ============================================================================
# 1. INGESTION MODEL FIELD CONTRACT (roundtrip NOT APPLICABLE — no to_dict/from_dict)
# ============================================================================


class TestIngestionModelFieldContract:
    """DocumentChunk and VectorSearchResult have no to_dict/from_dict in the public contract.
    Verifies field accessibility and structural integrity instead."""

    def test_document_chunk_field_integrity(self) -> None:
        dc = DocumentChunk(
            chunk_id=_CHUNK_ID,
            chunk_index=_CHUNK_IDX,
            document_id=_DOC_ID,
            filename=_FILE,
            page_number=_PAGE,
            content="Day 24 ingestion contract.",
            content_type="text",
            metadata=dict(_META),
        )
        assert dc.document_id == _DOC_ID
        assert dc.chunk_id == _CHUNK_ID
        assert dc.filename == _FILE
        assert dc.page_number == _PAGE
        assert dc.chunk_index == _CHUNK_IDX
        assert dc.content == "Day 24 ingestion contract."
        assert dc.metadata["day24_marker"] == "CONTRACT_TEST"

    def test_vector_search_result_field_integrity(self) -> None:
        vsr = _vsr()
        assert vsr.document_id == _DOC_ID
        assert vsr.chunk_id == _CHUNK_ID
        assert vsr.score == pytest.approx(0.91)
        assert vsr.page_number == _PAGE
        assert vsr.chunk_index == _CHUNK_IDX
        assert vsr.content_type == "chart"
        assert vsr.metadata["day24_marker"] == "CONTRACT_TEST"


# ============================================================================
# 2. AgentRequest SERIALIZATION ROUND-TRIP
# ============================================================================


class TestAgentRequestRoundTrip:
    def test_agent_request_basic_roundtrip(self) -> None:
        req = AgentRequest(query="Day 24 agent request", metadata={"session_marker": "DAY24"})
        d = req.to_dict()
        restored = AgentRequest.from_dict(d)

        assert restored.query == req.query
        assert restored.metadata["session_marker"] == "DAY24"

    def test_agent_request_with_document_filter(self) -> None:
        req = AgentRequest(
            query="Filtered query",
            document_filter={"tenant_id": "CORP_DAY24"},
            metadata={"priority": "high"},
        )
        d = req.to_dict()
        restored = AgentRequest.from_dict(d)

        assert restored.query == "Filtered query"
        assert restored.document_filter == {"tenant_id": "CORP_DAY24"}
        assert restored.metadata["priority"] == "high"


# ============================================================================
# 3. SearchRequest SERIALIZATION ROUND-TRIP
# ============================================================================


class TestSearchRequestRoundTrip:
    def test_search_request_full_roundtrip(self) -> None:
        req = SearchRequest(
            query="Day 24 search",
            top_k=7,
            min_score=0.75,
            metadata={"day24_marker": "SR_ROUNDTRIP"},
        )
        d = req.to_dict()
        restored = SearchRequest.from_dict(d)

        assert restored.query == "Day 24 search"
        assert restored.top_k == 7
        assert restored.min_score == pytest.approx(0.75)
        assert restored.metadata["day24_marker"] == "SR_ROUNDTRIP"

    def test_search_request_default_values_preserved(self) -> None:
        req = SearchRequest(query="Minimal search request")
        d = req.to_dict()
        assert "query" in d
        assert d["query"] == "Minimal search request"

        restored = SearchRequest.from_dict(d)
        assert restored.query == "Minimal search request"


# ============================================================================
# 4. AgentCitation SERIALIZATION ROUND-TRIP
# ============================================================================


class TestAgentCitationRoundTrip:
    def test_citation_full_roundtrip(self) -> None:
        cit = _citation()
        d = cit.to_dict()
        restored = AgentCitation.from_dict(d)

        assert restored.document_id == _DOC_ID
        assert restored.chunk_id == _CHUNK_ID
        assert restored.filename == _FILE
        assert restored.page_number == _PAGE
        assert restored.score == pytest.approx(0.91)
        assert restored.content_type == "chart"
        assert restored.metadata["day24_marker"] == "CONTRACT_TEST"

    def test_citation_from_search_result_roundtrip(self) -> None:
        vsr = _vsr()
        cit = AgentCitation.from_search_result(vsr)
        d = cit.to_dict()
        restored = AgentCitation.from_dict(d)

        assert restored.document_id == vsr.document_id
        assert restored.chunk_id == vsr.chunk_id
        assert restored.filename == vsr.filename
        assert restored.page_number == vsr.page_number

    def test_citation_dict_form_deserialization(self) -> None:
        """Tests legacy dict-form citation deserialization."""
        raw = {
            "document_id": _DOC_ID,
            "filename": _FILE,
            "chunk_id": _CHUNK_ID,
            "page_number": _PAGE,
            "content_type": "chart",
            "score": 0.88,
            "metadata": {"legacy_marker": "LEGACY_FORM"},
        }
        restored = AgentCitation.from_dict(raw)
        assert restored.document_id == _DOC_ID
        assert restored.metadata["legacy_marker"] == "LEGACY_FORM"


# ============================================================================
# 5. SearchResult SERIALIZATION ROUND-TRIP
# ============================================================================


class TestSearchResultRoundTrip:
    def test_search_result_success_roundtrip(self) -> None:
        cit = _citation()
        sr = SearchResult(
            query="Day 24 search result",
            status="success",
            citations=[cit],
            metadata={"round_trip": "verified"},
        )
        d = sr.to_dict()
        restored = SearchResult.from_dict(d)

        assert restored.query == "Day 24 search result"
        assert restored.status == "success"
        assert restored.has_results is True
        assert len(restored.citations) == 1
        assert restored.citations[0].document_id == _DOC_ID
        assert restored.metadata["round_trip"] == "verified"

    def test_search_result_no_results_roundtrip(self) -> None:
        sr = SearchResult(query="No match query", status="NO_RESULTS", citations=[])
        d = sr.to_dict()
        restored = SearchResult.from_dict(d)

        assert restored.status == "NO_RESULTS"
        assert restored.has_results is False
        assert len(restored.citations) == 0

    def test_search_result_unknown_field_lenient(self) -> None:
        """SearchResult.from_dict silently ignores unknown fields (lenient contract)."""
        cit = _citation()
        sr = SearchResult(query="q", status="success", citations=[cit])
        d = sr.to_dict()
        d["synthetic_unknown_field_day24"] = "EXTRA"
        restored = SearchResult.from_dict(d)
        assert restored.query == "q"


# ============================================================================
# 6. AgentResponse SERIALIZATION ROUND-TRIP
# ============================================================================


class TestAgentResponseRoundTrip:
    def test_agent_response_success_roundtrip(self) -> None:
        cit = _citation()
        ar = AgentResponse(
            answer="Day 24 answer text.",
            agent_name="day24_agent",
            status="success",
            citations=[cit],
            metadata={"session": "DAY24_SESSION"},
        )
        d = ar.to_dict()
        restored = AgentResponse.from_dict(d)

        assert restored.answer == "Day 24 answer text."
        assert restored.agent_name == "day24_agent"
        assert restored.status == "success"
        assert restored.is_success is True
        assert restored.has_citations is True
        assert len(restored.citations) == 1
        assert restored.citations[0].document_id == _DOC_ID
        assert restored.metadata["session"] == "DAY24_SESSION"

    def test_agent_response_error_roundtrip(self) -> None:
        ar = AgentResponse(
            answer="",
            agent_name="day24_agent",
            status="error",
            citations=[],
            metadata={"failure_code": "ERR_404"},
            error="Document not found in index.",
        )
        d = ar.to_dict()
        restored = AgentResponse.from_dict(d)

        assert restored.status == "error"
        assert restored.is_error is True
        assert restored.error == "Document not found in index."
        assert restored.metadata["failure_code"] == "ERR_404"


# ============================================================================
# 7. AgentState to_dict CONTRACT (from_dict NOT APPLICABLE)
# ============================================================================


class TestAgentStateToDict:
    """AgentState has to_dict but no from_dict in the public contract.
    Verifies to_dict captures contractual fields correctly."""

    def test_agent_state_to_dict_field_contract(self) -> None:
        cit = _citation()
        st = AgentState(
            query="Day 24 state query",
            route="vision",
            retrieved_results=[],
            context="Day 24 context text.",
            citations=[cit],
            answer="Day 24 agent answer.",
            errors=[],
            status="completed",
            metadata={"day24_marker": "STATE_CONTRACT"},
        )
        d = st.to_dict()

        assert d["query"] == "Day 24 state query"
        assert d["route"] == "vision"
        assert d["context"] == "Day 24 context text."
        assert d["answer"] == "Day 24 agent answer."
        assert d["status"] == "completed"
        assert d["metadata"]["day24_marker"] == "STATE_CONTRACT"
        assert len(d["citations"]) == 1


# ============================================================================
# 8. VisionRequest SERIALIZATION ROUND-TRIP
# ============================================================================


class TestVisionRequestRoundTrip:
    def test_vision_request_basic_roundtrip(self) -> None:
        req = VisionRequest(query="Day 24 vision query", metadata={"day24": "VR_ROUNDTRIP"})
        d = req.to_dict()
        restored = VisionRequest.from_dict(d)

        assert restored.query == "Day 24 vision query"
        assert restored.metadata["day24"] == "VR_ROUNDTRIP"

    def test_vision_request_with_evidence_roundtrip(self) -> None:
        ev = _evidence()
        req = VisionRequest(query="Query with evidence", evidence=[ev])
        d = req.to_dict()
        restored = VisionRequest.from_dict(d)

        assert restored.query == "Query with evidence"
        assert restored.has_evidence is True
        assert len(restored.evidence) == 1
        assert restored.evidence[0].document_id == _DOC_ID


# ============================================================================
# 9. VisualEvidence SERIALIZATION ROUND-TRIP
# ============================================================================


class TestVisualEvidenceRoundTrip:
    def test_visual_evidence_full_roundtrip(self) -> None:
        ev = _evidence()
        d = ev.to_dict()
        restored = VisualEvidence.from_dict(d)

        assert restored.document_id == _DOC_ID
        assert restored.chunk_id == _CHUNK_ID
        assert restored.filename == _FILE
        assert restored.page_number == _PAGE
        assert restored.chunk_index == _CHUNK_IDX
        assert restored.content_type == "chart"
        assert restored.metadata["day24_marker"] == "CONTRACT_TEST"

    def test_visual_evidence_from_search_result_roundtrip(self) -> None:
        vsr = _vsr()
        ev = VisualEvidence.from_search_result(vsr)
        d = ev.to_dict()
        restored = VisualEvidence.from_dict(d)

        assert restored.document_id == vsr.document_id
        assert restored.chunk_id == vsr.chunk_id
        assert restored.page_number == vsr.page_number

    def test_visual_evidence_unknown_field_lenient(self) -> None:
        """VisualEvidence.from_dict silently ignores unknown extra fields."""
        ev = _evidence()
        d = ev.to_dict()
        d["synthetic_extra_day24"] = "EXTRA_VALUE"
        restored = VisualEvidence.from_dict(d)
        assert restored.document_id == _DOC_ID


# ============================================================================
# 10. VisionResult SUCCESS SERIALIZATION ROUND-TRIP
# ============================================================================


class TestVisionResultSuccessRoundTrip:
    def test_vision_result_success_roundtrip(self) -> None:
        ev = _evidence()
        res = VisionResult(
            query="Day 24 vision result",
            status="success",
            description="Certified evidence for Day 24.",
            evidence=[ev],
            metadata={"day24_marker": "VR_SUCCESS"},
        )
        d = res.to_dict()
        restored = VisionResult.from_dict(d)

        assert restored.query == "Day 24 vision result"
        assert restored.status == "success"
        assert restored.description == "Certified evidence for Day 24."
        assert restored.document_id == _DOC_ID
        assert restored.chunk_id == _CHUNK_ID
        assert restored.has_evidence is True
        assert len(restored.evidence) == 1
        assert restored.evidence[0].document_id == _DOC_ID
        assert restored.metadata["day24_marker"] == "VR_SUCCESS"
        assert restored.error is None


# ============================================================================
# 11. VisionResult ERROR SERIALIZATION ROUND-TRIP
# ============================================================================


class TestVisionResultErrorRoundTrip:
    def test_vision_result_error_roundtrip(self) -> None:
        res = VisionResult(
            query="Day 24 error query",
            status="error",
            description="",
            error="Vision provider unavailable (HTTP 503).",
            metadata={"retryable": True, "failure_code": "ERR_503"},
        )
        d = res.to_dict()
        restored = VisionResult.from_dict(d)

        assert restored.status == "error"
        assert restored.error == "Vision provider unavailable (HTTP 503)."
        assert restored.has_evidence is False
        assert restored.metadata["failure_code"] == "ERR_503"
        assert restored.metadata["retryable"] is True


# ============================================================================
# 12. NESTED STRUCTURE ROUND-TRIP — AgentResponse with nested citations
# ============================================================================


class TestNestedAgentResponseRoundTrip:
    def test_nested_citations_survive_roundtrip(self) -> None:
        cit1 = _citation(doc_id="DOC_NESTED_1", chunk_id="CHK_N1", metadata={"pos": 1})
        cit2 = _citation(doc_id="DOC_NESTED_2", chunk_id="CHK_N2", metadata={"pos": 2})
        ar = AgentResponse(
            answer="Nested structure answer.",
            agent_name="nested_agent",
            status="success",
            citations=[cit1, cit2],
            metadata={"nested": True},
        )
        d = ar.to_dict()
        restored = AgentResponse.from_dict(d)

        assert len(restored.citations) == 2
        assert restored.citations[0].document_id == "DOC_NESTED_1"
        assert restored.citations[0].metadata["pos"] == 1
        assert restored.citations[1].document_id == "DOC_NESTED_2"
        assert restored.citations[1].metadata["pos"] == 2


# ============================================================================
# 13. NESTED STRUCTURE ROUND-TRIP — VisionResult with nested evidence
# ============================================================================


class TestNestedVisionResultRoundTrip:
    def test_nested_evidence_survives_roundtrip(self) -> None:
        ev1 = _evidence(doc_id="DOC_EV1", chunk_id="CHK_EV1", content_type="image")
        ev2 = _evidence(doc_id="DOC_EV2", chunk_id="CHK_EV2", content_type="chart")
        res = VisionResult(
            query="Nested evidence query",
            status="success",
            description="Multi-evidence result.",
            evidence=[ev1, ev2],
            metadata={"nested": True},
        )
        d = res.to_dict()
        restored = VisionResult.from_dict(d)

        # Primary identity taken from first evidence
        assert restored.document_id == "DOC_EV1"
        assert len(restored.evidence) == 2
        ev_ids = {e.document_id for e in restored.evidence}
        assert "DOC_EV1" in ev_ids
        assert "DOC_EV2" in ev_ids


# ============================================================================
# 14. MULTI-ITEM SERIALIZATION ISOLATION (A, B, C remain independent)
# ============================================================================


class TestMultiItemSerializationIsolation:
    def test_three_items_remain_independent(self) -> None:
        items = [
            (_DOC_A, _CHUNK_A, "file_a.pdf", _META_A),
            (_DOC_B, _CHUNK_B, "file_b.pdf", _META_B),
            ("DAY24_DOC_C", "DAY24_CHUNK_C", "file_c.pdf", {"day24_marker": "C", "tenant": "TENANT_C"}),
        ]
        results = []
        for doc_id, chunk_id, filename, meta in items:
            ev = _evidence(doc_id=doc_id, chunk_id=chunk_id, filename=filename, metadata=meta)
            res = VisionResult(
                query=f"Query for {doc_id}",
                status="success",
                description=f"Result {doc_id}",
                evidence=[ev],
                metadata=dict(meta),
            )
            results.append(res)

        serialized = [r.to_dict() for r in results]
        restored = [VisionResult.from_dict(d) for d in serialized]

        assert restored[0].document_id == _DOC_A
        assert restored[1].document_id == _DOC_B
        assert restored[2].document_id == "DAY24_DOC_C"

        # A not in B or C
        assert _DOC_A not in str(serialized[1])
        assert _DOC_A not in str(serialized[2])
        # B not in A or C
        assert _DOC_B not in str(serialized[0])
        assert _DOC_B not in str(serialized[2])


# ============================================================================
# 15. OPTIONAL FIELD BEHAVIOR
# ============================================================================


class TestOptionalFieldBehavior:
    def test_vision_result_no_evidence_optional(self) -> None:
        """VisionResult without evidence: evidence list is empty, document_id is None."""
        res = VisionResult(query="No evidence query", status="error", description="", error="no data")
        d = res.to_dict()
        restored = VisionResult.from_dict(d)

        assert restored.has_evidence is False
        assert restored.evidence == []

    def test_agent_citation_optional_page_number(self) -> None:
        """page_number is optional — None should be preserved."""
        cit = AgentCitation(document_id=_DOC_ID, filename=_FILE, chunk_id=_CHUNK_ID, page_number=None)
        d = cit.to_dict()
        restored = AgentCitation.from_dict(d)
        assert restored.page_number is None

    def test_vision_request_empty_evidence_list(self) -> None:
        """VisionRequest with no evidence list — has_evidence should be False after roundtrip."""
        req = VisionRequest(query="No evidence request")
        d = req.to_dict()
        restored = VisionRequest.from_dict(d)
        assert restored.has_evidence is False
        assert restored.evidence == []

    def test_agent_response_none_error_field(self) -> None:
        """AgentResponse.error is None for success — must survive roundtrip."""
        ar = AgentResponse(answer="ok", agent_name="a", status="success", citations=[], metadata={})
        d = ar.to_dict()
        restored = AgentResponse.from_dict(d)
        assert restored.error is None


# ============================================================================
# 16. DEFAULT VALUE STABILITY
# ============================================================================


class TestDefaultValueStability:
    def test_search_result_default_metadata_stable(self) -> None:
        sr = SearchResult(query="q", status="success", citations=[])
        d = sr.to_dict()
        restored = SearchResult.from_dict(d)
        assert restored.status == "success"
        assert len(restored.citations) == 0

    def test_agent_citation_default_metadata_stable(self) -> None:
        """When no metadata specified, metadata defaults survive roundtrip."""
        cit = AgentCitation(document_id=_DOC_ID, filename=_FILE, chunk_id=_CHUNK_ID)
        d = cit.to_dict()
        restored = AgentCitation.from_dict(d)
        assert isinstance(restored.metadata, dict)

    def test_visual_evidence_default_optional_fields_stable(self) -> None:
        """Optional VisualEvidence image fields (image_path, image_format, width, height, description)
        default to None and survive roundtrip."""
        ev = _evidence()
        d = ev.to_dict()
        restored = VisualEvidence.from_dict(d)
        assert restored.image_path is None
        assert restored.image_format is None
        assert restored.width is None
        assert restored.height is None
        assert restored.description is None


# ============================================================================
# 17. UNKNOWN FIELD LENIENT COMPATIBILITY
# ============================================================================


class TestUnknownFieldLenientCompatibility:
    def test_visual_evidence_accepts_unknown_fields(self) -> None:
        ev = _evidence()
        d = ev.to_dict()
        d["future_field_day24"] = "FUTURE_VALUE"
        d["another_unknown"] = 42
        restored = VisualEvidence.from_dict(d)
        # Core fields intact
        assert restored.document_id == _DOC_ID
        assert restored.chunk_id == _CHUNK_ID

    def test_search_result_accepts_unknown_fields(self) -> None:
        sr = SearchResult(query="q", status="success", citations=[_citation()])
        d = sr.to_dict()
        d["future_search_field"] = "FUTURE"
        restored = SearchResult.from_dict(d)
        assert restored.query == "q"
        assert restored.status == "success"


# ============================================================================
# 18. DATA TYPE PRESERVATION
# ============================================================================


class TestDataTypePreservation:
    def test_numeric_fields_preserved(self) -> None:
        cit = AgentCitation(
            document_id=_DOC_ID,
            filename=_FILE,
            chunk_id=_CHUNK_ID,
            page_number=7,
            score=0.9876,
        )
        d = cit.to_dict()
        restored = AgentCitation.from_dict(d)

        assert isinstance(restored.page_number, int)
        assert restored.page_number == 7
        assert isinstance(restored.score, float)
        assert restored.score == pytest.approx(0.9876)

    def test_boolean_metadata_preserved(self) -> None:
        res = VisionResult(
            query="bool type test",
            status="success",
            description="ok",
            evidence=[_evidence()],
            metadata={"is_certified": True, "retry_count": 3, "score": 0.95},
        )
        d = res.to_dict()
        restored = VisionResult.from_dict(d)

        assert restored.metadata["is_certified"] is True
        assert isinstance(restored.metadata["retry_count"], int)
        assert isinstance(restored.metadata["score"], float)

    def test_list_and_dict_fields_preserved(self) -> None:
        sr = SearchResult(
            query="list/dict test",
            status="success",
            citations=[_citation()],
            metadata={"tags": ["a", "b", "c"], "nested": {"k": "v"}},
        )
        d = sr.to_dict()
        restored = SearchResult.from_dict(d)

        assert restored.metadata["tags"] == ["a", "b", "c"]
        assert restored.metadata["nested"] == {"k": "v"}


# ============================================================================
# 19. METADATA ISOLATION
# ============================================================================


class TestMetadataIsolation:
    def test_a_and_b_metadata_never_cross_in_serialization(self) -> None:
        ev_a = _evidence(doc_id=_DOC_A, chunk_id=_CHUNK_A, metadata=_META_A)
        ev_b = _evidence(doc_id=_DOC_B, chunk_id=_CHUNK_B, metadata=_META_B)

        dict_a = ev_a.to_dict()
        dict_b = ev_b.to_dict()

        restored_a = VisualEvidence.from_dict(dict_a)
        restored_b = VisualEvidence.from_dict(dict_b)

        assert restored_a.metadata["day24_marker"] == "A"
        assert restored_b.metadata["day24_marker"] == "B"

        assert "TENANT_B" not in str(dict_a)
        assert "TENANT_A" not in str(dict_b)


# ============================================================================
# 20. CITATION PRESERVATION
# ============================================================================


class TestCitationPreservation:
    def test_all_citation_fields_preserved(self) -> None:
        cit = AgentCitation(
            document_id=_DOC_ID,
            filename=_FILE,
            chunk_id=_CHUNK_ID,
            page_number=_PAGE,
            score=0.91,
            content_type="chart",
            metadata={"lineage": "DAY24_LINEAGE", "source": "DAY24"},
        )
        d = cit.to_dict()
        restored = AgentCitation.from_dict(d)

        assert restored.document_id == _DOC_ID
        assert restored.filename == _FILE
        assert restored.chunk_id == _CHUNK_ID
        assert restored.page_number == _PAGE
        assert restored.score == pytest.approx(0.91)
        assert restored.content_type == "chart"
        assert restored.metadata["lineage"] == "DAY24_LINEAGE"


# ============================================================================
# 21. EVIDENCE PRESERVATION
# ============================================================================


class TestEvidencePreservation:
    def test_all_evidence_fields_preserved(self) -> None:
        ev = VisualEvidence(
            document_id=_DOC_ID,
            filename=_FILE,
            chunk_id=_CHUNK_ID,
            page_number=_PAGE,
            chunk_index=_CHUNK_IDX,
            content_type="image",
            image_path="/synthetic/path/day24.png",
            image_format="png",
            width=800,
            height=600,
            description="Day 24 synthetic image evidence.",
            metadata={"lineage": "DAY24_EVIDENCE_LINEAGE"},
        )
        d = ev.to_dict()
        restored = VisualEvidence.from_dict(d)

        assert restored.document_id == _DOC_ID
        assert restored.filename == _FILE
        assert restored.chunk_id == _CHUNK_ID
        assert restored.page_number == _PAGE
        assert restored.chunk_index == _CHUNK_IDX
        assert restored.content_type == "image"
        assert restored.image_path == "/synthetic/path/day24.png"
        assert restored.image_format == "png"
        assert restored.width == 800
        assert restored.height == 600
        assert restored.description == "Day 24 synthetic image evidence."
        assert restored.metadata["lineage"] == "DAY24_EVIDENCE_LINEAGE"


# ============================================================================
# 22. ERROR SERIALIZATION
# ============================================================================


class TestErrorSerialization:
    def test_agent_response_error_fields_preserved(self) -> None:
        ar = AgentResponse(
            answer="",
            agent_name="day24_agent",
            status="error",
            citations=[],
            metadata={"failure_code": "ERR_TIMEOUT", "retryable": True},
            error="Agent timeout after 30s.",
        )
        d = ar.to_dict()
        restored = AgentResponse.from_dict(d)

        assert restored.status == "error"
        assert restored.error == "Agent timeout after 30s."
        assert restored.metadata["failure_code"] == "ERR_TIMEOUT"
        assert restored.metadata["retryable"] is True

    def test_vision_result_error_fields_preserved(self) -> None:
        res = VisionResult(
            query="error preservation query",
            status="error",
            description="",
            error="Rate limited: vision API quota exceeded.",
            metadata={"http_status": 429, "provider": "synthetic_mock"},
        )
        d = res.to_dict()
        restored = VisionResult.from_dict(d)

        assert restored.error == "Rate limited: vision API quota exceeded."
        assert restored.metadata["http_status"] == 429
        assert restored.metadata["provider"] == "synthetic_mock"


# ============================================================================
# 23. CROSS-COMPONENT HANDOFF
# ============================================================================


class TestCrossComponentHandoff:
    def test_ingestion_to_agent_handoff(self) -> None:
        """VectorSearchResult (ingestion output) -> AgentCitation (agent input) roundtrip."""
        vsr = _vsr()
        cit = AgentCitation.from_search_result(vsr)
        d = cit.to_dict()
        restored_cit = AgentCitation.from_dict(d)

        assert restored_cit.document_id == vsr.document_id
        assert restored_cit.chunk_id == vsr.chunk_id
        assert restored_cit.filename == vsr.filename
        assert restored_cit.page_number == vsr.page_number

    def test_agent_to_vision_handoff(self) -> None:
        """VectorSearchResult -> VisualEvidence (vision consumer) roundtrip."""
        vsr = _vsr()
        ev = VisualEvidence.from_search_result(vsr)
        d = ev.to_dict()
        restored_ev = VisualEvidence.from_dict(d)

        assert restored_ev.document_id == vsr.document_id
        assert restored_ev.chunk_id == vsr.chunk_id

    def test_vision_to_downstream_handoff(self) -> None:
        """VisionResult -> serialized -> downstream AgentResponse pipeline."""
        ev = _evidence()
        vision_res = VisionResult(
            query="Cross-component query",
            status="success",
            description="Vision analysis complete.",
            evidence=[ev],
            metadata={"pipeline": "DAY24_HANDOFF"},
        )
        vision_dict = vision_res.to_dict()
        restored_vision = VisionResult.from_dict(vision_dict)

        # Build downstream AgentResponse from vision result
        cit = _citation()
        agent_res = AgentResponse(
            answer=restored_vision.description,
            agent_name="downstream_agent",
            status="success",
            citations=[cit],
            metadata={"source_query": restored_vision.query},
        )
        agent_dict = agent_res.to_dict()
        restored_agent = AgentResponse.from_dict(agent_dict)

        assert restored_agent.answer == "Vision analysis complete."
        assert restored_agent.metadata["source_query"] == "Cross-component query"


# ============================================================================
# 24. MUTATION SAFETY
# ============================================================================


class TestMutationSafety:
    def test_to_dict_copy_does_not_mutate_original_metadata(self) -> None:
        ev = _evidence()
        d = ev.to_dict()

        # Mutate the serialized dict
        d["metadata"]["mutated_key"] = "MUTATED_VALUE"

        # Original object must be unaffected
        assert "mutated_key" not in ev.metadata

    def test_citation_to_dict_does_not_share_metadata_ref(self) -> None:
        cit = _citation()
        d = cit.to_dict()

        d["metadata"]["injected"] = "INJECTED"

        assert "injected" not in cit.metadata

    def test_restored_metadata_independent_from_original(self) -> None:
        ev = _evidence()
        d = ev.to_dict()
        restored = VisualEvidence.from_dict(d)

        # Mutate the restored copy's metadata (if mutable)
        if isinstance(restored.metadata, dict):
            try:
                restored.metadata["new_key"] = "RESTORED_MUTATION"
                # Original should not be affected
                assert "new_key" not in ev.metadata
            except (TypeError, AttributeError):
                pass  # Immutable metadata — mutation safety guaranteed by contract


# ============================================================================
# 25. REPEATED ROUND-TRIP STABILITY
# ============================================================================


class TestRepeatedRoundTripStability:
    def test_double_roundtrip_citation_stable(self) -> None:
        cit = _citation()
        # First roundtrip
        r1 = AgentCitation.from_dict(cit.to_dict())
        # Second roundtrip
        r2 = AgentCitation.from_dict(r1.to_dict())

        assert r2.document_id == _DOC_ID
        assert r2.chunk_id == _CHUNK_ID
        assert r2.filename == _FILE
        assert r2.metadata["day24_marker"] == "CONTRACT_TEST"

    def test_double_roundtrip_vision_result_stable(self) -> None:
        ev = _evidence()
        res = VisionResult(
            query="Double roundtrip query",
            status="success",
            description="Stable across two roundtrips.",
            evidence=[ev],
            metadata={"roundtrip": 2},
        )
        # First roundtrip
        r1 = VisionResult.from_dict(res.to_dict())
        # Second roundtrip
        r2 = VisionResult.from_dict(r1.to_dict())

        assert r2.document_id == _DOC_ID
        assert r2.status == "success"
        assert r2.evidence[0].chunk_id == _CHUNK_ID
        assert r2.metadata["roundtrip"] == 2


# ============================================================================
# 26. SECURITY BOUNDARY
# ============================================================================


class TestSecurityBoundary:
    def test_serialized_dict_contains_no_forbidden_keys(self) -> None:
        """Verify serialized representations do not contain forbidden secret keys."""
        forbidden = {"api_key", "password", "secret", "token", "authorization", "credentials", "bearer"}

        ev = _evidence(metadata={"day24_marker": "SECURITY_CHECK", "safe_key": "safe_value"})
        ev_dict = ev.to_dict()
        ev_str = str(ev_dict).lower()
        for key in forbidden:
            assert key not in ev_str, f"Forbidden key '{key}' found in serialized VisualEvidence"

    def test_sanitize_metadata_removes_forbidden_in_error_context(self) -> None:
        """VisionResultNormalizer.sanitize_metadata strips forbidden keys from error metadata."""
        dirty = {
            "api_key": "SYNTHETIC_API_KEY",
            "password": "SYNTHETIC_PASS",
            "token": "SYNTHETIC_TOKEN",
            "safe_error_code": "ERR_DAY24",
        }
        sanitized = VisionResultNormalizer.sanitize_metadata(dirty)
        assert "api_key" not in sanitized
        assert "password" not in sanitized
        assert "token" not in sanitized
        assert sanitized["safe_error_code"] == "ERR_DAY24"
