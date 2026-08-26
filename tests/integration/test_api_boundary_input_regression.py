"""
OmniBrain Member 4 -- Day 17 API Contract Fuzzing & Boundary-Input Regression Tests.

Verifies that existing public APIs behave safely and predictably when given unusual,
empty, minimal, maximal, malformed, or boundary-shaped inputs.

Concern areas:
 1. Empty inputs -- empty string, whitespace, empty list, None where applicable
 2. Minimal valid inputs -- smallest valid objects that the contract accepts
 3. Boundary string values -- single-char, long, Unicode, newline, punctuation
 4. Boundary numeric values -- 0, 1, negative, boundary-range values
 5. Optional fields -- omitted, explicit None, empty metadata/collections
 6. Unknown fields in deserialization -- tolerated vs. silently ignored
 7. Invalid types -- string vs int, list vs dict, None for required fields
 8. Document/chunk boundaries -- single, many, empty metadata, minimal content
 9. Evidence boundaries -- zero, one, many, optional fields omitted
10. Request boundaries -- minimal, optional omitted, explicit defaults
11. Serialization boundaries -- to_dict()->from_dict()->to_dict() with edge inputs
12. Error boundaries -- exception category, no secrets, no unrelated state
13. Cross-member boundary -- edge inputs survive Ingestion->Search->Vision handoff
14. State isolation -- valid, invalid, valid sequence; invalid does not contaminate

Constraints:
 - 100% Offline: No external APIs, network, real LLMs, or production secrets.
 - Zero production code modified.
 - Only observable behavior guaranteed by existing public contracts tested.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# Ingestion Subsystem (Member 1)
from ingestion.models import DocumentChunk, VectorSearchResult

# Search / Agents Subsystem (Member 2)
from agents.models import (
    AgentCitation,
    AgentRequest,
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
from vision.result_normalizer import VisionExecutionTrace, VisionResultNormalizer


# ============================================================================
# Helpers
# ============================================================================

def _minimal_chunk(
    chunk_id: str = "chk-bnd-001",
    document_id: str = "doc-bnd-001",
    filename: str = "bnd.pdf",
    content: str = "x",
    content_type: str = "image",
    page_number: int | None = 1,
    chunk_index: int = 0,
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
        metadata=metadata if metadata is not None else {},
    )


def _minimal_vsr(
    chunk_id: str = "chk-bnd-001",
    document_id: str = "doc-bnd-001",
    filename: str = "bnd.pdf",
    content_type: str = "image",
    page_number: int | None = 1,
    score: float = 0.0,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        score=score,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        chunk_index=0,
        content_type=content_type,
        content="x",
        metadata={},
    )


def _minimal_ev(
    chunk_id: str = "chk-bnd-001",
    document_id: str = "doc-bnd-001",
    filename: str = "bnd.pdf",
    content_type: str = "image",
) -> VisualEvidence:
    return VisualEvidence(
        document_id=document_id,
        filename=filename,
        chunk_id=chunk_id,
        content_type=content_type,
    )


# ============================================================================
# 1. EMPTY INPUTS
# ============================================================================


class TestEmptyInputs:
    """Verifies existing contract behavior when empty inputs are provided."""

    # --- AgentCitation ---

    def test_citation_empty_document_id_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="f.pdf", chunk_id="c1")

    def test_citation_whitespace_document_id_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="   ", filename="f.pdf", chunk_id="c1")

    def test_citation_empty_filename_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="doc1", filename="", chunk_id="c1")

    def test_citation_empty_chunk_id_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="doc1", filename="f.pdf", chunk_id="")

    # --- SearchRequest ---

    def test_search_request_empty_query_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="")

    def test_search_request_whitespace_query_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="   ")

    # --- AgentRequest ---

    def test_agent_request_empty_query_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentRequest(query="")

    # --- VisionRequest ---

    def test_vision_request_empty_query_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="")

    def test_vision_request_whitespace_query_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="   ")

    def test_vision_request_empty_evidence_list_accepted(self) -> None:
        """Empty evidence list is valid -- query is required but evidence is optional."""
        req = VisionRequest(query="boundary query", evidence=[])
        assert req.has_evidence is False
        assert req.total_evidence == 0

    # --- VisualEvidence ---

    def test_visual_evidence_empty_document_id_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename="f.pdf", chunk_id="c1", content_type="image")

    def test_visual_evidence_empty_filename_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="doc1", filename="", chunk_id="c1", content_type="image")

    def test_visual_evidence_empty_chunk_id_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="doc1", filename="f.pdf", chunk_id="", content_type="image")

    def test_visual_evidence_invalid_content_type_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="doc1", filename="f.pdf", chunk_id="c1", content_type="video")

    # --- VisionExecutionTrace ---

    def test_trace_empty_stage_raises(self) -> None:
        trace = VisionExecutionTrace()
        with pytest.raises(VisionInputValidationError):
            trace.add_stage("")

    def test_trace_whitespace_stage_raises(self) -> None:
        trace = VisionExecutionTrace()
        with pytest.raises(VisionInputValidationError):
            trace.add_stage("   ")


# ============================================================================
# 2. MINIMAL VALID INPUTS
# ============================================================================


class TestMinimalValidInputs:
    """Verifies the smallest valid objects accepted by the existing public contracts."""

    def test_minimal_agent_citation(self) -> None:
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c")
        assert cit.document_id == "d"
        assert cit.filename == "f.pdf"
        assert cit.chunk_id == "c"
        assert cit.page_number is None
        assert cit.score == 0.0
        assert cit.metadata == {}
        assert isinstance(cit.to_dict(), dict)

    def test_minimal_search_request(self) -> None:
        req = SearchRequest(query="q")
        assert req.query == "q"
        assert req.top_k is None
        assert req.min_score is None
        assert req.max_results is None
        assert req.metadata == {}

    def test_minimal_agent_request(self) -> None:
        req = AgentRequest(query="q")
        assert req.query == "q"
        assert req.session_id is None
        assert req.document_filter is None
        assert req.metadata == {}

    def test_minimal_vision_request(self) -> None:
        req = VisionRequest(query="q")
        assert req.query == "q"
        assert req.evidence == []
        assert req.has_evidence is False

    def test_minimal_visual_evidence(self) -> None:
        ev = _minimal_ev()
        assert ev.document_id == "doc-bnd-001"
        assert ev.page_number is None
        assert ev.image_path is None
        assert ev.image_bytes is None
        assert ev.metadata == {}

    def test_minimal_vision_result(self) -> None:
        result = VisionResult(query="q", status="success", description="ok.")
        assert result.status == "success"
        assert result.error is None
        assert result.has_evidence is False

    def test_minimal_document_chunk(self) -> None:
        chunk = _minimal_chunk()
        assert chunk.chunk_id == "chk-bnd-001"
        assert chunk.metadata == {}

    def test_minimal_vector_search_result(self) -> None:
        vsr = _minimal_vsr()
        assert vsr.chunk_id == "chk-bnd-001"
        assert vsr.score == 0.0


# ============================================================================
# 3. BOUNDARY STRING VALUES
# ============================================================================


class TestBoundaryStringValues:
    """Verifies APIs handle safe boundary string shapes without unexpected crashes."""

    BOUNDARY_STRINGS = [
        "a",                              # single char
        "A" * 1000,                       # long string
        "Query with spaces and  tabs\t",  # whitespace containing
        "Unicode: \u4e2d\u6587 \u00e9\u00e0\u00fc",  # Unicode
        "Punctuation: !@#$%^&*()_+-=",    # punctuation
        "Multi\nline\ncontent",            # newlines
        "mixed: ABC123abc!?./",            # mixed
    ]

    def test_boundary_strings_as_query(self) -> None:
        for s in self.BOUNDARY_STRINGS:
            stripped = s.strip()
            if not stripped:
                continue  # skip empty/whitespace-only
            req = VisionRequest(query=s)
            assert req.query == stripped  # VisionRequest strips the query

    def test_boundary_strings_as_document_id(self) -> None:
        for s in self.BOUNDARY_STRINGS:
            stripped = s.strip()
            if not stripped:
                continue
            ev = VisualEvidence(
                document_id=s, filename="f.pdf", chunk_id="c", content_type="image"
            )
            assert ev.document_id == s

    def test_boundary_strings_as_metadata_values(self) -> None:
        for s in self.BOUNDARY_STRINGS:
            cit = AgentCitation(
                document_id="doc-bnd",
                filename="f.pdf",
                chunk_id="c-bnd",
                metadata={"test_value": s},
            )
            assert cit.metadata["test_value"] == s

    def test_long_content_string_in_chunk(self) -> None:
        long_content = "Token " * 500
        chunk = _minimal_chunk(content=long_content)
        assert len(chunk.content) > 1000


# ============================================================================
# 4. BOUNDARY NUMERIC VALUES
# ============================================================================


class TestBoundaryNumericValues:
    """Verifies existing validation behavior at contract-relevant numeric boundaries."""

    # --- page_number ---

    def test_page_number_none_accepted(self) -> None:
        ev = VisualEvidence(document_id="d", filename="f.pdf", chunk_id="c",
                            content_type="image", page_number=None)
        assert ev.page_number is None

    def test_page_number_one_accepted(self) -> None:
        ev = VisualEvidence(document_id="d", filename="f.pdf", chunk_id="c",
                            content_type="image", page_number=1)
        assert ev.page_number == 1

    def test_page_number_zero_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d", filename="f.pdf", chunk_id="c",
                           content_type="image", page_number=0)

    def test_page_number_negative_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d", filename="f.pdf", chunk_id="c",
                           content_type="image", page_number=-1)

    # --- AgentCitation page_number ---

    def test_citation_page_number_none_accepted(self) -> None:
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", page_number=None)
        assert cit.page_number is None

    def test_citation_page_number_one_accepted(self) -> None:
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", page_number=1)
        assert cit.page_number == 1

    def test_citation_page_number_zero_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", page_number=0)

    def test_citation_page_number_negative_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", page_number=-5)

    # --- score ---

    def test_citation_score_zero_accepted(self) -> None:
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", score=0.0)
        assert cit.score == 0.0

    def test_citation_score_negative_accepted(self) -> None:
        """Score validation only requires finite numeric -- negative is allowed."""
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", score=-0.5)
        assert cit.score == -0.5

    def test_citation_score_one_accepted(self) -> None:
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", score=1.0)
        assert cit.score == 1.0

    def test_citation_score_inf_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", score=math.inf)

    def test_citation_score_nan_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", score=math.nan)

    # --- SearchRequest top_k ---

    def test_search_request_top_k_one_accepted(self) -> None:
        req = SearchRequest(query="q", top_k=1)
        assert req.top_k == 1

    def test_search_request_top_k_zero_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="q", top_k=0)

    def test_search_request_top_k_negative_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="q", top_k=-1)

    # --- SearchRequest min_score ---

    def test_search_request_min_score_boundary_minus_one_accepted(self) -> None:
        req = SearchRequest(query="q", min_score=-1.0)
        assert req.min_score == -1.0

    def test_search_request_min_score_boundary_one_accepted(self) -> None:
        req = SearchRequest(query="q", min_score=1.0)
        assert req.min_score == 1.0

    def test_search_request_min_score_above_one_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="q", min_score=1.01)

    def test_search_request_min_score_below_minus_one_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="q", min_score=-1.01)

    # --- VisualEvidence width/height ---

    def test_visual_evidence_width_positive_accepted(self) -> None:
        ev = VisualEvidence(document_id="d", filename="f.pdf", chunk_id="c",
                            content_type="image", width=1)
        assert ev.width == 1

    def test_visual_evidence_width_zero_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d", filename="f.pdf", chunk_id="c",
                           content_type="image", width=0)

    def test_visual_evidence_height_negative_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d", filename="f.pdf", chunk_id="c",
                           content_type="image", height=-1)


# ============================================================================
# 5. OPTIONAL FIELDS
# ============================================================================


class TestOptionalFields:
    """Verifies models behave correctly with optional fields omitted or set to None/empty."""

    def test_citation_all_optional_omitted(self) -> None:
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c")
        assert cit.page_number is None
        assert cit.content_type == "text"
        assert cit.score == 0.0
        assert cit.metadata == {}

    def test_vision_request_no_evidence_no_session(self) -> None:
        req = VisionRequest(query="q")
        assert req.session_id is None
        assert req.evidence == []
        assert req.metadata == {}

    def test_vision_result_no_evidence_no_error(self) -> None:
        result = VisionResult(query="q", status="no_evidence", description="")
        assert result.error is None
        assert result.evidence == []
        assert result.has_evidence is False

    def test_visual_evidence_optional_fields_all_none(self) -> None:
        ev = VisualEvidence(
            document_id="d", filename="f.pdf", chunk_id="c", content_type="diagram",
            page_number=None, image_path=None, image_bytes=None,
            image_format=None, width=None, height=None, description=None,
        )
        assert ev.page_number is None
        assert ev.image_path is None
        assert ev.image_bytes is None
        assert ev.description is None

    def test_search_request_all_optional_omitted(self) -> None:
        req = SearchRequest(query="q")
        assert req.top_k is None
        assert req.min_score is None
        assert req.max_results is None
        assert req.collection_name is None
        assert req.session_id is None
        assert req.document_filter is None
        assert req.metadata == {}

    def test_agent_state_empty_metadata_accepted(self) -> None:
        state = AgentState(query="q", metadata={})
        assert state.metadata == {}

    def test_citation_empty_metadata_dict(self) -> None:
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", metadata={})
        assert cit.metadata == {}


# ============================================================================
# 6. UNKNOWN FIELDS IN DESERIALIZATION
# ============================================================================


class TestUnknownFieldsDeserialization:
    """Verifies that from_dict with extra unknown keys does not crash.
    Note: The contract silently ignores unknown keys (uses .get() for known fields).
    """

    def test_citation_from_dict_with_unknown_fields(self) -> None:
        data = {
            "document_id": "doc-unk",
            "filename": "unk.pdf",
            "chunk_id": "chk-unk",
            "score": 0.75,
            "page_number": 3,
            "test_unknown_field_day17": "IGNORED_VALUE",
            "another_unknown": 999,
        }
        cit = AgentCitation.from_dict(data)
        assert cit.document_id == "doc-unk"
        assert cit.chunk_id == "chk-unk"
        assert cit.score == 0.75

    def test_search_request_from_dict_with_unknown_fields(self) -> None:
        data = {
            "query": "boundary query",
            "top_k": 5,
            "unknown_future_param": "FUTURE_VALUE",
        }
        req = SearchRequest.from_dict(data)
        assert req.query == "boundary query"
        assert req.top_k == 5

    def test_vision_result_from_dict_with_unknown_fields(self) -> None:
        ev = _minimal_ev()
        orig = VisionResult(query="q", status="success", description="ok.", evidence=[ev])
        data = orig.to_dict()
        data["test_unknown_day17"] = "BOUNDARY_UNKNOWN"
        restored = VisionResult.from_dict(data)
        assert restored.status == "success"
        assert restored.query == "q"


# ============================================================================
# 7. INVALID TYPES
# ============================================================================


class TestInvalidTypes:
    """Verifies invalid types produce expected validation errors."""

    def test_citation_document_id_int_raises(self) -> None:
        with pytest.raises((AgentValidationError, TypeError)):
            AgentCitation(document_id=123, filename="f.pdf", chunk_id="c")  # type: ignore[arg-type]

    def test_citation_score_string_raises(self) -> None:
        with pytest.raises((AgentValidationError, TypeError, ValueError)):
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", score="high")  # type: ignore[arg-type]

    def test_citation_metadata_list_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", metadata=["a", "b"])  # type: ignore[arg-type]

    def test_search_request_query_int_raises(self) -> None:
        with pytest.raises((AgentValidationError, TypeError)):
            SearchRequest(query=42)  # type: ignore[arg-type]

    def test_search_request_top_k_float_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="q", top_k=5.5)  # type: ignore[arg-type]

    def test_vision_request_evidence_dict_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="q", evidence={"key": "value"})  # type: ignore[arg-type]

    def test_vision_request_evidence_with_non_evidence_item_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="q", evidence=["not_an_evidence"])  # type: ignore[list-item]

    def test_visual_evidence_page_number_bool_raises(self) -> None:
        """bool is a subclass of int -- the contract explicitly rejects booleans as page_number."""
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d", filename="f.pdf", chunk_id="c",
                           content_type="image", page_number=True)


# ============================================================================
# 8. DOCUMENT / CHUNK BOUNDARIES
# ============================================================================


class TestDocumentChunkBoundaries:
    """Verifies IDs, metadata, lineage remain consistent at document/chunk boundaries."""

    def test_single_document_single_chunk(self) -> None:
        chunk = _minimal_chunk(chunk_id="c0", document_id="doc-single", filename="single.pdf")
        assert chunk.document_id == "doc-single"
        assert chunk.chunk_index == 0

    def test_multiple_chunks_same_document(self) -> None:
        chunks = [
            _minimal_chunk(
                chunk_id=f"c{i}", document_id="doc-multi", filename="multi.pdf",
                chunk_index=i, content=f"Content {i}",
            )
            for i in range(10)
        ]
        assert all(c.document_id == "doc-multi" for c in chunks)
        assert [c.chunk_index for c in chunks] == list(range(10))

    def test_multiple_documents(self) -> None:
        docs = {f"doc-{i}": f"doc_{i}.pdf" for i in range(5)}
        chunks = [
            _minimal_chunk(chunk_id=f"c-d{i}", document_id=did, filename=fn)
            for i, (did, fn) in enumerate(docs.items())
        ]
        for i, (did, fn) in enumerate(docs.items()):
            assert chunks[i].document_id == did
            assert chunks[i].filename == fn

    def test_chunk_with_empty_metadata(self) -> None:
        chunk = _minimal_chunk()
        assert chunk.metadata == {}

    def test_chunk_with_rich_metadata(self) -> None:
        meta = {"key_a": "value_a", "key_b": 42, "nested": {"inner": True}}
        chunk = _minimal_chunk(metadata=meta)
        assert chunk.metadata["key_a"] == "value_a"
        assert chunk.metadata["key_b"] == 42


# ============================================================================
# 9. EVIDENCE BOUNDARIES
# ============================================================================


class TestEvidenceBoundaries:
    """Verifies existing result contracts at evidence count boundaries."""

    def test_zero_evidence_request_and_result(self) -> None:
        req = VisionRequest(query="zero evidence", evidence=[])
        assert req.has_evidence is False
        result = VisionResult(query=req.query, status="no_evidence", description="")
        assert result.has_evidence is False

    def test_one_evidence_request_and_result(self) -> None:
        ev = _minimal_ev()
        req = VisionRequest(query="one evidence", evidence=[ev])
        assert req.total_evidence == 1
        result = VisionResult(query=req.query, status="success", description="ok.", evidence=[ev])
        assert result.has_evidence is True
        assert len(result.evidence) == 1

    def test_many_evidence_request_and_result(self) -> None:
        evidences = [
            _minimal_ev(
                chunk_id=f"c{i}", document_id=f"d{i}", filename=f"f{i}.pdf",
                content_type="chart" if i % 2 == 0 else "diagram",
            )
            for i in range(12)
        ]
        req = VisionRequest(query="many evidence", evidence=evidences)
        assert req.total_evidence == 12
        result = VisionResult(query=req.query, status="success", description="ok.",
                              evidence=req.evidence)
        assert len(result.evidence) == 12

    def test_all_valid_visual_content_types_accepted(self) -> None:
        for ct in sorted(VALID_VISUAL_CONTENT_TYPES):
            ev = VisualEvidence(
                document_id="d", filename="f.pdf", chunk_id="c",
                content_type=ct,
            )
            assert ev.content_type == ct

    def test_evidence_minimal_optional_fields(self) -> None:
        ev = VisualEvidence(
            document_id="d", filename="f.pdf", chunk_id="c", content_type="image",
        )
        assert ev.width is None
        assert ev.height is None
        assert ev.description is None
        assert ev.image_path is None
        assert ev.image_bytes is None
        assert isinstance(ev.to_dict(), dict)


# ============================================================================
# 10. REQUEST BOUNDARIES
# ============================================================================


class TestRequestBoundaries:
    """Verifies contract behavior at request-level boundaries."""

    def test_minimal_search_request_serialization(self) -> None:
        req = SearchRequest(query="min")
        data = req.to_dict()
        restored = SearchRequest.from_dict(data)
        assert restored.query == "min"
        assert restored.top_k is None

    def test_search_request_with_all_optional_set(self) -> None:
        req = SearchRequest(
            query="full request",
            top_k=20,
            min_score=0.5,
            max_results=10,
            collection_name="test_collection",
            session_id="sess-bnd-001",
            document_filter={"doc_id": "d1"},
            metadata={"priority": "high"},
        )
        assert req.top_k == 20
        assert req.min_score == 0.5
        assert req.max_results == 10
        assert req.collection_name == "test_collection"
        assert req.session_id == "sess-bnd-001"
        assert req.metadata["priority"] == "high"

    def test_vision_request_with_empty_metadata_dict(self) -> None:
        req = VisionRequest(query="q", metadata={})
        assert req.metadata == {}

    def test_vision_request_session_id_stripped(self) -> None:
        req = VisionRequest(query="q", session_id="  sess-001  ")
        assert req.session_id == "sess-001"

    def test_vision_request_session_id_whitespace_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="q", session_id="   ")


# ============================================================================
# 11. SERIALIZATION BOUNDARIES
# ============================================================================


class TestSerializationBoundaries:
    """Verifies to_dict()->from_dict()->to_dict() stability with boundary inputs."""

    def test_minimal_citation_round_trip(self) -> None:
        orig = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c")
        d1 = orig.to_dict()
        restored = AgentCitation.from_dict(d1)
        d2 = restored.to_dict()
        assert d1 == d2

    def test_citation_with_unicode_metadata_round_trip(self) -> None:
        orig = AgentCitation(
            document_id="doc-uni",
            filename="\u6587\u4ef6.pdf",
            chunk_id="chk-uni",
            metadata={"unicode_key": "\u4e2d\u6587\u5185\u5bb9", "emoji": "\U0001f9e0"},
        )
        d1 = orig.to_dict()
        restored = AgentCitation.from_dict(d1)
        assert restored.filename == "\u6587\u4ef6.pdf"
        assert restored.metadata["unicode_key"] == "\u4e2d\u6587\u5185\u5bb9"

    def test_vision_result_zero_evidence_round_trip(self) -> None:
        orig = VisionResult(query="round trip zero", status="no_evidence", description="")
        d1 = orig.to_dict()
        restored = VisionResult.from_dict(d1)
        d2 = restored.to_dict()
        assert d1["status"] == d2["status"]
        assert d1["query"] == d2["query"]

    def test_vision_result_with_evidence_round_trip(self) -> None:
        ev = VisualEvidence(
            document_id="doc-rt",
            filename="\u00e9tude.pdf",
            chunk_id="chk-rt",
            content_type="chart",
            page_number=3,
            metadata={"note": "boundary test"},
        )
        orig = VisionResult(
            query="round trip evidence",
            status="success",
            description="RT verified.",
            evidence=[ev],
        )
        d1 = orig.to_dict()
        restored = VisionResult.from_dict(d1)
        d2 = restored.to_dict()
        assert d1["document_id"] == d2["document_id"]
        assert len(d1["evidence"]) == len(d2["evidence"])
        assert d1["evidence"][0]["metadata"] == d2["evidence"][0]["metadata"]

    def test_vision_result_with_newline_description_round_trip(self) -> None:
        orig = VisionResult(
            query="newline test",
            status="success",
            description="Line one.\nLine two.\nLine three.",
        )
        d1 = orig.to_dict()
        restored = VisionResult.from_dict(d1)
        assert restored.description == orig.description

    def test_vision_execution_trace_serialization(self) -> None:
        trace = VisionExecutionTrace(initial_stages=["s1", "s2", "s3"])
        data = trace.to_dict()
        assert data["stages"] == ["s1", "s2", "s3"]
        assert data["stage_count"] == 3


# ============================================================================
# 12. ERROR BOUNDARIES
# ============================================================================


class TestErrorBoundaries:
    """Verifies error behavior at API boundaries: correct exception, no secrets, no cross-state."""

    def test_validation_error_message_does_not_expose_secrets(self) -> None:
        try:
            AgentCitation(document_id="", filename="f.pdf", chunk_id="c")
        except AgentValidationError as exc:
            msg = str(exc)
            assert "password" not in msg.lower()
            assert "secret" not in msg.lower()
            assert "api_key" not in msg.lower()

    def test_vision_error_message_does_not_expose_secrets(self) -> None:
        try:
            VisualEvidence(document_id="", filename="f.pdf", chunk_id="c", content_type="image")
        except VisionEvidenceError as exc:
            msg = str(exc)
            assert "password" not in msg.lower()
            assert "api_key" not in msg.lower()

    def test_error_result_contains_no_other_request_state(self) -> None:
        result_ok = VisionResult(query="OK query", status="success", description="ok.")
        result_err = VisionResult(
            query="Error query", status="error", description="", error="Timeout."
        )
        assert "OK query" not in str(result_err.to_dict())
        assert result_ok.status == "success"
        assert result_ok.error is None

    def test_repeated_boundary_errors_stay_consistent(self) -> None:
        for _ in range(4):
            with pytest.raises(AgentValidationError):
                AgentCitation(document_id="", filename="f.pdf", chunk_id="c")


# ============================================================================
# 13. CROSS-MEMBER BOUNDARY
# ============================================================================


class TestCrossMemberBoundary:
    """Verifies edge inputs survive the Ingestion->Search->Vision handoff."""

    def test_minimal_chunk_to_vsr_to_citation_to_evidence_chain(self) -> None:
        chunk = _minimal_chunk(
            chunk_id="chk-cross-001",
            document_id="doc-cross-001",
            filename="cross.pdf",
            content="x",
            content_type="image",
        )
        vsr = VectorSearchResult(
            chunk_id=chunk.chunk_id,
            score=0.0,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content_type=chunk.content_type,
            content=chunk.content,
            metadata=chunk.metadata,
        )
        cit = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_citation(cit)
        result = VisionResult(
            query="Cross member boundary test",
            status="success",
            description="Cross-member boundary verified.",
            evidence=[ev],
        )

        assert result.document_id == "doc-cross-001"
        assert result.chunk_id == "chk-cross-001"
        assert cit.score == 0.0
        assert ev.content_type == "image"

    def test_minimal_evidence_with_all_content_types_survive_handoff(self) -> None:
        for ct in sorted(VALID_VISUAL_CONTENT_TYPES):
            vsr = _minimal_vsr(
                chunk_id=f"c-{ct}", document_id=f"d-{ct}",
                filename=f"{ct}.pdf", content_type=ct,
            )
            cit = AgentCitation.from_search_result(vsr)
            ev = VisualEvidence.from_citation(cit)
            result = VisionResult(
                query=f"Boundary query for {ct}",
                status="success",
                description=f"Boundary {ct} ok.",
                evidence=[ev],
            )
            assert result.has_evidence is True
            assert result.evidence[0].content_type == ct


# ============================================================================
# 14. STATE ISOLATION — valid, invalid, valid
# ============================================================================


class TestStateIsolation:
    """Verifies invalid request does not corrupt subsequent valid request."""

    def test_valid_invalid_valid_sequence_isolation(self) -> None:
        # First valid
        ev1 = _minimal_ev(chunk_id="c-first", document_id="doc-first", filename="first.pdf")
        result1 = VisionResult(
            query="First valid query",
            status="success",
            description="First ok.",
            evidence=[ev1],
        )

        # Invalid -- raises, no side effects
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="bad.pdf", chunk_id="c-bad")

        # Second valid -- must be completely clean
        ev2 = _minimal_ev(chunk_id="c-second", document_id="doc-second", filename="second.pdf")
        result2 = VisionResult(
            query="Second valid query",
            status="success",
            description="Second ok.",
            evidence=[ev2],
        )

        assert result1.document_id == "doc-first"
        assert result2.document_id == "doc-second"
        assert result1.status == "success"
        assert result2.status == "success"
        assert "doc-second" not in str(result1.to_dict())
        assert "doc-first" not in str(result2.to_dict())

    def test_agent_state_not_contaminated_after_citation_failure(self) -> None:
        state = AgentState(query="Isolation state query")

        # Attempt to add a bad citation -- must not affect state
        try:
            bad_cit = AgentCitation(document_id="", filename="f.pdf", chunk_id="c")
            state.add_citation(bad_cit)
        except AgentValidationError:
            pass

        assert len(state.citations) == 0

        # Good citation added after failure
        good_cit = AgentCitation(document_id="doc-good", filename="g.pdf", chunk_id="c-good")
        state.add_citation(good_cit)
        assert len(state.citations) == 1
        assert state.citations[0].document_id == "doc-good"
