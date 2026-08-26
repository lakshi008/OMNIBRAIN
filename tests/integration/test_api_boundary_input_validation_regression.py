"""
OmniBrain Member 4 — Day 33 API Boundary & Input Validation Regression Certification.

Validates the existing public API boundaries across:
  - Ingestion (DocumentChunk, ChunkingResult, ChunkValidationResult,
               EmbeddingPreparationResult, VectorSearchResult,
               validate_chunks, normalize_chunks, prepare_for_embedding,
               process_retrieval_results, build_retrieval_context)
  - Agents / Search (AgentCitation, AgentRequest, SearchRequest,
                     AgentResponse, SearchResult, AgentState)
  - Vision (VisualEvidence, VisionRequest, VisionResult,
             VisualEvidenceAdapter, VisionResultNormalizer,
             VisionExecutionTrace)

Focus areas:
  1.  Valid inputs.
  2.  Empty inputs.
  3.  None values.
  4.  Wrong types.
  5.  Missing required fields.
  6.  Optional fields.
  7.  Extra / unknown fields.
  8.  Keyword arguments.
  9.  Numeric boundaries.
  10. String boundaries.
  11. Batch/list boundaries.
  12. Nested object validation.
  13. Serialization boundaries.
  14. Round-trip validation.
  15. Cross-component compatibility.
  16. Error contract.
  17. Input mutation safety.
  18. Invalid to Valid recovery.
  19. Concurrent isolation.

Constraints:
  - 100% offline.
  - Zero production code modified.
  - No validators, schemas, wrappers, middleware, or adapters added.
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

from ingestion.models import (
    ChunkingResult,
    ChunkValidationResult,
    DocumentChunk,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    VectorSearchResult,
)
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.retrieval_processor import (
    build_retrieval_context,
    process_retrieval_results,
)
from ingestion.ingestion_errors import (
    IngestionError,
    IngestionValidationError,
    IngestionChunkingError,
    IngestionEmbeddingError,
)
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
    AgentValidationError,
    AgentRoutingError,
    AgentExecutionError,
)
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
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
)

DOC_ID = "DAY33_DOC_ID"
FILENAME = "day33_regression.pdf"
QUERY = "Day 33 API boundary regression query"


def _make_chunk(
    chunk_id: str = "chk_d33_00",
    chunk_index: int = 0,
    document_id: str = DOC_ID,
    filename: str = FILENAME,
    page_number: int | None = 1,
    content: str = "Day 33 synthetic chunk content",
    content_type: str = "text",
    metadata: dict | None = None,
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


def _make_image_chunk(idx: int = 0) -> DocumentChunk:
    return _make_chunk(
        chunk_id=f"chk_img_{idx:02d}",
        chunk_index=idx,
        content_type="image",
        content=f"[image reference {idx}]",
    )


def _make_vsr(
    chunk_id: str = "vsr_d33_00",
    score: float = 0.90,
    document_id: str = DOC_ID,
    filename: str = FILENAME,
    page_number: int = 1,
    chunk_index: int = 0,
    content_type: str = "image",
    content: str = "Day 33 synthetic retrieval content",
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
        metadata={},
    )


def _make_citation(
    document_id: str = DOC_ID,
    filename: str = FILENAME,
    chunk_id: str = "chk_d33_00",
    page_number: int | None = 1,
    content_type: str = "image",
    score: float = 0.90,
) -> AgentCitation:
    return AgentCitation(
        document_id=document_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        content_type=content_type,
        score=score,
    )


# ===========================================================================
# 1. INGESTION — validate_chunks valid inputs
# ===========================================================================

class TestValidateChunksValidInputs:
    def test_valid_single_text_chunk(self) -> None:
        chunks = [_make_chunk()]
        result = validate_chunks(chunks)
        assert isinstance(result, ChunkValidationResult)
        assert result.is_valid is True
        assert result.total_chunks == 1
        assert result.valid_chunks == 1
        assert result.invalid_chunks == 0
        assert result.errors == []

    def test_valid_multiple_chunks_mixed_types(self) -> None:
        chunks = [
            _make_chunk(chunk_id="c0", chunk_index=0, content_type="text"),
            _make_chunk(chunk_id="c1", chunk_index=1, content_type="table", content="| col1 | col2 |"),
            _make_chunk(chunk_id="c2", chunk_index=2, content_type="image", content="[image ref]"),
        ]
        result = validate_chunks(chunks)
        assert result.is_valid is True
        assert result.total_chunks == 3

    def test_valid_chunking_result_input(self) -> None:
        chunks = [_make_chunk(chunk_id="c0", chunk_index=0), _make_chunk(chunk_id="c1", chunk_index=1)]
        cr = ChunkingResult(document_id=DOC_ID, filename=FILENAME, chunks=chunks)
        result = validate_chunks(cr)
        assert result.is_valid is True
        assert result.total_chunks == 2

    def test_valid_chunk_with_none_page_number(self) -> None:
        chunk = _make_chunk(page_number=None)
        result = validate_chunks([chunk])
        assert result.is_valid is True

    def test_valid_chunk_with_rich_metadata(self) -> None:
        chunk = _make_chunk(metadata={"source": "test", "classification": "public"})
        result = validate_chunks([chunk])
        assert result.is_valid is True


# ===========================================================================
# 2. INGESTION — validate_chunks empty inputs
# ===========================================================================

class TestValidateChunksEmptyInputs:
    def test_empty_list_returns_valid_result(self) -> None:
        result = validate_chunks([])
        assert isinstance(result, ChunkValidationResult)
        assert result.is_valid is True
        assert result.total_chunks == 0

    def test_empty_chunking_result_returns_valid(self) -> None:
        cr = ChunkingResult(document_id=DOC_ID, filename=FILENAME, chunks=[])
        result = validate_chunks(cr)
        assert result.is_valid is True
        assert result.total_chunks == 0


# ===========================================================================
# 3. INGESTION — validate_chunks wrong types
# ===========================================================================

class TestValidateChunksWrongTypes:
    def test_string_input_returns_invalid_result(self) -> None:
        result = validate_chunks("not a list")
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_dict_input_returns_invalid_result(self) -> None:
        result = validate_chunks({"chunks": []})
        assert result.is_valid is False

    def test_none_input_returns_invalid_result(self) -> None:
        result = validate_chunks(None)
        assert result.is_valid is False


# ===========================================================================
# 4. INGESTION — validate_chunks missing required chunk fields
# ===========================================================================

class TestValidateChunksMissingFields:
    def test_whitespace_chunk_id_invalid(self) -> None:
        bad_chunk = _make_chunk(chunk_id="  ", chunk_index=1)
        result = validate_chunks([bad_chunk])
        assert result.is_valid is False
        assert result.invalid_chunks > 0

    def test_whitespace_document_id_invalid(self) -> None:
        bad_chunk = DocumentChunk(
            chunk_id="chk_bad", chunk_index=0, document_id="   ",
            filename=FILENAME, page_number=1, content="some content", content_type="text",
        )
        result = validate_chunks([bad_chunk])
        assert result.is_valid is False

    def test_whitespace_only_content_invalid(self) -> None:
        bad_chunk = DocumentChunk(
            chunk_id="chk_bad", chunk_index=0, document_id=DOC_ID,
            filename=FILENAME, page_number=1, content="   ", content_type="text",
        )
        result = validate_chunks([bad_chunk])
        assert result.is_valid is False

    def test_invalid_content_type_invalid(self) -> None:
        bad_chunk = DocumentChunk(
            chunk_id="chk_bad", chunk_index=0, document_id=DOC_ID,
            filename=FILENAME, page_number=1, content="some content", content_type="video",
        )
        result = validate_chunks([bad_chunk])
        assert result.is_valid is False

    def test_page_number_zero_invalid(self) -> None:
        bad_chunk = DocumentChunk(
            chunk_id="chk_bad", chunk_index=0, document_id=DOC_ID,
            filename=FILENAME, page_number=0, content="some content", content_type="text",
        )
        result = validate_chunks([bad_chunk])
        assert result.is_valid is False


# ===========================================================================
# 5. INGESTION — normalize_chunks boundaries
# ===========================================================================

class TestNormalizeChunksBoundaries:
    def test_normalize_empty_list(self) -> None:
        result = normalize_chunks([])
        assert result == []

    def test_normalize_single_chunk_strips_whitespace(self) -> None:
        chunk = _make_chunk(content="  leading and trailing  ")
        result = normalize_chunks([chunk])
        assert len(result) == 1
        assert result[0].content == "leading and trailing"

    def test_normalize_preserves_chunk_id_and_index(self) -> None:
        chunk = _make_chunk(chunk_id="specific_id", chunk_index=7)
        result = normalize_chunks([chunk])
        assert result[0].chunk_id == "specific_id"
        assert result[0].chunk_index == 7

    def test_normalize_wrong_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            normalize_chunks("not a list")

    def test_normalize_multiple_chunks_preserves_count(self) -> None:
        chunks = [_make_chunk(chunk_id=f"c{i}", chunk_index=i) for i in range(5)]
        result = normalize_chunks(chunks)
        assert len(result) == 5


# ===========================================================================
# 6. INGESTION — prepare_for_embedding boundaries
# ===========================================================================

class TestPrepareForEmbeddingBoundaries:
    def test_valid_list_returns_embedding_result(self) -> None:
        result = prepare_for_embedding([_make_chunk()])
        assert isinstance(result, EmbeddingPreparationResult)
        assert result.is_ready is True
        assert result.total_items == 1

    def test_empty_list_returns_ready_empty_result(self) -> None:
        result = prepare_for_embedding([])
        assert result.is_ready is True
        assert result.total_items == 0

    def test_chunking_result_input_accepted(self) -> None:
        cr = ChunkingResult(document_id=DOC_ID, filename=FILENAME, chunks=[_make_chunk()])
        result = prepare_for_embedding(cr)
        assert result.is_ready is True
        assert result.document_id == DOC_ID

    def test_wrong_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            prepare_for_embedding("not valid")

    def test_invalid_chunk_raises_value_error(self) -> None:
        bad_chunk = DocumentChunk(
            chunk_id="", chunk_index=0, document_id=DOC_ID,
            filename=FILENAME, page_number=1, content="content", content_type="text",
        )
        with pytest.raises(ValueError):
            prepare_for_embedding([bad_chunk])

    def test_embedding_records_have_correct_fields(self) -> None:
        chunk = _make_chunk(chunk_id="emb_chk_01", chunk_index=0, content="Embedding content")
        result = prepare_for_embedding([chunk])
        rec = result.items[0]
        assert isinstance(rec, EmbeddingRecord)
        assert rec.chunk_id == "emb_chk_01"
        assert rec.document_id == DOC_ID
        assert rec.content == "Embedding content"

    def test_deterministic_ordering_by_chunk_index(self) -> None:
        chunks = [
            _make_chunk(chunk_id="c2", chunk_index=2, content="Third"),
            _make_chunk(chunk_id="c0", chunk_index=0, content="First"),
            _make_chunk(chunk_id="c1", chunk_index=1, content="Second"),
        ]
        result = prepare_for_embedding(chunks)
        assert [r.chunk_index for r in result.items] == [0, 1, 2]


# ===========================================================================
# 7. INGESTION — process_retrieval_results boundaries
# ===========================================================================

class TestProcessRetrievalResultsBoundaries:
    def test_valid_list_returns_filtered_results(self) -> None:
        vsrs = [_make_vsr(chunk_id=f"v{i}", chunk_index=i, score=round(0.95 - i * 0.1, 2)) for i in range(3)]
        result = process_retrieval_results(vsrs, min_score=0.5, max_results=10)
        assert len(result) == 3
        assert all(r.score >= 0.5 for r in result)

    def test_empty_list_returns_empty(self) -> None:
        assert process_retrieval_results([], min_score=0.0, max_results=5) == []

    def test_min_score_filters_correctly(self) -> None:
        vsrs = [
            _make_vsr(chunk_id="high", chunk_index=0, score=0.90),
            _make_vsr(chunk_id="low", chunk_index=1, score=0.30),
        ]
        result = process_retrieval_results(vsrs, min_score=0.5, max_results=10)
        assert len(result) == 1
        assert result[0].chunk_id == "high"

    def test_max_results_limits_output(self) -> None:
        vsrs = [_make_vsr(chunk_id=f"v{i}", chunk_index=i, score=round(0.9 - i * 0.05, 2)) for i in range(5)]
        result = process_retrieval_results(vsrs, min_score=0.0, max_results=2)
        assert len(result) == 2

    def test_wrong_type_not_list_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            process_retrieval_results("not a list", min_score=0.0, max_results=5)

    def test_wrong_item_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            process_retrieval_results(["not a vsr"], min_score=0.0, max_results=5)

    def test_invalid_min_score_too_high_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            process_retrieval_results([_make_vsr()], min_score=2.0, max_results=5)

    def test_invalid_max_results_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            process_retrieval_results([_make_vsr()], min_score=0.0, max_results=0)

    def test_results_sorted_by_score_descending(self) -> None:
        vsrs = [
            _make_vsr(chunk_id="low", chunk_index=1, score=0.50),
            _make_vsr(chunk_id="high", chunk_index=0, score=0.95),
        ]
        result = process_retrieval_results(vsrs, min_score=0.0, max_results=10)
        assert result[0].chunk_id == "high"


# ===========================================================================
# 8. INGESTION — build_retrieval_context boundaries
# ===========================================================================

class TestBuildRetrievalContextBoundaries:
    def test_empty_list_returns_empty_string(self) -> None:
        assert build_retrieval_context([]) == ""

    def test_single_result_produces_source_block(self) -> None:
        vsr = _make_vsr(filename=FILENAME, page_number=1, content="Test content")
        ctx = build_retrieval_context([vsr])
        assert "[Source 1]" in ctx
        assert FILENAME in ctx
        assert "Test content" in ctx

    def test_multiple_results_produce_numbered_blocks(self) -> None:
        vsrs = [
            _make_vsr(chunk_id="v0", chunk_index=0, content="First"),
            _make_vsr(chunk_id="v1", chunk_index=1, content="Second"),
        ]
        ctx = build_retrieval_context(vsrs)
        assert "[Source 1]" in ctx
        assert "[Source 2]" in ctx

    def test_wrong_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            build_retrieval_context("not a list")


# ===========================================================================
# 9. AGENTS — AgentCitation valid inputs and boundaries
# ===========================================================================

class TestAgentCitationValidInputs:
    def test_valid_citation_with_all_fields(self) -> None:
        cit = AgentCitation(
            document_id=DOC_ID, filename=FILENAME, chunk_id="chk_00",
            page_number=1, content_type="image", score=0.95,
            metadata={"key": "value"},
        )
        assert cit.document_id == DOC_ID
        assert cit.score == 0.95

    def test_valid_citation_defaults(self) -> None:
        cit = AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="chk_00")
        assert cit.page_number is None
        assert cit.content_type == "text"
        assert cit.score == 0.0
        assert cit.metadata == {}

    def test_citation_from_search_result(self) -> None:
        vsr = _make_vsr()
        cit = AgentCitation.from_search_result(vsr)
        assert isinstance(cit, AgentCitation)
        assert cit.document_id == DOC_ID

    def test_citation_score_zero_boundary(self) -> None:
        cit = AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", score=0.0)
        assert cit.score == 0.0

    def test_citation_score_one_boundary(self) -> None:
        cit = AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", score=1.0)
        assert cit.score == 1.0

    def test_citation_negative_score_accepted(self) -> None:
        cit = AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", score=-0.5)
        assert cit.score == -0.5

    def test_citation_unicode_document_id(self) -> None:
        cit = AgentCitation(document_id="日本語ドキュメント", filename=FILENAME, chunk_id="chk")
        assert cit.document_id == "日本語ドキュメント"

    def test_citation_long_filename(self) -> None:
        long_name = "a" * 250 + ".pdf"
        cit = AgentCitation(document_id=DOC_ID, filename=long_name, chunk_id="c0")
        assert cit.filename == long_name


# ===========================================================================
# 10. AGENTS — AgentCitation invalid inputs
# ===========================================================================

class TestAgentCitationInvalidInputs:
    def test_empty_document_id_raises(self) -> None:
        with pytest.raises(AgentValidationError) as exc_info:
            AgentCitation(document_id="", filename=FILENAME, chunk_id="c0")
        assert "document_id" in str(exc_info.value).lower()

    def test_whitespace_document_id_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="   ", filename=FILENAME, chunk_id="c0")

    def test_empty_filename_raises(self) -> None:
        with pytest.raises(AgentValidationError) as exc_info:
            AgentCitation(document_id=DOC_ID, filename="", chunk_id="c0")
        assert "filename" in str(exc_info.value).lower()

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(AgentValidationError) as exc_info:
            AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="")
        assert "chunk_id" in str(exc_info.value).lower()

    def test_page_number_zero_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", page_number=0)

    def test_page_number_negative_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", page_number=-1)

    def test_non_finite_score_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", score=float("inf"))

    def test_non_numeric_score_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", score="high")

    def test_non_dict_metadata_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", metadata="bad")

    def test_exception_is_agent_error(self) -> None:
        with pytest.raises(AgentError):
            AgentCitation(document_id="", filename=FILENAME, chunk_id="c0")


# ===========================================================================
# 11. AGENTS — AgentCitation serialization boundaries
# ===========================================================================

class TestAgentCitationSerializationBoundaries:
    def test_to_dict_produces_correct_keys(self) -> None:
        d = _make_citation().to_dict()
        assert set(d.keys()) >= {"document_id", "filename", "chunk_id",
                                  "page_number", "content_type", "score", "metadata"}

    def test_round_trip_to_dict_from_dict(self) -> None:
        cit = _make_citation()
        d1 = cit.to_dict()
        cit2 = AgentCitation.from_dict(d1)
        assert cit2.document_id == cit.document_id
        assert cit2.chunk_id == cit.chunk_id
        assert cit2.score == cit.score
        assert cit2.to_dict() == d1

    def test_from_dict_missing_document_id_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation.from_dict({"filename": FILENAME, "chunk_id": "c0"})

    def test_from_dict_non_dict_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation.from_dict("not a dict")

    def test_from_dict_extra_fields_absorbed(self) -> None:
        d = {
            "document_id": DOC_ID,
            "filename": FILENAME,
            "chunk_id": "c0",
            "day33_unknown_field": "synthetic_extra",
        }
        cit = AgentCitation.from_dict(d)
        assert cit.document_id == DOC_ID

    def test_from_dict_wrong_score_type_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation.from_dict({
                "document_id": DOC_ID, "filename": FILENAME,
                "chunk_id": "c0", "score": "not_a_float",
            })


# ===========================================================================
# 12. AGENTS — SearchRequest boundaries
# ===========================================================================

class TestSearchRequestBoundaries:
    def test_valid_minimal(self) -> None:
        req = SearchRequest(query=QUERY)
        assert req.query == QUERY
        assert req.top_k is None
        assert req.min_score is None

    def test_valid_with_all_optional_fields(self) -> None:
        req = SearchRequest(query=QUERY, top_k=10, min_score=0.5, max_results=5,
                            collection_name="omnibrain_v1", session_id="s1",
                            document_filter=["doc_a"], metadata={"run": "day33"})
        assert req.top_k == 10
        assert req.min_score == 0.5

    def test_empty_query_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="")

    def test_whitespace_query_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="   ")

    def test_invalid_top_k_zero_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query=QUERY, top_k=0)

    def test_invalid_top_k_negative_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query=QUERY, top_k=-5)

    def test_min_score_too_high_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query=QUERY, min_score=1.5)

    def test_min_score_too_low_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query=QUERY, min_score=-2.0)

    def test_max_results_zero_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query=QUERY, max_results=0)

    def test_empty_collection_name_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query=QUERY, collection_name="")

    def test_to_dict_structure(self) -> None:
        req = SearchRequest(query=QUERY, top_k=5, min_score=0.5)
        d = req.to_dict()
        assert d["query"] == QUERY
        assert d["top_k"] == 5


# ===========================================================================
# 13. AGENTS — AgentRequest boundaries
# ===========================================================================

class TestAgentRequestBoundaries:
    def test_valid_minimal(self) -> None:
        req = AgentRequest(query=QUERY)
        assert req.query == QUERY
        assert req.session_id is None
        assert req.metadata == {}

    def test_empty_query_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentRequest(query="")

    def test_none_session_id_accepted(self) -> None:
        req = AgentRequest(query=QUERY, session_id=None)
        assert req.session_id is None

    def test_empty_session_id_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentRequest(query=QUERY, session_id="")

    def test_document_filter_list_accepted(self) -> None:
        req = AgentRequest(query=QUERY, document_filter=["doc_a", "doc_b"])
        assert req.document_filter == ["doc_a", "doc_b"]

    def test_document_filter_dict_accepted(self) -> None:
        req = AgentRequest(query=QUERY, document_filter={"field": "value"})
        assert isinstance(req.document_filter, dict)

    def test_round_trip(self) -> None:
        req = AgentRequest(query=QUERY, metadata={"x": 1})
        req2 = AgentRequest.from_dict(req.to_dict())
        assert req2.query == req.query
        assert req2.metadata == req.metadata


# ===========================================================================
# 14. AGENTS — AgentResponse boundaries
# ===========================================================================

class TestAgentResponseBoundaries:
    def _make_resp(self, answer: str = "Day 33 answer") -> AgentResponse:
        return AgentResponse(
            answer=answer, agent_name="TestAgent", status="success",
            citations=[_make_citation()], metadata={"query": QUERY},
        )

    def test_valid_success_response(self) -> None:
        resp = self._make_resp()
        assert resp.is_success is True
        assert resp.error is None
        assert resp.has_citations is True

    def test_empty_citations_accepted(self) -> None:
        resp = AgentResponse(answer="answer", agent_name="Agent", citations=[])
        assert resp.has_citations is False

    def test_invalid_answer_type_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentResponse(answer=123, agent_name="Agent")

    def test_empty_agent_name_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentResponse(answer="answer", agent_name="")

    def test_wrong_citation_type_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentResponse(answer="answer", agent_name="Agent", citations=["bad"])

    def test_round_trip_serialization(self) -> None:
        resp = self._make_resp()
        d = resp.to_dict()
        resp2 = AgentResponse.from_dict(d)
        assert resp2.answer == resp.answer
        assert resp2.to_dict() == d

    def test_from_dict_non_dict_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentResponse.from_dict("not a dict")

    def test_image_results_property(self) -> None:
        images = self._make_resp().image_results
        assert all(c.content_type == "image" for c in images)

    def test_unique_documents_property_sorted(self) -> None:
        docs = self._make_resp().unique_documents
        assert docs == sorted(docs)


# ===========================================================================
# 15. AGENTS — SearchResult boundaries
# ===========================================================================

class TestSearchResultBoundaries:
    def _make_agent_resp(self) -> AgentResponse:
        return AgentResponse(
            answer="Search answer", agent_name="SearchAgent", status="success",
            citations=[_make_citation()], metadata={"query": QUERY},
        )

    def test_from_response_valid(self) -> None:
        sr = SearchResult.from_response(self._make_agent_resp())
        assert isinstance(sr, SearchResult)
        assert sr.query == QUERY
        assert sr.status == "RESULTS_FOUND"

    def test_from_response_empty_citations_no_results(self) -> None:
        resp = AgentResponse(answer="No", agent_name="A", citations=[], metadata={"query": QUERY})
        sr = SearchResult.from_response(resp)
        assert sr.status == "NO_RESULTS"

    def test_from_response_missing_query_raises(self) -> None:
        resp = AgentResponse(answer="a", agent_name="A", citations=[], metadata={})
        with pytest.raises(AgentValidationError):
            SearchResult.from_response(resp)

    def test_from_response_wrong_type_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchResult.from_response("not a response")

    def test_round_trip_serialization(self) -> None:
        sr = SearchResult.from_response(self._make_agent_resp())
        d = sr.to_dict()
        sr2 = SearchResult.from_dict(d)
        assert sr2.query == sr.query
        assert sr2.status == sr.status

    def test_from_dict_non_dict_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchResult.from_dict(42)


# ===========================================================================
# 16. AGENTS — AgentState boundaries
# ===========================================================================

class TestAgentStateBoundaries:
    def test_valid_state_minimal(self) -> None:
        state = AgentState(query=QUERY)
        assert state.query == QUERY
        assert state.status == "initialized"
        assert state.route is None
        assert state.errors == []

    def test_add_error_valid(self) -> None:
        state = AgentState(query=QUERY)
        state.add_error("Something went wrong")
        assert len(state.errors) == 1

    def test_add_error_empty_raises(self) -> None:
        state = AgentState(query=QUERY)
        with pytest.raises(AgentValidationError):
            state.add_error("")

    def test_add_citation_valid(self) -> None:
        state = AgentState(query=QUERY)
        state.add_citation(_make_citation())
        assert len(state.citations) == 1

    def test_add_citation_wrong_type_raises(self) -> None:
        state = AgentState(query=QUERY)
        with pytest.raises(AgentValidationError):
            state.add_citation("not a citation")

    def test_update_valid_field(self) -> None:
        state = AgentState(query=QUERY)
        state.update(status="running")
        assert state.status == "running"

    def test_update_unknown_field_raises(self) -> None:
        state = AgentState(query=QUERY)
        with pytest.raises(AgentValidationError):
            state.update(nonexistent_field="value")

    def test_empty_query_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentState(query="")

    def test_none_route_accepted(self) -> None:
        state = AgentState(query=QUERY, route=None)
        assert state.route is None

    def test_non_string_route_raises(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentState(query=QUERY, route=42)

    def test_to_dict_contains_expected_keys(self) -> None:
        d = AgentState(query=QUERY).to_dict()
        assert "query" in d and "status" in d and "citations" in d


# ===========================================================================
# 17. VISION — VisualEvidence valid inputs
# ===========================================================================

class TestVisualEvidenceValidInputs:
    def test_valid_evidence_minimal(self) -> None:
        ev = VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="chk_ev")
        assert ev.document_id == DOC_ID
        assert ev.content_type == "image"
        assert ev.page_number is None
        assert ev.chunk_index == 0

    def test_valid_evidence_chart_type(self) -> None:
        ev = VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", content_type="chart")
        assert ev.content_type == "chart"

    def test_valid_evidence_diagram_type(self) -> None:
        ev = VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", content_type="diagram")
        assert ev.content_type == "diagram"

    def test_none_page_number_accepted(self) -> None:
        ev = VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", page_number=None)
        assert ev.page_number is None

    def test_valid_content_types_set(self) -> None:
        assert "image" in VALID_VISUAL_CONTENT_TYPES
        assert "chart" in VALID_VISUAL_CONTENT_TYPES
        assert "diagram" in VALID_VISUAL_CONTENT_TYPES

    def test_round_trip_serialization(self) -> None:
        ev = VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="c0",
                            page_number=3, chunk_index=2, content_type="chart",
                            metadata={"src": "unit"})
        d = ev.to_dict()
        ev2 = VisualEvidence.from_dict(d)
        assert ev2.document_id == ev.document_id
        assert ev2.to_dict() == d


# ===========================================================================
# 18. VISION — VisualEvidence invalid inputs
# ===========================================================================

class TestVisualEvidenceInvalidInputs:
    def test_empty_document_id_raises(self) -> None:
        with pytest.raises(VisionEvidenceError) as exc_info:
            VisualEvidence(document_id="", filename=FILENAME, chunk_id="c0")
        assert "document_id" in str(exc_info.value).lower()

    def test_empty_filename_raises(self) -> None:
        with pytest.raises(VisionEvidenceError) as exc_info:
            VisualEvidence(document_id=DOC_ID, filename="", chunk_id="c0")
        assert "filename" in str(exc_info.value).lower()

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(VisionEvidenceError) as exc_info:
            VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="")
        assert "chunk_id" in str(exc_info.value).lower()

    def test_invalid_content_type_text_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", content_type="text")

    def test_page_number_zero_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", page_number=0)

    def test_negative_chunk_index_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="c0", chunk_index=-1)

    def test_exception_is_vision_agent_error(self) -> None:
        with pytest.raises(VisionAgentError):
            VisualEvidence(document_id="", filename=FILENAME, chunk_id="c0")


# ===========================================================================
# 19. VISION — VisionRequest boundaries
# ===========================================================================

class TestVisionRequestBoundaries:
    def _make_ev(self) -> VisualEvidence:
        return VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="chk_ev")

    def test_valid_request_with_evidence(self) -> None:
        req = VisionRequest(query=QUERY, evidence=[self._make_ev()])
        assert req.query == QUERY
        assert req.has_evidence is True

    def test_valid_request_empty_evidence(self) -> None:
        req = VisionRequest(query=QUERY, evidence=[])
        assert req.has_evidence is False

    def test_empty_query_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="")

    def test_whitespace_query_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="   ")

    def test_invalid_evidence_wrong_type_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query=QUERY, evidence=["not_evidence"])

    def test_round_trip_serialization(self) -> None:
        req = VisionRequest(query=QUERY, evidence=[self._make_ev()])
        d = req.to_dict()
        req2 = VisionRequest.from_dict(d)
        assert req2.query == req.query
        assert len(req2.evidence) == len(req.evidence)

    def test_none_session_id_accepted(self) -> None:
        req = VisionRequest(query=QUERY, session_id=None)
        assert req.session_id is None

    def test_whitespace_query_stripped(self) -> None:
        req = VisionRequest(query="  valid query  ")
        assert req.query == "valid query"


# ===========================================================================
# 20. VISION — VisionResult boundaries
# ===========================================================================

class TestVisionResultBoundaries:
    def _make_ev(self) -> VisualEvidence:
        return VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="chk_ev")

    def test_valid_result_with_evidence(self) -> None:
        res = VisionResult(query=QUERY, status="success", description="Visual analysis", evidence=[self._make_ev()])
        assert res.is_success is True
        assert res.has_evidence is True
        assert res.document_id == DOC_ID

    def test_valid_result_no_evidence(self) -> None:
        res = VisionResult(query=QUERY, status="no_evidence", evidence=[])
        assert res.has_evidence is False

    def test_empty_query_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionResult(query="", status="success")

    def test_non_string_description_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionResult(query=QUERY, status="success", description=42)

    def test_round_trip_serialization(self) -> None:
        res = VisionResult(query=QUERY, status="success", description="Analysis", evidence=[self._make_ev()])
        d = res.to_dict()
        res2 = VisionResult.from_dict(d)
        assert res2.query == res.query
        assert res2.to_dict() == d

    def test_from_dict_non_dict_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionResult.from_dict("not a dict")

    def test_lineage_inherited_from_first_evidence(self) -> None:
        res = VisionResult(query=QUERY, evidence=[self._make_ev()])
        assert res.document_id == DOC_ID
        assert res.filename == FILENAME


# ===========================================================================
# 21. VISION — VisualEvidenceAdapter boundaries
# ===========================================================================

class TestVisualEvidenceAdapterBoundaries:
    def test_adapt_citation_valid(self) -> None:
        ev = VisualEvidenceAdapter.adapt_citation(_make_citation(content_type="image"))
        assert isinstance(ev, VisualEvidence)
        assert ev.document_id == DOC_ID

    def test_adapt_citation_none_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisualEvidenceAdapter.adapt_citation(None)

    def test_adapt_citation_text_type_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidenceAdapter.adapt_citation(_make_citation(content_type="text"))

    def test_adapt_search_result_valid(self) -> None:
        ev = VisualEvidenceAdapter.adapt_search_result(_make_vsr(content_type="image"))
        assert isinstance(ev, VisualEvidence)

    def test_adapt_search_result_none_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisualEvidenceAdapter.adapt_search_result(None)

    def test_adapt_chunk_valid(self) -> None:
        ev = VisualEvidenceAdapter.adapt_chunk(_make_image_chunk())
        assert isinstance(ev, VisualEvidence)

    def test_adapt_chunk_none_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisualEvidenceAdapter.adapt_chunk(None)

    def test_adapt_chunk_text_type_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidenceAdapter.adapt_chunk(_make_chunk(content_type="text"))

    def test_adapt_batch_empty_list_returns_empty(self) -> None:
        assert VisualEvidenceAdapter.adapt_batch([]) == []

    def test_adapt_batch_non_list_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisualEvidenceAdapter.adapt_batch("not a list")

    def test_adapt_batch_filters_non_visual_by_default(self) -> None:
        result = VisualEvidenceAdapter.adapt_batch([
            _make_citation(content_type="image"),
            _make_citation(content_type="text", chunk_id="txt_chk"),
        ])
        assert len(result) == 1
        assert result[0].content_type == "image"

    def test_adapt_batch_strict_non_visual_raises(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidenceAdapter.adapt_batch([_make_citation(content_type="text")], strict=True)

    def test_adapt_search_package_agent_response(self) -> None:
        resp = AgentResponse(
            answer="test", agent_name="Agent",
            citations=[_make_citation(content_type="image")],
            metadata={"query": QUERY},
        )
        result = VisualEvidenceAdapter.adapt_search_package(resp)
        assert len(result) == 1

    def test_adapt_search_package_none_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisualEvidenceAdapter.adapt_search_package(None)

    def test_adapt_search_package_wrong_type_raises(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisualEvidenceAdapter.adapt_search_package("bad input")

    def test_is_visual_content_type_image_true(self) -> None:
        assert VisualEvidenceAdapter.is_visual_content_type("image") is True

    def test_is_visual_content_type_text_false(self) -> None:
        assert VisualEvidenceAdapter.is_visual_content_type("text") is False

    def test_is_visual_content_type_none_false(self) -> None:
        assert VisualEvidenceAdapter.is_visual_content_type(None) is False


# ===========================================================================
# 22. VISION — VisionExecutionTrace and VisionResultNormalizer
# ===========================================================================

class TestVisionExecutionTraceBoundaries:
    def test_empty_trace_has_no_stages(self) -> None:
        trace = VisionExecutionTrace()
        assert trace.stages == []

    def test_add_stage_valid(self) -> None:
        trace = VisionExecutionTrace()
        trace.add_stage("validation_started")
        assert "validation_started" in trace.stages

    def test_add_stage_empty_raises(self) -> None:
        trace = VisionExecutionTrace()
        with pytest.raises(VisionInputValidationError):
            trace.add_stage("")

    def test_add_stage_non_string_raises(self) -> None:
        trace = VisionExecutionTrace()
        with pytest.raises(VisionInputValidationError):
            trace.add_stage(42)

    def test_create_default_has_stages(self) -> None:
        trace = VisionExecutionTrace.create_default()
        assert len(trace.stages) == len(VisionExecutionTrace.DEFAULT_STAGES)

    def test_to_dict_structure(self) -> None:
        trace = VisionExecutionTrace()
        trace.add_stage("step_one")
        d = trace.to_dict()
        assert "stages" in d and "stage_count" in d
        assert d["stage_count"] == 1

    def test_stages_property_returns_copy(self) -> None:
        trace = VisionExecutionTrace()
        trace.add_stage("step_a")
        s1 = trace.stages
        s1.append("tampered")
        assert "tampered" not in trace.stages


class TestVisionResultNormalizerBoundaries:
    def _make_result(self) -> VisionResult:
        ev = VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="chk_n")
        return VisionResult(query=QUERY, status="success", description="Normalized", evidence=[ev])

    def test_normalize_valid_vision_result(self) -> None:
        normalized = VisionResultNormalizer.normalize(self._make_result())
        assert isinstance(normalized, VisionResult)
        assert normalized.status == "success"

    def test_normalize_none_raises(self) -> None:
        with pytest.raises(VisionProcessingError):
            VisionResultNormalizer.normalize(None)

    def test_normalize_wrong_type_raises(self) -> None:
        with pytest.raises(VisionProcessingError):
            VisionResultNormalizer.normalize("not a result")

    def test_normalize_dict_input_accepted(self) -> None:
        normalized = VisionResultNormalizer.normalize(self._make_result().to_dict())
        assert isinstance(normalized, VisionResult)

    def test_sanitize_metadata_strips_forbidden_keys(self) -> None:
        raw = {"query": QUERY, "api_key": "stripped", "secret": "stripped", "safe_key": "kept"}
        clean = VisionResultNormalizer.sanitize_metadata(raw)
        assert "api_key" not in clean
        assert "secret" not in clean
        assert "safe_key" in clean

    def test_sanitize_metadata_non_dict_returns_empty(self) -> None:
        assert VisionResultNormalizer.sanitize_metadata("not a dict") == {}

    def test_normalize_with_trace_attaches_trace(self) -> None:
        trace = VisionExecutionTrace()
        trace.add_stage("provider_started")
        normalized = VisionResultNormalizer.normalize(self._make_result(), trace=trace)
        assert "execution_trace" in normalized.metadata


# ===========================================================================
# 23. CROSS-COMPONENT COMPATIBILITY
# ===========================================================================

class TestCrossComponentCompatibility:
    def test_vsr_to_agent_citation(self) -> None:
        vsr = _make_vsr(content_type="image")
        cit = AgentCitation.from_search_result(vsr)
        assert cit.document_id == vsr.document_id
        assert cit.chunk_id == vsr.chunk_id

    def test_agent_citation_to_visual_evidence(self) -> None:
        cit = _make_citation(content_type="image")
        ev = VisualEvidenceAdapter.adapt_citation(cit)
        assert ev.document_id == cit.document_id

    def test_full_pipeline_ingestion_to_vision(self) -> None:
        chunk = _make_image_chunk()
        vsr = _make_vsr(chunk_id=chunk.chunk_id, content_type=chunk.content_type, content=chunk.content)
        processed = process_retrieval_results([vsr], min_score=0.0, max_results=5)
        assert len(processed) == 1
        cit = AgentCitation.from_search_result(processed[0])
        resp = AgentResponse(
            answer="Cross-component answer", agent_name="Pipeline",
            citations=[cit], metadata={"query": QUERY},
        )
        evidence = VisualEvidenceAdapter.adapt_batch(resp.image_results)
        assert len(evidence) == 1
        vision_result = VisionResult(
            query=QUERY, status="success", description="E2E", evidence=evidence,
        )
        assert vision_result.is_success is True
        assert vision_result.document_id == DOC_ID

    def test_text_citation_rejected_by_vision_strict_adapter(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidenceAdapter.adapt_citation(_make_citation(content_type="text"))

    def test_search_result_to_vision_adapter(self) -> None:
        resp = AgentResponse(
            answer="test", agent_name="SearchAgent",
            citations=[_make_citation(content_type="image")],
            metadata={"query": QUERY},
        )
        sr = SearchResult.from_response(resp)
        evidence = VisualEvidenceAdapter.adapt_search_package(sr)
        assert len(evidence) == 1


# ===========================================================================
# 24. ERROR CONTRACT
# ===========================================================================

class TestErrorContract:
    def test_agent_validation_error_is_agent_error(self) -> None:
        with pytest.raises(AgentError):
            AgentCitation(document_id="", filename=FILENAME, chunk_id="c0")

    def test_vision_evidence_error_is_vision_agent_error(self) -> None:
        with pytest.raises(VisionAgentError):
            VisualEvidence(document_id="", filename=FILENAME, chunk_id="c0")

    def test_vision_input_validation_error_is_vision_agent_error(self) -> None:
        with pytest.raises(VisionAgentError):
            VisionRequest(query="")

    def test_vision_processing_error_is_vision_agent_error(self) -> None:
        with pytest.raises(VisionAgentError):
            VisionResultNormalizer.normalize(None)

    def test_ingestion_validation_error_hierarchy(self) -> None:
        err = IngestionValidationError("test")
        assert isinstance(err, IngestionError)
        assert isinstance(err, ValueError)

    def test_ingestion_chunking_error_hierarchy(self) -> None:
        err = IngestionChunkingError("test")
        assert isinstance(err, IngestionError)
        assert isinstance(err, ValueError)

    def test_ingestion_embedding_error_hierarchy(self) -> None:
        err = IngestionEmbeddingError("test")
        assert isinstance(err, IngestionError)
        assert isinstance(err, ValueError)

    def test_agent_routing_error_is_agent_error(self) -> None:
        assert isinstance(AgentRoutingError("r"), AgentError)

    def test_agent_execution_error_is_agent_error(self) -> None:
        assert isinstance(AgentExecutionError("e"), AgentError)

    def test_vision_evidence_error_has_message_attribute(self) -> None:
        with pytest.raises(VisionEvidenceError) as exc_info:
            VisualEvidence(document_id="", filename=FILENAME, chunk_id="c0")
        assert hasattr(exc_info.value, "message")


# ===========================================================================
# 25. INPUT MUTATION SAFETY
# ===========================================================================

class TestInputMutationSafety:
    def test_validate_chunks_does_not_mutate_input_list(self) -> None:
        chunks = [_make_chunk()]
        original_len = len(chunks)
        validate_chunks(chunks)
        assert len(chunks) == original_len

    def test_normalize_chunks_does_not_mutate_caller_metadata(self) -> None:
        meta = {"classification": "public", "user": "Alice"}
        meta_snapshot = copy.deepcopy(meta)
        normalize_chunks([_make_chunk(metadata=meta)])
        assert meta == meta_snapshot

    def test_process_retrieval_results_does_not_mutate_input(self) -> None:
        vsrs = [_make_vsr()]
        original_count = len(vsrs)
        process_retrieval_results(vsrs, min_score=0.0, max_results=5)
        assert len(vsrs) == original_count

    def test_agent_response_metadata_not_mutated(self) -> None:
        meta = {"query": QUERY, "extra": "data"}
        meta_snapshot = copy.deepcopy(meta)
        AgentResponse(answer="answer", agent_name="Agent",
                      citations=[_make_citation()], metadata=meta)
        assert meta == meta_snapshot

    def test_vision_result_evidence_list_not_mutated(self) -> None:
        ev = VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="c0")
        evidence_list = [ev]
        VisionResult(query=QUERY, evidence=evidence_list)
        assert len(evidence_list) == 1


# ===========================================================================
# 26. INVALID TO VALID RECOVERY
# ===========================================================================

class TestInvalidToValidRecovery:
    def test_agent_citation_invalid_then_valid(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename=FILENAME, chunk_id="c0")
        cit = AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="c0")
        assert cit.document_id == DOC_ID

    def test_vision_evidence_invalid_then_valid(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename=FILENAME, chunk_id="c0")
        ev = VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="c0")
        assert ev.document_id == DOC_ID

    def test_chunk_validation_invalid_then_valid(self) -> None:
        bad = DocumentChunk(chunk_id="", chunk_index=0, document_id=DOC_ID,
                            filename=FILENAME, page_number=1, content="content", content_type="text")
        r1 = validate_chunks([bad])
        assert r1.is_valid is False
        r2 = validate_chunks([_make_chunk()])
        assert r2.is_valid is True


# ===========================================================================
# 27. VALID - INVALID - VALID SEQUENCE
# ===========================================================================

class TestValidInvalidValidSequence:
    def test_citation_valid_invalid_valid(self) -> None:
        c1 = AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="c1")
        assert c1.document_id == DOC_ID
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename=FILENAME, chunk_id="c2")
        c3 = AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="c3")
        assert c3.chunk_id == "c3"

    def test_vision_request_valid_invalid_valid(self) -> None:
        r1 = VisionRequest(query=QUERY)
        assert r1.query == QUERY
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="")
        r3 = VisionRequest(query="Another valid query")
        assert r3.query == "Another valid query"


# ===========================================================================
# 28. REPEATED INVALID INPUT
# ===========================================================================

class TestRepeatedInvalidInput:
    def test_repeated_empty_document_id_always_raises(self) -> None:
        for _ in range(5):
            with pytest.raises(AgentValidationError):
                AgentCitation(document_id="", filename=FILENAME, chunk_id="c0")

    def test_repeated_invalid_vision_evidence_always_raises(self) -> None:
        for _ in range(5):
            with pytest.raises(VisionEvidenceError):
                VisualEvidence(document_id="", filename=FILENAME, chunk_id="c0")

    def test_repeated_invalid_retrieval_always_raises(self) -> None:
        vsrs = [_make_vsr()]
        for _ in range(5):
            with pytest.raises(ValueError):
                process_retrieval_results(vsrs, min_score=99.0, max_results=5)


# ===========================================================================
# 29. CONCURRENT ISOLATION
# ===========================================================================

class TestConcurrentIsolation:
    def test_concurrent_valid_invalid_agent_citation(self) -> None:
        def valid_task(_: int) -> str:
            return AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="c0").document_id

        def invalid_task(_: int) -> str:
            try:
                AgentCitation(document_id="", filename=FILENAME, chunk_id="c0")
                return "no_error"
            except AgentValidationError as e:
                return f"error:{e}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            valid_results = [f.result() for f in [executor.submit(valid_task, i) for i in range(4)]]
            invalid_results = [f.result() for f in [executor.submit(invalid_task, i) for i in range(4)]]

        assert all(r == DOC_ID for r in valid_results)
        assert all(r.startswith("error:") for r in invalid_results)

    def test_concurrent_vision_evidence_valid_invalid(self) -> None:
        def create_valid(_: int) -> str:
            return VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="c0").document_id

        def create_invalid(_: int) -> str:
            try:
                VisualEvidence(document_id="", filename=FILENAME, chunk_id="c0")
                return "no_error"
            except VisionEvidenceError as e:
                return f"error:{e}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            valid_results = [f.result() for f in [executor.submit(create_valid, i) for i in range(4)]]
            invalid_results = [f.result() for f in [executor.submit(create_invalid, i) for i in range(4)]]

        assert all(r == DOC_ID for r in valid_results)
        assert all(r.startswith("error:") for r in invalid_results)


# ===========================================================================
# 30. STRING BOUNDARY TESTING
# ===========================================================================

class TestStringBoundaries:
    def test_short_query_accepted(self) -> None:
        req = SearchRequest(query="Q")
        assert req.query == "Q"

    def test_unicode_query_accepted(self) -> None:
        req = SearchRequest(query="Ünïcödé qüéry für Tëst")
        assert "Ünïcödé" in req.query

    def test_long_query_accepted(self) -> None:
        req = SearchRequest(query=("word " * 100).strip())
        assert len(req.query) > 200

    def test_single_char_document_id_accepted(self) -> None:
        cit = AgentCitation(document_id="X", filename=FILENAME, chunk_id="c0")
        assert cit.document_id == "X"

    def test_whitespace_chunk_id_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="   ")
