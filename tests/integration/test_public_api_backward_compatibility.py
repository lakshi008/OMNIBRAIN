"""
OmniBrain Member 4 -- Day 10 Public API Contract and Backward Compatibility Tests.
14 concern sections. 100% offline. No LLM. No network. No production code modified.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

from ingestion.models import (
    ChunkValidationResult, ChunkingResult, DocumentChunk, DocumentMetadata,
    EmbeddingGenerationResult, EmbeddingPreparationResult, EmbeddingRecord,
    EmbeddingVectorRecord, RetrievalServiceResult, VectorSearchResult, PageData,
)
from ingestion.ingestion_errors import (
    IngestionChunkingError, IngestionEmbeddingError, IngestionError,
    IngestionExtractionError, IngestionPipelineError, IngestionValidationError,
)
from ingestion.exceptions import CorruptedPDFError, InvalidFileTypeError, PDFNotFoundError
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.retrieval_processor import build_retrieval_context, process_retrieval_results
from agents.exceptions import AgentError, AgentExecutionError, AgentRoutingError, AgentValidationError
from agents.models import AgentCitation, AgentRequest, AgentResponse, AgentState, SearchRequest, SearchResult
from vision.exceptions import (
    VisionAgentError, VisionCancellationError, VisionError, VisionEvidenceError,
    VisionInputValidationError, VisionProcessingError, VisionProviderError, VisionTimeoutError,
)
from vision.models import VALID_VISUAL_CONTENT_TYPES, VisionRequest, VisionResult, VisualEvidence
from vision.evidence_adapter import VisualEvidenceAdapter


def _make_vsr(**kw):
    d = dict(chunk_id="ck-001", score=0.9, document_id="doc-001", filename="report.pdf",
             page_number=1, chunk_index=0, content_type="text",
             content="Sample content.", metadata={"source": "test"})
    d.update(kw)
    return VectorSearchResult(**d)


def _make_chunk(**kw):
    d = dict(chunk_id="ck-001", chunk_index=0, document_id="doc-001", filename="report.pdf",
             page_number=1, content="Sample chunk.", content_type="text", metadata={})
    d.update(kw)
    return DocumentChunk(**d)


def _make_citation(**kw):
    d = dict(document_id="doc-001", filename="report.pdf", chunk_id="ck-001",
             page_number=1, content_type="text", score=0.9, metadata={})
    d.update(kw)
    return AgentCitation(**d)


def _make_evidence(**kw):
    d = dict(document_id="doc-vis-001", filename="diagram.pdf", chunk_id="ck-vis-001",
             page_number=3, chunk_index=0, content_type="image", metadata={})
    d.update(kw)
    return VisualEvidence(**d)


# ---- SECTION 1: PUBLIC FUNCTION SIGNATURES ----

class TestPublicFunctionSignatures:
    def test_validate_chunks_accepts_list(self):
        assert isinstance(validate_chunks([_make_chunk()]), ChunkValidationResult)

    def test_validate_chunks_accepts_chunking_result(self):
        cr = ChunkingResult(document_id="d", filename="f.pdf", chunks=[_make_chunk()])
        assert isinstance(validate_chunks(cr), ChunkValidationResult)

    def test_normalize_chunks_signature(self):
        result = normalize_chunks([_make_chunk()])
        assert isinstance(result, list) and all(isinstance(c, DocumentChunk) for c in result)

    def test_prepare_for_embedding_accepts_list(self):
        assert isinstance(prepare_for_embedding([_make_chunk()]), EmbeddingPreparationResult)

    def test_prepare_for_embedding_accepts_chunking_result(self):
        cr = ChunkingResult(document_id="d", filename="f.pdf", chunks=[_make_chunk()])
        assert isinstance(prepare_for_embedding(cr), EmbeddingPreparationResult)

    def test_process_retrieval_results_defaults(self):
        assert isinstance(process_retrieval_results([_make_vsr()]), list)

    def test_process_retrieval_results_explicit_params(self):
        result = process_retrieval_results(results=[_make_vsr(score=0.8)], min_score=0.5, max_results=10)
        assert len(result) == 1

    def test_build_retrieval_context_returns_str(self):
        assert isinstance(build_retrieval_context([_make_vsr()]), str)

    def test_build_retrieval_context_empty_list(self):
        assert build_retrieval_context([]) == ""

    def test_agent_citation_from_search_result(self):
        assert isinstance(AgentCitation.from_search_result(_make_vsr()), AgentCitation)

    def test_search_request_to_agent_request(self):
        assert isinstance(SearchRequest(query="q").to_agent_request(), AgentRequest)

    def test_search_request_from_agent_request(self):
        assert isinstance(SearchRequest.from_agent_request(AgentRequest(query="q")), SearchRequest)

    def test_search_result_from_response(self):
        resp = AgentResponse(answer="A", agent_name="S", citations=[_make_citation()],
                              metadata={"query": "q", "context": ""})
        assert isinstance(SearchResult.from_response(resp), SearchResult)

    def test_visual_evidence_adapter_is_visual(self):
        assert VisualEvidenceAdapter.is_visual(_make_evidence()) is True

    def test_visual_evidence_adapter_is_visual_content_type(self):
        assert VisualEvidenceAdapter.is_visual_content_type("image") is True
        assert VisualEvidenceAdapter.is_visual_content_type("chart") is True
        assert VisualEvidenceAdapter.is_visual_content_type("diagram") is True
        assert VisualEvidenceAdapter.is_visual_content_type("text") is False

    def test_visual_evidence_from_citation(self):
        assert isinstance(VisualEvidence.from_citation(_make_citation(content_type="image")), VisualEvidence)

    def test_visual_evidence_from_search_result(self):
        assert isinstance(VisualEvidence.from_search_result(_make_vsr(content_type="image")), VisualEvidence)

    def test_agent_state_add_error(self):
        s = AgentState(query="q")
        s.add_error("err")
        assert "err" in s.errors

    def test_agent_state_add_citation(self):
        s = AgentState(query="q")
        c = _make_citation()
        s.add_citation(c)
        assert c in s.citations


# ---- SECTION 2: INGESTION REQUEST-MODEL COMPATIBILITY ----

class TestIngestionRequestModelCompatibility:
    def test_page_data_required_fields(self):
        pd = PageData(page_number=1, text="Hello.", char_count=6, has_content=True)
        assert pd.page_number == 1 and pd.text == "Hello." and pd.char_count == 6

    def test_document_metadata_required_fields(self):
        dm = DocumentMetadata(document_id="d", filename="f.pdf", total_pages=5,
                               content_type="application/pdf", created_at="2026-01-01T00:00:00Z",
                               pages_with_content=4, pages_without_content=1)
        assert dm.document_id == "d" and dm.total_pages == 5 and dm.pages_with_content == 4

    def test_document_chunk_optional_metadata_defaults(self):
        chunk = DocumentChunk(chunk_id="c", chunk_index=0, document_id="d", filename="f.pdf",
                               page_number=1, content="t", content_type="text")
        assert chunk.metadata == {}

    def test_document_chunk_optional_page_number_none(self):
        chunk = DocumentChunk(chunk_id="c", chunk_index=0, document_id="d", filename="f.pdf",
                               page_number=None, content="t", content_type="text")
        assert chunk.page_number is None

    def test_vector_search_result_fields(self):
        vsr = _make_vsr()
        for attr in ("chunk_id", "score", "document_id", "filename", "page_number",
                     "chunk_index", "content_type", "content", "metadata"):
            assert hasattr(vsr, attr), f"Missing field: {attr}"

    def test_retrieval_service_result_fields(self):
        rsr = RetrievalServiceResult(query_vector_dimension=4, results=[_make_vsr()], context="ctx")
        assert rsr.query_vector_dimension == 4 and len(rsr.results) == 1

    def test_retrieval_service_result_properties(self):
        rsr = RetrievalServiceResult(query_vector_dimension=4, results=[_make_vsr(content_type="text")], context="")
        assert rsr.total_results == 1 and rsr.has_results is True and rsr.text_results == 1

    def test_chunking_result_properties(self):
        cr = ChunkingResult(document_id="d", filename="f.pdf", chunks=[_make_chunk(content_type="text")])
        assert cr.total_chunks == 1 and cr.text_chunks == 1 and cr.has_chunks is True

    def test_embedding_preparation_result_properties(self):
        rec = EmbeddingRecord(chunk_id="c", document_id="d", filename="f.pdf",
                               chunk_index=0, page_number=1, content="t", content_type="text")
        epr = EmbeddingPreparationResult(document_id="d", filename="f.pdf", items=[rec], is_ready=True)
        assert epr.total_items == 1 and epr.text_items == 1 and epr.is_ready is True

    def test_embedding_generation_result_properties(self):
        rec = EmbeddingVectorRecord(chunk_id="c", document_id="d", filename="f.pdf",
                                     chunk_index=0, page_number=1, content_type="text",
                                     vector=[0.1, 0.2, 0.3, 0.4])
        egr = EmbeddingGenerationResult(document_id="d", filename="f.pdf", items=[rec], dimension=4, is_ready=True)
        assert egr.total_items == 1 and egr.dimension == 4 and egr.is_ready is True

    def test_chunk_validation_result_fields(self):
        result = validate_chunks([_make_chunk()])
        for attr in ("is_valid", "total_chunks", "valid_chunks", "invalid_chunks", "errors", "warnings"):
            assert hasattr(result, attr), f"Missing field: {attr}"


# ---- SECTION 3: SEARCH / AGENTS REQUEST-MODEL COMPATIBILITY ----

class TestSearchAgentsRequestModelCompatibility:
    def test_agent_citation_required_fields(self):
        c = AgentCitation(document_id="d", filename="f.pdf", chunk_id="ck")
        assert c.document_id == "d" and c.filename == "f.pdf" and c.chunk_id == "ck"

    def test_agent_citation_optional_defaults(self):
        c = AgentCitation(document_id="d", filename="f.pdf", chunk_id="ck")
        assert c.page_number is None and c.content_type == "text" and c.score == 0.0 and c.metadata == {}

    def test_agent_request_required_field_only(self):
        ar = AgentRequest(query="q")
        assert ar.query == "q" and ar.session_id is None and ar.document_filter is None

    def test_agent_request_all_fields(self):
        ar = AgentRequest(query="q", session_id="s", document_filter={"doc": "d"}, metadata={"k": "v"})
        assert ar.session_id == "s" and ar.document_filter == {"doc": "d"} and ar.metadata["k"] == "v"

    def test_search_request_query_only(self):
        sr = SearchRequest(query="q")
        assert sr.query == "q" and sr.top_k is None and sr.min_score is None

    def test_search_request_optional_params(self):
        sr = SearchRequest(query="q", top_k=10, min_score=0.5, max_results=5, collection_name="col")
        assert sr.top_k == 10 and sr.min_score == 0.5 and sr.max_results == 5

    def test_agent_response_required_fields(self):
        r = AgentResponse(answer="A", agent_name="S")
        assert r.answer == "A" and r.agent_name == "S" and r.status == "success" and r.error is None

    def test_agent_response_with_citations(self):
        r = AgentResponse(answer="A", agent_name="S", citations=[_make_citation()])
        assert len(r.citations) == 1 and r.has_citations is True

    def test_search_result_required_field(self):
        sr = SearchResult(query="q")
        assert sr.query == "q" and sr.status == "NO_RESULTS" and sr.citations == [] and sr.context == ""

    def test_agent_state_default_fields(self):
        s = AgentState(query="q")
        assert s.query == "q" and s.route is None and s.status == "initialized" and s.errors == []


# ---- SECTION 4: VISION REQUEST-MODEL COMPATIBILITY ----

class TestVisionRequestModelCompatibility:
    def test_visual_evidence_required_fields(self):
        ev = VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck")
        assert ev.document_id == "d" and ev.filename == "f.pdf" and ev.chunk_id == "ck"

    def test_visual_evidence_optional_defaults(self):
        ev = VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck")
        assert ev.page_number is None and ev.chunk_index == 0 and ev.content_type == "image"
        assert ev.image_path is None and ev.image_bytes is None and ev.image_format is None
        assert ev.width is None and ev.height is None and ev.description is None

    def test_visual_evidence_content_type_normalization(self):
        ev = VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck", content_type="IMAGE")
        assert ev.content_type == "image"

    def test_valid_visual_content_types_constant(self):
        assert isinstance(VALID_VISUAL_CONTENT_TYPES, frozenset)
        assert {"image", "chart", "diagram"}.issubset(VALID_VISUAL_CONTENT_TYPES)

    def test_vision_request_required_field(self):
        vr = VisionRequest(query="Describe.")
        assert vr.query == "Describe." and vr.evidence == [] and vr.session_id is None

    def test_vision_request_with_evidence(self):
        ev = _make_evidence()
        vr = VisionRequest(query="A", evidence=[ev])
        assert len(vr.evidence) == 1 and vr.has_evidence is True and vr.total_evidence == 1

    def test_vision_result_required_field(self):
        vr = VisionResult(query="q")
        assert vr.query == "q" and vr.status == "success" and vr.description == "" and vr.error is None

    def test_vision_result_with_evidence_inherits_lineage(self):
        ev = _make_evidence(document_id="doc-inh", filename="inh.pdf", chunk_id="ck-inh", page_number=7)
        vr = VisionResult(query="q", evidence=[ev])
        assert vr.document_id == "doc-inh" and vr.chunk_id == "ck-inh" and vr.page_number == 7


# ---- SECTION 5: RESULT-MODEL FIELD COMPLETENESS ----

class TestResultModelFieldCompleteness:
    def test_vision_result_all_public_fields_present(self):
        ev = _make_evidence()
        vr = VisionResult(query="q", status="success", description="bar chart", evidence=[ev],
                           document_id="d", filename="f.pdf", page_number=2, chunk_id="ck",
                           content_type="image", metadata={"latency_ms": 42}, error=None)
        assert vr.status == "success" and vr.description == "bar chart" and vr.metadata["latency_ms"] == 42

    def test_agent_citation_all_public_fields_present(self):
        c = _make_citation(page_number=5, content_type="table", score=0.85, metadata={"tag": "fin"})
        assert c.page_number == 5 and c.content_type == "table" and c.score == 0.85

    def test_search_result_all_public_fields_present(self):
        sr = SearchResult(query="q", status="RESULTS_FOUND", citations=[_make_citation()],
                           context="ctx", metadata={"top_k": 10})
        assert sr.has_results is True and sr.total_results == 1 and sr.evidence_count == 1

    def test_agent_response_all_public_fields_present(self):
        r = AgentResponse(answer="a", agent_name="S", status="success",
                           citations=[_make_citation()], metadata={"latency": 120}, error=None)
        assert r.is_success is True and r.is_error is False and len(r.citations) == 1


# ---- SECTION 6: RETURN-TYPE COMPATIBILITY ----

class TestReturnTypeCompatibility:
    def test_validate_chunks_returns_chunk_validation_result(self):
        assert type(validate_chunks([_make_chunk()])).__name__ == "ChunkValidationResult"

    def test_normalize_chunks_returns_list(self):
        assert isinstance(normalize_chunks([_make_chunk()]), list)

    def test_prepare_for_embedding_returns_embedding_preparation_result(self):
        assert type(prepare_for_embedding([_make_chunk()])).__name__ == "EmbeddingPreparationResult"

    def test_process_retrieval_results_returns_list_of_vsr(self):
        result = process_retrieval_results([_make_vsr()])
        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], VectorSearchResult)

    def test_build_retrieval_context_returns_str(self):
        assert isinstance(build_retrieval_context([_make_vsr()]), str)

    def test_agent_citation_from_search_result_returns_agent_citation(self):
        assert isinstance(AgentCitation.from_search_result(_make_vsr()), AgentCitation)

    def test_search_result_from_response_returns_search_result(self):
        resp = AgentResponse(answer="A", agent_name="S", citations=[_make_citation()],
                              metadata={"query": "q?", "context": ""})
        assert isinstance(SearchResult.from_response(resp), SearchResult)

    def test_visual_evidence_from_citation_returns_visual_evidence(self):
        assert isinstance(VisualEvidence.from_citation(_make_citation(content_type="image")), VisualEvidence)

    def test_visual_evidence_from_search_result_returns_visual_evidence(self):
        assert isinstance(VisualEvidence.from_search_result(_make_vsr(content_type="image")), VisualEvidence)

    def test_to_dict_methods_return_dict(self):
        assert isinstance(_make_citation().to_dict(), dict)
        assert isinstance(AgentRequest(query="q").to_dict(), dict)
        assert isinstance(SearchRequest(query="q").to_dict(), dict)
        assert isinstance(VisionResult(query="q").to_dict(), dict)


# ---- SECTION 7: EXCEPTION CONTRACT STABILITY ----

class TestExceptionContractStability:
    def test_ingestion_error_is_base(self):
        exc = IngestionError(message="base")
        assert isinstance(exc, Exception) and exc.message == "base" and exc.stage == "PIPELINE"

    def test_ingestion_error_stage_attribute(self):
        assert IngestionError(message="t", stage="EXTRACTION").stage == "EXTRACTION"

    def test_ingestion_validation_error_hierarchy(self):
        exc = IngestionValidationError(message="v")
        assert isinstance(exc, IngestionError) and isinstance(exc, ValueError)

    def test_ingestion_extraction_error_hierarchy(self):
        assert isinstance(IngestionExtractionError(message="e"), IngestionError)

    def test_ingestion_chunking_error_hierarchy(self):
        exc = IngestionChunkingError(message="c")
        assert isinstance(exc, IngestionError) and isinstance(exc, ValueError)

    def test_ingestion_embedding_error_hierarchy(self):
        exc = IngestionEmbeddingError(message="emb")
        assert isinstance(exc, IngestionError) and isinstance(exc, ValueError)

    def test_ingestion_pipeline_error_hierarchy(self):
        assert isinstance(IngestionPipelineError(message="p"), IngestionError)

    def test_pdf_not_found_error(self):
        exc = PDFNotFoundError("/p/missing.pdf")
        assert isinstance(exc, IngestionExtractionError) and exc.filepath == "/p/missing.pdf"

    def test_invalid_file_type_error(self):
        exc = InvalidFileTypeError("/p/d.txt", ".txt")
        assert isinstance(exc, IngestionExtractionError) and exc.extension == ".txt"

    def test_corrupted_pdf_error(self):
        exc = CorruptedPDFError("/p/bad.pdf", reason="truncated")
        assert isinstance(exc, IngestionExtractionError) and exc.reason == "truncated"

    def test_agent_exception_hierarchy(self):
        assert isinstance(AgentError("b"), Exception)
        assert isinstance(AgentValidationError("v"), AgentError)
        assert isinstance(AgentRoutingError("r"), AgentError)
        assert isinstance(AgentExecutionError("e"), AgentError)

    def test_vision_error_alias(self):
        assert VisionError is VisionAgentError

    def test_vision_exception_hierarchy(self):
        assert isinstance(VisionAgentError(), Exception)
        assert isinstance(VisionInputValidationError("v"), VisionAgentError)
        assert isinstance(VisionEvidenceError("e"), VisionAgentError)
        assert isinstance(VisionProcessingError("p"), VisionAgentError)
        assert isinstance(VisionProviderError("pr"), VisionAgentError)
        assert isinstance(VisionCancellationError("c"), VisionAgentError)

    def test_vision_timeout_error_hierarchy(self):
        exc = VisionTimeoutError("timeout")
        assert isinstance(exc, VisionProviderError) and isinstance(exc, VisionProcessingError)

    def test_agent_citation_empty_doc_id_raises(self):
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="f.pdf", chunk_id="ck")

    def test_agent_request_whitespace_query_raises(self):
        with pytest.raises(AgentValidationError):
            AgentRequest(query="   ")

    def test_search_request_empty_query_raises(self):
        with pytest.raises(AgentValidationError):
            SearchRequest(query="")

    def test_search_request_invalid_top_k_raises(self):
        with pytest.raises(AgentValidationError):
            SearchRequest(query="q", top_k=0)

    def test_search_request_invalid_min_score_raises(self):
        with pytest.raises(AgentValidationError):
            SearchRequest(query="q", min_score=2.0)

    def test_vision_request_empty_query_raises(self):
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="")

    def test_visual_evidence_invalid_content_type_raises(self):
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck", content_type="text")

    def test_visual_evidence_empty_doc_id_raises(self):
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename="f.pdf", chunk_id="ck")

    def test_process_retrieval_results_non_list_raises_type_error(self):
        with pytest.raises(TypeError):
            process_retrieval_results("not a list")

    def test_process_retrieval_results_invalid_min_score_raises_value_error(self):
        with pytest.raises(ValueError):
            process_retrieval_results([_make_vsr()], min_score=5.0)

    def test_process_retrieval_results_zero_max_results_raises_value_error(self):
        with pytest.raises(ValueError):
            process_retrieval_results([_make_vsr()], max_results=0)

    def test_build_retrieval_context_none_raises_type_error(self):
        with pytest.raises(TypeError):
            build_retrieval_context(None)


# ---- SECTION 8: SERIALIZATION COMPATIBILITY ----

class TestSerializationCompatibility:
    def test_agent_citation_roundtrip(self):
        orig = _make_citation(document_id="d", filename="f.pdf", chunk_id="ck",
                               page_number=3, content_type="table", score=0.72,
                               metadata={"fiscal_year": 2024})
        r = AgentCitation.from_dict(orig.to_dict())
        assert r.document_id == orig.document_id and r.chunk_id == orig.chunk_id
        assert r.page_number == orig.page_number and r.score == orig.score
        assert r.metadata == orig.metadata

    def test_agent_citation_to_dict_keys(self):
        d = _make_citation().to_dict()
        for key in ("document_id", "filename", "chunk_id", "page_number", "content_type", "score", "metadata"):
            assert key in d, f"Missing key: {key}"

    def test_agent_request_roundtrip(self):
        orig = AgentRequest(query="q?", session_id="s", document_filter={"doc": "d"}, metadata={"user": "t"})
        r = AgentRequest.from_dict(orig.to_dict())
        assert r.query == orig.query and r.session_id == orig.session_id and r.metadata == orig.metadata

    def test_search_request_roundtrip(self):
        orig = SearchRequest(query="q", top_k=10, min_score=0.5, max_results=5,
                              collection_name="col", session_id="s", metadata={"p": "h"})
        r = SearchRequest.from_dict(orig.to_dict())
        assert r.query == orig.query and r.top_k == orig.top_k and r.collection_name == orig.collection_name

    def test_agent_response_roundtrip(self):
        orig = AgentResponse(answer="ans", agent_name="S", status="success",
                              citations=[_make_citation()], metadata={"lat": 100}, error=None)
        r = AgentResponse.from_dict(orig.to_dict())
        assert r.answer == orig.answer and len(r.citations) == 1 and r.error is None

    def test_search_result_roundtrip(self):
        orig = SearchResult(query="q", status="RESULTS_FOUND", citations=[_make_citation()],
                             context="ctx", metadata={"k": 5})
        r = SearchResult.from_dict(orig.to_dict())
        assert r.query == orig.query and r.status == orig.status and len(r.citations) == 1

    def test_search_result_to_dict_keys(self):
        d = SearchResult(query="q", status="NO_RESULTS").to_dict()
        for key in ("query", "status", "citations", "context", "total_results", "evidence_count",
                    "has_results", "text_count", "table_count", "image_count",
                    "unique_document_count", "unique_documents", "metadata"):
            assert key in d, f"Missing key: {key}"

    def test_visual_evidence_roundtrip(self):
        orig = _make_evidence(document_id="d", filename="f.pdf", chunk_id="ck",
                               page_number=4, chunk_index=2, content_type="chart",
                               metadata={"tool": "camelot"})
        r = VisualEvidence.from_dict(orig.to_dict())
        assert r.document_id == orig.document_id and r.chunk_id == orig.chunk_id
        assert r.page_number == orig.page_number and r.content_type == orig.content_type
        assert r.metadata == orig.metadata

    def test_visual_evidence_to_dict_keys(self):
        d = _make_evidence().to_dict()
        for key in ("document_id", "filename", "chunk_id", "page_number", "chunk_index",
                    "content_type", "image_path", "image_format", "width", "height",
                    "description", "metadata"):
            assert key in d, f"Missing key: {key}"

    def test_vision_request_roundtrip(self):
        orig = VisionRequest(query="desc", evidence=[_make_evidence()], session_id="s", metadata={"p": "h"})
        r = VisionRequest.from_dict(orig.to_dict())
        assert r.query == orig.query and len(r.evidence) == 1 and r.session_id == orig.session_id

    def test_vision_result_roundtrip(self):
        orig = VisionResult(query="q", status="success", description="bar chart",
                             evidence=[_make_evidence()], document_id="d", filename="f.pdf",
                             page_number=5, chunk_id="ck", content_type="chart",
                             metadata={"m": "t"}, error=None)
        r = VisionResult.from_dict(orig.to_dict())
        assert r.query == orig.query and r.description == orig.description
        assert r.document_id == orig.document_id and len(r.evidence) == 1 and r.error is None

    def test_vision_result_to_dict_keys(self):
        d = VisionResult(query="q").to_dict()
        for key in ("query", "status", "description", "evidence", "document_id", "filename",
                    "page_number", "chunk_id", "content_type", "metadata", "error"):
            assert key in d, f"Missing key: {key}"

    def test_serialization_preserves_citation_lineage(self):
        c = AgentCitation(document_id="lineage-doc", filename="lr.pdf", chunk_id="lk-42",
                           page_number=10, content_type="text", score=0.88,
                           metadata={"section": "intro"})
        r = AgentCitation.from_dict(c.to_dict())
        assert r.document_id == "lineage-doc" and r.chunk_id == "lk-42" and r.page_number == 10
        assert r.metadata["section"] == "intro"

    def test_serialization_preserves_error_field(self):
        ok = AgentResponse(answer="ok", agent_name="A", error=None)
        assert ok.to_dict()["error"] is None
        assert AgentResponse.from_dict(ok.to_dict()).error is None
        err = AgentResponse(answer="", agent_name="A", status="error", error="Failed.")
        assert err.to_dict()["error"] == "Failed."
        assert AgentResponse.from_dict(err.to_dict()).error == "Failed."


# ---- SECTION 9: CROSS-MEMBER COMPATIBILITY ----

class TestCrossMemberCompatibility:
    def test_vector_search_result_to_agent_citation(self):
        vsr = _make_vsr(chunk_id="ck-x", score=0.91, document_id="doc-x",
                        filename="x.pdf", page_number=2, content_type="text")
        c = AgentCitation.from_search_result(vsr)
        assert c.chunk_id == vsr.chunk_id and c.document_id == vsr.document_id
        assert c.page_number == vsr.page_number and c.score == vsr.score

    def test_agent_citation_to_visual_evidence(self):
        c = AgentCitation(document_id="d", filename="img.pdf", chunk_id="ck",
                           page_number=6, content_type="image", score=0.82,
                           metadata={"chunk_index": 3})
        ev = VisualEvidence.from_citation(c)
        assert ev.document_id == c.document_id and ev.chunk_id == c.chunk_id
        assert ev.page_number == c.page_number and ev.content_type == c.content_type

    def test_vector_search_result_to_visual_evidence(self):
        vsr = _make_vsr(chunk_id="ck-img", document_id="doc-img", filename="charts.pdf",
                        page_number=8, chunk_index=4, content_type="image")
        ev = VisualEvidence.from_search_result(vsr)
        assert ev.document_id == vsr.document_id and ev.chunk_id == vsr.chunk_id
        assert ev.page_number == vsr.page_number and ev.chunk_index == vsr.chunk_index

    def test_visual_evidence_to_vision_request_to_vision_result(self):
        ev = _make_evidence(document_id="e2e-doc", filename="e2e.pdf",
                             chunk_id="e2e-ck", page_number=1, content_type="chart")
        req = VisionRequest(query="Describe.", evidence=[ev])
        res = VisionResult(query=req.query, status="success", description="Stacked bar.", evidence=req.evidence)
        assert res.document_id == "e2e-doc" and res.chunk_id == "e2e-ck" and res.page_number == 1

    def test_ingestion_chunk_to_embedding_record_preserves_lineage(self):
        chunk = DocumentChunk(chunk_id="ck-emb", chunk_index=0, document_id="doc-emb",
                               filename="emb.pdf", page_number=2,
                               content="Embedding test.", content_type="text")
        prep = prepare_for_embedding([chunk])
        assert prep.is_ready is True and len(prep.items) == 1
        rec = prep.items[0]
        assert rec.chunk_id == chunk.chunk_id and rec.document_id == chunk.document_id
        assert rec.page_number == chunk.page_number

    def test_adapter_is_visual_rejects_text(self):
        assert VisualEvidenceAdapter.is_visual(_make_vsr(content_type="text")) is False

    def test_adapter_is_visual_accepts_image(self):
        assert VisualEvidenceAdapter.is_visual(_make_vsr(content_type="image")) is True

    def test_retrieval_service_result_feeds_process_retrieval_results(self):
        rsr = RetrievalServiceResult(query_vector_dimension=4, results=[_make_vsr()], context="")
        processed = process_retrieval_results(rsr.results, min_score=0.0, max_results=10)
        assert isinstance(processed, list) and len(processed) == 1


# ---- SECTION 10: SUPPORTED CALLING PATTERNS ----

class TestSupportedCallingPatterns:
    def test_agent_citation_positional_args(self):
        c = AgentCitation("doc-p", "file.pdf", "ck-p")
        assert c.document_id == "doc-p" and c.filename == "file.pdf" and c.chunk_id == "ck-p"

    def test_agent_request_positional_query(self):
        assert AgentRequest("my question").query == "my question"

    def test_search_request_positional_query(self):
        assert SearchRequest("my search").query == "my search"

    def test_document_chunk_explicit_empty_metadata(self):
        chunk = DocumentChunk(chunk_id="ck", chunk_index=0, document_id="d", filename="f.pdf",
                               page_number=1, content="c", content_type="text", metadata={})
        assert chunk.metadata == {}

    def test_vector_search_result_empty_metadata(self):
        vsr = VectorSearchResult(chunk_id="ck", score=0.5, document_id="d", filename="f.pdf",
                                  page_number=None, chunk_index=0, content_type="text",
                                  content="c", metadata={})
        assert vsr.metadata == {}

    def test_retrieval_service_result_empty_results(self):
        rsr = RetrievalServiceResult(query_vector_dimension=4, results=[], context="")
        assert rsr.total_results == 0 and rsr.has_results is False

    def test_search_result_from_dict_with_dict_citations(self):
        data = {"query": "q?", "status": "RESULTS_FOUND", "context": "ctx", "metadata": {},
                "citations": [{"document_id": "d", "filename": "f.pdf", "chunk_id": "ck",
                               "page_number": 1, "content_type": "text", "score": 0.9, "metadata": {}}]}
        sr = SearchResult.from_dict(data)
        assert sr.query == "q?" and len(sr.citations) == 1 and isinstance(sr.citations[0], AgentCitation)

    def test_agent_response_from_dict_with_dict_citations(self):
        data = {"answer": "a", "agent_name": "S", "status": "success", "metadata": {}, "error": None,
                "citations": [{"document_id": "d", "filename": "f.pdf", "chunk_id": "ck",
                               "page_number": None, "content_type": "text", "score": 0.75, "metadata": {}}]}
        resp = AgentResponse.from_dict(data)
        assert resp.answer == "a" and len(resp.citations) == 1

    def test_vision_request_from_dict_with_dict_evidence(self):
        data = {"query": "analyze", "metadata": {}, "session_id": None,
                "evidence": [{"document_id": "d", "filename": "f.pdf", "chunk_id": "ck",
                              "page_number": 1, "chunk_index": 0, "content_type": "image", "metadata": {}}]}
        vr = VisionRequest.from_dict(data)
        assert vr.query == "analyze" and len(vr.evidence) == 1

    def test_chunking_result_get_chunks_by_type(self):
        cr = ChunkingResult(document_id="d", filename="f.pdf",
                             chunks=[_make_chunk(chunk_id="t", content_type="text"),
                                     _make_chunk(chunk_id="i", content_type="image")])
        assert len(cr.get_chunks_by_type("text")) == 1


# ---- SECTION 11: OPTIONAL FIELD BEHAVIOR ----

class TestOptionalFieldBehavior:
    def test_agent_citation_optional_field_defaults(self):
        c = AgentCitation(document_id="d", filename="f.pdf", chunk_id="ck")
        assert c.page_number is None and c.metadata == {}

    def test_agent_request_optional_field_defaults(self):
        ar = AgentRequest(query="q")
        assert ar.session_id is None and ar.document_filter is None

    def test_search_request_optional_field_defaults(self):
        sr = SearchRequest(query="q")
        assert sr.top_k is None and sr.min_score is None and sr.max_results is None

    def test_vision_result_optional_field_defaults(self):
        vr = VisionResult(query="q")
        assert vr.evidence == [] and vr.has_evidence is False and vr.error is None

    def test_visual_evidence_optional_image_field_defaults(self):
        ev = VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck")
        assert ev.image_path is None and ev.image_bytes is None and ev.image_format is None
        assert ev.width is None and ev.height is None and ev.description is None

    def test_agent_response_optional_error_default(self):
        assert AgentResponse(answer="", agent_name="A").error is None

    def test_process_retrieval_results_default_min_score_includes_zero(self):
        assert len(process_retrieval_results([_make_vsr(score=0.0)])) == 1

    def test_document_chunk_page_number_none_is_accepted(self):
        assert isinstance(validate_chunks([_make_chunk(page_number=None)]), ChunkValidationResult)


# ---- SECTION 12: UNKNOWN FIELD BEHAVIOR ----

class TestUnknownFieldBehavior:
    def test_agent_citation_from_dict_ignores_unknown_keys(self):
        data = {"document_id": "d", "filename": "f.pdf", "chunk_id": "ck",
                "page_number": 1, "content_type": "text", "score": 0.9, "metadata": {},
                "unknown_field": "ignored", "another_extra": 42}
        assert AgentCitation.from_dict(data).document_id == "d"

    def test_agent_request_from_dict_ignores_unknown_keys(self):
        data = {"query": "q", "session_id": None, "document_filter": None,
                "metadata": {}, "extra_key": "ignored"}
        assert AgentRequest.from_dict(data).query == "q"

    def test_search_request_from_dict_ignores_unknown_keys(self):
        data = {"query": "q", "top_k": None, "min_score": None, "max_results": None,
                "collection_name": None, "session_id": None, "document_filter": None,
                "metadata": {}, "future_field": "ignored"}
        assert SearchRequest.from_dict(data).query == "q"

    def test_vision_result_from_dict_ignores_unknown_keys(self):
        data = {"query": "q", "status": "success", "description": "", "evidence": [],
                "document_id": "", "filename": "", "page_number": None, "chunk_id": "",
                "content_type": "image", "metadata": {}, "error": None, "totally_unknown": "noop"}
        assert VisionResult.from_dict(data).query == "q"

    def test_visual_evidence_from_dict_ignores_unknown_keys(self):
        data = {"document_id": "d", "filename": "f.pdf", "chunk_id": "ck",
                "page_number": 1, "chunk_index": 0, "content_type": "image",
                "image_path": None, "image_bytes": None, "image_format": None,
                "width": None, "height": None, "description": None,
                "metadata": {}, "future_unused_key": "ignored"}
        assert VisualEvidence.from_dict(data).document_id == "d"


# ---- SECTION 13: REQUEST / RESULT ROUND TRIP ----

class TestRequestResultRoundTrip:
    def test_agent_request_round_trip(self):
        orig = AgentRequest(query="rt", session_id="s", metadata={"v": "1.0"})
        r = AgentRequest.from_dict(orig.to_dict())
        assert r.query == orig.query and r.session_id == orig.session_id and r.metadata["v"] == "1.0"

    def test_vision_request_evidence_round_trip(self):
        ev = _make_evidence(document_id="rt-doc", chunk_id="rt-ck", page_number=5)
        req = VisionRequest(query="RT", evidence=[ev], session_id="s")
        r = VisionRequest.from_dict(req.to_dict())
        assert len(r.evidence) == 1
        assert r.evidence[0].document_id == "rt-doc" and r.evidence[0].page_number == 5

    def test_search_result_multi_citations_round_trip(self):
        citations = [_make_citation(chunk_id=f"ck-{i}", document_id=f"doc-{i}", page_number=i)
                     for i in range(1, 4)]
        sr = SearchResult(query="multi", status="RESULTS_FOUND", citations=citations)
        r = SearchResult.from_dict(sr.to_dict())
        assert len(r.citations) == 3
        for i, c in enumerate(r.citations, start=1):
            assert c.chunk_id == f"ck-{i}" and c.document_id == f"doc-{i}"

    def test_vision_result_with_evidence_round_trip(self):
        ev = _make_evidence(document_id="rv-doc", filename="rv.pdf", chunk_id="rv-ck",
                             page_number=3, content_type="chart", metadata={"src": "rt"})
        orig = VisionResult(query="RT vision", status="success", description="line chart",
                             evidence=[ev], document_id="rv-doc", filename="rv.pdf",
                             page_number=3, chunk_id="rv-ck", content_type="chart")
        r = VisionResult.from_dict(orig.to_dict())
        assert r.query == orig.query and r.document_id == "rv-doc" and len(r.evidence) == 1
        assert r.evidence[0].metadata["src"] == "rt"

    def test_agent_response_with_citations_round_trip(self):
        citations = [_make_citation(chunk_id="rt-a"), _make_citation(chunk_id="rt-b")]
        orig = AgentResponse(answer="Revenue +20%.", agent_name="S", citations=citations, metadata={"lat": 55})
        r = AgentResponse.from_dict(orig.to_dict())
        assert r.answer == orig.answer and len(r.citations) == 2


# ---- SECTION 14: MUTATION SAFETY ----

class TestMutationSafety:
    def test_process_retrieval_results_does_not_mutate_input_list(self):
        items = [_make_vsr(chunk_id="a", score=0.9), _make_vsr(chunk_id="b", score=0.5)]
        original_ids = [id(v) for v in items]
        process_retrieval_results(items, min_score=0.6, max_results=10)
        assert [id(v) for v in items] == original_ids

    def test_validate_chunks_does_not_mutate_input_chunks(self):
        chunk = _make_chunk(content="original")
        validate_chunks([chunk])
        assert chunk.content == "original"

    def test_agent_citation_metadata_is_independent_copy(self):
        meta = {"key": "value"}
        c = AgentCitation(document_id="d", filename="f.pdf", chunk_id="ck", metadata=meta)
        meta["key"] = "mutated"
        assert c.metadata.get("key") == "value"

    def test_agent_request_metadata_is_independent_copy(self):
        meta = {"src": "initial"}
        ar = AgentRequest(query="q", metadata=meta)
        meta["src"] = "mutated"
        assert ar.metadata.get("src") == "initial"

    def test_search_request_metadata_is_independent_copy(self):
        meta = {"priority": "low"}
        sr = SearchRequest(query="q", metadata=meta)
        meta["priority"] = "high"
        assert sr.metadata.get("priority") == "low"

    def test_visual_evidence_metadata_is_independent_copy(self):
        meta = {"tool": "camelot"}
        ev = VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck", metadata=meta)
        meta["tool"] = "mutated"
        assert ev.metadata.get("tool") == "camelot"

    def test_vision_result_evidence_list_length_at_construction(self):
        vres = VisionResult(query="q", evidence=[_make_evidence()])
        assert len(vres.evidence) == 1

    def test_search_result_from_response_does_not_mutate_response_citations(self):
        c = _make_citation()
        resp = AgentResponse(answer="A", agent_name="S", citations=[c], metadata={"query": "q", "context": ""})
        original_len = len(resp.citations)
        SearchResult.from_response(resp)
        assert len(resp.citations) == original_len

    def test_prepare_for_embedding_does_not_mutate_input(self):
        chunk = _make_chunk(content="immutable")
        prepare_for_embedding([chunk])
        assert chunk.content == "immutable"

    def test_build_retrieval_context_does_not_mutate_input(self):
        vsr = _make_vsr(content="ctx content")
        build_retrieval_context([vsr])
        assert vsr.content == "ctx content"

    def test_agent_state_to_dict_does_not_mutate_citations(self):
        state = AgentState(query="q", citations=[_make_citation()])
        orig_len = len(state.citations)
        state.to_dict()
        assert len(state.citations) == orig_len
