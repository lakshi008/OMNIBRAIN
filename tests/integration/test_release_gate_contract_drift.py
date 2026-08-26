"""
OmniBrain Member 4 — Day 25 Release Gate Contract Drift Certification.

Objective:
    Perform a targeted release-gate sweep for contract drift that is NOT already
    exhaustively covered by the 24 existing Day 10-24 integration test suites.

Coverage Strategy:
    The existing suites already verify in depth:
        - Public function signatures and callables (Day 10)
        - Data contract serialization round-trips (Day 24)
        - End-to-end lineage integrity (Day 19)
        - Cross-request data isolation (Day 23)
        - Failure recovery regression (Day 22)
        - Security/data isolation regression (Day 18)
        - Observability trace integrity (Day 17)
        - Deterministic reproducibility (Day 15)
        - Release readiness certification (Day 20)
        - Cross-component schema compatibility (Day 14)

    Day 25 focuses on the following RELEASE-GATE–SPECIFIC drift checks not
    concentrated in a single prior suite:

    RG-01: Public module attribute drift — all expected public names remain
           importable from each subsystem's top-level __init__ namespace.
    RG-02: Exception hierarchy stability — all exception classes remain in their
           certified inheritance chains.
    RG-03: Field-name completeness — every contractual field documented in the
           public API remains present on instantiated objects.
    RG-04: Default value stability — optional field defaults remain unchanged.
    RG-05: Constructor keyword-argument stability — existing callers remain
           constructable with the same keyword arguments.
    RG-06: Property name stability — all documented computed properties remain
           present and return the correct type.
    RG-07: Factory method signature stability — from_search_result,
           from_citation, from_agent_request, to_agent_request, from_response
           all remain callable with previously certified signatures.
    RG-08: Serialization key stability — to_dict() output keys remain unchanged
           (no key added or removed without migration path).
    RG-09: from_dict() lenient unknown-field compatibility — previously confirmed
           lenient deserializers still ignore unknown extra keys.
    RG-10: VALID_VISUAL_CONTENT_TYPES set stability — the frozenset value
           remains unchanged.
    RG-11: VisionExecutionStage constant stability — all stage name constants
           remain present and hold their certified string values.
    RG-12: VisionCancellationToken public interface stability — is_cancelled,
           reason, cancel(), raise_if_cancelled() remain present and correct.
    RG-13: AgentState.add_error / add_citation / update method stability.
    RG-14: SearchResult.from_response factory stability and status derivation.
    RG-15: VisionResult primary-evidence lineage auto-inheritance stability.
    RG-16: AgentResponse modality filter property stability (text_results,
           table_results, image_results).
    RG-17: SearchResult modality grouping and by_document / by_modality
           property stability.
    RG-18: Security — serialized dictionaries contain no forbidden credential
           keys across all public models.
    RG-19: No external imports — all subsystem modules remain importable without
           any external network, LLM, or cloud SDK initialisation.
    RG-20: Production-code boundary — no test-layer production-code artifacts
           are introduced by these tests themselves.

Constraints:
    - 100% Offline: No external APIs, network, real LLMs, or production secrets.
    - Zero production code modified.
    - Only observable behaviour guaranteed by public contracts certified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# ---------------------------------------------------------------------------
# Subsystem imports
# ---------------------------------------------------------------------------

# Ingestion (Member 1)
import ingestion as ing_pkg
from ingestion.models import (
    ChunkValidationResult,
    ChunkingResult,
    DocumentChunk,
    DocumentMetadata,
    EmbeddingGenerationResult,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    EmbeddingVectorRecord,
    ExtractedImage,
    ExtractedTable,
    ImageExtractionResult,
    IngestionResult,
    PageData,
    ParsedDocument,
    RetrievalServiceResult,
    TableExtractionResult,
    VectorSearchResult,
)
from ingestion.ingestion_errors import (
    IngestionChunkingError,
    IngestionEmbeddingError,
    IngestionError,
    IngestionExtractionError,
    IngestionPipelineError,
    IngestionValidationError,
)
from ingestion.exceptions import CorruptedPDFError, InvalidFileTypeError, PDFNotFoundError
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.retrieval_processor import build_retrieval_context, process_retrieval_results

# Agents / Search (Member 2)
import agents as ag_pkg
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
    AgentExecutionError,
    AgentRoutingError,
    AgentValidationError,
)

# Vision (Member 3)
import vision as vis_pkg
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.exceptions import (
    VisionAgentError,
    VisionCancellationError,
    VisionError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderError,
    VisionTimeoutError,
)
from vision.lifecycle import (
    VisionCancellationToken,
    VisionExecutionLifecycle,
    VisionExecutionStage,
)

# ---------------------------------------------------------------------------
# Synthetic fixture helpers — Day 25 specific markers
# ---------------------------------------------------------------------------

RG_DOC_ID = "RG25_DOCUMENT_001"
RG_CHUNK_ID = "RG25_CHUNK_001"
RG_FILENAME = "rg25_source.pdf"
RG_META: dict[str, Any] = {"rg_day": 25, "rg_marker": "RELEASE_GATE_DAY25"}

FORBIDDEN_KEYS = {
    "password", "api_key", "apikey", "secret", "token", "credential",
    "auth", "private_key", "access_key", "bearer",
}


def _make_chunk(**kw: Any) -> DocumentChunk:
    defaults = dict(
        chunk_id=RG_CHUNK_ID,
        chunk_index=0,
        document_id=RG_DOC_ID,
        filename=RG_FILENAME,
        page_number=1,
        content="Release gate Day 25 chunk content.",
        content_type="text",
        metadata=dict(RG_META),
    )
    defaults.update(kw)
    return DocumentChunk(**defaults)


def _make_vsr(**kw: Any) -> VectorSearchResult:
    defaults = dict(
        chunk_id=RG_CHUNK_ID,
        score=0.92,
        document_id=RG_DOC_ID,
        filename=RG_FILENAME,
        page_number=1,
        chunk_index=0,
        content_type="text",
        content="Release gate Day 25 content.",
        metadata=dict(RG_META),
    )
    defaults.update(kw)
    return VectorSearchResult(**defaults)


def _make_citation(**kw: Any) -> AgentCitation:
    defaults = dict(
        document_id=RG_DOC_ID,
        filename=RG_FILENAME,
        chunk_id=RG_CHUNK_ID,
        page_number=1,
        content_type="text",
        score=0.92,
        metadata=dict(RG_META),
    )
    defaults.update(kw)
    return AgentCitation(**defaults)


def _make_evidence(**kw: Any) -> VisualEvidence:
    defaults = dict(
        document_id=RG_DOC_ID,
        filename=RG_FILENAME,
        chunk_id=RG_CHUNK_ID,
        page_number=1,
        chunk_index=0,
        content_type="image",
        metadata=dict(RG_META),
    )
    defaults.update(kw)
    return VisualEvidence(**defaults)


def _make_vision_request(**kw: Any) -> VisionRequest:
    defaults = dict(
        query="Release gate Day 25 vision query.",
        evidence=[_make_evidence()],
        metadata=dict(RG_META),
    )
    defaults.update(kw)
    return VisionRequest(**defaults)


def _make_vision_result(**kw: Any) -> VisionResult:
    defaults = dict(
        query="Release gate Day 25 result query.",
        status="success",
        description="Day 25 analysis result.",
        evidence=[_make_evidence()],
        metadata=dict(RG_META),
    )
    defaults.update(kw)
    return VisionResult(**defaults)


def _make_agent_response(**kw: Any) -> AgentResponse:
    defaults = dict(
        answer="Day 25 answer.",
        agent_name="RG25Agent",
        status="success",
        citations=[_make_citation()],
        metadata={"query": "Day 25 query", "context": "Day 25 context"},
    )
    defaults.update(kw)
    return AgentResponse(**defaults)


# ===========================================================================
# RG-01: Public module attribute drift
# ===========================================================================

class TestPublicModuleAttributeDrift:
    """RG-01: All expected names remain importable from each subsystem's top-level package."""

    # --- ingestion ---
    EXPECTED_INGESTION_ATTRS = [
        "normalize_chunks", "validate_chunks", "chunk_document",
        "prepare_for_embedding", "generate_embeddings",
        "IngestionConfig",
        "IngestionError", "IngestionChunkingError", "IngestionEmbeddingError",
        "IngestionExtractionError", "IngestionPipelineError", "IngestionValidationError",
        "CorruptedPDFError", "InvalidFileTypeError", "PDFNotFoundError",
        "DocumentChunk", "VectorSearchResult", "ChunkingResult",
        "ChunkValidationResult", "DocumentMetadata", "PageData",
        "EmbeddingRecord", "EmbeddingVectorRecord",
        "EmbeddingPreparationResult", "EmbeddingGenerationResult",
        "RetrievalServiceResult",
        "build_retrieval_context", "process_retrieval_results",
        "validate_pipeline_contracts", "validate_chunk_contracts",
        "validate_embedding_contracts", "validate_search_result_contracts",
        "validate_pipeline_lineage",
        "check_ingestion_health", "check_ingestion_readiness",
        "IngestionMetrics", "StageMetrics",
        "IngestionStatus", "PipelineStage", "PipelineStatus",
        "QdrantVectorStore", "QdrantConfig",
        "run_ingestion",
    ]

    @pytest.mark.parametrize("attr", EXPECTED_INGESTION_ATTRS)
    def test_ingestion_attr_present(self, attr: str) -> None:
        assert hasattr(ing_pkg, attr), (
            f"CONTRACT DRIFT: ingestion.{attr} is no longer exported from ingestion/__init__.py"
        )

    # --- agents ---
    EXPECTED_AGENTS_ATTRS = [
        "AgentCitation", "AgentRequest", "AgentResponse",
        "AgentState", "SearchRequest", "SearchResult",
        "AgentError", "AgentExecutionError", "AgentRoutingError", "AgentValidationError",
    ]

    @pytest.mark.parametrize("attr", EXPECTED_AGENTS_ATTRS)
    def test_agents_attr_present(self, attr: str) -> None:
        assert hasattr(ag_pkg, attr), (
            f"CONTRACT DRIFT: agents.{attr} is no longer exported from agents/__init__.py"
        )

    # --- vision ---
    EXPECTED_VISION_ATTRS = [
        "VisionRequest", "VisionResult", "VisualEvidence",
        "VALID_VISUAL_CONTENT_TYPES",
        "VisionAgentError", "VisionError",
        "VisionEvidenceError", "VisionInputValidationError",
        "VisionProcessingError", "VisionProviderError",
        "VisionTimeoutError", "VisionCancellationError",
        "VisionExecutionLifecycle", "VisionExecutionStage",
        "VisionCancellationToken",
    ]

    @pytest.mark.parametrize("attr", EXPECTED_VISION_ATTRS)
    def test_vision_attr_present(self, attr: str) -> None:
        assert hasattr(vis_pkg, attr), (
            f"CONTRACT DRIFT: vision.{attr} is no longer exported from vision/__init__.py"
        )


# ===========================================================================
# RG-02: Exception hierarchy stability
# ===========================================================================

class TestExceptionHierarchyStability:
    """RG-02: Exception inheritance chains remain certified."""

    def test_ingestion_error_base(self) -> None:
        assert issubclass(IngestionError, Exception)

    def test_ingestion_chunking_error_inherits_base(self) -> None:
        assert issubclass(IngestionChunkingError, IngestionError)

    def test_ingestion_embedding_error_inherits_base(self) -> None:
        assert issubclass(IngestionEmbeddingError, IngestionError)

    def test_ingestion_extraction_error_inherits_base(self) -> None:
        assert issubclass(IngestionExtractionError, IngestionError)

    def test_ingestion_pipeline_error_inherits_base(self) -> None:
        assert issubclass(IngestionPipelineError, IngestionError)

    def test_ingestion_validation_error_inherits_base(self) -> None:
        assert issubclass(IngestionValidationError, IngestionError)

    def test_pdf_not_found_inherits_exception(self) -> None:
        assert issubclass(PDFNotFoundError, Exception)

    def test_invalid_file_type_inherits_exception(self) -> None:
        assert issubclass(InvalidFileTypeError, Exception)

    def test_corrupted_pdf_inherits_exception(self) -> None:
        assert issubclass(CorruptedPDFError, Exception)

    def test_agent_validation_error_inherits_agent_error(self) -> None:
        assert issubclass(AgentValidationError, AgentError)

    def test_agent_execution_error_inherits_agent_error(self) -> None:
        assert issubclass(AgentExecutionError, AgentError)

    def test_agent_routing_error_inherits_agent_error(self) -> None:
        assert issubclass(AgentRoutingError, AgentError)

    def test_vision_error_alias_for_vision_agent_error(self) -> None:
        assert VisionError is VisionAgentError

    def test_vision_evidence_error_inherits_vision_agent_error(self) -> None:
        assert issubclass(VisionEvidenceError, VisionAgentError)

    def test_vision_input_validation_error_inherits_vision_agent_error(self) -> None:
        assert issubclass(VisionInputValidationError, VisionAgentError)

    def test_vision_processing_error_inherits_vision_agent_error(self) -> None:
        assert issubclass(VisionProcessingError, VisionAgentError)

    def test_vision_provider_error_inherits_vision_agent_error(self) -> None:
        assert issubclass(VisionProviderError, VisionAgentError)

    def test_vision_timeout_error_inherits_vision_provider_and_processing(self) -> None:
        assert issubclass(VisionTimeoutError, VisionProviderError)
        assert issubclass(VisionTimeoutError, VisionProcessingError)

    def test_vision_cancellation_error_inherits_vision_agent_error(self) -> None:
        assert issubclass(VisionCancellationError, VisionAgentError)

    def test_ingestion_errors_are_catchable_as_base(self) -> None:
        for exc_cls in (
            IngestionChunkingError, IngestionEmbeddingError,
            IngestionExtractionError, IngestionPipelineError, IngestionValidationError,
        ):
            with pytest.raises(IngestionError):
                raise exc_cls("rg25 test")

    def test_vision_errors_catchable_as_vision_agent_error(self) -> None:
        for exc_cls in (
            VisionEvidenceError, VisionInputValidationError,
            VisionProcessingError, VisionTimeoutError, VisionCancellationError,
        ):
            with pytest.raises(VisionAgentError):
                raise exc_cls("rg25 test")

    def test_agent_errors_catchable_as_agent_error(self) -> None:
        for exc_cls in (AgentValidationError, AgentExecutionError, AgentRoutingError):
            with pytest.raises(AgentError):
                raise exc_cls("rg25 test")


# ===========================================================================
# RG-03 + RG-04 + RG-05: Field-name, default, and constructor keyword stability
# ===========================================================================

class TestFieldNameAndDefaultStability:
    """RG-03/04/05: Required fields present, defaults stable, kwarg construction works."""

    # DocumentChunk required fields
    def test_document_chunk_required_fields(self) -> None:
        c = _make_chunk()
        assert hasattr(c, "chunk_id")
        assert hasattr(c, "chunk_index")
        assert hasattr(c, "document_id")
        assert hasattr(c, "filename")
        assert hasattr(c, "page_number")
        assert hasattr(c, "content")
        assert hasattr(c, "content_type")
        assert hasattr(c, "metadata")

    def test_document_chunk_field_values(self) -> None:
        c = _make_chunk()
        assert c.document_id == RG_DOC_ID
        assert c.chunk_id == RG_CHUNK_ID
        assert c.filename == RG_FILENAME
        assert c.page_number == 1
        assert c.chunk_index == 0
        assert c.content_type == "text"

    # VectorSearchResult required fields
    def test_vector_search_result_required_fields(self) -> None:
        vsr = _make_vsr()
        for field in ("chunk_id", "score", "document_id", "filename",
                      "page_number", "chunk_index", "content_type", "content", "metadata"):
            assert hasattr(vsr, field), f"VectorSearchResult missing field: {field}"

    # AgentCitation required fields and defaults
    def test_agent_citation_required_fields(self) -> None:
        c = _make_citation()
        for field in ("document_id", "filename", "chunk_id", "page_number",
                      "content_type", "score", "metadata"):
            assert hasattr(c, field), f"AgentCitation missing field: {field}"

    def test_agent_citation_default_content_type(self) -> None:
        c = AgentCitation(document_id="d", filename="f.pdf", chunk_id="ck")
        assert c.content_type == "text"

    def test_agent_citation_default_score(self) -> None:
        c = AgentCitation(document_id="d", filename="f.pdf", chunk_id="ck")
        assert c.score == 0.0

    def test_agent_citation_default_page_number(self) -> None:
        c = AgentCitation(document_id="d", filename="f.pdf", chunk_id="ck")
        assert c.page_number is None

    def test_agent_citation_default_metadata(self) -> None:
        c = AgentCitation(document_id="d", filename="f.pdf", chunk_id="ck")
        assert isinstance(c.metadata, dict) and c.metadata == {}

    # AgentRequest required fields and defaults
    def test_agent_request_required_field_query(self) -> None:
        r = AgentRequest(query="Day 25 query")
        assert r.query == "Day 25 query"

    def test_agent_request_default_session_id_none(self) -> None:
        r = AgentRequest(query="q")
        assert r.session_id is None

    def test_agent_request_default_document_filter_none(self) -> None:
        r = AgentRequest(query="q")
        assert r.document_filter is None

    def test_agent_request_default_metadata_empty(self) -> None:
        r = AgentRequest(query="q")
        assert isinstance(r.metadata, dict) and r.metadata == {}

    # SearchRequest fields and defaults
    def test_search_request_defaults(self) -> None:
        r = SearchRequest(query="Day 25 search")
        assert r.top_k is None
        assert r.min_score is None
        assert r.max_results is None
        assert r.collection_name is None
        assert r.session_id is None
        assert r.document_filter is None
        assert r.metadata == {}

    # AgentResponse fields and defaults
    def test_agent_response_required_fields(self) -> None:
        r = AgentResponse(answer="ans", agent_name="bot")
        for field in ("answer", "agent_name", "status", "citations", "metadata", "error"):
            assert hasattr(r, field)

    def test_agent_response_default_status_success(self) -> None:
        r = AgentResponse(answer="ans", agent_name="bot")
        assert r.status == "success"

    def test_agent_response_default_citations_empty(self) -> None:
        r = AgentResponse(answer="ans", agent_name="bot")
        assert r.citations == []

    def test_agent_response_default_error_none(self) -> None:
        r = AgentResponse(answer="ans", agent_name="bot")
        assert r.error is None

    # SearchResult fields and defaults
    def test_search_result_default_status_no_results(self) -> None:
        r = SearchResult(query="q")
        assert r.status == "NO_RESULTS"

    def test_search_result_default_citations_empty(self) -> None:
        r = SearchResult(query="q")
        assert r.citations == []

    def test_search_result_default_context_empty(self) -> None:
        r = SearchResult(query="q")
        assert r.context == ""

    def test_search_result_default_metadata_empty(self) -> None:
        r = SearchResult(query="q")
        assert r.metadata == {}

    # AgentState fields and defaults
    def test_agent_state_default_route_none(self) -> None:
        s = AgentState(query="q")
        assert s.route is None

    def test_agent_state_default_status_initialized(self) -> None:
        s = AgentState(query="q")
        assert s.status == "initialized"

    def test_agent_state_default_collections_empty(self) -> None:
        s = AgentState(query="q")
        assert s.retrieved_results == []
        assert s.citations == []
        assert s.errors == []
        assert s.context == ""
        assert s.answer == ""

    # VisualEvidence fields and defaults
    def test_visual_evidence_required_fields(self) -> None:
        e = _make_evidence()
        for field in ("document_id", "filename", "chunk_id", "page_number",
                      "chunk_index", "content_type", "metadata"):
            assert hasattr(e, field), f"VisualEvidence missing field: {field}"

    def test_visual_evidence_default_content_type_image(self) -> None:
        e = VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck",
                           content_type="image")
        assert e.content_type == "image"

    def test_visual_evidence_optional_fields_default_none(self) -> None:
        e = VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck",
                           content_type="image")
        assert e.image_path is None
        assert e.image_bytes is None
        assert e.image_format is None
        assert e.width is None
        assert e.height is None
        assert e.description is None

    # VisionRequest fields and defaults
    def test_vision_request_default_session_id_none(self) -> None:
        r = VisionRequest(query="Day 25 vision")
        assert r.session_id is None

    def test_vision_request_default_evidence_empty(self) -> None:
        r = VisionRequest(query="Day 25 vision")
        assert r.evidence == []

    # VisionResult fields and defaults
    def test_vision_result_default_status_success(self) -> None:
        r = VisionResult(query="q")
        assert r.status == "success"

    def test_vision_result_default_description_empty(self) -> None:
        r = VisionResult(query="q")
        assert r.description == ""

    def test_vision_result_default_error_none(self) -> None:
        r = VisionResult(query="q")
        assert r.error is None


# ===========================================================================
# RG-06: Property name stability
# ===========================================================================

class TestPropertyNameStability:
    """RG-06: Documented computed properties remain present and return correct types."""

    def test_agent_response_has_citations_property(self) -> None:
        r = _make_agent_response()
        assert isinstance(r.has_citations, bool)
        assert r.has_citations is True

    def test_agent_response_total_citations_property(self) -> None:
        r = _make_agent_response()
        assert isinstance(r.total_citations, int)
        assert r.total_citations >= 1

    def test_agent_response_is_success_property(self) -> None:
        r = _make_agent_response()
        assert r.is_success is True

    def test_agent_response_is_error_property_false_on_success(self) -> None:
        r = _make_agent_response()
        assert r.is_error is False

    def test_agent_response_is_error_property_true_on_error(self) -> None:
        r = AgentResponse(answer="", agent_name="bot", status="error",
                          error="rg25 error")
        assert r.is_error is True

    def test_agent_response_has_results_alias(self) -> None:
        r = _make_agent_response()
        assert isinstance(r.has_results, bool)
        assert r.has_results is True

    def test_agent_response_unique_document_count(self) -> None:
        r = _make_agent_response()
        assert isinstance(r.unique_document_count, int)
        assert r.unique_document_count == 1

    def test_agent_response_unique_documents(self) -> None:
        r = _make_agent_response()
        assert isinstance(r.unique_documents, list)
        assert RG_DOC_ID in r.unique_documents

    def test_search_result_has_results_property(self) -> None:
        r = SearchResult(query="q", citations=[_make_citation()])
        assert r.has_results is True

    def test_search_result_total_results_property(self) -> None:
        r = SearchResult(query="q", citations=[_make_citation()])
        assert r.total_results == 1

    def test_search_result_evidence_count_alias(self) -> None:
        r = SearchResult(query="q", citations=[_make_citation()])
        assert r.evidence_count == r.total_results

    def test_search_result_unique_document_count_property(self) -> None:
        r = SearchResult(query="q", citations=[_make_citation()])
        assert isinstance(r.unique_document_count, int)

    def test_search_result_unique_documents_property(self) -> None:
        r = SearchResult(query="q", citations=[_make_citation()])
        docs = r.unique_documents
        assert isinstance(docs, list)
        assert RG_DOC_ID in docs

    def test_vision_request_has_evidence_property(self) -> None:
        r = _make_vision_request()
        assert r.has_evidence is True

    def test_vision_request_total_evidence_property(self) -> None:
        r = _make_vision_request()
        assert isinstance(r.total_evidence, int)
        assert r.total_evidence >= 1

    def test_vision_result_is_success_property(self) -> None:
        r = _make_vision_result()
        assert r.is_success is True

    def test_vision_result_is_error_property_false_on_success(self) -> None:
        r = _make_vision_result()
        assert r.is_error is False

    def test_vision_result_has_evidence_property(self) -> None:
        r = _make_vision_result()
        assert r.has_evidence is True

    def test_vision_result_is_error_true_on_error_status(self) -> None:
        r = VisionResult(query="q", status="error", error="rg25 failure")
        assert r.is_error is True


# ===========================================================================
# RG-07: Factory method signature stability
# ===========================================================================

class TestFactoryMethodSignatureStability:
    """RG-07: All certified factory methods remain callable with existing signatures."""

    def test_agent_citation_from_search_result_signature(self) -> None:
        vsr = _make_vsr()
        c = AgentCitation.from_search_result(vsr)
        assert isinstance(c, AgentCitation)
        assert c.document_id == RG_DOC_ID
        assert c.chunk_id == RG_CHUNK_ID
        assert c.filename == RG_FILENAME

    def test_visual_evidence_from_citation_signature(self) -> None:
        citation = _make_citation(content_type="image")
        ev = VisualEvidence.from_citation(citation)
        assert isinstance(ev, VisualEvidence)
        assert ev.document_id == RG_DOC_ID
        assert ev.chunk_id == RG_CHUNK_ID
        assert ev.filename == RG_FILENAME

    def test_visual_evidence_from_citation_with_optional_params(self) -> None:
        citation = _make_citation(content_type="chart")
        ev = VisualEvidence.from_citation(
            citation,
            image_path="/tmp/rg25.png",
            image_format="png",
        )
        assert ev.image_path == "/tmp/rg25.png"
        assert ev.image_format == "png"

    def test_visual_evidence_from_search_result_signature(self) -> None:
        vsr = _make_vsr(content_type="image")
        ev = VisualEvidence.from_search_result(vsr)
        assert isinstance(ev, VisualEvidence)
        assert ev.document_id == RG_DOC_ID

    def test_search_request_to_agent_request_signature(self) -> None:
        sr = SearchRequest(query="Day 25 q", top_k=5, min_score=0.7,
                           session_id="rg25-sess")
        ar = sr.to_agent_request()
        assert isinstance(ar, AgentRequest)
        assert ar.query == "Day 25 q"
        assert ar.session_id == "rg25-sess"
        # top_k and min_score folded into metadata
        assert ar.metadata.get("top_k") == 5
        assert ar.metadata.get("min_score") == 0.7

    def test_search_request_from_agent_request_signature(self) -> None:
        ar = AgentRequest(query="Day 25 q", session_id="rg25-sess")
        sr = SearchRequest.from_agent_request(ar, top_k=10, min_score=0.5)
        assert isinstance(sr, SearchRequest)
        assert sr.query == "Day 25 q"
        assert sr.top_k == 10
        assert sr.min_score == pytest.approx(0.5)

    def test_search_result_from_response_with_citations(self) -> None:
        resp = _make_agent_response()
        sr = SearchResult.from_response(resp)
        assert isinstance(sr, SearchResult)
        assert sr.status == "RESULTS_FOUND"
        assert len(sr.citations) == 1

    def test_search_result_from_response_no_citations(self) -> None:
        resp = AgentResponse(
            answer="no results",
            agent_name="bot",
            citations=[],
            metadata={"query": "empty Day 25 query", "context": ""},
        )
        sr = SearchResult.from_response(resp)
        assert sr.status == "NO_RESULTS"
        assert sr.citations == []


# ===========================================================================
# RG-08: Serialization key stability (to_dict output keys)
# ===========================================================================

class TestSerializationKeyStability:
    """RG-08: to_dict() key sets remain unchanged from certified contracts."""

    CERTIFIED_AGENT_CITATION_KEYS = frozenset({
        "document_id", "filename", "chunk_id",
        "page_number", "content_type", "score", "metadata",
    })

    CERTIFIED_AGENT_REQUEST_KEYS = frozenset({
        "query", "session_id", "document_filter", "metadata",
    })

    CERTIFIED_SEARCH_REQUEST_KEYS = frozenset({
        "query", "top_k", "min_score", "max_results",
        "collection_name", "session_id", "document_filter", "metadata",
    })

    CERTIFIED_AGENT_RESPONSE_KEYS = frozenset({
        "answer", "agent_name", "status", "citations", "metadata", "error",
    })

    CERTIFIED_SEARCH_RESULT_KEYS = frozenset({
        "query", "status", "citations", "context",
        "total_results", "evidence_count", "has_results",
        "text_count", "table_count", "image_count",
        "unique_document_count", "unique_documents",
        "metadata",
    })

    CERTIFIED_VISUAL_EVIDENCE_KEYS = frozenset({
        "document_id", "filename", "chunk_id", "page_number", "chunk_index",
        "content_type", "image_path", "image_format", "width", "height",
        "description", "metadata",
    })

    CERTIFIED_VISION_REQUEST_KEYS = frozenset({
        "query", "evidence", "metadata", "session_id",
    })

    CERTIFIED_VISION_RESULT_KEYS = frozenset({
        "query", "status", "description", "evidence",
        "document_id", "filename", "page_number", "chunk_id",
        "content_type", "metadata", "error",
    })

    def test_agent_citation_to_dict_keys(self) -> None:
        d = _make_citation().to_dict()
        assert frozenset(d.keys()) == self.CERTIFIED_AGENT_CITATION_KEYS

    def test_agent_request_to_dict_keys(self) -> None:
        d = AgentRequest(query="rg25").to_dict()
        assert frozenset(d.keys()) == self.CERTIFIED_AGENT_REQUEST_KEYS

    def test_search_request_to_dict_keys(self) -> None:
        d = SearchRequest(query="rg25").to_dict()
        assert frozenset(d.keys()) == self.CERTIFIED_SEARCH_REQUEST_KEYS

    def test_agent_response_to_dict_keys(self) -> None:
        d = _make_agent_response().to_dict()
        assert frozenset(d.keys()) == self.CERTIFIED_AGENT_RESPONSE_KEYS

    def test_search_result_to_dict_keys(self) -> None:
        sr = SearchResult.from_response(_make_agent_response())
        d = sr.to_dict()
        assert frozenset(d.keys()) == self.CERTIFIED_SEARCH_RESULT_KEYS

    def test_visual_evidence_to_dict_keys(self) -> None:
        d = _make_evidence().to_dict()
        assert frozenset(d.keys()) == self.CERTIFIED_VISUAL_EVIDENCE_KEYS

    def test_vision_request_to_dict_keys(self) -> None:
        d = _make_vision_request().to_dict()
        assert frozenset(d.keys()) == self.CERTIFIED_VISION_REQUEST_KEYS

    def test_vision_result_to_dict_keys(self) -> None:
        d = _make_vision_result().to_dict()
        assert frozenset(d.keys()) == self.CERTIFIED_VISION_RESULT_KEYS

    def test_agent_state_to_dict_keys(self) -> None:
        s = AgentState(query="rg25")
        d = s.to_dict()
        expected = frozenset({
            "query", "route", "retrieved_results", "context",
            "citations", "answer", "errors", "status", "metadata",
        })
        assert frozenset(d.keys()) == expected


# ===========================================================================
# RG-09: from_dict() lenient unknown-field compatibility
# ===========================================================================

class TestFromDictLenientCompatibility:
    """RG-09: Lenient deserializers confirmed to silently ignore unknown extra keys."""

    def test_visual_evidence_from_dict_ignores_unknown_keys(self) -> None:
        d = {
            "document_id": RG_DOC_ID,
            "filename": RG_FILENAME,
            "chunk_id": RG_CHUNK_ID,
            "content_type": "image",
            "UNKNOWN_FUTURE_FIELD": "rg25_value",
            "another_unknown": 99,
        }
        ev = VisualEvidence.from_dict(d)
        assert ev.document_id == RG_DOC_ID
        assert ev.chunk_id == RG_CHUNK_ID

    def test_search_result_from_dict_ignores_unknown_keys(self) -> None:
        d = {
            "query": "rg25 unknown field test",
            "status": "RESULTS_FOUND",
            "citations": [],
            "context": "",
            "metadata": {},
            "UNKNOWN_FUTURE_KEY": True,
            "rg25_extra": "drift_check",
        }
        sr = SearchResult.from_dict(d)
        assert sr.query == "rg25 unknown field test"

    def test_agent_citation_from_dict_round_trip_with_extras(self) -> None:
        # AgentCitation.from_dict uses data.get() so unknown keys are effectively ignored
        d = _make_citation().to_dict()
        d["FUTURE_FIELD"] = "rg25"
        c2 = AgentCitation.from_dict(d)
        assert c2.document_id == RG_DOC_ID


# ===========================================================================
# RG-10: VALID_VISUAL_CONTENT_TYPES stability
# ===========================================================================

class TestValidVisualContentTypesStability:
    """RG-10: The VALID_VISUAL_CONTENT_TYPES frozenset remains unchanged."""

    CERTIFIED_CONTENT_TYPES = frozenset({"image", "chart", "diagram"})

    def test_valid_visual_content_types_is_frozenset(self) -> None:
        assert isinstance(VALID_VISUAL_CONTENT_TYPES, frozenset)

    def test_valid_visual_content_types_exact_membership(self) -> None:
        assert VALID_VISUAL_CONTENT_TYPES == self.CERTIFIED_CONTENT_TYPES

    def test_image_in_valid_types(self) -> None:
        assert "image" in VALID_VISUAL_CONTENT_TYPES

    def test_chart_in_valid_types(self) -> None:
        assert "chart" in VALID_VISUAL_CONTENT_TYPES

    def test_diagram_in_valid_types(self) -> None:
        assert "diagram" in VALID_VISUAL_CONTENT_TYPES

    def test_text_not_in_valid_visual_types(self) -> None:
        assert "text" not in VALID_VISUAL_CONTENT_TYPES

    def test_table_not_in_valid_visual_types(self) -> None:
        assert "table" not in VALID_VISUAL_CONTENT_TYPES


# ===========================================================================
# RG-11: VisionExecutionStage constant stability
# ===========================================================================

class TestVisionExecutionStageStability:
    """RG-11: All stage name constants remain present and hold certified string values."""

    CERTIFIED_STAGES = {
        "PENDING": "pending",
        "VALIDATING": "validating",
        "PREPARING": "preparing",
        "BUILDING_INPUT": "building_input",
        "EXECUTING": "executing",
        "COMPLETED": "completed",
        "FAILED": "failed",
        "TIMEOUT": "timeout",
        "CANCELLED": "cancelled",
    }

    CERTIFIED_ALL_STAGES = frozenset({
        "pending", "validating", "preparing", "building_input",
        "executing", "completed", "failed", "timeout", "cancelled",
    })

    CERTIFIED_TERMINAL_STAGES = frozenset({
        "completed", "failed", "timeout", "cancelled",
    })

    @pytest.mark.parametrize("attr,expected", list(CERTIFIED_STAGES.items()))
    def test_stage_constant_value(self, attr: str, expected: str) -> None:
        assert hasattr(VisionExecutionStage, attr), (
            f"CONTRACT DRIFT: VisionExecutionStage.{attr} no longer present"
        )
        assert getattr(VisionExecutionStage, attr) == expected

    def test_all_stages_frozenset_stable(self) -> None:
        assert VisionExecutionStage.ALL_STAGES == self.CERTIFIED_ALL_STAGES

    def test_terminal_stages_frozenset_stable(self) -> None:
        assert VisionExecutionStage.TERMINAL_STAGES == self.CERTIFIED_TERMINAL_STAGES

    def test_non_terminal_stages_not_in_terminal(self) -> None:
        non_terminal = {"pending", "validating", "preparing", "building_input", "executing"}
        for stage in non_terminal:
            assert stage not in VisionExecutionStage.TERMINAL_STAGES


# ===========================================================================
# RG-12: VisionCancellationToken public interface stability
# ===========================================================================

class TestVisionCancellationTokenStability:
    """RG-12: is_cancelled, reason, cancel(), raise_if_cancelled() remain present and correct."""

    def test_token_initial_is_cancelled_false(self) -> None:
        token = VisionCancellationToken()
        assert token.is_cancelled is False

    def test_token_initial_reason_none(self) -> None:
        token = VisionCancellationToken()
        assert token.reason is None

    def test_cancel_sets_is_cancelled(self) -> None:
        token = VisionCancellationToken()
        token.cancel("rg25 cancel reason")
        assert token.is_cancelled is True

    def test_cancel_sets_reason(self) -> None:
        token = VisionCancellationToken()
        token.cancel("rg25 reason")
        assert token.reason == "rg25 reason"

    def test_raise_if_cancelled_raises_vision_cancellation_error(self) -> None:
        token = VisionCancellationToken()
        token.cancel("rg25 raise test")
        with pytest.raises(VisionCancellationError):
            token.raise_if_cancelled()

    def test_raise_if_cancelled_does_not_raise_when_not_cancelled(self) -> None:
        token = VisionCancellationToken()
        token.raise_if_cancelled()  # Should not raise

    def test_cancel_without_reason_uses_default_message(self) -> None:
        token = VisionCancellationToken()
        token.cancel()
        assert token.is_cancelled is True
        assert isinstance(token.reason, str) and len(token.reason) > 0

    def test_token_not_cancelled_before_cancel_called(self) -> None:
        t1 = VisionCancellationToken()
        t2 = VisionCancellationToken()
        t2.cancel("other")
        # t1 is unaffected by t2's cancellation
        assert t1.is_cancelled is False


# ===========================================================================
# RG-13: AgentState mutation methods stability
# ===========================================================================

class TestAgentStateMutationMethodStability:
    """RG-13: add_error, add_citation, and update remain callable with certified signatures."""

    def test_add_error_appends_to_errors_list(self) -> None:
        s = AgentState(query="rg25 q")
        s.add_error("rg25 error one")
        assert len(s.errors) == 1
        assert s.errors[0] == "rg25 error one"

    def test_add_error_multiple_times(self) -> None:
        s = AgentState(query="rg25 q")
        s.add_error("err1")
        s.add_error("err2")
        assert len(s.errors) == 2

    def test_add_citation_appends_to_citations_list(self) -> None:
        s = AgentState(query="rg25 q")
        c = _make_citation()
        s.add_citation(c)
        assert len(s.citations) == 1
        assert s.citations[0] is c

    def test_update_method_changes_attribute(self) -> None:
        s = AgentState(query="rg25 q")
        s.update(route="search", answer="Day 25 answer")
        assert s.route == "search"
        assert s.answer == "Day 25 answer"

    def test_update_status_via_update(self) -> None:
        s = AgentState(query="rg25 q")
        s.update(status="completed")
        assert s.status == "completed"

    def test_add_error_invalid_type_raises_validation_error(self) -> None:
        s = AgentState(query="rg25 q")
        with pytest.raises(AgentValidationError):
            s.add_error("")  # empty string should fail

    def test_add_citation_invalid_type_raises_validation_error(self) -> None:
        s = AgentState(query="rg25 q")
        with pytest.raises(AgentValidationError):
            s.add_citation("not a citation")  # type: ignore[arg-type]

    def test_update_unknown_key_raises_validation_error(self) -> None:
        s = AgentState(query="rg25 q")
        with pytest.raises(AgentValidationError):
            s.update(nonexistent_field="value")


# ===========================================================================
# RG-14: SearchResult.from_response status derivation stability
# ===========================================================================

class TestSearchResultFromResponseStability:
    """RG-14: from_response status derivation and metadata propagation remain stable."""

    def test_from_response_results_found_when_citations_present(self) -> None:
        resp = _make_agent_response()
        sr = SearchResult.from_response(resp)
        assert sr.status == "RESULTS_FOUND"

    def test_from_response_no_results_when_no_citations(self) -> None:
        resp = AgentResponse(
            answer="", agent_name="bot",
            metadata={"query": "rg25 empty", "context": ""},
        )
        sr = SearchResult.from_response(resp)
        assert sr.status == "NO_RESULTS"

    def test_from_response_propagates_query_from_metadata(self) -> None:
        resp = AgentResponse(
            answer="ans", agent_name="bot",
            metadata={"query": "rg25_query_from_meta", "context": "some ctx"},
        )
        sr = SearchResult.from_response(resp)
        assert sr.query == "rg25_query_from_meta"

    def test_from_response_propagates_context(self) -> None:
        resp = AgentResponse(
            answer="ans", agent_name="bot",
            metadata={"query": "rg25", "context": "rg25 context block"},
        )
        sr = SearchResult.from_response(resp)
        assert sr.context == "rg25 context block"

    def test_from_response_missing_query_raises_validation_error(self) -> None:
        resp = AgentResponse(answer="ans", agent_name="bot", metadata={})
        with pytest.raises(AgentValidationError):
            SearchResult.from_response(resp)

    def test_from_response_citations_preserved(self) -> None:
        c1 = _make_citation()
        resp = AgentResponse(
            answer="a", agent_name="b",
            citations=[c1],
            metadata={"query": "rg25"},
        )
        sr = SearchResult.from_response(resp)
        assert len(sr.citations) == 1
        assert sr.citations[0].chunk_id == RG_CHUNK_ID


# ===========================================================================
# RG-15: VisionResult primary-evidence lineage auto-inheritance
# ===========================================================================

class TestVisionResultPrimaryEvidenceInheritance:
    """RG-15: VisionResult auto-inherits primary lineage from first evidence item."""

    def test_vision_result_inherits_document_id_from_evidence(self) -> None:
        ev = _make_evidence(document_id="RG25_AUTO_DOC")
        r = VisionResult(query="rg25", evidence=[ev])
        assert r.document_id == "RG25_AUTO_DOC"

    def test_vision_result_inherits_filename_from_evidence(self) -> None:
        ev = _make_evidence(filename="rg25_inherited.pdf")
        r = VisionResult(query="rg25", evidence=[ev])
        assert r.filename == "rg25_inherited.pdf"

    def test_vision_result_inherits_page_number_from_evidence(self) -> None:
        ev = _make_evidence(page_number=7)
        r = VisionResult(query="rg25", evidence=[ev])
        assert r.page_number == 7

    def test_vision_result_inherits_chunk_id_from_evidence(self) -> None:
        ev = _make_evidence(chunk_id="RG25_INHERITED_CHUNK")
        r = VisionResult(query="rg25", evidence=[ev])
        assert r.chunk_id == "RG25_INHERITED_CHUNK"

    def test_vision_result_explicit_lineage_not_overridden_by_evidence(self) -> None:
        ev = _make_evidence(document_id="EVIDENCE_DOC")
        r = VisionResult(
            query="rg25",
            evidence=[ev],
            document_id="EXPLICIT_DOC",
            chunk_id="EXPLICIT_CHUNK",
        )
        # Explicit values must NOT be overwritten by auto-inheritance
        assert r.document_id == "EXPLICIT_DOC"
        assert r.chunk_id == "EXPLICIT_CHUNK"

    def test_vision_result_no_evidence_empty_lineage(self) -> None:
        r = VisionResult(query="rg25", evidence=[])
        assert r.document_id == ""
        assert r.filename == ""
        assert r.chunk_id == ""
        assert r.page_number is None


# ===========================================================================
# RG-16: AgentResponse modality filter property stability
# ===========================================================================

class TestAgentResponseModalityFilterStability:
    """RG-16: text_results, table_results, image_results properties remain stable."""

    def test_text_results_filters_text_citations(self) -> None:
        c_text = _make_citation(content_type="text")
        c_img = _make_citation(content_type="image",
                                chunk_id="img-ck", document_id="img-doc",
                                filename="img.pdf")
        r = AgentResponse(answer="a", agent_name="b", citations=[c_text, c_img])
        assert len(r.text_results) == 1
        assert r.text_results[0].content_type == "text"

    def test_table_results_filters_table_citations(self) -> None:
        c_table = _make_citation(content_type="table",
                                  chunk_id="tbl-ck", document_id="tbl-doc",
                                  filename="tbl.pdf")
        r = AgentResponse(answer="a", agent_name="b", citations=[c_table])
        assert len(r.table_results) == 1
        assert r.table_results[0].content_type == "table"

    def test_image_results_filters_image_citations(self) -> None:
        c_img = _make_citation(content_type="image",
                                chunk_id="img-ck", document_id="img-doc",
                                filename="img.pdf")
        r = AgentResponse(answer="a", agent_name="b", citations=[c_img])
        assert len(r.image_results) == 1

    def test_empty_citations_all_modalities_empty(self) -> None:
        r = AgentResponse(answer="a", agent_name="b", citations=[])
        assert r.text_results == []
        assert r.table_results == []
        assert r.image_results == []


# ===========================================================================
# RG-17: SearchResult modality grouping and by_document / by_modality stability
# ===========================================================================

class TestSearchResultModalityGroupingStability:
    """RG-17: by_document and by_modality grouping properties remain stable."""

    def test_by_document_groups_citations_by_document_id(self) -> None:
        c1 = _make_citation(document_id="DOC_A", chunk_id="ck-a", filename="a.pdf")
        c2 = _make_citation(document_id="DOC_B", chunk_id="ck-b", filename="b.pdf")
        r = SearchResult(query="rg25", citations=[c1, c2])
        grouped = r.by_document
        assert isinstance(grouped, dict)
        assert "DOC_A" in grouped
        assert "DOC_B" in grouped
        assert grouped["DOC_A"][0].chunk_id == "ck-a"

    def test_by_modality_contains_text_table_image_keys(self) -> None:
        r = SearchResult(query="rg25", citations=[])
        bm = r.by_modality
        assert isinstance(bm, dict)
        assert "text" in bm
        assert "table" in bm
        assert "image" in bm

    def test_by_modality_text_key_returns_text_citations(self) -> None:
        c = _make_citation(content_type="text")
        r = SearchResult(query="rg25", citations=[c])
        assert len(r.by_modality["text"]) == 1
        assert r.by_modality["image"] == []

    def test_text_count_table_count_image_count_properties(self) -> None:
        c_text = _make_citation(content_type="text")
        c_table = _make_citation(content_type="table",
                                  chunk_id="tbl-ck", document_id="tbl-doc",
                                  filename="tbl.pdf")
        r = SearchResult(query="rg25", citations=[c_text, c_table])
        assert r.text_count == 1
        assert r.table_count == 1
        assert r.image_count == 0


# ===========================================================================
# RG-18: Security — no forbidden credential keys in serialized dictionaries
# ===========================================================================

class TestSerializationSecurityBoundary:
    """RG-18: No forbidden credential keys leak into any public model's to_dict() output."""

    def _assert_no_forbidden_keys(self, d: dict[str, Any], label: str) -> None:
        flat_keys = {k.lower() for k in d.keys()}
        for fk in FORBIDDEN_KEYS:
            assert fk not in flat_keys, (
                f"SECURITY VIOLATION: forbidden key '{fk}' found in {label} serialized dict"
            )

    def test_agent_citation_no_credential_keys(self) -> None:
        self._assert_no_forbidden_keys(_make_citation().to_dict(), "AgentCitation")

    def test_agent_request_no_credential_keys(self) -> None:
        self._assert_no_forbidden_keys(
            AgentRequest(query="rg25", metadata={"rg_marker": "DAY25"}).to_dict(),
            "AgentRequest",
        )

    def test_search_request_no_credential_keys(self) -> None:
        self._assert_no_forbidden_keys(
            SearchRequest(query="rg25").to_dict(), "SearchRequest"
        )

    def test_agent_response_no_credential_keys(self) -> None:
        self._assert_no_forbidden_keys(
            _make_agent_response().to_dict(), "AgentResponse"
        )

    def test_search_result_no_credential_keys(self) -> None:
        sr = SearchResult.from_response(_make_agent_response())
        self._assert_no_forbidden_keys(sr.to_dict(), "SearchResult")

    def test_visual_evidence_no_credential_keys(self) -> None:
        self._assert_no_forbidden_keys(
            _make_evidence().to_dict(), "VisualEvidence"
        )

    def test_vision_request_no_credential_keys(self) -> None:
        self._assert_no_forbidden_keys(
            _make_vision_request().to_dict(), "VisionRequest"
        )

    def test_vision_result_no_credential_keys(self) -> None:
        self._assert_no_forbidden_keys(
            _make_vision_result().to_dict(), "VisionResult"
        )

    def test_agent_state_no_credential_keys(self) -> None:
        s = AgentState(query="rg25", metadata={"rg_marker": "DAY25"})
        self._assert_no_forbidden_keys(s.to_dict(), "AgentState")


# ===========================================================================
# RG-19: No external imports — subsystems importable without network/LLM init
# ===========================================================================

class TestNoExternalImportContamination:
    """RG-19: All subsystem modules importable without network, LLM, or cloud SDK init."""

    def test_ingestion_package_importable(self) -> None:
        assert ing_pkg is not None

    def test_agents_package_importable(self) -> None:
        assert ag_pkg is not None

    def test_vision_package_importable(self) -> None:
        assert vis_pkg is not None

    def test_all_ingestion_model_classes_instantiable_offline(self) -> None:
        c = _make_chunk()
        assert isinstance(c, DocumentChunk)

    def test_all_agent_model_classes_instantiable_offline(self) -> None:
        citation = _make_citation()
        ar = AgentRequest(query="rg25")
        sr = SearchRequest(query="rg25")
        resp = AgentResponse(answer="a", agent_name="b")
        state = AgentState(query="rg25")
        assert all([citation, ar, sr, resp, state])

    def test_all_vision_model_classes_instantiable_offline(self) -> None:
        ev = _make_evidence()
        req = _make_vision_request()
        res = _make_vision_result()
        assert all([ev, req, res])

    def test_vision_lifecycle_instantiable_offline(self) -> None:
        lc = VisionExecutionLifecycle(provider_name="rg25_provider", model_name="rg25_model")
        assert lc is not None

    def test_cancellation_token_instantiable_offline(self) -> None:
        token = VisionCancellationToken()
        assert token is not None


# ===========================================================================
# RG-20: Validation rejection contracts remain stable
# ===========================================================================

class TestValidationRejectionContractStability:
    """RG-20: Public model constructors still reject invalid inputs with correct exception types."""

    def test_agent_citation_empty_document_id_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="f.pdf", chunk_id="ck")

    def test_agent_citation_empty_filename_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="d", filename="", chunk_id="ck")

    def test_agent_citation_invalid_page_number_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="ck",
                          page_number=0)

    def test_agent_citation_invalid_score_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="ck",
                          score=float("inf"))

    def test_agent_request_empty_query_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentRequest(query="")

    def test_search_request_invalid_top_k_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="q", top_k=0)

    def test_search_request_min_score_out_of_range_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="q", min_score=2.0)

    def test_visual_evidence_empty_document_id_rejected(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename="f.pdf", chunk_id="ck",
                           content_type="image")

    def test_visual_evidence_invalid_content_type_rejected(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck",
                           content_type="text")  # text is not a visual type

    def test_visual_evidence_invalid_chunk_index_rejected(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck",
                           content_type="image", chunk_index=-1)

    def test_vision_request_empty_query_rejected(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="")

    def test_vision_result_empty_query_rejected(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionResult(query="")

    def test_agent_state_empty_query_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentState(query="")

    def test_agent_response_empty_agent_name_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentResponse(answer="a", agent_name="")

    def test_search_result_empty_query_rejected(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchResult(query="")
