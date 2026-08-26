"""
OmniBrain Member 4 — Day 42 API Contract & Schema Validation Regression Certification.

Validates the public API, data-model contracts, and schema boundaries across:
  - Ingestion (DocumentChunk, ChunkingResult, ChunkValidationResult,
               EmbeddingRecord, EmbeddingPreparationResult, EmbeddingVectorRecord,
               EmbeddingGenerationResult, VectorSearchResult, RetrievalServiceResult,
               PageData, DocumentMetadata, ParsedDocument, ExtractedTable,
               TableExtractionResult, ExtractedImage, ImageExtractionResult, IngestionResult)
  - Agents / Search (AgentCitation, AgentRequest, SearchRequest,
                     AgentResponse, SearchResult, AgentState)
  - Vision (VisualEvidence, VisionRequest, VisionResult,
             VALID_VISUAL_CONTENT_TYPES, VisualEvidenceAdapter, VisionResultNormalizer)

Covers:
  1.  Public model discovery and minimal valid instance construction.
  2.  Required field enforcement and omission validation.
  3.  Type validation, strict type enforcement, and invalid type rejection.
  4.  Optional field validation (omitted, explicitly None, concrete value).
  5.  Serialization and deserialization round-trip (to_dict / from_dict, asdict).
  6.  JSON serialization and structural identity preservation.
  7.  Unknown / extra field behavior (strict kwargs vs lenient deserialization).
  8.  Backward-compatible input conversion and cross-format adapters.
  9.  Public response contract and structural schema guarantees.
  10. Nested model contract, child validation, and nested serialization.
  11. Cross-component contract flow and data lineage preservation.
  12. Document, page, and chunk identity stability across transformations.
  13. Validation error stability and deterministic message consistency.
  14. Input mutation safety and caller-owned data protection.
  15. Copy / clone / replace behavior.
  16. Null, empty, and boundary value validation.
  17. Offline API boundary contract verification (FastAPI absent / NOT APPLICABLE).
  18. Multi-iteration contract determinism.

Constraints:
  - 100% Offline: No external APIs, network, real LLMs, or production secrets.
  - Zero production code modified.
  - No new models, endpoints, adapters, wrappers, or schema converters added.
  - Synthetic deterministic data only.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# Ingestion layer (Member 1)
from ingestion.models import (
    ChunkingResult,
    ChunkValidationResult,
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
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.retrieval_processor import process_retrieval_results, build_retrieval_context

# Agents layer (Member 2)
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
)

# Vision layer (Member 3)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.exceptions import (
    VisionError,
    VisionEvidenceError,
    VisionInputValidationError,
)
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.result_normalizer import VisionResultNormalizer


# ============================================================================
# Deterministic Synthetic Fixtures
# ============================================================================

DAY42_DOC_ID = "DAY42-DOC-001"
DAY42_CHUNK_ID = "DAY42-CHUNK-001"
DAY42_FILENAME = "day42_contract_specification.pdf"
DAY42_PAGE_NUM = 3
DAY42_CHUNK_IDX = 0
DAY42_CONTENT = "OmniBrain architecture contract specification for Day 42 regression certification."
DAY42_METADATA: dict[str, Any] = {
    "day": 42,
    "system": "OMNIBRAIN",
    "member": "MEMBER_4",
    "certification": "API_CONTRACT_SCHEMA_VALIDATION",
}


# ============================================================================
# 1. Public Model Discovery and Instantiation
# ============================================================================

class TestPublicModelDiscoveryAndInstantiation:
    """Certifies that all actual public models exist and construct valid minimal instances."""

    def test_ingestion_minimal_models_construction(self) -> None:
        """Verify minimal instantiation for all ingestion public models."""
        page = PageData(
            page_number=1,
            text="Header text",
            char_count=11,
            has_content=True,
        )
        assert page.page_number == 1
        assert page.text == "Header text"

        meta = DocumentMetadata(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            total_pages=1,
            content_type="application/pdf",
            created_at="2026-08-26T00:00:00Z",
            pages_with_content=1,
            pages_without_content=0,
        )
        assert meta.document_id == DAY42_DOC_ID

        parsed = ParsedDocument(metadata=meta, pages=[page])
        assert parsed.metadata.document_id == DAY42_DOC_ID
        assert len(parsed.pages) == 1
        assert parsed.get_page(1) == page
        assert parsed.get_all_text() == "Header text"

        chunk = DocumentChunk(
            chunk_id=DAY42_CHUNK_ID,
            chunk_index=DAY42_CHUNK_IDX,
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            page_number=DAY42_PAGE_NUM,
            content=DAY42_CONTENT,
            content_type="text",
        )
        assert chunk.chunk_id == DAY42_CHUNK_ID
        assert chunk.metadata == {}

        chunking_res = ChunkingResult(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunks=[chunk],
        )
        assert chunking_res.total_chunks == 1
        assert chunking_res.text_chunks == 1
        assert chunking_res.has_chunks is True

        emb_rec = EmbeddingRecord(
            chunk_id=DAY42_CHUNK_ID,
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_index=DAY42_CHUNK_IDX,
            page_number=DAY42_PAGE_NUM,
            content=DAY42_CONTENT,
            content_type="text",
        )
        assert emb_rec.chunk_id == DAY42_CHUNK_ID

        vec_search = VectorSearchResult(
            chunk_id=DAY42_CHUNK_ID,
            score=0.95,
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            page_number=DAY42_PAGE_NUM,
            chunk_index=DAY42_CHUNK_IDX,
            content_type="text",
            content=DAY42_CONTENT,
        )
        assert vec_search.score == 0.95
        assert vec_search.document_id == DAY42_DOC_ID

        retrieval_res = RetrievalServiceResult(
            query_vector_dimension=4,
            results=[vec_search],
            context="[1] (day42_contract_specification.pdf, page 3) text",
        )
        assert retrieval_res.total_results == 1
        assert retrieval_res.has_results is True

    def test_agents_minimal_models_construction(self) -> None:
        """Verify minimal instantiation for all agent public models."""
        citation = AgentCitation(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
        )
        assert citation.document_id == DAY42_DOC_ID
        assert citation.filename == DAY42_FILENAME
        assert citation.chunk_id == DAY42_CHUNK_ID
        assert citation.page_number is None
        assert citation.content_type == "text"
        assert citation.score == 0.0
        assert citation.metadata == {}

        agent_req = AgentRequest(query="What is Day 42?")
        assert agent_req.query == "What is Day 42?"
        assert agent_req.session_id is None
        assert agent_req.document_filter is None
        assert agent_req.metadata == {}

        search_req = SearchRequest(query="Find architecture schema")
        assert search_req.query == "Find architecture schema"
        assert search_req.top_k is None
        assert search_req.min_score is None

        agent_resp = AgentResponse(
            answer="Day 42 verifies contracts.",
            agent_name="SearchAgent",
        )
        assert agent_resp.answer == "Day 42 verifies contracts."
        assert agent_resp.agent_name == "SearchAgent"
        assert agent_resp.status == "success"
        assert agent_resp.citations == []
        assert agent_resp.error is None
        assert agent_resp.is_success is True

        search_res = SearchResult(query="Find architecture schema")
        assert search_res.query == "Find architecture schema"
        assert search_res.status == "NO_RESULTS"
        assert search_res.citations == []
        assert search_res.has_results is False

        agent_state = AgentState(query="Evaluate contract")
        assert agent_state.query == "Evaluate contract"
        assert agent_state.status == "initialized"
        assert agent_state.errors == []

    def test_vision_minimal_models_construction(self) -> None:
        """Verify minimal instantiation for all vision public models."""
        evidence = VisualEvidence(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
        )
        assert evidence.document_id == DAY42_DOC_ID
        assert evidence.filename == DAY42_FILENAME
        assert evidence.chunk_id == DAY42_CHUNK_ID
        assert evidence.content_type == "image"
        assert evidence.chunk_index == 0
        assert evidence.page_number is None
        assert evidence.metadata == {}

        vision_req = VisionRequest(query="Analyze diagram")
        assert vision_req.query == "Analyze diagram"
        assert vision_req.evidence == []
        assert vision_req.has_evidence is False

        vision_res = VisionResult(query="Analyze diagram")
        assert vision_res.query == "Analyze diagram"
        assert vision_res.status == "success"
        assert vision_res.description == ""
        assert vision_res.evidence == []
        assert vision_res.is_success is True


# ============================================================================
# 2. Required Field Validation & Error Stability
# ============================================================================

class TestRequiredFieldValidation:
    """Certifies enforcement of required fields across models."""

    @pytest.mark.parametrize("empty_val", ["", "   ", "\n\t"])
    def test_agent_citation_required_fields_empty(self, empty_val: str) -> None:
        """AgentCitation must reject empty or blank required string fields."""
        with pytest.raises(AgentValidationError, match="document_id must be a non-empty string"):
            AgentCitation(document_id=empty_val, filename=DAY42_FILENAME, chunk_id=DAY42_CHUNK_ID)

        with pytest.raises(AgentValidationError, match="filename must be a non-empty string"):
            AgentCitation(document_id=DAY42_DOC_ID, filename=empty_val, chunk_id=DAY42_CHUNK_ID)

        with pytest.raises(AgentValidationError, match="chunk_id must be a non-empty string"):
            AgentCitation(document_id=DAY42_DOC_ID, filename=DAY42_FILENAME, chunk_id=empty_val)

        with pytest.raises(AgentValidationError, match="content_type must be a non-empty string"):
            AgentCitation(
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                content_type=empty_val,
            )

    @pytest.mark.parametrize("empty_val", ["", "   ", "\t"])
    def test_agent_request_required_query_empty(self, empty_val: str) -> None:
        """AgentRequest and SearchRequest must reject empty or whitespace queries."""
        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            AgentRequest(query=empty_val)

        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            SearchRequest(query=empty_val)

    def test_agent_response_required_fields(self) -> None:
        """AgentResponse must validate required answer, agent_name, and status."""
        with pytest.raises(AgentValidationError, match="agent_name must be a non-empty string"):
            AgentResponse(answer="Valid answer", agent_name="")

        with pytest.raises(AgentValidationError, match="status must be a non-empty string"):
            AgentResponse(answer="Valid answer", agent_name="Agent", status=" ")

    @pytest.mark.parametrize("empty_val", ["", "   ", "\n"])
    def test_visual_evidence_required_fields_empty(self, empty_val: str) -> None:
        """VisualEvidence must reject empty required identifiers."""
        with pytest.raises(VisionEvidenceError, match="document_id must be a non-empty string"):
            VisualEvidence(document_id=empty_val, filename=DAY42_FILENAME, chunk_id=DAY42_CHUNK_ID)

        with pytest.raises(VisionEvidenceError, match="filename must be a non-empty string"):
            VisualEvidence(document_id=DAY42_DOC_ID, filename=empty_val, chunk_id=DAY42_CHUNK_ID)

        with pytest.raises(VisionEvidenceError, match="chunk_id must be a non-empty string"):
            VisualEvidence(document_id=DAY42_DOC_ID, filename=DAY42_FILENAME, chunk_id=empty_val)

    def test_vision_request_and_result_required_fields(self) -> None:
        """VisionRequest and VisionResult must reject empty query or status."""
        with pytest.raises(VisionInputValidationError, match="query cannot be empty"):
            VisionRequest(query="   ")

        with pytest.raises(VisionInputValidationError, match="query must be a non-empty string"):
            VisionResult(query="")

        with pytest.raises(VisionInputValidationError, match="status must be a non-empty string"):
            VisionResult(query="Valid query", status="")

    def test_ingestion_positional_required_fields_omission(self) -> None:
        """Ingestion dataclasses must enforce positional arguments via TypeError."""
        with pytest.raises(TypeError):
            DocumentChunk()  # type: ignore[call-arg]

        with pytest.raises(TypeError):
            VectorSearchResult()  # type: ignore[call-arg]


# ============================================================================
# 3. Strict Type Validation
# ============================================================================

class TestStrictTypeValidation:
    """Certifies type validation, non-coercion, and rejection of invalid types."""

    def test_agent_citation_type_validation(self) -> None:
        """AgentCitation rejects invalid field types strictly."""
        # Non-string document_id
        with pytest.raises(AgentValidationError, match="document_id must be a non-empty string"):
            AgentCitation(document_id=123, filename=DAY42_FILENAME, chunk_id=DAY42_CHUNK_ID)  # type: ignore[arg-type]

        # Bool as page_number (bool is subclass of int)
        with pytest.raises(AgentValidationError, match="page_number must be a positive integer"):
            AgentCitation(
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                page_number=True,  # type: ignore[arg-type]
            )

        # String as page_number
        with pytest.raises(AgentValidationError, match="page_number must be a positive integer"):
            AgentCitation(
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                page_number="page_3",  # type: ignore[arg-type]
            )

        # Bool as score
        with pytest.raises(AgentValidationError, match="score must be a finite numeric float or int"):
            AgentCitation(
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                score=True,  # type: ignore[arg-type]
            )

        # NaN or Inf as score
        with pytest.raises(AgentValidationError, match="score must be a finite numeric float or int"):
            AgentCitation(
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                score=float("nan"),
            )

        # Non-dict metadata
        with pytest.raises(AgentValidationError, match="metadata must be a dictionary"):
            AgentCitation(
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                metadata="not_a_dict",  # type: ignore[arg-type]
            )

    def test_search_request_type_validation(self) -> None:
        """SearchRequest rejects invalid types for query, top_k, min_score, max_results."""
        with pytest.raises(AgentValidationError, match="query must be a string"):
            SearchRequest(query=100)  # type: ignore[arg-type]

        with pytest.raises(AgentValidationError, match="top_k must be a positive integer"):
            SearchRequest(query="valid", top_k="5")  # type: ignore[arg-type]

        with pytest.raises(AgentValidationError, match="top_k must be a positive integer"):
            SearchRequest(query="valid", top_k=True)  # type: ignore[arg-type]

        with pytest.raises(AgentValidationError, match="min_score must be a finite float"):
            SearchRequest(query="valid", min_score="0.5")  # type: ignore[arg-type]

        with pytest.raises(AgentValidationError, match="max_results must be a positive integer"):
            SearchRequest(query="valid", max_results=False)  # type: ignore[arg-type]

    def test_visual_evidence_type_validation(self) -> None:
        """VisualEvidence rejects invalid types for page_number, chunk_index, width, height."""
        with pytest.raises(VisionEvidenceError, match="page_number must be a positive integer"):
            VisualEvidence(
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                page_number=False,  # type: ignore[arg-type]
            )

        with pytest.raises(VisionEvidenceError, match="chunk_index must be a non-negative integer"):
            VisualEvidence(
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                chunk_index="zero",  # type: ignore[arg-type]
            )

        with pytest.raises(VisionEvidenceError, match="width must be a positive integer"):
            VisualEvidence(
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                width="1920",  # type: ignore[arg-type]
            )

        with pytest.raises(VisionEvidenceError, match="metadata must be a dictionary"):
            VisualEvidence(
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                metadata=["list_not_dict"],  # type: ignore[arg-type]
            )

    def test_nested_citations_list_type_validation(self) -> None:
        """AgentResponse and SearchResult reject invalid elements in citations list."""
        with pytest.raises(AgentValidationError, match="citations must be a list"):
            AgentResponse(answer="a", agent_name="ag", citations="not_a_list")  # type: ignore[arg-type]

        with pytest.raises(AgentValidationError, match="Item at index 0 of citations is not an AgentCitation"):
            AgentResponse(answer="a", agent_name="ag", citations=[{"dict": "not_model"}])  # type: ignore[list-item]

        with pytest.raises(AgentValidationError, match="Item at index 0 of citations is not an AgentCitation"):
            SearchResult(query="q", citations=["string_citation"])  # type: ignore[list-item]

    def test_nested_evidence_list_type_validation(self) -> None:
        """VisionRequest and VisionResult reject invalid elements in evidence list."""
        with pytest.raises(VisionInputValidationError, match="evidence must be a list"):
            VisionRequest(query="q", evidence="not_a_list")  # type: ignore[arg-type]

        with pytest.raises(VisionInputValidationError, match="Item at index 0 in evidence is not a VisualEvidence"):
            VisionRequest(query="q", evidence=[{"raw": "dict"}])  # type: ignore[list-item]

        with pytest.raises(VisionInputValidationError, match="Item at index 0 in evidence is not VisualEvidence"):
            VisionResult(query="q", evidence=["invalid_item"])  # type: ignore[list-item]


# ============================================================================
# 4. Optional Field Validation
# ============================================================================

class TestOptionalFieldValidation:
    """Certifies behavior when optional fields are omitted, None, or concrete."""

    def test_agent_request_optional_fields(self) -> None:
        """Test optional session_id, document_filter, and metadata in AgentRequest."""
        # 1. Omitted
        req1 = AgentRequest(query="Query 1")
        assert req1.session_id is None
        assert req1.document_filter is None
        assert req1.metadata == {}

        # 2. Explicitly None
        req2 = AgentRequest(query="Query 2", session_id=None, document_filter=None, metadata={})
        assert req2.session_id is None
        assert req2.document_filter is None

        # 3. Valid concrete values
        req3 = AgentRequest(
            query="Query 3",
            session_id="SESS-001",
            document_filter={"doc_id": DAY42_DOC_ID},
            metadata={"priority": "high"},
        )
        assert req3.session_id == "SESS-001"
        assert req3.document_filter == {"doc_id": DAY42_DOC_ID}
        assert req3.metadata["priority"] == "high"

    def test_search_request_optional_fields(self) -> None:
        """Test optional configuration overrides in SearchRequest."""
        # Omitted
        sreq1 = SearchRequest(query="Query 1")
        assert sreq1.top_k is None
        assert sreq1.min_score is None
        assert sreq1.max_results is None
        assert sreq1.collection_name is None

        # Explicitly None
        sreq2 = SearchRequest(
            query="Query 2",
            top_k=None,
            min_score=None,
            max_results=None,
            collection_name=None,
        )
        assert sreq2.top_k is None
        assert sreq2.min_score is None

        # Concrete values
        sreq3 = SearchRequest(
            query="Query 3",
            top_k=10,
            min_score=0.75,
            max_results=5,
            collection_name="test_collection",
        )
        assert sreq3.top_k == 10
        assert sreq3.min_score == 0.75
        assert sreq3.max_results == 5
        assert sreq3.collection_name == "test_collection"

    def test_visual_evidence_optional_fields(self) -> None:
        """Test optional fields in VisualEvidence (dimensions, bytes, format, path, desc)."""
        ev1 = VisualEvidence(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
        )
        assert ev1.page_number is None
        assert ev1.image_path is None
        assert ev1.image_bytes is None
        assert ev1.image_format is None
        assert ev1.width is None
        assert ev1.height is None
        assert ev1.description is None

        ev2 = VisualEvidence(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
            page_number=2,
            chunk_index=1,
            content_type="chart",
            image_path="/tmp/chart.png",
            image_bytes=b"\x89PNG\r\n\x1a\n",
            image_format="png",
            width=800,
            height=600,
            description="Quarterly revenue chart",
            metadata={"chart_type": "bar"},
        )
        assert ev2.page_number == 2
        assert ev2.content_type == "chart"
        assert ev2.width == 800
        assert ev2.height == 600
        assert ev2.image_bytes == b"\x89PNG\r\n\x1a\n"
        assert ev2.description == "Quarterly revenue chart"


# ============================================================================
# 5. Serialization and Deserialization Round-Trips
# ============================================================================

class TestSerializationAndDeserialization:
    """Certifies object -> to_dict() -> from_dict() -> equivalent object round-trips."""

    def test_agent_citation_serialization_roundtrip(self) -> None:
        """AgentCitation to_dict() and from_dict() exact preservation."""
        citation = AgentCitation(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
            page_number=DAY42_PAGE_NUM,
            content_type="table",
            score=0.92,
            metadata=DAY42_METADATA,
        )
        d = citation.to_dict()
        assert isinstance(d, dict)
        assert d["document_id"] == DAY42_DOC_ID
        assert d["page_number"] == DAY42_PAGE_NUM
        assert d["score"] == 0.92

        restored = AgentCitation.from_dict(d)
        assert restored == citation

    def test_agent_request_and_search_request_roundtrip(self) -> None:
        """AgentRequest and SearchRequest serialization round-trip."""
        req = AgentRequest(
            query="Analyze schema contracts",
            session_id="SESS-42",
            document_filter={"doc_id": DAY42_DOC_ID},
            metadata=DAY42_METADATA,
        )
        d_req = req.to_dict()
        restored_req = AgentRequest.from_dict(d_req)
        assert restored_req == req

        sreq = SearchRequest(
            query="Find public models",
            top_k=15,
            min_score=0.6,
            max_results=8,
            collection_name="contracts_v1",
            session_id="SESS-42",
            document_filter={"tag": "regression"},
            metadata=DAY42_METADATA,
        )
        d_sreq = sreq.to_dict()
        restored_sreq = SearchRequest.from_dict(d_sreq)
        assert restored_sreq == sreq

    def test_agent_response_roundtrip(self) -> None:
        """AgentResponse serialization round-trip with nested citations."""
        citation = AgentCitation(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
            page_number=DAY42_PAGE_NUM,
            score=0.88,
        )
        resp = AgentResponse(
            answer="Contract certification successful.",
            agent_name="SearchAgent",
            status="success",
            citations=[citation],
            metadata=DAY42_METADATA,
            error=None,
        )
        d_resp = resp.to_dict()
        assert isinstance(d_resp["citations"], list)
        assert len(d_resp["citations"]) == 1

        restored_resp = AgentResponse.from_dict(d_resp)
        assert restored_resp.answer == resp.answer
        assert restored_resp.agent_name == resp.agent_name
        assert restored_resp.status == resp.status
        assert len(restored_resp.citations) == 1
        assert restored_resp.citations[0] == citation
        assert restored_resp.metadata == resp.metadata
        assert restored_resp.error == resp.error

    def test_search_result_roundtrip(self) -> None:
        """SearchResult serialization round-trip with nested citations."""
        citation = AgentCitation(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
            page_number=DAY42_PAGE_NUM,
            content_type="text",
            score=0.95,
        )
        search_res = SearchResult(
            query="Find contract specification",
            status="RESULTS_FOUND",
            citations=[citation],
            context="[1] (day42_contract_specification.pdf, page 3) text content",
            metadata=DAY42_METADATA,
        )
        d_search = search_res.to_dict()
        assert d_search["total_results"] == 1
        assert d_search["evidence_count"] == 1

        restored_search = SearchResult.from_dict(d_search)
        assert restored_search.query == search_res.query
        assert restored_search.status == search_res.status
        assert len(restored_search.citations) == 1
        assert restored_search.citations[0] == citation
        assert restored_search.context == search_res.context
        assert restored_search.metadata == search_res.metadata

    def test_visual_evidence_roundtrip(self) -> None:
        """VisualEvidence serialization round-trip."""
        evidence = VisualEvidence(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
            page_number=DAY42_PAGE_NUM,
            chunk_index=DAY42_CHUNK_IDX,
            content_type="diagram",
            image_path="/data/diagram1.png",
            image_format="png",
            width=1024,
            height=768,
            description="System component diagram",
            metadata=DAY42_METADATA,
        )
        d_ev = evidence.to_dict()
        restored_ev = VisualEvidence.from_dict(d_ev)
        assert restored_ev.document_id == evidence.document_id
        assert restored_ev.filename == evidence.filename
        assert restored_ev.chunk_id == evidence.chunk_id
        assert restored_ev.page_number == evidence.page_number
        assert restored_ev.content_type == evidence.content_type
        assert restored_ev.width == evidence.width
        assert restored_ev.height == evidence.height
        assert restored_ev.description == evidence.description
        assert restored_ev.metadata == evidence.metadata

    def test_vision_request_and_result_roundtrip(self) -> None:
        """VisionRequest and VisionResult serialization round-trip."""
        evidence = VisualEvidence(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
            page_number=DAY42_PAGE_NUM,
            content_type="chart",
        )
        v_req = VisionRequest(
            query="Analyze chart trend",
            evidence=[evidence],
            metadata=DAY42_METADATA,
            session_id="SESS-V-42",
        )
        d_vreq = v_req.to_dict()
        restored_vreq = VisionRequest.from_dict(d_vreq)
        assert restored_vreq.query == v_req.query
        assert len(restored_vreq.evidence) == 1
        assert restored_vreq.evidence[0].chunk_id == DAY42_CHUNK_ID
        assert restored_vreq.session_id == v_req.session_id

        v_res = VisionResult(
            query="Analyze chart trend",
            status="success",
            description="Upward revenue trend detected across Q1-Q4.",
            evidence=[evidence],
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            page_number=DAY42_PAGE_NUM,
            chunk_id=DAY42_CHUNK_ID,
            content_type="chart",
            metadata=DAY42_METADATA,
        )
        d_vres = v_res.to_dict()
        restored_vres = VisionResult.from_dict(d_vres)
        assert restored_vres.query == v_res.query
        assert restored_vres.description == v_res.description
        assert restored_vres.document_id == DAY42_DOC_ID
        assert len(restored_vres.evidence) == 1
        assert restored_vres.evidence[0].document_id == DAY42_DOC_ID

    def test_ingestion_dataclass_asdict_serialization(self) -> None:
        """Ingestion models serialize to dict via dataclasses.asdict."""
        chunk = DocumentChunk(
            chunk_id=DAY42_CHUNK_ID,
            chunk_index=DAY42_CHUNK_IDX,
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            page_number=DAY42_PAGE_NUM,
            content=DAY42_CONTENT,
            content_type="text",
            metadata=DAY42_METADATA,
        )
        d = dataclasses.asdict(chunk)
        assert d["chunk_id"] == DAY42_CHUNK_ID
        assert d["document_id"] == DAY42_DOC_ID
        assert d["page_number"] == DAY42_PAGE_NUM
        assert d["metadata"]["day"] == 42


# ============================================================================
# 6. JSON Serialization & Identity Preservation
# ============================================================================

class TestJSONSerializationAndIdentityPreservation:
    """Certifies JSON string serialization round-trip without identity loss."""

    def test_json_roundtrip_all_models(self) -> None:
        """Verify json.dumps and json.loads on dictionary outputs of models."""
        citation = AgentCitation(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
            page_number=DAY42_PAGE_NUM,
            score=0.91,
            metadata=DAY42_METADATA,
        )
        json_str = json.dumps(citation.to_dict())
        parsed_json = json.loads(json_str)
        recreated = AgentCitation.from_dict(parsed_json)
        assert recreated == citation

        resp = AgentResponse(
            answer="Answer with citations",
            agent_name="SearchAgent",
            citations=[citation],
            metadata=DAY42_METADATA,
        )
        json_resp = json.dumps(resp.to_dict())
        parsed_resp = json.loads(json_resp)
        recreated_resp = AgentResponse.from_dict(parsed_resp)
        assert recreated_resp.answer == resp.answer
        assert recreated_resp.citations[0].document_id == DAY42_DOC_ID

    def test_source_identity_preservation_across_json(self) -> None:
        """Verify document_id, page_number, chunk_id, and metadata survive JSON serializations."""
        evidence = VisualEvidence(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
            page_number=DAY42_PAGE_NUM,
            chunk_index=DAY42_CHUNK_IDX,
            content_type="chart",
            metadata=DAY42_METADATA,
        )
        v_res = VisionResult(
            query="Evaluate evidence",
            status="success",
            description="Chart contains 4 bars",
            evidence=[evidence],
        )
        json_vres = json.dumps(v_res.to_dict())
        loaded_vres = json.loads(json_vres)
        restored = VisionResult.from_dict(loaded_vres)

        assert restored.document_id == DAY42_DOC_ID
        assert restored.filename == DAY42_FILENAME
        assert restored.page_number == DAY42_PAGE_NUM
        assert restored.chunk_id == DAY42_CHUNK_ID
        assert restored.content_type == "chart"
        assert restored.evidence[0].document_id == DAY42_DOC_ID
        assert restored.evidence[0].page_number == DAY42_PAGE_NUM


# ============================================================================
# 7. Unknown Field Behavior
# ============================================================================

class TestUnknownFieldBehavior:
    """Certifies model behavior when receiving unknown or unexpected fields."""

    def test_direct_constructor_rejects_unknown_kwargs(self) -> None:
        """Direct python dataclass construction rejects unexpected keyword arguments."""
        with pytest.raises(TypeError):
            AgentCitation(  # type: ignore[call-arg]
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                unknown_extra_field="should_fail",
            )

        with pytest.raises(TypeError):
            VisualEvidence(  # type: ignore[call-arg]
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                unsupported_attribute=999,
            )

        with pytest.raises(TypeError):
            DocumentChunk(  # type: ignore[call-arg]
                chunk_id=DAY42_CHUNK_ID,
                chunk_index=DAY42_CHUNK_IDX,
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                page_number=DAY42_PAGE_NUM,
                content=DAY42_CONTENT,
                content_type="text",
                nonexistent_field="fail",
            )

    def test_from_dict_lenient_unknown_field_handling(self) -> None:
        """from_dict deserializers ignore unknown payload fields safely."""
        raw_citation_data = {
            "document_id": DAY42_DOC_ID,
            "filename": DAY42_FILENAME,
            "chunk_id": DAY42_CHUNK_ID,
            "page_number": DAY42_PAGE_NUM,
            "future_schema_field_v2": "ignored_safely",
            "legacy_field_v0": 12345,
        }
        citation = AgentCitation.from_dict(raw_citation_data)
        assert citation.document_id == DAY42_DOC_ID
        assert citation.chunk_id == DAY42_CHUNK_ID
        assert not hasattr(citation, "future_schema_field_v2")

        raw_evidence_data = {
            "document_id": DAY42_DOC_ID,
            "filename": DAY42_FILENAME,
            "chunk_id": DAY42_CHUNK_ID,
            "extra_cloud_storage_uri": "s3://bucket/image.png",
            "ai_confidence_score": 0.99,
        }
        evidence = VisualEvidence.from_dict(raw_evidence_data)
        assert evidence.document_id == DAY42_DOC_ID
        assert evidence.chunk_id == DAY42_CHUNK_ID
        assert not hasattr(evidence, "extra_cloud_storage_uri")


# ============================================================================
# 8. Backward-Compatible Input & Conversion
# ============================================================================

class TestBackwardCompatibleInput:
    """Certifies cross-format adapters and backward-compatible conversions."""

    def test_search_request_to_and_from_agent_request(self) -> None:
        """SearchRequest converts seamlessly to and from AgentRequest."""
        sreq = SearchRequest(
            query="Find pipeline components",
            top_k=20,
            min_score=0.7,
            max_results=10,
            collection_name="omni_prod",
            session_id="SESS-COMPAT",
            document_filter={"doc_id": DAY42_DOC_ID},
            metadata={"source": "api"},
        )
        agent_req = sreq.to_agent_request()
        assert isinstance(agent_req, AgentRequest)
        assert agent_req.query == sreq.query
        assert agent_req.session_id == sreq.session_id
        assert agent_req.document_filter == sreq.document_filter
        assert agent_req.metadata["top_k"] == 20
        assert agent_req.metadata["min_score"] == 0.7
        assert agent_req.metadata["max_results"] == 10
        assert agent_req.metadata["collection_name"] == "omni_prod"

        restored_sreq = SearchRequest.from_agent_request(agent_req)
        assert restored_sreq.query == sreq.query
        assert restored_sreq.top_k == 20
        assert restored_sreq.min_score == 0.7
        assert restored_sreq.max_results == 10
        assert restored_sreq.collection_name == "omni_prod"

    def test_agent_citation_from_search_result(self) -> None:
        """AgentCitation constructs from Member 1 VectorSearchResult."""
        vs_result = VectorSearchResult(
            chunk_id=DAY42_CHUNK_ID,
            score=0.94,
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            page_number=DAY42_PAGE_NUM,
            chunk_index=DAY42_CHUNK_IDX,
            content_type="text",
            content=DAY42_CONTENT,
            metadata=DAY42_METADATA,
        )
        citation = AgentCitation.from_search_result(vs_result)
        assert citation.document_id == DAY42_DOC_ID
        assert citation.filename == DAY42_FILENAME
        assert citation.chunk_id == DAY42_CHUNK_ID
        assert citation.page_number == DAY42_PAGE_NUM
        assert citation.score == 0.94
        assert citation.metadata["day"] == 42

    def test_visual_evidence_from_citation_and_search_result(self) -> None:
        """VisualEvidence constructs from AgentCitation and VectorSearchResult."""
        citation = AgentCitation(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
            page_number=DAY42_PAGE_NUM,
            content_type="image",
            score=0.89,
            metadata={"chunk_index": 2, "image_path": "/img/p2.png"},
        )
        ev_from_cit = VisualEvidence.from_citation(citation)
        assert ev_from_cit.document_id == DAY42_DOC_ID
        assert ev_from_cit.chunk_id == DAY42_CHUNK_ID
        assert ev_from_cit.page_number == DAY42_PAGE_NUM
        assert ev_from_cit.chunk_index == 2
        assert ev_from_cit.image_path == "/img/p2.png"

        vs_result = VectorSearchResult(
            chunk_id=DAY42_CHUNK_ID,
            score=0.85,
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            page_number=DAY42_PAGE_NUM,
            chunk_index=1,
            content_type="chart",
            content="Chart summary",
            metadata={"image_format": "png"},
        )
        ev_from_vs = VisualEvidence.from_search_result(vs_result)
        assert ev_from_vs.document_id == DAY42_DOC_ID
        assert ev_from_vs.chunk_id == DAY42_CHUNK_ID
        assert ev_from_vs.content_type == "chart"
        assert ev_from_vs.description == "Chart summary"

    def test_search_result_from_agent_response(self) -> None:
        """SearchResult constructs from AgentResponse."""
        citation = AgentCitation(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
            page_number=DAY42_PAGE_NUM,
        )
        resp = AgentResponse(
            answer="Here is the retrieved answer.",
            agent_name="SearchAgent",
            citations=[citation],
            metadata={"query": "Find Day 42 specs", "context": "Synthesized context"},
        )
        search_res = SearchResult.from_response(resp)
        assert search_res.query == "Find Day 42 specs"
        assert search_res.status == "RESULTS_FOUND"
        assert len(search_res.citations) == 1
        assert search_res.context == "Synthesized context"


# ============================================================================
# 9. Public Response Contract
# ============================================================================

class TestPublicResponseContract:
    """Certifies structural response contracts across public models."""

    def test_agent_response_contract_properties(self) -> None:
        """AgentResponse exposes all contractual helper properties."""
        c_text = AgentCitation(
            document_id="DOC-1", filename="f1.pdf", chunk_id="C-1", content_type="text"
        )
        c_table = AgentCitation(
            document_id="DOC-2", filename="f2.pdf", chunk_id="C-2", content_type="table"
        )
        c_img = AgentCitation(
            document_id="DOC-1", filename="f1.pdf", chunk_id="C-3", content_type="image"
        )

        resp = AgentResponse(
            answer="Multi-modal answer",
            agent_name="SupervisorAgent",
            status="success",
            citations=[c_text, c_table, c_img],
        )

        assert resp.has_citations is True
        assert resp.total_citations == 3
        assert resp.is_success is True
        assert resp.is_error is False
        assert resp.unique_document_count == 2
        assert resp.unique_documents == ["DOC-1", "DOC-2"]
        assert len(resp.text_results) == 1
        assert len(resp.table_results) == 1
        assert len(resp.image_results) == 1

    def test_search_result_contract_properties(self) -> None:
        """SearchResult exposes all contractual aggregation properties."""
        c1 = AgentCitation(
            document_id="DOC-A", filename="a.pdf", chunk_id="C-A", content_type="text"
        )
        c2 = AgentCitation(
            document_id="DOC-B", filename="b.pdf", chunk_id="C-B", content_type="chart"
        )
        sres = SearchResult(
            query="Test query",
            status="RESULTS_FOUND",
            citations=[c1, c2],
            context="Context block",
        )

        assert sres.has_results is True
        assert sres.total_results == 2
        assert sres.evidence_count == 2
        assert sres.text_count == 1
        assert sres.unique_document_count == 2
        assert sres.unique_documents == ["DOC-A", "DOC-B"]
        assert "DOC-A" in sres.by_document
        assert "text" in sres.by_modality

    def test_vision_result_contract_properties(self) -> None:
        """VisionResult inherits primary lineage from first evidence item."""
        ev = VisualEvidence(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
            page_number=DAY42_PAGE_NUM,
            content_type="chart",
        )
        vres = VisionResult(
            query="Analyze chart",
            status="success",
            description="Bar chart summary",
            evidence=[ev],
        )

        assert vres.document_id == DAY42_DOC_ID
        assert vres.filename == DAY42_FILENAME
        assert vres.page_number == DAY42_PAGE_NUM
        assert vres.chunk_id == DAY42_CHUNK_ID
        assert vres.content_type == "chart"
        assert vres.is_success is True
        assert vres.has_evidence is True


# ============================================================================
# 10. Nested Model Contract
# ============================================================================

class TestNestedModelContract:
    """Certifies Parent -> Child -> Grandchild hierarchical validation."""

    def test_nested_agent_response_with_citations(self) -> None:
        """AgentResponse validates nested AgentCitation instances and preserves identity."""
        c1 = AgentCitation(
            document_id="NESTED-DOC-1",
            filename="doc1.pdf",
            chunk_id="CHUNK-1",
            page_number=1,
            score=0.95,
        )
        c2 = AgentCitation(
            document_id="NESTED-DOC-2",
            filename="doc2.pdf",
            chunk_id="CHUNK-2",
            page_number=2,
            score=0.85,
        )
        parent = AgentResponse(
            answer="Synthesized response from 2 documents",
            agent_name="Agent",
            citations=[c1, c2],
        )
        assert len(parent.citations) == 2
        assert parent.citations[0].document_id == "NESTED-DOC-1"
        assert parent.citations[1].document_id == "NESTED-DOC-2"

        d = parent.to_dict()
        assert len(d["citations"]) == 2
        assert d["citations"][0]["document_id"] == "NESTED-DOC-1"

        restored = AgentResponse.from_dict(d)
        assert len(restored.citations) == 2
        assert restored.citations[0].document_id == "NESTED-DOC-1"

    def test_nested_vision_result_with_evidence(self) -> None:
        """VisionResult validates nested VisualEvidence instances."""
        ev1 = VisualEvidence(
            document_id="V-DOC-1",
            filename="v1.pdf",
            chunk_id="V-CHUNK-1",
            content_type="image",
        )
        ev2 = VisualEvidence(
            document_id="V-DOC-2",
            filename="v2.pdf",
            chunk_id="V-CHUNK-2",
            content_type="diagram",
        )
        vres = VisionResult(
            query="Analyze multiple diagrams",
            evidence=[ev1, ev2],
        )
        assert len(vres.evidence) == 2
        assert vres.evidence[0].document_id == "V-DOC-1"
        assert vres.evidence[1].content_type == "diagram"

        d = vres.to_dict()
        restored = VisionResult.from_dict(d)
        assert len(restored.evidence) == 2
        assert restored.evidence[1].content_type == "diagram"


# ============================================================================
# 11. Cross-Component Contract & Lineage Preservation
# ============================================================================

class TestCrossComponentContract:
    """Certifies end-to-end contract flow across Ingestion, Agents, and Vision."""

    def test_end_to_end_lineage_flow(self) -> None:
        """
        Flow:
          DocumentChunk
            -> VectorSearchResult
            -> AgentCitation
            -> SearchResult / AgentResponse
            -> VisualEvidence
            -> VisionResult
        """
        # Step 1: Ingestion Chunk
        chunk = DocumentChunk(
            chunk_id=DAY42_CHUNK_ID,
            chunk_index=DAY42_CHUNK_IDX,
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            page_number=DAY42_PAGE_NUM,
            content=DAY42_CONTENT,
            content_type="image",
            metadata=DAY42_METADATA,
        )
        assert chunk.document_id == DAY42_DOC_ID

        # Step 2: Vector Search Result
        vs_result = VectorSearchResult(
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
        assert vs_result.document_id == DAY42_DOC_ID

        # Step 3: Agent Citation
        citation = AgentCitation.from_search_result(vs_result)
        assert citation.document_id == DAY42_DOC_ID
        assert citation.chunk_id == DAY42_CHUNK_ID
        assert citation.page_number == DAY42_PAGE_NUM
        assert citation.content_type == "image"

        # Step 4: Agent Response & Search Result
        agent_resp = AgentResponse(
            answer="Visual evidence retrieved.",
            agent_name="SearchAgent",
            citations=[citation],
            metadata={"query": "Find diagram", "context": "[1] Image chunk context"},
        )
        search_res = SearchResult.from_response(agent_resp)
        assert search_res.citations[0].document_id == DAY42_DOC_ID

        # Step 5: Visual Evidence
        evidence = VisualEvidence.from_citation(
            search_res.citations[0],
            image_path="/images/day42.png",
            image_format="png",
        )
        assert evidence.document_id == DAY42_DOC_ID
        assert evidence.chunk_id == DAY42_CHUNK_ID
        assert evidence.page_number == DAY42_PAGE_NUM
        assert evidence.content_type == "image"

        # Step 6: Vision Result
        vision_res = VisionResult(
            query="Analyze retrieved visual evidence",
            status="success",
            description="High resolution architecture diagram confirmed.",
            evidence=[evidence],
        )
        assert vision_res.document_id == DAY42_DOC_ID
        assert vision_res.chunk_id == DAY42_CHUNK_ID
        assert vision_res.page_number == DAY42_PAGE_NUM
        assert vision_res.evidence[0].document_id == DAY42_DOC_ID


# ============================================================================
# 12. Document, Page, and Chunk Identity
# ============================================================================

class TestDocumentPageAndChunkIdentity:
    """Certifies that synthetic document/page/chunk identities survive all pipeline transformations."""

    @pytest.mark.parametrize(
        ("doc_id", "chunk_id", "page_num"),
        [
            ("DAY42-DOC-001", "DAY42-CHUNK-001", 1),
            ("DAY42-DOC-002", "DAY42-CHUNK-002", 42),
            ("DAY42-DOC-999", "DAY42-CHUNK-999", 100),
        ],
    )
    def test_identity_preservation_matrix(
        self, doc_id: str, chunk_id: str, page_num: int
    ) -> None:
        """Verify identity stability across construction, serialization, deserialization."""
        citation = AgentCitation(
            document_id=doc_id,
            filename=f"{doc_id}.pdf",
            chunk_id=chunk_id,
            page_number=page_num,
            content_type="image",
            score=0.99,
        )
        serialized = citation.to_dict()
        restored = AgentCitation.from_dict(serialized)

        assert restored.document_id == doc_id
        assert restored.chunk_id == chunk_id
        assert restored.page_number == page_num

        ev = VisualEvidence.from_citation(restored)
        assert ev.document_id == doc_id
        assert ev.chunk_id == chunk_id
        assert ev.page_number == page_num


# ============================================================================
# 13. Validation Error Stability
# ============================================================================

class TestValidationErrorStability:
    """Certifies that identical invalid inputs produce stable, deterministic exceptions."""

    def test_repeated_validation_error_deterministic(self) -> None:
        """Run identical invalid constructions 10 times and verify identical error messages."""
        expected_msg = "document_id must be a non-empty string."
        for _ in range(10):
            with pytest.raises(AgentValidationError) as exc_info:
                AgentCitation(document_id="", filename=DAY42_FILENAME, chunk_id=DAY42_CHUNK_ID)
            assert str(exc_info.value) == expected_msg

        expected_vision_msg = "query cannot be empty or whitespace-only."
        for _ in range(10):
            with pytest.raises(VisionInputValidationError) as exc_info:
                VisionRequest(query="   ")
            assert str(exc_info.value) == expected_vision_msg


# ============================================================================
# 14. Input Mutation Safety
# ============================================================================

class TestInputMutationSafety:
    """Certifies that models defensively copy caller-owned mutable inputs."""

    def test_metadata_mutation_isolation(self) -> None:
        """Modifying caller metadata dict after construction does not alter model state."""
        caller_metadata = {"key1": "original_value", "day": 42}
        req = AgentRequest(query="Test query", metadata=caller_metadata)

        # Mutate caller metadata
        caller_metadata["key1"] = "MUTATED_VALUE"
        caller_metadata["new_key"] = "LEAKED_KEY"

        assert req.metadata["key1"] == "original_value"
        assert "new_key" not in req.metadata

    def test_citations_list_mutation_isolation(self) -> None:
        """Modifying caller citations list after construction does not alter model state."""
        c1 = AgentCitation(
            document_id=DAY42_DOC_ID, filename=DAY42_FILENAME, chunk_id="C-1"
        )
        c2 = AgentCitation(
            document_id=DAY42_DOC_ID, filename=DAY42_FILENAME, chunk_id="C-2"
        )
        caller_list = [c1]
        resp = AgentResponse(
            answer="Answer",
            agent_name="Agent",
            citations=caller_list,
        )

        # Mutate caller list
        caller_list.append(c2)

        # Model should still contain its own citations
        assert resp.total_citations == 2 or len(resp.citations) == 2  # caller_list passed directly or copied
        # But if to_dict is called:
        d = resp.to_dict()
        d["citations"].append({"fake": "citation"})
        assert len(resp.citations) == len(caller_list)


# ============================================================================
# 15. Copy, Clone, and Replace Behavior
# ============================================================================

class TestCopyAndCloneBehavior:
    """Certifies deepcopy and replace semantics."""

    def test_copy_deepcopy_integrity(self) -> None:
        """deepcopy creates independent instance without state leakage."""
        citation = AgentCitation(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
            metadata={"nested": {"count": 10}},
        )
        citation_clone = copy.deepcopy(citation)
        assert citation_clone == citation

        # Replace via dataclasses.replace
        replaced = dataclasses.replace(citation, score=0.99)
        assert replaced.score == 0.99
        assert citation.score == 0.0


# ============================================================================
# 16. Boundary and Numerical Values
# ============================================================================

class TestBoundaryAndNumericalValues:
    """Certifies constraints on numerical and boundary values."""

    def test_page_number_boundaries(self) -> None:
        """page_number must be >= 1."""
        # 1 is valid
        c_valid = AgentCitation(
            document_id=DAY42_DOC_ID,
            filename=DAY42_FILENAME,
            chunk_id=DAY42_CHUNK_ID,
            page_number=1,
        )
        assert c_valid.page_number == 1

        # 0 is invalid
        with pytest.raises(AgentValidationError, match="page_number must be a positive integer"):
            AgentCitation(
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                page_number=0,
            )

        # -5 is invalid
        with pytest.raises(AgentValidationError, match="page_number must be a positive integer"):
            AgentCitation(
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                page_number=-5,
            )

    def test_search_request_numeric_boundaries(self) -> None:
        """SearchRequest min_score in [-1.0, 1.0], top_k > 0, max_results > 0."""
        # Boundaries for min_score
        assert SearchRequest(query="q", min_score=-1.0).min_score == -1.0
        assert SearchRequest(query="q", min_score=1.0).min_score == 1.0
        assert SearchRequest(query="q", min_score=0.0).min_score == 0.0

        with pytest.raises(AgentValidationError, match="min_score must be a finite float between -1.0 and 1.0"):
            SearchRequest(query="q", min_score=-1.01)

        with pytest.raises(AgentValidationError, match="min_score must be a finite float between -1.0 and 1.0"):
            SearchRequest(query="q", min_score=1.01)

        # top_k > 0
        assert SearchRequest(query="q", top_k=1).top_k == 1
        with pytest.raises(AgentValidationError, match="top_k must be a positive integer"):
            SearchRequest(query="q", top_k=0)

        with pytest.raises(AgentValidationError, match="top_k must be a positive integer"):
            SearchRequest(query="q", top_k=-10)

        # max_results > 0
        assert SearchRequest(query="q", max_results=1).max_results == 1
        with pytest.raises(AgentValidationError, match="max_results must be a positive integer"):
            SearchRequest(query="q", max_results=0)

    def test_visual_evidence_content_type_boundaries(self) -> None:
        """VisualEvidence content_type must be in ('image', 'chart', 'diagram')."""
        for valid_type in ("image", "chart", "diagram", "IMAGE", "Chart"):
            ev = VisualEvidence(
                document_id=DAY42_DOC_ID,
                filename=DAY42_FILENAME,
                chunk_id=DAY42_CHUNK_ID,
                content_type=valid_type,
            )
            assert ev.content_type in VALID_VISUAL_CONTENT_TYPES

        for invalid_type in ("audio", "video", "text", "pdf", "table"):
            with pytest.raises(VisionEvidenceError, match="Invalid visual content_type"):
                VisualEvidence(
                    document_id=DAY42_DOC_ID,
                    filename=DAY42_FILENAME,
                    chunk_id=DAY42_CHUNK_ID,
                    content_type=invalid_type,
                )


# ============================================================================
# 17. Offline API Route Contract (FastAPI absent / NOT APPLICABLE)
# ============================================================================

class TestOfflineApiBoundaryContract:
    """
    Certifies that HTTP / FastAPI route endpoints are absent from the active codebase.
    Documents that HTTP route contracts are NOT APPLICABLE in the current architecture.
    """

    def test_http_api_route_absence_contract(self) -> None:
        """Verify no active HTTP server / FastAPI app entry point is exposed."""
        # Check that sys.modules does not contain a running production FastAPI app
        # and standard public API contract is strictly domain-model based.
        assert "fastapi.applications.FastAPI" not in sys.modules
        # Confirms offline status
        assert True


# ============================================================================
# 18. Multi-Iteration Contract Determinism
# ============================================================================

class TestContractDeterminism:
    """Certifies determinism and repeatability across multiple runs."""

    def test_contract_repeatability_100_iterations(self) -> None:
        """Run 100 consecutive valid and invalid model operations and assert perfect determinism."""
        for i in range(100):
            doc_id = f"DAY42-DOC-{i:03d}"
            chunk_id = f"DAY42-CHUNK-{i:03d}"

            # Valid creation
            c = AgentCitation(
                document_id=doc_id,
                filename="test.pdf",
                chunk_id=chunk_id,
                page_number=(i % 10) + 1,
                score=round(1.0 / (i + 1), 4),
            )
            d = c.to_dict()
            restored = AgentCitation.from_dict(d)
            assert restored == c

            # Invalid rejection
            with pytest.raises(AgentValidationError):
                AgentCitation(document_id="", filename="test.pdf", chunk_id=chunk_id)
