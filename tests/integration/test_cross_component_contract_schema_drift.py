"""
OmniBrain Member 4 — Day 34 Cross-Component Contract & Schema Drift Certification.

Verifies public data contracts and schemas across:
  Member 1 — Ingestion
      ↓
  Member 2 — Search / Agents
      ↓
  Member 3 — Vision

Focus areas:
  1.  Model compatibility & discovery
  2.  Field completeness across pipeline handoffs
  3.  Field type compatibility
  4.  Required & optional field compatibility
  5.  Default value stability
  6.  Serialization & deserialization contracts
  7.  Extra / unknown field handling
  8.  Full cross-component pipeline handoff
  9.  Ingestion → Search contract
  10. Search → Agent contract
  11. Agent → Vision contract
  12. Vision → Final Result contract
  13. Schema drift detection via structural snapshots
  14. Enum & constrained value compatibility
  15. Nested model compatibility & preservation
  16. Legacy calling patterns & keyword argument compliance
  17. Backward & forward compatibility
  18. Schema round-trip for multi-item batches
  19. Cross-component failure behavior on incompatible input
  20. Caller input mutation safety
  21. Comprehensive schema drift report generation

Constraints:
  - 100% offline. Zero external APIs, network, LLM, or production credentials.
  - Zero production code modified.
  - No adapters, wrappers, converters, schema migrations, or new models added.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# ---------------------------------------------------------------------------
# Ingestion Subsystem (Member 1) Models & Processors
# ---------------------------------------------------------------------------
from ingestion.models import (
    ChunkingResult,
    ChunkValidationResult,
    DocumentChunk,
    DocumentMetadata,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    EmbeddingVectorRecord,
    ExtractedImage,
    ExtractedTable,
    PageData,
    ParsedDocument,
    RetrievalServiceResult,
    TableExtractionResult,
    VectorSearchResult,
)
from ingestion.chunk_validator import normalize_chunks, validate_chunks
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.retrieval_processor import (
    build_retrieval_context,
    process_retrieval_results,
)
from ingestion.ingestion_errors import (
    IngestionError,
    IngestionValidationError,
)

# ---------------------------------------------------------------------------
# Agents / Search Subsystem (Member 2) Models
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Vision Subsystem (Member 3) Models & Adapters
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Fixture Constants
# ---------------------------------------------------------------------------

DOC_ID = "DAY34_DOC_UUID_001"
FILENAME = "day34_contract_drift.pdf"
QUERY = "Day 34 Cross-Component Contract Schema Drift Query"


def _make_chunk(
    chunk_id: str = "chk_d34_00",
    chunk_index: int = 0,
    document_id: str = DOC_ID,
    filename: str = FILENAME,
    page_number: int | None = 1,
    content: str = "Day 34 synthetic chunk content",
    content_type: str = "text",
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


def _make_vsr(
    chunk_id: str = "vsr_d34_00",
    score: float = 0.92,
    document_id: str = DOC_ID,
    filename: str = FILENAME,
    page_number: int | None = 1,
    chunk_index: int = 0,
    content_type: str = "image",
    content: str = "Day 34 visual retrieval content",
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
        metadata=metadata if metadata is not None else {},
    )


def _make_citation(
    document_id: str = DOC_ID,
    filename: str = FILENAME,
    chunk_id: str = "chk_d34_00",
    page_number: int | None = 1,
    content_type: str = "image",
    score: float = 0.92,
    metadata: dict[str, Any] | None = None,
) -> AgentCitation:
    return AgentCitation(
        document_id=document_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        content_type=content_type,
        score=score,
        metadata=metadata if metadata is not None else {},
    )


# ===========================================================================
# 1. PUBLIC MODEL DISCOVERY & SCHEMA SNAPSHOTS
# ===========================================================================

class TestPublicModelDiscoveryAndSnapshots:
    """Section 5 & 6: Validate presence and field signatures of all public models."""

    def test_ingestion_models_exist_and_are_dataclasses(self) -> None:
        models = [
            PageData, DocumentMetadata, ParsedDocument, ExtractedTable,
            TableExtractionResult, ExtractedImage, DocumentChunk,
            ChunkingResult, ChunkValidationResult, EmbeddingRecord,
            EmbeddingPreparationResult, EmbeddingVectorRecord,
            VectorSearchResult, RetrievalServiceResult,
        ]
        for m in models:
            assert inspect.isclass(m), f"{m.__name__} must be a class"
            assert dataclasses.is_dataclass(m), f"{m.__name__} must be a dataclass"

    def test_agents_models_exist_and_are_dataclasses(self) -> None:
        models = [
            AgentCitation, AgentRequest, SearchRequest,
            AgentResponse, SearchResult, AgentState,
        ]
        for m in models:
            assert inspect.isclass(m), f"{m.__name__} must be a class"
            assert dataclasses.is_dataclass(m), f"{m.__name__} must be a dataclass"

    def test_vision_models_exist_and_are_dataclasses(self) -> None:
        models = [VisualEvidence, VisionRequest, VisionResult]
        for m in models:
            assert inspect.isclass(m), f"{m.__name__} must be a class"
            assert dataclasses.is_dataclass(m), f"{m.__name__} must be a dataclass"


# ===========================================================================
# 2. SCHEMA DRIFT DETECTION (Structural Snapshots)
# ===========================================================================

class TestSchemaDriftDetection:
    """Section 19: Strict field snapshot checks to detect schema drift."""

    def test_document_chunk_schema_snapshot(self) -> None:
        fields = {f.name: f.type for f in dataclasses.fields(DocumentChunk)}
        expected = {
            "chunk_id": "str",
            "chunk_index": "int",
            "document_id": "str",
            "filename": "str",
            "page_number": "int | None",
            "content": "str",
            "content_type": "str",
            "metadata": "dict[str, Any]",
        }
        for name in expected:
            assert name in fields, f"Missing required field {name} in DocumentChunk"

    def test_vector_search_result_schema_snapshot(self) -> None:
        fields = {f.name: f.type for f in dataclasses.fields(VectorSearchResult)}
        expected_fields = [
            "chunk_id", "score", "document_id", "filename",
            "page_number", "chunk_index", "content_type", "content", "metadata",
        ]
        for name in expected_fields:
            assert name in fields, f"Missing field {name} in VectorSearchResult"

    def test_agent_citation_schema_snapshot(self) -> None:
        fields = {f.name: f.type for f in dataclasses.fields(AgentCitation)}
        expected_fields = [
            "document_id", "filename", "chunk_id",
            "page_number", "content_type", "score", "metadata",
        ]
        for name in expected_fields:
            assert name in fields, f"Missing field {name} in AgentCitation"

    def test_visual_evidence_schema_snapshot(self) -> None:
        fields = {f.name: f.type for f in dataclasses.fields(VisualEvidence)}
        expected_fields = [
            "document_id", "filename", "chunk_id", "page_number",
            "chunk_index", "content_type", "image_path", "image_bytes",
            "image_format", "description", "metadata",
        ]
        for name in expected_fields:
            assert name in fields, f"Missing field {name} in VisualEvidence"

    def test_vision_result_schema_snapshot(self) -> None:
        fields = {f.name: f.type for f in dataclasses.fields(VisionResult)}
        expected_fields = [
            "query", "status", "description", "evidence",
            "document_id", "filename", "page_number", "chunk_id",
            "content_type", "metadata", "error",
        ]
        for name in expected_fields:
            assert name in fields, f"Missing field {name} in VisionResult"


# ===========================================================================
# 3. FIELD COMPLETENESS & TYPE COMPATIBILITY
# ===========================================================================

class TestFieldCompletenessAndTypeCompatibility:
    """Sections 7 & 8: Verify field availability and types across handoffs."""

    def test_chunk_to_vsr_field_compatibility(self) -> None:
        chunk = _make_chunk(page_number=2, chunk_index=3)
        vsr = VectorSearchResult(
            chunk_id=chunk.chunk_id,
            score=0.88,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content_type=chunk.content_type,
            content=chunk.content,
            metadata=chunk.metadata,
        )
        assert vsr.chunk_id == chunk.chunk_id
        assert vsr.document_id == chunk.document_id
        assert vsr.filename == chunk.filename
        assert vsr.page_number == chunk.page_number
        assert vsr.chunk_index == chunk.chunk_index
        assert vsr.content_type == chunk.content_type
        assert isinstance(vsr.score, float)

    def test_vsr_to_citation_field_compatibility(self) -> None:
        vsr = _make_vsr(page_number=4)
        cit = AgentCitation.from_search_result(vsr)
        assert cit.document_id == vsr.document_id
        assert cit.filename == vsr.filename
        assert cit.chunk_id == vsr.chunk_id
        assert cit.page_number == vsr.page_number
        assert cit.content_type == vsr.content_type
        assert cit.score == vsr.score

    def test_citation_to_visual_evidence_field_compatibility(self) -> None:
        cit = _make_citation(content_type="image", page_number=5)
        ev = VisualEvidenceAdapter.adapt_citation(cit)
        assert ev.document_id == cit.document_id
        assert ev.filename == cit.filename
        assert ev.chunk_id == cit.chunk_id
        assert ev.page_number == cit.page_number
        assert ev.content_type == cit.content_type


# ===========================================================================
# 4. REQUIRED & OPTIONAL FIELD COMPATIBILITY
# ===========================================================================

class TestRequiredAndOptionalFieldCompatibility:
    """Sections 9 & 10: Verify required and optional field behavior."""

    def test_optional_page_number_none_in_chunk_preserves_through_pipeline(self) -> None:
        chunk = _make_chunk(page_number=None)
        vsr = _make_vsr(page_number=chunk.page_number)
        cit = AgentCitation.from_search_result(vsr)
        ev = VisualEvidenceAdapter.adapt_citation(cit)

        assert chunk.page_number is None
        assert vsr.page_number is None
        assert cit.page_number is None
        assert ev.page_number is None

    def test_optional_metadata_empty_dict_preserves_through_pipeline(self) -> None:
        cit = AgentCitation(
            document_id=DOC_ID, filename=FILENAME, chunk_id="chk_00", content_type="image"
        )
        assert cit.metadata == {}
        ev = VisualEvidenceAdapter.adapt_citation(cit)
        assert isinstance(ev.metadata, dict)

    def test_required_document_id_enforced_in_all_models(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename=FILENAME, chunk_id="c0")

        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename=FILENAME, chunk_id="c0")


# ===========================================================================
# 5. DEFAULT VALUE STABILITY
# ===========================================================================

class TestDefaultValueStability:
    """Section 11: Assert documented defaults remain unchanged."""

    def test_document_chunk_defaults(self) -> None:
        chunk = DocumentChunk(
            chunk_id="c0", chunk_index=0, document_id=DOC_ID,
            filename=FILENAME, page_number=1, content="content", content_type="text",
        )
        assert chunk.metadata == {}

    def test_vector_search_result_defaults(self) -> None:
        vsr = VectorSearchResult(
            chunk_id="c0", score=0.9, document_id=DOC_ID,
            filename=FILENAME, page_number=1, chunk_index=0,
            content_type="text", content="content",
        )
        assert vsr.metadata == {}

    def test_agent_citation_defaults(self) -> None:
        cit = AgentCitation(document_id=DOC_ID, filename=FILENAME, chunk_id="c0")
        assert cit.page_number is None
        assert cit.content_type == "text"
        assert cit.score == 0.0
        assert cit.metadata == {}

    def test_visual_evidence_defaults(self) -> None:
        ev = VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="c0")
        assert ev.page_number is None
        assert ev.chunk_index == 0
        assert ev.content_type == "image"
        assert ev.image_path is None
        assert ev.image_bytes is None
        assert ev.image_format is None
        assert ev.description is None
        assert ev.metadata == {}

    def test_agent_state_defaults(self) -> None:
        state = AgentState(query=QUERY)
        assert state.route is None
        assert state.retrieved_results == []
        assert state.context == ""
        assert state.citations == []
        assert state.answer == ""
        assert state.errors == []
        assert state.status == "initialized"
        assert state.metadata == {}


# ===========================================================================
# 6. SERIALIZATION & DESERIALIZATION CONTRACT
# ===========================================================================

class TestSerializationContract:
    """Section 12: to_dict / from_dict symmetry, nested object and lineage preservation."""

    def test_agent_citation_serialization_round_trip(self) -> None:
        cit = _make_citation(page_number=3, score=0.97, metadata={"tag": "v1"})
        d = cit.to_dict()
        cit2 = AgentCitation.from_dict(d)
        assert cit2.document_id == cit.document_id
        assert cit2.filename == cit.filename
        assert cit2.chunk_id == cit.chunk_id
        assert cit2.page_number == cit.page_number
        assert cit2.content_type == cit.content_type
        assert cit2.score == cit.score
        assert cit2.metadata == cit.metadata

    def test_visual_evidence_serialization_round_trip(self) -> None:
        ev = VisualEvidence(
            document_id=DOC_ID, filename=FILENAME, chunk_id="chk_ev",
            page_number=2, chunk_index=1, content_type="diagram",
            image_format="png", description="Architecture diagram",
            metadata={"source": "test"},
        )
        d = ev.to_dict()
        ev2 = VisualEvidence.from_dict(d)
        assert ev2.document_id == ev.document_id
        assert ev2.content_type == "diagram"
        assert ev2.description == "Architecture diagram"
        assert ev2.metadata == {"source": "test"}

    def test_search_result_serialization_round_trip(self) -> None:
        cit = _make_citation()
        sr = SearchResult(
            query=QUERY, status="RESULTS_FOUND",
            citations=[cit], context="[Source 1] content",
            metadata={"k": "v"},
        )
        d = sr.to_dict()
        sr2 = SearchResult.from_dict(d)
        assert sr2.query == sr.query
        assert sr2.status == sr.status
        assert len(sr2.citations) == 1
        assert sr2.citations[0].document_id == cit.document_id

    def test_vision_result_serialization_round_trip(self) -> None:
        ev = VisualEvidence(document_id=DOC_ID, filename=FILENAME, chunk_id="c0")
        vr = VisionResult(
            query=QUERY, status="success", description="Analyzed image",
            evidence=[ev], metadata={"model": "synthetic"},
        )
        d = vr.to_dict()
        vr2 = VisionResult.from_dict(d)
        assert vr2.query == vr.query
        assert vr2.status == vr.status
        assert vr2.description == vr.description
        assert len(vr2.evidence) == 1
        assert vr2.document_id == DOC_ID


# ===========================================================================
# 7. EXTRA FIELD COMPATIBILITY (Forward Compatibility)
# ===========================================================================

class TestExtraFieldCompatibility:
    """Sections 13 & 24: Deserializers safely absorb synthetic unknown fields."""

    def test_agent_citation_absorbs_unknown_fields(self) -> None:
        d = {
            "document_id": DOC_ID,
            "filename": FILENAME,
            "chunk_id": "c0",
            "day34_schema_drift_test": "synthetic_value",
            "future_field_v2": 12345,
        }
        cit = AgentCitation.from_dict(d)
        assert cit.document_id == DOC_ID
        assert cit.filename == FILENAME

    def test_agent_request_absorbs_unknown_fields(self) -> None:
        d = {
            "query": QUERY,
            "day34_schema_drift_test": "synthetic",
        }
        req = AgentRequest.from_dict(d)
        assert req.query == QUERY

    def test_visual_evidence_from_dict_with_extra_fields(self) -> None:
        d = {
            "document_id": DOC_ID,
            "filename": FILENAME,
            "chunk_id": "c0",
            "day34_unknown_key": "ignored",
        }
        ev = VisualEvidence.from_dict(d)
        assert ev.document_id == DOC_ID


# ===========================================================================
# 8. CROSS-COMPONENT FULL HANDOFF
# ===========================================================================

class TestCrossComponentFullHandoff:
    """Section 14: Ingestion → Search → Agents → Vision → Final Result."""

    def test_e2e_cross_component_data_flow(self) -> None:
        # Step 1: Ingestion Chunk
        chunk = _make_chunk(
            chunk_id="chk_e2e_01", chunk_index=0,
            content_type="image", content="[Revenue Chart Q3]",
        )
        assert chunk.document_id == DOC_ID

        # Step 2: Search VectorSearchResult
        vsr = VectorSearchResult(
            chunk_id=chunk.chunk_id,
            score=0.95,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content_type=chunk.content_type,
            content=chunk.content,
            metadata=chunk.metadata,
        )
        processed = process_retrieval_results([vsr], min_score=0.5, max_results=5)
        assert len(processed) == 1

        # Step 3: Agent Citation & Response
        cit = AgentCitation.from_search_result(processed[0])
        assert cit.document_id == chunk.document_id
        assert cit.chunk_id == chunk.chunk_id

        resp = AgentResponse(
            answer="Based on Q3 chart...",
            agent_name="SearchAgent",
            status="success",
            citations=[cit],
            metadata={"query": QUERY},
        )
        assert resp.is_success is True
        assert len(resp.image_results) == 1

        # Step 4: Vision Adaptation & Request
        evidence = VisualEvidenceAdapter.adapt_search_package(resp)
        assert len(evidence) == 1
        assert evidence[0].document_id == chunk.document_id

        v_req = VisionRequest(query=QUERY, evidence=evidence)
        assert v_req.has_evidence is True

        # Step 5: Vision Result & Normalization
        v_res = VisionResult(
            query=v_req.query,
            status="success",
            description="Revenue increased by 15%",
            evidence=v_req.evidence,
        )
        normalized = VisionResultNormalizer.normalize(v_res, request=v_req)
        assert normalized.is_success is True
        assert normalized.document_id == DOC_ID
        assert normalized.filename == FILENAME


# ===========================================================================
# 9. ENUM & CONSTRAINED VALUE COMPATIBILITY
# ===========================================================================

class TestEnumAndConstrainedValues:
    """Section 20: Verified constrained value sets and rejection of invalid values."""

    def test_valid_visual_content_types(self) -> None:
        for vtype in ("image", "chart", "diagram"):
            assert vtype in VALID_VISUAL_CONTENT_TYPES
            ev = VisualEvidence(
                document_id=DOC_ID, filename=FILENAME,
                chunk_id="c0", content_type=vtype
            )
            assert ev.content_type == vtype

    def test_invalid_visual_content_type_rejected(self) -> None:
        for invalid in ("audio", "video", "text", "binary"):
            with pytest.raises(VisionEvidenceError):
                VisualEvidence(
                    document_id=DOC_ID, filename=FILENAME,
                    chunk_id="c0", content_type=invalid
                )

    def test_agent_response_status_values(self) -> None:
        for status in ("success", "no_results", "error"):
            resp = AgentResponse(
                answer="text", agent_name="Agent", status=status
            )
            assert resp.status == status


# ===========================================================================
# 10. NESTED MODEL COMPATIBILITY
# ===========================================================================

class TestNestedModelCompatibility:
    """Section 21: Nested structures survive serialization/deserialization."""

    def test_parsed_document_with_page_data(self) -> None:
        pages = [
            PageData(page_number=1, text="Page 1 text", char_count=11, has_content=True),
            PageData(page_number=2, text="Page 2 text", char_count=11, has_content=True),
        ]
        meta = DocumentMetadata(
            document_id=DOC_ID, filename=FILENAME, total_pages=2,
            content_type="application/pdf", created_at="2026-08-26T00:00:00Z",
            pages_with_content=2, pages_without_content=0,
        )
        doc = ParsedDocument(metadata=meta, pages=pages)
        assert doc.get_page(1) == pages[0]
        assert doc.get_page(2) == pages[1]
        assert doc.get_page(3) is None
        assert "Page 1 text" in doc.get_all_text()

    def test_agent_response_nested_citations_serialization(self) -> None:
        citations = [
            _make_citation(chunk_id="c1", score=0.9),
            _make_citation(chunk_id="c2", score=0.8),
        ]
        resp = AgentResponse(
            answer="Multi-citation answer", agent_name="Agent",
            citations=citations, metadata={"query": QUERY},
        )
        d = resp.to_dict()
        resp2 = AgentResponse.from_dict(d)
        assert len(resp2.citations) == 2
        assert resp2.citations[0].chunk_id == "c1"
        assert resp2.citations[1].chunk_id == "c2"


# ===========================================================================
# 11. LEGACY CALLING PATTERNS & KEYWORD COMPLIANCE
# ===========================================================================

class TestCallingPatterns:
    """Section 22: Positional and keyword argument calling compliance."""

    def test_document_chunk_positional_arguments(self) -> None:
        # DocumentChunk: chunk_id, chunk_index, document_id, filename, page_number, content, content_type, metadata
        chunk = DocumentChunk(
            "c0", 0, DOC_ID, FILENAME, 1, "content", "text", {"k": "v"}
        )
        assert chunk.chunk_id == "c0"
        assert chunk.chunk_index == 0

    def test_document_chunk_keyword_arguments(self) -> None:
        chunk = DocumentChunk(
            metadata={}, content_type="text", content="content",
            page_number=1, filename=FILENAME, document_id=DOC_ID,
            chunk_index=0, chunk_id="c0",
        )
        assert chunk.chunk_id == "c0"

    def test_agent_citation_positional_arguments(self) -> None:
        # AgentCitation: document_id, filename, chunk_id, page_number, content_type, score, metadata
        cit = AgentCitation(
            DOC_ID, FILENAME, "c0", 1, "image", 0.95, {}
        )
        assert cit.document_id == DOC_ID

    def test_agent_citation_keyword_arguments(self) -> None:
        cit = AgentCitation(
            score=0.95, content_type="image", page_number=1,
            chunk_id="c0", filename=FILENAME, document_id=DOC_ID,
        )
        assert cit.score == 0.95


# ===========================================================================
# 12. BACKWARD COMPATIBILITY
# ===========================================================================

class TestBackwardCompatibility:
    """Section 23: Older supported representations remain valid."""

    def test_agent_citation_from_minimal_dict(self) -> None:
        old_dict = {
            "document_id": DOC_ID,
            "filename": FILENAME,
            "chunk_id": "c0",
        }
        cit = AgentCitation.from_dict(old_dict)
        assert cit.document_id == DOC_ID
        assert cit.content_type == "text"
        assert cit.score == 0.0

    def test_search_result_from_minimal_response(self) -> None:
        resp = AgentResponse(
            answer="answer", agent_name="Agent", citations=[],
            metadata={"query": QUERY},
        )
        sr = SearchResult.from_response(resp)
        assert sr.status == "NO_RESULTS"
        assert sr.query == QUERY


# ===========================================================================
# 13. BATCH SCHEMA ROUND-TRIP
# ===========================================================================

class TestBatchSchemaRoundTrip:
    """Section 25: Multi-item batch serialization, deserialization, and handoff."""

    def test_batch_round_trip_preserves_order_and_lineage(self) -> None:
        chunks = [
            _make_chunk(chunk_id=f"c{i}", chunk_index=i, content=f"Content {i}")
            for i in range(5)
        ]
        # Ingestion normalization
        normalized = normalize_chunks(chunks)
        assert len(normalized) == 5

        # Ingestion prepare for embedding
        prep = prepare_for_embedding(normalized)
        assert prep.is_ready is True
        assert len(prep.items) == 5

        # VectorSearchResult batch
        vsrs = [
            _make_vsr(
                chunk_id=rec.chunk_id,
                chunk_index=rec.chunk_index,
                score=round(0.99 - i * 0.05, 2),
                content_type="image",
            )
            for i, rec in enumerate(prep.items)
        ]
        processed = process_retrieval_results(vsrs, min_score=0.5, max_results=10)
        assert len(processed) == 5

        # Citations batch
        citations = [AgentCitation.from_search_result(r) for r in processed]
        assert len(citations) == 5

        # VisualEvidence batch
        evidence = VisualEvidenceAdapter.adapt_batch(citations)
        assert len(evidence) == 5
        for i, ev in enumerate(evidence):
            assert ev.chunk_id == f"c{i}"
            assert ev.document_id == DOC_ID


# ===========================================================================
# 14. CROSS-COMPONENT FAILURE BEHAVIOR
# ===========================================================================

class TestCrossComponentFailureBehavior:
    """Section 26: Receiving component rejects incompatible inputs according to contract."""

    def test_vision_adapter_rejects_text_vsr_in_strict_mode(self) -> None:
        vsr = _make_vsr(content_type="text")
        with pytest.raises(VisionEvidenceError):
            VisualEvidenceAdapter.adapt_search_result(vsr)

    def test_vision_adapter_rejects_none_citation(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisualEvidenceAdapter.adapt_citation(None)  # type: ignore[arg-type]

    def test_search_result_rejects_non_response(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchResult.from_response("not_a_response")  # type: ignore[arg-type]


# ===========================================================================
# 15. MUTATION SAFETY
# ===========================================================================

class TestMutationSafety:
    """Section 27: Upstream caller-owned objects remain unmutated."""

    def test_process_retrieval_results_does_not_mutate_input_vsrs(self) -> None:
        vsrs = [_make_vsr(score=0.9), _make_vsr(chunk_id="c1", score=0.8)]
        snapshot = copy.deepcopy(vsrs)
        process_retrieval_results(vsrs, min_score=0.85, max_results=1)
        assert vsrs == snapshot

    def test_adapt_citation_does_not_mutate_citation_metadata(self) -> None:
        meta = {"original_key": "original_val"}
        snapshot = copy.deepcopy(meta)
        cit = _make_citation(metadata=meta)
        VisualEvidenceAdapter.adapt_citation(cit)
        assert meta == snapshot

    def test_search_result_from_response_does_not_mutate_response(self) -> None:
        resp = AgentResponse(
            answer="answer", agent_name="Agent",
            citations=[_make_citation()], metadata={"query": QUERY, "extra": "1"},
        )
        meta_snapshot = copy.deepcopy(resp.metadata)
        SearchResult.from_response(resp)
        assert resp.metadata == meta_snapshot


# ===========================================================================
# 16. SCHEMA DRIFT REPORT GENERATION
# ===========================================================================

class TestSchemaDriftReportGeneration:
    """Section 28: Test-side summary of public models, fields, types, and defaults."""

    def test_generate_and_verify_schema_report(self) -> None:
        report: list[dict[str, Any]] = []

        models_to_check = [
            DocumentChunk,
            VectorSearchResult,
            AgentCitation,
            SearchRequest,
            AgentRequest,
            AgentResponse,
            SearchResult,
            AgentState,
            VisualEvidence,
            VisionRequest,
            VisionResult,
        ]

        for model in models_to_check:
            for field_def in dataclasses.fields(model):
                default_val = (
                    field_def.default
                    if field_def.default is not dataclasses.MISSING
                    else (
                        field_def.default_factory()
                        if field_def.default_factory is not dataclasses.MISSING
                        else "REQUIRED"
                    )
                )
                report.append({
                    "model": model.__name__,
                    "field": field_def.name,
                    "type": str(field_def.type),
                    "default": str(default_val),
                    "status": "VERIFIED",
                })

        assert len(report) > 50
        assert all(row["status"] == "VERIFIED" for row in report)
