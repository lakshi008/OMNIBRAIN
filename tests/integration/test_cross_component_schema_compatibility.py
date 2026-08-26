"""
OmniBrain Member 4 -- Day 18 Cross-Component Data Contract & Schema Compatibility Tests.

Verifies that data structures exchanged between Member 1, Member 2, Member 3,
and downstream components remain strictly schema-compatible.

Pipeline Handoff Focus:
  Ingestion (Member 1)
      ↓
  Search / Retrieval (Member 2)
      ↓
  Vision (Member 3)
      ↓
  Supervisor / Downstream Consumers

Concern areas:
 1. Ingestion -> Search schema compatibility (DocumentChunk -> VectorSearchResult -> SearchAgent inputs)
 2. Search -> Vision schema compatibility (VectorSearchResult -> AgentCitation -> VisualEvidence)
 3. Vision -> Downstream schema compatibility (VisionRequest -> VisionResult -> AgentResponse / AgentState)
 4. Required field compatibility across all cross-component models
 5. Optional field compatibility (omitted, explicit None, empty metadata/collections, defaults)
 6. Field type compatibility (str, int, float, dict, list)
 7. Identifier preservation (document_id, chunk_id, filename across the full chain)
 8. Metadata preservation across schema transformations
 9. Content type preservation (text, image, chart, diagram)
10. Citation compatibility (AgentCitation construction, conversion, lineage locking)
11. Serialization compatibility (to_dict -> from_dict -> to_dict round-trip)
12. Unknown field compatibility in deserialization
13. Multi-item compatibility (multi-chunk, multi-evidence, multi-citation)
14. Cross-document isolation (Document A -> A objects, Document B -> B objects)
15. Failure schema compatibility (standard error fields, exception categories)
16. Backward compatibility (legacy model construction and from_dict patterns)
17. State isolation (A, B, A again: second A completely independent of B)

Constraints:
 - 100% Offline: No external APIs, network, real LLMs, or production secrets.
 - Zero production code modified.
 - Only observable behavior guaranteed by existing public contracts tested.
"""

from __future__ import annotations

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
    DocumentMetadata,
    PageData,
    ParsedDocument,
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
from vision.result_normalizer import VisionExecutionTrace, VisionResultNormalizer


# ============================================================================
# Helpers & Fixtures
# ============================================================================

def _chunk(
    chunk_id: str = "chk-schema-001",
    document_id: str = "doc-schema-001",
    filename: str = "schema_doc.pdf",
    page_number: int | None = 2,
    chunk_index: int = 0,
    content: str = "Quarterly financial chart and breakdown.",
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
        metadata=metadata if metadata is not None else {"schema_test": "DAY18", "source": "TEST_DOCUMENT"},
    )


def _vsr(
    chunk_id: str = "chk-schema-001",
    score: float = 0.95,
    document_id: str = "doc-schema-001",
    filename: str = "schema_doc.pdf",
    page_number: int | None = 2,
    chunk_index: int = 0,
    content: str = "Quarterly financial chart and breakdown.",
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
        metadata=metadata if metadata is not None else {"schema_test": "DAY18", "source": "TEST_DOCUMENT"},
    )


def _ev(
    chunk_id: str = "chk-schema-001",
    document_id: str = "doc-schema-001",
    filename: str = "schema_doc.pdf",
    page_number: int | None = 2,
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
        metadata=metadata if metadata is not None else {"schema_test": "DAY18", "source": "TEST_DOCUMENT"},
    )


# ============================================================================
# 1. INGESTION → SEARCH SCHEMA COMPATIBILITY
# ============================================================================


class TestIngestionToSearchSchema:
    """Verifies that Member 1 ingestion data structures seamlessly integrate with Member 2 search."""

    def test_document_chunk_to_vector_search_result_compatibility(self) -> None:
        chunk = _chunk(
            chunk_id="chk-i2s-01",
            document_id="doc-i2s-01",
            filename="quarterly.pdf",
            page_number=5,
            chunk_index=3,
            content="Revenue breakdown by division.",
            content_type="table",
            metadata={"department": "finance", "year": 2026},
        )

        vsr = VectorSearchResult(
            chunk_id=chunk.chunk_id,
            score=0.92,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content_type=chunk.content_type,
            content=chunk.content,
            metadata=chunk.metadata,
        )

        # Ingestion -> Search: All fields preserved
        assert vsr.chunk_id == chunk.chunk_id
        assert vsr.document_id == chunk.document_id
        assert vsr.filename == chunk.filename
        assert vsr.page_number == chunk.page_number
        assert vsr.chunk_index == chunk.chunk_index
        assert vsr.content_type == chunk.content_type
        assert vsr.content == chunk.content
        assert vsr.metadata == chunk.metadata
        assert vsr.score == 0.92

    def test_agent_citation_from_search_result(self) -> None:
        vsr = _vsr(
            chunk_id="chk-i2s-02",
            score=0.88,
            document_id="doc-i2s-02",
            filename="tech_spec.pdf",
            page_number=12,
            content_type="text",
            metadata={"spec_version": "2.1"},
        )
        citation = AgentCitation.from_search_result(vsr)

        assert citation.document_id == vsr.document_id
        assert citation.filename == vsr.filename
        assert citation.chunk_id == vsr.chunk_id
        assert citation.page_number == vsr.page_number
        assert citation.content_type == vsr.content_type
        assert citation.score == vsr.score
        assert citation.metadata["spec_version"] == "2.1"


# ============================================================================
# 2. SEARCH → VISION SCHEMA COMPATIBILITY
# ============================================================================


class TestSearchToVisionSchema:
    """Verifies that Member 2 Search results and citations safely construct Member 3 VisualEvidence."""

    def test_visual_evidence_from_search_result(self) -> None:
        vsr = _vsr(
            chunk_id="chk-s2v-01",
            score=0.94,
            document_id="doc-s2v-01",
            filename="architecture.pdf",
            page_number=4,
            chunk_index=1,
            content_type="diagram",
            metadata={"diagram_type": "component_flow", "image_path": "/path/to/diagram.png"},
        )
        ev = VisualEvidence.from_search_result(vsr)

        assert ev.document_id == vsr.document_id
        assert ev.filename == vsr.filename
        assert ev.chunk_id == vsr.chunk_id
        assert ev.page_number == vsr.page_number
        assert ev.chunk_index == vsr.chunk_index
        assert ev.content_type == "diagram"
        assert ev.image_path == "/path/to/diagram.png"
        assert ev.description == vsr.content
        assert ev.metadata["diagram_type"] == "component_flow"

    def test_visual_evidence_from_citation(self) -> None:
        cit = AgentCitation(
            document_id="doc-s2v-02",
            filename="charts.pdf",
            chunk_id="chk-s2v-02",
            page_number=7,
            content_type="chart",
            score=0.89,
            metadata={"chart_title": "Q3 Growth", "chunk_index": 2},
        )
        ev = VisualEvidence.from_citation(cit)

        assert ev.document_id == cit.document_id
        assert ev.filename == cit.filename
        assert ev.chunk_id == cit.chunk_id
        assert ev.page_number == cit.page_number
        assert ev.chunk_index == 2
        assert ev.content_type == "chart"
        assert ev.metadata["chart_title"] == "Q3 Growth"


# ============================================================================
# 3. VISION → DOWNSTREAM SCHEMA COMPATIBILITY
# ============================================================================


class TestVisionToDownstreamSchema:
    """Verifies that Member 3 Vision output is compatible with downstream agent workflows."""

    def test_vision_result_compatibility_with_agent_state_and_response(self) -> None:
        ev = _ev(
            chunk_id="chk-v2d-01",
            document_id="doc-v2d-01",
            filename="report.pdf",
            page_number=3,
            content_type="chart",
        )
        vision_result = VisionResult(
            query="Analyze Q3 chart",
            status="success",
            description="Q3 chart indicates 15% growth year-over-year.",
            evidence=[ev],
            metadata={"confidence": "high"},
        )

        # Downstream: AgentState incorporates vision result
        state = AgentState(query=vision_result.query)
        cit = AgentCitation(
            document_id=vision_result.document_id,
            filename=vision_result.filename,
            chunk_id=vision_result.chunk_id,
            page_number=vision_result.page_number,
            content_type=vision_result.content_type,
            score=1.0,
            metadata=vision_result.metadata,
        )
        state.add_citation(cit)

        # Downstream: AgentResponse encapsulates answer and citations
        response = AgentResponse(
            answer=vision_result.description,
            agent_name="SupervisorAgent",
            citations=state.citations,
            status="success",
            metadata={"vision_status": vision_result.status, "query": state.query},
        )

        assert response.metadata["query"] == vision_result.query
        assert response.answer == vision_result.description
        assert len(response.citations) == 1
        assert response.citations[0].document_id == "doc-v2d-01"
        assert response.citations[0].chunk_id == "chk-v2d-01"
        assert response.citations[0].page_number == 3
        assert response.citations[0].content_type == "chart"
        assert response.status == "success"


# ============================================================================
# 4. REQUIRED FIELD COMPATIBILITY
# ============================================================================


class TestRequiredFieldCompatibility:
    """Verifies that all required fields are validated and present on public cross-component models."""

    def test_document_chunk_required_fields(self) -> None:
        chunk = DocumentChunk(
            chunk_id="c1",
            chunk_index=0,
            document_id="d1",
            filename="f.pdf",
            page_number=1,
            content="content",
            content_type="text",
        )
        assert hasattr(chunk, "chunk_id")
        assert hasattr(chunk, "document_id")
        assert hasattr(chunk, "filename")
        assert hasattr(chunk, "content")
        assert hasattr(chunk, "content_type")

    def test_vector_search_result_required_fields(self) -> None:
        vsr = VectorSearchResult(
            chunk_id="c1",
            score=0.9,
            document_id="d1",
            filename="f.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="content",
        )
        assert hasattr(vsr, "chunk_id")
        assert hasattr(vsr, "score")
        assert hasattr(vsr, "document_id")
        assert hasattr(vsr, "filename")
        assert hasattr(vsr, "content_type")

    def test_agent_citation_required_fields(self) -> None:
        cit = AgentCitation(document_id="d1", filename="f.pdf", chunk_id="c1")
        assert hasattr(cit, "document_id")
        assert hasattr(cit, "filename")
        assert hasattr(cit, "chunk_id")

    def test_visual_evidence_required_fields(self) -> None:
        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", content_type="image")
        assert hasattr(ev, "document_id")
        assert hasattr(ev, "filename")
        assert hasattr(ev, "chunk_id")
        assert hasattr(ev, "content_type")

    def test_vision_result_required_fields(self) -> None:
        res = VisionResult(query="q", status="success", description="desc")
        assert hasattr(res, "query")
        assert hasattr(res, "status")
        assert hasattr(res, "description")


# ============================================================================
# 5. OPTIONAL FIELD COMPATIBILITY
# ============================================================================


class TestOptionalFieldCompatibility:
    """Verifies models handle omitted optional fields, None, and empty containers across handoffs."""

    def test_end_to_end_optional_field_omission_survives_handoff(self) -> None:
        # Minimal chunk: page_number is None, metadata is empty
        chunk = DocumentChunk(
            chunk_id="chk-opt-01",
            chunk_index=0,
            document_id="doc-opt-01",
            filename="opt.pdf",
            page_number=None,
            content="minimal content",
            content_type="image",
            metadata={},
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
            metadata={},
        )
        cit = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_citation(cit)
        v_res = VisionResult(
            query="Optional field query",
            status="success",
            description="Optional fields verified.",
            evidence=[ev],
        )

        assert cit.page_number is None
        assert cit.metadata == {}
        assert ev.page_number is None
        assert ev.image_path is None
        assert ev.image_bytes is None
        assert v_res.page_number is None
        assert v_res.error is None


# ============================================================================
# 6. FIELD TYPE COMPATIBILITY
# ============================================================================


class TestFieldTypeCompatibility:
    """Verifies that public field types conform to expected types across handoffs."""

    def test_public_model_field_types(self) -> None:
        chunk = _chunk()
        vsr = _vsr()
        cit = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_citation(cit)
        vres = VisionResult(query="type test", status="success", description="type ok.", evidence=[ev])

        assert isinstance(chunk.chunk_id, str)
        assert isinstance(chunk.document_id, str)
        assert isinstance(chunk.page_number, int)
        assert isinstance(chunk.metadata, dict)

        assert isinstance(vsr.score, float)
        assert isinstance(cit.score, float)
        assert isinstance(ev.chunk_index, int)
        assert isinstance(vres.has_evidence, bool)
        assert isinstance(vres.evidence, list)


# ============================================================================
# 7. IDENTIFIER PRESERVATION
# ============================================================================


class TestIdentifierPreservation:
    """Verifies Document ID, Chunk ID, and Filename survive unchanged across the entire pipeline."""

    def test_full_pipeline_identifier_preservation(self) -> None:
        DOC_ID = "DOC-CORP-SEC-2026-X99"
        CHUNK_ID = "CHK-SEC-0042-UUID"
        FILENAME = "sec_filing_q3_2026.pdf"

        chunk = _chunk(chunk_id=CHUNK_ID, document_id=DOC_ID, filename=FILENAME, page_number=14)
        vsr = _vsr(chunk_id=chunk.chunk_id, document_id=chunk.document_id, filename=chunk.filename, page_number=chunk.page_number)
        cit = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_citation(cit)
        v_res = VisionResult(query="SEC analysis", status="success", description="SEC filing analyzed.", evidence=[ev])

        # Verification at every hop
        assert chunk.document_id == DOC_ID
        assert vsr.document_id == DOC_ID
        assert cit.document_id == DOC_ID
        assert ev.document_id == DOC_ID
        assert v_res.document_id == DOC_ID

        assert chunk.chunk_id == CHUNK_ID
        assert vsr.chunk_id == CHUNK_ID
        assert cit.chunk_id == CHUNK_ID
        assert ev.chunk_id == CHUNK_ID
        assert v_res.chunk_id == CHUNK_ID

        assert chunk.filename == FILENAME
        assert vsr.filename == FILENAME
        assert cit.filename == FILENAME
        assert ev.filename == FILENAME
        assert v_res.filename == FILENAME


# ============================================================================
# 8. METADATA PRESERVATION
# ============================================================================


class TestMetadataPreservation:
    """Verifies metadata dictionaries survive schema handoffs and transformations intact."""

    def test_metadata_dict_preservation(self) -> None:
        test_meta = {
            "schema_test": "DAY18",
            "source": "TEST_DOCUMENT",
            "audit_id": "AUD-9988",
            "nested": {"level": 2, "verified": True},
        }

        chunk = _chunk(metadata=test_meta)
        vsr = _vsr(metadata=chunk.metadata)
        cit = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_citation(cit)
        v_res = VisionResult(query="Meta test", status="success", description="Meta verified.", evidence=[ev])

        assert cit.metadata["schema_test"] == "DAY18"
        assert cit.metadata["source"] == "TEST_DOCUMENT"
        assert cit.metadata["audit_id"] == "AUD-9988"
        assert cit.metadata["nested"]["level"] == 2

        assert ev.metadata["schema_test"] == "DAY18"
        assert ev.metadata["source"] == "TEST_DOCUMENT"
        assert ev.metadata["audit_id"] == "AUD-9988"
        assert ev.metadata["nested"]["verified"] is True

        assert v_res.evidence[0].metadata["schema_test"] == "DAY18"


# ============================================================================
# 9. CONTENT TYPE PRESERVATION
# ============================================================================


class TestContentTypePreservation:
    """Verifies supported content types (text, table, image, chart, diagram) remain consistent."""

    def test_all_supported_visual_content_types_preserved(self) -> None:
        for ct in sorted(VALID_VISUAL_CONTENT_TYPES):
            vsr = _vsr(content_type=ct)
            cit = AgentCitation.from_search_result(vsr)
            ev = VisualEvidence.from_citation(cit)
            res = VisionResult(query=f"Analyze {ct}", status="success", description=f"{ct} ok.", evidence=[ev])

            assert cit.content_type == ct
            assert ev.content_type == ct
            assert res.content_type == ct


# ============================================================================
# 10. CITATION COMPATIBILITY
# ============================================================================


class TestCitationCompatibility:
    """Verifies AgentCitation preserves provenance and is interoperable across search and agents."""

    def test_citation_to_dict_and_from_dict_compatibility(self) -> None:
        orig = AgentCitation(
            document_id="doc-cit-01",
            filename="cit.pdf",
            chunk_id="chk-cit-01",
            page_number=4,
            content_type="table",
            score=0.91,
            metadata={"schema_test": "DAY18"},
        )
        data = orig.to_dict()
        restored = AgentCitation.from_dict(data)

        assert restored.document_id == orig.document_id
        assert restored.filename == orig.filename
        assert restored.chunk_id == orig.chunk_id
        assert restored.page_number == orig.page_number
        assert restored.content_type == orig.content_type
        assert restored.score == orig.score
        assert restored.metadata == orig.metadata


# ============================================================================
# 11. SERIALIZATION COMPATIBILITY
# ============================================================================


class TestSerializationCompatibility:
    """Verifies to_dict -> from_dict -> to_dict round-trip retains full contract compatibility."""

    def test_full_pipeline_round_trip_serialization(self) -> None:
        ev = _ev(
            chunk_id="chk-ser-01",
            document_id="doc-ser-01",
            filename="ser.pdf",
            page_number=6,
            content_type="chart",
            metadata={"schema_test": "DAY18", "format": "vector"},
        )
        orig_res = VisionResult(
            query="Serialization pipeline test",
            status="success",
            description="Serialization compatibility verified.",
            evidence=[ev],
            metadata={"pipeline": "ingest->search->vision"},
        )

        d1 = orig_res.to_dict()
        restored_res = VisionResult.from_dict(d1)
        d2 = restored_res.to_dict()

        assert d1["document_id"] == d2["document_id"]
        assert d1["filename"] == d2["filename"]
        assert d1["chunk_id"] == d2["chunk_id"]
        assert d1["page_number"] == d2["page_number"]
        assert d1["status"] == d2["status"]
        assert d1["description"] == d2["description"]
        assert d1["evidence"][0]["metadata"]["schema_test"] == "DAY18"

    def test_search_request_and_response_serialization(self) -> None:
        req = SearchRequest(
            query="schema search query",
            top_k=15,
            min_score=0.7,
            max_results=5,
            collection_name="enterprise_docs",
            session_id="sess-schema-001",
            document_filter={"doc_id": "doc-001"},
            metadata={"role": "tester"},
        )
        data = req.to_dict()
        restored_req = SearchRequest.from_dict(data)

        assert restored_req.query == req.query
        assert restored_req.top_k == req.top_k
        assert restored_req.min_score == req.min_score
        assert restored_req.collection_name == req.collection_name
        assert restored_req.session_id == req.session_id


# ============================================================================
# 12. UNKNOWN FIELD COMPATIBILITY
# ============================================================================


class TestUnknownFieldCompatibility:
    """Verifies from_dict methods tolerate harmless synthetic unknown fields."""

    def test_unknown_fields_in_deserialization_ignored_safely(self) -> None:
        data = {
            "document_id": "doc-unk-01",
            "filename": "unk.pdf",
            "chunk_id": "chk-unk-01",
            "page_number": 1,
            "content_type": "text",
            "score": 0.85,
            "metadata": {"schema_test": "DAY18"},
            "future_schema_field_v2": "IGNORABLE_PAYLOAD",
            "unknown_extra_metric": 42.0,
        }
        cit = AgentCitation.from_dict(data)
        assert cit.document_id == "doc-unk-01"
        assert cit.score == 0.85


# ============================================================================
# 13. MULTI-ITEM COMPATIBILITY
# ============================================================================


class TestMultiItemCompatibility:
    """Verifies that collections of chunks, citations, and evidence items remain intact without dropping items."""

    def test_multi_chunk_to_multi_evidence_handoff(self) -> None:
        ITEM_COUNT = 8
        chunks = [
            _chunk(
                chunk_id=f"chk-multi-{i}",
                document_id="doc-multi-01",
                filename="multi.pdf",
                page_number=i + 1,
                chunk_index=i,
                content_type="chart" if i % 2 == 0 else "diagram",
                metadata={"item_index": i},
            )
            for i in range(ITEM_COUNT)
        ]

        vsrs = [
            VectorSearchResult(
                chunk_id=c.chunk_id,
                score=0.80 + (i * 0.02),
                document_id=c.document_id,
                filename=c.filename,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                content_type=c.content_type,
                content=c.content,
                metadata=c.metadata,
            )
            for i, c in enumerate(chunks)
        ]

        citations = [AgentCitation.from_search_result(vsr) for vsr in vsrs]
        evidences = [VisualEvidence.from_search_result(vsr) for vsr in vsrs]
        req = VisionRequest(query="Analyze all charts and diagrams", evidence=evidences)
        result = VisionResult(query=req.query, status="success", description="All items analyzed.", evidence=req.evidence)

        assert len(result.evidence) == ITEM_COUNT
        for i, ev_item in enumerate(result.evidence):
            assert ev_item.chunk_id == f"chk-multi-{i}"
            assert ev_item.page_number == i + 1
            assert ev_item.chunk_index == i
            assert ev_item.metadata["item_index"] == i


# ============================================================================
# 14. CROSS-DOCUMENT ISOLATION
# ============================================================================


class TestCrossDocumentIsolation:
    """Verifies Document A data stays with Document A, Document B data stays with Document B."""

    def test_cross_document_isolation_in_schema_conversion(self) -> None:
        doc_a_chunk = _chunk(
            chunk_id="chk-docA-01",
            document_id="doc-A-alpha",
            filename="document_a.pdf",
            page_number=1,
            metadata={"doc_tag": "ALPHA"},
        )
        doc_b_chunk = _chunk(
            chunk_id="chk-docB-01",
            document_id="doc-B-beta",
            filename="document_b.pdf",
            page_number=2,
            metadata={"doc_tag": "BETA"},
        )

        vsr_a = _vsr(
            chunk_id=doc_a_chunk.chunk_id,
            document_id=doc_a_chunk.document_id,
            filename=doc_a_chunk.filename,
            metadata=doc_a_chunk.metadata,
        )
        vsr_b = _vsr(
            chunk_id=doc_b_chunk.chunk_id,
            document_id=doc_b_chunk.document_id,
            filename=doc_b_chunk.filename,
            metadata=doc_b_chunk.metadata,
        )

        ev_a = VisualEvidence.from_search_result(vsr_a)
        ev_b = VisualEvidence.from_search_result(vsr_b)

        res_a = VisionResult(query="Query A", status="success", description="A done.", evidence=[ev_a])
        res_b = VisionResult(query="Query B", status="success", description="B done.", evidence=[ev_b])

        assert res_a.document_id == "doc-A-alpha"
        assert res_a.filename == "document_a.pdf"
        assert res_a.evidence[0].metadata["doc_tag"] == "ALPHA"
        assert "BETA" not in str(res_a.to_dict())

        assert res_b.document_id == "doc-B-beta"
        assert res_b.filename == "document_b.pdf"
        assert res_b.evidence[0].metadata["doc_tag"] == "BETA"
        assert "ALPHA" not in str(res_b.to_dict())


# ============================================================================
# 15. FAILURE SCHEMA COMPATIBILITY
# ============================================================================


class TestFailureSchemaCompatibility:
    """Verifies that error schemas remain consistent across subsystem boundaries."""

    def test_error_vision_result_schema_compatibility(self) -> None:
        err_res = VisionResult(
            query="Failing vision request",
            status="error",
            description="",
            error="Model provider returned non-retryable 400 Bad Request.",
            metadata={"schema_test": "DAY18_ERR"},
        )

        assert err_res.status == "error"
        assert err_res.error == "Model provider returned non-retryable 400 Bad Request."
        assert err_res.has_evidence is False
        assert err_res.evidence == []

        # Serialization of error result
        data = err_res.to_dict()
        restored = VisionResult.from_dict(data)
        assert restored.status == "error"
        assert restored.error == err_res.error

    def test_validation_exceptions_are_distinct_and_typed(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="f.pdf", chunk_id="c")

        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename="f.pdf", chunk_id="c", content_type="image")

        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="")


# ============================================================================
# 16. BACKWARD COMPATIBILITY
# ============================================================================


class TestBackwardCompatibility:
    """Verifies legacy model construction and dictionary formats continue to function."""

    def test_agent_citation_legacy_construction_positional(self) -> None:
        cit = AgentCitation("doc-leg-01", "leg.pdf", "chk-leg-01")
        assert cit.document_id == "doc-leg-01"
        assert cit.filename == "leg.pdf"
        assert cit.chunk_id == "chk-leg-01"
        assert cit.page_number is None
        assert cit.score == 0.0

    def test_visual_evidence_legacy_from_dict_minimal(self) -> None:
        data = {
            "document_id": "doc-leg-02",
            "filename": "leg2.pdf",
            "chunk_id": "chk-leg-02",
            "content_type": "image",
        }
        ev = VisualEvidence.from_dict(data)
        assert ev.document_id == "doc-leg-02"
        assert ev.content_type == "image"
        assert ev.page_number is None

    def test_search_request_to_agent_request_conversion(self) -> None:
        sr = SearchRequest(query="legacy conversion query", top_k=10, min_score=0.8)
        ar = sr.to_agent_request()
        assert isinstance(ar, AgentRequest)
        assert ar.query == "legacy conversion query"
        assert ar.metadata.get("top_k") == 10
        assert ar.metadata.get("min_score") == 0.8


# ============================================================================
# 17. STATE ISOLATION (A, B, A again)
# ============================================================================


class TestStateIsolation:
    """Verifies running Request A, Request B, Request A again leaves the second A completely pristine."""

    def test_schema_handoff_state_isolation(self) -> None:
        def _execute_a() -> dict[str, Any]:
            chunk_a = _chunk(
                chunk_id="chk-isoA-01",
                document_id="doc-isoA-01",
                filename="isoA.pdf",
                metadata={"workflow": "ISO_A"},
            )
            vsr_a = _vsr(
                chunk_id=chunk_a.chunk_id,
                document_id=chunk_a.document_id,
                filename=chunk_a.filename,
                metadata=chunk_a.metadata,
            )
            cit_a = AgentCitation.from_search_result(vsr_a)
            ev_a = VisualEvidence.from_citation(cit_a)
            res_a = VisionResult(
                query="Isolation query A",
                status="success",
                description="Workflow A complete.",
                evidence=[ev_a],
            )
            return res_a.to_dict()

        def _execute_b() -> dict[str, Any]:
            chunk_b = _chunk(
                chunk_id="chk-isoB-02",
                document_id="doc-isoB-02",
                filename="isoB.pdf",
                metadata={"workflow": "ISO_B"},
            )
            vsr_b = _vsr(
                chunk_id=chunk_b.chunk_id,
                document_id=chunk_b.document_id,
                filename=chunk_b.filename,
                metadata=chunk_b.metadata,
            )
            cit_b = AgentCitation.from_search_result(vsr_b)
            ev_b = VisualEvidence.from_citation(cit_b)
            res_b = VisionResult(
                query="Isolation query B",
                status="success",
                description="Workflow B complete.",
                evidence=[ev_b],
            )
            return res_b.to_dict()

        run_a1 = _execute_a()
        _run_b = _execute_b()  # noqa: F841
        run_a2 = _execute_a()

        assert run_a1 == run_a2
        assert "ISO_B" not in str(run_a2)
        assert "doc-isoB-02" not in str(run_a2)
