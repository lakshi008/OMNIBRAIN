"""
OmniBrain Member 4 -- Day 11 End-to-End Contract Regression & Evidence Integrity Tests.

Verifies the complete end-to-end contract regression and evidence integrity across:
  Ingestion -> Search -> Evidence -> Vision -> Supervisor / Downstream -> Final Result

Concern areas:
  1. Ingestion -> Search contract flow
  2. Search -> Evidence & Citation conversion
  3. Evidence -> Vision Request & Input integrity
  4. Vision -> Supervisor / Downstream consumption
  5. Multi-Evidence flow and deterministic ordering
  6. Multi-Document flow and cross-document isolation
  7. Multimodal Content Type preservation
  8. End-to-end Citation & complete Lineage preservation
  9. Metadata preservation & cross-document non-leakage
 10. Intermediate & Final Object Serialization round-trips
 11. Contract Error Propagation across all subsystem boundaries
 12. Repeated Execution & Stale State Isolation
 13. Concurrent Request Execution Isolation
 14. Final Result Integrity & Request Boundary Protection

Constraints:
  - 100% Offline: No external APIs, real LLMs, network, or production secrets.
  - Zero production code modified.
  - Zero compatibility wrappers, new models, or new serializers.
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

# Ingestion Subsystem (Member 1)
from ingestion.models import (
    ChunkValidationResult,
    ChunkingResult,
    DocumentChunk,
    DocumentMetadata,
    EmbeddingGenerationResult,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    EmbeddingVectorRecord,
    PageData,
    RetrievalServiceResult,
    VectorSearchResult,
)
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.retrieval_processor import build_retrieval_context, process_retrieval_results
from ingestion.ingestion_errors import (
    IngestionChunkingError,
    IngestionEmbeddingError,
    IngestionError,
    IngestionExtractionError,
    IngestionPipelineError,
    IngestionValidationError,
)
from ingestion.exceptions import (
    CorruptedPDFError,
    InvalidFileTypeError,
    PDFNotFoundError,
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
from agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentRoutingError,
    AgentValidationError,
)

# Vision Subsystem (Member 3)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.evidence_adapter import VisualEvidenceAdapter
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


# ============================================================================
# Shared Fixtures & Helpers
# ============================================================================


def _create_document_chunk(
    chunk_id: str = "chunk-001",
    chunk_index: int = 0,
    document_id: str = "doc-001",
    filename: str = "quarterly_report.pdf",
    page_number: int | None = 1,
    content: str = "Revenue increased by 14% year-over-year.",
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
        metadata=metadata if metadata is not None else {"source_dept": "finance"},
    )


def _create_vector_search_result(
    chunk_id: str = "chunk-001",
    score: float = 0.92,
    document_id: str = "doc-001",
    filename: str = "quarterly_report.pdf",
    page_number: int | None = 1,
    chunk_index: int = 0,
    content_type: str = "text",
    content: str = "Revenue increased by 14% year-over-year.",
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
        metadata=metadata if metadata is not None else {"source_dept": "finance"},
    )


# ============================================================================
# 1. INGESTION -> SEARCH CONTRACT
# ============================================================================


class TestIngestionToSearchContractRegression:
    """Verifies Document -> DocumentChunk -> Embedding -> VectorSearchResult contract flow."""

    def test_document_chunk_to_search_result_lineage_preservation(self) -> None:
        chunk = _create_document_chunk(
            chunk_id="chunk-ingest-001",
            chunk_index=2,
            document_id="doc-ingest-alpha",
            filename="financial_summary.pdf",
            page_number=5,
            content="Q3 operating income was $4.2M.",
            content_type="text",
            metadata={"quarter": "Q3", "audited": True},
        )
        # Validation & Normalization step
        val_result = validate_chunks([chunk])
        assert val_result.is_valid is True
        assert val_result.valid_chunks == 1

        normalized = normalize_chunks([chunk])
        assert len(normalized) == 1
        assert normalized[0].chunk_id == chunk.chunk_id

        # Embedding preparation step
        prep_result = prepare_for_embedding(normalized)
        assert prep_result.is_ready is True
        assert prep_result.total_items == 1
        assert prep_result.items[0].document_id == "doc-ingest-alpha"
        assert prep_result.items[0].page_number == 5

        # Simulated Vector Search Output preserves exact identities
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
        assert vsr.chunk_id == "chunk-ingest-001"
        assert vsr.document_id == "doc-ingest-alpha"
        assert vsr.filename == "financial_summary.pdf"
        assert vsr.page_number == 5
        assert vsr.chunk_index == 2
        assert vsr.content == "Q3 operating income was $4.2M."
        assert vsr.metadata["quarter"] == "Q3"

    def test_retrieval_processing_and_context_building(self) -> None:
        vsr1 = _create_vector_search_result(chunk_id="c1", score=0.90, content="Income grew.")
        vsr2 = _create_vector_search_result(chunk_id="c2", score=0.85, content="Expenses dropped.")
        vsr_low = _create_vector_search_result(chunk_id="c3", score=0.40, content="Noise.")

        filtered = process_retrieval_results([vsr1, vsr2, vsr_low], min_score=0.70, max_results=5)
        assert len(filtered) == 2
        assert [r.chunk_id for r in filtered] == ["c1", "c2"]

        context = build_retrieval_context(filtered)
        assert "Income grew." in context
        assert "Expenses dropped." in context
        assert "Noise." not in context


# ============================================================================
# 2. SEARCH -> EVIDENCE CONTRACT
# ============================================================================


class TestSearchToEvidenceContractRegression:
    """Verifies Search results are converted into AgentCitation and VisualEvidence contracts."""

    def test_search_result_to_agent_citation_conversion(self) -> None:
        vsr = _create_vector_search_result(
            chunk_id="chk-cit-01",
            score=0.88,
            document_id="doc-cit-01",
            filename="annual_report.pdf",
            page_number=12,
            content_type="text",
            metadata={"sec_filing": "10-K"},
        )
        citation = AgentCitation.from_search_result(vsr)
        assert isinstance(citation, AgentCitation)
        assert citation.document_id == "doc-cit-01"
        assert citation.filename == "annual_report.pdf"
        assert citation.chunk_id == "chk-cit-01"
        assert citation.page_number == 12
        assert citation.content_type == "text"
        assert citation.score == 0.88
        assert citation.metadata["sec_filing"] == "10-K"

    def test_search_result_to_visual_evidence_conversion(self) -> None:
        vsr_image = _create_vector_search_result(
            chunk_id="chk-img-01",
            score=0.91,
            document_id="doc-img-01",
            filename="diagrams.pdf",
            page_number=8,
            chunk_index=3,
            content_type="image",
            metadata={"caption": "System Architecture"},
        )
        assert VisualEvidenceAdapter.is_visual(vsr_image) is True

        ev = VisualEvidence.from_search_result(vsr_image)
        assert isinstance(ev, VisualEvidence)
        assert ev.document_id == "doc-img-01"
        assert ev.filename == "diagrams.pdf"
        assert ev.chunk_id == "chk-img-01"
        assert ev.page_number == 8
        assert ev.chunk_index == 3
        assert ev.content_type == "image"
        assert ev.metadata["caption"] == "System Architecture"

    def test_citation_to_visual_evidence_conversion(self) -> None:
        citation = AgentCitation(
            document_id="doc-chart-01",
            filename="metrics.pdf",
            chunk_id="chk-chart-01",
            page_number=3,
            content_type="chart",
            score=0.85,
            metadata={"chart_type": "bar", "chunk_index": 1},
        )
        assert VisualEvidenceAdapter.is_visual(citation) is True

        ev = VisualEvidence.from_citation(citation)
        assert ev.document_id == "doc-chart-01"
        assert ev.filename == "metrics.pdf"
        assert ev.chunk_id == "chk-chart-01"
        assert ev.page_number == 3
        assert ev.content_type == "chart"
        assert ev.metadata["chart_type"] == "bar"


# ============================================================================
# 3. EVIDENCE -> VISION CONTRACT
# ============================================================================


class TestEvidenceToVisionContractRegression:
    """Verifies VisionRequest receives correct evidence and maintains exact properties."""

    def test_evidence_in_vision_request_preservation(self) -> None:
        ev = VisualEvidence(
            document_id="doc-vis-001",
            filename="blueprint.pdf",
            chunk_id="chk-bp-01",
            page_number=14,
            chunk_index=2,
            content_type="diagram",
            image_path="/path/to/bp01.png",
            image_format="PNG",
            width=1920,
            height=1080,
            description="Network topology diagram",
            metadata={"network_zone": "DMZ"},
        )
        req = VisionRequest(
            query="Analyze the network topology diagram.",
            evidence=[ev],
            session_id="sess-e2e-001",
            metadata={"requester": "secops"},
        )
        assert req.has_evidence is True
        assert req.total_evidence == 1
        assert req.query == "Analyze the network topology diagram."
        assert req.session_id == "sess-e2e-001"

        retrieved_ev = req.evidence[0]
        assert retrieved_ev.document_id == "doc-vis-001"
        assert retrieved_ev.filename == "blueprint.pdf"
        assert retrieved_ev.chunk_id == "chk-bp-01"
        assert retrieved_ev.page_number == 14
        assert retrieved_ev.content_type == "diagram"
        assert retrieved_ev.image_format == "PNG"
        assert retrieved_ev.width == 1920
        assert retrieved_ev.height == 1080
        assert retrieved_ev.metadata["network_zone"] == "DMZ"


# ============================================================================
# 4. VISION -> SUPERVISOR / DOWNSTREAM CONTRACT
# ============================================================================


class TestVisionToSupervisorContractRegression:
    """Verifies VisionResult is consumable by Supervisor and downstream Agent contracts."""

    def test_vision_result_to_supervisor_agent_response_flow(self) -> None:
        ev = VisualEvidence(
            document_id="doc-sup-01",
            filename="quarterly_deck.pdf",
            chunk_id="chk-deck-01",
            page_number=7,
            content_type="chart",
            metadata={"source": "treasury"},
        )
        vision_res = VisionResult(
            query="What is the gross margin trend?",
            status="success",
            description="Gross margin grew by 3.2% to 48.5%.",
            evidence=[ev],
            metadata={"model": "offline-stub", "latency_ms": 35},
        )
        assert vision_res.document_id == "doc-sup-01"
        assert vision_res.filename == "quarterly_deck.pdf"
        assert vision_res.page_number == 7
        assert vision_res.chunk_id == "chk-deck-01"

        # Supervisor builds citation and response
        citation = AgentCitation(
            document_id=vision_res.document_id,
            filename=vision_res.filename,
            chunk_id=vision_res.chunk_id,
            page_number=vision_res.page_number,
            content_type=vision_res.content_type,
            score=0.95,
            metadata=vision_res.metadata,
        )
        response = AgentResponse(
            answer=vision_res.description,
            agent_name="VisionAgent",
            status=vision_res.status,
            citations=[citation],
            metadata={"session_id": "sess-sup-01", "stage": "SUPERVISOR_EVALUATION"},
        )
        assert response.is_success is True
        assert response.has_citations is True
        assert response.answer == "Gross margin grew by 3.2% to 48.5%."
        assert response.citations[0].document_id == "doc-sup-01"
        assert response.citations[0].filename == "quarterly_deck.pdf"
        assert response.citations[0].page_number == 7

        # AgentState ingestion of Supervisor response
        state = AgentState(query=vision_res.query)
        state.answer = response.answer
        state.status = "completed"
        for c in response.citations:
            state.add_citation(c)
        assert len(state.citations) == 1
        assert state.citations[0].document_id == "doc-sup-01"


# ============================================================================
# 5. MULTI-EVIDENCE FLOW & ORDERING
# ============================================================================


class TestMultiEvidenceFlowRegression:
    """Verifies multiple evidence items maintain exact count, ordering, and 1:1 lineage."""

    def test_multi_evidence_ordering_and_preservation(self) -> None:
        ev_a = VisualEvidence(document_id="doc-multi", filename="multi.pdf", chunk_id="ck-01", page_number=1, chunk_index=0, content_type="image", metadata={"order": 1})
        ev_b = VisualEvidence(document_id="doc-multi", filename="multi.pdf", chunk_id="ck-02", page_number=2, chunk_index=1, content_type="chart", metadata={"order": 2})
        ev_c = VisualEvidence(document_id="doc-multi", filename="multi.pdf", chunk_id="ck-03", page_number=3, chunk_index=2, content_type="diagram", metadata={"order": 3})

        req = VisionRequest(query="Analyze all 3 figures.", evidence=[ev_a, ev_b, ev_c])
        assert req.total_evidence == 3
        assert [e.chunk_id for e in req.evidence] == ["ck-01", "ck-02", "ck-03"]
        assert [e.metadata["order"] for e in req.evidence] == [1, 2, 3]

        res = VisionResult(
            query=req.query,
            status="success",
            description="All 3 figures analyzed.",
            evidence=req.evidence,
        )
        assert len(res.evidence) == 3
        assert [e.chunk_id for e in res.evidence] == ["ck-01", "ck-02", "ck-03"]
        # Primary lineage inherits from first evidence
        assert res.document_id == "doc-multi"
        assert res.chunk_id == "ck-01"


# ============================================================================
# 6. MULTI-DOCUMENT FLOW & ISOLATION
# ============================================================================


class TestMultiDocumentFlowRegression:
    """Verifies evidence from multiple documents maintains strict cross-document isolation."""

    def test_multi_document_lineage_and_metadata_isolation(self) -> None:
        ev_doc1 = VisualEvidence(
            document_id="doc-alpha-100",
            filename="alpha.pdf",
            chunk_id="chk-alpha-01",
            page_number=3,
            content_type="chart",
            metadata={"secret_proj": "Apollo", "doc_owner": "Alice"},
        )
        ev_doc2 = VisualEvidence(
            document_id="doc-beta-200",
            filename="beta.pdf",
            chunk_id="chk-beta-01",
            page_number=9,
            content_type="diagram",
            metadata={"secret_proj": "Zeus", "doc_owner": "Bob"},
        )

        cit1 = AgentCitation.from_dict({
            "document_id": ev_doc1.document_id,
            "filename": ev_doc1.filename,
            "chunk_id": ev_doc1.chunk_id,
            "page_number": ev_doc1.page_number,
            "content_type": ev_doc1.content_type,
            "score": 0.94,
            "metadata": ev_doc1.metadata,
        })
        cit2 = AgentCitation.from_dict({
            "document_id": ev_doc2.document_id,
            "filename": ev_doc2.filename,
            "chunk_id": ev_doc2.chunk_id,
            "page_number": ev_doc2.page_number,
            "content_type": ev_doc2.content_type,
            "score": 0.89,
            "metadata": ev_doc2.metadata,
        })

        # Verify Doc 1
        assert cit1.document_id == "doc-alpha-100"
        assert cit1.filename == "alpha.pdf"
        assert cit1.metadata["secret_proj"] == "Apollo"
        assert "Zeus" not in cit1.metadata.values()

        # Verify Doc 2
        assert cit2.document_id == "doc-beta-200"
        assert cit2.filename == "beta.pdf"
        assert cit2.metadata["secret_proj"] == "Zeus"
        assert "Apollo" not in cit2.metadata.values()

        # Combined in SearchResult maintains distinct documents
        sr = SearchResult(query="Compare Alpha and Beta", status="RESULTS_FOUND", citations=[cit1, cit2])
        assert sr.unique_document_count == 2
        assert set(sr.unique_documents) == {"doc-alpha-100", "doc-beta-200"}


# ============================================================================
# 7. CONTENT TYPE PRESERVATION
# ============================================================================


class TestContentTypePreservationRegression:
    """Verifies all supported content types survive the entire pipeline."""

    @pytest.mark.parametrize("content_type", ["image", "chart", "diagram"])
    def test_visual_content_types_preservation(self, content_type: str) -> None:
        vsr = _create_vector_search_result(
            chunk_id=f"chk-{content_type}",
            content_type=content_type,
        )
        assert VisualEvidenceAdapter.is_visual_content_type(content_type) is True
        assert VisualEvidenceAdapter.is_visual(vsr) is True

        ev = VisualEvidence.from_search_result(vsr)
        assert ev.content_type == content_type

        req = VisionRequest(query=f"Describe {content_type}", evidence=[ev])
        assert req.evidence[0].content_type == content_type

        res = VisionResult(query=req.query, evidence=req.evidence)
        assert res.content_type == content_type

    def test_text_and_table_content_types_handling(self) -> None:
        vsr_text = _create_vector_search_result(content_type="text")
        vsr_table = _create_vector_search_result(content_type="table")

        assert VisualEvidenceAdapter.is_visual(vsr_text) is False
        assert VisualEvidenceAdapter.is_visual(vsr_table) is False

        cit_text = AgentCitation.from_search_result(vsr_text)
        cit_table = AgentCitation.from_search_result(vsr_table)

        assert cit_text.content_type == "text"
        assert cit_table.content_type == "table"


# ============================================================================
# 8. CITATION & LINEAGE INTEGRITY
# ============================================================================


class TestCitationAndLineageIntegrityRegression:
    """Verifies full unbroken lineage chain across all stages."""

    def test_unbroken_lineage_across_all_six_stages(self) -> None:
        # Stage 1: Document & Chunk
        doc_id = "doc-chain-999"
        filename = "master_spec.pdf"
        chunk_id = "ck-spec-042"
        page_num = 18
        chunk_idx = 7

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            chunk_index=chunk_idx,
            document_id=doc_id,
            filename=filename,
            page_number=page_num,
            content="Spec specification chart.",
            content_type="chart",
            metadata={"spec_version": "2.4"},
        )

        # Stage 2: Search Result
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

        # Stage 3: Evidence Adapter
        ev = VisualEvidence.from_search_result(vsr)

        # Stage 4: Vision Request
        req = VisionRequest(query="Extract chart values.", evidence=[ev])

        # Stage 5: Vision Result
        vres = VisionResult(
            query=req.query,
            status="success",
            description="Chart values: A=10, B=20.",
            evidence=req.evidence,
        )

        # Stage 6: Final Agent Response Citation
        final_citation = AgentCitation(
            document_id=vres.document_id,
            filename=vres.filename,
            chunk_id=vres.chunk_id,
            page_number=vres.page_number,
            content_type=vres.content_type,
            score=vsr.score,
            metadata=vres.metadata,
        )

        # Assert unbroken lineage
        for obj in (chunk, vsr, ev, req.evidence[0], vres, final_citation):
            assert getattr(obj, "document_id") == doc_id
            assert getattr(obj, "filename") == filename
            assert getattr(obj, "chunk_id") == chunk_id
            assert getattr(obj, "page_number") == page_num


# ============================================================================
# 9. METADATA PRESERVATION
# ============================================================================


class TestMetadataPreservationRegression:
    """Verifies complex, nested metadata survives end-to-end without mutation or cross-talk."""

    def test_complex_metadata_preservation(self) -> None:
        custom_metadata = {
            "department": "Engineering",
            "confidentiality": "Internal",
            "nested": {"tags": ["q4", "roadmap"], "version": 3},
            "timestamp": "2026-08-25T12:00:00Z",
        }
        chunk = _create_document_chunk(content_type="chart", metadata=custom_metadata)
        vsr = _create_vector_search_result(content_type="chart", metadata=chunk.metadata)
        ev = VisualEvidence.from_search_result(vsr)
        cit = AgentCitation.from_search_result(vsr)

        assert ev.metadata == custom_metadata
        assert cit.metadata == custom_metadata
        assert ev.metadata["nested"]["tags"] == ["q4", "roadmap"]


# ============================================================================
# 10. SERIALIZATION ROUND-TRIPS
# ============================================================================


class TestSerializationRoundTripsRegression:
    """Verifies lossless to_dict / from_dict round-trips for all intermediate and final contracts."""

    def test_agent_citation_serialization(self) -> None:
        orig = AgentCitation(document_id="d1", filename="f1.pdf", chunk_id="c1", page_number=2, score=0.88, metadata={"m": 1})
        restored = AgentCitation.from_dict(orig.to_dict())
        assert restored.document_id == orig.document_id
        assert restored.score == orig.score
        assert restored.metadata == orig.metadata

    def test_visual_evidence_serialization(self) -> None:
        orig = VisualEvidence(document_id="d2", filename="f2.pdf", chunk_id="c2", page_number=4, content_type="chart", metadata={"k": "v"})
        restored = VisualEvidence.from_dict(orig.to_dict())
        assert restored.document_id == orig.document_id
        assert restored.content_type == orig.content_type
        assert restored.metadata == orig.metadata

    def test_vision_request_serialization(self) -> None:
        ev = VisualEvidence(document_id="d3", filename="f3.pdf", chunk_id="c3")
        orig = VisionRequest(query="Describe", evidence=[ev], session_id="s1", metadata={"tag": "t"})
        restored = VisionRequest.from_dict(orig.to_dict())
        assert restored.query == orig.query
        assert len(restored.evidence) == 1
        assert restored.evidence[0].document_id == "d3"

    def test_vision_result_serialization(self) -> None:
        ev = VisualEvidence(document_id="d4", filename="f4.pdf", chunk_id="c4")
        orig = VisionResult(query="What is this?", status="success", description="desc", evidence=[ev], document_id="d4", filename="f4.pdf")
        restored = VisionResult.from_dict(orig.to_dict())
        assert restored.query == orig.query
        assert restored.description == orig.description
        assert restored.document_id == "d4"

    def test_search_result_serialization(self) -> None:
        cit = AgentCitation(document_id="d5", filename="f5.pdf", chunk_id="c5")
        orig = SearchResult(query="Find docs", status="RESULTS_FOUND", citations=[cit], context="ctx text")
        restored = SearchResult.from_dict(orig.to_dict())
        assert restored.query == orig.query
        assert len(restored.citations) == 1
        assert restored.citations[0].document_id == "d5"

    def test_agent_response_serialization(self) -> None:
        cit = AgentCitation(document_id="d6", filename="f6.pdf", chunk_id="c6")
        orig = AgentResponse(answer="Final answer", agent_name="TestAgent", citations=[cit], error=None)
        restored = AgentResponse.from_dict(orig.to_dict())
        assert restored.answer == orig.answer
        assert restored.agent_name == orig.agent_name
        assert len(restored.citations) == 1


# ============================================================================
# 11. CONTRACT ERROR PROPAGATION
# ============================================================================


class TestContractErrorPropagationRegression:
    """Verifies failures and invalid contracts trigger expected domain errors cleanly."""

    def test_agent_citation_validation_failure(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="f.pdf", chunk_id="ck-01")

    def test_visual_evidence_invalid_content_type_failure(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d", filename="f.pdf", chunk_id="ck", content_type="text")

    def test_vision_request_empty_query_failure(self) -> None:
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="   ")

    def test_search_request_invalid_score_failure(self) -> None:
        with pytest.raises(AgentValidationError):
            SearchRequest(query="test", min_score=2.5)

    def test_vision_result_error_status_contract(self) -> None:
        vres_err = VisionResult(
            query="analyze",
            status="error",
            description="",
            error="Vision processing failed due to timeout.",
        )
        assert vres_err.status == "error"
        assert vres_err.error == "Vision processing failed due to timeout."

        # AgentResponse error propagation
        resp_err = AgentResponse(
            answer="",
            agent_name="VisionAgent",
            status="error",
            error=vres_err.error,
        )
        assert resp_err.is_error is True
        assert resp_err.is_success is False
        assert resp_err.error == "Vision processing failed due to timeout."


# ============================================================================
# 12. REPEATED EXECUTION ISOLATION
# ============================================================================


class TestRepeatedExecutionIsolationRegression:
    """Verifies deterministic contract behavior across consecutive executions without stale state."""

    def test_repeated_pipeline_runs_are_deterministic(self) -> None:
        results = []
        for run_idx in range(5):
            chunk = _create_document_chunk(
                chunk_id=f"chk-run-{run_idx}",
                document_id="doc-repeatable",
                content=f"Run {run_idx} data",
            )
            vsr = _create_vector_search_result(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
            )
            cit = AgentCitation.from_search_result(vsr)
            results.append((cit.chunk_id, cit.document_id))

        assert len(results) == 5
        for idx, (cid, did) in enumerate(results):
            assert cid == f"chk-run-{idx}"
            assert did == "doc-repeatable"


# ============================================================================
# 13. CONCURRENT REQUEST EXECUTION ISOLATION
# ============================================================================


class TestConcurrentExecutionIsolationRegression:
    """Verifies concurrent requests across threads remain strictly isolated."""

    def test_concurrent_pipeline_execution_isolation(self) -> None:
        def _execute_mock_pipeline(req_id: int) -> dict[str, Any]:
            doc_id = f"doc-concurrent-{req_id}"
            chunk_id = f"chk-concurrent-{req_id}"
            metadata = {"req_id": req_id, "token": f"tok-{req_id}"}

            chunk = _create_document_chunk(chunk_id=chunk_id, document_id=doc_id, metadata=metadata)
            vsr = _create_vector_search_result(chunk_id=chunk.chunk_id, document_id=chunk.document_id, metadata=chunk.metadata)
            cit = AgentCitation.from_search_result(vsr)
            resp = AgentResponse(
                answer=f"Result for req {req_id}",
                agent_name="ConcurrentAgent",
                citations=[cit],
                metadata=metadata,
            )
            return {
                "req_id": req_id,
                "doc_id": resp.citations[0].document_id,
                "chunk_id": resp.citations[0].chunk_id,
                "token": resp.metadata["token"],
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_execute_mock_pipeline, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        for res in results:
            req_id = res["req_id"]
            assert res["doc_id"] == f"doc-concurrent-{req_id}"
            assert res["chunk_id"] == f"chk-concurrent-{req_id}"
            assert res["token"] == f"tok-{req_id}"


# ============================================================================
# 14. FINAL RESULT INTEGRITY & BOUNDARY PROTECTION
# ============================================================================


class TestFinalResultIntegrityRegression:
    """Verifies final result contains exclusively information belonging to current request."""

    def test_final_result_integrity_and_field_completeness(self) -> None:
        doc_id = "doc-final-777"
        filename = "final_brief.pdf"
        chunk_id = "ck-final-01"
        page_num = 1

        ev = VisualEvidence(
            document_id=doc_id,
            filename=filename,
            chunk_id=chunk_id,
            page_number=page_num,
            content_type="chart",
            metadata={"priority": "P0"},
        )
        vision_res = VisionResult(
            query="Summarize P0 brief chart.",
            status="success",
            description="P0 Brief shows 100% test passing.",
            evidence=[ev],
        )

        citation = AgentCitation(
            document_id=vision_res.document_id,
            filename=vision_res.filename,
            chunk_id=vision_res.chunk_id,
            page_number=vision_res.page_number,
            content_type=vision_res.content_type,
            score=0.99,
            metadata=vision_res.metadata,
        )
        response = AgentResponse(
            answer=vision_res.description,
            agent_name="OmniBrainSupervisor",
            status="success",
            citations=[citation],
            metadata={"request_id": "REQ-FINAL-777"},
        )

        # Verification of complete final result integrity
        assert response.is_success is True
        assert response.answer == "P0 Brief shows 100% test passing."
        assert len(response.citations) == 1
        assert response.citations[0].document_id == doc_id
        assert response.citations[0].filename == filename
        assert response.citations[0].chunk_id == chunk_id
        assert response.citations[0].page_number == 1
        assert response.citations[0].score == 0.99
        assert response.metadata["request_id"] == "REQ-FINAL-777"
