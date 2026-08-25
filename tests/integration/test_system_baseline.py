"""
OmniBrain Member 4 — Day 1 System Evaluation & Integration Baseline Tests.

Verifies the baseline accessibility and cross-member contract compatibility across:
- Member 1 (Ingestion subsystem)
- Member 2 (Search / Agent subsystem)
- Member 3 (Vision Agent subsystem)
- Member 4 (System-level Integration, Evaluation & Downstream Handoff Baseline)

Ensures that:
1. All public contracts and domain models from Members 1, 2, and 3 are accessible.
2. Cross-member contract handoffs are strictly compatible and preserve lineage:
   - Ingestion (VectorSearchResult, DocumentChunk) -> Search (AgentCitation, SearchResult)
   - Search (AgentCitation, SearchResult, AgentResponse) -> Vision (VisualEvidence, VisionRequest)
   - Vision (VisionResult) -> Downstream / Supervisor (AgentState, AgentResponse)
3. Zero network calls, zero external APIs, zero real LLMs, zero credentials, zero production side-effects.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path for test runners executing this file directly
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# Member 1 public imports
import ingestion
from ingestion.models import (
    DocumentChunk,
    DocumentMetadata,
    IngestionResult,
    ParsedDocument,
    VectorSearchResult,
)

# Member 2 public imports
import agents
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    AgentState,
    SearchRequest,
    SearchResult,
)
from agents.search_agent import SearchAgent

# Member 3 public imports
import vision
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.pipeline import VisionPipeline
from vision.vision_agent import VisionAgent


# ============================================================================
# 1. PUBLIC CONTRACTS & EXPORTS BASELINE
# ============================================================================


class TestPublicExportsBaseline:
    """Verifies that all subsystems export expected public interfaces."""

    def test_member1_ingestion_public_exports(self) -> None:
        """Verify Member 1 public API exports."""
        expected_symbols = [
            "ingest_pdf",
            "extract_text",
            "extract_tables",
            "extract_images",
            "chunk_document",
            "validate_chunks",
            "normalize_chunks",
            "prepare_for_embedding",
            "generate_embeddings",
            "EmbeddingProvider",
            "QdrantVectorStore",
            "QdrantConfig",
            "retrieve",
            "process_retrieval_results",
            "build_retrieval_context",
            "retrieve_context",
            "run_ingestion",
            "IngestionStatus",
            "PipelineStatus",
            "PipelineStage",
            "IngestionConfig",
            "IngestionMetrics",
            "StageMetrics",
            "IngestionLogger",
            "get_ingestion_logger",
            "IngestionHealthResult",
            "check_ingestion_health",
            "check_ingestion_readiness",
            "IngestionValidationResult",
            "validate_chunk_contracts",
            "validate_embedding_contracts",
            "validate_search_result_contracts",
            "validate_pipeline_lineage",
            "validate_pipeline_contracts",
            "IngestionResult",
            "ChunkingResult",
            "ChunkValidationResult",
            "EmbeddingRecord",
            "EmbeddingPreparationResult",
            "EmbeddingVectorRecord",
            "EmbeddingGenerationResult",
            "VectorSearchResult",
            "RetrievalServiceResult",
            "DocumentChunk",
            "ParsedDocument",
            "PageData",
            "DocumentMetadata",
            "ExtractedTable",
            "TableExtractionResult",
            "ExtractedImage",
            "ImageExtractionResult",
            "IngestionError",
            "IngestionValidationError",
            "IngestionExtractionError",
            "IngestionChunkingError",
            "IngestionEmbeddingError",
            "IngestionPipelineError",
            "PDFNotFoundError",
            "InvalidFileTypeError",
            "CorruptedPDFError",
        ]
        for sym in expected_symbols:
            assert hasattr(ingestion, sym), f"Member 1 missing exported symbol: {sym}"

    def test_member2_agents_public_exports(self) -> None:
        """Verify Member 2 public API exports."""
        expected_symbols = [
            "AgentRequest",
            "SearchRequest",
            "AgentResponse",
            "AgentCitation",
            "AgentState",
            "SearchResult",
            "SearchAgent",
            "AgentError",
            "AgentValidationError",
            "AgentRoutingError",
            "AgentExecutionError",
        ]
        for sym in expected_symbols:
            assert hasattr(agents, sym), f"Member 2 missing exported symbol: {sym}"

    def test_member3_vision_public_exports(self) -> None:
        """Verify Member 3 public API exports."""
        expected_symbols = [
            "VisualEvidence",
            "VisionRequest",
            "VisionResult",
            "VALID_VISUAL_CONTENT_TYPES",
            "PreparedImageEvidence",
            "ImageEvidencePreparator",
            "OversizedImagePolicy",
            "SUPPORTED_IMAGE_FORMATS",
            "prepare_image_evidence",
            "VisionModelInput",
            "VisionInputBuilder",
            "build_vision_input",
            "VisionModelProvider",
            "VisionProviderRegistry",
            "VisionProviderConfig",
            "VisionProviderCapabilities",
            "VisionExecutionAdapter",
            "VisionExecutionStage",
            "VisionExecutionLifecycle",
            "VisionExecutionObservation",
            "VisionCancellationToken",
            "VisionRetryPolicy",
            "execute_vision_request",
            "VisionResultNormalizer",
            "VisionExecutionTrace",
            "FORBIDDEN_METADATA_KEYS",
            "VisionPipeline",
            "run_vision_pipeline",
            "VisualEvidenceAdapter",
            "VisionAgent",
            "VisionAgentError",
            "VisionError",
            "VisionCancellationError",
            "VisionInputValidationError",
            "VisionEvidenceError",
            "VisionProcessingError",
            "VisionProviderError",
            "VisionProviderConfigError",
            "VisionProviderExecutionError",
            "VisionProviderUnavailableError",
            "VisionUnsupportedCapabilityError",
            "VisionTimeoutError",
        ]
        for sym in expected_symbols:
            assert hasattr(vision, sym), f"Member 3 missing exported symbol: {sym}"


# ============================================================================
# 2. INGESTION -> SEARCH CONTRACT VERIFICATION
# ============================================================================


class TestIngestionToSearchContract:
    """Verifies that Member 1 ingestion outputs satisfy Member 2 search requirements."""

    def test_vector_search_result_to_agent_citation(self) -> None:
        """Verify VectorSearchResult can be converted into AgentCitation with lineage intact."""
        vs_result = VectorSearchResult(
            document_id="doc-101",
            filename="quarterly_report.pdf",
            chunk_id="chunk-001",
            page_number=4,
            chunk_index=0,
            content="Revenue increased by 15% year-over-year.",
            score=0.92,
            content_type="text",
            metadata={"source_section": "financial_summary", "author": "Finance Dept"},
        )

        citation = AgentCitation.from_search_result(vs_result)

        assert citation.document_id == "doc-101"
        assert citation.filename == "quarterly_report.pdf"
        assert citation.chunk_id == "chunk-001"
        assert citation.page_number == 4
        assert citation.content_type == "text"
        assert citation.score == 0.92
        assert citation.metadata["source_section"] == "financial_summary"
        assert citation.metadata["author"] == "Finance Dept"

    def test_document_chunk_contract_compatibility(self) -> None:
        """Verify DocumentChunk attributes align with AgentCitation requirements."""
        doc_chunk = DocumentChunk(
            chunk_id="chunk-img-002",
            document_id="doc-202",
            filename="architecture_diagram.pdf",
            page_number=2,
            chunk_index=1,
            content="System architecture overview diagram",
            content_type="image",
            metadata={"image_path": "B:/tmp/test_img.png", "resolution": "1080p"},
        )

        citation = AgentCitation(
            document_id=doc_chunk.document_id,
            filename=doc_chunk.filename,
            chunk_id=doc_chunk.chunk_id,
            page_number=doc_chunk.page_number,
            content_type=doc_chunk.content_type,
            score=0.88,
            metadata=dict(doc_chunk.metadata),
        )

        assert citation.document_id == doc_chunk.document_id
        assert citation.filename == doc_chunk.filename
        assert citation.chunk_id == doc_chunk.chunk_id
        assert citation.page_number == doc_chunk.page_number
        assert citation.content_type == doc_chunk.content_type
        assert citation.metadata["image_path"] == "B:/tmp/test_img.png"

    def test_search_request_from_agent_request(self) -> None:
        """Verify SearchRequest creation and overrides from AgentRequest."""
        agent_req = AgentRequest(
            query="What is the net profit margin?",
            session_id="session-42",
            document_filter={"doc_id": "doc-101"},
            metadata={"top_k": 5, "min_score": 0.75, "priority": "high"},
        )

        search_req = SearchRequest.from_agent_request(agent_req)
        assert search_req.query == "What is the net profit margin?"
        assert search_req.session_id == "session-42"
        assert search_req.document_filter == {"doc_id": "doc-101"}
        assert search_req.top_k == 5
        assert search_req.min_score == 0.75

    def test_search_result_packaging(self) -> None:
        """Verify SearchResult packaging with multiple citations."""
        citations = [
            AgentCitation(
                document_id="doc-01",
                filename="f1.pdf",
                chunk_id="c-01",
                page_number=1,
                content_type="text",
                score=0.95,
            ),
            AgentCitation(
                document_id="doc-01",
                filename="f1.pdf",
                chunk_id="c-02",
                page_number=2,
                content_type="image",
                score=0.89,
            ),
            AgentCitation(
                document_id="doc-02",
                filename="f2.pdf",
                chunk_id="c-03",
                page_number=1,
                content_type="table",
                score=0.85,
            ),
        ]

        search_result = SearchResult(
            query="Analyze Q3 figures",
            status="RESULTS_FOUND",
            citations=citations,
            context="[1] Financial growth details.\n[2] Architecture flow.",
            metadata={"engine": "qdrant_mock"},
        )

        assert search_result.has_results is True
        assert search_result.total_results == 3
        assert len(search_result.text_results) == 1
        assert len(search_result.image_results) == 1
        assert len(search_result.table_results) == 1
        assert search_result.unique_document_count == 2
        assert search_result.unique_documents == ["doc-01", "doc-02"]


# ============================================================================
# 3. SEARCH -> VISION CONTRACT VERIFICATION
# ============================================================================


class TestSearchToVisionContract:
    """Verifies that Member 2 Search evidence satisfies Member 3 Vision input contracts."""

    def test_adapt_visual_citation_to_visual_evidence(self) -> None:
        """Verify visual AgentCitation adapts into VisualEvidence preserving lineage."""
        citation = AgentCitation(
            document_id="doc-vision-01",
            filename="diagrams.pdf",
            chunk_id="c-img-99",
            page_number=3,
            content_type="diagram",
            score=0.91,
            metadata={"chunk_index": 2, "image_path": "B:/tmp/diagram.png"},
        )

        assert VisualEvidenceAdapter.is_visual(citation) is True
        visual_evidence = VisualEvidenceAdapter.adapt_citation(citation)

        assert isinstance(visual_evidence, VisualEvidence)
        assert visual_evidence.document_id == "doc-vision-01"
        assert visual_evidence.filename == "diagrams.pdf"
        assert visual_evidence.chunk_id == "c-img-99"
        assert visual_evidence.page_number == 3
        assert visual_evidence.chunk_index == 2
        assert visual_evidence.content_type == "diagram"
        assert visual_evidence.image_path == "B:/tmp/diagram.png"

    def test_adapt_vector_search_result_directly(self) -> None:
        """Verify Member 1 VectorSearchResult with visual modality adapts to VisualEvidence."""
        vs_result = VectorSearchResult(
            document_id="doc-chart-02",
            filename="charts.pdf",
            chunk_id="c-chart-01",
            page_number=5,
            chunk_index=4,
            content="Bar chart showing quarterly performance",
            score=0.94,
            content_type="chart",
            metadata={"image_path": "B:/tmp/chart.png", "width": 800, "height": 600},
        )

        visual_evidence = VisualEvidenceAdapter.adapt_search_result(vs_result)
        assert visual_evidence.document_id == "doc-chart-02"
        assert visual_evidence.filename == "charts.pdf"
        assert visual_evidence.chunk_id == "c-chart-01"
        assert visual_evidence.page_number == 5
        assert visual_evidence.chunk_index == 4
        assert visual_evidence.content_type == "chart"
        assert visual_evidence.description == "Bar chart showing quarterly performance"
        assert visual_evidence.image_path == "B:/tmp/chart.png"

    def test_adapt_search_package_filters_non_visual(self) -> None:
        """Verify adapt_search_package extracts only visual modalities in original order."""
        citations = [
            AgentCitation(
                document_id="doc-1",
                filename="f1.pdf",
                chunk_id="c-1",
                page_number=1,
                content_type="text",
                score=0.95,
            ),
            AgentCitation(
                document_id="doc-1",
                filename="f1.pdf",
                chunk_id="c-2",
                page_number=2,
                content_type="image",
                score=0.90,
            ),
            AgentCitation(
                document_id="doc-2",
                filename="f2.pdf",
                chunk_id="c-3",
                page_number=3,
                content_type="chart",
                score=0.85,
            ),
            AgentCitation(
                document_id="doc-2",
                filename="f2.pdf",
                chunk_id="c-4",
                page_number=4,
                content_type="table",
                score=0.80,
            ),
        ]
        search_pkg = SearchResult(
            query="Visual summary query",
            status="RESULTS_FOUND",
            citations=citations,
            context="context block",
        )

        visual_list = VisualEvidenceAdapter.adapt_search_package(search_pkg, strict=False)
        assert len(visual_list) == 2
        assert visual_list[0].chunk_id == "c-2"
        assert visual_list[0].content_type == "image"
        assert visual_list[1].chunk_id == "c-3"
        assert visual_list[1].content_type == "chart"

    def test_vision_request_assembly(self) -> None:
        """Verify VisionRequest encapsulates adapted visual evidence properly."""
        evidence = [
            VisualEvidence(
                document_id="doc-vr-1",
                filename="vr.pdf",
                chunk_id="c-vr-1",
                page_number=1,
                content_type="image",
            )
        ]
        req = VisionRequest(
            query="Analyze this diagram",
            evidence=evidence,
            session_id="sess-vision-01",
            metadata={"priority": "normal"},
        )

        assert req.query == "Analyze this diagram"
        assert req.has_evidence is True
        assert req.total_evidence == 1
        assert req.session_id == "sess-vision-01"
        assert req.evidence[0].document_id == "doc-vr-1"


# ============================================================================
# 4. VISION -> DOWNSTREAM / SUPERVISOR CONTRACT VERIFICATION
# ============================================================================


class TestVisionToDownstreamContract:
    """Verifies that Member 3 Vision outputs satisfy Downstream/Supervisor requirements."""

    def test_vision_result_contract_and_lineage_inheritance(self) -> None:
        """Verify VisionResult inherits lineage from primary visual evidence."""
        evidence_item = VisualEvidence(
            document_id="doc-downstream-01",
            filename="system_spec.pdf",
            chunk_id="chunk-spec-9",
            page_number=7,
            chunk_index=3,
            content_type="diagram",
            image_path="B:/tmp/spec.png",
        )

        vision_result = VisionResult(
            query="Explain the dataflow in the diagram",
            status="success",
            description="The diagram depicts a 4-member distributed RAG workflow.",
            evidence=[evidence_item],
            metadata={"confidence": 0.98, "provider": "mock_vision_provider"},
        )

        # Lineage inherited from primary evidence
        assert vision_result.document_id == "doc-downstream-01"
        assert vision_result.filename == "system_spec.pdf"
        assert vision_result.chunk_id == "chunk-spec-9"
        assert vision_result.page_number == 7
        assert vision_result.content_type == "diagram"
        assert vision_result.status == "success"
        assert "4-member" in vision_result.description
        assert vision_result.metadata["confidence"] == 0.98

    def test_vision_result_consumption_by_agent_state(self) -> None:
        """Verify AgentState can consume VisionResult and lineage for workflow tracking."""
        state = AgentState(
            query="What does the architecture diagram illustrate?",
            route="vision",
        )

        # Simulate vision processing output
        evidence_item = VisualEvidence(
            document_id="doc-arch-01",
            filename="arch.pdf",
            chunk_id="chunk-arch-1",
            page_number=1,
            content_type="diagram",
        )
        vision_result = VisionResult(
            query=state.query,
            status="success",
            description="Architecture contains Ingestion, Search, Vision, and Evaluation modules.",
            evidence=[evidence_item],
        )

        # Convert visual evidence to citation for AgentState provenance
        citation = AgentCitation(
            document_id=vision_result.document_id,
            filename=vision_result.filename,
            chunk_id=vision_result.chunk_id,
            page_number=vision_result.page_number,
            content_type=vision_result.content_type,
            score=1.0,
            metadata={"vision_status": vision_result.status},
        )

        state.add_citation(citation)
        state.update(
            answer=vision_result.description,
            status="completed",
            metadata={"vision_metadata": vision_result.metadata},
        )

        assert state.status == "completed"
        assert state.answer == vision_result.description
        assert len(state.citations) == 1
        assert state.citations[0].document_id == "doc-arch-01"
        assert state.citations[0].chunk_id == "chunk-arch-1"
        assert state.citations[0].content_type == "diagram"

    def test_vision_result_to_agent_response(self) -> None:
        """Verify AgentResponse can be constructed from VisionResult and downstream citations."""
        evidence_item = VisualEvidence(
            document_id="doc-summary-01",
            filename="summary.pdf",
            chunk_id="c-sum-1",
            page_number=2,
            content_type="chart",
        )
        vision_result = VisionResult(
            query="Summarize performance chart",
            status="success",
            description="Performance shows 3x throughput improvement.",
            evidence=[evidence_item],
        )

        citation = AgentCitation(
            document_id=vision_result.document_id,
            filename=vision_result.filename,
            chunk_id=vision_result.chunk_id,
            page_number=vision_result.page_number,
            content_type=vision_result.content_type,
            score=1.0,
        )

        response = AgentResponse(
            answer=vision_result.description,
            agent_name="VisionAgent",
            status=vision_result.status,
            citations=[citation],
            metadata={"query": vision_result.query},
        )

        assert response.is_success is True
        assert response.agent_name == "VisionAgent"
        assert response.total_citations == 1
        assert response.citations[0].document_id == "doc-summary-01"


# ============================================================================
# 5. END-TO-END CROSS-MEMBER DATA FLOW BASELINE
# ============================================================================


class TestCrossMemberEndToEndDataFlow:
    """Verifies complete end-to-end data flow across Member 1 -> 2 -> 3 -> 4."""

    def test_full_pipeline_lineage_preservation(self) -> None:
        """Verify exact document lineage is preserved across all 4 member contracts."""
        # 1. Ingestion (Member 1)
        doc_id = "doc-omnibrain-full-01"
        fname = "omnibrain_whitepaper.pdf"
        cid = "chunk-full-007"
        page = 5
        ctype = "diagram"

        ingestion_chunk = DocumentChunk(
            chunk_id=cid,
            document_id=doc_id,
            filename=fname,
            page_number=page,
            chunk_index=2,
            content="OmniBrain end-to-end system architecture flowchart",
            content_type=ctype,
            metadata={"author": "OmniBrain Team", "version": "1.0"},
        )

        # 2. Search / Retrieval (Member 2)
        citation = AgentCitation(
            document_id=ingestion_chunk.document_id,
            filename=ingestion_chunk.filename,
            chunk_id=ingestion_chunk.chunk_id,
            page_number=ingestion_chunk.page_number,
            content_type=ingestion_chunk.content_type,
            score=0.96,
            metadata=dict(ingestion_chunk.metadata),
        )

        search_result = SearchResult(
            query="Explain OmniBrain system architecture",
            status="RESULTS_FOUND",
            citations=[citation],
            context=f"[1] {ingestion_chunk.content}",
        )

        # 3. Vision (Member 3)
        visual_evidence_list = VisualEvidenceAdapter.adapt_search_package(search_result)
        assert len(visual_evidence_list) == 1

        v_evidence = visual_evidence_list[0]
        assert v_evidence.document_id == doc_id
        assert v_evidence.filename == fname
        assert v_evidence.chunk_id == cid
        assert v_evidence.page_number == page
        assert v_evidence.content_type == ctype

        vision_req = VisionRequest(
            query=search_result.query,
            evidence=visual_evidence_list,
        )

        vision_res = VisionResult(
            query=vision_req.query,
            status="success",
            description="Verified: System consists of Ingestion, Search, Vision, and Evaluation subsystems.",
            evidence=vision_req.evidence,
        )

        # 4. Downstream / Supervisor State (Member 4 / System Orchestrator)
        state = AgentState(
            query=vision_res.query,
            route="vision",
        )
        state.add_citation(
            AgentCitation(
                document_id=vision_res.document_id,
                filename=vision_res.filename,
                chunk_id=vision_res.chunk_id,
                page_number=vision_res.page_number,
                content_type=vision_res.content_type,
                score=1.0,
            )
        )
        state.update(
            answer=vision_res.description,
            status="completed",
        )

        # Final verification: All fields intact through complete 4-member chain
        assert state.citations[0].document_id == doc_id
        assert state.citations[0].filename == fname
        assert state.citations[0].chunk_id == cid
        assert state.citations[0].page_number == page
        assert state.citations[0].content_type == ctype
        assert state.answer == vision_res.description
        assert state.status == "completed"
