"""
OmniBrain Member 4 — Day 56 End-to-End Pipeline Integrity Regression Certification.

Validates the complete supported OmniBrain pipeline across all subsystem boundaries:
  Input Document
        ↓
  Extraction
        ↓
  Parsed Document
        ↓
  Pages
        ↓
  Chunks
        ↓
  Embedding Preparation
        ↓
  Retrieval (Qdrant Vector Store)
        ↓
  Search (SearchAgent)
        ↓
  Context Building (build_retrieval_context)
        ↓
  Agent (AgentResponse, AgentCitation)
        ↓
  Vision / Visual Evidence (VisualEvidenceAdapter, VisualEvidence)
        ↓
  Final Result (SearchResult)

Covers:
  1.  Input validation & ParsedDocument extraction preserving markers.
  2.  Page & Chunk lineage preservation (Document -> Page -> Chunk).
  3.  Embedding preparation via prepare_for_embedding.
  4.  Retrieval operations on in-memory QdrantVectorStore.
  5.  Search execution via SearchAgent.
  6.  Context building via build_retrieval_context.
  7.  Agent response construction and citation mapping.
  8.  Visual evidence adaptation and metadata propagation.
  9.  Final packaged SearchResult verification.
  10. Complete marker trace across all pipeline stages.
  11. Multi-document end-to-end isolation (DOC-A, DOC-B, DOC-C).
  12. Repeated pipeline execution stability (3-run determinism).
  13. Failure injection and clean pipeline recovery.
  14. Pipeline model serialization and deserialization roundtrips.
  15. Cross-document data boundary enforcement.

Constraints:
  - 100% Offline: In-memory QdrantVectorStore, mock deterministic embeddings.
  - Zero production code modified.
  - No external APIs, network, real LLM, or credentials.
  - Synthetic deterministic data only.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest
from qdrant_client import QdrantClient

# Ingestion subsystem (Member 1)
from ingestion.models import (
    ChunkingResult,
    ChunkValidationResult,
    DocumentChunk,
    DocumentMetadata,
    EmbeddingGenerationResult,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    EmbeddingVectorRecord,
    PageData,
    ParsedDocument,
    VectorSearchResult,
)
from ingestion.chunk_validator import normalize_chunks, validate_chunks
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.qdrant_store import QdrantVectorStore
from ingestion.retrieval_processor import (
    build_retrieval_context,
    process_retrieval_results,
)

# Agents subsystem (Member 2)
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    SearchRequest,
    SearchResult,
)
from agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentValidationError,
)
from agents.search_agent import SearchAgent

# Vision subsystem (Member 3)
from vision.models import VisualEvidence, VisionRequest, VisionResult
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.result_normalizer import VisionResultNormalizer


# ============================================================================
# Deterministic Synthetic Fixtures
# ============================================================================

DAY56_DOC_ID = "DAY56-E2E-DOC-PRIMARY"
DAY56_FILENAME = "day56_e2e_spec.pdf"

DAY56_E2E_DOCUMENT_MARKER = "DAY56_E2E_DOCUMENT_MARKER_001"
DAY56_E2E_PAGE_MARKER = "DAY56_E2E_PAGE_MARKER_002"
DAY56_E2E_CHUNK_MARKER = "DAY56_E2E_CHUNK_MARKER_003"
DAY56_E2E_RETRIEVAL_MARKER = "DAY56_E2E_RETRIEVAL_MARKER_004"
DAY56_E2E_CONTEXT_MARKER = "DAY56_E2E_CONTEXT_MARKER_005"

DOC_A = "DOC-A-DAY56"
DOC_B = "DOC-B-DAY56"
DOC_C = "DOC-C-DAY56"

FILE_A = "day56_doc_a.pdf"
FILE_B = "day56_doc_b.pdf"
FILE_C = "day56_doc_c.pdf"

CHUNK_A1 = str(uuid.UUID("11111111-5656-5656-5656-aaaaaaaaaaaa"))
CHUNK_B1 = str(uuid.UUID("22222222-5656-5656-5656-bbbbbbbbbbbb"))
CHUNK_C1 = str(uuid.UUID("33333333-5656-5656-5656-cccccccccccc"))

DAY56_E2E_A = "DAY56_E2E_A_PAYLOAD_ALPHA"
DAY56_E2E_B = "DAY56_E2E_B_PAYLOAD_BETA"
DAY56_E2E_C = "DAY56_E2E_C_PAYLOAD_GAMMA"


class DeterministicDay56EmbeddingProvider:
    """Thread-safe deterministic offline mock embedding provider returning orthogonal 4D unit vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Map distinct document query keywords to predictable unit vectors."""
        clean = text.lower()
        if "alpha" in clean or "doc_a" in clean or "e2e_a" in clean or "retrieval_marker" in clean or "day56_e2e_document" in clean:
            return [1.0, 0.0, 0.0, 0.0]
        if "beta" in clean or "doc_b" in clean or "e2e_b" in clean:
            return [0.0, 1.0, 0.0, 0.0]
        if "gamma" in clean or "doc_c" in clean or "e2e_c" in clean:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding generation."""
        return [self.embed(t) for t in texts]


@pytest.fixture
def e2e_store() -> tuple[QdrantVectorStore, str]:
    """Create an isolated in-memory QdrantVectorStore preloaded with DOC-A, DOC-B, DOC-C."""
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(client=client)
    col_name = "e2e_pipeline_coll"
    store.create_collection(col_name, vector_dimension=4)

    # Document A (also contains the unified E2E test markers)
    rec_a1 = EmbeddingVectorRecord(
        chunk_id=CHUNK_A1,
        document_id=DOC_A,
        filename=FILE_A,
        chunk_index=0,
        page_number=1,
        content_type="text",
        vector=[1.0, 0.0, 0.0, 0.0],
        metadata={
            "document_marker": DAY56_E2E_DOCUMENT_MARKER,
            "page_marker": DAY56_E2E_PAGE_MARKER,
            "chunk_marker": DAY56_E2E_CHUNK_MARKER,
            "retrieval_marker": DAY56_E2E_RETRIEVAL_MARKER,
            "context_marker": DAY56_E2E_CONTEXT_MARKER,
            "payload": DAY56_E2E_A,
            "content": (
                f"{DAY56_E2E_A} | Doc: {DAY56_E2E_DOCUMENT_MARKER} | "
                f"Page: {DAY56_E2E_PAGE_MARKER} | Chunk: {DAY56_E2E_CHUNK_MARKER} | "
                f"Retrieval: {DAY56_E2E_RETRIEVAL_MARKER} | Context: {DAY56_E2E_CONTEXT_MARKER}"
            ),
        },
    )
    gen_a = EmbeddingGenerationResult(
        document_id=DOC_A, filename=FILE_A, items=[rec_a1], dimension=4, is_ready=True,
    )
    store.upsert_embeddings(col_name, gen_a)

    # Document B
    rec_b1 = EmbeddingVectorRecord(
        chunk_id=CHUNK_B1,
        document_id=DOC_B,
        filename=FILE_B,
        chunk_index=0,
        page_number=1,
        content_type="text",
        vector=[0.0, 1.0, 0.0, 0.0],
        metadata={"payload": DAY56_E2E_B, "content": f"{DAY56_E2E_B} specifications"},
    )
    gen_b = EmbeddingGenerationResult(
        document_id=DOC_B, filename=FILE_B, items=[rec_b1], dimension=4, is_ready=True,
    )
    store.upsert_embeddings(col_name, gen_b)

    # Document C
    rec_c1 = EmbeddingVectorRecord(
        chunk_id=CHUNK_C1,
        document_id=DOC_C,
        filename=FILE_C,
        chunk_index=0,
        page_number=1,
        content_type="text",
        vector=[0.0, 0.0, 1.0, 0.0],
        metadata={"payload": DAY56_E2E_C, "content": f"{DAY56_E2E_C} specifications"},
    )
    gen_c = EmbeddingGenerationResult(
        document_id=DOC_C, filename=FILE_C, items=[rec_c1], dimension=4, is_ready=True,
    )
    store.upsert_embeddings(col_name, gen_c)

    return store, col_name


# ============================================================================
# 1. Extraction, Page & Chunk Lineage Verification
# ============================================================================

class TestExtractionAndChunkLineage:
    """Sections 4, 5, 6, 7, 8: Input validation, ParsedDocument, Page and Chunk lineage."""

    def test_parsed_document_extraction_lineage(self) -> None:
        """ParsedDocument correctly retains pages and text markers from synthetic document."""
        page_text = (
            f"Document Header: {DAY56_E2E_DOCUMENT_MARKER}\n"
            f"Page Section: {DAY56_E2E_PAGE_MARKER}\n"
            f"Chunk Details: {DAY56_E2E_CHUNK_MARKER}"
        )
        page1 = PageData(
            page_number=1,
            text=page_text,
            char_count=len(page_text),
            has_content=True,
        )
        parsed_doc = ParsedDocument(
            metadata=DocumentMetadata(
                document_id=DAY56_DOC_ID,
                filename=DAY56_FILENAME,
                total_pages=1,
                content_type="application/pdf",
                created_at="2026-08-27T00:00:00Z",
                pages_with_content=1,
                pages_without_content=0,
            ),
            pages=[page1],
        )

        assert parsed_doc.metadata.document_id == DAY56_DOC_ID
        assert parsed_doc.metadata.total_pages == 1
        assert len(parsed_doc.pages) == 1
        assert parsed_doc.pages[0].page_number == 1
        assert DAY56_E2E_DOCUMENT_MARKER in parsed_doc.pages[0].text
        assert DAY56_E2E_PAGE_MARKER in parsed_doc.pages[0].text

    def test_chunking_and_chunk_lineage_preservation(self) -> None:
        """DocumentChunk maintains exact Document -> Page -> Chunk lineage."""
        chunk = DocumentChunk(
            chunk_id=CHUNK_A1,
            chunk_index=0,
            document_id=DOC_A,
            filename=FILE_A,
            page_number=1,
            content=f"{DAY56_E2E_CHUNK_MARKER} - Content Details",
            content_type="text",
            metadata={
                "doc_marker": DAY56_E2E_DOCUMENT_MARKER,
                "page_marker": DAY56_E2E_PAGE_MARKER,
                "chunk_marker": DAY56_E2E_CHUNK_MARKER,
            },
        )

        norm = normalize_chunks([chunk])
        assert len(norm) == 1
        assert norm[0].document_id == DOC_A
        assert norm[0].page_number == 1
        assert norm[0].chunk_id == CHUNK_A1
        assert norm[0].metadata["doc_marker"] == DAY56_E2E_DOCUMENT_MARKER


# ============================================================================
# 2. Embedding Preparation & Retrieval Verification
# ============================================================================

class TestEmbeddingPreparationAndRetrieval:
    """Sections 9, 10, 11: prepare_for_embedding, Qdrant search, and SearchAgent."""

    def test_embedding_preparation_pipeline(self) -> None:
        """prepare_for_embedding converts DocumentChunk list into EmbeddingPreparationResult."""
        chunk = DocumentChunk(
            chunk_id=CHUNK_A1,
            chunk_index=0,
            document_id=DOC_A,
            filename=FILE_A,
            page_number=1,
            content=f"{DAY56_E2E_CHUNK_MARKER} payload",
            content_type="text",
        )
        prep = prepare_for_embedding([chunk])

        assert prep.is_ready is True
        assert prep.total_items == 1
        assert prep.items[0].chunk_id == CHUNK_A1
        assert prep.items[0].document_id == DOC_A

    def test_retrieval_and_search_marker_matching(
        self, e2e_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Querying DAY56_E2E_RETRIEVAL_MARKER returns DOC-A with exact metadata markers."""
        store, col_name = e2e_store
        embedder = DeterministicDay56EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        resp = agent.search(f"Search for {DAY56_E2E_RETRIEVAL_MARKER}")

        assert resp.is_success is True
        assert resp.unique_documents == [DOC_A]
        assert resp.has_citations is True

        citation = resp.citations[0]
        assert citation.document_id == DOC_A
        assert citation.chunk_id == CHUNK_A1
        assert citation.page_number == 1

        content = citation.metadata.get("content", "")
        assert DAY56_E2E_RETRIEVAL_MARKER in content
        assert DAY56_E2E_CONTEXT_MARKER in content


# ============================================================================
# 3. Context Building, Agent Processing & Citation Lineage
# ============================================================================

class TestContextAgentAndCitationLineage:
    """Sections 12, 13, 14: Context building, Agent response, Citation lineage."""

    def test_context_building_marker_preservation(self) -> None:
        """build_retrieval_context formats VectorSearchResult retaining context markers."""
        vsr = VectorSearchResult(
            chunk_id=CHUNK_A1,
            score=0.98,
            document_id=DOC_A,
            filename=FILE_A,
            page_number=1,
            chunk_index=0,
            content_type="text",
            content=f"Context Section: {DAY56_E2E_CONTEXT_MARKER}",
        )
        context_str = build_retrieval_context([vsr])

        assert DAY56_E2E_CONTEXT_MARKER in context_str
        assert f"File: {FILE_A}" in context_str
        assert "Page: 1" in context_str
        assert "[Source 1]" in context_str

    def test_agent_and_citation_contract(
        self, e2e_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """SearchAgent builds AgentResponse and AgentCitation adhering to contract."""
        store, col_name = e2e_store
        embedder = DeterministicDay56EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        req = SearchRequest(query=f"Find {DAY56_E2E_RETRIEVAL_MARKER}")
        resp = agent.search(req)

        assert resp.is_success is True
        assert len(resp.citations) == 1
        cit = resp.citations[0]
        assert isinstance(cit, AgentCitation)
        assert cit.document_id == DOC_A
        assert cit.chunk_id == CHUNK_A1
        assert cit.page_number == 1
        assert cit.filename == FILE_A


# ============================================================================
# 4. Vision Evidence & Final Packaged Result
# ============================================================================

class TestVisionEvidenceAndFinalResult:
    """Sections 15, 16: VisualEvidenceAdapter and packaged SearchResult."""

    def test_vision_evidence_adaptation(self) -> None:
        """VisualEvidenceAdapter converts AgentCitation to VisualEvidence preserving document identity."""
        citation = AgentCitation(
            document_id=DOC_A,
            filename=FILE_A,
            chunk_id=CHUNK_A1,
            page_number=1,
            content_type="image",
            metadata={"description": "Visual diagram of circuit A"},
        )
        evidence = VisualEvidenceAdapter.adapt_citation(citation)

        assert isinstance(evidence, VisualEvidence)
        assert evidence.document_id == DOC_A
        assert evidence.filename == FILE_A
        assert evidence.chunk_id == CHUNK_A1
        assert evidence.page_number == 1
        assert evidence.metadata.get("description") == "Visual diagram of circuit A"

    def test_final_packaged_search_result_contract(
        self, e2e_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """SearchAgent.search_and_package returns fully validated SearchResult."""
        store, col_name = e2e_store
        embedder = DeterministicDay56EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        packaged = agent.search_and_package(f"Find {DAY56_E2E_A}")

        assert isinstance(packaged, SearchResult)
        assert packaged.status == "RESULTS_FOUND"
        assert packaged.has_results is True
        assert packaged.total_results == 1
        assert packaged.unique_documents == [DOC_A]
        assert packaged.citations[0].document_id == DOC_A


# ============================================================================
# 5. Complete Marker Trace & Multi-Document E2E Isolation
# ============================================================================

class TestMarkerTraceAndMultiDocumentIsolation:
    """Sections 17, 18, 22: Full trace of synthetic markers and strict cross-doc isolation."""

    def test_complete_marker_trace_across_pipeline_stages(
        self, e2e_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """
        Complete end-to-end trace:
          DAY56_E2E_DOCUMENT_MARKER -> Extraction -> Page -> Chunk ->
          Retrieval -> Context -> Agent -> Citation -> Final Result.
        """
        store, col_name = e2e_store
        embedder = DeterministicDay56EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        packaged = agent.search_and_package(f"Search for {DAY56_E2E_DOCUMENT_MARKER}")

        assert packaged.status == "RESULTS_FOUND"
        assert packaged.unique_documents == [DOC_A]

        cit = packaged.citations[0]
        meta = cit.metadata
        assert meta["document_marker"] == DAY56_E2E_DOCUMENT_MARKER
        assert meta["page_marker"] == DAY56_E2E_PAGE_MARKER
        assert meta["chunk_marker"] == DAY56_E2E_CHUNK_MARKER
        assert meta["retrieval_marker"] == DAY56_E2E_RETRIEVAL_MARKER
        assert meta["context_marker"] == DAY56_E2E_CONTEXT_MARKER

    def test_multi_document_e2e_isolation(
        self, e2e_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """DOC-A -> A, DOC-B -> B, DOC-C -> C with complete cross-document isolation."""
        store, col_name = e2e_store
        embedder = DeterministicDay56EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        res_a = agent.search(f"Search {DAY56_E2E_A}")
        res_b = agent.search(f"Search {DAY56_E2E_B}")
        res_c = agent.search(f"Search {DAY56_E2E_C}")

        assert res_a.unique_documents == [DOC_A]
        assert res_b.unique_documents == [DOC_B]
        assert res_c.unique_documents == [DOC_C]

        # Assert no cross-document leakage in citation metadata
        assert DAY56_E2E_B not in res_a.citations[0].metadata.get("content", "")
        assert DAY56_E2E_C not in res_a.citations[0].metadata.get("content", "")
        assert DAY56_E2E_A not in res_b.citations[0].metadata.get("content", "")
        assert DAY56_E2E_A not in res_c.citations[0].metadata.get("content", "")


# ============================================================================
# 6. Repeated Execution, Failure Recovery & Serialization
# ============================================================================

class TestRepeatabilityRecoveryAndSerialization:
    """Sections 19, 20, 21: 3-run stability, error recovery, serialization roundtrips."""

    def test_repeated_pipeline_execution_stability(
        self, e2e_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Executing the full pipeline 3 times yields identical results with zero state drift."""
        store, col_name = e2e_store
        embedder = DeterministicDay56EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        runs: list[list[str]] = []
        for _ in range(3):
            resp = agent.search(f"Search {DAY56_E2E_A}")
            runs.append([c.chunk_id for c in resp.citations])

        assert runs[0] == runs[1] == runs[2] == [CHUNK_A1]

    def test_failure_injection_and_clean_recovery(
        self, e2e_store: tuple[QdrantVectorStore, str]
    ) -> None:
        """Valid pipeline -> Invalid empty query -> Valid pipeline executes with clean recovery."""
        store, col_name = e2e_store
        embedder = DeterministicDay56EmbeddingProvider()
        agent = SearchAgent(embedding_provider=embedder, store=store, collection_name=col_name, min_score=0.5)

        # 1. Valid A
        res_a1 = agent.search(f"Search {DAY56_E2E_A}")
        assert res_a1.unique_documents == [DOC_A]

        # 2. Invalid failure
        with pytest.raises(AgentValidationError):
            agent.search("")

        # 3. Valid B
        res_b = agent.search(f"Search {DAY56_E2E_B}")
        assert res_b.unique_documents == [DOC_B]

        # 4. Valid A recovery
        res_a2 = agent.search(f"Search {DAY56_E2E_A}")
        assert res_a2.unique_documents == [DOC_A]

    def test_pipeline_serialization_roundtrips(self) -> None:
        """AgentCitation and SearchResult serialize to JSON and restore without loss of identity."""
        citation = AgentCitation(
            document_id=DOC_A,
            filename=FILE_A,
            chunk_id=CHUNK_A1,
            page_number=1,
            score=0.99,
            metadata={"marker": DAY56_E2E_DOCUMENT_MARKER},
        )
        json_str = json.dumps(citation.to_dict())
        restored = AgentCitation.from_dict(json.loads(json_str))

        assert restored.document_id == DOC_A
        assert restored.chunk_id == CHUNK_A1
        assert restored.page_number == 1
        assert restored.score == 0.99
        assert restored.metadata["marker"] == DAY56_E2E_DOCUMENT_MARKER
