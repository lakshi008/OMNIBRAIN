"""
OmniBrain Member 4 — Day 2 Ingestion -> Search Contract Integration Tests.

Verifies the existing handoff and data contracts between:
- Member 1 (Ingestion subsystem: VectorSearchResult, DocumentChunk, RetrievalServiceResult)
- Member 2 (Search subsystem: AgentCitation, SearchRequest, SearchResult, AgentResponse, SearchAgent)

Ensures that:
1. Valid Member 1 ingestion & retrieval outputs are correctly ingested and transformed by Member 2.
2. Complete document identity, chunk identity, page lineage, content, and metadata are preserved without data loss or corruption.
3. Multi-document outputs maintain strict isolation without cross-document data leakage.
4. Malformed or invalid upstream data triggers expected validation/execution errors.
5. All operations are deterministic, offline, and side-effect free.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure repo root is on sys.path for test runners executing this file directly
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

from agents.exceptions import AgentExecutionError, AgentValidationError
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    SearchRequest,
    SearchResult,
)
from agents.search_agent import SearchAgent
from ingestion.models import (
    DocumentChunk,
    RetrievalServiceResult,
    VectorSearchResult,
)
from ingestion.qdrant_store import QdrantVectorStore


class MockEmbeddingProvider:
    """Deterministic in-memory embedding provider stub."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        return [0.1] * self.dimension

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]


# ============================================================================
# 1. SINGLE DOCUMENT INGESTION -> SEARCH HANDOFF
# ============================================================================


class TestSingleDocumentIngestionToSearchHandoff:
    """Verifies that a single document's chunks/search results correctly translate to Search citations."""

    def test_single_chunk_lineage_preservation(self) -> None:
        """Verify single chunk handoff preserves exact document identity, chunk identity, page, and metadata."""
        vs_result = VectorSearchResult(
            chunk_id="chunk-alpha-001",
            score=0.95,
            document_id="doc-alpha",
            filename="financial_report.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="OmniBrain quarterly revenue increased by 20%.",
            metadata={"department": "finance", "fiscal_year": 2024, "char_count": 46},
        )

        citation = AgentCitation.from_search_result(vs_result)

        assert citation.document_id == "doc-alpha"
        assert citation.filename == "financial_report.pdf"
        assert citation.chunk_id == "chunk-alpha-001"
        assert citation.page_number == 1
        assert citation.content_type == "text"
        assert citation.score == 0.95
        assert citation.metadata["department"] == "finance"
        assert citation.metadata["fiscal_year"] == 2024
        assert citation.metadata["char_count"] == 46

    def test_multi_chunk_multi_page_handoff(self) -> None:
        """Verify multiple chunks across multiple pages maintain correct sequential lineage."""
        chunks = [
            VectorSearchResult(
                chunk_id=f"chunk-page-{page}-idx-{idx}",
                score=0.90 - (idx * 0.05),
                document_id="doc-multi-page",
                filename="technical_manual.pdf",
                page_number=page,
                chunk_index=idx,
                content_type="text",
                content=f"Technical specification section on page {page}.",
                metadata={"section_id": f"sec_{idx}"},
            )
            for page in range(1, 4)
            for idx in range(2)
        ]

        citations = [AgentCitation.from_search_result(c) for c in chunks]

        assert len(citations) == 6
        for i, cit in enumerate(citations):
            expected_page = (i // 2) + 1
            expected_idx = i % 2
            assert cit.document_id == "doc-multi-page"
            assert cit.filename == "technical_manual.pdf"
            assert cit.page_number == expected_page
            assert cit.chunk_id == f"chunk-page-{expected_page}-idx-{expected_idx}"
            assert cit.metadata["section_id"] == f"sec_{expected_idx}"

    def test_search_result_packaging_from_ingestion_results(self) -> None:
        """Verify SearchResult packaging preserves ordered citations, query, and context."""
        vs_results = [
            VectorSearchResult(
                chunk_id="chunk-01",
                score=0.96,
                document_id="doc-pkg-1",
                filename="contract.pdf",
                page_number=1,
                chunk_index=0,
                content_type="text",
                content="Clause 1: Service Level Agreement terms.",
                metadata={"clause": 1},
            ),
            VectorSearchResult(
                chunk_id="chunk-02",
                score=0.88,
                document_id="doc-pkg-1",
                filename="contract.pdf",
                page_number=2,
                chunk_index=1,
                content_type="text",
                content="Clause 2: Payment and billing schedules.",
                metadata={"clause": 2},
            ),
        ]

        citations = [AgentCitation.from_search_result(r) for r in vs_results]
        search_result = SearchResult(
            query="What are the SLA terms and payment schedule?",
            status="RESULTS_FOUND",
            citations=citations,
            context="[Source 1] Clause 1: Service Level Agreement terms.\n[Source 2] Clause 2: Payment and billing schedules.",
            metadata={"engine": "qdrant"},
        )

        assert search_result.has_results is True
        assert search_result.total_results == 2
        assert search_result.evidence_count == 2
        assert search_result.text_count == 2
        assert search_result.unique_document_count == 1
        assert search_result.unique_documents == ["doc-pkg-1"]
        assert search_result.citations[0].chunk_id == "chunk-01"
        assert search_result.citations[1].chunk_id == "chunk-02"


# ============================================================================
# 2. MULTIMODAL INGESTION CONTRACT COMPATIBILITY
# ============================================================================


class TestMultimodalIngestionSearchContract:
    """Verifies that all ingestion modalities (text, table, image) map cleanly to Search contracts."""

    def test_text_table_and_image_modalities(self) -> None:
        """Verify text, table, and image chunks are accurately represented and grouped in SearchResult."""
        results = [
            VectorSearchResult(
                chunk_id="chk-text",
                score=0.95,
                document_id="doc-multimodal",
                filename="multimodal_doc.pdf",
                page_number=1,
                chunk_index=0,
                content_type="text",
                content="Executive summary text paragraph.",
                metadata={"modality": "text"},
            ),
            VectorSearchResult(
                chunk_id="chk-table",
                score=0.90,
                document_id="doc-multimodal",
                filename="multimodal_doc.pdf",
                page_number=2,
                chunk_index=1,
                content_type="table",
                content="| Year | Growth |\n| 2023 | 15% |",
                metadata={"modality": "table", "rows": 2, "cols": 2},
            ),
            VectorSearchResult(
                chunk_id="chk-image",
                score=0.85,
                document_id="doc-multimodal",
                filename="multimodal_doc.pdf",
                page_number=3,
                chunk_index=2,
                content_type="image",
                content="System architecture block diagram",
                metadata={"modality": "image", "image_path": "B:/tmp/arch.png"},
            ),
        ]

        citations = [AgentCitation.from_search_result(r) for r in results]
        search_pkg = SearchResult(
            query="Summarize performance and architecture",
            status="RESULTS_FOUND",
            citations=citations,
            context="Multimodal synthesized context block",
        )

        assert search_pkg.text_count == 1
        assert search_pkg.table_count == 1
        assert search_pkg.image_count == 1
        assert len(search_pkg.text_results) == 1
        assert len(search_pkg.table_results) == 1
        assert len(search_pkg.image_results) == 1
        assert search_pkg.text_results[0].chunk_id == "chk-text"
        assert search_pkg.table_results[0].chunk_id == "chk-table"
        assert search_pkg.image_results[0].chunk_id == "chk-image"

        by_modality = search_pkg.by_modality
        assert len(by_modality["text"]) == 1
        assert len(by_modality["table"]) == 1
        assert len(by_modality["image"]) == 1


# ============================================================================
# 3. MULTI-DOCUMENT ISOLATION & PROVENANCE
# ============================================================================


class TestMultiDocumentIsolation:
    """Verifies that chunks from distinct documents maintain strict isolation without cross-talk."""

    def test_multi_document_association_and_grouping(self) -> None:
        """Verify citations from multiple documents are cleanly segregated and grouped."""
        doc_a_results = [
            VectorSearchResult(
                chunk_id="doc-a-c1",
                score=0.98,
                document_id="doc-A",
                filename="policy_a.pdf",
                page_number=1,
                chunk_index=0,
                content_type="text",
                content="Policy A: Employee onboarding guidelines.",
                metadata={"dept": "HR"},
            ),
            VectorSearchResult(
                chunk_id="doc-a-c2",
                score=0.91,
                document_id="doc-A",
                filename="policy_a.pdf",
                page_number=2,
                chunk_index=1,
                content_type="text",
                content="Policy A: Code of conduct rules.",
                metadata={"dept": "HR"},
            ),
        ]

        doc_b_results = [
            VectorSearchResult(
                chunk_id="doc-b-c1",
                score=0.88,
                document_id="doc-B",
                filename="security_b.pdf",
                page_number=1,
                chunk_index=0,
                content_type="text",
                content="Security B: Password complexity standards.",
                metadata={"dept": "IT Security"},
            ),
            VectorSearchResult(
                chunk_id="doc-b-c2",
                score=0.82,
                document_id="doc-B",
                filename="security_b.pdf",
                page_number=3,
                chunk_index=1,
                content_type="text",
                content="Security B: Multi-factor authentication policy.",
                metadata={"dept": "IT Security"},
            ),
        ]

        all_results = doc_a_results + doc_b_results
        citations = [AgentCitation.from_search_result(r) for r in all_results]

        search_result = SearchResult(
            query="Company policies and security guidelines",
            status="RESULTS_FOUND",
            citations=citations,
            context="Combined context from both policy documents.",
        )

        assert search_result.total_results == 4
        assert search_result.unique_document_count == 2
        assert search_result.unique_documents == ["doc-A", "doc-B"]

        by_doc = search_result.by_document
        assert "doc-A" in by_doc
        assert "doc-B" in by_doc
        assert len(by_doc["doc-A"]) == 2
        assert len(by_doc["doc-B"]) == 2

        # Verify Doc A citations contain only Doc A data
        for cit in by_doc["doc-A"]:
            assert cit.document_id == "doc-A"
            assert cit.filename == "policy_a.pdf"
            assert cit.metadata["dept"] == "HR"
            assert "Policy A" in cit.chunk_id or "doc-a" in cit.chunk_id

        # Verify Doc B citations contain only Doc B data
        for cit in by_doc["doc-B"]:
            assert cit.document_id == "doc-B"
            assert cit.filename == "security_b.pdf"
            assert cit.metadata["dept"] == "IT Security"
            assert "Security B" in cit.chunk_id or "doc-b" in cit.chunk_id


# ============================================================================
# 4. SEARCH AGENT INTEGRATION & RESULT INTEGRITY VERIFICATION
# ============================================================================


class TestSearchAgentIntegrityValidation:
    """Verifies that SearchAgent executes Member 1 result integrity validation properly."""

    def test_search_agent_result_validation_success(self) -> None:
        """Verify SearchAgent._validate_result_integrity passes for fully compliant VectorSearchResult."""
        valid_item = VectorSearchResult(
            chunk_id="chunk-valid-01",
            score=0.92,
            document_id="doc-valid-01",
            filename="report.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Compliant content text.",
            metadata={"key": "value"},
        )
        # Should execute without raising
        SearchAgent._validate_result_integrity(valid_item, idx=0)

    def test_search_agent_result_policy_deduplication_and_ranking(self) -> None:
        """Verify SearchAgent._apply_member2_result_policy deduplicates, sorts, and caps correctly."""
        results = [
            VectorSearchResult(
                chunk_id="chunk-dup",
                score=0.80,
                document_id="doc-1",
                filename="f.pdf",
                page_number=1,
                chunk_index=0,
                content_type="text",
                content="Lower score duplicate.",
            ),
            VectorSearchResult(
                chunk_id="chunk-dup",
                score=0.95,
                document_id="doc-1",
                filename="f.pdf",
                page_number=1,
                chunk_index=0,
                content_type="text",
                content="Higher score duplicate.",
            ),
            VectorSearchResult(
                chunk_id="chunk-unique-2",
                score=0.90,
                document_id="doc-1",
                filename="f.pdf",
                page_number=2,
                chunk_index=1,
                content_type="text",
                content="Second unique chunk.",
            ),
            VectorSearchResult(
                chunk_id="chunk-unique-3",
                score=0.85,
                document_id="doc-1",
                filename="f.pdf",
                page_number=3,
                chunk_index=2,
                content_type="text",
                content="Third unique chunk.",
            ),
        ]

        # Request cap of 2
        filtered = SearchAgent._apply_member2_result_policy(results, max_results=2)

        assert len(filtered) == 2
        # Highest scored chunk-dup (0.95) should be first
        assert filtered[0].chunk_id == "chunk-dup"
        assert filtered[0].score == 0.95
        # Second should be chunk-unique-2 (0.90)
        assert filtered[1].chunk_id == "chunk-unique-2"
        assert filtered[1].score == 0.90

    def test_search_agent_build_evidence_context(self) -> None:
        """Verify SearchAgent._build_evidence_context produces structured citation-numbered text."""
        items = [
            VectorSearchResult(
                chunk_id="c1",
                score=0.95,
                document_id="doc1",
                filename="file1.pdf",
                page_number=1,
                chunk_index=0,
                content_type="text",
                content="Primary finding.",
            ),
            VectorSearchResult(
                chunk_id="c2",
                score=0.90,
                document_id="doc1",
                filename="file1.pdf",
                page_number=2,
                chunk_index=1,
                content_type="text",
                content="Secondary finding.",
            ),
        ]

        context = SearchAgent._build_evidence_context(items)
        assert "[Source 1]" in context
        assert "File: file1.pdf" in context
        assert "Page: 1" in context
        assert "Type: text" in context
        assert "Primary finding." in context
        assert "[Source 2]" in context
        assert "Page: 2" in context
        assert "Secondary finding." in context


# ============================================================================
# 5. INVALID INPUT & ERROR BOUNDARY TESTS
# ============================================================================


class TestInvalidInputHandling:
    """Verifies that malformed upstream data raises expected domain errors."""

    def test_missing_document_id_rejection_in_citation(self) -> None:
        """Verify AgentCitation rejects empty or missing document_id."""
        with pytest.raises(AgentValidationError, match="document_id"):
            AgentCitation(
                document_id="",
                filename="test.pdf",
                chunk_id="c1",
                page_number=1,
                content_type="text",
            )

    def test_missing_chunk_id_rejection_in_citation(self) -> None:
        """Verify AgentCitation rejects empty or missing chunk_id."""
        with pytest.raises(AgentValidationError, match="chunk_id"):
            AgentCitation(
                document_id="doc1",
                filename="test.pdf",
                chunk_id="   ",
                page_number=1,
                content_type="text",
            )

    def test_missing_filename_rejection_in_citation(self) -> None:
        """Verify AgentCitation rejects empty or missing filename."""
        with pytest.raises(AgentValidationError, match="filename"):
            AgentCitation(
                document_id="doc1",
                filename="",
                chunk_id="c1",
                page_number=1,
                content_type="text",
            )

    def test_invalid_page_number_rejection_in_citation(self) -> None:
        """Verify AgentCitation rejects non-positive or boolean page_number."""
        with pytest.raises(AgentValidationError, match="page_number"):
            AgentCitation(
                document_id="doc1",
                filename="test.pdf",
                chunk_id="c1",
                page_number=0,  # Must be >= 1 or None
                content_type="text",
            )

        with pytest.raises(AgentValidationError, match="page_number"):
            AgentCitation(
                document_id="doc1",
                filename="test.pdf",
                chunk_id="c1",
                page_number=True,  # Bools not allowed
                content_type="text",
            )

    def test_invalid_score_rejection_in_citation(self) -> None:
        """Verify AgentCitation rejects NaN, infinite, or boolean scores."""
        with pytest.raises(AgentValidationError, match="score"):
            AgentCitation(
                document_id="doc1",
                filename="test.pdf",
                chunk_id="c1",
                page_number=1,
                content_type="text",
                score=float("nan"),
            )

        with pytest.raises(AgentValidationError, match="score"):
            AgentCitation(
                document_id="doc1",
                filename="test.pdf",
                chunk_id="c1",
                page_number=1,
                content_type="text",
                score=float("inf"),
            )

    def test_search_agent_integrity_rejects_missing_chunk_id(self) -> None:
        """Verify SearchAgent._validate_result_integrity raises AgentExecutionError on missing chunk_id."""
        invalid_item = VectorSearchResult(
            chunk_id="",
            score=0.9,
            document_id="doc1",
            filename="f.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Sample text",
        )
        with pytest.raises(AgentExecutionError, match="chunk_id is missing or empty"):
            SearchAgent._validate_result_integrity(invalid_item, idx=0)

    def test_search_agent_integrity_rejects_missing_document_id(self) -> None:
        """Verify SearchAgent._validate_result_integrity raises AgentExecutionError on missing document_id."""
        invalid_item = VectorSearchResult(
            chunk_id="c1",
            score=0.9,
            document_id="   ",
            filename="f.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Sample text",
        )
        with pytest.raises(AgentExecutionError, match="document_id is missing or empty"):
            SearchAgent._validate_result_integrity(invalid_item, idx=0)

    def test_search_agent_integrity_rejects_empty_content(self) -> None:
        """Verify SearchAgent._validate_result_integrity raises AgentExecutionError on empty content."""
        invalid_item = VectorSearchResult(
            chunk_id="c1",
            score=0.9,
            document_id="doc1",
            filename="f.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="   ",
        )
        with pytest.raises(AgentExecutionError, match="content is empty or missing"):
            SearchAgent._validate_result_integrity(invalid_item, idx=0)

    def test_search_agent_integrity_rejects_negative_chunk_index(self) -> None:
        """Verify SearchAgent._validate_result_integrity raises AgentExecutionError on negative chunk_index."""
        invalid_item = VectorSearchResult(
            chunk_id="c1",
            score=0.9,
            document_id="doc1",
            filename="f.pdf",
            page_number=1,
            chunk_index=-1,
            content_type="text",
            content="Sample text",
        )
        with pytest.raises(AgentExecutionError, match="chunk_index must be a non-negative integer"):
            SearchAgent._validate_result_integrity(invalid_item, idx=0)


# ============================================================================
# 6. DETERMINISTIC REPEATABILITY & IMMUTABILITY
# ============================================================================


class TestDeterministicBehavior:
    """Verifies that contract transformation is strictly deterministic and free of mutations."""

    def test_repeated_citation_conversions_identical(self) -> None:
        """Verify multiple conversions of the same VectorSearchResult produce identical immutable citations."""
        item = VectorSearchResult(
            chunk_id="chk-det-01",
            score=0.93,
            document_id="doc-det-01",
            filename="deterministic.pdf",
            page_number=5,
            chunk_index=3,
            content_type="text",
            content="Deterministic content verification.",
            metadata={"seed": 42, "checksum": "abc123xyz"},
        )

        cit1 = AgentCitation.from_search_result(item)
        cit2 = AgentCitation.from_search_result(item)
        cit3 = AgentCitation.from_search_result(item)

        assert cit1 == cit2 == cit3
        assert cit1.to_dict() == cit2.to_dict() == cit3.to_dict()

    def test_input_object_immutability(self) -> None:
        """Verify upstream VectorSearchResult is never mutated during conversion or search packaging."""
        original_meta = {"key": "immutable_val", "nested": {"a": 1}}
        item = VectorSearchResult(
            chunk_id="chk-imm-01",
            score=0.90,
            document_id="doc-imm",
            filename="imm.pdf",
            page_number=1,
            chunk_index=0,
            content_type="text",
            content="Immutable content.",
            metadata=dict(original_meta),
        )

        citation = AgentCitation.from_search_result(item)
        citation.metadata["key"] = "modified_in_citation"

        # Upstream object must remain untouched
        assert item.metadata["key"] == "immutable_val"
        assert item.chunk_id == "chk-imm-01"
        assert item.score == 0.90
