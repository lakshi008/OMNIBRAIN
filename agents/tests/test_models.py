"""
Unit tests for agent domain models.
"""

from __future__ import annotations

import math
from typing import Any
import pytest

from agents.exceptions import AgentValidationError
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    AgentState,
)
from ingestion.models import VectorSearchResult


class TestAgentRequest:
    """Test suite for AgentRequest model."""

    def test_valid_minimal_request(self) -> None:
        """Verify creation with minimal valid query."""
        req = AgentRequest(query="What was the total revenue in 2023?")
        assert req.query == "What was the total revenue in 2023?"
        assert req.session_id is None
        assert req.document_filter is None
        assert req.metadata == {}

    def test_valid_request_with_all_optional_fields(self) -> None:
        """Verify creation with all optional fields provided."""
        req = AgentRequest(
            query="Summarize executive compensation.",
            session_id="session-12345",
            document_filter={"document_id": "doc-abc"},
            metadata={"priority": "high", "timeout_sec": 30},
        )
        assert req.query == "Summarize executive compensation."
        assert req.session_id == "session-12345"
        assert req.document_filter == {"document_id": "doc-abc"}
        assert req.metadata == {"priority": "high", "timeout_sec": 30}

    def test_empty_query_raises_validation_error(self) -> None:
        """Verify empty query string is rejected."""
        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            AgentRequest(query="")

    def test_whitespace_query_raises_validation_error(self) -> None:
        """Verify whitespace-only query is rejected."""
        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            AgentRequest(query="   \t\n  ")

    def test_non_string_query_raises_validation_error(self) -> None:
        """Verify non-string query is rejected."""
        with pytest.raises(AgentValidationError, match="query must be a string"):
            AgentRequest(query=12345)  # type: ignore[arg-type]

        with pytest.raises(AgentValidationError, match="query must be a string"):
            AgentRequest(query=None)  # type: ignore[arg-type]

    def test_invalid_session_id_raises_validation_error(self) -> None:
        """Verify empty or non-string session_id is rejected."""
        with pytest.raises(AgentValidationError, match="session_id must be a non-empty string"):
            AgentRequest(query="valid query", session_id="")

        with pytest.raises(AgentValidationError, match="session_id must be a non-empty string"):
            AgentRequest(query="valid query", session_id=123)  # type: ignore[arg-type]

    def test_invalid_document_filter_type_raises_validation_error(self) -> None:
        """Verify non-dict, non-list document_filter is rejected."""
        with pytest.raises(AgentValidationError, match="document_filter must be a dict"):
            AgentRequest(query="valid query", document_filter=12345)  # type: ignore[arg-type]

    def test_invalid_metadata_type_raises_validation_error(self) -> None:
        """Verify non-dict metadata is rejected."""
        with pytest.raises(AgentValidationError, match="metadata must be a dictionary"):
            AgentRequest(query="valid query", metadata="not a dict")  # type: ignore[arg-type]

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """Verify serialization to and from dictionary."""
        original = AgentRequest(
            query="Analyze Q3 earnings report",
            session_id="session-999",
            document_filter=["doc-1", "doc-2"],
            metadata={"user_role": "analyst"},
        )
        d = original.to_dict()
        assert d == {
            "query": "Analyze Q3 earnings report",
            "session_id": "session-999",
            "document_filter": ["doc-1", "doc-2"],
            "metadata": {"user_role": "analyst"},
        }
        reconstructed = AgentRequest.from_dict(d)
        assert reconstructed == original

    def test_from_dict_with_invalid_data_raises_validation_error(self) -> None:
        """Verify from_dict fails when passed non-dict."""
        with pytest.raises(AgentValidationError, match="Input data must be a dictionary"):
            AgentRequest.from_dict(["not a dict"])  # type: ignore[arg-type]


class TestAgentCitation:
    """Test suite for AgentCitation model and Member 1 lineage preservation."""

    def test_valid_citation(self) -> None:
        """Verify valid citation construction."""
        citation = AgentCitation(
            document_id="doc-uuid-1234",
            filename="financial_report_2023.pdf",
            chunk_id="chunk-uuid-5678",
            page_number=12,
            content_type="table",
            score=0.895,
            metadata={"table_index": 0, "rows": 5},
        )
        assert citation.document_id == "doc-uuid-1234"
        assert citation.filename == "financial_report_2023.pdf"
        assert citation.chunk_id == "chunk-uuid-5678"
        assert citation.page_number == 12
        assert citation.content_type == "table"
        assert citation.score == 0.895
        assert citation.metadata == {"table_index": 0, "rows": 5}

    def test_citation_with_none_page_number(self) -> None:
        """Verify citation supports page_number=None (e.g. document-level content)."""
        citation = AgentCitation(
            document_id="doc-1",
            filename="doc.pdf",
            chunk_id="chunk-1",
            page_number=None,
        )
        assert citation.page_number is None

    @pytest.mark.parametrize(
        ("field_name", "kwargs"),
        [
            ("document_id", {"document_id": "", "filename": "f.pdf", "chunk_id": "c1"}),
            ("document_id", {"document_id": "   ", "filename": "f.pdf", "chunk_id": "c1"}),
            ("document_id", {"document_id": 123, "filename": "f.pdf", "chunk_id": "c1"}),
            ("filename", {"document_id": "d1", "filename": "", "chunk_id": "c1"}),
            ("filename", {"document_id": "d1", "filename": "   ", "chunk_id": "c1"}),
            ("filename", {"document_id": "d1", "filename": None, "chunk_id": "c1"}),
            ("chunk_id", {"document_id": "d1", "filename": "f.pdf", "chunk_id": ""}),
            ("chunk_id", {"document_id": "d1", "filename": "f.pdf", "chunk_id": "   "}),
            ("chunk_id", {"document_id": "d1", "filename": "f.pdf", "chunk_id": None}),
            ("content_type", {"document_id": "d1", "filename": "f.pdf", "chunk_id": "c1", "content_type": ""}),
        ],
    )
    def test_missing_or_invalid_string_fields_raise_error(
        self, field_name: str, kwargs: dict[str, Any]
    ) -> None:
        """Verify invalid or empty string attributes raise AgentValidationError."""
        with pytest.raises(AgentValidationError, match=f"{field_name} must be a non-empty string"):
            AgentCitation(**kwargs)

    @pytest.mark.parametrize("invalid_page", [0, -1, -100, 1.5, True, False, "1", [1]])
    def test_invalid_page_number_raises_error(self, invalid_page: Any) -> None:
        """Verify invalid page numbers raise AgentValidationError."""
        with pytest.raises(AgentValidationError, match="page_number must be a positive integer"):
            AgentCitation(
                document_id="doc-1",
                filename="f.pdf",
                chunk_id="chunk-1",
                page_number=invalid_page,
            )

    @pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), float("-inf"), "0.9", True, None])
    def test_invalid_score_raises_error(self, invalid_score: Any) -> None:
        """Verify non-finite or non-numeric scores raise AgentValidationError."""
        with pytest.raises(AgentValidationError, match="score must be a finite numeric float"):
            AgentCitation(
                document_id="doc-1",
                filename="f.pdf",
                chunk_id="chunk-1",
                score=invalid_score,
            )

    def test_integer_score_coerced_to_float(self) -> None:
        """Verify integer score is accepted and converted to float."""
        citation = AgentCitation(
            document_id="doc-1",
            filename="f.pdf",
            chunk_id="chunk-1",
            score=1,
        )
        assert citation.score == 1.0
        assert isinstance(citation.score, float)

    def test_from_member_1_search_result(self) -> None:
        """Verify compatibility and provenance preservation when constructing from Member 1 VectorSearchResult."""
        search_res = VectorSearchResult(
            chunk_id="chunk-999",
            score=0.942,
            document_id="doc-777",
            filename="annual_report.pdf",
            page_number=4,
            chunk_index=15,
            content_type="text",
            content="Total net income reached 4.2 billion USD.",
            metadata={"section": "Financial Highlights", "char_count": 42},
        )

        citation = AgentCitation.from_search_result(search_res)
        assert citation.document_id == "doc-777"
        assert citation.filename == "annual_report.pdf"
        assert citation.chunk_id == "chunk-999"
        assert citation.page_number == 4
        assert citation.content_type == "text"
        assert citation.score == 0.942
        assert citation.metadata == {"section": "Financial Highlights", "char_count": 42}

    def test_from_search_result_missing_attribute_raises_error(self) -> None:
        """Verify from_search_result fails if required attribute is missing."""
        class IncompleteResult:
            document_id = "doc-1"
            filename = "f.pdf"
            # chunk_id is missing

        with pytest.raises(AgentValidationError, match="missing required attribute 'chunk_id'"):
            AgentCitation.from_search_result(IncompleteResult())

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """Verify serialization to and from dictionary representation."""
        citation = AgentCitation(
            document_id="d-1",
            filename="doc.pdf",
            chunk_id="c-1",
            page_number=3,
            content_type="image",
            score=0.78,
            metadata={"width": 800, "height": 600},
        )
        d = citation.to_dict()
        assert d == {
            "document_id": "d-1",
            "filename": "doc.pdf",
            "chunk_id": "c-1",
            "page_number": 3,
            "content_type": "image",
            "score": 0.78,
            "metadata": {"width": 800, "height": 600},
        }
        reconstructed = AgentCitation.from_dict(d)
        assert reconstructed == citation


class TestAgentResponse:
    """Test suite for AgentResponse model."""

    def test_successful_response(self) -> None:
        """Verify successful response structure and helper properties."""
        citation = AgentCitation(
            document_id="doc-1",
            filename="report.pdf",
            chunk_id="chunk-1",
            page_number=5,
            score=0.91,
        )
        resp = AgentResponse(
            answer="The revenue grew by 15% in 2023.",
            agent_name="SearchAgent",
            status="success",
            citations=[citation],
            metadata={"latency_ms": 250},
        )
        assert resp.answer == "The revenue grew by 15% in 2023."
        assert resp.agent_name == "SearchAgent"
        assert resp.status == "success"
        assert resp.citations == [citation]
        assert resp.has_citations is True
        assert resp.total_citations == 1
        assert resp.is_success is True
        assert resp.is_error is False
        assert resp.error is None

    def test_failure_response(self) -> None:
        """Verify failure response representation and error properties."""
        resp = AgentResponse(
            answer="",
            agent_name="SupervisorAgent",
            status="error",
            error="Vector store query timed out.",
        )
        assert resp.is_success is False
        assert resp.is_error is True
        assert resp.error == "Vector store query timed out."
        assert resp.has_citations is False
        assert resp.total_citations == 0

    def test_invalid_citations_list_raises_error(self) -> None:
        """Verify non-AgentCitation items in citations list raise AgentValidationError."""
        with pytest.raises(AgentValidationError, match="Item at index 0 of citations is not an AgentCitation"):
            AgentResponse(
                answer="Sample answer",
                agent_name="Agent",
                citations=["not a citation"],  # type: ignore[list-item]
            )

    def test_invalid_agent_name_raises_error(self) -> None:
        """Verify empty agent_name raises error."""
        with pytest.raises(AgentValidationError, match="agent_name must be a non-empty string"):
            AgentResponse(answer="Answer", agent_name="")

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """Verify serialization and deserialization of response with citations."""
        citation = AgentCitation(
            document_id="doc-10",
            filename="financials.pdf",
            chunk_id="chk-10",
            page_number=2,
            score=0.88,
        )
        original = AgentResponse(
            answer="Operating profit reached 2.1M.",
            agent_name="RetrievalAgent",
            status="success",
            citations=[citation],
            metadata={"model": "claude"},
        )
        d = original.to_dict()
        reconstructed = AgentResponse.from_dict(d)
        assert reconstructed.answer == original.answer
        assert reconstructed.agent_name == original.agent_name
        assert reconstructed.status == original.status
        assert reconstructed.citations == original.citations
        assert reconstructed.metadata == original.metadata


class TestAgentState:
    """Test suite for AgentState workflow context."""

    def test_initial_state_defaults(self) -> None:
        """Verify initial default values of AgentState."""
        state = AgentState(query="Explain the tax liability.")
        assert state.query == "Explain the tax liability."
        assert state.route is None
        assert state.retrieved_results == []
        assert state.context == ""
        assert state.citations == []
        assert state.answer == ""
        assert state.errors == []
        assert state.status == "initialized"
        assert state.metadata == {}

    def test_empty_query_raises_validation_error(self) -> None:
        """Verify empty query string raises validation error on state creation."""
        with pytest.raises(AgentValidationError, match="query cannot be empty"):
            AgentState(query="  ")

    def test_add_error_and_add_citation(self) -> None:
        """Verify adding error messages and citations to state."""
        state = AgentState(query="Summarize findings")
        state.add_error("First fallback triggered")
        assert state.errors == ["First fallback triggered"]

        citation = AgentCitation(
            document_id="d1",
            filename="f.pdf",
            chunk_id="c1",
            page_number=1,
            score=0.9,
        )
        state.add_citation(citation)
        assert state.citations == [citation]

    def test_add_invalid_error_raises_validation_error(self) -> None:
        """Verify adding non-string or empty error raises error."""
        state = AgentState(query="Query")
        with pytest.raises(AgentValidationError, match="error must be a non-empty string"):
            state.add_error("")

        with pytest.raises(AgentValidationError, match="error must be a non-empty string"):
            state.add_error(123)  # type: ignore[arg-type]

    def test_add_invalid_citation_raises_validation_error(self) -> None:
        """Verify adding non-AgentCitation raises error."""
        state = AgentState(query="Query")
        with pytest.raises(AgentValidationError, match="citation must be an AgentCitation"):
            state.add_citation({"doc": "dict"}  # type: ignore[arg-type]
            )

    def test_update_state(self) -> None:
        """Verify updating state attributes via update method."""
        state = AgentState(query="Initial query")
        state.update(
            route="retrieval",
            context="Formatted source context",
            answer="Final generated answer",
            status="completed",
        )
        assert state.route == "retrieval"
        assert state.context == "Formatted source context"
        assert state.answer == "Final generated answer"
        assert state.status == "completed"

    def test_update_unknown_attribute_raises_validation_error(self) -> None:
        """Verify updating non-existent field raises error."""
        state = AgentState(query="Query")
        with pytest.raises(AgentValidationError, match="Unknown state attribute: non_existent"):
            state.update(non_existent="value")

    def test_to_dict_representation(self) -> None:
        """Verify state conversion to dictionary."""
        citation = AgentCitation(
            document_id="d1",
            filename="doc.pdf",
            chunk_id="c1",
            page_number=1,
            score=0.95,
        )
        state = AgentState(
            query="Analyze balance sheet",
            route="search",
            context="Source context",
            citations=[citation],
            answer="Net assets increased.",
            errors=["Minor notice"],
            status="completed",
            metadata={"duration_sec": 1.2},
        )
        d = state.to_dict()
        assert d["query"] == "Analyze balance sheet"
        assert d["route"] == "search"
        assert d["context"] == "Source context"
        assert len(d["citations"]) == 1
        assert d["citations"][0]["chunk_id"] == "c1"
        assert d["answer"] == "Net assets increased."
        assert d["errors"] == ["Minor notice"]
        assert d["status"] == "completed"
        assert d["metadata"] == {"duration_sec": 1.2}
