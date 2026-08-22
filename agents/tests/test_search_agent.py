"""
Unit and integration tests for SearchAgent.
"""

from __future__ import annotations

import math
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agents.exceptions import AgentExecutionError, AgentValidationError
from agents.models import AgentCitation, AgentRequest, AgentResponse, SearchRequest
from agents.search_agent import SearchAgent
from ingestion.models import (
    EmbeddingVectorRecord,
    RetrievalServiceResult,
    VectorSearchResult,
)
from ingestion.qdrant_config import QdrantConfig
from ingestion.qdrant_store import QdrantVectorStore


class MockEmbeddingProvider:
    """Test stub for EmbeddingProvider."""

    def __init__(self, dimension: int = 4, return_vector: list[float] | None = None) -> None:
        self.dimension = dimension
        self.return_vector = return_vector or [0.1] * dimension

    def embed(self, text: str) -> list[float]:
        return list(self.return_vector)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self.return_vector) for _ in texts]


class MockBatchOnlyEmbeddingProvider:
    """Test stub for provider implementing only embed_batch."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.2] * self.dimension for _ in texts]


@pytest.fixture
def mock_store() -> MagicMock:
    """Fixture providing a mock QdrantVectorStore."""
    store = MagicMock(spec=QdrantVectorStore)
    return store


@pytest.fixture
def sample_search_results() -> list[VectorSearchResult]:
    """Fixture providing real Member 1 VectorSearchResult items."""
    return [
        VectorSearchResult(
            chunk_id="chk-001",
            score=0.92,
            document_id="doc-100",
            filename="financial_report_2023.pdf",
            page_number=3,
            chunk_index=0,
            content_type="text",
            content="Total net sales for the fiscal year reached 12.5 billion USD.",
            metadata={"department": "finance", "char_count": 65},
        ),
        VectorSearchResult(
            chunk_id="chk-002",
            score=0.85,
            document_id="doc-100",
            filename="financial_report_2023.pdf",
            page_number=7,
            chunk_index=4,
            content_type="table",
            content="Revenue by Region: North America 60%, Europe 25%, APAC 15%.",
            metadata={"table_index": 1, "rows": 3},
        ),
    ]


class TestSearchAgentInitialization:
    """Test suite for SearchAgent dependency and configuration validation."""

    def test_successful_initialization(self, mock_store: MagicMock) -> None:
        """Verify initialization with valid dependencies."""
        provider = MockEmbeddingProvider()
        agent = SearchAgent(
            embedding_provider=provider,
            store=mock_store,
            collection_name="test_collection",
            top_k=10,
            min_score=0.25,
            max_results=8,
            agent_name="CustomSearchAgent",
        )
        assert agent.embedding_provider is provider
        assert agent.store is mock_store
        assert agent.collection_name == "test_collection"
        assert agent.top_k == 10
        assert agent.min_score == 0.25
        assert agent.max_results == 8
        assert agent.agent_name == "CustomSearchAgent"

    def test_default_parameters(self, mock_store: MagicMock) -> None:
        """Verify default parameters on initialization."""
        provider = MockEmbeddingProvider()
        agent = SearchAgent(embedding_provider=provider, store=mock_store)
        assert agent.collection_name == "documents"
        assert agent.top_k == 5
        assert agent.min_score == 0.0
        assert agent.max_results == 5
        assert agent.agent_name == "SearchAgent"

    def test_invalid_embedding_provider_raises_validation_error(self, mock_store: MagicMock) -> None:
        """Verify invalid embedding provider is rejected."""
        with pytest.raises(AgentValidationError, match="embedding_provider must implement"):
            SearchAgent(embedding_provider=None, store=mock_store)  # type: ignore[arg-type]

        class InvalidProvider:
            pass

        with pytest.raises(AgentValidationError, match="embedding_provider must implement"):
            SearchAgent(embedding_provider=InvalidProvider(), store=mock_store)  # type: ignore[arg-type]

    def test_invalid_store_raises_validation_error(self) -> None:
        """Verify invalid store is rejected."""
        provider = MockEmbeddingProvider()
        with pytest.raises(AgentValidationError, match="store must be an instance of QdrantVectorStore"):
            SearchAgent(embedding_provider=provider, store=None)  # type: ignore[arg-type]

        with pytest.raises(AgentValidationError, match="store must be an instance of QdrantVectorStore"):
            SearchAgent(embedding_provider=provider, store="not_a_store")  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_col", ["", "   ", None, 123])
    def test_invalid_collection_name_raises_validation_error(
        self, mock_store: MagicMock, invalid_col: Any
    ) -> None:
        """Verify invalid collection name is rejected."""
        provider = MockEmbeddingProvider()
        with pytest.raises(AgentValidationError, match="collection_name must be a non-empty string"):
            SearchAgent(embedding_provider=provider, store=mock_store, collection_name=invalid_col)

    @pytest.mark.parametrize("invalid_top_k", [0, -5, "5", True, False, 3.5])
    def test_invalid_top_k_raises_validation_error(
        self, mock_store: MagicMock, invalid_top_k: Any
    ) -> None:
        """Verify invalid top_k is rejected."""
        provider = MockEmbeddingProvider()
        with pytest.raises(AgentValidationError, match="top_k must be a positive integer"):
            SearchAgent(embedding_provider=provider, store=mock_store, top_k=invalid_top_k)

    @pytest.mark.parametrize("invalid_score", [-1.5, 1.5, "0.5", True, float("nan"), float("inf")])
    def test_invalid_min_score_raises_validation_error(
        self, mock_store: MagicMock, invalid_score: Any
    ) -> None:
        """Verify invalid min_score is rejected."""
        provider = MockEmbeddingProvider()
        with pytest.raises(AgentValidationError, match="min_score must be a finite float"):
            SearchAgent(embedding_provider=provider, store=mock_store, min_score=invalid_score)

    @pytest.mark.parametrize("invalid_max", [0, -1, "10", True, 2.5])
    def test_invalid_max_results_raises_validation_error(
        self, mock_store: MagicMock, invalid_max: Any
    ) -> None:
        """Verify invalid max_results is rejected."""
        provider = MockEmbeddingProvider()
        with pytest.raises(AgentValidationError, match="max_results must be a positive integer"):
            SearchAgent(embedding_provider=provider, store=mock_store, max_results=invalid_max)

    @pytest.mark.parametrize("invalid_name", ["", "   ", None, 123])
    def test_invalid_agent_name_raises_validation_error(
        self, mock_store: MagicMock, invalid_name: Any
    ) -> None:
        """Verify invalid agent_name is rejected."""
        provider = MockEmbeddingProvider()
        with pytest.raises(AgentValidationError, match="agent_name must be a non-empty string"):
            SearchAgent(embedding_provider=provider, store=mock_store, agent_name=invalid_name)


class TestSearchAgentExecution:
    """Test suite for search execution flow, validation, and citation conversion."""

    @patch("agents.search_agent.retrieve_context")
    def test_successful_search_with_string_query(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        sample_search_results: list[VectorSearchResult],
    ) -> None:
        """Verify successful search with raw query string."""
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=sample_search_results,
            context="[Source 1]\nFile: financial_report_2023.pdf\nPage: 3\nType: text\nContent:\nTotal net sales...",
        )

        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store, collection_name="docs")

        response = agent.search("What were the net sales in 2023?")

        assert isinstance(response, AgentResponse)
        assert response.agent_name == "SearchAgent"
        assert response.status == "success"
        assert response.answer == ""  # No LLM answer generated
        assert response.error is None
        assert response.has_citations is True
        assert response.total_citations == 2

        # Verify citation conversion & lineage preservation
        cit1 = response.citations[0]
        assert cit1.chunk_id == "chk-001"
        assert cit1.document_id == "doc-100"
        assert cit1.filename == "financial_report_2023.pdf"
        assert cit1.page_number == 3
        assert cit1.content_type == "text"
        assert cit1.score == 0.92
        assert cit1.metadata == {"department": "finance", "char_count": 65}

        cit2 = response.citations[1]
        assert cit2.chunk_id == "chk-002"
        assert cit2.document_id == "doc-100"
        assert cit2.filename == "financial_report_2023.pdf"
        assert cit2.page_number == 7
        assert cit2.content_type == "table"
        assert cit2.score == 0.85
        assert cit2.metadata == {"table_index": 1, "rows": 3}

        # Verify metadata
        assert response.metadata["query"] == "What were the net sales in 2023?"
        assert "[Source 1]" in response.metadata["context"]
        assert response.metadata["total_results"] == 2
        assert response.metadata["text_results"] == 1
        assert response.metadata["table_results"] == 1
        assert response.metadata["image_results"] == 0
        assert response.metadata["collection_name"] == "docs"

    @patch("agents.search_agent.retrieve_context")
    def test_search_with_agent_request(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        sample_search_results: list[VectorSearchResult],
    ) -> None:
        """Verify search accepting typed AgentRequest with metadata."""
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=sample_search_results,
            context="Formatted context",
        )

        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        req = AgentRequest(
            query="Analyze revenue breakdown",
            session_id="sess-42",
            document_filter={"doc_type": "finance"},
            metadata={"user_tier": "premium"},
        )
        response = agent.search(req)

        assert response.status == "success"
        assert response.total_citations == 2
        assert response.metadata["query"] == "Analyze revenue breakdown"
        assert response.metadata["session_id"] == "sess-42"
        assert response.metadata["document_filter"] == {"doc_type": "finance"}
        assert response.metadata["user_tier"] == "premium"

    @patch("agents.search_agent.retrieve_context")
    def test_lineage_preservation_field_by_field(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify all fields of VectorSearchResult are mapped into AgentCitation without loss."""
        search_res = VectorSearchResult(
            chunk_id="chunk-abc-123",
            score=0.987,
            document_id="doc-xyz-789",
            filename="quarterly_report.pdf",
            page_number=42,
            chunk_index=12,
            content_type="text",
            content="Specific paragraph text for validation.",
            metadata={"source_author": "Finance Team", "verified": True},
        )
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=[search_res],
            context="[Source 1] ...",
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)
        response = agent.search("Audit check")

        assert len(response.citations) == 1
        citation = response.citations[0]
        assert citation.document_id == "doc-xyz-789"
        assert citation.filename == "quarterly_report.pdf"
        assert citation.chunk_id == "chunk-abc-123"
        assert citation.page_number == 42
        assert citation.content_type == "text"
        assert citation.score == 0.987
        assert citation.metadata == {"source_author": "Finance Team", "verified": True}

    @patch("agents.search_agent.retrieve_context")
    def test_search_with_batch_only_provider(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify search works with provider that only implements embed_batch."""
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=[],
            context="",
        )

        provider = MockBatchOnlyEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)
        response = agent.search("Batch test query")
        assert response.status == "success"
        assert response.total_citations == 0

    @patch("agents.search_agent.retrieve_context")
    def test_zero_retrieval_results(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify handling of zero search results without fabricating citations."""
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=[],
            context="",
        )

        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Non-existent entity information")

        assert response.status == "success"
        assert response.citations == []
        assert response.has_citations is False
        assert response.total_citations == 0
        assert response.metadata["context"] == ""
        assert response.metadata["total_results"] == 0

    @patch("agents.search_agent.retrieve_context")
    def test_runtime_parameter_overrides(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify runtime overrides for top_k, min_score, max_results, and collection."""
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=[],
            context="",
        )

        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(
            embedding_provider=provider,
            store=mock_store,
            collection_name="default_coll",
            top_k=5,
            min_score=0.0,
            max_results=5,
        )

        agent.search(
            "Query with overrides",
            top_k=20,
            min_score=0.75,
            max_results=10,
            collection_name="custom_coll",
        )

        mock_retrieve_context.assert_called_once_with(
            query_vector=[0.1, 0.1, 0.1, 0.1],
            store=mock_store,
            collection_name="custom_coll",
            top_k=20,
            min_score=0.75,
            max_results=10,
        )

    def test_callable_and_run_aliases(self, mock_store: MagicMock) -> None:
        """Verify agent instance is callable and provides .run() method."""
        with patch.object(SearchAgent, "search") as mock_search:
            mock_search.return_value = AgentResponse(
                answer="",
                agent_name="SearchAgent",
                status="success",
            )
            provider = MockEmbeddingProvider()
            agent = SearchAgent(embedding_provider=provider, store=mock_store)

            resp1 = agent("Callable query")
            assert mock_search.call_count == 1

            resp2 = agent.run("Run query")
            assert mock_search.call_count == 2


class TestSearchAgentValidationAndErrorHandling:
    """Test suite for query validation and error handling."""

    @pytest.mark.parametrize("invalid_query", ["", "   ", "\t\n", None, 12345, [1, 2], {"q": "text"}])
    def test_invalid_query_raises_validation_error(
        self, mock_store: MagicMock, invalid_query: Any
    ) -> None:
        """Verify invalid query values raise AgentValidationError."""
        provider = MockEmbeddingProvider()
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError):
            agent.search(invalid_query)

    def test_invalid_runtime_params_raise_validation_error(self, mock_store: MagicMock) -> None:
        """Verify invalid runtime parameter overrides raise AgentValidationError."""
        provider = MockEmbeddingProvider()
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError, match="top_k must be a positive integer"):
            agent.search("Valid query", top_k=-1)

        with pytest.raises(AgentValidationError, match="min_score must be a finite float"):
            agent.search("Valid query", min_score=2.0)

        with pytest.raises(AgentValidationError, match="max_results must be a positive integer"):
            agent.search("Valid query", max_results=0)

    def test_embedding_provider_exception_raises_execution_error(
        self, mock_store: MagicMock
    ) -> None:
        """Verify exceptions during embedding generation raise AgentExecutionError."""
        failing_provider = MagicMock()
        failing_provider.embed.side_effect = RuntimeError("Embedding service unavailable")

        agent = SearchAgent(embedding_provider=failing_provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Failed to generate query embedding"):
            agent.search("Query triggering error")

    def test_embedding_provider_invalid_vector_raises_execution_error(
        self, mock_store: MagicMock
    ) -> None:
        """Verify non-numeric or empty vector from provider raises AgentExecutionError."""
        provider = MagicMock()
        provider.embed.return_value = ["not", "numeric"]

        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Failed to generate query embedding"):
            agent.search("Query with invalid vector")

    @patch("agents.search_agent.retrieve_context")
    def test_retrieval_service_exception_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify exceptions during vector retrieval raise AgentExecutionError."""
        mock_retrieve_context.side_effect = ConnectionError("Qdrant unreachable")

        provider = MockEmbeddingProvider()
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Retrieval execution failed"):
            agent.search("Query with retrieval failure")


class TestSearchAgentMember1Integration:
    """Integration test suite verifying end-to-end compatibility with real Member 1 components."""

    def test_real_qdrant_store_integration(self) -> None:
        """Verify SearchAgent against real in-memory QdrantVectorStore and Member 1 models."""
        config = QdrantConfig(url=":memory:")
        real_store = QdrantVectorStore(config=config)
        collection_name = "test_member1_integration"

        # Create collection
        real_store.create_collection(collection_name=collection_name, vector_dimension=4)

        # Upsert real Member 1 vector records
        chunk_id_1 = str(uuid.uuid4())
        chunk_id_2 = str(uuid.uuid4())
        records = [
            EmbeddingVectorRecord(
                chunk_id=chunk_id_1,
                document_id="doc-real-10",
                filename="annual_report.pdf",
                chunk_index=0,
                page_number=1,
                content_type="text",
                vector=[1.0, 0.0, 0.0, 0.0],
                metadata={"section": "Executive Summary", "content": "Total net income reached 4.2 billion USD."},
            ),
            EmbeddingVectorRecord(
                chunk_id=chunk_id_2,
                document_id="doc-real-10",
                filename="annual_report.pdf",
                chunk_index=1,
                page_number=2,
                content_type="table",
                vector=[0.0, 1.0, 0.0, 0.0],
                metadata={"table_index": 0, "content": "Table: Operating margins."},
            ),
        ]
        from ingestion.models import EmbeddingGenerationResult

        gen_result = EmbeddingGenerationResult(
            document_id="doc-real-10",
            filename="annual_report.pdf",
            items=records,
            dimension=4,
            is_ready=True,
        )
        real_store.upsert_embeddings(collection_name=collection_name, result=gen_result)

        # Create provider returning vector pointing towards chunk 1
        class DirectionalProvider:
            def embed(self, text: str) -> list[float]:
                return [0.9, 0.1, 0.0, 0.0]

        agent = SearchAgent(
            embedding_provider=DirectionalProvider(),
            store=real_store,
            collection_name=collection_name,
            top_k=5,
            min_score=0.5,
            max_results=2,
        )

        response = agent.search("Executive overview")

        assert response.status == "success"
        assert response.total_citations >= 1
        top_cit = response.citations[0]
        assert top_cit.chunk_id == chunk_id_1
        assert top_cit.document_id == "doc-real-10"
        assert top_cit.filename == "annual_report.pdf"
        assert top_cit.page_number == 1
        assert top_cit.content_type == "text"
        assert top_cit.metadata == {"section": "Executive Summary", "content": "Total net income reached 4.2 billion USD."}
        assert "annual_report.pdf" in response.metadata["context"]


class TestSearchRequestContract:
    """Test suite for SearchRequest contract, validation, and conversion helpers."""

    def test_minimal_search_request(self) -> None:
        """Verify minimal SearchRequest creation with query only."""
        req = SearchRequest(query="Quarterly revenue breakdown")
        assert req.query == "Quarterly revenue breakdown"
        assert req.top_k is None
        assert req.min_score is None
        assert req.max_results is None
        assert req.collection_name is None
        assert req.session_id is None
        assert req.document_filter is None
        assert req.metadata == {}

    def test_full_search_request(self) -> None:
        """Verify SearchRequest creation with all fields."""
        req = SearchRequest(
            query="Detailed balance sheet",
            top_k=15,
            min_score=0.35,
            max_results=8,
            collection_name="financial_docs",
            session_id="session-xyz",
            document_filter={"year": 2023},
            metadata={"user_tier": "enterprise"},
        )
        assert req.query == "Detailed balance sheet"
        assert req.top_k == 15
        assert req.min_score == 0.35
        assert req.max_results == 8
        assert req.collection_name == "financial_docs"
        assert req.session_id == "session-xyz"
        assert req.document_filter == {"year": 2023}
        assert req.metadata == {"user_tier": "enterprise"}

    @pytest.mark.parametrize("invalid_query", ["", "   ", "\t\n", None, 1234, [1, 2]])
    def test_invalid_query_raises_validation_error(self, invalid_query: Any) -> None:
        """Verify invalid query values raise AgentValidationError."""
        with pytest.raises(AgentValidationError):
            SearchRequest(query=invalid_query)

    @pytest.mark.parametrize("invalid_top_k", [0, -1, "10", True, 2.5])
    def test_invalid_top_k_raises_validation_error(self, invalid_top_k: Any) -> None:
        """Verify invalid top_k raises AgentValidationError."""
        with pytest.raises(AgentValidationError, match="top_k must be a positive integer"):
            SearchRequest(query="Valid query", top_k=invalid_top_k)

    @pytest.mark.parametrize("invalid_score", [-1.5, 1.5, "0.5", True, float("nan"), float("inf")])
    def test_invalid_min_score_raises_validation_error(self, invalid_score: Any) -> None:
        """Verify invalid min_score raises AgentValidationError."""
        with pytest.raises(AgentValidationError, match="min_score must be a finite float"):
            SearchRequest(query="Valid query", min_score=invalid_score)

    @pytest.mark.parametrize("invalid_max", [0, -2, "5", True, 4.5])
    def test_invalid_max_results_raises_validation_error(self, invalid_max: Any) -> None:
        """Verify invalid max_results raises AgentValidationError."""
        with pytest.raises(AgentValidationError, match="max_results must be a positive integer"):
            SearchRequest(query="Valid query", max_results=invalid_max)

    @pytest.mark.parametrize("invalid_col", ["", "   ", 123])
    def test_invalid_collection_raises_validation_error(self, invalid_col: Any) -> None:
        """Verify invalid collection_name raises AgentValidationError."""
        with pytest.raises(AgentValidationError, match="collection_name must be a non-empty string"):
            SearchRequest(query="Valid query", collection_name=invalid_col)

    @pytest.mark.parametrize("invalid_session", ["", "   ", 456])
    def test_invalid_session_id_raises_validation_error(self, invalid_session: Any) -> None:
        """Verify invalid session_id raises AgentValidationError."""
        with pytest.raises(AgentValidationError, match="session_id must be a non-empty string"):
            SearchRequest(query="Valid query", session_id=invalid_session)

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """Verify dictionary serialization roundtrip."""
        req = SearchRequest(
            query="Testing roundtrip",
            top_k=7,
            min_score=0.4,
            max_results=3,
            collection_name="test_col",
            session_id="sess-007",
            document_filter=["doc1", "doc2"],
            metadata={"source": "api"},
        )
        d = req.to_dict()
        reconstructed = SearchRequest.from_dict(d)
        assert reconstructed == req

    def test_conversion_to_and_from_agent_request(self) -> None:
        """Verify conversion to and from AgentRequest."""
        search_req = SearchRequest(
            query="Convert to AgentRequest",
            top_k=10,
            min_score=0.5,
            max_results=5,
            collection_name="contracts",
            session_id="sess-1",
            document_filter={"type": "pdf"},
            metadata={"caller": "supervisor"},
        )
        agent_req = search_req.to_agent_request()
        assert isinstance(agent_req, AgentRequest)
        assert agent_req.query == "Convert to AgentRequest"
        assert agent_req.session_id == "sess-1"
        assert agent_req.document_filter == {"type": "pdf"}
        assert agent_req.metadata["top_k"] == 10
        assert agent_req.metadata["min_score"] == 0.5
        assert agent_req.metadata["max_results"] == 5
        assert agent_req.metadata["collection_name"] == "contracts"
        assert agent_req.metadata["caller"] == "supervisor"

        reconstructed_search_req = SearchRequest.from_agent_request(agent_req)
        assert reconstructed_search_req.query == search_req.query
        assert reconstructed_search_req.top_k == search_req.top_k
        assert reconstructed_search_req.min_score == search_req.min_score
        assert reconstructed_search_req.max_results == search_req.max_results
        assert reconstructed_search_req.collection_name == search_req.collection_name
        assert reconstructed_search_req.session_id == search_req.session_id
        assert reconstructed_search_req.document_filter == search_req.document_filter
        assert reconstructed_search_req.metadata == {"caller": "supervisor"}


class TestSearchAgentProductionHardening:
    """Test suite for Day 23 production hardening, result normalization, and error boundaries."""

    @patch("agents.search_agent.retrieve_context")
    def test_search_with_search_request_instance(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        sample_search_results: list[VectorSearchResult],
    ) -> None:
        """Verify executing search using typed SearchRequest."""
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=sample_search_results,
            context="Hardened context string",
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(
            embedding_provider=provider,
            store=mock_store,
            collection_name="default_col",
            top_k=5,
            min_score=0.0,
            max_results=5,
        )

        req = SearchRequest(
            query="Search request execution",
            top_k=12,
            min_score=0.6,
            max_results=4,
            collection_name="override_col",
            session_id="session-prod-1",
            document_filter={"scope": "global"},
            metadata={"trace_id": "tr-987"},
        )

        response = agent.search(req)

        assert response.status == "success"
        assert response.total_citations == 2
        assert response.metadata["query"] == "Search request execution"
        assert response.metadata["top_k"] == 12
        assert response.metadata["min_score"] == 0.6
        assert response.metadata["max_results"] == 4
        assert response.metadata["collection_name"] == "override_col"
        assert response.metadata["session_id"] == "session-prod-1"
        assert response.metadata["document_filter"] == {"scope": "global"}
        assert response.metadata["trace_id"] == "tr-987"
        assert response.metadata["results_by_modality"] == {"text": 1, "table": 1, "image": 0}

        mock_retrieve_context.assert_called_once_with(
            query_vector=[0.1, 0.1, 0.1, 0.1],
            store=mock_store,
            collection_name="override_col",
            top_k=12,
            min_score=0.6,
            max_results=4,
        )

    @patch("agents.search_agent.retrieve_context")
    def test_explicit_argument_precedence_over_search_request(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify explicit kwargs to search() take precedence over SearchRequest attributes."""
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=[],
            context="",
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        req = SearchRequest(query="Precedence test", top_k=8, min_score=0.2, max_results=3, collection_name="req_col")
        agent.search(req, top_k=25, min_score=0.8, max_results=10, collection_name="kwarg_col")

        mock_retrieve_context.assert_called_once_with(
            query_vector=[0.1, 0.1, 0.1, 0.1],
            store=mock_store,
            collection_name="kwarg_col",
            top_k=25,
            min_score=0.8,
            max_results=10,
        )

    @patch("agents.search_agent.retrieve_context")
    def test_non_retrieval_service_result_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify unexpected response type from retrieval service raises AgentExecutionError."""
        mock_retrieve_context.return_value = "invalid string response"
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Expected RetrievalServiceResult"):
            agent.search("Type safety check")

    @patch("agents.search_agent.retrieve_context")
    def test_non_list_results_in_service_result_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify non-list results attribute raises AgentExecutionError."""
        mock_retrieve_context.return_value = MagicMock(
            spec=RetrievalServiceResult,
            results="not_a_list",
            context="",
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Expected list of results"):
            agent.search("Type safety check")

    @patch("agents.search_agent.retrieve_context")
    def test_non_vector_search_result_item_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify non-VectorSearchResult item in results raises AgentExecutionError."""
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=["invalid_result_item"],  # type: ignore[list-item]
            context="",
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="not a VectorSearchResult"):
            agent.search("Type safety check")

    @patch("agents.search_agent.retrieve_context")
    def test_multi_modality_normalization(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify multi-modality evidence (text, table, image) is properly normalized."""
        results = [
            VectorSearchResult(
                chunk_id="chk-text",
                score=0.95,
                document_id="doc-1",
                filename="report.pdf",
                page_number=1,
                chunk_index=0,
                content_type="text",
                content="Text section content.",
            ),
            VectorSearchResult(
                chunk_id="chk-tbl",
                score=0.88,
                document_id="doc-1",
                filename="report.pdf",
                page_number=2,
                chunk_index=1,
                content_type="table",
                content="Table content.",
            ),
            VectorSearchResult(
                chunk_id="chk-img",
                score=0.82,
                document_id="doc-1",
                filename="report.pdf",
                page_number=3,
                chunk_index=2,
                content_type="image",
                content="Image content.",
            ),
        ]
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=results,
            context="[Source 1] Text\n\n[Source 2] Table\n\n[Source 3] Image",
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Multi modal query")

        assert response.status == "success"
        assert response.total_citations == 3
        assert response.metadata["total_results"] == 3
        assert response.metadata["text_results"] == 1
        assert response.metadata["table_results"] == 1
        assert response.metadata["image_results"] == 1
        assert response.metadata["results_by_modality"] == {"text": 1, "table": 1, "image": 1}
        assert response.citations[0].content_type == "text"
        assert response.citations[1].content_type == "table"
        assert response.citations[2].content_type == "image"
