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


# ---------------------------------------------------------------------------
# Day 24 — Evidence Quality & Citation Integrity
# ---------------------------------------------------------------------------


def _make_valid_result(**overrides: Any) -> VectorSearchResult:
    """Return a valid VectorSearchResult with any field replaced via overrides."""
    defaults: dict[str, Any] = dict(
        chunk_id="chk-valid-001",
        score=0.75,
        document_id="doc-valid-001",
        filename="evidence.pdf",
        page_number=1,
        chunk_index=0,
        content_type="text",
        content="Valid evidence content.",
        metadata={"source": "unit-test"},
    )
    defaults.update(overrides)
    return VectorSearchResult(**defaults)


def _make_retrieval_result(results: list[VectorSearchResult]) -> RetrievalServiceResult:
    """Wrap a list of VectorSearchResult into a RetrievalServiceResult."""
    return RetrievalServiceResult(
        query_vector_dimension=4,
        results=results,
        context="[Source 1] Valid evidence content.",
    )


class TestEvidenceQualityAndCitationIntegrity:
    """Day 24 — Boundary-level evidence integrity and citation fidelity tests.

    Verifies that SearchAgent._validate_result_integrity rejects any
    VectorSearchResult with malformed lineage fields before citation conversion
    and that valid results produce citations with exact, unmodified lineage.
    """

    # ------------------------------------------------------------------
    # 1. Rejection: empty / whitespace chunk_id
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_chunk_id", ["", "   ", "\t", "\n"])
    @patch("agents.search_agent.retrieve_context")
    def test_empty_chunk_id_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        bad_chunk_id: str,
    ) -> None:
        """Verify that a VectorSearchResult with an empty chunk_id is rejected."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(chunk_id=bad_chunk_id)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="chunk_id is missing or empty"):
            agent.search("Test evidence integrity")

    # ------------------------------------------------------------------
    # 2. Rejection: empty / whitespace document_id
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_doc_id", ["", "   ", "\t"])
    @patch("agents.search_agent.retrieve_context")
    def test_empty_document_id_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        bad_doc_id: str,
    ) -> None:
        """Verify that a VectorSearchResult with an empty document_id is rejected."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(document_id=bad_doc_id)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="document_id is missing or empty"):
            agent.search("Test document lineage")

    # ------------------------------------------------------------------
    # 3. Rejection: empty / whitespace filename
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_filename", ["", "   ", "\t"])
    @patch("agents.search_agent.retrieve_context")
    def test_empty_filename_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        bad_filename: str,
    ) -> None:
        """Verify that a VectorSearchResult with an empty filename is rejected."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(filename=bad_filename)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="filename is missing or empty"):
            agent.search("Test filename attribution")

    # ------------------------------------------------------------------
    # 4. Rejection: non-finite score (NaN)
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_nan_score_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify that a VectorSearchResult with a NaN score is rejected."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(score=float("nan"))]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="score is not a finite numeric value"):
            agent.search("NaN score query")

    # ------------------------------------------------------------------
    # 5. Rejection: non-finite score (Inf)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_score", [float("inf"), float("-inf")])
    @patch("agents.search_agent.retrieve_context")
    def test_infinite_score_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        bad_score: float,
    ) -> None:
        """Verify that a VectorSearchResult with an infinite score is rejected."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(score=bad_score)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="score is not a finite numeric value"):
            agent.search("Infinite score query")

    # ------------------------------------------------------------------
    # 6. Rejection: empty / whitespace content_type
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_ct", ["", "   ", "\t"])
    @patch("agents.search_agent.retrieve_context")
    def test_empty_content_type_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        bad_ct: str,
    ) -> None:
        """Verify that a VectorSearchResult with an empty content_type is rejected."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(content_type=bad_ct)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="content_type is missing or empty"):
            agent.search("Content type routing query")

    # ------------------------------------------------------------------
    # 7. Rejection: negative chunk_index
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_idx", [-1, -100])
    @patch("agents.search_agent.retrieve_context")
    def test_negative_chunk_index_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        bad_idx: int,
    ) -> None:
        """Verify that a VectorSearchResult with a negative chunk_index is rejected."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(chunk_index=bad_idx)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="chunk_index must be a non-negative integer"):
            agent.search("Chunk index query")

    # ------------------------------------------------------------------
    # 8. Score boundary acceptance: -1.0 and 1.0 are valid
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("boundary_score", [-1.0, 0.0, 1.0])
    @patch("agents.search_agent.retrieve_context")
    def test_boundary_scores_are_accepted(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        boundary_score: float,
    ) -> None:
        """Verify that boundary score values (-1.0, 0.0, 1.0) pass integrity validation."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(score=boundary_score)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Boundary score query")

        assert response.status == "success"
        assert response.total_citations == 1
        assert response.citations[0].score == boundary_score

    # ------------------------------------------------------------------
    # 9. Score exact preservation — no rounding or modification
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_score_is_exactly_preserved_from_member1(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify the Member 1 similarity score is propagated to AgentCitation without modification."""
        precise_score = 0.8761234567890123
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(score=precise_score)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Score preservation query")

        assert response.citations[0].score == precise_score

    # ------------------------------------------------------------------
    # 10. Ranking order preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_ranking_order_is_preserved_in_citations(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify citations are produced in exact Member 1 ranking order (highest score first)."""
        results = [
            _make_valid_result(chunk_id="chk-rank-1", score=0.95, document_id="doc-A", filename="a.pdf"),
            _make_valid_result(chunk_id="chk-rank-2", score=0.87, document_id="doc-B", filename="b.pdf"),
            _make_valid_result(chunk_id="chk-rank-3", score=0.72, document_id="doc-C", filename="c.pdf"),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Ranking order query")

        assert response.total_citations == 3
        assert response.citations[0].chunk_id == "chk-rank-1"
        assert response.citations[0].score == 0.95
        assert response.citations[1].chunk_id == "chk-rank-2"
        assert response.citations[1].score == 0.87
        assert response.citations[2].chunk_id == "chk-rank-3"
        assert response.citations[2].score == 0.72

    # ------------------------------------------------------------------
    # 11. Exact lineage propagation to AgentCitation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_exact_lineage_propagated_to_citation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify all VectorSearchResult lineage fields are faithfully copied to AgentCitation."""
        result = _make_valid_result(
            chunk_id="chk-lineage-99",
            document_id="doc-lineage-42",
            filename="annual_report_2024.pdf",
            page_number=17,
            content_type="table",
            score=0.931,
            metadata={"section": "financial", "rows": 5},
        )
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Lineage propagation query")

        assert response.total_citations == 1
        citation = response.citations[0]
        assert citation.chunk_id == "chk-lineage-99"
        assert citation.document_id == "doc-lineage-42"
        assert citation.filename == "annual_report_2024.pdf"
        assert citation.page_number == 17
        assert citation.content_type == "table"
        assert citation.score == 0.931
        assert citation.metadata["section"] == "financial"
        assert citation.metadata["rows"] == 5

    # ------------------------------------------------------------------
    # 12. Malformed item at non-zero index aborts entire response
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_malformed_item_at_index_1_aborts_response(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify a malformed result at index 1 aborts the whole response (no partial citations)."""
        results = [
            _make_valid_result(chunk_id="chk-good", score=0.9),
            _make_valid_result(chunk_id="chk-bad", score=float("nan")),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="score is not a finite numeric value"):
            agent.search("Partial abort query")

    # ------------------------------------------------------------------
    # 13. Error message includes the 0-based index of the failing result
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_error_message_includes_result_index(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify the integrity error message identifies which result index failed."""
        results = [
            _make_valid_result(chunk_id="chk-ok-0"),
            _make_valid_result(chunk_id="chk-ok-1"),
            _make_valid_result(chunk_id="", score=0.8),  # index 2 fails
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Result at index 2"):
            agent.search("Index identification query")

    # ------------------------------------------------------------------
    # 14. _validate_result_integrity is publicly accessible as a static method
    # ------------------------------------------------------------------

    def test_validate_result_integrity_is_static_method(self) -> None:
        """Verify _validate_result_integrity can be invoked as a standalone static method."""
        valid = _make_valid_result()
        # Must not raise for a fully valid item
        SearchAgent._validate_result_integrity(valid, 0)

    # ------------------------------------------------------------------
    # 15. chunk_index=0 is accepted (zero is valid)
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_chunk_index_zero_is_accepted(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Verify chunk_index=0 passes integrity validation (zero is the minimum valid value)."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(chunk_index=0)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Zero chunk index query")
        assert response.status == "success"
        assert response.total_citations == 1


# ---------------------------------------------------------------------------
# Day 25 — Query Handling & Retrieval Orchestration
# ---------------------------------------------------------------------------


class TestDay25QueryHandlingAndOrchestration:
    """Day 25 — Comprehensive query handling and retrieval orchestration tests.

    Covers all 36 required scenarios: query validation, embedding validation,
    retrieval orchestration, citation fidelity, determinism, Member 1 API reuse,
    multimodal handling, and partial evidence failure.
    """

    # ------------------------------------------------------------------
    # Scenario 1 — Valid query produces successful AgentResponse
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_01_valid_query_produces_success_response(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Valid query produces a well-formed AgentResponse with status='success'."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(chunk_id="chk-s01", score=0.88)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("What is the revenue for Q3?")

        assert isinstance(response, AgentResponse)
        assert response.status == "success"
        assert response.error is None
        assert response.answer == ""
        assert response.agent_name == "SearchAgent"

    # ------------------------------------------------------------------
    # Scenario 2 — Empty query string
    # ------------------------------------------------------------------

    def test_02_empty_query_raises_validation_error(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Empty string query raises AgentValidationError."""
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError):
            agent.search("")

    # ------------------------------------------------------------------
    # Scenario 3 — Whitespace-only query string
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("ws_query", ["   ", "\t", "\n", "  \t  \n"])
    def test_03_whitespace_only_query_raises_validation_error(
        self,
        mock_store: MagicMock,
        ws_query: str,
    ) -> None:
        """Whitespace-only query raises AgentValidationError."""
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError):
            agent.search(ws_query)

    # ------------------------------------------------------------------
    # Scenario 4 — None query
    # ------------------------------------------------------------------

    def test_04_none_query_raises_validation_error(
        self,
        mock_store: MagicMock,
    ) -> None:
        """None query raises AgentValidationError."""
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError):
            agent.search(None)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Scenario 5 — Non-string query
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("non_str", [42, 3.14, ["list"], {"dict": 1}, True])
    def test_05_non_string_query_raises_validation_error(
        self,
        mock_store: MagicMock,
        non_str: Any,
    ) -> None:
        """Non-string query raises AgentValidationError."""
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError):
            agent.search(non_str)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Scenario 6 — Query normalization: strip but preserve content
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_06_query_normalization_strips_whitespace_preserves_content(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Leading/trailing whitespace is stripped; meaningful content is preserved."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("  What is the net profit?  ")

        # Query stored in metadata must be stripped
        assert response.metadata["query"] == "What is the net profit?"

    # ------------------------------------------------------------------
    # Scenario 7 — Valid embedding passed to retrieve_context
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_07_valid_embedding_passed_to_retrieval(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """The exact vector from embed() is forwarded to retrieve_context."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4, return_vector=[0.25, 0.50, 0.75, 1.0])
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search("Embedding pass-through check")

        call_kwargs = mock_retrieve_context.call_args.kwargs
        assert call_kwargs["query_vector"] == [0.25, 0.50, 0.75, 1.0]

    # ------------------------------------------------------------------
    # Scenario 8 — Empty embedding from provider raises error
    # ------------------------------------------------------------------

    def test_08_empty_embedding_from_provider_raises_execution_error(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Provider returning empty list raises AgentExecutionError."""
        provider = MagicMock()
        provider.embed.return_value = []

        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Failed to generate query embedding"):
            agent.search("Empty embedding query")

    # ------------------------------------------------------------------
    # Scenario 9 — Malformed embedding (non-numeric values)
    # ------------------------------------------------------------------

    def test_09_malformed_embedding_non_numeric_raises_execution_error(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Provider returning non-numeric values raises AgentExecutionError."""
        provider = MagicMock()
        provider.embed.return_value = ["a", "b", "c", "d"]

        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Failed to generate query embedding"):
            agent.search("Malformed embedding query")

    # ------------------------------------------------------------------
    # Scenario 10 — NaN in embedding
    # ------------------------------------------------------------------

    def test_10_nan_in_embedding_raises_execution_error(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Provider returning NaN in vector raises AgentExecutionError."""
        provider = MagicMock()
        provider.embed.return_value = [0.1, float("nan"), 0.3, 0.4]

        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Failed to generate query embedding"):
            agent.search("NaN in embedding")

    # ------------------------------------------------------------------
    # Scenario 11 — Infinite value in embedding
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("inf_val", [float("inf"), float("-inf")])
    def test_11_infinite_value_in_embedding_raises_execution_error(
        self,
        mock_store: MagicMock,
        inf_val: float,
    ) -> None:
        """Provider returning Inf/−Inf in vector raises AgentExecutionError."""
        provider = MagicMock()
        provider.embed.return_value = [0.1, inf_val, 0.3, 0.4]

        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Failed to generate query embedding"):
            agent.search("Infinite embedding query")

    # ------------------------------------------------------------------
    # Scenario 12 — Embedding dimension mismatch
    # ------------------------------------------------------------------

    def test_12_embedding_dimension_mismatch_raises_execution_error(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Vector dimension != expected_dimension raises AgentExecutionError."""
        provider = MockEmbeddingProvider(dimension=4)  # returns 4-d vector
        agent = SearchAgent(
            embedding_provider=provider,
            store=mock_store,
            expected_dimension=768,  # expects 768-d
        )

        with pytest.raises(AgentExecutionError, match="dimension mismatch"):
            agent.search("Dimension mismatch query")

    def test_12b_matching_dimension_passes(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Vector dimension == expected_dimension passes validation."""
        with patch("agents.search_agent.retrieve_context") as mock_rc:
            mock_rc.return_value = _make_retrieval_result([])
            provider = MockEmbeddingProvider(dimension=4)
            agent = SearchAgent(
                embedding_provider=provider,
                store=mock_store,
                expected_dimension=4,
            )
            response = agent.search("Correct dimension")
            assert response.status == "success"

    def test_12c_invalid_expected_dimension_raises_validation_error(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Invalid expected_dimension value raises AgentValidationError at init."""
        provider = MockEmbeddingProvider(dimension=4)

        with pytest.raises(AgentValidationError, match="expected_dimension"):
            SearchAgent(embedding_provider=provider, store=mock_store, expected_dimension=0)

        with pytest.raises(AgentValidationError, match="expected_dimension"):
            SearchAgent(embedding_provider=provider, store=mock_store, expected_dimension=-1)

        with pytest.raises(AgentValidationError, match="expected_dimension"):
            SearchAgent(embedding_provider=provider, store=mock_store, expected_dimension=True)

    # ------------------------------------------------------------------
    # Scenario 13 — Embedding provider raises exception
    # ------------------------------------------------------------------

    def test_13_embedding_provider_exception_wrapped_in_execution_error(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Exception from provider.embed() is wrapped in AgentExecutionError."""
        provider = MagicMock()
        provider.embed.side_effect = RuntimeError("GPU OOM")

        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Failed to generate query embedding"):
            agent.search("GPU failure query")

    # ------------------------------------------------------------------
    # Scenario 14 — Retrieval service called with correct parameters
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_14_retrieval_service_called_with_correct_parameters(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """retrieve_context receives exact query_vector, store, collection, top_k, min_score, max_results."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4, return_vector=[0.1, 0.2, 0.3, 0.4])
        agent = SearchAgent(
            embedding_provider=provider,
            store=mock_store,
            collection_name="legal_docs",
            top_k=10,
            min_score=0.3,
            max_results=6,
        )

        agent.search("Parameter verification query")

        mock_retrieve_context.assert_called_once_with(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            store=mock_store,
            collection_name="legal_docs",
            top_k=10,
            min_score=0.3,
            max_results=6,
        )

    # ------------------------------------------------------------------
    # Scenario 15 — top_k handling via SearchRequest
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_15_top_k_from_search_request_forwarded_to_retrieval(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """top_k from SearchRequest is forwarded to retrieve_context."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store, top_k=5)

        req = SearchRequest(query="Top-k test", top_k=20)
        agent.search(req)

        assert mock_retrieve_context.call_args.kwargs["top_k"] == 20

    # ------------------------------------------------------------------
    # Scenario 16 — Invalid top_k at runtime
    # ------------------------------------------------------------------

    def test_16_invalid_runtime_top_k_raises_validation_error(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Invalid top_k override at call time raises AgentValidationError."""
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError, match="top_k must be a positive integer"):
            agent.search("Invalid top-k", top_k=0)

        with pytest.raises(AgentValidationError, match="top_k must be a positive integer"):
            agent.search("Invalid top-k", top_k=-5)

    # ------------------------------------------------------------------
    # Scenario 17 — min_score handling
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_17_min_score_forwarded_correctly_to_retrieval(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """min_score from SearchRequest overrides default and is forwarded."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store, min_score=0.0)

        req = SearchRequest(query="Min score test", min_score=0.65)
        agent.search(req)

        assert mock_retrieve_context.call_args.kwargs["min_score"] == 0.65

    # ------------------------------------------------------------------
    # Scenario 18 — Invalid min_score at runtime
    # ------------------------------------------------------------------

    def test_18_invalid_runtime_min_score_raises_validation_error(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Invalid min_score override at call time raises AgentValidationError."""
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError, match="min_score must be a finite float"):
            agent.search("Invalid min_score", min_score=2.0)

        with pytest.raises(AgentValidationError, match="min_score must be a finite float"):
            agent.search("Invalid min_score", min_score=float("nan"))

    # ------------------------------------------------------------------
    # Scenario 19 — Successful retrieval produces AgentResponse
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_19_successful_retrieval_produces_agent_response(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Successful retrieval flow produces a well-formed AgentResponse."""
        result = _make_valid_result(
            chunk_id="chk-s19",
            document_id="doc-s19",
            filename="s19_report.pdf",
            score=0.91,
        )
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Successful retrieval")

        assert isinstance(response, AgentResponse)
        assert response.status == "success"
        assert response.total_citations == 1
        assert response.citations[0].chunk_id == "chk-s19"
        assert response.citations[0].document_id == "doc-s19"

    # ------------------------------------------------------------------
    # Scenario 20 — Zero retrieval results
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_20_zero_retrieval_results_returns_empty_citations(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Zero results from retrieval → valid response with 0 citations, no exception."""
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=[],
            context="",
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Query with no matching documents")

        assert response.status == "success"
        assert response.citations == []
        assert response.has_citations is False
        assert response.total_citations == 0
        assert response.metadata["total_results"] == 0
        assert response.metadata["context"] == ""

    # ------------------------------------------------------------------
    # Scenario 21 — Retrieval failure
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_21_retrieval_failure_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Exception from retrieve_context is wrapped in AgentExecutionError."""
        mock_retrieve_context.side_effect = ConnectionError("Qdrant cluster unreachable")
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Retrieval execution failed"):
            agent.search("Retrieval failure scenario")

    # ------------------------------------------------------------------
    # Scenario 22 — Malformed retrieval result (non-VectorSearchResult item)
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_22_malformed_retrieval_result_item_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Non-VectorSearchResult item in results list raises AgentExecutionError."""
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=["raw_string_result"],  # type: ignore[list-item]
            context="",
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="not a VectorSearchResult"):
            agent.search("Malformed result item")

    # ------------------------------------------------------------------
    # Scenario 23 — Citation creation: list of AgentCitation objects
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_23_citations_are_agent_citation_instances(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Every item in response.citations is an AgentCitation instance."""
        results = [
            _make_valid_result(chunk_id="chk-c1", score=0.9),
            _make_valid_result(chunk_id="chk-c2", score=0.8),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Citation type check")

        assert len(response.citations) == 2
        for cit in response.citations:
            assert isinstance(cit, AgentCitation)

    # ------------------------------------------------------------------
    # Scenario 24 — Citation lineage preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_24_citation_lineage_preserved_from_vector_search_result(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """All VectorSearchResult lineage fields are faithfully propagated to AgentCitation."""
        result = _make_valid_result(
            chunk_id="lineage-chk",
            document_id="lineage-doc",
            filename="lineage.pdf",
            page_number=12,
            chunk_index=3,
            content_type="table",
            score=0.777,
            metadata={"verified": True, "author": "QA"},
        )
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Lineage check")

        cit = response.citations[0]
        assert cit.chunk_id == "lineage-chk"
        assert cit.document_id == "lineage-doc"
        assert cit.filename == "lineage.pdf"
        assert cit.page_number == 12
        assert cit.content_type == "table"
        assert cit.score == 0.777
        assert cit.metadata["verified"] is True
        assert cit.metadata["author"] == "QA"

    # ------------------------------------------------------------------
    # Scenario 25 — Context preservation: Member 1 context passed through
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_25_member1_context_preserved_in_response_metadata(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """The context string from Member 1 RetrievalServiceResult is stored as-is."""
        expected_context = (
            "[Source 1]\nFile: evidence.pdf\nPage: 1\nType: text\nContent:\nValid evidence content."
        )
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=[_make_valid_result()],
            context=expected_context,
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Context preservation test")

        assert response.metadata["context"] == expected_context

    # ------------------------------------------------------------------
    # Scenario 26 — Result count in metadata
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_26_result_count_reported_accurately_in_metadata(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """total_results, text_results, table_results, image_results reflect actual data."""
        results = [
            _make_valid_result(chunk_id="chk-t1", content_type="text", score=0.9),
            _make_valid_result(chunk_id="chk-t2", content_type="text", score=0.8),
            _make_valid_result(chunk_id="chk-tb1", content_type="table", score=0.7),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Count accuracy test")

        assert response.metadata["total_results"] == 3
        assert response.metadata["text_results"] == 2
        assert response.metadata["table_results"] == 1
        assert response.metadata["image_results"] == 0
        assert response.metadata["results_by_modality"] == {"text": 2, "table": 1, "image": 0}

    # ------------------------------------------------------------------
    # Scenario 27 — Deterministic response
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_27_same_inputs_produce_identical_response(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Identical query + config + retrieval results always yield the same AgentResponse."""
        result = _make_valid_result(chunk_id="chk-det", score=0.85)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4, return_vector=[0.1, 0.2, 0.3, 0.4])
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response_a = agent.search("Deterministic query")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        response_b = agent.search("Deterministic query")

        assert response_a.status == response_b.status
        assert response_a.total_citations == response_b.total_citations
        assert response_a.citations[0].chunk_id == response_b.citations[0].chunk_id
        assert response_a.citations[0].score == response_b.citations[0].score
        assert response_a.metadata["query"] == response_b.metadata["query"]

    # ------------------------------------------------------------------
    # Scenario 28 — No fake vectors: embed() is actually called
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_28_embed_method_is_called_on_real_provider(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Production code calls the real provider's embed(), no hardcoded/random vector."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MagicMock()
        provider.embed.return_value = [0.11, 0.22, 0.33, 0.44]

        agent = SearchAgent(embedding_provider=provider, store=mock_store)
        agent.search("Real embed call")

        provider.embed.assert_called_once_with("Real embed call")

    # ------------------------------------------------------------------
    # Scenario 29 — No fake retrieval: retrieve_context is called
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_29_retrieve_context_is_called_not_bypassed(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """SearchAgent calls retrieve_context exactly once per search() invocation."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search("Real retrieval call")

        assert mock_retrieve_context.call_count == 1

    # ------------------------------------------------------------------
    # Scenario 30 — No fake citations: citations come only from VectorSearchResult
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_30_citations_sourced_exclusively_from_retrieval_results(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Citations count exactly matches the number of results returned by Member 1."""
        n_results = 3
        results = [
            _make_valid_result(chunk_id=f"chk-{i}", score=0.9 - i * 0.1)
            for i in range(n_results)
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Citation source check")

        assert response.total_citations == n_results

    # ------------------------------------------------------------------
    # Scenario 31 — Existing Member 1 retrieval API reused (import verify)
    # ------------------------------------------------------------------

    def test_31_retrieve_context_imported_from_ingestion_retrieval_service(self) -> None:
        """search_agent imports retrieve_context from ingestion.retrieval_service."""
        import agents.search_agent as sa_module
        from ingestion.retrieval_service import retrieve_context as rc

        # The retrieve_context used inside the module must be the real Member 1 function
        assert sa_module.retrieve_context is rc

    # ------------------------------------------------------------------
    # Scenario 32 — Existing Member 1 embedding API reused (import verify)
    # ------------------------------------------------------------------

    def test_32_embedding_provider_protocol_imported_from_ingestion(self) -> None:
        """search_agent imports EmbeddingProvider from ingestion.embedding_generator."""
        import agents.search_agent as sa_module
        from ingestion.embedding_generator import EmbeddingProvider

        assert sa_module.EmbeddingProvider is EmbeddingProvider

    # ------------------------------------------------------------------
    # Scenario 33 — Multimodal: text result
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_33_multimodal_text_result_citation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Text-modality VectorSearchResult produces AgentCitation with content_type='text'."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(chunk_id="chk-mm-text", content_type="text", score=0.9)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Text modality check")

        assert response.citations[0].content_type == "text"
        assert response.metadata["text_results"] == 1

    # ------------------------------------------------------------------
    # Scenario 34 — Multimodal: table result
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_34_multimodal_table_result_citation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Table-modality VectorSearchResult produces AgentCitation with content_type='table'."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(chunk_id="chk-mm-table", content_type="table", score=0.85)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Table modality check")

        assert response.citations[0].content_type == "table"
        assert response.metadata["table_results"] == 1

    # ------------------------------------------------------------------
    # Scenario 35 — Multimodal: image result
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_35_multimodal_image_result_citation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Image-modality VectorSearchResult produces AgentCitation with content_type='image'."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(chunk_id="chk-mm-image", content_type="image", score=0.80)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Image modality check")

        assert response.citations[0].content_type == "image"
        assert response.metadata["image_results"] == 1

    # ------------------------------------------------------------------
    # Scenario 36 — Partial evidence failure: NaN score in second item
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_36_partial_evidence_failure_aborts_entire_response(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """A NaN score in the second result causes the entire search to fail (no partial citations)."""
        results = [
            _make_valid_result(chunk_id="chk-ok", score=0.9),
            _make_valid_result(chunk_id="chk-corrupt", score=float("nan")),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="score is not a finite numeric value"):
            agent.search("Partial failure scenario")
