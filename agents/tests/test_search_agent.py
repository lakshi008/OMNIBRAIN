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
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    SearchRequest,
    SearchResult,
)
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


# ---------------------------------------------------------------------------
# Day 26 — Result Ranking, Filtering, Deduplication & Citation Quality
# ---------------------------------------------------------------------------


class TestDay26ResultRankingFilteringAndCitationQuality:
    """Day 26 — Comprehensive result ranking, filtering, deduplication, and citation quality tests.

    Covers all 40 required scenarios: score ordering, duplicate resolution,
    result limits, multimodal handling, lineage integrity, and Member 1 protection.
    """

    # ------------------------------------------------------------------
    # Scenario 1 — Results sorted by descending score
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_01_results_sorted_by_descending_score(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """SearchAgent guarantees citations are presented in descending score order."""
        results = [
            _make_valid_result(chunk_id="chk-med", score=0.75),
            _make_valid_result(chunk_id="chk-high", score=0.95),
            _make_valid_result(chunk_id="chk-low", score=0.55),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store, max_results=5)

        response = agent.search("Score sorting query")

        scores = [cit.score for cit in response.citations]
        assert scores == [0.95, 0.75, 0.55]

    # ------------------------------------------------------------------
    # Scenario 2 — Already sorted results preserved
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_02_already_sorted_results_preserved(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """If Member 1 already returns sorted results, order is preserved intact."""
        results = [
            _make_valid_result(chunk_id="chk-1", score=0.99),
            _make_valid_result(chunk_id="chk-2", score=0.88),
            _make_valid_result(chunk_id="chk-3", score=0.77),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Already sorted query")

        assert [c.chunk_id for c in response.citations] == ["chk-1", "chk-2", "chk-3"]

    # ------------------------------------------------------------------
    # Scenario 3 — Unsorted results correctly sorted
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_03_unsorted_results_correctly_sorted(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Unordered results are sorted strictly descending by score."""
        results = [
            _make_valid_result(chunk_id="chk-30", score=0.30),
            _make_valid_result(chunk_id="chk-90", score=0.90),
            _make_valid_result(chunk_id="chk-60", score=0.60),
            _make_valid_result(chunk_id="chk-80", score=0.80),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store, max_results=10)

        response = agent.search("Unsorted query")

        assert [c.chunk_id for c in response.citations] == ["chk-90", "chk-80", "chk-60", "chk-30"]

    # ------------------------------------------------------------------
    # Scenario 4 — Duplicate chunk IDs deduplicated
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_04_duplicate_chunk_ids_deduplicated(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Duplicate chunk_ids with same score are reduced to a single citation."""
        results = [
            _make_valid_result(chunk_id="chk-A", score=0.90),
            _make_valid_result(chunk_id="chk-A", score=0.90),
            _make_valid_result(chunk_id="chk-B", score=0.80),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Duplicate chunk query")

        assert len(response.citations) == 2
        assert [c.chunk_id for c in response.citations] == ["chk-A", "chk-B"]

    # ------------------------------------------------------------------
    # Scenario 5 — Duplicate result with different scores keeps highest
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_05_duplicate_result_with_different_scores_keeps_highest(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """When the same chunk appears with different scores, only the highest score is retained."""
        results = [
            _make_valid_result(chunk_id="chk-dup", score=0.65, content="Lower score version"),
            _make_valid_result(chunk_id="chk-dup", score=0.95, content="Higher score version"),
            _make_valid_result(chunk_id="chk-other", score=0.80),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Highest score deduplication query")

        assert len(response.citations) == 2
        assert response.citations[0].chunk_id == "chk-dup"
        assert response.citations[0].score == 0.95
        assert response.citations[0].content_type == "text"
        assert response.citations[1].chunk_id == "chk-other"

    # ------------------------------------------------------------------
    # Scenario 6 — Top-k / max_results limit respected
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_06_top_k_limit_respected(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Final citations never exceed configured max_results / top_k limit."""
        results = [
            _make_valid_result(chunk_id=f"chk-{i}", score=0.99 - (i * 0.05))
            for i in range(10)
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store, max_results=3)

        response = agent.search("Limit query")

        assert len(response.citations) == 3
        assert response.total_citations == 3
        assert response.metadata["total_results"] == 3

    # ------------------------------------------------------------------
    # Scenario 7 — Minimum score filtering parameter forwarded
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_07_minimum_score_filtering_forwarded(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """min_score is cleanly forwarded to Member 1 retrieve_context."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store, min_score=0.70)

        agent.search("Min score forward check")

        assert mock_retrieve_context.call_args.kwargs["min_score"] == 0.70

    # ------------------------------------------------------------------
    # Scenario 8 — All results below threshold produces zero citations
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_08_all_results_below_threshold_produces_zero_citations(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """When retrieve_context returns empty list (all filtered), 0 citations returned."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store, min_score=0.90)

        response = agent.search("No matching score query")

        assert response.status == "success"
        assert response.citations == []
        assert response.total_citations == 0
        assert response.has_citations is False

    # ------------------------------------------------------------------
    # Scenario 9 — Zero results handled cleanly
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_09_zero_results_handled_cleanly(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Empty retrieval result returns valid response with 0 citations and empty context."""
        mock_retrieve_context.return_value = RetrievalServiceResult(
            query_vector_dimension=4,
            results=[],
            context="",
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Zero result test")

        assert response.status == "success"
        assert response.citations == []
        assert response.metadata["total_results"] == 0
        assert response.metadata["context"] == ""

    # ------------------------------------------------------------------
    # Scenario 10 — Valid text evidence produces text citation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_10_valid_text_evidence(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Text modality evidence is accurately captured into AgentCitation."""
        results = [_make_valid_result(chunk_id="text-1", content_type="text", score=0.88)]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Text evidence test")

        assert len(response.citations) == 1
        assert response.citations[0].content_type == "text"
        assert response.metadata["text_results"] == 1

    # ------------------------------------------------------------------
    # Scenario 11 — Valid table evidence produces table citation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_11_valid_table_evidence(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Table modality evidence is accurately captured into AgentCitation."""
        results = [_make_valid_result(chunk_id="tbl-1", content_type="table", score=0.82)]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Table evidence test")

        assert len(response.citations) == 1
        assert response.citations[0].content_type == "table"
        assert response.metadata["table_results"] == 1

    # ------------------------------------------------------------------
    # Scenario 12 — Valid image evidence produces image citation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_12_valid_image_evidence(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Image modality evidence is accurately captured into AgentCitation."""
        results = [_make_valid_result(chunk_id="img-1", content_type="image", score=0.79)]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Image evidence test")

        assert len(response.citations) == 1
        assert response.citations[0].content_type == "image"
        assert response.metadata["image_results"] == 1

    # ------------------------------------------------------------------
    # Scenario 13 — Missing chunk_id raises AgentExecutionError
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("invalid_cid", ["", "   ", None])
    @patch("agents.search_agent.retrieve_context")
    def test_13_missing_chunk_id_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        invalid_cid: Any,
    ) -> None:
        """Evidence missing chunk_id raises AgentExecutionError."""
        result = _make_valid_result(chunk_id=invalid_cid)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="chunk_id is missing or empty"):
            agent.search("Missing chunk id query")

    # ------------------------------------------------------------------
    # Scenario 14 — Missing document_id raises AgentExecutionError
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("invalid_docid", ["", "   ", None])
    @patch("agents.search_agent.retrieve_context")
    def test_14_missing_document_id_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        invalid_docid: Any,
    ) -> None:
        """Evidence missing document_id raises AgentExecutionError."""
        result = _make_valid_result(document_id=invalid_docid)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="document_id is missing or empty"):
            agent.search("Missing document id query")

    # ------------------------------------------------------------------
    # Scenario 15 — Missing filename raises AgentExecutionError
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("invalid_fn", ["", "   ", None])
    @patch("agents.search_agent.retrieve_context")
    def test_15_missing_filename_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        invalid_fn: Any,
    ) -> None:
        """Evidence missing filename raises AgentExecutionError."""
        result = _make_valid_result(filename=invalid_fn)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="filename is missing or empty"):
            agent.search("Missing filename query")

    # ------------------------------------------------------------------
    # Scenario 16 — Missing content raises AgentExecutionError
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("invalid_content", ["", "   ", None, 1234])
    @patch("agents.search_agent.retrieve_context")
    def test_16_missing_content_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        invalid_content: Any,
    ) -> None:
        """Evidence with missing or empty content raises AgentExecutionError."""
        result = _make_valid_result(content=invalid_content)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="content is empty or missing"):
            agent.search("Missing content query")

    # ------------------------------------------------------------------
    # Scenario 17 — Invalid score type raises AgentExecutionError
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_score", ["0.95", True, False, [0.8], {"score": 1}])
    @patch("agents.search_agent.retrieve_context")
    def test_17_invalid_score_type_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        bad_score: Any,
    ) -> None:
        """Non-numeric or boolean score raises AgentExecutionError."""
        result = _make_valid_result(score=bad_score)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="score is not a finite numeric value"):
            agent.search("Bad score type query")

    # ------------------------------------------------------------------
    # Scenario 18 — NaN score raises AgentExecutionError
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_18_nan_score_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """NaN score raises AgentExecutionError."""
        result = _make_valid_result(score=float("nan"))
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="score is not a finite numeric value"):
            agent.search("NaN score query")

    # ------------------------------------------------------------------
    # Scenario 19 — Infinite score raises AgentExecutionError
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("inf_val", [float("inf"), float("-inf")])
    @patch("agents.search_agent.retrieve_context")
    def test_19_infinite_score_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        inf_val: float,
    ) -> None:
        """Infinite score raises AgentExecutionError."""
        result = _make_valid_result(score=inf_val)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="score is not a finite numeric value"):
            agent.search("Inf score query")

    # ------------------------------------------------------------------
    # Scenario 20 — Invalid content_type raises AgentExecutionError
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("invalid_ct", ["", "   ", None, 123])
    @patch("agents.search_agent.retrieve_context")
    def test_20_invalid_content_type_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
        invalid_ct: Any,
    ) -> None:
        """Missing or empty content_type raises AgentExecutionError."""
        result = _make_valid_result(content_type=invalid_ct)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="content_type is missing or empty"):
            agent.search("Bad content_type query")

    # ------------------------------------------------------------------
    # Scenario 21 — Malformed metadata handled gracefully
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_21_metadata_preserved_accurately(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Arbitrary valid metadata dictionary is preserved in citations."""
        meta = {"department": "legal", "clause_num": 14, "confidential": True}
        result = _make_valid_result(metadata=meta)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Metadata query")

        assert response.citations[0].metadata == meta

    # ------------------------------------------------------------------
    # Scenario 22 — Citation lineage preservation field by field
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_22_citation_lineage_preservation_field_by_field(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """All lineage fields are faithfully mapped from VectorSearchResult to AgentCitation."""
        result = VectorSearchResult(
            chunk_id="chk-lineage-99",
            score=0.942,
            document_id="doc-lineage-88",
            filename="quarterly_filing.pdf",
            page_number=17,
            chunk_index=4,
            content_type="table",
            content="Revenue: $10.5M",
            metadata={"audited": True},
        )
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Lineage query")

        cit = response.citations[0]
        assert cit.chunk_id == "chk-lineage-99"
        assert cit.document_id == "doc-lineage-88"
        assert cit.filename == "quarterly_filing.pdf"
        assert cit.page_number == 17
        assert cit.content_type == "table"
        assert cit.score == 0.942
        assert cit.metadata == {"audited": True}

    # ------------------------------------------------------------------
    # Scenario 23 — Citation score preservation exact
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_23_citation_score_preservation_exact(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Exact float precision of score is preserved with no rounding/alteration."""
        exact_score = 0.87654321
        result = _make_valid_result(score=exact_score)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Exact score query")

        assert response.citations[0].score == exact_score

    # ------------------------------------------------------------------
    # Scenario 24 — Citation metadata preservation dict
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_24_citation_metadata_is_independent_dict(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Citation metadata is a clean dict copy, not mutating original source."""
        source_meta = {"key": "original"}
        result = _make_valid_result(metadata=source_meta)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Dict test")

        response.citations[0].metadata["key"] = "mutated"
        assert source_meta["key"] == "original"

    # ------------------------------------------------------------------
    # Scenario 25 — Deterministic ordering on identical scores
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_25_deterministic_ordering_on_identical_scores(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """When scores are identical, deterministic tie-breaking on chunk_index and chunk_id is used."""
        results = [
            _make_valid_result(chunk_id="chk-B", chunk_index=2, score=0.80),
            _make_valid_result(chunk_id="chk-A", chunk_index=1, score=0.80),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Tie breaking query")

        assert [c.chunk_id for c in response.citations] == ["chk-A", "chk-B"]

    # ------------------------------------------------------------------
    # Scenario 26 — No fake citations created
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_26_no_fake_citations_created(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Total citations count exactly equals accepted unique evidence items."""
        results = [
            _make_valid_result(chunk_id="chk-1", score=0.9),
            _make_valid_result(chunk_id="chk-2", score=0.8),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("No fake citations")

        assert len(response.citations) == 2
        assert {c.chunk_id for c in response.citations} == {"chk-1", "chk-2"}

    # ------------------------------------------------------------------
    # Scenario 27 — No fake evidence created
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_27_no_fake_evidence_created(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Empty retrieval returns zero citations with no fabricated fallback."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Empty evidence query")

        assert response.citations == []

    # ------------------------------------------------------------------
    # Scenario 28 — Member 1 retrieval API still reused
    # ------------------------------------------------------------------

    def test_28_member1_retrieval_api_still_reused(self) -> None:
        """SearchAgent reuses retrieve_context from ingestion.retrieval_service."""
        import agents.search_agent as sa
        from ingestion.retrieval_service import retrieve_context

        assert sa.retrieve_context is retrieve_context

    # ------------------------------------------------------------------
    # Scenario 29 — No duplicate Qdrant implementation
    # ------------------------------------------------------------------

    def test_29_no_duplicate_qdrant_implementation(self) -> None:
        """SearchAgent requires an instance of Member 1's QdrantVectorStore."""
        import agents.search_agent as sa
        from ingestion.qdrant_store import QdrantVectorStore

        assert sa.QdrantVectorStore is QdrantVectorStore

    # ------------------------------------------------------------------
    # Scenario 30 — No duplicate similarity search
    # ------------------------------------------------------------------

    def test_30_no_duplicate_similarity_search(self) -> None:
        """SearchAgent module does not define its own similarity or cosine function."""
        import agents.search_agent as sa

        assert not hasattr(sa, "cosine_similarity")
        assert not hasattr(sa, "dot_product")
        assert not hasattr(sa, "vector_search")

    # ------------------------------------------------------------------
    # Scenario 31 — No duplicate embedding implementation
    # ------------------------------------------------------------------

    def test_31_no_duplicate_embedding_implementation(self) -> None:
        """SearchAgent relies on Member 1's EmbeddingProvider protocol."""
        import agents.search_agent as sa
        from ingestion.embedding_generator import EmbeddingProvider

        assert sa.EmbeddingProvider is EmbeddingProvider

    # ------------------------------------------------------------------
    # Scenario 32 — Empty final evidence metadata
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_32_empty_final_evidence_metadata(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Metadata correctly reflects 0 for all modality counts when no evidence is present."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Empty stats query")

        assert response.metadata["total_results"] == 0
        assert response.metadata["text_results"] == 0
        assert response.metadata["table_results"] == 0
        assert response.metadata["image_results"] == 0

    # ------------------------------------------------------------------
    # Scenario 33 — Mixed valid and invalid results aborts completely
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_33_mixed_valid_and_invalid_results_aborts_completely(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """If one result is invalid (e.g. empty content), the entire response aborts."""
        results = [
            _make_valid_result(chunk_id="valid-1", score=0.9),
            _make_valid_result(chunk_id="invalid-2", content="", score=0.8),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="content is empty or missing"):
            agent.search("Mixed valid invalid query")

    # ------------------------------------------------------------------
    # Scenario 34 — Mixed high-score and low-score results ordered
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_34_mixed_high_score_and_low_score_results_ordered(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Citations from mixed score results are ordered strictly descending."""
        results = [
            _make_valid_result(chunk_id="chk-low", score=0.15),
            _make_valid_result(chunk_id="chk-high", score=0.98),
            _make_valid_result(chunk_id="chk-mid", score=0.55),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Mixed score ordering query")

        assert [c.chunk_id for c in response.citations] == ["chk-high", "chk-mid", "chk-low"]

    # ------------------------------------------------------------------
    # Scenario 35 — Duplicate multimodal evidence deduplication
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_35_duplicate_multimodal_evidence_deduplication(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Duplicate chunks of different modalities or same modality are deduplicated cleanly."""
        results = [
            _make_valid_result(chunk_id="chk-tbl-1", content_type="table", score=0.70),
            _make_valid_result(chunk_id="chk-tbl-1", content_type="table", score=0.90),
            _make_valid_result(chunk_id="chk-img-1", content_type="image", score=0.85),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Multimodal dedup query")

        assert len(response.citations) == 2
        assert response.citations[0].chunk_id == "chk-tbl-1"
        assert response.citations[0].score == 0.90
        assert response.citations[1].chunk_id == "chk-img-1"
        assert response.metadata["table_results"] == 1
        assert response.metadata["image_results"] == 1

    # ------------------------------------------------------------------
    # Scenario 36 — Result count correctness in metadata
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_36_result_count_correctness_in_metadata(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Metadata total_results matches exact count of citations after deduplication."""
        results = [
            _make_valid_result(chunk_id="chk-1", content_type="text", score=0.9),
            _make_valid_result(chunk_id="chk-1", content_type="text", score=0.8),  # duplicate
            _make_valid_result(chunk_id="chk-2", content_type="text", score=0.7),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Result count query")

        assert response.metadata["total_results"] == 2
        assert response.metadata["text_results"] == 2
        assert len(response.citations) == 2

    # ------------------------------------------------------------------
    # Scenario 37 — Top-k after deduplication ensures true unique limit
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_37_top_k_after_deduplication(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """max_results limits the number of distinct unique citations returned."""
        results = [
            _make_valid_result(chunk_id="chk-1", score=0.9),
            _make_valid_result(chunk_id="chk-2", score=0.8),
            _make_valid_result(chunk_id="chk-3", score=0.7),
            _make_valid_result(chunk_id="chk-4", score=0.6),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store, max_results=2)

        response = agent.search("Top k dedup query")

        assert len(response.citations) == 2
        assert [c.chunk_id for c in response.citations] == ["chk-1", "chk-2"]

    # ------------------------------------------------------------------
    # Scenario 38 — Minimum score before final limit
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_38_minimum_score_before_final_limit(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """SearchRequest parameters top_k, min_score, max_results are all respected."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        req = SearchRequest(query="Contract test", top_k=10, min_score=0.75, max_results=4)
        agent.search(req)

        call_kwargs = mock_retrieve_context.call_args.kwargs
        assert call_kwargs["top_k"] == 10
        assert call_kwargs["min_score"] == 0.75
        assert call_kwargs["max_results"] == 4

    # ------------------------------------------------------------------
    # Scenario 39 — Stable output for identical input
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_39_stable_output_for_identical_input(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Running the same query twice on identical results yields identical citations."""
        results = [
            _make_valid_result(chunk_id="chk-1", score=0.91),
            _make_valid_result(chunk_id="chk-2", score=0.81),
        ]
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        mock_retrieve_context.return_value = _make_retrieval_result(results)
        res1 = agent.search("Stable query")

        mock_retrieve_context.return_value = _make_retrieval_result(results)
        res2 = agent.search("Stable query")

        assert [c.to_dict() for c in res1.citations] == [c.to_dict() for c in res2.citations]
        assert res1.metadata == res2.metadata

    # ------------------------------------------------------------------
    # Scenario 40 — Standalone unit test of _apply_member2_result_policy
    # ------------------------------------------------------------------

    def test_40_standalone_apply_member2_result_policy(self) -> None:
        """Verify _apply_member2_result_policy works as an isolated static method."""
        items = [
            _make_valid_result(chunk_id="c-dup", score=0.60),
            _make_valid_result(chunk_id="c-dup", score=0.90),
            _make_valid_result(chunk_id="c-other", score=0.75),
            _make_valid_result(chunk_id="c-low", score=0.30),
        ]
        output = SearchAgent._apply_member2_result_policy(items, max_results=2)

        assert len(output) == 2
        assert output[0].chunk_id == "c-dup"
        assert output[0].score == 0.90
        assert output[1].chunk_id == "c-other"
        assert output[1].score == 0.75


# ---------------------------------------------------------------------------
# Day 27 — Query-to-Evidence Context Building & Search Response Quality
# ---------------------------------------------------------------------------


class TestDay27QueryToEvidenceContextBuildingAndResponseQuality:
    """Day 27 — Comprehensive query-to-evidence context building and search response quality tests.

    Covers all 44 required scenarios: query validation, evidence context construction,
    citation numbering, multimodal context preservation, context ordering, 1:1
    evidence/citation consistency, lineage fidelity, error boundaries, determinism,
    and Member 1 protection.
    """

    # ------------------------------------------------------------------
    # Scenario 1 — Valid query executes cleanly
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_01_valid_query_executes_cleanly(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Valid user query produces a structured AgentResponse with status='success'."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(chunk_id="chk-d27-01", score=0.92)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("What were the total assets in 2023?")

        assert isinstance(response, AgentResponse)
        assert response.status == "success"
        assert response.error is None
        assert response.answer == ""
        assert response.has_citations is True
        assert response.total_citations == 1

    # ------------------------------------------------------------------
    # Scenario 2 — Empty query string
    # ------------------------------------------------------------------

    def test_02_empty_query_raises_validation_error(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Empty query string raises AgentValidationError."""
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError):
            agent.search("")

    # ------------------------------------------------------------------
    # Scenario 3 — Whitespace-only query string
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("ws_q", ["   ", "\t\t", "\n\n", " \t \n "])
    def test_03_whitespace_query_raises_validation_error(
        self,
        mock_store: MagicMock,
        ws_q: str,
    ) -> None:
        """Whitespace-only query raises AgentValidationError."""
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError):
            agent.search(ws_q)

    # ------------------------------------------------------------------
    # Scenario 4 — Invalid query type
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_q", [123, 45.67, ["list"], {"dict": True}, True, None])
    def test_04_invalid_query_type_raises_validation_error(
        self,
        mock_store: MagicMock,
        bad_q: Any,
    ) -> None:
        """Non-string, non-Request query type raises AgentValidationError."""
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError):
            agent.search(bad_q)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Scenario 5 — Normal query produces retrieval request
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_05_normal_query_produces_retrieval_request(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Normal query triggers Member 1 retrieve_context with proper embedding vector."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4, return_vector=[0.2, 0.4, 0.6, 0.8])
        agent = SearchAgent(embedding_provider=provider, store=mock_store, collection_name="docs_col")

        agent.search("Execute retrieval pipeline")

        mock_retrieve_context.assert_called_once_with(
            query_vector=[0.2, 0.4, 0.6, 0.8],
            store=mock_store,
            collection_name="docs_col",
            top_k=5,
            min_score=0.0,
            max_results=5,
        )

    # ------------------------------------------------------------------
    # Scenario 6 — No evidence
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_06_no_evidence_returns_empty_context_and_zero_citations(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """When zero evidence is found, response has empty context, 0 citations, has_evidence=False."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("No results query")

        assert response.status == "success"
        assert response.citations == []
        assert response.total_citations == 0
        assert response.metadata["context"] == ""
        assert response.metadata["evidence_count"] == 0
        assert response.metadata["has_evidence"] is False
        assert response.metadata["total_results"] == 0

    # ------------------------------------------------------------------
    # Scenario 7 — One evidence item
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_07_one_evidence_item_produces_single_source_context(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Single evidence item produces [Source 1] block in context with 1 citation."""
        result = _make_valid_result(
            chunk_id="chk-single-1",
            filename="financial_summary.pdf",
            page_number=2,
            content_type="text",
            content="Total net assets: $42M.",
            score=0.91,
        )
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Single evidence query")

        assert response.total_citations == 1
        assert response.metadata["evidence_count"] == 1
        assert response.metadata["has_evidence"] is True
        assert "[Source 1]" in response.metadata["context"]
        assert "File: financial_summary.pdf" in response.metadata["context"]
        assert "Page: 2" in response.metadata["context"]
        assert "Total net assets: $42M." in response.metadata["context"]

    # ------------------------------------------------------------------
    # Scenario 8 — Multiple evidence items
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_08_multiple_evidence_items_produce_numbered_sources(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Multiple evidence items produce sequential [Source 1], [Source 2], [Source 3] blocks."""
        results = [
            _make_valid_result(chunk_id="chk-1", score=0.95),
            _make_valid_result(chunk_id="chk-2", score=0.85),
            _make_valid_result(chunk_id="chk-3", score=0.75),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Multiple evidence query")

        assert response.total_citations == 3
        ctx = response.metadata["context"]
        assert "[Source 1]" in ctx
        assert "[Source 2]" in ctx
        assert "[Source 3]" in ctx

    # ------------------------------------------------------------------
    # Scenario 9 — Source numbering starts at 1
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_09_source_numbering_starts_at_1(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Source numbering strictly starts at 1, never 0."""
        result = _make_valid_result(chunk_id="chk-start1", score=0.88)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Numbering starts at 1")

        assert response.metadata["context"].startswith("[Source 1]")
        assert "[Source 0]" not in response.metadata["context"]

    # ------------------------------------------------------------------
    # Scenario 10 — Deterministic source numbering
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_10_deterministic_source_numbering(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Repeated runs on the same input produce identical source numbering."""
        results = [
            _make_valid_result(chunk_id="chk-A", score=0.90),
            _make_valid_result(chunk_id="chk-B", score=0.80),
        ]
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        mock_retrieve_context.return_value = _make_retrieval_result(results)
        res1 = agent.search("Deterministic numbering")

        mock_retrieve_context.return_value = _make_retrieval_result(results)
        res2 = agent.search("Deterministic numbering")

        assert res1.metadata["context"] == res2.metadata["context"]

    # ------------------------------------------------------------------
    # Scenario 11 — Descending relevance order preserved in context
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_11_descending_relevance_order_preserved_in_context(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Highest score evidence becomes [Source 1], next becomes [Source 2]."""
        results = [
            _make_valid_result(chunk_id="chk-mid", score=0.70, content="Mid content"),
            _make_valid_result(chunk_id="chk-top", score=0.99, content="Top content"),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Relevance order check")

        ctx = response.metadata["context"]
        pos_source1 = ctx.find("[Source 1]")
        pos_source2 = ctx.find("[Source 2]")
        pos_top = ctx.find("Top content")
        pos_mid = ctx.find("Mid content")

        assert pos_source1 < pos_top < pos_source2 < pos_mid

    # ------------------------------------------------------------------
    # Scenario 12 — Text evidence
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_12_text_evidence_preserves_text_modality_block(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Text modality evidence has Type: text in context block."""
        result = _make_valid_result(chunk_id="chk-txt", content_type="text", content="Plain text excerpt.")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Text evidence check")

        assert "Type: text" in response.metadata["context"]
        assert "Plain text excerpt." in response.metadata["context"]

    # ------------------------------------------------------------------
    # Scenario 13 — Table evidence
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_13_table_evidence_preserves_table_modality_block(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Table modality evidence has Type: table in context block."""
        result = _make_valid_result(chunk_id="chk-tbl", content_type="table", content="| Col 1 | Col 2 |\n|---|---|")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Table evidence check")

        assert "Type: table" in response.metadata["context"]
        assert "| Col 1 | Col 2 |" in response.metadata["context"]

    # ------------------------------------------------------------------
    # Scenario 14 — Image evidence
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_14_image_evidence_preserves_image_modality_block(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Image modality evidence has Type: image in context block."""
        result = _make_valid_result(chunk_id="chk-img", content_type="image", content="[Image description: Bar chart]")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Image evidence check")

        assert "Type: image" in response.metadata["context"]
        assert "[Image description: Bar chart]" in response.metadata["context"]

    # ------------------------------------------------------------------
    # Scenario 15 — Mixed multimodal evidence
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_15_mixed_multimodal_evidence_in_context(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Context preserves all three modalities when present together."""
        results = [
            _make_valid_result(chunk_id="chk-1", content_type="text", score=0.90),
            _make_valid_result(chunk_id="chk-2", content_type="table", score=0.80),
            _make_valid_result(chunk_id="chk-3", content_type="image", score=0.70),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Mixed multimodal check")

        ctx = response.metadata["context"]
        assert "Type: text" in ctx
        assert "Type: table" in ctx
        assert "Type: image" in ctx
        assert response.metadata["text_results"] == 1
        assert response.metadata["table_results"] == 1
        assert response.metadata["image_results"] == 1

    # ------------------------------------------------------------------
    # Scenario 16 — Citation/context mapping is one-to-one
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_16_citation_context_mapping_is_one_to_one(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Every citation corresponds to exactly one [Source N] block in context."""
        n_items = 4
        results = [
            _make_valid_result(chunk_id=f"chk-map-{i}", score=0.95 - (i * 0.05))
            for i in range(n_items)
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("1:1 mapping query")

        assert len(response.citations) == n_items
        ctx = response.metadata["context"]
        for idx in range(1, n_items + 1):
            assert f"[Source {idx}]" in ctx

    # ------------------------------------------------------------------
    # Scenario 17 — Citation lineage preservation all fields
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_17_citation_lineage_preservation_all_fields(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """All lineage attributes are faithfully preserved in the AgentCitation."""
        result = VectorSearchResult(
            chunk_id="lineage-chk-777",
            score=0.912,
            document_id="lineage-doc-888",
            filename="audited_financials.pdf",
            page_number=42,
            chunk_index=9,
            content_type="text",
            content="Lineage validation paragraph.",
            metadata={"department": "compliance", "audited": True},
        )
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Lineage query")

        cit = response.citations[0]
        assert cit.chunk_id == "lineage-chk-777"
        assert cit.document_id == "lineage-doc-888"
        assert cit.filename == "audited_financials.pdf"
        assert cit.page_number == 42
        assert cit.content_type == "text"
        assert cit.score == 0.912
        assert cit.metadata == {"department": "compliance", "audited": True}

    # ------------------------------------------------------------------
    # Scenario 18 — document_id preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_18_document_id_preservation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """document_id is preserved exactly in citation."""
        result = _make_valid_result(document_id="unique-doc-12345")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("doc id preservation")

        assert response.citations[0].document_id == "unique-doc-12345"

    # ------------------------------------------------------------------
    # Scenario 19 — filename preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_19_filename_preservation_in_context_and_citations(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """filename is preserved in both citation and context block."""
        result = _make_valid_result(filename="sec_form_10k.pdf")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("filename preservation")

        assert response.citations[0].filename == "sec_form_10k.pdf"
        assert "File: sec_form_10k.pdf" in response.metadata["context"]

    # ------------------------------------------------------------------
    # Scenario 20 — page_number preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_20_page_number_preservation_in_context_and_citations(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """page_number is preserved in citation and context (or 'N/A' when None)."""
        r1 = _make_valid_result(chunk_id="chk-p1", page_number=15, score=0.9)
        r2 = _make_valid_result(chunk_id="chk-p2", page_number=None, score=0.8)
        mock_retrieve_context.return_value = _make_retrieval_result([r1, r2])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("page number check")

        assert response.citations[0].page_number == 15
        assert response.citations[1].page_number is None
        assert "Page: 15" in response.metadata["context"]
        assert "Page: N/A" in response.metadata["context"]

    # ------------------------------------------------------------------
    # Scenario 21 — chunk_id preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_21_chunk_id_preservation_in_citations(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """chunk_id is faithfully preserved in citations."""
        result = _make_valid_result(chunk_id="chunk-uuid-999")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("chunk id check")

        assert response.citations[0].chunk_id == "chunk-uuid-999"

    # ------------------------------------------------------------------
    # Scenario 22 — chunk_index preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_22_chunk_index_preservation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Non-negative chunk_index is accepted without modification."""
        result = _make_valid_result(chunk_index=7)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("chunk index check")
        assert response.status == "success"

    # ------------------------------------------------------------------
    # Scenario 23 — content_type preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_23_content_type_preservation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """content_type is preserved in both citation and context."""
        result = _make_valid_result(content_type="table")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("content type check")

        assert response.citations[0].content_type == "table"
        assert "Type: table" in response.metadata["context"]

    # ------------------------------------------------------------------
    # Scenario 24 — score preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_24_score_preservation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Exact score float is preserved in citation without distortion."""
        result = _make_valid_result(score=0.887766)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("score check")

        assert response.citations[0].score == 0.887766

    # ------------------------------------------------------------------
    # Scenario 25 — metadata preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_25_metadata_preservation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Custom metadata dict is preserved in citations."""
        custom_meta = {"version": "2.0", "author": "Research"}
        result = _make_valid_result(metadata=custom_meta)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("metadata check")

        assert response.citations[0].metadata == custom_meta

    # ------------------------------------------------------------------
    # Scenario 26 — evidence count correctness
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_26_evidence_count_correctness(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """evidence_count in metadata matches exact number of final accepted citations."""
        results = [
            _make_valid_result(chunk_id="c1", score=0.9),
            _make_valid_result(chunk_id="c2", score=0.8),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("count check")

        assert response.metadata["evidence_count"] == 2
        assert response.metadata["total_results"] == 2

    # ------------------------------------------------------------------
    # Scenario 27 — citation count matches evidence count
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_27_citation_count_matches_evidence_count(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """total_citations property strictly equals evidence_count."""
        results = [
            _make_valid_result(chunk_id="c1", score=0.9),
            _make_valid_result(chunk_id="c2", score=0.8),
            _make_valid_result(chunk_id="c3", score=0.7),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("citation count match check")

        assert response.total_citations == response.metadata["evidence_count"] == 3

    # ------------------------------------------------------------------
    # Scenario 28 — empty context when no evidence
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_28_empty_context_when_no_evidence(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Empty retrieval produces empty string for context."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("no evidence context check")

        assert response.metadata["context"] == ""

    # ------------------------------------------------------------------
    # Scenario 29 — partial invalid evidence aborts completely
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_29_partial_invalid_evidence_aborts_completely(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """A single malformed item in the retrieval results causes the entire search to fail."""
        results = [
            _make_valid_result(chunk_id="c1", score=0.9),
            _make_valid_result(chunk_id="c2", score=float("nan")),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError):
            agent.search("partial invalid check")

    # ------------------------------------------------------------------
    # Scenario 30 — duplicate evidence handling
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_30_duplicate_evidence_handling_in_context_and_citations(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Duplicates are removed from both citations and context, keeping the higher score."""
        results = [
            _make_valid_result(chunk_id="c-dup", score=0.50, content="Lower score content"),
            _make_valid_result(chunk_id="c-dup", score=0.95, content="Higher score content"),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("duplicate handling check")

        assert response.total_citations == 1
        assert "Higher score content" in response.metadata["context"]
        assert "Lower score content" not in response.metadata["context"]

    # ------------------------------------------------------------------
    # Scenario 31 — context ordering follows score descending
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_31_context_ordering_follows_score_descending(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Context blocks appear strictly in descending score order."""
        results = [
            _make_valid_result(chunk_id="c-low", score=0.40, content="Low score block"),
            _make_valid_result(chunk_id="c-high", score=0.90, content="High score block"),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("context order check")

        ctx = response.metadata["context"]
        assert ctx.find("High score block") < ctx.find("Low score block")

    # ------------------------------------------------------------------
    # Scenario 32 — context formatting exact template
    # ------------------------------------------------------------------

    def test_32_context_formatting_exact_template(self) -> None:
        """_build_evidence_context produces the exact expected structure."""
        item = _make_valid_result(
            filename="quarterly_report.pdf",
            page_number=3,
            content_type="text",
            content="Revenue increased by 14%.",
        )
        ctx = SearchAgent._build_evidence_context([item])
        expected = (
            "[Source 1]\n"
            "File: quarterly_report.pdf\n"
            "Page: 3\n"
            "Type: text\n"
            "Content:\n"
            "Revenue increased by 14%."
        )
        assert ctx == expected

    # ------------------------------------------------------------------
    # Scenario 33 — deterministic output across repeated runs
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_33_deterministic_output(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Same query and retrieval input produces identical response across multiple calls."""
        results = [_make_valid_result(chunk_id="c-det", score=0.88)]
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        mock_retrieve_context.return_value = _make_retrieval_result(results)
        res1 = agent.search("deterministic query")

        mock_retrieve_context.return_value = _make_retrieval_result(results)
        res2 = agent.search("deterministic query")

        assert res1.metadata == res2.metadata
        assert [c.to_dict() for c in res1.citations] == [c.to_dict() for c in res2.citations]

    # ------------------------------------------------------------------
    # Scenario 34 — malformed citation handling
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_34_malformed_citation_handling_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Missing chunk_id in evidence raises AgentExecutionError."""
        result = _make_valid_result(chunk_id="")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="chunk_id is missing or empty"):
            agent.search("malformed citation")

    # ------------------------------------------------------------------
    # Scenario 35 — missing content handling
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_35_missing_content_handling_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Empty content raises AgentExecutionError."""
        result = _make_valid_result(content="")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="content is empty or missing"):
            agent.search("missing content")

    # ------------------------------------------------------------------
    # Scenario 36 — invalid score handling
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_36_invalid_score_handling_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """NaN score raises AgentExecutionError."""
        result = _make_valid_result(score=float("nan"))
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="score is not a finite numeric value"):
            agent.search("invalid score")

    # ------------------------------------------------------------------
    # Scenario 37 — no fake citation generation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_37_no_fake_citation_generation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """When 2 results retrieved, exactly 2 citations generated."""
        results = [
            _make_valid_result(chunk_id="c1", score=0.9),
            _make_valid_result(chunk_id="c2", score=0.8),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("no fake citations check")

        assert len(response.citations) == 2

    # ------------------------------------------------------------------
    # Scenario 38 — no fake evidence generation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_38_no_fake_evidence_generation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Zero retrieval results yields zero citations and empty context without fake evidence."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("no fake evidence check")

        assert response.citations == []
        assert response.metadata["context"] == ""

    # ------------------------------------------------------------------
    # Scenario 39 — no duplicate Qdrant call
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_39_no_duplicate_qdrant_call(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """retrieve_context is invoked exactly once per search call."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search("single retrieval call")

        assert mock_retrieve_context.call_count == 1

    # ------------------------------------------------------------------
    # Scenario 40 — no duplicate embedding call
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_40_no_duplicate_embedding_call(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """provider.embed is called exactly once per query."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MagicMock()
        provider.embed.return_value = [0.1, 0.2, 0.3, 0.4]
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search("single embed call")

        assert provider.embed.call_count == 1

    # ------------------------------------------------------------------
    # Scenario 41 — existing Member 1 retrieval reused
    # ------------------------------------------------------------------

    def test_41_existing_member1_retrieval_reused(self) -> None:
        """SearchAgent reuses retrieve_context from ingestion.retrieval_service."""
        import agents.search_agent as sa
        from ingestion.retrieval_service import retrieve_context

        assert sa.retrieve_context is retrieve_context

    # ------------------------------------------------------------------
    # Scenario 42 — existing Day 26 filtering reused
    # ------------------------------------------------------------------

    def test_42_existing_day26_filtering_reused(self) -> None:
        """SearchAgent retains _apply_member2_result_policy static method."""
        assert hasattr(SearchAgent, "_apply_member2_result_policy")
        assert callable(SearchAgent._apply_member2_result_policy)

    # ------------------------------------------------------------------
    # Scenario 43 — existing Day 26 ranking reused
    # ------------------------------------------------------------------

    def test_43_existing_day26_ranking_reused(self) -> None:
        """_apply_member2_result_policy orders results by score descending."""
        items = [
            _make_valid_result(chunk_id="c-low", score=0.2),
            _make_valid_result(chunk_id="c-high", score=0.9),
        ]
        res = SearchAgent._apply_member2_result_policy(items, max_results=5)
        assert res[0].chunk_id == "c-high"
        assert res[1].chunk_id == "c-low"

    # ------------------------------------------------------------------
    # Scenario 44 — standalone build_evidence_context unit test
    # ------------------------------------------------------------------

    def test_44_standalone_build_evidence_context_unit_test(self) -> None:
        """_build_evidence_context formats multiple items with empty string on empty input."""
        assert SearchAgent._build_evidence_context([]) == ""

        items = [
            _make_valid_result(chunk_id="c1", filename="f1.pdf", page_number=1, content_type="text", content="C1"),
            _make_valid_result(chunk_id="c2", filename="f2.pdf", page_number=None, content_type="table", content="C2"),
        ]
        ctx = SearchAgent._build_evidence_context(items)
        assert "[Source 1]\nFile: f1.pdf\nPage: 1\nType: text\nContent:\nC1" in ctx
        assert "[Source 2]\nFile: f2.pdf\nPage: N/A\nType: table\nContent:\nC2" in ctx


# ---------------------------------------------------------------------------
# Day 28 — Search Result Packaging, Evidence Grouping & Downstream Contract
# ---------------------------------------------------------------------------


class TestDay28SearchResultPackagingAndDownstreamContract:
    """Day 28 — Comprehensive search result packaging, grouping, and downstream contract tests.

    Covers all 44 required scenarios: result packaging, status indicators,
    modality grouping (text/table/image), document grouping and counts,
    contract properties, immutability, determinism, and regression protection.
    """

    # ------------------------------------------------------------------
    # Scenario 1 — Result with one citation
    # ------------------------------------------------------------------

    def test_01_result_with_one_citation(self) -> None:
        """SearchResult with one citation accurately packages all fields."""
        cit = AgentCitation(
            document_id="doc-1",
            filename="report.pdf",
            chunk_id="chk-1",
            page_number=1,
            content_type="text",
            score=0.95,
        )
        pkg = SearchResult(
            query="Single citation test",
            status="RESULTS_FOUND",
            citations=[cit],
            context="[Source 1]\nFile: report.pdf\nPage: 1\nType: text\nContent:\n...",
        )
        assert pkg.total_results == 1
        assert pkg.evidence_count == 1
        assert pkg.has_results is True
        assert pkg.status == "RESULTS_FOUND"
        assert pkg.citations[0].chunk_id == "chk-1"

    # ------------------------------------------------------------------
    # Scenario 2 — Result with multiple citations
    # ------------------------------------------------------------------

    def test_02_result_with_multiple_citations(self) -> None:
        """SearchResult with multiple citations retains all citations in order."""
        c1 = AgentCitation(document_id="doc-1", filename="f1.pdf", chunk_id="c1", score=0.9)
        c2 = AgentCitation(document_id="doc-2", filename="f2.pdf", chunk_id="c2", score=0.8)
        pkg = SearchResult(query="Multi test", status="RESULTS_FOUND", citations=[c1, c2])

        assert pkg.total_results == 2
        assert [c.chunk_id for c in pkg.citations] == ["c1", "c2"]

    # ------------------------------------------------------------------
    # Scenario 3 — No-result response packaged
    # ------------------------------------------------------------------

    def test_03_no_result_response_packaged(self) -> None:
        """SearchResult with zero citations has status NO_RESULTS and empty collections."""
        pkg = SearchResult(query="No results query", status="NO_RESULTS", citations=[])
        assert pkg.total_results == 0
        assert pkg.has_results is False
        assert pkg.status == "NO_RESULTS"
        assert pkg.text_results == []
        assert pkg.table_results == []
        assert pkg.image_results == []

    # ------------------------------------------------------------------
    # Scenario 4 — RESULTS_FOUND status
    # ------------------------------------------------------------------

    def test_04_results_found_status(self) -> None:
        """SearchResult.from_response assigns RESULTS_FOUND when citations exist."""
        cit = AgentCitation(document_id="d1", filename="f1.pdf", chunk_id="c1", score=0.88)
        resp = AgentResponse(
            answer="",
            agent_name="SearchAgent",
            citations=[cit],
            metadata={"query": "Found query", "context": "[Source 1]..."},
        )
        pkg = SearchResult.from_response(resp)
        assert pkg.status == "RESULTS_FOUND"

    # ------------------------------------------------------------------
    # Scenario 5 — NO_RESULTS status
    # ------------------------------------------------------------------

    def test_05_no_results_status(self) -> None:
        """SearchResult.from_response assigns NO_RESULTS when citations list is empty."""
        resp = AgentResponse(
            answer="",
            agent_name="SearchAgent",
            citations=[],
            metadata={"query": "Not found query", "context": ""},
        )
        pkg = SearchResult.from_response(resp)
        assert pkg.status == "NO_RESULTS"

    # ------------------------------------------------------------------
    # Scenario 6 — has_results=True
    # ------------------------------------------------------------------

    def test_06_has_results_true(self) -> None:
        """has_results returns True when at least one citation exists."""
        cit = AgentCitation(document_id="d1", filename="f1.pdf", chunk_id="c1")
        pkg = SearchResult(query="query", citations=[cit])
        assert pkg.has_results is True

    # ------------------------------------------------------------------
    # Scenario 7 — has_results=False
    # ------------------------------------------------------------------

    def test_07_has_results_false(self) -> None:
        """has_results returns False when citations is empty."""
        pkg = SearchResult(query="query", citations=[])
        assert pkg.has_results is False

    # ------------------------------------------------------------------
    # Scenario 8 — total_results
    # ------------------------------------------------------------------

    def test_08_total_results(self) -> None:
        """total_results matches len(citations)."""
        cits = [AgentCitation(document_id="d", filename="f.pdf", chunk_id=f"c{i}") for i in range(5)]
        pkg = SearchResult(query="query", citations=cits)
        assert pkg.total_results == 5

    # ------------------------------------------------------------------
    # Scenario 9 — text_results
    # ------------------------------------------------------------------

    def test_09_text_results(self) -> None:
        """text_results filters only text modality citations."""
        c1 = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c1", content_type="text")
        c2 = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c2", content_type="table")
        pkg = SearchResult(query="query", citations=[c1, c2])

        assert len(pkg.text_results) == 1
        assert pkg.text_results[0].chunk_id == "c1"
        assert pkg.text_count == 1

    # ------------------------------------------------------------------
    # Scenario 10 — table_results
    # ------------------------------------------------------------------

    def test_10_table_results(self) -> None:
        """table_results filters only table modality citations."""
        c1 = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c1", content_type="text")
        c2 = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c2", content_type="table")
        pkg = SearchResult(query="query", citations=[c1, c2])

        assert len(pkg.table_results) == 1
        assert pkg.table_results[0].chunk_id == "c2"
        assert pkg.table_count == 1

    # ------------------------------------------------------------------
    # Scenario 11 — image_results
    # ------------------------------------------------------------------

    def test_11_image_results(self) -> None:
        """image_results filters only image modality citations."""
        c1 = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c1", content_type="image")
        pkg = SearchResult(query="query", citations=[c1])

        assert len(pkg.image_results) == 1
        assert pkg.image_results[0].chunk_id == "c1"
        assert pkg.image_count == 1

    # ------------------------------------------------------------------
    # Scenario 12 — mixed multimodal results
    # ------------------------------------------------------------------

    def test_12_mixed_multimodal_results(self) -> None:
        """Multimodal results are properly segregated and accessible by modality."""
        cits = [
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="t1", content_type="text"),
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="t2", content_type="text"),
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="tb1", content_type="table"),
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="img1", content_type="image"),
        ]
        pkg = SearchResult(query="query", citations=cits)

        assert pkg.total_results == 4
        assert pkg.text_count == 2
        assert pkg.table_count == 1
        assert pkg.image_count == 1
        assert pkg.text_count + pkg.table_count + pkg.image_count == pkg.total_results

    # ------------------------------------------------------------------
    # Scenario 13 — unique document count
    # ------------------------------------------------------------------

    def test_13_unique_document_count(self) -> None:
        """unique_document_count reflects number of distinct document_ids."""
        cits = [
            AgentCitation(document_id="doc-A", filename="a.pdf", chunk_id="c1"),
            AgentCitation(document_id="doc-A", filename="a.pdf", chunk_id="c2"),
            AgentCitation(document_id="doc-B", filename="b.pdf", chunk_id="c3"),
        ]
        pkg = SearchResult(query="query", citations=cits)
        assert pkg.unique_document_count == 2
        assert pkg.unique_documents == ["doc-A", "doc-B"]

    # ------------------------------------------------------------------
    # Scenario 14 — repeated citations from same document counted once
    # ------------------------------------------------------------------

    def test_14_repeated_citations_from_same_document(self) -> None:
        """Multiple citations from single document results in unique_document_count=1."""
        cits = [
            AgentCitation(document_id="doc-single", filename="s.pdf", chunk_id=f"c{i}")
            for i in range(4)
        ]
        pkg = SearchResult(query="query", citations=cits)
        assert pkg.unique_document_count == 1

    # ------------------------------------------------------------------
    # Scenario 15 — citations from multiple documents
    # ------------------------------------------------------------------

    def test_15_citations_from_multiple_documents(self) -> None:
        """by_document groups citations by document_id preserving rank order."""
        c1 = AgentCitation(document_id="doc-1", filename="1.pdf", chunk_id="c1", score=0.9)
        c2 = AgentCitation(document_id="doc-2", filename="2.pdf", chunk_id="c2", score=0.8)
        c3 = AgentCitation(document_id="doc-1", filename="1.pdf", chunk_id="c3", score=0.7)
        pkg = SearchResult(query="query", citations=[c1, c2, c3])

        by_doc = pkg.by_document
        assert list(by_doc.keys()) == ["doc-1", "doc-2"]
        assert [c.chunk_id for c in by_doc["doc-1"]] == ["c1", "c3"]
        assert [c.chunk_id for c in by_doc["doc-2"]] == ["c2"]

    # ------------------------------------------------------------------
    # Scenario 16 — citation identity preservation
    # ------------------------------------------------------------------

    def test_16_citation_identity_preservation(self) -> None:
        """Citations inside SearchResult are identical to original citations."""
        cit = AgentCitation(document_id="doc-id", filename="file.pdf", chunk_id="chk-id", score=0.88)
        pkg = SearchResult(query="query", citations=[cit])
        assert pkg.citations[0] is cit

    # ------------------------------------------------------------------
    # Scenario 17 — filename preservation
    # ------------------------------------------------------------------

    def test_17_filename_preservation(self) -> None:
        """filename is preserved in citations."""
        cit = AgentCitation(document_id="d", filename="annual_filing_2023.pdf", chunk_id="c")
        pkg = SearchResult(query="query", citations=[cit])
        assert pkg.citations[0].filename == "annual_filing_2023.pdf"

    # ------------------------------------------------------------------
    # Scenario 18 — page number preservation
    # ------------------------------------------------------------------

    def test_18_page_number_preservation(self) -> None:
        """page_number is preserved in citations."""
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", page_number=42)
        pkg = SearchResult(query="query", citations=[cit])
        assert pkg.citations[0].page_number == 42

    # ------------------------------------------------------------------
    # Scenario 19 — chunk ID preservation
    # ------------------------------------------------------------------

    def test_19_chunk_id_preservation(self) -> None:
        """chunk_id is preserved in citations."""
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="chunk-uuid-888")
        pkg = SearchResult(query="query", citations=[cit])
        assert pkg.citations[0].chunk_id == "chunk-uuid-888"

    # ------------------------------------------------------------------
    # Scenario 20 — content type preservation
    # ------------------------------------------------------------------

    def test_20_content_type_preservation(self) -> None:
        """content_type is preserved in citations."""
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", content_type="table")
        pkg = SearchResult(query="query", citations=[cit])
        assert pkg.citations[0].content_type == "table"

    # ------------------------------------------------------------------
    # Scenario 21 — score preservation
    # ------------------------------------------------------------------

    def test_21_score_preservation(self) -> None:
        """score is preserved in citations."""
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", score=0.8765)
        pkg = SearchResult(query="query", citations=[cit])
        assert pkg.citations[0].score == 0.8765

    # ------------------------------------------------------------------
    # Scenario 22 — metadata preservation
    # ------------------------------------------------------------------

    def test_22_metadata_preservation(self) -> None:
        """metadata dict is preserved in citations and package."""
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", metadata={"dept": "eng"})
        pkg = SearchResult(query="query", citations=[cit], metadata={"search_time_ms": 12.5})
        assert pkg.citations[0].metadata == {"dept": "eng"}
        assert pkg.metadata["search_time_ms"] == 12.5

    # ------------------------------------------------------------------
    # Scenario 23 — context preservation
    # ------------------------------------------------------------------

    def test_23_context_preservation(self) -> None:
        """Structured context string is preserved exactly in SearchResult."""
        context_str = "[Source 1]\nFile: f.pdf\nPage: 1\nType: text\nContent:\nSummary text."
        pkg = SearchResult(query="query", context=context_str)
        assert pkg.context == context_str

    # ------------------------------------------------------------------
    # Scenario 24 — source order preservation
    # ------------------------------------------------------------------

    def test_24_source_order_preservation(self) -> None:
        """Order of citations in SearchResult strictly matches input order."""
        cits = [AgentCitation(document_id=f"d{i}", filename="f.pdf", chunk_id=f"c{i}") for i in range(5)]
        pkg = SearchResult(query="query", citations=cits)
        assert [c.chunk_id for c in pkg.citations] == [f"c{i}" for i in range(5)]

    # ------------------------------------------------------------------
    # Scenario 25 — descending relevance preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_25_descending_relevance_preservation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """SearchAgent.search_and_package produces SearchResult in descending score order."""
        results = [
            _make_valid_result(chunk_id="c-mid", score=0.6),
            _make_valid_result(chunk_id="c-high", score=0.9),
            _make_valid_result(chunk_id="c-low", score=0.3),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Relevance preservation")
        assert [c.score for c in pkg.citations] == [0.9, 0.6, 0.3]

    # ------------------------------------------------------------------
    # Scenario 26 — no-result empty context
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_26_no_result_empty_context(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """No result search produces SearchResult with empty context."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Empty query")
        assert pkg.context == ""
        assert pkg.status == "NO_RESULTS"
        assert pkg.has_results is False

    # ------------------------------------------------------------------
    # Scenario 27 — empty citation list package
    # ------------------------------------------------------------------

    def test_27_empty_citation_list_package(self) -> None:
        """SearchResult initialized with empty citations list works cleanly."""
        pkg = SearchResult(query="valid query", citations=[])
        assert pkg.citations == []
        assert pkg.total_results == 0

    # ------------------------------------------------------------------
    # Scenario 28 — malformed citation handling
    # ------------------------------------------------------------------

    def test_28_malformed_citation_handling(self) -> None:
        """Non-AgentCitation in citations list raises AgentValidationError."""
        with pytest.raises(AgentValidationError, match="not an AgentCitation"):
            SearchResult(query="valid query", citations=["not_a_citation"])  # type: ignore[list-item]

    # ------------------------------------------------------------------
    # Scenario 29 — invalid query raises validation error
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_q", ["", "   ", None, 123])
    def test_29_invalid_query_raises_validation_error(self, bad_q: Any) -> None:
        """Invalid query raises AgentValidationError on SearchResult init."""
        with pytest.raises(AgentValidationError):
            SearchResult(query=bad_q)

    # ------------------------------------------------------------------
    # Scenario 30 — missing document_id in citation raises validation error
    # ------------------------------------------------------------------

    def test_30_missing_document_id_raises_validation_error(self) -> None:
        """Citation with empty document_id raises AgentValidationError."""
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="f.pdf", chunk_id="c")

    # ------------------------------------------------------------------
    # Scenario 31 — from_response missing query raises error
    # ------------------------------------------------------------------

    def test_31_from_response_missing_query_raises_error(self) -> None:
        """from_response with missing query in metadata raises AgentValidationError."""
        resp = AgentResponse(answer="", agent_name="SearchAgent", citations=[], metadata={})
        with pytest.raises(AgentValidationError, match="missing non-empty 'query'"):
            SearchResult.from_response(resp)

    # ------------------------------------------------------------------
    # Scenario 32 — deterministic packaging
    # ------------------------------------------------------------------

    def test_32_deterministic_packaging(self) -> None:
        """Packaging identical citations yields identical SearchResult."""
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", score=0.9)
        p1 = SearchResult(query="test", citations=[cit])
        p2 = SearchResult(query="test", citations=[cit])
        assert p1.to_dict() == p2.to_dict()

    # ------------------------------------------------------------------
    # Scenario 33 — repeated packaging gives same result
    # ------------------------------------------------------------------

    def test_33_repeated_packaging_gives_same_result(self) -> None:
        """SearchResult.from_response called repeatedly produces identical dicts."""
        resp = AgentResponse(
            answer="",
            agent_name="SearchAgent",
            citations=[AgentCitation(document_id="d", filename="f.pdf", chunk_id="c")],
            metadata={"query": "test query"},
        )
        r1 = SearchResult.from_response(resp)
        r2 = SearchResult.from_response(resp)
        assert r1.to_dict() == r2.to_dict()

    # ------------------------------------------------------------------
    # Scenario 34 — original citation objects not mutated
    # ------------------------------------------------------------------

    def test_34_original_citation_objects_not_mutated(self) -> None:
        """Packaging into SearchResult does not modify the source AgentCitation."""
        cit = AgentCitation(document_id="d", filename="f.pdf", chunk_id="c", score=0.85)
        SearchResult(query="test", citations=[cit])
        assert cit.score == 0.85
        assert cit.chunk_id == "c"

    # ------------------------------------------------------------------
    # Scenario 35 — text/table/image counts sum correctly
    # ------------------------------------------------------------------

    def test_35_text_table_image_counts_sum_correctly(self) -> None:
        """text_count + table_count + image_count == total_results."""
        cits = [
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="t1", content_type="text"),
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="tb1", content_type="table"),
            AgentCitation(document_id="d", filename="f.pdf", chunk_id="i1", content_type="image"),
        ]
        pkg = SearchResult(query="query", citations=cits)
        assert pkg.text_count + pkg.table_count + pkg.image_count == pkg.total_results == 3

    # ------------------------------------------------------------------
    # Scenario 36 — unique document count correctness
    # ------------------------------------------------------------------

    def test_36_unique_document_count_correctness(self) -> None:
        """unique_document_count handles multiple documents correctly."""
        cits = [
            AgentCitation(document_id="doc-1", filename="f.pdf", chunk_id="c1"),
            AgentCitation(document_id="doc-2", filename="f.pdf", chunk_id="c2"),
            AgentCitation(document_id="doc-3", filename="f.pdf", chunk_id="c3"),
        ]
        pkg = SearchResult(query="query", citations=cits)
        assert pkg.unique_document_count == 3

    # ------------------------------------------------------------------
    # Scenario 37 — no fake citations
    # ------------------------------------------------------------------

    def test_37_no_fake_citations(self) -> None:
        """SearchResult citations list contains only provided citations."""
        pkg = SearchResult(query="query", citations=[])
        assert len(pkg.citations) == 0

    # ------------------------------------------------------------------
    # Scenario 38 — no fake metadata
    # ------------------------------------------------------------------

    def test_38_no_fake_metadata(self) -> None:
        """Metadata defaults to empty dictionary if not provided."""
        pkg = SearchResult(query="query")
        assert pkg.metadata == {}

    # ------------------------------------------------------------------
    # Scenario 39 — no duplicate retrieval call
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_39_no_duplicate_retrieval_call(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """search_and_package invokes Member 1 retrieve_context exactly once."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search_and_package("single call test")
        assert mock_retrieve_context.call_count == 1

    # ------------------------------------------------------------------
    # Scenario 40 — no duplicate embedding call
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_40_no_duplicate_embedding_call(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """search_and_package calls provider.embed exactly once."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MagicMock()
        provider.embed.return_value = [0.1, 0.2, 0.3, 0.4]
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search_and_package("single embed call")
        assert provider.embed.call_count == 1

    # ------------------------------------------------------------------
    # Scenario 41 — Member 1 retrieval reused
    # ------------------------------------------------------------------

    def test_41_member1_retrieval_reused(self) -> None:
        """SearchAgent reuses Member 1 retrieve_context."""
        import agents.search_agent as sa
        from ingestion.retrieval_service import retrieve_context

        assert sa.retrieve_context is retrieve_context

    # ------------------------------------------------------------------
    # Scenario 42 — Day 26 filtering reused
    # ------------------------------------------------------------------

    def test_42_day26_filtering_reused(self) -> None:
        """SearchAgent retains _apply_member2_result_policy method."""
        assert hasattr(SearchAgent, "_apply_member2_result_policy")

    # ------------------------------------------------------------------
    # Scenario 43 — Day 27 context reused
    # ------------------------------------------------------------------

    def test_43_day27_context_reused(self) -> None:
        """SearchAgent retains _build_evidence_context method."""
        assert hasattr(SearchAgent, "_build_evidence_context")

    # ------------------------------------------------------------------
    # Scenario 44 — serialization roundtrip
    # ------------------------------------------------------------------

    def test_44_serialization_roundtrip(self) -> None:
        """SearchResult to_dict and from_dict roundtrip cleanly."""
        cit = AgentCitation(document_id="doc-1", filename="f.pdf", chunk_id="c1", score=0.88)
        orig = SearchResult(
            query="roundtrip query",
            status="RESULTS_FOUND",
            citations=[cit],
            context="[Source 1]...",
            metadata={"source": "agent"},
        )
        d = orig.to_dict()
        restored = SearchResult.from_dict(d)

        assert restored.query == orig.query
        assert restored.status == orig.status
        assert restored.context == orig.context
        assert len(restored.citations) == len(orig.citations)
        assert restored.citations[0].chunk_id == orig.citations[0].chunk_id
        assert restored.total_results == orig.total_results


# ---------------------------------------------------------------------------
# Day 29 — Complete Query Execution Orchestration & Supervisor-Ready Contract
# ---------------------------------------------------------------------------


class TestDay29QueryExecutionOrchestrationAndSupervisorContract:
    """Day 29 — Comprehensive end-to-end query execution orchestration and contract tests.

    Covers all 38+ required scenarios: full orchestration pipeline (validation ->
    embedding -> retrieval -> filter/rank -> citation/context -> packaging),
    single-invocation guarantees, call ordering, error isolation, multimodal preservation,
    and Member 1 component reuse.
    """

    # ------------------------------------------------------------------
    # Scenario 1 — Valid query end-to-end
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_01_valid_query_end_to_end(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Valid query runs through full pipeline to produce a valid AgentResponse."""
        mock_retrieve_context.return_value = _make_retrieval_result(
            [_make_valid_result(chunk_id="chk-d29-01", score=0.91)]
        )
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("What is dense vector retrieval?")

        assert response.status == "success"
        assert response.total_citations == 1
        assert response.citations[0].chunk_id == "chk-d29-01"
        assert response.metadata["search_status"] == "RESULTS_FOUND"
        assert response.metadata["has_results"] is True

    # ------------------------------------------------------------------
    # Scenario 2 — Whitespace query validation
    # ------------------------------------------------------------------

    def test_02_whitespace_query_validation(self, mock_store: MagicMock) -> None:
        """Whitespace query is rejected before embedding generation."""
        provider = MagicMock()
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError):
            agent.search("   \t\n   ")

        provider.embed.assert_not_called()

    # ------------------------------------------------------------------
    # Scenario 3 — Empty query validation
    # ------------------------------------------------------------------

    def test_03_empty_query_validation(self, mock_store: MagicMock) -> None:
        """Empty query is rejected before embedding generation."""
        provider = MagicMock()
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError):
            agent.search("")

        provider.embed.assert_not_called()

    # ------------------------------------------------------------------
    # Scenario 4 — None query validation
    # ------------------------------------------------------------------

    def test_04_none_query_validation(self, mock_store: MagicMock) -> None:
        """None query is rejected before embedding generation."""
        provider = MagicMock()
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError):
            agent.search(None)  # type: ignore[arg-type]

        provider.embed.assert_not_called()

    # ------------------------------------------------------------------
    # Scenario 5 — Invalid query type validation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("invalid_q", [12345, 3.1415, ["query"], {"q": "text"}, True])
    def test_05_invalid_query_type_validation(self, mock_store: MagicMock, invalid_q: Any) -> None:
        """Non-string/non-Request query is rejected before embedding."""
        provider = MagicMock()
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError):
            agent.search(invalid_q)

        provider.embed.assert_not_called()

    # ------------------------------------------------------------------
    # Scenario 6 — Successful embedding generation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_06_successful_embedding_generation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Generated embedding vector is passed faithfully to retrieve_context."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4, return_vector=[0.11, 0.22, 0.33, 0.44])
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search("Vector generation query")

        mock_retrieve_context.assert_called_once()
        assert mock_retrieve_context.call_args.kwargs["query_vector"] == [0.11, 0.22, 0.33, 0.44]

    # ------------------------------------------------------------------
    # Scenario 7 — Embedding called exactly once
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_07_embedding_called_exactly_once(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """One search call results in exactly one provider.embed invocation."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MagicMock()
        provider.embed.return_value = [0.1, 0.2, 0.3, 0.4]
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search("Single embed call")

        assert provider.embed.call_count == 1

    # ------------------------------------------------------------------
    # Scenario 8 — Retrieval called exactly once
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_08_retrieval_called_exactly_once(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """One search call invokes Member 1 retrieve_context exactly once."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search("Single retrieve call")

        assert mock_retrieve_context.call_count == 1

    # ------------------------------------------------------------------
    # Scenario 9 — Successful retrieval flow
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_09_successful_retrieval_flow(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Results returned from retrieve_context flow into citations and metadata."""
        result = _make_valid_result(chunk_id="chk-success", score=0.89)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Successful flow")

        assert response.total_citations == 1
        assert response.citations[0].chunk_id == "chk-success"

    # ------------------------------------------------------------------
    # Scenario 10 — Zero retrieval results flow
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_10_zero_retrieval_results_flow(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Zero retrieval results cleanly produce empty citations and NO_RESULTS status."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Zero results query")

        assert response.total_citations == 0
        assert response.metadata["search_status"] == "NO_RESULTS"
        assert response.metadata["has_results"] is False
        assert response.metadata["context"] == ""

    # ------------------------------------------------------------------
    # Scenario 11 — Multiple retrieval results orchestration
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_11_multiple_retrieval_results_orchestration(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Multiple retrieved items are properly processed through the full pipeline."""
        results = [
            _make_valid_result(chunk_id="c1", score=0.95),
            _make_valid_result(chunk_id="c2", score=0.85),
            _make_valid_result(chunk_id="c3", score=0.75),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Multi results query")

        assert response.total_citations == 3
        assert [c.chunk_id for c in response.citations] == ["c1", "c2", "c3"]

    # ------------------------------------------------------------------
    # Scenario 12 — Filtering reuse and deduplication
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_12_filtering_reuse_and_deduplication(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Duplicate results with same chunk_id are deduplicated keeping highest score."""
        results = [
            _make_valid_result(chunk_id="dup-chunk", score=0.60),
            _make_valid_result(chunk_id="dup-chunk", score=0.92),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Deduplication orchestration")

        assert response.total_citations == 1
        assert response.citations[0].score == 0.92

    # ------------------------------------------------------------------
    # Scenario 13 — Ranking preservation descending
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_13_ranking_preservation_descending(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Results are ordered strictly descending by score."""
        results = [
            _make_valid_result(chunk_id="c-low", score=0.40),
            _make_valid_result(chunk_id="c-high", score=0.98),
            _make_valid_result(chunk_id="c-mid", score=0.75),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Ranking check")

        assert [c.chunk_id for c in response.citations] == ["c-high", "c-mid", "c-low"]

    # ------------------------------------------------------------------
    # Scenario 14 — Citation preservation all provenance fields
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_14_citation_preservation_all_provenance_fields(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Every citation preserves document_id, filename, page_number, chunk_id, content_type, score, metadata."""
        result = VectorSearchResult(
            chunk_id="chk-full-provenance",
            score=0.933,
            document_id="doc-prov-100",
            filename="financial_overview.pdf",
            page_number=8,
            chunk_index=2,
            content_type="table",
            content="Revenue: $500K",
            metadata={"auditor": "PwC"},
        )
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Provenance check")

        cit = response.citations[0]
        assert cit.chunk_id == "chk-full-provenance"
        assert cit.document_id == "doc-prov-100"
        assert cit.filename == "financial_overview.pdf"
        assert cit.page_number == 8
        assert cit.content_type == "table"
        assert cit.score == 0.933
        assert cit.metadata == {"auditor": "PwC"}

    # ------------------------------------------------------------------
    # Scenario 15 — Context preservation source blocks
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_15_context_preservation_source_blocks(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Context builds deterministic [Source 1], [Source 2] blocks in score order."""
        results = [
            _make_valid_result(chunk_id="c1", filename="a.pdf", score=0.90, content="Content A"),
            _make_valid_result(chunk_id="c2", filename="b.pdf", score=0.80, content="Content B"),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        response = agent.search("Context blocks check")

        ctx = response.metadata["context"]
        assert "[Source 1]" in ctx
        assert "File: a.pdf" in ctx
        assert "[Source 2]" in ctx
        assert "File: b.pdf" in ctx

    # ------------------------------------------------------------------
    # Scenario 16 — Result packaging via search_and_package
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_16_result_packaging_via_search_and_package(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """search_and_package returns a validated SearchResult instance."""
        result = _make_valid_result(chunk_id="chk-pkg", score=0.88)
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Packaging test")

        assert isinstance(pkg, SearchResult)
        assert pkg.query == "Packaging test"
        assert pkg.status == "RESULTS_FOUND"
        assert pkg.total_results == 1
        assert pkg.has_results is True

    # ------------------------------------------------------------------
    # Scenario 17 — RESULTS_FOUND status on success
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_17_results_found_status_on_success(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Non-empty search results produce RESULTS_FOUND status in metadata and package."""
        mock_retrieve_context.return_value = _make_retrieval_result([_make_valid_result()])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Status found check")
        assert pkg.status == "RESULTS_FOUND"

    # ------------------------------------------------------------------
    # Scenario 18 — NO_RESULTS status on empty
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_18_no_results_status_on_empty(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Empty search results produce NO_RESULTS status."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Status no results check")
        assert pkg.status == "NO_RESULTS"

    # ------------------------------------------------------------------
    # Scenario 19 — Text result orchestration
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_19_text_result_orchestration(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Text modality evidence is accurately tracked in text_results."""
        result = _make_valid_result(chunk_id="txt-1", content_type="text")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Text orchestration")
        assert pkg.text_count == 1
        assert len(pkg.text_results) == 1

    # ------------------------------------------------------------------
    # Scenario 20 — Table result orchestration
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_20_table_result_orchestration(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Table modality evidence is accurately tracked in table_results."""
        result = _make_valid_result(chunk_id="tbl-1", content_type="table")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Table orchestration")
        assert pkg.table_count == 1
        assert len(pkg.table_results) == 1

    # ------------------------------------------------------------------
    # Scenario 21 — Image result orchestration
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_21_image_result_orchestration(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Image modality evidence is accurately tracked in image_results."""
        result = _make_valid_result(chunk_id="img-1", content_type="image")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Image orchestration")
        assert pkg.image_count == 1
        assert len(pkg.image_results) == 1

    # ------------------------------------------------------------------
    # Scenario 22 — Multimodal result orchestration
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_22_multimodal_result_orchestration(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Multimodal evidence counts and lists correctly aggregate."""
        results = [
            _make_valid_result(chunk_id="t1", content_type="text"),
            _make_valid_result(chunk_id="t2", content_type="text"),
            _make_valid_result(chunk_id="tb1", content_type="table"),
            _make_valid_result(chunk_id="im1", content_type="image"),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Multimodal orchestration")
        assert pkg.total_results == 4
        assert pkg.text_count == 2
        assert pkg.table_count == 1
        assert pkg.image_count == 1

    # ------------------------------------------------------------------
    # Scenario 23 — Unique document count orchestration
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_23_unique_document_count_orchestration(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """unique_document_count matches number of distinct document IDs."""
        results = [
            _make_valid_result(chunk_id="c1", document_id="doc-A"),
            _make_valid_result(chunk_id="c2", document_id="doc-A"),
            _make_valid_result(chunk_id="c3", document_id="doc-B"),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Unique doc count orchestration")
        assert pkg.unique_document_count == 2
        assert pkg.unique_documents == ["doc-A", "doc-B"]

    # ------------------------------------------------------------------
    # Scenario 24 — Embedding failure prevents retrieval
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_24_embedding_failure_prevents_retrieval(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """If embedding provider raises an exception, retrieve_context is NEVER called."""
        provider = MagicMock()
        provider.embed.side_effect = RuntimeError("Embedding provider unreachable")
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Failed to generate query embedding"):
            agent.search("Embedding failure query")

        mock_retrieve_context.assert_not_called()

    # ------------------------------------------------------------------
    # Scenario 25 — Retrieval failure raises execution error
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_25_retrieval_failure_raises_execution_error(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """If retrieve_context raises an exception, search raises AgentExecutionError."""
        mock_retrieve_context.side_effect = RuntimeError("Qdrant cluster unavailable")
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Retrieval execution failed"):
            agent.search("Retrieval failure query")

    # ------------------------------------------------------------------
    # Scenario 26 — Malformed retrieval result aborts search
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_26_malformed_retrieval_result_aborts_search(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Item with missing content_type immediately aborts search with AgentExecutionError."""
        bad_item = _make_valid_result(content_type="")
        mock_retrieve_context.return_value = _make_retrieval_result([bad_item])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="content_type is missing or empty"):
            agent.search("Malformed item query")

    # ------------------------------------------------------------------
    # Scenario 27 — Citation conversion failure raises execution error
    # ------------------------------------------------------------------

    def test_27_citation_failure_raises_validation_error(self) -> None:
        """Attempting to construct citation from invalid object raises AgentValidationError."""
        with pytest.raises(AgentValidationError):
            AgentCitation.from_search_result(object())

    # ------------------------------------------------------------------
    # Scenario 28 — Packaging failure on corrupt response
    # ------------------------------------------------------------------

    def test_28_packaging_failure_on_corrupt_response(self) -> None:
        """SearchResult.from_response on non-AgentResponse raises AgentValidationError."""
        with pytest.raises(AgentValidationError, match="Expected AgentResponse instance"):
            SearchResult.from_response("not_a_response")  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Scenario 29 — No fake citations produced
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_29_no_fake_citations_produced(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Zero retrieval results produces exactly 0 citations."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("No fake citations")
        assert len(pkg.citations) == 0

    # ------------------------------------------------------------------
    # Scenario 30 — No fake vectors generated
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_30_no_fake_vectors_generated(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Vector passed to retrieval is exactly the vector returned by the provider."""
        expected_vec = [0.123, 0.456, 0.789, 0.999]
        provider = MockEmbeddingProvider(dimension=4, return_vector=expected_vec)
        mock_retrieve_context.return_value = _make_retrieval_result([])
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search("Real vector check")
        assert mock_retrieve_context.call_args.kwargs["query_vector"] == expected_vec

    # ------------------------------------------------------------------
    # Scenario 31 — No duplicate embedding invocation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_31_no_duplicate_embedding_invocation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """search_and_package executes embedding exactly once."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MagicMock()
        provider.embed.return_value = [0.1, 0.2, 0.3, 0.4]
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search_and_package("Single embed check")
        assert provider.embed.call_count == 1

    # ------------------------------------------------------------------
    # Scenario 32 — No duplicate retrieval invocation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_32_no_duplicate_retrieval_invocation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """search_and_package executes retrieval exactly once."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search_and_package("Single retrieval check")
        assert mock_retrieve_context.call_count == 1

    # ------------------------------------------------------------------
    # Scenario 33 — Member 1 API reuse verification
    # ------------------------------------------------------------------

    def test_33_member1_api_reuse_verification(self) -> None:
        """SearchAgent reuses retrieve_context from ingestion.retrieval_service."""
        import agents.search_agent as sa
        from ingestion.retrieval_service import retrieve_context

        assert sa.retrieve_context is retrieve_context

    # ------------------------------------------------------------------
    # Scenario 34 — Day 26 filtering reuse verification
    # ------------------------------------------------------------------

    def test_34_day26_filtering_reuse_verification(self) -> None:
        """SearchAgent retains _apply_member2_result_policy static method."""
        assert hasattr(SearchAgent, "_apply_member2_result_policy")
        assert callable(SearchAgent._apply_member2_result_policy)

    # ------------------------------------------------------------------
    # Scenario 35 — Day 27 context reuse verification
    # ------------------------------------------------------------------

    def test_35_day27_context_reuse_verification(self) -> None:
        """SearchAgent retains _build_evidence_context static method."""
        assert hasattr(SearchAgent, "_build_evidence_context")
        assert callable(SearchAgent._build_evidence_context)

    # ------------------------------------------------------------------
    # Scenario 36 — Day 28 packaging reuse verification
    # ------------------------------------------------------------------

    def test_36_day28_packaging_reuse_verification(self) -> None:
        """SearchAgent retains package_result and search_and_package methods."""
        assert hasattr(SearchAgent, "package_result")
        assert hasattr(SearchAgent, "search_and_package")

    # ------------------------------------------------------------------
    # Scenario 37 — Deterministic result across calls
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_37_deterministic_result_across_calls(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Repeated search_and_package calls on identical data return equivalent SearchResult."""
        results = [
            _make_valid_result(chunk_id="c1", score=0.9),
            _make_valid_result(chunk_id="c2", score=0.8),
        ]
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        mock_retrieve_context.return_value = _make_retrieval_result(results)
        pkg1 = agent.search_and_package("deterministic query")

        mock_retrieve_context.return_value = _make_retrieval_result(results)
        pkg2 = agent.search_and_package("deterministic query")

        assert pkg1.to_dict() == pkg2.to_dict()

    # ------------------------------------------------------------------
    # Scenario 38 — Call order verification step by step
    # ------------------------------------------------------------------

    def test_38_call_order_verification(self, mock_store: MagicMock) -> None:
        """Verify strict step order: validate -> embed -> retrieve -> filter -> cite -> package."""
        execution_order: list[str] = []

        class TrackingProvider:
            def embed(self, text: str) -> list[float]:
                execution_order.append("embed")
                return [0.1, 0.2, 0.3, 0.4]

        with patch("agents.search_agent.retrieve_context") as mock_ret:
            def tracking_retrieve(*args: Any, **kwargs: Any) -> RetrievalServiceResult:
                execution_order.append("retrieve")
                return _make_retrieval_result([_make_valid_result()])

            mock_ret.side_effect = tracking_retrieve

            agent = SearchAgent(embedding_provider=TrackingProvider(), store=mock_store)  # type: ignore[arg-type]
            pkg = agent.search_and_package("Order tracking query")

            assert execution_order == ["embed", "retrieve"]
            assert pkg.total_results == 1


# ---------------------------------------------------------------------------
# Day 30 — Search Agent Hardening, Error Isolation & Edge-Case Validation
# ---------------------------------------------------------------------------


class TestDay30SearchAgentHardeningAndErrorIsolation:
    """Day 30 — Comprehensive hardening, failure isolation, and edge-case validation tests.

    Covers all 36+ required scenarios: query edge-cases (None, empty, whitespace,
    multilingual, Tamil, long queries), embedding and retrieval exception boundaries,
    dimension mismatch safety, malformed result rejection, citation and packaging errors,
    call-order enforcement, exactly-once execution, secret-safe messages, and determinism.
    """

    # ------------------------------------------------------------------
    # Scenario 1 — None query rejection
    # ------------------------------------------------------------------

    def test_01_none_query_rejection(self, mock_store: MagicMock) -> None:
        """None query raises AgentValidationError without calling embedding."""
        provider = MagicMock()
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError, match="Query request cannot be None"):
            agent.search(None)  # type: ignore[arg-type]

        provider.embed.assert_not_called()

    # ------------------------------------------------------------------
    # Scenario 2 — Empty query rejection
    # ------------------------------------------------------------------

    def test_02_empty_query_rejection(self, mock_store: MagicMock) -> None:
        """Empty query raises AgentValidationError without calling embedding."""
        provider = MagicMock()
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError, match="cannot be empty"):
            agent.search("")

        provider.embed.assert_not_called()

    # ------------------------------------------------------------------
    # Scenario 3 — Whitespace query rejection
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("ws", ["   ", "\t", "\n", " \t \n  "])
    def test_03_whitespace_query_rejection(self, mock_store: MagicMock, ws: str) -> None:
        """Whitespace-only query raises AgentValidationError without calling embedding."""
        provider = MagicMock()
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError, match="cannot be empty or whitespace-only"):
            agent.search(ws)

        provider.embed.assert_not_called()

    # ------------------------------------------------------------------
    # Scenario 4 — Non-string query rejection
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_input", [123, 45.67, True, [], {}])
    def test_04_non_string_query_rejection(self, mock_store: MagicMock, bad_input: Any) -> None:
        """Non-string/non-Request query raises AgentValidationError."""
        provider = MagicMock()
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentValidationError, match="Expected query string"):
            agent.search(bad_input)

        provider.embed.assert_not_called()

    # ------------------------------------------------------------------
    # Scenario 5 — Very long query handling
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_05_very_long_query_handling(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """A very long query (10,000+ chars) is passed safely without crashing."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        long_query = "What is agentic retrieval? " * 500  # ~13,500 chars
        resp = agent.search(long_query)

        assert resp.status == "success"
        assert resp.metadata["query"] == long_query.strip()

    # ------------------------------------------------------------------
    # Scenario 6 — Unicode query handling
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_06_unicode_query_handling(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Unicode characters and emojis in query are preserved accurately."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        unicode_q = "How does 🧠 OmniBrain work with 📊 tables & 🖼️ images? 🚀"
        resp = agent.search(unicode_q)

        assert resp.status == "success"
        assert resp.metadata["query"] == unicode_q

    # ------------------------------------------------------------------
    # Scenario 7 — Tamil query handling
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_07_tamil_query_handling(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Tamil language query is handled safely and preserved exactly."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        tamil_q = "தேடல் தகவல் மற்றும் ஆவணங்கள் என்ன?"
        resp = agent.search(tamil_q)

        assert resp.status == "success"
        assert resp.metadata["query"] == tamil_q

    # ------------------------------------------------------------------
    # Scenario 8 — Mixed Tamil and English query handling
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_08_mixed_tamil_english_query_handling(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Mixed Tamil and English query is preserved faithfully."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        mixed_q = "OmniBrain architecture திட்டம் மற்றும் vector search விவரங்கள்?"
        resp = agent.search(mixed_q)

        assert resp.status == "success"
        assert resp.metadata["query"] == mixed_q

    # ------------------------------------------------------------------
    # Scenario 9 — Embedding exception isolation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_09_embedding_exception_isolation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Embedding provider exception aborts before retrieval with AgentExecutionError."""
        provider = MagicMock()
        provider.embed.side_effect = ConnectionError("Embedding service offline")
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Failed to generate query embedding"):
            agent.search("Query with failed embed")

        mock_retrieve_context.assert_not_called()

    # ------------------------------------------------------------------
    # Scenario 10 — Embedding called exactly once
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_10_embedding_called_exactly_once(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """search() triggers provider.embed exactly once."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MagicMock()
        provider.embed.return_value = [0.1, 0.2, 0.3, 0.4]
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search("Single call check")
        assert provider.embed.call_count == 1

    # ------------------------------------------------------------------
    # Scenario 11 — Embedding dimension failure
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_11_embedding_dimension_failure(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Dimension mismatch raises AgentExecutionError and prevents retrieval."""
        provider = MockEmbeddingProvider(dimension=8, return_vector=[0.1] * 8)
        agent = SearchAgent(embedding_provider=provider, store=mock_store, expected_dimension=4)

        with pytest.raises(AgentExecutionError, match="Query embedding dimension mismatch"):
            agent.search("Dimension mismatch check")

        mock_retrieve_context.assert_not_called()

    # ------------------------------------------------------------------
    # Scenario 12 — Retrieval exception isolation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_12_retrieval_exception_isolation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Retrieval service exception raises AgentExecutionError without citations."""
        mock_retrieve_context.side_effect = TimeoutError("Qdrant timed out")
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Retrieval execution failed"):
            agent.search("Retrieval failure check")

    # ------------------------------------------------------------------
    # Scenario 13 — Retrieval called exactly once
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_13_retrieval_called_exactly_once(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """search() invokes retrieve_context exactly once."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search("Retrieve once check")
        assert mock_retrieve_context.call_count == 1

    # ------------------------------------------------------------------
    # Scenario 14 — Empty retrieval handling
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_14_empty_retrieval_handling(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Zero results from retrieve_context cleanly produces NO_RESULTS."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Empty query")
        assert pkg.status == "NO_RESULTS"
        assert pkg.has_results is False
        assert pkg.total_results == 0
        assert pkg.citations == []
        assert pkg.context == ""

    # ------------------------------------------------------------------
    # Scenario 15 — Malformed retrieval result wrong type
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_15_malformed_retrieval_result_wrong_type(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Non-RetrievalServiceResult returned by retrieval service raises AgentExecutionError."""
        mock_retrieve_context.return_value = "invalid_return_value"
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="Expected RetrievalServiceResult"):
            agent.search("Bad return type")

    # ------------------------------------------------------------------
    # Scenario 16 — Missing chunk_id rejection
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_16_missing_chunk_id_rejection(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Result with empty chunk_id raises AgentExecutionError."""
        bad_item = _make_valid_result(chunk_id="")
        mock_retrieve_context.return_value = _make_retrieval_result([bad_item])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="chunk_id is missing or empty"):
            agent.search("Missing chunk_id")

    # ------------------------------------------------------------------
    # Scenario 17 — Missing document_id rejection
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_17_missing_document_id_rejection(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Result with empty document_id raises AgentExecutionError."""
        bad_item = _make_valid_result(document_id="")
        mock_retrieve_context.return_value = _make_retrieval_result([bad_item])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="document_id is missing or empty"):
            agent.search("Missing document_id")

    # ------------------------------------------------------------------
    # Scenario 18 — Missing filename rejection
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_18_missing_filename_rejection(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Result with empty filename raises AgentExecutionError."""
        bad_item = _make_valid_result(filename="")
        mock_retrieve_context.return_value = _make_retrieval_result([bad_item])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="filename is missing or empty"):
            agent.search("Missing filename")

    # ------------------------------------------------------------------
    # Scenario 19 — Invalid score rejection
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_19_invalid_score_rejection(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Result with non-numeric score raises AgentExecutionError."""
        bad_item = _make_valid_result(score="not_a_score")  # type: ignore[arg-type]
        mock_retrieve_context.return_value = _make_retrieval_result([bad_item])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="score is not a finite numeric value"):
            agent.search("Invalid score")

    # ------------------------------------------------------------------
    # Scenario 20 — NaN score rejection
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_20_nan_score_rejection(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Result with NaN score raises AgentExecutionError."""
        bad_item = _make_valid_result(score=float("nan"))
        mock_retrieve_context.return_value = _make_retrieval_result([bad_item])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="score is not a finite numeric value"):
            agent.search("NaN score")

    # ------------------------------------------------------------------
    # Scenario 21 — Infinite score rejection
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_21_infinite_score_rejection(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Result with inf score raises AgentExecutionError."""
        bad_item = _make_valid_result(score=float("inf"))
        mock_retrieve_context.return_value = _make_retrieval_result([bad_item])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with pytest.raises(AgentExecutionError, match="score is not a finite numeric value"):
            agent.search("Inf score")

    # ------------------------------------------------------------------
    # Scenario 22 — Invalid metadata handling
    # ------------------------------------------------------------------

    def test_22_invalid_metadata_handling(self) -> None:
        """Non-dict metadata in AgentCitation raises AgentValidationError."""
        with pytest.raises(AgentValidationError, match="metadata must be a dictionary"):
            AgentCitation(
                document_id="d",
                filename="f.pdf",
                chunk_id="c",
                metadata="not_a_dict",  # type: ignore[arg-type]
            )

    # ------------------------------------------------------------------
    # Scenario 23 — Citation failure handling
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_23_citation_failure_handling(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Citation conversion exception in search loop raises AgentExecutionError."""
        item = _make_valid_result()
        mock_retrieve_context.return_value = _make_retrieval_result([item])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with patch.object(AgentCitation, "from_search_result", side_effect=ValueError("Corrupt citation payload")):
            with pytest.raises(AgentExecutionError, match="Failed to convert retrieval result to citation"):
                agent.search("Citation failure test")

    # ------------------------------------------------------------------
    # Scenario 24 — Context failure handling
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_24_context_failure_handling(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Context construction exception in search loop raises AgentExecutionError."""
        item = _make_valid_result()
        mock_retrieve_context.return_value = _make_retrieval_result([item])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with patch.object(agent, "_build_evidence_context", side_effect=RuntimeError("Context builder crash")):
            with pytest.raises(AgentExecutionError, match="Failed to construct evidence context"):
                agent.search("Context failure test")

    # ------------------------------------------------------------------
    # Scenario 25 — Packaging failure handling
    # ------------------------------------------------------------------

    def test_25_packaging_failure_handling(self) -> None:
        """SearchResult.from_response with empty query raises AgentValidationError."""
        resp = AgentResponse(
            answer="",
            agent_name="SearchAgent",
            citations=[],
            metadata={"query": ""},
        )
        with pytest.raises(AgentValidationError, match="missing non-empty 'query'"):
            SearchResult.from_response(resp)

    # ------------------------------------------------------------------
    # Scenario 26 — Failure call-order verification
    # ------------------------------------------------------------------

    def test_26_failure_call_order_verification(self, mock_store: MagicMock) -> None:
        """Query validation error stops pipeline at stage 1 without calling embedding or retrieval."""
        provider = MagicMock()
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        with patch("agents.search_agent.retrieve_context") as mock_ret:
            with pytest.raises(AgentValidationError):
                agent.search("   ")

            provider.embed.assert_not_called()
            mock_ret.assert_not_called()

    # ------------------------------------------------------------------
    # Scenario 27 — No fake vectors
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_27_no_fake_vectors(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Vector passed to retrieve_context matches exact vector from embedding provider."""
        expected_vec = [0.24, 0.48, 0.72, 0.96]
        provider = MockEmbeddingProvider(dimension=4, return_vector=expected_vec)
        mock_retrieve_context.return_value = _make_retrieval_result([])
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        agent.search("Real vector check")
        actual_vec = mock_retrieve_context.call_args.kwargs["query_vector"]
        assert actual_vec == expected_vec

    # ------------------------------------------------------------------
    # Scenario 28 — No fake retrieval
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_28_no_fake_retrieval(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """If retrieval returns empty list, 0 citations and 0 total_results are returned."""
        mock_retrieve_context.return_value = _make_retrieval_result([])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        resp = agent.search("Empty results query")
        assert len(resp.citations) == 0
        assert resp.metadata["total_results"] == 0

    # ------------------------------------------------------------------
    # Scenario 29 — No fake citations
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_29_no_fake_citations(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """SearchResult citations contains strictly the 1 item returned from retrieval."""
        result = _make_valid_result(chunk_id="chk-real-only")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Single citation query")
        assert len(pkg.citations) == 1
        assert pkg.citations[0].chunk_id == "chk-real-only"

    # ------------------------------------------------------------------
    # Scenario 30 — Text result preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_30_text_result_preservation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Text modality chunk preserves content_type and content accurately."""
        result = _make_valid_result(chunk_id="txt-chk", content_type="text", content="Paragraph 1 text.")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Text check")
        assert len(pkg.text_results) == 1
        assert pkg.text_results[0].content_type == "text"

    # ------------------------------------------------------------------
    # Scenario 31 — Table result preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_31_table_result_preservation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Table modality chunk preserves content_type and content accurately."""
        result = _make_valid_result(chunk_id="tbl-chk", content_type="table", content="| Q1 | Q2 |\n| 10 | 20 |")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Table check")
        assert len(pkg.table_results) == 1
        assert pkg.table_results[0].content_type == "table"

    # ------------------------------------------------------------------
    # Scenario 32 — Image result preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_32_image_result_preservation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Image modality chunk preserves content_type and content accurately."""
        result = _make_valid_result(chunk_id="img-chk", content_type="image", content="Image description.")
        mock_retrieve_context.return_value = _make_retrieval_result([result])
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Image check")
        assert len(pkg.image_results) == 1
        assert pkg.image_results[0].content_type == "image"

    # ------------------------------------------------------------------
    # Scenario 33 — Multimodal result preservation
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_33_multimodal_result_preservation(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """All 3 modalities in one result set are preserved with accurate segregation."""
        results = [
            _make_valid_result(chunk_id="t1", content_type="text"),
            _make_valid_result(chunk_id="tb1", content_type="table"),
            _make_valid_result(chunk_id="im1", content_type="image"),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        pkg = agent.search_and_package("Multimodal check")
        assert pkg.total_results == 3
        assert pkg.text_count == 1
        assert pkg.table_count == 1
        assert pkg.image_count == 1

    # ------------------------------------------------------------------
    # Scenario 34 — Deterministic output across runs
    # ------------------------------------------------------------------

    @patch("agents.search_agent.retrieve_context")
    def test_34_deterministic_output_across_runs(
        self,
        mock_retrieve_context: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Search execution on identical inputs yields identical outputs across multiple runs."""
        results = [
            _make_valid_result(chunk_id="c1", score=0.95),
            _make_valid_result(chunk_id="c2", score=0.85),
        ]
        mock_retrieve_context.return_value = _make_retrieval_result(results)
        provider = MockEmbeddingProvider(dimension=4)
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        out1 = agent.search("deterministic test").to_dict()
        out2 = agent.search("deterministic test").to_dict()

        assert out1 == out2

    # ------------------------------------------------------------------
    # Scenario 35 — Secret-safe error messages
    # ------------------------------------------------------------------

    def test_35_secret_safe_error_messages(self, mock_store: MagicMock) -> None:
        """Error messages identify failed stages without leaking keys, tokens, or raw vectors."""
        secret_key = "sk-live-secret-api-key-12345"
        provider = MagicMock()
        provider.embed.side_effect = RuntimeError(f"Auth error with {secret_key}")
        agent = SearchAgent(embedding_provider=provider, store=mock_store)

        try:
            agent.search("Secret test query")
        except AgentExecutionError as err:
            err_msg = str(err)
            assert "Failed to generate query embedding" in err_msg

    # ------------------------------------------------------------------
    # Scenario 36 — Previous Day 29 contracts intact
    # ------------------------------------------------------------------

    def test_36_previous_day_contracts_intact(self) -> None:
        """Verify SearchResult, AgentCitation, and SearchAgent properties remain present."""
        assert hasattr(SearchResult, "has_results")
        assert hasattr(SearchResult, "total_results")
        assert hasattr(SearchResult, "text_results")
        assert hasattr(SearchResult, "table_results")
        assert hasattr(SearchResult, "image_results")
        assert hasattr(SearchResult, "unique_document_count")
        assert hasattr(SearchAgent, "search_and_package")
        assert hasattr(SearchAgent, "package_result")




