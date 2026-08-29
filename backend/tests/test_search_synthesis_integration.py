"""
Integration tests for search and LLM Answer Synthesis.

Tests full pipeline flow:
User query
  ↓
query embedding (RealEmbeddingProvider)
  ↓
Qdrant retrieval (QdrantVectorStore)
  ↓
context construction
  ↓
LLM answer generation (AnswerSynthesizer)
  ↓
citations in final response (SearchResponse)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agents.search_agent import SearchAgent
from backend.embedding_provider import SentenceTransformerEmbeddingProvider
from backend.llm.provider import LLMProvider
from backend.llm.synthesizer import AnswerSynthesizer, NO_CONTEXT_FALLBACK_MESSAGE
from backend.main import app
from backend.services.search_service import run_search
from ingestion.models import (
    EmbeddingGenerationResult,
    EmbeddingVectorRecord,
)
from ingestion.qdrant_config import QdrantConfig
from ingestion.qdrant_store import QdrantVectorStore


class MockGroundedLLM(LLMProvider):
    """Deterministic mock LLM for testing search synthesis integration."""

    def __init__(self, answer: str = "OmniBrain uses a multi-agent LangGraph workflow [Source 1].") -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        self.calls += 1
        return self.answer

    async def agenerate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        self.calls += 1
        return self.answer


class TestSearchSynthesisIntegration:
    @pytest.fixture
    def test_environment(self):
        """Set up in-memory Qdrant store with sample document embeddings."""
        store = QdrantVectorStore(config=QdrantConfig(url=":memory:"))
        collection = "test_synthesis_collection"
        dim = 384
        store.create_collection(collection, vector_dimension=dim)

        # Create embedding provider
        encoder_provider = SentenceTransformerEmbeddingProvider(dimension=dim)

        # Generate a real embedding for our test chunk
        chunk_content = "OmniBrain architecture is built on an enterprise-grade agentic multi-modal RAG system with LangGraph supervisor."
        real_vector = encoder_provider.embed(chunk_content)

        record = EmbeddingVectorRecord(
            chunk_id="00000000-0000-0000-0000-000000000001",
            document_id="00000000-0000-0000-0000-000000000002",
            filename="omni_specs.pdf",
            chunk_index=0,
            page_number=1,
            content_type="text",
            vector=real_vector,
            metadata={"content": chunk_content},
        )

        gen_result = EmbeddingGenerationResult(
            document_id="00000000-0000-0000-0000-000000000002",
            filename="omni_specs.pdf",
            items=[record],
            dimension=dim,
            is_ready=True,
        )

        store.upsert_embeddings(collection, gen_result)

        search_agent = SearchAgent(
            embedding_provider=encoder_provider,
            store=store,
            collection_name=collection,
            top_k=5,
            min_score=0.0,
            max_results=5,
            expected_dimension=dim,
        )

        return {
            "store": store,
            "collection": collection,
            "provider": encoder_provider,
            "agent": search_agent,
            "content": chunk_content,
        }

    @pytest.mark.asyncio
    async def test_full_search_and_synthesis_flow(self, test_environment):
        agent = test_environment["agent"]
        mock_llm = MockGroundedLLM(
            answer="Based on the specifications, OmniBrain uses an agentic multi-modal RAG system [Source 1]."
        )
        synthesizer = AnswerSynthesizer(llm_provider=mock_llm)

        response = await run_search(
            query="What is OmniBrain architecture?",
            search_agent=agent,
            synthesizer=synthesizer,
            top_k=3,
        )

        # Verify search status and total results
        assert response.status == "RESULTS_FOUND"
        assert response.total_results == 1
        assert len(response.results) == 1

        # Verify citation lineage
        citation = response.results[0]
        assert citation.filename == "omni_specs.pdf"
        assert citation.page == 1
        assert citation.chunk_id == "00000000-0000-0000-0000-000000000001"
        assert citation.score > 0.0
        assert "omni_specs.pdf — Page 1" in citation.citation
        assert test_environment["content"] in citation.content

        # Verify LLM answer was synthesized and attached
        assert "OmniBrain uses an agentic multi-modal RAG system" in response.answer
        assert "[Source 1]" in response.answer
        assert mock_llm.calls == 1

        # Verify context is preserved
        assert "[Source 1]" in response.context
        assert "omni_specs.pdf" in response.context

    @pytest.mark.asyncio
    async def test_empty_collection_returns_fallback_without_llm_call(self, test_environment):
        store = test_environment["store"]
        empty_col = "empty_test_collection"
        store.create_collection(empty_col, vector_dimension=384)

        agent = SearchAgent(
            embedding_provider=test_environment["provider"],
            store=store,
            collection_name=empty_col,
            top_k=5,
            expected_dimension=384,
        )

        mock_llm = MockGroundedLLM()
        synthesizer = AnswerSynthesizer(llm_provider=mock_llm)

        response = await run_search(
            query="Find anything in empty collection",
            search_agent=agent,
            synthesizer=synthesizer,
        )

        assert response.status == "NO_RESULTS"
        assert response.total_results == 0
        assert response.answer == NO_CONTEXT_FALLBACK_MESSAGE
        # Confirms Rule 15: LLM was not called when no context exists
        assert mock_llm.calls == 0

    def test_api_endpoint_search_with_synthesizer(self):
        client = TestClient(app)
        response = client.post(
            "/api/search",
            json={"query": "test query for API endpoint", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "answer" in data
        assert "results" in data
        assert "status" in data
        assert isinstance(data["results"], list)
