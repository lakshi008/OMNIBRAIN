"""
Search Agent for OmniBrain Member 2 Agentic RAG subsystem.

Provides structured query validation, embedding generation via Member 1 interfaces,
vector retrieval from Qdrant, result processing, citation conversion preserving
exact document lineage, and typed AgentResponse delivery.
"""

from __future__ import annotations

import math
from typing import Any

from agents.exceptions import AgentExecutionError, AgentValidationError
from agents.models import AgentCitation, AgentRequest, AgentResponse
from ingestion.embedding_generator import EmbeddingProvider
from ingestion.models import RetrievalServiceResult, VectorSearchResult
from ingestion.qdrant_store import QdrantVectorStore
from ingestion.retrieval_service import retrieve_context


class SearchAgent:
    """Retrieval agent responsible for evidence gathering and citation extraction.

    Coordinates query validation, dense embedding generation, Qdrant similarity
    search, citation conversion, and context building without generating LLM answers.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        store: QdrantVectorStore,
        collection_name: str = "documents",
        top_k: int = 5,
        min_score: float = 0.0,
        max_results: int = 5,
        agent_name: str = "SearchAgent",
    ) -> None:
        """Initialize SearchAgent with injected dependencies and search parameters.

        Args:
            embedding_provider: Real Member 1 EmbeddingProvider implementation.
            store: Real Member 1 QdrantVectorStore instance.
            collection_name: Name of target Qdrant collection.
            top_k: Default initial nearest neighbors to retrieve from Qdrant.
            min_score: Default minimum similarity score threshold (-1.0 to 1.0).
            max_results: Default maximum final processed results to return.
            agent_name: Name identifier for this agent.

        Raises:
            AgentValidationError: If dependencies or configuration parameters are invalid.
        """
        if embedding_provider is None or not (
            (hasattr(embedding_provider, "embed") and callable(getattr(embedding_provider, "embed")))
            or (hasattr(embedding_provider, "embed_batch") and callable(getattr(embedding_provider, "embed_batch")))
        ):
            raise AgentValidationError(
                "embedding_provider must implement 'embed' or 'embed_batch' method."
            )

        if not isinstance(store, QdrantVectorStore):
            raise AgentValidationError(
                f"store must be an instance of QdrantVectorStore, got {type(store).__name__}."
            )

        if not collection_name or not isinstance(collection_name, str) or not collection_name.strip():
            raise AgentValidationError("collection_name must be a non-empty string.")

        self._validate_search_params(top_k=top_k, min_score=min_score, max_results=max_results)

        if not agent_name or not isinstance(agent_name, str) or not agent_name.strip():
            raise AgentValidationError("agent_name must be a non-empty string.")

        self.embedding_provider = embedding_provider
        self.store = store
        self.collection_name = collection_name.strip()
        self.top_k = top_k
        self.min_score = float(min_score)
        self.max_results = max_results
        self.agent_name = agent_name.strip()

    @staticmethod
    def _validate_search_params(top_k: int, min_score: float, max_results: int) -> None:
        """Validate numeric search parameters."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise AgentValidationError(f"top_k must be a positive integer > 0, got {top_k!r}.")

        if (
            not isinstance(min_score, (int, float))
            or isinstance(min_score, bool)
            or not math.isfinite(min_score)
            or min_score < -1.0
            or min_score > 1.0
        ):
            raise AgentValidationError(
                f"min_score must be a finite float between -1.0 and 1.0, got {min_score!r}."
            )

        if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results <= 0:
            raise AgentValidationError(
                f"max_results must be a positive integer > 0, got {max_results!r}."
            )

    def _extract_and_validate_query(self, request: str | AgentRequest) -> tuple[str, dict[str, Any]]:
        """Validate and extract query string and optional request metadata."""
        if request is None:
            raise AgentValidationError("Query request cannot be None.")

        if isinstance(request, str):
            cleaned = request.strip()
            if not cleaned:
                raise AgentValidationError("Query string cannot be empty or whitespace-only.")
            return cleaned, {}

        if isinstance(request, AgentRequest):
            cleaned = request.query.strip()
            if not cleaned:
                raise AgentValidationError("Query in AgentRequest cannot be empty or whitespace-only.")
            metadata = dict(request.metadata)
            if request.session_id is not None:
                metadata["session_id"] = request.session_id
            if request.document_filter is not None:
                metadata["document_filter"] = request.document_filter
            return cleaned, metadata

        raise AgentValidationError(
            f"Expected query string or AgentRequest, got {type(request).__name__}."
        )

    def _generate_query_vector(self, query: str) -> list[float]:
        """Generate embedding vector for user query using Member 1 provider."""
        try:
            if hasattr(self.embedding_provider, "embed") and callable(getattr(self.embedding_provider, "embed")):
                vector = self.embedding_provider.embed(query)
            elif hasattr(self.embedding_provider, "embed_batch") and callable(getattr(self.embedding_provider, "embed_batch")):
                batch_vectors = self.embedding_provider.embed_batch([query])
                if not isinstance(batch_vectors, list) or len(batch_vectors) == 0:
                    raise ValueError(f"Provider returned invalid batch vector: {batch_vectors!r}")
                vector = batch_vectors[0]
            else:
                raise TypeError("Embedding provider does not implement 'embed' or 'embed_batch'.")

            if not isinstance(vector, list) or len(vector) == 0:
                raise ValueError(f"Provider returned empty or non-list vector: {vector!r}")

            cleaned_vector: list[float] = []
            for idx, val in enumerate(vector):
                if not isinstance(val, (int, float)) or isinstance(val, bool) or math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Query vector contains invalid numeric value at index {idx}: {val!r}")
                cleaned_vector.append(float(val))

            return cleaned_vector

        except Exception as err:
            if isinstance(err, AgentValidationError):
                raise
            raise AgentExecutionError(
                f"Failed to generate query embedding: {err}"
            ) from err

    def search(
        self,
        request: str | AgentRequest,
        top_k: int | None = None,
        min_score: float | None = None,
        max_results: int | None = None,
        collection_name: str | None = None,
    ) -> AgentResponse:
        """Execute structured search workflow for user query.

        Workflow:
            1. Validates query request.
            2. Generates dense query vector using Member 1 EmbeddingProvider.
            3. Executes similarity search and context building via Member 1 retrieve_context.
            4. Converts real VectorSearchResult objects into AgentCitation objects preserving lineage.
            5. Returns typed AgentResponse with evidence and metadata without LLM answers.

        Args:
            request: Raw string query or AgentRequest instance.
            top_k: Optional override for initial nearest neighbors to retrieve.
            min_score: Optional override for minimum similarity score threshold.
            max_results: Optional override for maximum final results to return.
            collection_name: Optional override for target Qdrant collection.

        Returns:
            AgentResponse containing citations, formatted context in metadata, and search stats.

        Raises:
            AgentValidationError: If query or parameter validation fails.
            AgentExecutionError: If embedding generation or vector retrieval fails.
        """
        # 1. Validate query input
        query_text, request_metadata = self._extract_and_validate_query(request)

        # 2. Determine effective search parameters
        effective_top_k = self.top_k if top_k is None else top_k
        effective_min_score = self.min_score if min_score is None else float(min_score)
        effective_max_results = self.max_results if max_results is None else max_results
        effective_collection = (
            self.collection_name
            if collection_name is None or not collection_name.strip()
            else collection_name.strip()
        )

        self._validate_search_params(
            top_k=effective_top_k,
            min_score=effective_min_score,
            max_results=effective_max_results,
        )

        # 3. Generate query embedding vector
        query_vector = self._generate_query_vector(query_text)

        # 4. Perform vector retrieval via Member 1 retrieval service
        try:
            retrieval_result: RetrievalServiceResult = retrieve_context(
                query_vector=query_vector,
                store=self.store,
                collection_name=effective_collection,
                top_k=effective_top_k,
                min_score=effective_min_score,
                max_results=effective_max_results,
            )
        except Exception as err:
            raise AgentExecutionError(
                f"Retrieval execution failed: {err}"
            ) from err

        # 5. Convert VectorSearchResult objects to AgentCitation objects preserving lineage
        citations: list[AgentCitation] = [
            AgentCitation.from_search_result(res)
            for res in retrieval_result.results
        ]

        # 6. Build response metadata
        response_metadata: dict[str, Any] = {
            **request_metadata,
            "query": query_text,
            "context": retrieval_result.context,
            "total_results": retrieval_result.total_results,
            "text_results": retrieval_result.text_results,
            "table_results": retrieval_result.table_results,
            "image_results": retrieval_result.image_results,
            "query_vector_dimension": retrieval_result.query_vector_dimension,
            "collection_name": effective_collection,
            "top_k": effective_top_k,
            "min_score": effective_min_score,
            "max_results": effective_max_results,
        }

        # 7. Construct AgentResponse (Retrieval agent does NOT generate LLM answers)
        return AgentResponse(
            answer="",
            agent_name=self.agent_name,
            status="success",
            citations=citations,
            metadata=response_metadata,
            error=None,
        )

    def __call__(
        self,
        request: str | AgentRequest,
        **kwargs: Any,
    ) -> AgentResponse:
        """Allow calling agent instance directly as a callable."""
        return self.search(request, **kwargs)

    def run(
        self,
        request: str | AgentRequest,
        **kwargs: Any,
    ) -> AgentResponse:
        """Alias for search method."""
        return self.search(request, **kwargs)
