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
from agents.models import AgentCitation, AgentRequest, AgentResponse, SearchRequest
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
        expected_dimension: int | None = None,
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
            expected_dimension: Optional positive integer for pre-retrieval query vector
                dimension validation. When provided, the generated embedding vector
                length is checked against this value before calling retrieve_context.
                A mismatch raises AgentExecutionError immediately — the vector is
                never truncated or padded.

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

        if expected_dimension is not None:
            if (
                not isinstance(expected_dimension, int)
                or isinstance(expected_dimension, bool)
                or expected_dimension <= 0
            ):
                raise AgentValidationError(
                    f"expected_dimension must be a positive integer or None, got {expected_dimension!r}."
                )

        self.embedding_provider = embedding_provider
        self.store = store
        self.collection_name = collection_name.strip()
        self.top_k = top_k
        self.min_score = float(min_score)
        self.max_results = max_results
        self.agent_name = agent_name.strip()
        self.expected_dimension = expected_dimension

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

    @staticmethod
    def _validate_result_integrity(item: VectorSearchResult, idx: int) -> None:
        """Validate field integrity of a single VectorSearchResult before citation conversion.

        Enforces that each retrieved evidence item carries complete, trustworthy lineage
        before it is promoted to an AgentCitation.  Raises immediately on the first
        violation — items are never silently skipped, repaired, or fabricated.

        Checks performed:
            - chunk_id: non-empty string (unique chunk provenance key).
            - document_id: non-empty string (parent document lineage key).
            - filename: non-empty string (source file identifier).
            - score: finite numeric float (as-returned by Member 1 retrieval, not modified).
            - content_type: non-empty string (modality tag required for supervisor routing).
            - chunk_index: non-negative integer (sequential position within document).

        Args:
            item: VectorSearchResult to validate.
            idx: 0-based position in the results list, used in error messages.

        Raises:
            AgentExecutionError: If any field fails integrity validation.
        """
        if not isinstance(item.chunk_id, str) or not item.chunk_id.strip():
            raise AgentExecutionError(
                f"Result at index {idx}: chunk_id is missing or empty — "
                "citation lineage cannot be established."
            )

        if not isinstance(item.document_id, str) or not item.document_id.strip():
            raise AgentExecutionError(
                f"Result at index {idx}: document_id is missing or empty — "
                "citation lineage cannot be established."
            )

        if not isinstance(item.filename, str) or not item.filename.strip():
            raise AgentExecutionError(
                f"Result at index {idx}: filename is missing or empty — "
                "source file attribution cannot be established."
            )

        if (
            not isinstance(item.score, (int, float))
            or isinstance(item.score, bool)
            or not math.isfinite(item.score)
        ):
            raise AgentExecutionError(
                f"Result at index {idx}: score is not a finite numeric value "
                f"({item.score!r}) — Member 1 relevance signal cannot be trusted."
            )

        if not isinstance(item.content_type, str) or not item.content_type.strip():
            raise AgentExecutionError(
                f"Result at index {idx}: content_type is missing or empty — "
                "modality routing for the supervisor cannot be determined."
            )

        if (
            not isinstance(item.chunk_index, int)
            or isinstance(item.chunk_index, bool)
            or item.chunk_index < 0
        ):
            raise AgentExecutionError(
                f"Result at index {idx}: chunk_index must be a non-negative integer, "
                f"got {item.chunk_index!r}."
            )

        # Day 26: content quality gate — evidence with no readable content cannot be cited
        if not isinstance(item.content, str) or not item.content.strip():
            raise AgentExecutionError(
                f"Result at index {idx}: content is empty or missing — "
                "evidence quality is insufficient for citation."
            )

    @staticmethod
    def _apply_member2_result_policy(
        results: list[VectorSearchResult],
        max_results: int,
    ) -> list[VectorSearchResult]:
        """Apply Member 2-level evidence policy after Member 1 retrieval returns.

        Provides a defensive boundary guarantee over the results returned by
        Member 1's retrieve_context without repeating the Qdrant search or
        duplicating any Member 1 processing logic.  Three operations:

        1. **Deduplication** — for any chunk_id that appears more than once,
           retain only the highest-scored entry.  When scores are equal the
           first occurrence in the sorted order wins.
        2. **Score ordering** — results are sorted strictly by score descending
           with deterministic tie-breaking on chunk_index (ascending) then
           chunk_id (lexicographic ascending).
        3. **Result cap** — the final list is sliced to max_results so the
           citation set never exceeds the configured limit.

        Member 1 already performs equivalent operations during retrieval.  This
        layer is the Member 2 contract enforcement point — it does not call
        retrieve_context again, does not perform vector similarity, and does
        not modify scores.

        Args:
            results: Type-checked list[VectorSearchResult] from Member 1.
            max_results: Maximum citations to return (positive integer).

        Returns:
            Deduplicated, score-ranked, capped list[VectorSearchResult].
        """
        if not results:
            return []

        # 1. Deduplicate by chunk_id: keep highest-scored entry per unique chunk.
        #    Items with invalid/empty chunk_ids are grouped under the same key;
        #    integrity validation (_validate_result_integrity) will reject them.
        seen: dict[str, VectorSearchResult] = {}
        for item in results:
            chunk_key = item.chunk_id
            if chunk_key not in seen or item.score > seen[chunk_key].score:
                seen[chunk_key] = item

        # 2. Sort descending by score; deterministic tie-breaking ensures stable output.
        deduped = list(seen.values())
        deduped.sort(key=lambda r: (-r.score, r.chunk_index, r.chunk_id))

        # 3. Enforce Member 2-side result cap (no additional retrieval calls).
        return deduped[:max_results]

    @staticmethod
    def _build_evidence_context(results: list[VectorSearchResult]) -> str:
        """Build structured, deterministic, citation-numbered context from accepted evidence items.

        Assigns 1-indexed source numbers ([Source 1], [Source 2], ...) following
        the exact descending relevance order of the filtered evidence items.
        Preserves all multimodal content types (text, table, image) faithfully.
        Returns an empty string if results is empty.

        Args:
            results: Ordered list of validated VectorSearchResult objects.

        Returns:
            Clean, formatted textual context string with deterministic [Source N] blocks.
        """
        if not results:
            return ""

        source_blocks: list[str] = []
        for idx, res in enumerate(results, start=1):
            page_str = str(res.page_number) if res.page_number is not None else "N/A"
            block = (
                f"[Source {idx}]\n"
                f"File: {res.filename}\n"
                f"Page: {page_str}\n"
                f"Type: {res.content_type}\n"
                f"Content:\n"
                f"{res.content}"
            )
            source_blocks.append(block)

        return "\n\n".join(source_blocks)

    def _extract_and_validate_request(
        self,
        request: str | AgentRequest | SearchRequest,
        top_k: int | None = None,
        min_score: float | None = None,
        max_results: int | None = None,
        collection_name: str | None = None,
    ) -> tuple[str, int, float, int, str, dict[str, Any]]:
        """Validate and extract query string, parameters, and metadata from request."""
        if request is None:
            raise AgentValidationError("Query request cannot be None.")

        if isinstance(request, str):
            cleaned = request.strip()
            if not cleaned:
                raise AgentValidationError("Query string cannot be empty or whitespace-only.")
            eff_top_k = self.top_k if top_k is None else top_k
            eff_min_score = self.min_score if min_score is None else float(min_score)
            eff_max_results = self.max_results if max_results is None else max_results
            eff_coll = (
                self.collection_name
                if collection_name is None or not collection_name.strip()
                else collection_name.strip()
            )
            return cleaned, eff_top_k, eff_min_score, eff_max_results, eff_coll, {}

        if isinstance(request, SearchRequest):
            cleaned = request.query.strip()
            if not cleaned:
                raise AgentValidationError("Query in SearchRequest cannot be empty or whitespace-only.")

            eff_top_k = (
                request.top_k
                if request.top_k is not None
                else (self.top_k if top_k is None else top_k)
            ) if top_k is None else top_k

            eff_min_score = (
                request.min_score
                if request.min_score is not None
                else (self.min_score if min_score is None else float(min_score))
            ) if min_score is None else float(min_score)

            eff_max_results = (
                request.max_results
                if request.max_results is not None
                else (self.max_results if max_results is None else max_results)
            ) if max_results is None else max_results

            req_coll = request.collection_name
            if collection_name is not None and collection_name.strip():
                eff_coll = collection_name.strip()
            elif req_coll is not None and req_coll.strip():
                eff_coll = req_coll.strip()
            else:
                eff_coll = self.collection_name

            metadata = dict(request.metadata)
            if request.session_id is not None:
                metadata["session_id"] = request.session_id
            if request.document_filter is not None:
                metadata["document_filter"] = request.document_filter
            return cleaned, eff_top_k, eff_min_score, eff_max_results, eff_coll, metadata

        if isinstance(request, AgentRequest):
            cleaned = request.query.strip()
            if not cleaned:
                raise AgentValidationError("Query in AgentRequest cannot be empty or whitespace-only.")
            metadata = dict(request.metadata)

            req_top_k = metadata.pop("top_k", None)
            eff_top_k = (
                req_top_k
                if isinstance(req_top_k, int) and not isinstance(req_top_k, bool)
                else (self.top_k if top_k is None else top_k)
            ) if top_k is None else top_k

            req_min_score = metadata.pop("min_score", None)
            eff_min_score = (
                float(req_min_score)
                if isinstance(req_min_score, (int, float)) and not isinstance(req_min_score, bool)
                else (self.min_score if min_score is None else float(min_score))
            ) if min_score is None else float(min_score)

            req_max_results = metadata.pop("max_results", None)
            eff_max_results = (
                req_max_results
                if isinstance(req_max_results, int) and not isinstance(req_max_results, bool)
                else (self.max_results if max_results is None else max_results)
            ) if max_results is None else max_results

            req_coll = metadata.pop("collection_name", None)
            if collection_name is not None and collection_name.strip():
                eff_coll = collection_name.strip()
            elif isinstance(req_coll, str) and req_coll.strip():
                eff_coll = req_coll.strip()
            else:
                eff_coll = self.collection_name

            if request.session_id is not None:
                metadata["session_id"] = request.session_id
            if request.document_filter is not None:
                metadata["document_filter"] = request.document_filter
            return cleaned, eff_top_k, eff_min_score, eff_max_results, eff_coll, metadata

        raise AgentValidationError(
            f"Expected query string, AgentRequest, or SearchRequest, got {type(request).__name__}."
        )

    def _generate_query_vector(self, query: str) -> list[float]:
        """Generate embedding vector for user query using Member 1 provider.

        Validates the produced vector for list type, non-empty length, numeric
        finite values, and (when self.expected_dimension is set) that the
        vector length matches the expected collection dimension.
        """
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

            # Day 25: dimension consistency check against expected collection dimension
            if self.expected_dimension is not None and len(cleaned_vector) != self.expected_dimension:
                raise AgentExecutionError(
                    f"Query embedding dimension mismatch: expected {self.expected_dimension}, "
                    f"got {len(cleaned_vector)} from provider. "
                    "Do not truncate or pad the vector — fix the provider or expected_dimension."
                )

            return cleaned_vector

        except Exception as err:
            if isinstance(err, AgentExecutionError):
                raise
            if isinstance(err, AgentValidationError):
                raise
            raise AgentExecutionError(
                f"Failed to generate query embedding: {err}"
            ) from err

    def search(
        self,
        request: str | AgentRequest | SearchRequest,
        top_k: int | None = None,
        min_score: float | None = None,
        max_results: int | None = None,
        collection_name: str | None = None,
    ) -> AgentResponse:
        """Execute structured search workflow for user query.

        Workflow:
            1. Validates query request and resolves search configuration.
            2. Generates dense query vector using Member 1 EmbeddingProvider.
            3. Executes similarity search and context building via Member 1 retrieve_context.
            4. Validates retrieval service response structure for type safety.
            5. Applies Member 2 evidence policy: dedup by chunk_id, descending sort, result cap.
            6. Validates per-item evidence field integrity (lineage, score, content).
            7. Converts validated VectorSearchResult objects into AgentCitation objects.
            8. Returns typed AgentResponse with normalized evidence and metadata.

        Args:
            request: Raw string query, AgentRequest instance, or SearchRequest instance.
            top_k: Optional override for initial nearest neighbors to retrieve.
            min_score: Optional override for minimum similarity score threshold.
            max_results: Optional override for maximum final results to return.
            collection_name: Optional override for target Qdrant collection.

        Returns:
            AgentResponse containing citations, formatted context in metadata, and search stats.

        Raises:
            AgentValidationError: If query or parameter validation fails.
            AgentExecutionError: If embedding generation, vector retrieval, or response processing fails.
        """
        # 1. Extract and validate query input and search parameters
        (
            query_text,
            effective_top_k,
            effective_min_score,
            effective_max_results,
            effective_collection,
            request_metadata,
        ) = self._extract_and_validate_request(
            request=request,
            top_k=top_k,
            min_score=min_score,
            max_results=max_results,
            collection_name=collection_name,
        )

        self._validate_search_params(
            top_k=effective_top_k,
            min_score=effective_min_score,
            max_results=effective_max_results,
        )

        # 2. Generate query embedding vector
        query_vector = self._generate_query_vector(query_text)

        # 3. Perform vector retrieval via Member 1 retrieval service
        try:
            retrieval_result = retrieve_context(
                query_vector=query_vector,
                store=self.store,
                collection_name=effective_collection,
                top_k=effective_top_k,
                min_score=effective_min_score,
                max_results=effective_max_results,
            )
        except Exception as err:
            if isinstance(err, (AgentValidationError, AgentExecutionError)):
                raise
            raise AgentExecutionError(
                f"Retrieval execution failed: {err}"
            ) from err

        # 4. Result type safety validation
        if not isinstance(retrieval_result, RetrievalServiceResult):
            raise AgentExecutionError(
                f"Expected RetrievalServiceResult from retrieval service, got {type(retrieval_result).__name__}."
            )

        if not isinstance(retrieval_result.results, list):
            raise AgentExecutionError(
                f"Expected list of results in RetrievalServiceResult, got {type(retrieval_result.results).__name__}."
            )

        for idx, item in enumerate(retrieval_result.results):
            if not isinstance(item, VectorSearchResult):
                raise AgentExecutionError(
                    f"Result item at index {idx} is not a VectorSearchResult: got {type(item).__name__}."
                )

        # 4b. Per-item evidence field integrity validation (Days 24/26)
        #     Validates lineage fields, score quality, content, and modality.
        #     Any malformed item immediately aborts the response — no silent repair.
        for idx, item in enumerate(retrieval_result.results):
            self._validate_result_integrity(item, idx)

        # 5. Member 2 evidence policy (Day 26)
        #    Deduplicates by chunk_id (highest score wins), sorts descending, caps at max_results.
        #    Does not repeat Qdrant search or any Member 1 computation.
        filtered_results = self._apply_member2_result_policy(
            retrieval_result.results, effective_max_results
        )

        # 6. Convert filtered VectorSearchResult objects to AgentCitation objects
        citations: list[AgentCitation] = []
        for item in filtered_results:
            try:
                citations.append(AgentCitation.from_search_result(item))
            except Exception as err:
                raise AgentExecutionError(
                    f"Failed to convert retrieval result to citation: {err}"
                ) from err

        # 7. Build citation-aware evidence context from the filtered result set
        evidence_context = self._build_evidence_context(filtered_results)

        # 8. Build normalized response metadata from the filtered result set
        final_text = sum(1 for r in filtered_results if r.content_type == "text")
        final_table = sum(1 for r in filtered_results if r.content_type == "table")
        final_image = sum(1 for r in filtered_results if r.content_type == "image")

        response_metadata: dict[str, Any] = {
            **request_metadata,
            "query": query_text,
            "context": evidence_context,
            "total_results": len(filtered_results),
            "evidence_count": len(filtered_results),
            "has_evidence": len(filtered_results) > 0,
            "text_results": final_text,
            "table_results": final_table,
            "image_results": final_image,
            "results_by_modality": {
                "text": final_text,
                "table": final_table,
                "image": final_image,
            },
            "query_vector_dimension": retrieval_result.query_vector_dimension,
            "collection_name": effective_collection,
            "top_k": effective_top_k,
            "min_score": effective_min_score,
            "max_results": effective_max_results,
        }

        # 9. Construct AgentResponse (Retrieval agent does NOT generate LLM answers)
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
        request: str | AgentRequest | SearchRequest,
        **kwargs: Any,
    ) -> AgentResponse:
        """Allow calling agent instance directly as a callable."""
        return self.search(request, **kwargs)

    def run(
        self,
        request: str | AgentRequest | SearchRequest,
        **kwargs: Any,
    ) -> AgentResponse:
        """Alias for search method."""
        return self.search(request, **kwargs)
