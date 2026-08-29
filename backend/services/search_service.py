"""
Search service with LLM Answer Synthesis.

Bridges FastAPI search routes to the existing SearchAgent retrieval layer
and AnswerSynthesizer grounded LLM generation layer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents.models import AgentCitation, AgentResponse
from agents.search_agent import SearchAgent
from backend.llm.synthesizer import AnswerSynthesizer
from backend.schemas.search import CitationItem, SearchResponse

logger = logging.getLogger(__name__)


def _make_citation_string(citation: AgentCitation) -> str:
    """Build a human-readable citation label from an AgentCitation."""
    page_part = f" — Page {citation.page_number}" if citation.page_number is not None else ""
    return f"{citation.filename}{page_part}"


def _citation_to_item(citation: AgentCitation) -> CitationItem:
    """Convert an AgentCitation to a CitationItem schema."""
    content = citation.metadata.get("content", "")
    return CitationItem(
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        filename=citation.filename,
        page=citation.page_number,
        content_type=citation.content_type,
        score=citation.score,
        citation=_make_citation_string(citation),
        content=content,
        metadata=dict(citation.metadata),
    )


async def run_search(
    query: str,
    search_agent: SearchAgent,
    synthesizer: AnswerSynthesizer | None = None,
    top_k: int = 5,
    min_score: float = 0.0,
    max_results: int = 5,
    collection_name: str | None = None,
) -> SearchResponse:
    """Execute a RAG search via SearchAgent and optionally synthesize grounded answer via LLM.

    Workflow:
    1. Vector retrieval & citation packaging via SearchAgent.
    2. Grounded answer synthesis via AnswerSynthesizer using retrieved context.
    3. Packaging into typed SearchResponse preserving complete lineage.

    Args:
        query: User question.
        search_agent: Configured SearchAgent instance.
        synthesizer: Optional AnswerSynthesizer instance.
        top_k: Initial candidate retrieval count.
        min_score: Minimum similarity threshold.
        max_results: Maximum final citations.
        collection_name: Optional collection override.

    Returns:
        Structured SearchResponse for the API.
    """
    try:
        kwargs: dict[str, Any] = {
            "top_k": top_k,
            "min_score": min_score,
            "max_results": max_results,
        }
        if collection_name:
            kwargs["collection_name"] = collection_name

        # 1. Retrieve candidates & build context
        response: AgentResponse = await asyncio.to_thread(
            search_agent.search, query, **kwargs
        )

        items = [_citation_to_item(c) for c in response.citations]
        status = "RESULTS_FOUND" if response.citations else "NO_RESULTS"
        eff_collection = response.metadata.get("collection_name", "")
        context = response.metadata.get("context", "")

        # 2. Synthesize answer if synthesizer is provided
        answer = response.answer
        if synthesizer is not None:
            synthesis = await synthesizer.asynthesize(
                query=query,
                citations=response.citations,
                context=context,
            )
            answer = synthesis.answer

        return SearchResponse(
            query=query,
            answer=answer,
            status=status,
            total_results=len(items),
            results=items,
            context=context,
            collection_name=eff_collection,
            error=response.error,
        )

    except Exception as exc:
        logger.error("Search failed for query=%r: %s", query, exc)
        return SearchResponse(
            query=query,
            answer="",
            status="ERROR",
            total_results=0,
            results=[],
            context="",
            collection_name=collection_name or "",
            error=str(exc),
        )
