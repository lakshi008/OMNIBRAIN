"""
Answer synthesizer for OmniBrain.

Constructs strictly grounded prompt instructions from retrieved citations and context,
calls the injected LLMProvider, and delivers grounded answers while strictly preserving
multimodal document lineage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agents.models import AgentCitation
from backend.llm.provider import LLMExecutionError, LLMProvider

logger = logging.getLogger(__name__)

GROUNDING_SYSTEM_PROMPT = """You are OmniBrain AI, an enterprise-grade multi-modal retrieval assistant.

Your task is to answer the user's question accurately, concisely, and STRICTLY using only the provided numbered sources below.

RULES:
1. Grounding: Answer ONLY based on the facts provided in the Context. Do NOT assume, extrapolate, or invent information not supported by the sources.
2. Citations: When making a factual claim, cite the supporting source using bracketed numbers like [Source 1], [Source 2], or [filename — Page X].
3. Insufficient Context: If the provided sources do not contain enough information to answer the question, state clearly: "Based on the provided documents, there is insufficient information to answer this question." Do not attempt to guess or answer from external knowledge.
4. Tone: Professional, objective, direct, and factual.
"""

NO_CONTEXT_FALLBACK_MESSAGE = "Based on the indexed documents, no relevant information was found for your query."


@dataclass
class SynthesisResult:
    """Result of LLM answer synthesis."""

    answer: str
    grounded: bool = True
    model_name: str = ""
    duration_seconds: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AnswerSynthesizer:
    """Orchestrates grounded answer generation over retrieved evidence context."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        system_prompt: str = GROUNDING_SYSTEM_PROMPT,
    ) -> None:
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt

    def _build_user_prompt(self, query: str, context: str) -> str:
        return f"""User Question:
{query.strip()}

Retrieved Evidence Context:
{context.strip()}

Instructions:
Synthesize a direct, grounded answer to the user question using only the context above. Include inline citations to [Source N] where appropriate."""

    def synthesize(
        self,
        query: str,
        citations: list[AgentCitation],
        context: str,
    ) -> SynthesisResult:
        """Synchronously synthesize an answer from retrieved citations and context."""
        start_time = time.perf_counter()

        # Rule 15: If no citations or empty context, do not hallucinate
        if not citations or not context.strip():
            return SynthesisResult(
                answer=NO_CONTEXT_FALLBACK_MESSAGE,
                grounded=True,
                duration_seconds=time.perf_counter() - start_time,
                metadata={"reason": "no_retrieved_context"},
            )

        if self.llm_provider is None:
            return SynthesisResult(
                answer="Retrieved relevant context and citations, but no LLM provider is configured to synthesize a summary answer.",
                grounded=False,
                duration_seconds=time.perf_counter() - start_time,
                error="LLM provider not configured",
                metadata={"reason": "no_llm_provider"},
            )

        prompt = self._build_user_prompt(query, context)

        try:
            raw_answer = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
            )
            duration = time.perf_counter() - start_time
            return SynthesisResult(
                answer=raw_answer,
                grounded=True,
                duration_seconds=duration,
                metadata={"citations_count": len(citations)},
            )
        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.warning("LLM answer synthesis failed: %s", exc)
            return SynthesisResult(
                answer=f"Retrieved {len(citations)} relevant citations, but failed to generate synthesis: {exc}",
                grounded=False,
                duration_seconds=duration,
                error=str(exc),
                metadata={"citations_count": len(citations), "error_type": type(exc).__name__},
            )

    async def asynthesize(
        self,
        query: str,
        citations: list[AgentCitation],
        context: str,
    ) -> SynthesisResult:
        """Asynchronously synthesize an answer from retrieved citations and context."""
        start_time = time.perf_counter()

        # Rule 15: If no citations or empty context, do not hallucinate
        if not citations or not context.strip():
            return SynthesisResult(
                answer=NO_CONTEXT_FALLBACK_MESSAGE,
                grounded=True,
                duration_seconds=time.perf_counter() - start_time,
                metadata={"reason": "no_retrieved_context"},
            )

        if self.llm_provider is None:
            return SynthesisResult(
                answer="Retrieved relevant context and citations, but no LLM provider is configured to synthesize a summary answer.",
                grounded=False,
                duration_seconds=time.perf_counter() - start_time,
                error="LLM provider not configured",
                metadata={"reason": "no_llm_provider"},
            )

        prompt = self._build_user_prompt(query, context)

        try:
            raw_answer = await self.llm_provider.agenerate(
                prompt=prompt,
                system_prompt=self.system_prompt,
            )
            duration = time.perf_counter() - start_time
            return SynthesisResult(
                answer=raw_answer,
                grounded=True,
                duration_seconds=duration,
                metadata={"citations_count": len(citations)},
            )
        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.warning("LLM async answer synthesis failed: %s", exc)
            return SynthesisResult(
                answer=f"Retrieved {len(citations)} relevant citations, but failed to generate synthesis: {exc}",
                grounded=False,
                duration_seconds=duration,
                error=str(exc),
                metadata={"citations_count": len(citations), "error_type": type(exc).__name__},
            )
