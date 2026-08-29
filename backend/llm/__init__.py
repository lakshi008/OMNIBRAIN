"""
LLM answer synthesis package for OmniBrain.

Provides pluggable LLM provider abstractions, prompt grounding engineering,
and citation-aware response synthesis.
"""

from __future__ import annotations

from backend.llm.provider import (
    BaseLLMProvider,
    GroqLLMProvider,
    LLMConfigurationError,
    LLMExecutionError,
    LLMProvider,
    OpenAICompatibleLLMProvider,
    create_llm_provider,
)
from backend.llm.synthesizer import (
    AnswerSynthesizer,
    GROUNDING_SYSTEM_PROMPT,
    NO_CONTEXT_FALLBACK_MESSAGE,
    SynthesisResult,
)

__all__ = [
    "LLMProvider",
    "BaseLLMProvider",
    "OpenAICompatibleLLMProvider",
    "GroqLLMProvider",
    "LLMConfigurationError",
    "LLMExecutionError",
    "create_llm_provider",
    "AnswerSynthesizer",
    "SynthesisResult",
    "GROUNDING_SYSTEM_PROMPT",
    "NO_CONTEXT_FALLBACK_MESSAGE",
]
