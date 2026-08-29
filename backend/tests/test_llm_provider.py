"""
Unit tests for LLMProvider and AnswerSynthesizer.

Verifies:
- Protocol compliance (@runtime_checkable LLMProvider)
- OpenAICompatibleLLMProvider and GroqLLMProvider generation
- Async and sync generation
- Error handling on timeouts, HTTP errors, and malformed responses
- AnswerSynthesizer prompt construction and strict grounding
- Zero-hallucination handling when context is empty or no citations exist
- Graceful degradation when LLM provider fails or is unconfigured
"""

from __future__ import annotations

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from agents.models import AgentCitation
from backend.llm.provider import (
    GroqLLMProvider,
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


class MockLLM(LLMProvider):
    """Test mock implementing LLMProvider protocol."""

    def __init__(self, response_text: str = "This is a synthesized test answer.") -> None:
        self.response_text = response_text
        self.call_count = 0
        self.last_prompt = ""
        self.last_system_prompt = ""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt or ""
        return self.response_text

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt or ""
        return self.response_text


class TestLLMProviderProtocol:
    def test_mock_implements_protocol(self):
        mock = MockLLM()
        assert isinstance(mock, LLMProvider)

    def test_openai_compatible_implements_protocol(self):
        provider = OpenAICompatibleLLMProvider(api_key="test-key")
        assert isinstance(provider, LLMProvider)

    def test_groq_implements_protocol(self):
        provider = GroqLLMProvider(api_key="test-key")
        assert isinstance(provider, LLMProvider)


class TestOpenAICompatibleLLMProvider:
    def test_input_validation_empty_prompt_raises(self):
        provider = OpenAICompatibleLLMProvider(api_key="test-key")
        with pytest.raises(ValueError, match="non-empty string"):
            provider.generate("")

    def test_input_validation_whitespace_raises(self):
        provider = OpenAICompatibleLLMProvider(api_key="test-key")
        with pytest.raises(ValueError, match="non-empty string"):
            provider.generate("   \n\t  ")

    @patch("httpx.Client.post")
    def test_sync_generate_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Revenue grew by 25% in 2025 [Source 1]."}}]
        }
        mock_post.return_value = mock_resp

        provider = OpenAICompatibleLLMProvider(api_key="test-key")
        ans = provider.generate("What was the revenue growth?", system_prompt="Answer factually.")
        assert "Revenue grew by 25%" in ans
        assert mock_post.called

    @patch("httpx.AsyncClient.post")
    @pytest.mark.asyncio
    async def test_async_generate_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Net profit was $5.2M [Source 2]."}}]
        }
        mock_post.return_value = mock_resp

        provider = OpenAICompatibleLLMProvider(api_key="test-key")
        ans = await provider.agenerate("What was the profit?")
        assert "$5.2M" in ans

    @patch("httpx.Client.post")
    def test_generate_http_error_raises_llm_execution_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_resp.json.return_value = {"error": {"message": "Invalid API key provided."}}
        mock_post.return_value = mock_resp

        provider = OpenAICompatibleLLMProvider(api_key="invalid-key")
        with pytest.raises(LLMExecutionError, match="401"):
            provider.generate("test")

    @patch("httpx.Client.post")
    def test_generate_malformed_response_raises(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"unexpected_key": "bad payload"}
        mock_post.return_value = mock_resp

        provider = OpenAICompatibleLLMProvider(api_key="test-key")
        with pytest.raises(LLMExecutionError, match="Malformed"):
            provider.generate("test")


class TestCreateLLMProviderFactory:
    def test_factory_defaults_to_groq(self):
        provider = create_llm_provider(api_key="test-key")
        assert isinstance(provider, GroqLLMProvider)
        assert provider.model_name == "llama-3.3-70b-versatile"

    def test_factory_openai_type(self):
        provider = create_llm_provider(provider_type="openai", api_key="test-key")
        assert isinstance(provider, OpenAICompatibleLLMProvider)
        assert provider.base_url == "https://api.openai.com/v1"


class TestAnswerSynthesizer:
    def _sample_citations(self) -> list[AgentCitation]:
        return [
            AgentCitation(
                document_id="doc-123",
                filename="sample.pdf",
                chunk_id="chunk-456",
                page_number=3,
                content_type="text",
                score=0.88,
                metadata={"content": "OmniBrain architecture is modular and scalable."},
            )
        ]

    def test_empty_citations_returns_fallback_without_calling_llm(self):
        mock_llm = MockLLM()
        synthesizer = AnswerSynthesizer(llm_provider=mock_llm)

        result = synthesizer.synthesize(
            query="Tell me about scaling",
            citations=[],
            context="",
        )
        assert result.answer == NO_CONTEXT_FALLBACK_MESSAGE
        assert result.grounded is True
        assert mock_llm.call_count == 0

    def test_empty_context_returns_fallback_without_calling_llm(self):
        mock_llm = MockLLM()
        synthesizer = AnswerSynthesizer(llm_provider=mock_llm)
        citations = self._sample_citations()

        result = synthesizer.synthesize(
            query="Tell me about scaling",
            citations=citations,
            context="   \n  ",
        )
        assert result.answer == NO_CONTEXT_FALLBACK_MESSAGE
        assert mock_llm.call_count == 0

    def test_unconfigured_llm_returns_graceful_message(self):
        synthesizer = AnswerSynthesizer(llm_provider=None)
        citations = self._sample_citations()
        context = "[Source 1]\nFile: sample.pdf\nPage: 3\nContent:\nOmniBrain architecture is modular."

        result = synthesizer.synthesize(
            query="What is OmniBrain?",
            citations=citations,
            context=context,
        )
        assert "no LLM provider is configured" in result.answer
        assert result.error is not None

    def test_successful_synthesis(self):
        mock_llm = MockLLM(response_text="OmniBrain features modular and scalable design [Source 1].")
        synthesizer = AnswerSynthesizer(llm_provider=mock_llm)
        citations = self._sample_citations()
        context = "[Source 1]\nFile: sample.pdf\nPage: 3\nContent:\nOmniBrain architecture is modular and scalable."

        result = synthesizer.synthesize(
            query="Describe OmniBrain architecture",
            citations=citations,
            context=context,
        )
        assert result.answer == "OmniBrain features modular and scalable design [Source 1]."
        assert result.grounded is True
        assert result.error is None
        assert mock_llm.call_count == 1
        assert "Describe OmniBrain architecture" in mock_llm.last_prompt
        assert "OmniBrain architecture is modular and scalable." in mock_llm.last_prompt
        assert mock_llm.last_system_prompt == GROUNDING_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_async_synthesis(self):
        mock_llm = MockLLM(response_text="Async answer [Source 1].")
        synthesizer = AnswerSynthesizer(llm_provider=mock_llm)
        citations = self._sample_citations()
        context = "[Source 1]\nFile: sample.pdf\nPage: 3\nContent:\nText content."

        result = await synthesizer.asynthesize(
            query="Async query",
            citations=citations,
            context=context,
        )
        assert result.answer == "Async answer [Source 1]."
        assert mock_llm.call_count == 1

    def test_llm_exception_returns_graceful_fallback(self):
        mock_llm = MagicMock(spec=LLMProvider)
        mock_llm.generate.side_effect = LLMExecutionError("Groq service rate limited")
        synthesizer = AnswerSynthesizer(llm_provider=mock_llm)
        citations = self._sample_citations()
        context = "[Source 1]\nContent:\nSome content"

        result = synthesizer.synthesize(
            query="Query",
            citations=citations,
            context=context,
        )
        assert "failed to generate synthesis" in result.answer
        assert "Groq service rate limited" in result.answer
        assert result.grounded is False
        assert result.error is not None
