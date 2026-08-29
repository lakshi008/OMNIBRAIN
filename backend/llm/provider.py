"""
Pluggable LLM provider abstraction for OmniBrain.

Defines the LLMProvider Protocol and production-grade implementations
supporting OpenAI-compatible REST APIs (including Groq, OpenAI, Ollama, DeepSeek, vLLM).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


# ── Exceptions ───────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Base exception for all LLM errors."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when LLM configuration or credentials are missing/invalid."""
    pass


class LLMExecutionError(LLMError):
    """Raised when an LLM API call fails, times out, or returns invalid data."""
    pass


# ── Protocol ─────────────────────────────────────────────────────────────────

@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for pluggable LLM generation backends."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronously generate text from a prompt.

        Args:
            prompt: User message content.
            system_prompt: Optional system grounding/instruction prompt.
            temperature: Optional sampling temperature override (0.0 - 2.0).
            max_tokens: Optional maximum output tokens override.
            **kwargs: Provider-specific additional parameters.

        Returns:
            Generated text string.
        """
        ...

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously generate text from a prompt.

        Args:
            prompt: User message content.
            system_prompt: Optional system grounding/instruction prompt.
            temperature: Optional sampling temperature override.
            max_tokens: Optional maximum output tokens override.
            **kwargs: Provider-specific additional parameters.

        Returns:
            Generated text string.
        """
        ...


# ── Base Class ───────────────────────────────────────────────────────────────

class BaseLLMProvider:
    """Base class for LLM providers with common parameter validation."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = max(0.0, min(2.0, temperature))
        self.max_tokens = max(1, max_tokens)
        self.timeout = max(1.0, timeout)

    def _validate_inputs(self, prompt: str) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")


# ── OpenAI-Compatible HTTP Implementation ────────────────────────────────────

class OpenAICompatibleLLMProvider(BaseLLMProvider):
    """Production LLM provider using OpenAI-compatible /chat/completions API.

    Works seamlessly with Groq, OpenAI, Together, DeepSeek, Local Ollama, etc.
    """

    def __init__(
        self,
        base_url: str = "https://api.groq.com/openai/v1",
        api_key: str | None = None,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        provider_name: str = "OpenAICompatible",
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self.base_url = base_url.rstrip("/")
        self.provider_name = provider_name

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            **kwargs,
        }
        return payload

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _parse_response(self, response_data: dict[str, Any]) -> str:
        try:
            choices = response_data.get("choices")
            if not choices or not isinstance(choices, list):
                raise ValueError("Response missing 'choices' array.")
            message = choices[0].get("message")
            if not message or not isinstance(message, dict):
                raise ValueError("Choice missing 'message' object.")
            content = message.get("content", "")
            if not isinstance(content, str):
                raise ValueError("Message content is not a string.")
            return content.strip()
        except Exception as exc:
            raise LLMExecutionError(f"Malformed response structure from LLM: {exc}") from exc

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronously generate answer text."""
        self._validate_inputs(prompt)
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()

        start_time = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload, headers=headers)
                if resp.status_code != 200:
                    error_detail = resp.text
                    try:
                        error_json = resp.json()
                        error_detail = error_json.get("error", {}).get("message", resp.text)
                    except Exception:
                        pass
                    raise LLMExecutionError(
                        f"{self.provider_name} API returned status {resp.status_code}: {error_detail}"
                    )
                data = resp.json()
                answer = self._parse_response(data)
                duration = time.perf_counter() - start_time
                logger.info(
                    "%s generated %d chars in %.2fs (model=%s)",
                    self.provider_name,
                    len(answer),
                    duration,
                    self.model_name,
                )
                return answer
        except httpx.TimeoutException as exc:
            raise LLMExecutionError(f"{self.provider_name} request timed out after {self.timeout}s") from exc
        except httpx.RequestError as exc:
            raise LLMExecutionError(f"{self.provider_name} network request failed: {exc}") from exc
        except LLMError:
            raise
        except Exception as exc:
            raise LLMExecutionError(f"Unexpected error during {self.provider_name} generation: {exc}") from exc

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously generate answer text."""
        self._validate_inputs(prompt)
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code != 200:
                    error_detail = resp.text
                    try:
                        error_json = resp.json()
                        error_detail = error_json.get("error", {}).get("message", resp.text)
                    except Exception:
                        pass
                    raise LLMExecutionError(
                        f"{self.provider_name} API returned status {resp.status_code}: {error_detail}"
                    )
                data = resp.json()
                answer = self._parse_response(data)
                duration = time.perf_counter() - start_time
                logger.info(
                    "%s async generated %d chars in %.2fs (model=%s)",
                    self.provider_name,
                    len(answer),
                    duration,
                    self.model_name,
                )
                return answer
        except httpx.TimeoutException as exc:
            raise LLMExecutionError(f"{self.provider_name} request timed out after {self.timeout}s") from exc
        except httpx.RequestError as exc:
            raise LLMExecutionError(f"{self.provider_name} network request failed: {exc}") from exc
        except LLMError:
            raise
        except Exception as exc:
            raise LLMExecutionError(f"Unexpected error during {self.provider_name} generation: {exc}") from exc


# ── Groq Specific Provider ───────────────────────────────────────────────────

class GroqLLMProvider(OpenAICompatibleLLMProvider):
    """Groq Cloud LLM provider (ultra-fast inference)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> None:
        resolved_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
        super().__init__(
            base_url="https://api.groq.com/openai/v1",
            api_key=resolved_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            provider_name="Groq",
        )


# ── Factory Function ─────────────────────────────────────────────────────────

def create_llm_provider(
    provider_type: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> LLMProvider | None:
    """Construct an LLMProvider instance from explicit arguments or environment variables.

    Environment variables:
        LLM_PROVIDER: 'groq' | 'openai' | 'openai_compatible' | 'ollama'
        LLM_API_KEY / GROQ_API_KEY / OPENAI_API_KEY
        LLM_MODEL
        LLM_BASE_URL
        LLM_TEMPERATURE
        LLM_MAX_TOKENS
        LLM_TIMEOUT
    """
    p_type = (provider_type or os.getenv("LLM_PROVIDER") or "groq").lower().strip()
    key = api_key or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    temp = float(os.getenv("LLM_TEMPERATURE", "0.1")) if temperature is None else temperature
    tokens = int(os.getenv("LLM_MAX_TOKENS", "1024")) if max_tokens is None else max_tokens
    t_out = float(os.getenv("LLM_TIMEOUT", "30.0")) if timeout is None else timeout

    if p_type in ("groq", "default"):
        model = model_name or "llama-3.3-70b-versatile"
        return GroqLLMProvider(
            api_key=key,
            model_name=model,
            temperature=temp,
            max_tokens=tokens,
            timeout=t_out,
        )

    if p_type == "openai":
        model = model_name or os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        b_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        return OpenAICompatibleLLMProvider(
            base_url=b_url,
            api_key=key,
            model_name=model,
            temperature=temp,
            max_tokens=tokens,
            timeout=t_out,
            provider_name="OpenAI",
        )

    if p_type in ("openai_compatible", "ollama", "custom"):
        model = model_name or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        b_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
        return OpenAICompatibleLLMProvider(
            base_url=b_url,
            api_key=key,
            model_name=model,
            temperature=temp,
            max_tokens=tokens,
            timeout=t_out,
            provider_name=p_type.capitalize(),
        )

    logger.warning("Unrecognized LLM_PROVIDER '%s'. Falling back to Groq.", p_type)
    return GroqLLMProvider(
        api_key=key,
        model_name=model_name or "llama-3.3-70b-versatile",
        temperature=temp,
        max_tokens=tokens,
        timeout=t_out,
    )
