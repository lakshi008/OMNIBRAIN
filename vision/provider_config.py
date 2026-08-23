"""
Vision Model Provider Configuration and Capability Contracts for OmniBrain Member 3.

Defines strongly typed, validated representations for Vision Model Provider
configuration parameters and provider capability descriptions.

Day 36 Scope:
  - VisionProviderConfig: Provider identity, model name, timeouts, generation parameters.
  - VisionProviderCapabilities: Modality support, format support, image limits, features.
  - Pure configuration and capability contracts -- zero network, zero hard-coded secrets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vision.exceptions import VisionProviderConfigError
from vision.image_preparation import SUPPORTED_IMAGE_FORMATS
from vision.input_builder import VisionModelInput
from vision.models import VALID_VISUAL_CONTENT_TYPES


# ---------------------------------------------------------------------------
# VisionProviderConfig
# ---------------------------------------------------------------------------


@dataclass
class VisionProviderConfig:
    """Configuration contract for a Vision Model Provider.

    Encapsulates vendor-agnostic configuration parameters such as provider identifier,
    model name, timeouts, token limits, temperature, and extension parameters.

    Attributes:
        provider_name: Identifier for the provider (e.g. 'openai', 'gemini', 'claude', 'local', 'test').
        model_name: Target model identifier or descriptor (e.g. 'gpt-4o', 'gemini-1.5-pro').
        timeout: Maximum execution timeout in seconds (> 0.0). Defaults to 30.0.
        max_tokens: Optional maximum output generation tokens (> 0 or None).
        temperature: Optional sampling temperature between 0.0 and 2.0 (or None).
        max_input_images: Maximum number of visual inputs accepted in a single request (> 0).
        extra_params: Optional dictionary of additional provider-specific parameters.
    """

    provider_name: str
    model_name: str
    timeout: float = 30.0
    max_tokens: int | None = None
    temperature: float | None = None
    max_input_images: int = 1
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate provider configuration fields."""
        # provider_name validation
        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise VisionProviderConfigError("provider_name must be a non-empty string.")
        self.provider_name = self.provider_name.strip()

        # model_name validation
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise VisionProviderConfigError("model_name must be a non-empty string.")
        self.model_name = self.model_name.strip()

        # timeout validation
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0.0
        ):
            raise VisionProviderConfigError(
                f"timeout must be a positive finite number (> 0), got {self.timeout!r}."
            )
        self.timeout = float(self.timeout)

        # max_tokens validation
        if self.max_tokens is not None:
            if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
                raise VisionProviderConfigError(
                    f"max_tokens must be a positive integer (> 0) or None, got {self.max_tokens!r}."
                )

        # temperature validation
        if self.temperature is not None:
            if (
                isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature)
                or not (0.0 <= self.temperature <= 2.0)
            ):
                raise VisionProviderConfigError(
                    f"temperature must be a finite number between 0.0 and 2.0 or None, got {self.temperature!r}."
                )
            self.temperature = float(self.temperature)

        # max_input_images validation
        if (
            isinstance(self.max_input_images, bool)
            or not isinstance(self.max_input_images, int)
            or self.max_input_images <= 0
        ):
            raise VisionProviderConfigError(
                f"max_input_images must be a positive integer (> 0), got {self.max_input_images!r}."
            )

        # extra_params validation
        if not isinstance(self.extra_params, (dict, Mapping)):
            raise VisionProviderConfigError("extra_params must be a dictionary.")
        self.extra_params = dict(self.extra_params)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary representation."""
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_input_images": self.max_input_images,
            "extra_params": dict(self.extra_params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisionProviderConfig:
        """Construct a VisionProviderConfig from a dictionary."""
        if not isinstance(data, dict):
            raise VisionProviderConfigError("data must be a dictionary.")

        if "provider_name" not in data:
            raise VisionProviderConfigError("Missing required key 'provider_name' in configuration dictionary.")
        if "model_name" not in data:
            raise VisionProviderConfigError("Missing required key 'model_name' in configuration dictionary.")

        return cls(
            provider_name=data["provider_name"],
            model_name=data["model_name"],
            timeout=data.get("timeout", 30.0),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            max_input_images=data.get("max_input_images", 1),
            extra_params=data.get("extra_params", {}),
        )


# ---------------------------------------------------------------------------
# VisionProviderCapabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisionProviderCapabilities:
    """Describes the functional capabilities supported by a Vision Model Provider.

    Allows providers to explicitly declare which visual modalities, image formats,
    and features (multi-image, streaming, system prompts) they can handle.

    Attributes:
        supported_modalities: Set of visual content types supported ('image', 'chart', 'diagram').
        supported_formats: Set of image formats supported ('png', 'jpeg', 'webp').
        max_images: Maximum number of images allowed per invocation (>= 1).
        supports_streaming: Whether the provider supports token streaming responses.
        supports_multi_image: Whether the provider supports multiple images in a single query.
        supports_system_prompt: Whether the provider accepts system/developer instructions.
    """

    supported_modalities: frozenset[str] = field(
        default_factory=lambda: frozenset(VALID_VISUAL_CONTENT_TYPES)
    )
    supported_formats: frozenset[str] = field(
        default_factory=lambda: frozenset(SUPPORTED_IMAGE_FORMATS)
    )
    max_images: int = 1
    supports_streaming: bool = False
    supports_multi_image: bool = False
    supports_system_prompt: bool = True

    def __post_init__(self) -> None:
        """Validate capability fields."""
        # Convert modalities if needed and validate
        if not isinstance(self.supported_modalities, (set, frozenset, list, tuple)):
            raise VisionProviderConfigError("supported_modalities must be a collection of strings.")

        modalities = frozenset(str(m).strip().lower() for m in self.supported_modalities)
        if not modalities:
            raise VisionProviderConfigError("supported_modalities cannot be empty.")

        for m in modalities:
            if m not in VALID_VISUAL_CONTENT_TYPES:
                raise VisionProviderConfigError(
                    f"Invalid modality '{m}' in supported_modalities. "
                    f"Must be one of {sorted(VALID_VISUAL_CONTENT_TYPES)}."
                )
        object.__setattr__(self, "supported_modalities", modalities)

        # Convert formats if needed and validate
        if not isinstance(self.supported_formats, (set, frozenset, list, tuple)):
            raise VisionProviderConfigError("supported_formats must be a collection of strings.")

        formats = frozenset(str(f).strip().lower() for f in self.supported_formats)
        if not formats:
            raise VisionProviderConfigError("supported_formats cannot be empty.")

        for f in formats:
            if f not in SUPPORTED_IMAGE_FORMATS:
                raise VisionProviderConfigError(
                    f"Invalid format '{f}' in supported_formats. "
                    f"Must be one of {sorted(SUPPORTED_IMAGE_FORMATS)}."
                )
        object.__setattr__(self, "supported_formats", formats)

        # max_images validation
        if isinstance(self.max_images, bool) or not isinstance(self.max_images, int) or self.max_images <= 0:
            raise VisionProviderConfigError(
                f"max_images must be a positive integer (> 0), got {self.max_images!r}."
            )

        # bool flags validation
        for flag in ("supports_streaming", "supports_multi_image", "supports_system_prompt"):
            val = getattr(self, flag)
            if not isinstance(val, bool):
                raise VisionProviderConfigError(f"{flag} must be a boolean.")

    def supports_modality(self, modality: str) -> bool:
        """Check whether a specific visual modality is supported."""
        if not isinstance(modality, str):
            return False
        return modality.strip().lower() in self.supported_modalities

    def supports_format(self, image_format: str) -> bool:
        """Check whether a specific image format is supported."""
        if not isinstance(image_format, str):
            return False
        return image_format.strip().lower() in self.supported_formats

    def can_handle(self, model_input: VisionModelInput) -> bool:
        """Check whether this provider can handle a specific VisionModelInput.

        Args:
            model_input: Validated VisionModelInput from Day 35.

        Returns:
            True if modality and image_format are within supported capabilities, False otherwise.
        """
        if not isinstance(model_input, VisionModelInput):
            return False

        if not self.supports_modality(model_input.content_type):
            return False

        if not self.supports_format(model_input.image_format):
            return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert capabilities to dictionary representation."""
        return {
            "supported_modalities": sorted(self.supported_modalities),
            "supported_formats": sorted(self.supported_formats),
            "max_images": self.max_images,
            "supports_streaming": self.supports_streaming,
            "supports_multi_image": self.supports_multi_image,
            "supports_system_prompt": self.supports_system_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisionProviderCapabilities:
        """Construct a VisionProviderCapabilities instance from a dictionary."""
        if not isinstance(data, dict):
            raise VisionProviderConfigError("data must be a dictionary.")

        modalities = data.get("supported_modalities", VALID_VISUAL_CONTENT_TYPES)
        formats = data.get("supported_formats", SUPPORTED_IMAGE_FORMATS)

        return cls(
            supported_modalities=frozenset(modalities),
            supported_formats=frozenset(formats),
            max_images=data.get("max_images", 1),
            supports_streaming=data.get("supports_streaming", False),
            supports_multi_image=data.get("supports_multi_image", False),
            supports_system_prompt=data.get("supports_system_prompt", True),
        )
