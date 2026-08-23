"""
Vision Model Provider Interface and Registry for OmniBrain Member 3 Vision Agent.

Establishes the provider abstraction contract between VisionModelInput and
future Vision Model provider implementations (local, cloud, commercial, or open-source).

Day 36 Scope:
  - VisionModelProvider: Abstract base class defining the provider execution contract.
  - VisionProviderRegistry: Central registry for registering and creating providers.
  - Input validation: Type checking, capability enforcement, lineage preservation.
  - Pure abstraction -- zero network, zero LLM, zero fake production inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vision.exceptions import (
    VisionInputValidationError,
    VisionProviderConfigError,
    VisionProviderError,
    VisionProviderUnavailableError,
    VisionUnsupportedCapabilityError,
)
from vision.input_builder import VisionModelInput
from vision.models import VisionResult
from vision.provider_config import (
    VisionProviderCapabilities,
    VisionProviderConfig,
)


# ---------------------------------------------------------------------------
# VisionModelProvider (Abstract Interface)
# ---------------------------------------------------------------------------


class VisionModelProvider(ABC):
    """Abstract interface that every Vision Model Provider implementation must satisfy.

    A concrete provider receives a standardized VisionModelInput, validates that the
    input conforms to its declared capabilities, executes vision reasoning (in future days),
    and returns a standardized VisionResult without exposing provider-specific details.

    Attributes:
        config: Strongly typed VisionProviderConfig instance.
        capabilities: Declared functional capabilities of this provider instance.
    """

    def __init__(
        self,
        config: VisionProviderConfig,
        capabilities: VisionProviderCapabilities | None = None,
    ) -> None:
        """Initialize the VisionModelProvider with configuration and capabilities.

        Args:
            config: Validated VisionProviderConfig instance.
            capabilities: Optional VisionProviderCapabilities descriptor.
                          Defaults to standard capabilities if None.

        Raises:
            VisionProviderConfigError: If config is not a VisionProviderConfig instance.
        """
        if not isinstance(config, VisionProviderConfig):
            raise VisionProviderConfigError(
                f"config must be a VisionProviderConfig instance, got {type(config).__name__}."
            )

        if capabilities is not None and not isinstance(capabilities, VisionProviderCapabilities):
            raise VisionProviderConfigError(
                f"capabilities must be a VisionProviderCapabilities instance or None, "
                f"got {type(capabilities).__name__}."
            )

        self._config: VisionProviderConfig = config
        self._capabilities: VisionProviderCapabilities = (
            capabilities if capabilities is not None else VisionProviderCapabilities()
        )

    @property
    def config(self) -> VisionProviderConfig:
        """Return the provider configuration."""
        return self._config

    @property
    def capabilities(self) -> VisionProviderCapabilities:
        """Return the declared capabilities of this provider."""
        return self._capabilities

    @property
    def provider_name(self) -> str:
        """Return the identifier of this provider."""
        return self._config.provider_name

    @property
    def model_name(self) -> str:
        """Return the model identifier of this provider."""
        return self._config.model_name

    def validate_input(self, model_input: VisionModelInput) -> None:
        """Validate that a VisionModelInput is valid and within this provider's capabilities.

        Args:
            model_input: Standardized VisionModelInput instance from Day 35.

        Raises:
            VisionInputValidationError: If model_input is None or wrong type.
            VisionUnsupportedCapabilityError: If modality or format is not supported.
        """
        if model_input is None:
            raise VisionInputValidationError("model_input cannot be None.")

        if not isinstance(model_input, VisionModelInput):
            raise VisionInputValidationError(
                f"Expected VisionModelInput instance, got {type(model_input).__name__}."
            )

        if not self.capabilities.supports_modality(model_input.content_type):
            raise VisionUnsupportedCapabilityError(
                f"Provider '{self.provider_name}' does not support visual modality "
                f"'{model_input.content_type}'. Supported modalities: "
                f"{sorted(self.capabilities.supported_modalities)}."
            )

        if not self.capabilities.supports_format(model_input.image_format):
            raise VisionUnsupportedCapabilityError(
                f"Provider '{self.provider_name}' does not support image format "
                f"'{model_input.image_format}'. Supported formats: "
                f"{sorted(self.capabilities.supported_formats)}."
            )

    @abstractmethod
    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        """Execute visual analysis for the provided VisionModelInput.

        Every concrete provider must implement this method. Concrete providers must
        first call self.validate_input(model_input) before execution.

        Args:
            model_input: Standardized, validated VisionModelInput.
            **kwargs: Additional runtime parameters passed to the provider.

        Returns:
            Standardized VisionResult maintaining lineage and analysis.

        Raises:
            VisionInputValidationError: If input validation fails.
            VisionUnsupportedCapabilityError: If input requires unsupported features.
            VisionProviderExecutionError: If provider execution fails.
            VisionProviderUnavailableError: If provider backend is unreachable.
        """
        raise NotImplementedError("Subclasses of VisionModelProvider must implement execute().")

    def process(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        """Alias for execute method."""
        return self.execute(model_input, **kwargs)

    def __call__(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        """Allow calling provider directly as a callable."""
        return self.execute(model_input, **kwargs)


# ---------------------------------------------------------------------------
# VisionProviderRegistry
# ---------------------------------------------------------------------------


class VisionProviderRegistry:
    """Registry and factory for VisionModelProvider implementations.

    Provides a clean, vendor-agnostic extension point for registering concrete
    providers, looking them up by name, and instantiating them with configuration.
    """

    _registry: dict[str, type[VisionModelProvider]] = {}

    @classmethod
    def register(
        cls,
        provider_name: str,
        provider_cls: type[VisionModelProvider],
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a concrete VisionModelProvider class under a unique name.

        Args:
            provider_name: String identifier for the provider (case-insensitive).
            provider_cls: Concrete subclass of VisionModelProvider.
            overwrite: If True, overwrite an existing registration. If False, raise error.

        Raises:
            VisionProviderConfigError: If provider_name or provider_cls are invalid.
            VisionProviderError: If provider is already registered and overwrite is False.
        """
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise VisionProviderConfigError("provider_name must be a non-empty string.")

        if not isinstance(provider_cls, type) or not issubclass(provider_cls, VisionModelProvider):
            raise VisionProviderConfigError(
                f"provider_cls must be a subclass of VisionModelProvider, "
                f"got {provider_cls!r}."
            )

        if provider_cls is VisionModelProvider:
            raise VisionProviderConfigError(
                "Cannot register abstract base class VisionModelProvider directly."
            )

        name = provider_name.strip().lower()
        if name in cls._registry and not overwrite:
            raise VisionProviderError(
                f"Provider '{name}' is already registered. Set overwrite=True to replace."
            )

        cls._registry[name] = provider_cls

    @classmethod
    def unregister(cls, provider_name: str) -> bool:
        """Unregister a provider by name.

        Args:
            provider_name: String identifier of provider to remove.

        Returns:
            True if provider was present and removed, False otherwise.
        """
        if not isinstance(provider_name, str):
            return False
        name = provider_name.strip().lower()
        if name in cls._registry:
            del cls._registry[name]
            return True
        return False

    @classmethod
    def get(cls, provider_name: str) -> type[VisionModelProvider] | None:
        """Retrieve a registered provider class by name.

        Args:
            provider_name: String identifier of provider.

        Returns:
            Provider class if found, None otherwise.
        """
        if not isinstance(provider_name, str):
            return None
        return cls._registry.get(provider_name.strip().lower())

    @classmethod
    def is_registered(cls, provider_name: str) -> bool:
        """Check whether a provider name is registered."""
        if not isinstance(provider_name, str):
            return False
        return provider_name.strip().lower() in cls._registry

    @classmethod
    def create(
        cls,
        provider_name: str,
        config: VisionProviderConfig | dict[str, Any] | None = None,
        capabilities: VisionProviderCapabilities | None = None,
        **kwargs: Any,
    ) -> VisionModelProvider:
        """Factory method to instantiate a registered provider.

        Args:
            provider_name: String identifier of the provider to instantiate.
            config: Optional VisionProviderConfig or dictionary with configuration keys.
            capabilities: Optional VisionProviderCapabilities descriptor.
            **kwargs: Extra arguments passed to configuration if config is created from scratch.

        Returns:
            An initialized instance of the requested VisionModelProvider.

        Raises:
            VisionProviderUnavailableError: If provider_name is not registered.
            VisionProviderConfigError: If configuration is invalid.
        """
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise VisionProviderConfigError("provider_name must be a non-empty string.")

        name = provider_name.strip().lower()
        provider_cls = cls.get(name)
        if provider_cls is None:
            raise VisionProviderUnavailableError(
                f"Vision provider '{name}' is not registered. "
                f"Available providers: {cls.list_providers()}."
            )

        # Resolve or construct configuration
        resolved_config: VisionProviderConfig
        if config is None:
            model_name = kwargs.pop("model_name", "default-vision-model")
            resolved_config = VisionProviderConfig(
                provider_name=name,
                model_name=model_name,
                **kwargs,
            )
        elif isinstance(config, dict):
            cfg_dict = dict(config)
            cfg_dict.setdefault("provider_name", name)
            resolved_config = VisionProviderConfig.from_dict(cfg_dict)
        elif isinstance(config, VisionProviderConfig):
            resolved_config = config
        else:
            raise VisionProviderConfigError(
                f"config must be a VisionProviderConfig, dict, or None; got {type(config).__name__}."
            )

        return provider_cls(config=resolved_config, capabilities=capabilities)

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return a sorted list of all registered provider names."""
        return sorted(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers (primarily for testing isolation)."""
        cls._registry.clear()
