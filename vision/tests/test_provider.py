"""
Comprehensive unit and integration tests for Day 36: Vision Model Provider Interface & Abstraction.

Tests cover:
  1.  Provider interface exists and cannot be directly instantiated (abstract).
  2.  Valid VisionProviderConfig creation and property inspection.
  3.  Missing or empty provider_name in config raises VisionProviderConfigError.
  4.  Missing or empty model_name in config raises VisionProviderConfigError.
  5.  Invalid timeout (<= 0, boolean, non-numeric, inf, nan) raises VisionProviderConfigError.
  6.  Invalid max_tokens (<= 0, boolean, non-int) raises VisionProviderConfigError.
  7.  Invalid temperature (< 0, > 2, boolean, non-numeric) raises VisionProviderConfigError.
  8.  Invalid max_input_images (<= 0, boolean, non-int) raises VisionProviderConfigError.
  9.  Invalid extra_params (non-dict) raises VisionProviderConfigError.
  10. Config serialization to_dict() and deserialization from_dict().
  11. VisionProviderCapabilities creation, defaults, and validation.
  12. Capabilities modality support check (supports_modality).
  13. Capabilities format support check (supports_format).
  14. Capabilities can_handle check on VisionModelInput.
  15. Capabilities serialization to_dict() and deserialization from_dict().
  16. Concrete provider subclassing and instantiation with custom capabilities.
  17. validate_input accepts valid VisionModelInput.
  18. validate_input rejects None and non-VisionModelInput types.
  19. validate_input raises VisionUnsupportedCapabilityError for unsupported modality.
  20. validate_input raises VisionUnsupportedCapabilityError for unsupported image format.
  21. Provider execute(), process(), and __call__() aliases execute correctly.
  22. Provider result contract conforms to VisionResult.
  23. Provider failure produces controlled VisionProviderExecutionError.
  24. Abstract execute() raises NotImplementedError if called directly.
  25. Provider registry register(), get(), is_registered(), and list_providers().
  26. Provider registry rejects registering abstract VisionModelProvider or non-subclass.
  27. Provider registry duplicate registration raises VisionProviderError unless overwrite=True.
  28. Provider registry unregister() removes provider.
  29. Provider registry create() instantiates registered provider with config object or dict.
  30. Provider registry create() for unknown provider raises VisionProviderUnavailableError.
  31. VisionAgent integration with concrete VisionModelProvider instance.
  32. VisionAgent analyze() with provider returns structured VisionResult.
  33. VisionAgent analyze() with provider and no evidence returns no_evidence status.
  34. VisionAgent without provider retains Day 32 controlled VisionProcessingError.
  35. VisionAgent rejects invalid provider types during initialization.
  36. No network calls or LLM client dependencies in provider code.
  37. No secrets or API keys required or leaked in config/capabilities.
  38. Deterministic input validation and repeatable behavior.
"""

from __future__ import annotations

import inspect
import io
from typing import Any

import pytest
from PIL import Image

from vision.exceptions import (
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderConfigError,
    VisionProviderError,
    VisionProviderExecutionError,
    VisionProviderUnavailableError,
    VisionUnsupportedCapabilityError,
)
from vision.image_preparation import prepare_image_evidence
from vision.input_builder import VisionModelInput, build_vision_input
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.provider import (
    VisionModelProvider,
    VisionProviderRegistry,
)
from vision.provider_config import (
    VisionProviderCapabilities,
    VisionProviderConfig,
)
from vision.vision_agent import VisionAgent


# ===========================================================================
# Test Helpers & Test Doubles (Conformed strictly to test scope only)
# ===========================================================================


def _make_test_png(width: int = 64, height: int = 64) -> bytes:
    """Generate minimal PNG bytes for test evidence."""
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_test_model_input(
    content_type: str = "image",
    image_format: str = "png",
    width: int = 64,
    height: int = 64,
    query: str = "Describe the diagram.",
) -> VisionModelInput:
    """Construct a valid VisionModelInput via Day 34 & Day 35 pipeline."""
    ev = VisualEvidence(
        document_id="doc-test-001",
        filename="test_doc.pdf",
        chunk_id="chunk-test-001",
        page_number=1,
        chunk_index=0,
        content_type=content_type,
        image_bytes=_make_test_png(width, height),
        metadata={"origin": "test-suite"},
    )
    prepared = prepare_image_evidence(ev)
    return build_vision_input(query, prepared)


class MinimalTestProvider(VisionModelProvider):
    """Minimal concrete test double to verify provider interface execution."""

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        return VisionResult(
            query=model_input.query,
            status="success",
            description="Verified visual analysis result from test provider.",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={"provider": self.provider_name, "model": self.model_name},
        )


class FailingTestProvider(VisionModelProvider):
    """Concrete test double that simulates a controlled provider execution failure."""

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        raise VisionProviderExecutionError("Simulated provider backend execution failure.")


# ===========================================================================
# 1. Test class: VisionModelProvider Abstract Interface
# ===========================================================================


class TestVisionModelProviderInterface:
    """Tests 1: VisionModelProvider cannot be instantiated directly."""

    def test_01_abstract_provider_cannot_be_instantiated(self) -> None:
        """VisionModelProvider is an abstract class and raises TypeError on direct instantiation."""
        config = VisionProviderConfig(provider_name="abstract-test", model_name="test-model")
        with pytest.raises(TypeError):
            VisionModelProvider(config)  # type: ignore[abstract]

    def test_02_concrete_provider_initialization(self) -> None:
        """Concrete provider subclass initializes with configuration and default capabilities."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-v1", timeout=45.0)
        provider = MinimalTestProvider(config)

        assert provider.config is config
        assert provider.provider_name == "test-prov"
        assert provider.model_name == "vlm-v1"
        assert isinstance(provider.capabilities, VisionProviderCapabilities)

    def test_03_concrete_provider_with_custom_capabilities(self) -> None:
        """Concrete provider accepts custom VisionProviderCapabilities."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-v1")
        caps = VisionProviderCapabilities(
            supported_modalities=frozenset({"chart"}),
            supported_formats=frozenset({"png"}),
            max_images=2,
            supports_streaming=True,
        )
        provider = MinimalTestProvider(config, capabilities=caps)

        assert provider.capabilities is caps
        assert provider.capabilities.supports_modality("chart") is True
        assert provider.capabilities.supports_modality("diagram") is False
        assert provider.capabilities.supports_streaming is True

    def test_04_provider_init_invalid_config_raises_error(self) -> None:
        """Passing non-VisionProviderConfig to provider init raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="VisionProviderConfig"):
            MinimalTestProvider(config={"provider_name": "test"})  # type: ignore[arg-type]

    def test_05_provider_init_invalid_capabilities_raises_error(self) -> None:
        """Passing invalid capabilities type raises VisionProviderConfigError."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-v1")
        with pytest.raises(VisionProviderConfigError, match="VisionProviderCapabilities"):
            MinimalTestProvider(config, capabilities="invalid")  # type: ignore[arg-type]


# ===========================================================================
# 2. Test class: VisionProviderConfig Validation & Serialization
# ===========================================================================


class TestVisionProviderConfig:
    """Tests 2-10: Configuration validation, error conditions, and serialization."""

    def test_06_valid_config_creation(self) -> None:
        """Valid configuration parameters are correctly set and typed."""
        config = VisionProviderConfig(
            provider_name="gemini",
            model_name="gemini-1.5-pro",
            timeout=60.0,
            max_tokens=2048,
            temperature=0.7,
            max_input_images=4,
            extra_params={"top_p": 0.95},
        )
        assert config.provider_name == "gemini"
        assert config.model_name == "gemini-1.5-pro"
        assert config.timeout == 60.0
        assert config.max_tokens == 2048
        assert config.temperature == 0.7
        assert config.max_input_images == 4
        assert config.extra_params == {"top_p": 0.95}

    @pytest.mark.parametrize("bad_name", ["", "   ", None, 123, []])
    def test_07_invalid_provider_name_raises_config_error(self, bad_name: Any) -> None:
        """Empty or non-string provider_name raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="provider_name"):
            VisionProviderConfig(provider_name=bad_name, model_name="m1")

    @pytest.mark.parametrize("bad_model", ["", "   ", None, 123, []])
    def test_08_invalid_model_name_raises_config_error(self, bad_model: Any) -> None:
        """Empty or non-string model_name raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="model_name"):
            VisionProviderConfig(provider_name="p1", model_name=bad_model)

    @pytest.mark.parametrize("bad_timeout", [0, -5.0, None, "30", True, float("inf"), float("nan")])
    def test_09_invalid_timeout_raises_config_error(self, bad_timeout: Any) -> None:
        """Non-positive, non-numeric, or non-finite timeout raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="timeout"):
            VisionProviderConfig(provider_name="p1", model_name="m1", timeout=bad_timeout)

    @pytest.mark.parametrize("bad_tokens", [0, -100, 1.5, "2048", True, []])
    def test_10_invalid_max_tokens_raises_config_error(self, bad_tokens: Any) -> None:
        """Non-positive or non-integer max_tokens raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="max_tokens"):
            VisionProviderConfig(provider_name="p1", model_name="m1", max_tokens=bad_tokens)

    @pytest.mark.parametrize("bad_temp", [-0.1, 2.1, 10.0, "0.5", True, float("inf"), float("nan")])
    def test_11_invalid_temperature_raises_config_error(self, bad_temp: Any) -> None:
        """Temperature outside 0.0-2.0 or non-numeric raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="temperature"):
            VisionProviderConfig(provider_name="p1", model_name="m1", temperature=bad_temp)

    @pytest.mark.parametrize("bad_images", [0, -1, 2.5, "1", True, []])
    def test_12_invalid_max_input_images_raises_config_error(self, bad_images: Any) -> None:
        """Non-positive or non-integer max_input_images raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="max_input_images"):
            VisionProviderConfig(provider_name="p1", model_name="m1", max_input_images=bad_images)

    @pytest.mark.parametrize("bad_extra", ["not-a-dict", 123, None, []])
    def test_13_invalid_extra_params_raises_config_error(self, bad_extra: Any) -> None:
        """Non-dict extra_params raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="extra_params"):
            VisionProviderConfig(provider_name="p1", model_name="m1", extra_params=bad_extra)

    def test_14_config_to_dict_and_from_dict_roundtrip(self) -> None:
        """to_dict() and from_dict() maintain full fidelity across serialization."""
        original = VisionProviderConfig(
            provider_name="claude",
            model_name="claude-3-5-sonnet",
            timeout=45.0,
            max_tokens=4096,
            temperature=0.2,
            max_input_images=2,
            extra_params={"system": "You are a chart analyzer."},
        )
        data = original.to_dict()
        reconstructed = VisionProviderConfig.from_dict(data)

        assert reconstructed.provider_name == original.provider_name
        assert reconstructed.model_name == original.model_name
        assert reconstructed.timeout == original.timeout
        assert reconstructed.max_tokens == original.max_tokens
        assert reconstructed.temperature == original.temperature
        assert reconstructed.max_input_images == original.max_input_images
        assert reconstructed.extra_params == original.extra_params

    def test_15_from_dict_invalid_inputs(self) -> None:
        """from_dict() raises VisionProviderConfigError on non-dict or missing required keys."""
        with pytest.raises(VisionProviderConfigError, match="dictionary"):
            VisionProviderConfig.from_dict("not-a-dict")  # type: ignore[arg-type]

        with pytest.raises(VisionProviderConfigError, match="provider_name"):
            VisionProviderConfig.from_dict({"model_name": "m1"})

        with pytest.raises(VisionProviderConfigError, match="model_name"):
            VisionProviderConfig.from_dict({"provider_name": "p1"})


# ===========================================================================
# 3. Test class: VisionProviderCapabilities Validation & Methods
# ===========================================================================


class TestVisionProviderCapabilities:
    """Tests 11-15: Capabilities declaration, modality and format checks, and serialization."""

    def test_16_default_capabilities(self) -> None:
        """Default capabilities cover all core modalities and formats."""
        caps = VisionProviderCapabilities()
        assert caps.supports_modality("image") is True
        assert caps.supports_modality("chart") is True
        assert caps.supports_modality("diagram") is True
        assert caps.supports_modality("text") is False
        assert caps.supports_format("png") is True
        assert caps.supports_format("jpeg") is True
        assert caps.supports_format("webp") is True
        assert caps.supports_format("gif") is False
        assert caps.max_images == 1
        assert caps.supports_streaming is False
        assert caps.supports_multi_image is False
        assert caps.supports_system_prompt is True

    def test_17_custom_capabilities_modality_checks(self) -> None:
        """supports_modality handles casing and non-string inputs safely."""
        caps = VisionProviderCapabilities(supported_modalities=frozenset({"chart"}))
        assert caps.supports_modality("CHART") is True
        assert caps.supports_modality(" chart ") is True
        assert caps.supports_modality("image") is False
        assert caps.supports_modality(123) is False  # type: ignore[arg-type]

    def test_18_custom_capabilities_format_checks(self) -> None:
        """supports_format handles casing and non-string inputs safely."""
        caps = VisionProviderCapabilities(supported_formats=frozenset({"png"}))
        assert caps.supports_format("PNG") is True
        assert caps.supports_format("png") is True
        assert caps.supports_format("jpeg") is False
        assert caps.supports_format(None) is False  # type: ignore[arg-type]

    def test_19_can_handle_model_input(self) -> None:
        """can_handle returns True only when both modality and format are supported."""
        caps = VisionProviderCapabilities(
            supported_modalities=frozenset({"image"}),
            supported_formats=frozenset({"png"}),
        )
        valid_input = _make_test_model_input(content_type="image", image_format="png")
        assert caps.can_handle(valid_input) is True

        chart_input = _make_test_model_input(content_type="chart", image_format="png")
        assert caps.can_handle(chart_input) is False

        assert caps.can_handle("not-a-model-input") is False  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_modality", [set(), {"unknown_modality"}, "image"])
    def test_20_invalid_supported_modalities_raises_error(self, bad_modality: Any) -> None:
        """Empty or unknown modalities raise VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="modalit"):
            VisionProviderCapabilities(supported_modalities=bad_modality)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_format", [set(), {"tiff"}, "png"])
    def test_21_invalid_supported_formats_raises_error(self, bad_format: Any) -> None:
        """Empty or unsupported formats raise VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="format"):
            VisionProviderCapabilities(supported_formats=bad_format)  # type: ignore[arg-type]

    def test_22_capabilities_to_dict_and_from_dict(self) -> None:
        """to_dict() and from_dict() serialize and deserialize correctly."""
        original = VisionProviderCapabilities(
            supported_modalities=frozenset({"chart", "diagram"}),
            supported_formats=frozenset({"png", "webp"}),
            max_images=3,
            supports_streaming=True,
            supports_multi_image=True,
            supports_system_prompt=False,
        )
        data = original.to_dict()
        reconstructed = VisionProviderCapabilities.from_dict(data)

        assert reconstructed.supported_modalities == original.supported_modalities
        assert reconstructed.supported_formats == original.supported_formats
        assert reconstructed.max_images == original.max_images
        assert reconstructed.supports_streaming == original.supports_streaming
        assert reconstructed.supports_multi_image == original.supports_multi_image
        assert reconstructed.supports_system_prompt == original.supports_system_prompt


# ===========================================================================
# 4. Test class: Input Validation & Capability Enforcement
# ===========================================================================


class TestProviderInputValidation:
    """Tests 17-20: validate_input enforcement on VisionModelInput."""

    def test_23_validate_input_accepts_valid_input(self) -> None:
        """validate_input succeeds without error for supported VisionModelInput."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="v1")
        provider = MinimalTestProvider(config)
        model_input = _make_test_model_input(content_type="image", image_format="png")

        # Must not raise
        provider.validate_input(model_input)

    @pytest.mark.parametrize("bad_input", [None, "string", {"query": "q"}, 123])
    def test_24_validate_input_rejects_invalid_types(self, bad_input: Any) -> None:
        """Passing None or non-VisionModelInput raises VisionInputValidationError."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="v1")
        provider = MinimalTestProvider(config)

        with pytest.raises(VisionInputValidationError):
            provider.validate_input(bad_input)

    def test_25_validate_input_unsupported_modality_raises_error(self) -> None:
        """validate_input raises VisionUnsupportedCapabilityError when modality is not supported."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="v1")
        # Provider only supports charts
        caps = VisionProviderCapabilities(supported_modalities=frozenset({"chart"}))
        provider = MinimalTestProvider(config, capabilities=caps)

        diagram_input = _make_test_model_input(content_type="diagram", image_format="png")
        with pytest.raises(VisionUnsupportedCapabilityError, match="modality"):
            provider.validate_input(diagram_input)

    def test_26_validate_input_unsupported_format_raises_error(self) -> None:
        """validate_input raises VisionUnsupportedCapabilityError when format is not supported."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="v1")
        # Provider only supports WEBP
        caps = VisionProviderCapabilities(supported_formats=frozenset({"webp"}))
        provider = MinimalTestProvider(config, capabilities=caps)

        png_input = _make_test_model_input(content_type="image", image_format="png")
        with pytest.raises(VisionUnsupportedCapabilityError, match="format"):
            provider.validate_input(png_input)


# ===========================================================================
# 5. Test class: Provider Execution & Result Contract
# ===========================================================================


class TestProviderExecutionAndResultContract:
    """Tests 21-24: Execution method aliases, result contract, and error propagation."""

    def test_27_provider_execution_returns_structured_vision_result(self) -> None:
        """execute() returns a valid VisionResult preserving query and lineage."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-v1")
        provider = MinimalTestProvider(config)
        model_input = _make_test_model_input(query="Analyze the growth curve.")

        result = provider.execute(model_input)

        assert isinstance(result, VisionResult)
        assert result.query == "Analyze the growth curve."
        assert result.is_success is True
        assert result.document_id == model_input.document_id
        assert result.filename == model_input.filename
        assert result.page_number == model_input.page_number
        assert result.chunk_id == model_input.chunk_id
        assert result.content_type == model_input.content_type
        assert result.metadata.get("provider") == "test-prov"

    def test_28_process_and_call_aliases(self) -> None:
        """process() and __call__() route directly to execute()."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-v1")
        provider = MinimalTestProvider(config)
        model_input = _make_test_model_input()

        res1 = provider.process(model_input)
        res2 = provider(model_input)

        assert isinstance(res1, VisionResult)
        assert isinstance(res2, VisionResult)
        assert res1.query == res2.query

    def test_29_provider_execution_error_propagates(self) -> None:
        """Failing provider raises controlled VisionProviderExecutionError."""
        config = VisionProviderConfig(provider_name="fail-prov", model_name="v1")
        provider = FailingTestProvider(config)
        model_input = _make_test_model_input()

        with pytest.raises(VisionProviderExecutionError, match="Simulated provider"):
            provider.execute(model_input)

    def test_30_abstract_execute_raises_not_implemented(self) -> None:
        """Calling base VisionModelProvider.execute directly raises NotImplementedError."""
        config = VisionProviderConfig(provider_name="test", model_name="m1")

        # Create dummy class that super() calls abstract method
        class SuperCallingProvider(VisionModelProvider):
            def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
                return super().execute(model_input, **kwargs)

        provider = SuperCallingProvider(config)
        model_input = _make_test_model_input()
        with pytest.raises(NotImplementedError):
            provider.execute(model_input)


# ===========================================================================
# 6. Test class: VisionProviderRegistry
# ===========================================================================


class TestVisionProviderRegistry:
    """Tests 25-30: Registration, lookup, factory instantiation, and error handling."""

    def setup_method(self) -> None:
        """Ensure clean registry before each test."""
        VisionProviderRegistry.clear()

    def teardown_method(self) -> None:
        """Clean registry after each test."""
        VisionProviderRegistry.clear()

    def test_31_register_and_get_provider(self) -> None:
        """Providers can be registered and retrieved by name."""
        VisionProviderRegistry.register("mock_provider", MinimalTestProvider)

        assert VisionProviderRegistry.is_registered("mock_provider") is True
        assert VisionProviderRegistry.is_registered("MOCK_PROVIDER") is True  # Case insensitive
        assert VisionProviderRegistry.get("mock_provider") is MinimalTestProvider
        assert "mock_provider" in VisionProviderRegistry.list_providers()

    def test_32_register_invalid_name_or_class_raises_error(self) -> None:
        """Invalid name or non-subclass raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="provider_name"):
            VisionProviderRegistry.register("", MinimalTestProvider)

        with pytest.raises(VisionProviderConfigError, match="subclass"):
            VisionProviderRegistry.register("bad_cls", object)  # type: ignore[arg-type]

        with pytest.raises(VisionProviderConfigError, match="abstract base class"):
            VisionProviderRegistry.register("base_cls", VisionModelProvider)

    def test_33_duplicate_registration_raises_error_unless_overwrite(self) -> None:
        """Duplicate registration raises VisionProviderError unless overwrite=True."""
        VisionProviderRegistry.register("test_prov", MinimalTestProvider)

        with pytest.raises(VisionProviderError, match="already registered"):
            VisionProviderRegistry.register("test_prov", MinimalTestProvider)

        # overwrite=True succeeds
        VisionProviderRegistry.register("test_prov", MinimalTestProvider, overwrite=True)
        assert VisionProviderRegistry.get("test_prov") is MinimalTestProvider

    def test_34_unregister_provider(self) -> None:
        """unregister removes a registered provider."""
        VisionProviderRegistry.register("to_remove", MinimalTestProvider)
        assert VisionProviderRegistry.unregister("to_remove") is True
        assert VisionProviderRegistry.is_registered("to_remove") is False
        assert VisionProviderRegistry.unregister("non_existent") is False

    def test_35_create_provider_with_config_object(self) -> None:
        """create() instantiates registered provider with VisionProviderConfig instance."""
        VisionProviderRegistry.register("minimal", MinimalTestProvider)
        config = VisionProviderConfig(provider_name="minimal", model_name="test-vlm")
        instance = VisionProviderRegistry.create("minimal", config=config)

        assert isinstance(instance, MinimalTestProvider)
        assert instance.provider_name == "minimal"
        assert instance.model_name == "test-vlm"

    def test_36_create_provider_with_dict_config(self) -> None:
        """create() instantiates registered provider with a dictionary config."""
        VisionProviderRegistry.register("minimal", MinimalTestProvider)
        instance = VisionProviderRegistry.create(
            "minimal",
            config={"model_name": "gemini-flash", "timeout": 20.0},
        )
        assert isinstance(instance, MinimalTestProvider)
        assert instance.model_name == "gemini-flash"
        assert instance.config.timeout == 20.0

    def test_37_create_provider_with_default_kwargs(self) -> None:
        """create() instantiates registered provider with default kwargs."""
        VisionProviderRegistry.register("minimal", MinimalTestProvider)
        instance = VisionProviderRegistry.create("minimal", model_name="auto-model")
        assert isinstance(instance, MinimalTestProvider)
        assert instance.model_name == "auto-model"

    def test_38_create_unregistered_provider_raises_unavailable_error(self) -> None:
        """create() for an unregistered provider raises VisionProviderUnavailableError."""
        with pytest.raises(VisionProviderUnavailableError, match="not registered"):
            VisionProviderRegistry.create("unregistered_provider")


# ===========================================================================
# 7. Test class: VisionAgent Integration with Provider
# ===========================================================================


class TestVisionAgentProviderIntegration:
    """Tests 31-35: VisionAgent execution when configured with a VisionModelProvider."""

    def test_39_vision_agent_with_provider_executes_analysis(self) -> None:
        """VisionAgent initialized with a provider successfully executes visual analysis."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-v1")
        provider = MinimalTestProvider(config)
        agent = VisionAgent(agent_name="IntegratedAgent", provider=provider)

        ev = VisualEvidence(
            document_id="doc-agent-01",
            filename="diagram.pdf",
            chunk_id="chunk-agent-01",
            page_number=2,
            content_type="diagram",
            image_bytes=_make_test_png(64, 64),
        )
        result = agent.analyze("Explain this diagram", evidence=[ev])

        assert isinstance(result, VisionResult)
        assert result.is_success is True
        assert result.query == "Explain this diagram"
        assert result.document_id == "doc-agent-01"
        assert result.filename == "diagram.pdf"
        assert result.content_type == "diagram"

    def test_40_vision_agent_with_provider_no_evidence_returns_no_evidence_status(self) -> None:
        """VisionAgent with provider returns status='no_evidence' when no visual evidence is supplied."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-v1")
        provider = MinimalTestProvider(config)
        agent = VisionAgent(provider=provider)

        result = agent.analyze("Query with no images", evidence=[])

        assert isinstance(result, VisionResult)
        assert result.status == "no_evidence"
        assert result.has_evidence is False

    def test_41_vision_agent_without_provider_retains_day32_error(self) -> None:
        """VisionAgent without a provider continues to raise controlled VisionProcessingError."""
        agent = VisionAgent(model_name="unconnected-model")
        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        with pytest.raises(VisionProcessingError, match="not implemented for Day 32 foundation"):
            agent.analyze("Describe", evidence=[ev])

    def test_42_vision_agent_invalid_provider_type_raises_error(self) -> None:
        """Passing an invalid provider type raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="VisionModelProvider"):
            VisionAgent(provider="invalid-provider")  # type: ignore[arg-type]


# ===========================================================================
# 8. Test class: Security, Zero-Network, and Offline Validation
# ===========================================================================


class TestProviderSecurityAndOfflineGuarantees:
    """Tests 36-38: Provider abstraction has no network dependencies, API keys, or secrets."""

    def test_43_no_network_imports_in_provider_modules(self) -> None:
        """Neither provider.py nor provider_config.py imports network or HTTP libraries."""
        from vision import provider, provider_config

        forbidden = [
            "import requests",
            "import httpx",
            "import aiohttp",
            "import urllib.request",
            "import socket",
            "http.client",
        ]

        for mod in (provider, provider_config):
            source = inspect.getsource(mod)
            for pattern in forbidden:
                assert pattern not in source, f"{mod.__name__} contains forbidden pattern '{pattern}'"

    def test_44_no_llm_client_imports_in_provider_modules(self) -> None:
        """Provider modules contain no vendor LLM client library imports."""
        from vision import provider, provider_config

        forbidden_imports = [
            "import openai",
            "from openai",
            "import anthropic",
            "from anthropic",
            "import google.generativeai",
            "from google.generativeai",
            "import transformers",
            "from transformers",
            "import langchain",
            "from langchain",
            "import langgraph",
            "from langgraph",
            "import fastapi",
            "from fastapi",
            "import streamlit",
            "from streamlit",
        ]

        for mod in (provider, provider_config):
            source = inspect.getsource(mod)
            for pattern in forbidden_imports:
                assert pattern not in source, f"{mod.__name__} contains forbidden import '{pattern}'"

    def test_45_no_secrets_in_config_or_capabilities_dicts(self) -> None:
        """to_dict() outputs for Config and Capabilities contain no credential fields."""
        config = VisionProviderConfig(provider_name="p1", model_name="m1")
        caps = VisionProviderCapabilities()

        config_keys = set(config.to_dict().keys())
        caps_keys = set(caps.to_dict().keys())

        forbidden_keys = {"api_key", "secret", "password", "token", "auth"}
        assert config_keys.isdisjoint(forbidden_keys)
        assert caps_keys.isdisjoint(forbidden_keys)

    def test_46_deterministic_validation_consistency(self) -> None:
        """Repeated validations of identical inputs yield identical results."""
        config = VisionProviderConfig(provider_name="test-prov", model_name="vlm-v1")
        provider = MinimalTestProvider(config)
        model_input = _make_test_model_input()

        for _ in range(10):
            # Must consistently succeed without stateful side-effects
            provider.validate_input(model_input)
