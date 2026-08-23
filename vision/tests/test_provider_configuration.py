"""
Day 44 — Vision Agent Provider Configuration & Capability Validation Tests.

Comprehensive test suite verifying:
  1.  VisionProviderConfig validation, boundaries, error conditions, and serialization.
  2.  VisionProviderCapabilities validation, modality/format sets, and query methods.
  3.  VisionModelProvider abstract base contract and input capability enforcement.
  4.  Capability-aware pipeline short-circuiting (unsupported modalities/formats fail before provider call).
  5.  VisionProviderRegistry registration, lookup, factory instantiation, and error handling.
  6.  Concurrency safety, thread isolation, and immutability under parallel execution.
  7.  Deterministic, repeatable behavior and strict offline execution guarantee.
"""

from __future__ import annotations

import concurrent.futures
import copy
import dataclasses
import io
import math
from typing import Any

import pytest
from PIL import Image

from vision import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionAgent,
    VisionExecutionAdapter,
    VisionModelInput,
    VisionModelProvider,
    VisionPipeline,
    VisionProviderCapabilities,
    VisionProviderConfig,
    VisionProviderRegistry,
    VisionRequest,
    VisionResult,
    VisualEvidence,
    execute_vision_request,
    run_vision_pipeline,
)
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
from vision.image_preparation import (
    SUPPORTED_IMAGE_FORMATS,
    prepare_image_evidence,
)
from vision.input_builder import build_vision_input
from vision.lifecycle import VisionExecutionStage


# ===========================================================================
# Test Helpers & Test Doubles
# ===========================================================================


def _make_test_image(
    format_name: str = "PNG",
    width: int = 48,
    height: int = 48,
    color: tuple[int, int, int] = (90, 140, 210),
) -> bytes:
    """Generate minimal valid image bytes for tests."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


def _make_valid_evidence(
    doc_id: str = "doc-cfg-001",
    filename: str = "chart_report.png",
    chunk_id: str = "chk-cfg-001",
    content_type: str = "chart",
    image_bytes: bytes | None = None,
) -> VisualEvidence:
    """Create a fully valid VisualEvidence instance."""
    return VisualEvidence(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        content_type=content_type,
        image_bytes=image_bytes or _make_test_image("PNG"),
        page_number=1,
        chunk_index=0,
        metadata={"source": "test_provider_configuration"},
    )


def _make_valid_model_input(
    content_type: str = "chart",
    image_format: str = "png",
    query: str = "Explain the quarterly chart.",
) -> VisionModelInput:
    """Construct a valid VisionModelInput."""
    ev = _make_valid_evidence(content_type=content_type)
    prep = prepare_image_evidence(ev)
    return build_vision_input(query, prep)


class RecordingConfigProvider(VisionModelProvider):
    """Test double that counts invocations and records received model inputs."""

    def __init__(
        self,
        config: VisionProviderConfig,
        capabilities: VisionProviderCapabilities | None = None,
    ) -> None:
        super().__init__(config, capabilities)
        self.invocation_count: int = 0
        self.recorded_inputs: list[VisionModelInput] = []

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        self.invocation_count += 1
        self.recorded_inputs.append(model_input)
        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Analysis by {self.provider_name}:{self.model_name}",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={"provider": self.provider_name, "model": self.model_name},
        )


class StubSubclassProvider(VisionModelProvider):
    """Minimal concrete provider for registry testing."""

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        return VisionResult(
            query=model_input.query,
            status="success",
            description="Stub result",
        )


# ===========================================================================
# 1. Test class: VisionProviderConfig Validation & Contracts
# ===========================================================================


class TestVisionProviderConfigValidation:
    """Tests for VisionProviderConfig instantiation, boundaries, and validation."""

    def test_01_valid_minimal_config(self) -> None:
        """VisionProviderConfig initializes correctly with required parameters and defaults."""
        cfg = VisionProviderConfig(provider_name="mock_provider", model_name="vlm-base")
        assert cfg.provider_name == "mock_provider"
        assert cfg.model_name == "vlm-base"
        assert cfg.timeout == 30.0
        assert cfg.max_tokens is None
        assert cfg.temperature is None
        assert cfg.max_input_images == 1
        assert cfg.extra_params == {}

    def test_02_valid_full_config(self) -> None:
        """VisionProviderConfig initializes with all explicit custom parameters."""
        cfg = VisionProviderConfig(
            provider_name="  openai_custom  ",
            model_name="  gpt-4o-mini  ",
            timeout=45.5,
            max_tokens=4096,
            temperature=0.3,
            max_input_images=5,
            extra_params={"seed": 42, "top_p": 0.9},
        )
        assert cfg.provider_name == "openai_custom"
        assert cfg.model_name == "gpt-4o-mini"
        assert cfg.timeout == 45.5
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 0.3
        assert cfg.max_input_images == 5
        assert cfg.extra_params == {"seed": 42, "top_p": 0.9}

    @pytest.mark.parametrize(
        "bad_provider",
        ["", "   ", "\t\n", None, 123, True, False, 3.14, [], {}, ()],
    )
    def test_03_invalid_provider_name_raises_error(self, bad_provider: Any) -> None:
        """Invalid or non-string provider_name raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="provider_name"):
            VisionProviderConfig(provider_name=bad_provider, model_name="m1")

    @pytest.mark.parametrize(
        "bad_model",
        ["", "   ", "\t\n", None, 123, True, False, 3.14, [], {}, ()],
    )
    def test_04_invalid_model_name_raises_error(self, bad_model: Any) -> None:
        """Invalid or non-string model_name raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="model_name"):
            VisionProviderConfig(provider_name="p1", model_name=bad_model)

    @pytest.mark.parametrize(
        "bad_timeout",
        [0, 0.0, -1, -0.001, -100.0, None, True, False, "30", "10s", float("inf"), float("-inf"), float("nan"), [], {}],
    )
    def test_05_invalid_timeout_raises_error(self, bad_timeout: Any) -> None:
        """Non-positive, non-finite, or non-numeric timeout raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="timeout"):
            VisionProviderConfig(provider_name="p1", model_name="m1", timeout=bad_timeout)

    @pytest.mark.parametrize(
        "bad_tokens",
        [0, -1, -500, 1.5, True, False, "1000", float("inf"), float("nan"), [], {}],
    )
    def test_06_invalid_max_tokens_raises_error(self, bad_tokens: Any) -> None:
        """Non-positive or non-integer max_tokens raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="max_tokens"):
            VisionProviderConfig(provider_name="p1", model_name="m1", max_tokens=bad_tokens)

    @pytest.mark.parametrize(
        "bad_temp",
        [-0.01, -1.0, 2.01, 3.0, 100.0, True, False, "0.7", float("inf"), float("-inf"), float("nan"), []],
    )
    def test_07_invalid_temperature_raises_error(self, bad_temp: Any) -> None:
        """Temperature outside [0.0, 2.0], boolean, or non-finite raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="temperature"):
            VisionProviderConfig(provider_name="p1", model_name="m1", temperature=bad_temp)

    def test_08_temperature_exact_boundaries(self) -> None:
        """Temperature boundary values 0.0 and 2.0 are valid."""
        cfg_zero = VisionProviderConfig(provider_name="p1", model_name="m1", temperature=0.0)
        assert cfg_zero.temperature == 0.0

        cfg_two = VisionProviderConfig(provider_name="p1", model_name="m1", temperature=2.0)
        assert cfg_two.temperature == 2.0

    @pytest.mark.parametrize(
        "bad_images",
        [0, -1, -10, 1.5, True, False, "5", None, float("inf"), []],
    )
    def test_09_invalid_max_input_images_raises_error(self, bad_images: Any) -> None:
        """Non-positive or non-integer max_input_images raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="max_input_images"):
            VisionProviderConfig(provider_name="p1", model_name="m1", max_input_images=bad_images)

    @pytest.mark.parametrize(
        "bad_extra",
        ["not_a_dict", 123, True, [("k", "v")], None],
    )
    def test_10_invalid_extra_params_raises_error(self, bad_extra: Any) -> None:
        """Non-dictionary extra_params raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="extra_params"):
            VisionProviderConfig(provider_name="p1", model_name="m1", extra_params=bad_extra)

    def test_11_extra_params_defensive_copy(self) -> None:
        """Mutating the source dictionary does not mutate the stored extra_params."""
        source = {"key1": "val1"}
        cfg = VisionProviderConfig(provider_name="p1", model_name="m1", extra_params=source)
        source["key2"] = "val2"
        assert "key2" not in cfg.extra_params
        assert cfg.extra_params == {"key1": "val1"}

    def test_12_to_dict_and_from_dict_roundtrip(self) -> None:
        """to_dict and from_dict produce equivalent configuration instances."""
        original = VisionProviderConfig(
            provider_name="anthropic",
            model_name="claude-3-5-sonnet",
            timeout=50.0,
            max_tokens=2048,
            temperature=0.5,
            max_input_images=3,
            extra_params={"system_prefix": "You are a visual analyst."},
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

    @pytest.mark.parametrize(
        "bad_dict",
        [
            "not_dict",
            123,
            None,
            [],
            {"model_name": "m1"},  # missing provider_name
            {"provider_name": "p1"},  # missing model_name
        ],
    )
    def test_13_from_dict_invalid_data_raises_error(self, bad_dict: Any) -> None:
        """from_dict raises VisionProviderConfigError on non-dict or missing required keys."""
        with pytest.raises(VisionProviderConfigError):
            VisionProviderConfig.from_dict(bad_dict)


# ===========================================================================
# 2. Test class: VisionProviderCapabilities Validation & Contracts
# ===========================================================================


class TestVisionProviderCapabilitiesValidation:
    """Tests for VisionProviderCapabilities instantiation, defaults, and capability checking."""

    def test_14_default_capabilities(self) -> None:
        """Default capabilities cover all standard modalities, formats, and single image."""
        caps = VisionProviderCapabilities()
        assert caps.supported_modalities == frozenset(VALID_VISUAL_CONTENT_TYPES)
        assert caps.supported_formats == frozenset(SUPPORTED_IMAGE_FORMATS)
        assert caps.max_images == 1
        assert caps.supports_streaming is False
        assert caps.supports_multi_image is False
        assert caps.supports_system_prompt is True

    def test_15_custom_valid_capabilities(self) -> None:
        """Custom valid capabilities are correctly set and normalized."""
        caps = VisionProviderCapabilities(
            supported_modalities=["CHART", "DIAGRAM"],
            supported_formats=["PNG", "WEBP"],
            max_images=4,
            supports_streaming=True,
            supports_multi_image=True,
            supports_system_prompt=False,
        )
        assert caps.supported_modalities == frozenset({"chart", "diagram"})
        assert caps.supported_formats == frozenset({"png", "webp"})
        assert caps.max_images == 4
        assert caps.supports_streaming is True
        assert caps.supports_multi_image is True
        assert caps.supports_system_prompt is False

    @pytest.mark.parametrize(
        "bad_modalities",
        [
            [],
            set(),
            frozenset(),
            ["audio"],
            ["image", "video_stream"],
            "not_a_collection",
            123,
            None,
        ],
    )
    def test_16_invalid_supported_modalities_raises_error(self, bad_modalities: Any) -> None:
        """Empty, invalid modality, or non-collection supported_modalities raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="supported_modalities"):
            VisionProviderCapabilities(supported_modalities=bad_modalities)

    @pytest.mark.parametrize(
        "bad_formats",
        [
            [],
            set(),
            frozenset(),
            ["gif"],
            ["png", "bmp", "tiff"],
            "not_a_collection",
            123,
            None,
        ],
    )
    def test_17_invalid_supported_formats_raises_error(self, bad_formats: Any) -> None:
        """Empty, invalid format, or non-collection supported_formats raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="supported_formats"):
            VisionProviderCapabilities(supported_formats=bad_formats)

    @pytest.mark.parametrize("bad_max_imgs", [0, -1, -5, 1.5, True, False, "2", None])
    def test_18_invalid_max_images_raises_error(self, bad_max_imgs: Any) -> None:
        """Non-positive, non-integer, or boolean max_images raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="max_images"):
            VisionProviderCapabilities(max_images=bad_max_imgs)

    @pytest.mark.parametrize(
        "flag_name",
        ["supports_streaming", "supports_multi_image", "supports_system_prompt"],
    )
    @pytest.mark.parametrize("bad_flag_val", [1, 0, "true", "false", None, [], {}])
    def test_19_invalid_boolean_flags_raises_error(self, flag_name: str, bad_flag_val: Any) -> None:
        """Non-boolean flag values raise VisionProviderConfigError."""
        kwargs = {flag_name: bad_flag_val}
        with pytest.raises(VisionProviderConfigError, match=flag_name):
            VisionProviderCapabilities(**kwargs)

    def test_20_supports_modality_checks(self) -> None:
        """supports_modality correctly validates case-insensitively and handles invalid input gracefully."""
        caps = VisionProviderCapabilities(supported_modalities=["chart", "diagram"])
        assert caps.supports_modality("chart") is True
        assert caps.supports_modality("CHART") is True
        assert caps.supports_modality(" Diagram ") is True
        assert caps.supports_modality("image") is False
        assert caps.supports_modality("unknown") is False
        assert caps.supports_modality(None) is False  # type: ignore[arg-type]
        assert caps.supports_modality(123) is False  # type: ignore[arg-type]

    def test_21_supports_format_checks(self) -> None:
        """supports_format correctly validates case-insensitively and handles invalid input gracefully."""
        caps = VisionProviderCapabilities(supported_formats=["png", "jpeg"])
        assert caps.supports_format("png") is True
        assert caps.supports_format("PNG") is True
        assert caps.supports_format(" JPEG ") is True
        assert caps.supports_format("webp") is False
        assert caps.supports_format("gif") is False
        assert caps.supports_format(None) is False  # type: ignore[arg-type]
        assert caps.supports_format(456) is False  # type: ignore[arg-type]

    def test_22_can_handle_model_input(self) -> None:
        """can_handle returns True only when both modality and format are supported."""
        caps = VisionProviderCapabilities(
            supported_modalities=["chart"],
            supported_formats=["png"],
        )
        valid_input = _make_valid_model_input(content_type="chart", image_format="png")
        assert caps.can_handle(valid_input) is True

        # Mismatched modality
        diagram_input = _make_valid_model_input(content_type="diagram", image_format="png")
        assert caps.can_handle(diagram_input) is False

        # Non-model-input
        assert caps.can_handle(None) is False  # type: ignore[arg-type]
        assert caps.can_handle("not an input") is False  # type: ignore[arg-type]

    def test_23_capabilities_to_dict_and_from_dict(self) -> None:
        """to_dict and from_dict serialize and reconstruct capabilities perfectly."""
        original = VisionProviderCapabilities(
            supported_modalities=["chart"],
            supported_formats=["png", "webp"],
            max_images=2,
            supports_streaming=True,
            supports_multi_image=True,
            supports_system_prompt=False,
        )
        d = original.to_dict()
        assert isinstance(d["supported_modalities"], list)
        assert isinstance(d["supported_formats"], list)

        reconstructed = VisionProviderCapabilities.from_dict(d)
        assert reconstructed.supported_modalities == original.supported_modalities
        assert reconstructed.supported_formats == original.supported_formats
        assert reconstructed.max_images == original.max_images
        assert reconstructed.supports_streaming == original.supports_streaming
        assert reconstructed.supports_multi_image == original.supports_multi_image
        assert reconstructed.supports_system_prompt == original.supports_system_prompt

    def test_24_capabilities_from_dict_invalid_type_raises_error(self) -> None:
        """from_dict raises VisionProviderConfigError on non-dict input."""
        with pytest.raises(VisionProviderConfigError, match="data must be a dictionary"):
            VisionProviderCapabilities.from_dict("invalid")  # type: ignore[arg-type]


# ===========================================================================
# 3. Test class: VisionModelProvider Abstract Contract & Validation
# ===========================================================================


class TestVisionModelProviderContract:
    """Tests for VisionModelProvider abstract methods, initialization, and validation."""

    def test_25_abstract_provider_cannot_be_instantiated(self) -> None:
        """VisionModelProvider is abstract and raises TypeError upon instantiation."""
        cfg = VisionProviderConfig(provider_name="base", model_name="base-v1")
        with pytest.raises(TypeError):
            VisionModelProvider(cfg)  # type: ignore[abstract]

    def test_26_provider_init_with_invalid_config_raises_error(self) -> None:
        """Passing non-VisionProviderConfig to subclass __init__ raises VisionProviderConfigError."""
        with pytest.raises(VisionProviderConfigError, match="VisionProviderConfig"):
            RecordingConfigProvider(config={"provider_name": "test", "model_name": "m1"})  # type: ignore[arg-type]

    def test_27_provider_init_with_invalid_capabilities_raises_error(self) -> None:
        """Passing non-VisionProviderCapabilities to subclass __init__ raises VisionProviderConfigError."""
        cfg = VisionProviderConfig(provider_name="test", model_name="m1")
        with pytest.raises(VisionProviderConfigError, match="VisionProviderCapabilities"):
            RecordingConfigProvider(config=cfg, capabilities="invalid_caps")  # type: ignore[arg-type]

    def test_28_provider_validate_input_none_or_wrong_type(self) -> None:
        """validate_input raises VisionInputValidationError on None or wrong type."""
        cfg = VisionProviderConfig(provider_name="test", model_name="m1")
        provider = RecordingConfigProvider(cfg)

        with pytest.raises(VisionInputValidationError, match="cannot be None"):
            provider.validate_input(None)  # type: ignore[arg-type]

        with pytest.raises(VisionInputValidationError, match="Expected VisionModelInput"):
            provider.validate_input("string_input")  # type: ignore[arg-type]

    def test_29_provider_validate_input_unsupported_modality_raises_error(self) -> None:
        """validate_input raises VisionUnsupportedCapabilityError for unsupported modality."""
        cfg = VisionProviderConfig(provider_name="chart_only_prov", model_name="chart-v1")
        caps = VisionProviderCapabilities(supported_modalities=["chart"])
        provider = RecordingConfigProvider(cfg, capabilities=caps)

        # Create input with 'diagram' modality
        diag_input = _make_valid_model_input(content_type="diagram")

        with pytest.raises(VisionUnsupportedCapabilityError, match="does not support visual modality 'diagram'"):
            provider.validate_input(diag_input)

    def test_30_provider_validate_input_unsupported_format_raises_error(self) -> None:
        """validate_input raises VisionUnsupportedCapabilityError for unsupported image format."""
        cfg = VisionProviderConfig(provider_name="png_only_prov", model_name="png-v1")
        caps = VisionProviderCapabilities(supported_formats=["png"])
        provider = RecordingConfigProvider(cfg, capabilities=caps)

        # Create input with 'jpeg' format
        jpeg_bytes = _make_test_image("JPEG")
        ev = _make_valid_evidence(content_type="chart", image_bytes=jpeg_bytes, filename="img.jpg")
        prep = prepare_image_evidence(ev)
        jpeg_input = build_vision_input("Explain jpeg chart", prep)

        with pytest.raises(VisionUnsupportedCapabilityError, match="does not support image format 'jpeg'"):
            provider.validate_input(jpeg_input)

    def test_31_provider_process_and_call_aliases(self) -> None:
        """process() and __call__() aliases delegate execution and return VisionResult."""
        cfg = VisionProviderConfig(provider_name="alias_prov", model_name="vlm-1")
        provider = RecordingConfigProvider(cfg)
        model_input = _make_valid_model_input()

        res1 = provider.process(model_input)
        assert isinstance(res1, VisionResult)
        assert res1.status == "success"

        res2 = provider(model_input)
        assert isinstance(res2, VisionResult)
        assert res2.status == "success"
        assert provider.invocation_count == 2


# ===========================================================================
# 4. Test class: Capability-Aware Pipeline Execution & Short-Circuiting
# ===========================================================================


class TestCapabilityAwarePipelineExecution:
    """Tests verifying capability enforcement and pipeline short-circuiting."""

    def test_32_pipeline_short_circuits_on_unsupported_modality(self) -> None:
        """Pipeline fails before provider execution when input modality is unsupported."""
        cfg = VisionProviderConfig(provider_name="chart_expert", model_name="chart-v2")
        caps = VisionProviderCapabilities(supported_modalities=["chart"])
        provider = RecordingConfigProvider(cfg, capabilities=caps)

        pipeline = VisionPipeline(provider=provider)
        diag_evidence = _make_valid_evidence(content_type="diagram", filename="arch.png")

        req = VisionRequest(
            query="Analyze architecture diagram",
            evidence=[diag_evidence],
        )

        with pytest.raises(VisionUnsupportedCapabilityError, match="does not support visual modality 'diagram'"):
            pipeline.run(req)

        # Crucial: Provider execute was never invoked!
        assert provider.invocation_count == 0
        assert len(provider.recorded_inputs) == 0

    def test_33_pipeline_short_circuits_on_unsupported_format(self) -> None:
        """Pipeline fails before provider execution when image format is unsupported."""
        cfg = VisionProviderConfig(provider_name="png_strict", model_name="png-v2")
        caps = VisionProviderCapabilities(supported_formats=["png"])
        provider = RecordingConfigProvider(cfg, capabilities=caps)

        pipeline = VisionPipeline(provider=provider)
        jpeg_bytes = _make_test_image("JPEG")
        jpeg_evidence = _make_valid_evidence(
            content_type="chart",
            filename="chart.jpg",
            image_bytes=jpeg_bytes,
        )

        req = VisionRequest(
            query="Analyze jpeg chart",
            evidence=[jpeg_evidence],
        )

        with pytest.raises(VisionUnsupportedCapabilityError, match="does not support image format 'jpeg'"):
            pipeline.run(req)

        assert provider.invocation_count == 0

    def test_34_vision_agent_capability_aware_execution(self) -> None:
        """VisionAgent correctly executes with capability-compatible evidence."""
        cfg = VisionProviderConfig(provider_name="gemini_vision", model_name="gemini-1.5-flash")
        caps = VisionProviderCapabilities(
            supported_modalities=["chart", "image"],
            supported_formats=["png", "jpeg"],
        )
        provider = RecordingConfigProvider(cfg, capabilities=caps)
        agent = VisionAgent(provider=provider)

        chart_ev = _make_valid_evidence(content_type="chart")
        result = agent.analyze(request="Explain chart data", evidence=[chart_ev])

        assert isinstance(result, VisionResult)
        assert result.status == "success"
        assert provider.invocation_count == 1
        assert provider.recorded_inputs[0].content_type == "chart"

    def test_35_execution_adapter_convenience_function_short_circuit(self) -> None:
        """execute_vision_request convenience function enforces capability validation."""
        cfg = VisionProviderConfig(provider_name="prov_chart", model_name="m1")
        caps = VisionProviderCapabilities(supported_modalities=["chart"])
        provider = RecordingConfigProvider(cfg, capabilities=caps)

        img_ev = _make_valid_evidence(content_type="image")
        with pytest.raises(VisionUnsupportedCapabilityError):
            execute_vision_request(provider=provider, request="Describe image", evidence=[img_ev])

        assert provider.invocation_count == 0


# ===========================================================================
# 5. Test class: VisionProviderRegistry Hardening & Factory
# ===========================================================================


class TestVisionProviderRegistryHardening:
    """Tests for VisionProviderRegistry operations, error handling, and factory creation."""

    def setup_method(self) -> None:
        """Clear registry before each test to guarantee complete test isolation."""
        VisionProviderRegistry.clear()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        VisionProviderRegistry.clear()

    def test_36_register_and_lookup_provider(self) -> None:
        """Registering concrete provider allows lookup and presence check."""
        assert VisionProviderRegistry.list_providers() == []
        VisionProviderRegistry.register("stub_provider", StubSubclassProvider)

        assert VisionProviderRegistry.is_registered("stub_provider") is True
        assert VisionProviderRegistry.is_registered("STUB_PROVIDER") is True  # case-insensitive
        assert VisionProviderRegistry.get("stub_provider") is StubSubclassProvider
        assert VisionProviderRegistry.list_providers() == ["stub_provider"]

    def test_37_duplicate_registration_raises_error(self) -> None:
        """Duplicate registration without overwrite raises VisionProviderError."""
        VisionProviderRegistry.register("mock_p", StubSubclassProvider)
        with pytest.raises(VisionProviderError, match="already registered"):
            VisionProviderRegistry.register("mock_p", StubSubclassProvider, overwrite=False)

    def test_38_duplicate_registration_with_overwrite(self) -> None:
        """Duplicate registration with overwrite=True successfully updates registry."""
        VisionProviderRegistry.register("mock_p", StubSubclassProvider)
        VisionProviderRegistry.register("mock_p", RecordingConfigProvider, overwrite=True)
        assert VisionProviderRegistry.get("mock_p") is RecordingConfigProvider

    def test_39_register_invalid_inputs_raise_config_error(self) -> None:
        """Registering invalid name, non-class, or abstract base class raises VisionProviderConfigError."""
        # Non-string / empty name
        with pytest.raises(VisionProviderConfigError, match="provider_name"):
            VisionProviderRegistry.register("", StubSubclassProvider)
        with pytest.raises(VisionProviderConfigError, match="provider_name"):
            VisionProviderRegistry.register(None, StubSubclassProvider)  # type: ignore[arg-type]

        # Non-subclass
        with pytest.raises(VisionProviderConfigError, match="subclass of VisionModelProvider"):
            VisionProviderRegistry.register("bad_cls", dict)  # type: ignore[arg-type]

        # Abstract base class itself
        with pytest.raises(VisionProviderConfigError, match="Cannot register abstract base class"):
            VisionProviderRegistry.register("abc", VisionModelProvider)

    def test_40_unregister_provider(self) -> None:
        """unregister removes registered provider and returns True; returns False if missing."""
        VisionProviderRegistry.register("removable", StubSubclassProvider)
        assert VisionProviderRegistry.unregister("removable") is True
        assert VisionProviderRegistry.is_registered("removable") is False
        assert VisionProviderRegistry.unregister("removable") is False
        assert VisionProviderRegistry.unregister(123) is False  # type: ignore[arg-type]

    def test_41_factory_create_with_config_instance(self) -> None:
        """create() instantiates registered provider with a VisionProviderConfig instance."""
        VisionProviderRegistry.register("rec_provider", RecordingConfigProvider)
        cfg = VisionProviderConfig(provider_name="rec_provider", model_name="rec-v1", timeout=40.0)
        caps = VisionProviderCapabilities(supported_modalities=["chart"])

        instance = VisionProviderRegistry.create("rec_provider", config=cfg, capabilities=caps)
        assert isinstance(instance, RecordingConfigProvider)
        assert instance.provider_name == "rec_provider"
        assert instance.model_name == "rec-v1"
        assert instance.config.timeout == 40.0
        assert instance.capabilities.supported_modalities == frozenset({"chart"})

    def test_42_factory_create_with_dict_config(self) -> None:
        """create() instantiates registered provider with dictionary configuration."""
        VisionProviderRegistry.register("rec_provider", RecordingConfigProvider)
        dict_cfg = {
            "provider_name": "rec_provider",
            "model_name": "rec-v2",
            "timeout": 25.0,
            "max_tokens": 1024,
        }
        instance = VisionProviderRegistry.create("rec_provider", config=dict_cfg)
        assert isinstance(instance, RecordingConfigProvider)
        assert instance.model_name == "rec-v2"
        assert instance.config.timeout == 25.0
        assert instance.config.max_tokens == 1024

    def test_43_factory_create_with_defaults(self) -> None:
        """create() constructs default configuration when config is None."""
        VisionProviderRegistry.register("stub", StubSubclassProvider)
        instance = VisionProviderRegistry.create("stub", model_name="custom-default")
        assert instance.provider_name == "stub"
        assert instance.model_name == "custom-default"

    def test_44_factory_create_unregistered_raises_unavailable_error(self) -> None:
        """create() raises VisionProviderUnavailableError for unknown provider."""
        with pytest.raises(VisionProviderUnavailableError, match="is not registered"):
            VisionProviderRegistry.create("non_existent_provider")

    def test_45_factory_create_invalid_config_type_raises_error(self) -> None:
        """create() raises VisionProviderConfigError for invalid config parameter type."""
        VisionProviderRegistry.register("stub", StubSubclassProvider)
        with pytest.raises(VisionProviderConfigError, match="must be a VisionProviderConfig, dict, or None"):
            VisionProviderRegistry.create("stub", config=["not", "a", "dict"])  # type: ignore[arg-type]


# ===========================================================================
# 6. Test class: Concurrency Safety & Immutability
# ===========================================================================


class TestProviderConfigurationConcurrencyAndImmutability:
    """Tests verifying thread safety and immutability of configuration & capabilities."""

    def test_46_capabilities_are_frozen(self) -> None:
        """VisionProviderCapabilities dataclass is frozen and cannot be mutated."""
        caps = VisionProviderCapabilities()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            caps.max_images = 10  # type: ignore[misc]

    def test_47_concurrent_capability_checks(self) -> None:
        """Concurrent capability queries across multiple threads are thread-safe and consistent."""
        caps = VisionProviderCapabilities(
            supported_modalities=["chart", "diagram"],
            supported_formats=["png", "jpeg"],
            max_images=2,
        )
        sample_input = _make_valid_model_input(content_type="chart", image_format="png")

        def worker_check(thread_id: int) -> bool:
            m_ok = caps.supports_modality("chart") and not caps.supports_modality("image")
            f_ok = caps.supports_format("png") and not caps.supports_format("webp")
            h_ok = caps.can_handle(sample_input)
            return m_ok and f_ok and h_ok

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker_check, i) for i in range(40)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 40
        assert all(results)

    def test_48_concurrent_pipeline_capability_validation(self) -> None:
        """Multiple concurrent requests through a shared provider validate capabilities independently."""
        cfg = VisionProviderConfig(provider_name="shared_vlm", model_name="vlm-thread-safe")
        caps = VisionProviderCapabilities(supported_modalities=["chart"])
        provider = RecordingConfigProvider(cfg, capabilities=caps)
        pipeline = VisionPipeline(provider=provider)

        chart_ev = _make_valid_evidence(content_type="chart")
        diag_ev = _make_valid_evidence(content_type="diagram")

        def run_valid(idx: int) -> str:
            req = VisionRequest(query=f"Valid query {idx}", evidence=[chart_ev])
            res = pipeline.run(req)
            return res.status

        def run_invalid(idx: int) -> str:
            req = VisionRequest(query=f"Invalid query {idx}", evidence=[diag_ev])
            try:
                pipeline.run(req)
                return "unexpected_success"
            except VisionUnsupportedCapabilityError:
                return "expected_failure"

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            valid_futures = [executor.submit(run_valid, i) for i in range(15)]
            invalid_futures = [executor.submit(run_invalid, i) for i in range(15)]

            valid_results = [f.result() for f in concurrent.futures.as_completed(valid_futures)]
            invalid_results = [f.result() for f in concurrent.futures.as_completed(invalid_futures)]

        assert len(valid_results) == 15
        assert all(r == "success" for r in valid_results)

        assert len(invalid_results) == 15
        assert all(r == "expected_failure" for r in invalid_results)

        # Provider execute was called exactly 15 times (only for valid requests)
        assert provider.invocation_count == 15


# ===========================================================================
# 7. Test class: Offline & Deterministic Guarantees
# ===========================================================================


class TestOfflineAndDeterministicGuarantees:
    """Tests verifying 100% offline operation and deterministic contracts."""

    def test_49_deterministic_validation_outputs(self) -> None:
        """Identical invalid configurations consistently raise identical exception types and messages."""
        for _ in range(10):
            with pytest.raises(VisionProviderConfigError, match="timeout must be a positive finite number"):
                VisionProviderConfig(provider_name="p", model_name="m", timeout=-5.0)

            with pytest.raises(VisionProviderConfigError, match="supported_modalities cannot be empty"):
                VisionProviderCapabilities(supported_modalities=[])

    def test_50_zero_network_imports_and_clean_repr(self) -> None:
        """ProviderConfig and Capabilities have clean representations without hidden networking."""
        cfg = VisionProviderConfig(provider_name="offline_prov", model_name="local-vlm")
        caps = VisionProviderCapabilities()

        assert "offline_prov" in repr(cfg)
        assert "local-vlm" in repr(cfg)
        assert "VisionProviderCapabilities" in repr(caps)
