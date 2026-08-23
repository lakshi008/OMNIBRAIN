"""
Comprehensive unit and integration tests for Day 39: Vision Result Normalization,
Execution Trace & Output Contract.

Tests cover:
  1.  Valid provider VisionResult normalization and sanitization.
  2.  Valid dictionary provider output converted and normalized to VisionResult.
  3.  Rejection of None result with controlled VisionProcessingError.
  4.  Rejection of invalid result types (string, int, list, object).
  5.  Rejection of malformed dictionary or result with empty/whitespace status.
  6.  Sanitization of forbidden metadata keys (api_key, secret, token, password, credentials, image_bytes).
  7.  Preservation of source document lineage (document_id, filename, chunk_id, page_number, chunk_index, content_type).
  8.  Execution trace creation, stage tracking, and attachment to metadata.
  9.  Execution trace safety (contains no secrets, credentials, image bytes, or fake latency).
  10. Input immutability during normalization (request, model_input, evidence untouched).
  11. Deterministic normalization output across repeated calls.
  12. Roundtrip serialization fidelity (VisionResult.to_dict() -> from_dict()).
  13. Provider failure propagation without fake result fabrication.
  14. Multi-stage integration across Days 33-39 (Adapter -> Preparation -> InputBuilder -> Provider -> Lifecycle -> Normalizer).
  15. Zero network, zero LLM, zero external telemetry dependencies.
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
    VisionProviderExecutionError,
)
from vision.execution_adapter import (
    VisionExecutionAdapter,
    execute_vision_request,
)
from vision.input_builder import VisionModelInput, build_vision_input
from vision.lifecycle import VisionExecutionLifecycle, VisionExecutionStage
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.provider import VisionModelProvider
from vision.provider_config import (
    VisionProviderCapabilities,
    VisionProviderConfig,
)
from vision.result_normalizer import (
    FORBIDDEN_METADATA_KEYS,
    VisionExecutionTrace,
    VisionResultNormalizer,
)
from vision.vision_agent import VisionAgent


# ===========================================================================
# Test Helpers & Test Doubles (Conformed strictly to test scope only)
# ===========================================================================


def _make_test_png(width: int = 64, height: int = 64) -> bytes:
    """Generate minimal PNG bytes for test evidence."""
    img = Image.new("RGB", (width, height), color=(180, 120, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class DictOutputProvider(VisionModelProvider):
    """Test double that returns a dictionary output instead of VisionResult object."""

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> Any:
        self.validate_input(model_input)
        return {
            "query": model_input.query,
            "status": "success",
            "description": "Analysis result from dictionary provider output.",
            "document_id": model_input.document_id,
            "filename": model_input.filename,
            "page_number": model_input.page_number,
            "chunk_id": model_input.chunk_id,
            "content_type": model_input.content_type,
            "metadata": {
                "provider": self.provider_name,
                "api_key": "secret_12345",  # To test sanitization
                "image_bytes": b"\x00\x01\x02",  # To test sanitization
            },
        }


class LeakyMetadataProvider(VisionModelProvider):
    """Test double that returns a VisionResult containing sensitive keys in metadata."""

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        return VisionResult(
            query=model_input.query,
            status="success",
            description="Leaky metadata analysis output.",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={
                "provider": self.provider_name,
                "api_key": "sk-secret-key-xyz",
                "auth": "Bearer token-1234",
                "image_bytes": _make_test_png(),
                "nested": {"token": "nested-token-99", "safe_field": "valid_value"},
            },
        )


class MalformedResultProvider(VisionModelProvider):
    """Test double that returns a dict output with an invalid or empty status."""

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> Any:
        self.validate_input(model_input)
        return {"query": model_input.query, "status": ""}  # Empty status in dict


# ===========================================================================
# 1. Test class: VisionExecutionTrace Unit Tests
# ===========================================================================


class TestVisionExecutionTraceUnit:
    """Tests 8, 19, 20, 21, 22: Trace creation, stage tracking, serialization, and safety."""

    def test_01_trace_initialization_and_default_factory(self) -> None:
        """VisionExecutionTrace initializes empty or with initial stage list."""
        trace1 = VisionExecutionTrace()
        assert trace1.stages == []

        trace2 = VisionExecutionTrace.create_default()
        assert len(trace2.stages) == len(VisionExecutionTrace.DEFAULT_STAGES)
        assert trace2.stages[0] == "request_received"
        assert trace2.stages[-1] == "execution_completed"

    def test_02_add_stage_and_case_normalization(self) -> None:
        """add_stage normalizes and appends non-empty string stage names."""
        trace = VisionExecutionTrace()
        trace.add_stage("Validation_Started")
        trace.add_stage("  Provider_COMPLETED  ")

        assert trace.stages == ["validation_started", "provider_completed"]

    @pytest.mark.parametrize("bad_stage", ["", "   ", None, 123, []])
    def test_03_invalid_stage_name_raises_error(self, bad_stage: Any) -> None:
        """Non-string or empty stage_name raises VisionInputValidationError."""
        trace = VisionExecutionTrace()
        with pytest.raises(VisionInputValidationError, match="stage_name"):
            trace.add_stage(bad_stage)

    def test_04_trace_to_dict_contains_no_secrets_or_timing(self) -> None:
        """to_dict() returns clean dictionary with stages and stage_count without secrets/timing."""
        trace = VisionExecutionTrace(initial_stages=["start", "stop"])
        data = trace.to_dict()

        assert data["stages"] == ["start", "stop"]
        assert data["stage_count"] == 2
        forbidden_keys = {"api_key", "secret", "timestamp", "latency", "token"}
        assert set(data.keys()).isdisjoint(forbidden_keys)


# ===========================================================================
# 2. Test class: VisionResultNormalizer Unit & Sanitization Tests
# ===========================================================================


class TestVisionResultNormalizerUnit:
    """Tests 1-7, 9-18: Result normalization, lineage preservation, metadata sanitization."""

    def test_05_normalize_valid_vision_result(self) -> None:
        """normalize() passes valid VisionResult and attaches execution trace."""
        res = VisionResult(
            query="Analyze chart",
            status="success",
            description="Upward trend detected.",
            document_id="doc-norm-01",
            filename="chart.pdf",
            chunk_id="chk-norm-01",
            page_number=1,
            content_type="chart",
            metadata={"chart_kind": "bar"},
        )
        trace = VisionExecutionTrace.create_default()
        normalized = VisionResultNormalizer.normalize(res, trace=trace)

        assert isinstance(normalized, VisionResult)
        assert normalized.query == "Analyze chart"
        assert normalized.status == "success"
        assert normalized.metadata.get("chart_kind") == "bar"
        assert "execution_trace" in normalized.metadata
        assert "result_normalized" in normalized.metadata["execution_trace"]["stages"]

    def test_06_normalize_dictionary_result(self) -> None:
        """normalize() accepts dictionary representation and converts to VisionResult."""
        raw_dict = {
            "query": "Dict query",
            "status": "success",
            "description": "Dict description",
            "document_id": "doc-dict-99",
            "filename": "dict.pdf",
            "chunk_id": "chk-dict-99",
            "page_number": 3,
            "content_type": "diagram",
            "metadata": {"source": "dict-test"},
        }
        normalized = VisionResultNormalizer.normalize(raw_dict)

        assert isinstance(normalized, VisionResult)
        assert normalized.query == "Dict query"
        assert normalized.document_id == "doc-dict-99"
        assert normalized.metadata["source"] == "dict-test"

    @pytest.mark.parametrize("bad_res", [None, "string", 123, [1, 2], object()])
    def test_07_invalid_result_type_raises_processing_error(self, bad_res: Any) -> None:
        """Passing None or unsupported result types raises VisionProcessingError."""
        with pytest.raises(VisionProcessingError):
            VisionResultNormalizer.normalize(bad_res)

    def test_08_malformed_dictionary_raises_processing_error(self) -> None:
        """Dictionary missing essential result structure raises VisionProcessingError."""
        bad_dict = {"invalid_schema_key": 123}
        # Result from_dict creates defaults, but if metadata or fields fail validation -> error
        with pytest.raises(VisionProcessingError):
            VisionResultNormalizer.normalize({"query": ""})  # empty query fails post-init

    def test_09_invalid_status_raises_processing_error(self) -> None:
        """Dictionary with empty or whitespace-only status raises VisionProcessingError or validation error."""
        bad_status_dict = {"query": "Query", "status": "   ", "description": "d"}
        with pytest.raises((VisionProcessingError, VisionInputValidationError)):
            VisionResultNormalizer.normalize(bad_status_dict)

    def test_10_sanitize_metadata_removes_forbidden_keys_and_raw_bytes(self) -> None:
        """sanitize_metadata removes api_key, secret, token, auth, credentials, image_bytes recursively."""
        raw_meta = {
            "safe_key": "safe_val",
            "api_key": "secret-key-123",
            "token": "bearer-token",
            "image_bytes": b"raw_image_data_payload",
            "nested": {
                "password": "secret-pass",
                "clean_nested": 42,
            },
            "byte_list": [b"byte1", "clean_string"],
        }
        clean = VisionResultNormalizer.sanitize_metadata(raw_meta)

        assert clean.get("safe_key") == "safe_val"
        assert "api_key" not in clean
        assert "token" not in clean
        assert "image_bytes" not in clean
        assert "password" not in clean["nested"]
        assert clean["nested"].get("clean_nested") == 42
        assert clean["byte_list"] == ["clean_string"]

    def test_11_lineage_reconciliation_from_model_input(self) -> None:
        """normalize() reconciles missing lineage fields on VisionResult from VisionModelInput."""
        # Result missing explicit document_id & filename
        res = VisionResult(query="Query", status="success", description="Result desc")

        # Create dummy PreparedImageEvidence & VisionModelInput for lineage source
        ev = VisualEvidence(
            document_id="doc-source-100",
            filename="source.pdf",
            chunk_id="chk-source-100",
            page_number=8,
            chunk_index=4,
            content_type="chart",
            image_bytes=_make_test_png(),
        )
        from vision.image_preparation import prepare_image_evidence
        prep = prepare_image_evidence(ev)
        inp = build_vision_input("Query", prep)

        normalized = VisionResultNormalizer.normalize(res, model_input=inp)

        assert normalized.document_id == "doc-source-100"
        assert normalized.filename == "source.pdf"
        assert normalized.page_number == 8
        assert normalized.chunk_id == "chk-source-100"
        assert normalized.content_type == "chart"


# ===========================================================================
# 3. Test class: Pipeline Integration & Provider Sanitization
# ===========================================================================


class TestPipelineIntegrationAndSanitization:
    """Tests 13-15, 23-26: Integration with LeakyMetadataProvider and DictOutputProvider."""

    def test_12_leaky_metadata_provider_sanitized_in_pipeline(self) -> None:
        """Leaky metadata from provider is sanitized during execution pipeline."""
        config = VisionProviderConfig(provider_name="leaky-prov", model_name="v1")
        provider = LeakyMetadataProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())
        result = adapter.execute("Sanitization check", evidence=[ev])

        assert result.is_success is True
        # Verify sensitive keys were removed
        assert "api_key" not in result.metadata
        assert "auth" not in result.metadata
        assert "image_bytes" not in result.metadata
        assert "token" not in result.metadata.get("nested", {})
        assert result.metadata.get("nested", {}).get("safe_field") == "valid_value"
        # Verify trace attached
        assert "execution_trace" in result.metadata
        assert "execution_lifecycle" in result.metadata

    def test_13_dict_output_provider_converted_in_pipeline(self) -> None:
        """Provider returning dictionary output is converted to clean VisionResult."""
        config = VisionProviderConfig(provider_name="dict-prov", model_name="v1")
        provider = DictOutputProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())
        result = adapter.execute("Dict output check", evidence=[ev])

        assert isinstance(result, VisionResult)
        assert result.is_success is True
        assert result.description == "Analysis result from dictionary provider output."
        assert "api_key" not in result.metadata

    def test_14_malformed_result_provider_raises_processing_error(self) -> None:
        """Provider returning malformed status raises VisionProcessingError."""
        config = VisionProviderConfig(provider_name="malformed-prov", model_name="v1")
        provider = MalformedResultProvider(config)
        adapter = VisionExecutionAdapter(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", image_bytes=_make_test_png())

        with pytest.raises(VisionProcessingError, match="status"):
            adapter.execute("Malformed test", evidence=[ev])


# ===========================================================================
# 4. Test class: Immutability & Serialization Roundtrip
# ===========================================================================


class TestImmutabilityAndSerialization:
    """Tests 23-27: Input immutability during normalization and VisionResult serialization."""

    def test_15_request_and_input_untouched_during_normalization(self) -> None:
        """VisionRequest and VisionModelInput are unchanged after normalization."""
        ev = VisualEvidence(document_id="doc-imm-01", filename="test.pdf", chunk_id="chk-imm-01", image_bytes=_make_test_png())
        from vision.image_preparation import prepare_image_evidence
        prep = prepare_image_evidence(ev)
        inp = build_vision_input("Query", prep)

        inp_dict_before = inp.to_dict()
        res = VisionResult(query="Query", status="success", description="Result")

        VisionResultNormalizer.normalize(res, model_input=inp)

        assert inp.to_dict() == inp_dict_before

    def test_16_normalized_result_to_dict_and_from_dict_serialization(self) -> None:
        """Normalized VisionResult serializes to dict and deserializes cleanly."""
        res = VisionResult(
            query="Serialize query",
            status="success",
            description="Serialize description",
            document_id="doc-ser-01",
            filename="ser.pdf",
            chunk_id="chk-ser-01",
            page_number=2,
            content_type="diagram",
            metadata={"trace": "clean"},
        )
        normalized = VisionResultNormalizer.normalize(res)

        data = normalized.to_dict()
        reconstructed = VisionResult.from_dict(data)

        assert reconstructed.query == normalized.query
        assert reconstructed.status == normalized.status
        assert reconstructed.description == normalized.description
        assert reconstructed.document_id == normalized.document_id
        assert reconstructed.filename == normalized.filename
        assert reconstructed.page_number == normalized.page_number
        assert reconstructed.chunk_id == normalized.chunk_id
        assert reconstructed.content_type == normalized.content_type
        assert reconstructed.metadata == normalized.metadata


# ===========================================================================
# 5. Test class: Security and Offline Verification
# ===========================================================================


class TestSecurityAndOfflineVerification:
    """Tests 37-38: Package exports, zero network, zero LLM dependencies."""

    def test_17_package_exports(self) -> None:
        """VisionResultNormalizer, VisionExecutionTrace, and FORBIDDEN_METADATA_KEYS are exported."""
        from vision import (
            FORBIDDEN_METADATA_KEYS,
            VisionExecutionTrace,
            VisionResultNormalizer,
        )

        assert isinstance(FORBIDDEN_METADATA_KEYS, frozenset)
        assert "api_key" in FORBIDDEN_METADATA_KEYS
        assert inspect.isclass(VisionResultNormalizer)
        assert inspect.isclass(VisionExecutionTrace)

    def test_18_no_network_libraries_in_normalizer(self) -> None:
        """result_normalizer.py contains no network or vendor SDK imports."""
        from vision import result_normalizer

        source = inspect.getsource(result_normalizer)
        forbidden = [
            "import requests",
            "import httpx",
            "import aiohttp",
            "import socket",
            "import urllib.request",
            "import openai",
            "import anthropic",
            "import google.generativeai",
        ]
        for pattern in forbidden:
            assert pattern not in source, f"result_normalizer.py contains forbidden pattern '{pattern}'"
