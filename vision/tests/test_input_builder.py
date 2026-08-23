"""
Comprehensive tests for Day 35: Vision Input Builder.

Tests cover:
  1.  valid query + valid prepared image evidence
  2.  query preservation (exact stripped value)
  3.  document_id preservation
  4.  filename preservation
  5.  page_number preservation
  6.  chunk_id preservation
  7.  chunk_index preservation
  8.  content_type preservation
  9.  metadata preservation
  10. valid image input (format, width, height, mode, size_bytes)
  11. supported visual modality accepted (image, chart, diagram)
  12. empty query rejected
  13. whitespace-only query rejected
  14. invalid query type rejected
  15. missing evidence (None) rejected
  16. invalid evidence type rejected
  17. raw/unprepared VisualEvidence rejected
  18. text-only evidence rejected at VisualEvidence construction
  19. unsupported modality (table) rejected at VisualEvidence construction
  20. deterministic construction -- same inputs -> identical to_dict()
  21. repeated construction consistency
  22. no generated/fake lineage fields
  23. frozen/immutable VisionModelInput
  24. malformed builder_metadata rejected
  25. correct error type for query errors
  26. correct error type for evidence errors
  27. no network calls
  28. no LLM calls (builder contains no async/network code)
  29. no external Vision API calls
  30. module-level build_vision_input convenience function
  31. builder_metadata preserved in output
  32. None builder_metadata becomes empty dict
  33. page_number=None preserved
  34. is_oversized preserved (True and False)
  35. to_dict schema has exactly documented keys
"""

from __future__ import annotations

import io
from typing import Any

import pytest

from vision.exceptions import (
    VisionEvidenceError,
    VisionInputValidationError,
)
from vision.image_preparation import (
    OversizedImagePolicy,
    PreparedImageEvidence,
    prepare_image_evidence,
)
from vision.input_builder import (
    VisionInputBuilder,
    VisionModelInput,
    build_vision_input,
)
from vision.models import VisualEvidence


# ===========================================================================
# Test helpers -- confined to this test module
# ===========================================================================


def _make_png_bytes(width: int = 64, height: int = 48) -> bytes:
    """Generate minimal PNG bytes using Pillow (test helper only)."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(80, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width: int = 64, height: int = 48) -> bytes:
    """Generate minimal JPEG bytes using Pillow (test helper only)."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(200, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_webp_bytes(width: int = 32, height: int = 32) -> bytes:
    """Generate minimal WEBP bytes using Pillow (test helper only)."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(60, 200, 100))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


def _make_visual_evidence(
    document_id: str = "doc-001",
    filename: str = "test.pdf",
    chunk_id: str = "chunk-001",
    page_number: int | None = 1,
    chunk_index: int = 0,
    content_type: str = "image",
    image_bytes: bytes | None = None,
    metadata: dict[str, Any] | None = None,
) -> VisualEvidence:
    """Construct a VisualEvidence for testing."""
    if image_bytes is None:
        image_bytes = _make_png_bytes()
    return VisualEvidence(
        document_id=document_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        chunk_index=chunk_index,
        content_type=content_type,
        image_bytes=image_bytes,
        metadata=metadata or {},
    )


def _make_prepared(
    document_id: str = "doc-001",
    filename: str = "test.pdf",
    chunk_id: str = "chunk-001",
    page_number: int | None = 1,
    chunk_index: int = 0,
    content_type: str = "image",
    image_bytes: bytes | None = None,
    metadata: dict[str, Any] | None = None,
) -> PreparedImageEvidence:
    """Construct a PreparedImageEvidence via the Day 34 pipeline."""
    ev = _make_visual_evidence(
        document_id=document_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        chunk_index=chunk_index,
        content_type=content_type,
        image_bytes=image_bytes or _make_png_bytes(),
        metadata=metadata,
    )
    return prepare_image_evidence(ev)


# ===========================================================================
# 1. Test class: Valid input construction
# ===========================================================================


class TestValidInputConstruction:
    """Tests 1-11: Valid inputs produce correct VisionModelInput."""

    def test_01_valid_query_and_evidence_returns_vision_model_input(self) -> None:
        """Valid query + valid PreparedImageEvidence returns a VisionModelInput."""
        prep = _make_prepared()
        result = build_vision_input("Describe this chart.", prep)
        assert isinstance(result, VisionModelInput)

    def test_02_query_preserved_exactly_after_strip(self) -> None:
        """Query is stored stripped -- leading/trailing whitespace is removed."""
        prep = _make_prepared()
        result = build_vision_input("  Explain the trend.  ", prep)
        assert result.query == "Explain the trend."

    def test_03_document_id_preserved(self) -> None:
        """document_id is copied verbatim from PreparedImageEvidence."""
        prep = _make_prepared(document_id="unique-doc-ABC")
        result = build_vision_input("Query", prep)
        assert result.document_id == "unique-doc-ABC"

    def test_04_filename_preserved(self) -> None:
        """filename is copied verbatim from PreparedImageEvidence."""
        prep = _make_prepared(filename="annual_report_2025.pdf")
        result = build_vision_input("Query", prep)
        assert result.filename == "annual_report_2025.pdf"

    def test_05_page_number_preserved(self) -> None:
        """page_number is copied verbatim from PreparedImageEvidence."""
        prep = _make_prepared(page_number=42)
        result = build_vision_input("Query", prep)
        assert result.page_number == 42

    def test_06_chunk_id_preserved(self) -> None:
        """chunk_id is copied verbatim from PreparedImageEvidence."""
        prep = _make_prepared(chunk_id="chk-XYZ-007")
        result = build_vision_input("Query", prep)
        assert result.chunk_id == "chk-XYZ-007"

    def test_07_chunk_index_preserved(self) -> None:
        """chunk_index is copied verbatim from PreparedImageEvidence."""
        prep = _make_prepared(chunk_index=15)
        result = build_vision_input("Query", prep)
        assert result.chunk_index == 15

    def test_08_content_type_preserved(self) -> None:
        """content_type is copied verbatim from PreparedImageEvidence."""
        for ct in ("image", "chart", "diagram"):
            prep = _make_prepared(content_type=ct)
            result = build_vision_input("Query", prep)
            assert result.content_type == ct

    def test_09_evidence_metadata_preserved(self) -> None:
        """Evidence metadata is copied verbatim to evidence_metadata."""
        meta = {"source": "extractor", "page_label": "IV"}
        prep = _make_prepared(metadata=meta)
        result = build_vision_input("Query", prep)
        assert result.evidence_metadata == meta

    def test_10_valid_image_technical_fields_captured(self) -> None:
        """image_format, width, height, mode, size_bytes are carried from PreparedImageEvidence."""
        png_bytes = _make_png_bytes(128, 96)
        prep = _make_prepared(image_bytes=png_bytes)
        result = build_vision_input("Analyse image.", prep)

        assert result.image_format == "png"
        assert result.width == 128
        assert result.height == 96
        assert isinstance(result.mode, str) and len(result.mode) > 0
        assert result.size_bytes == len(png_bytes)

    def test_11_supported_visual_modalities_accepted(self) -> None:
        """All three visual modalities are accepted through the full pipeline."""
        for ct in ("image", "chart", "diagram"):
            prep = _make_prepared(content_type=ct)
            result = build_vision_input("Query", prep)
            assert result.content_type == ct
            assert isinstance(result, VisionModelInput)


# ===========================================================================
# 2. Test class: Query validation
# ===========================================================================


class TestQueryValidation:
    """Tests 12-14: Invalid queries are rejected with VisionInputValidationError."""

    def test_12_empty_query_rejected(self) -> None:
        """Empty string query raises VisionInputValidationError."""
        prep = _make_prepared()
        with pytest.raises(VisionInputValidationError, match="empty"):
            build_vision_input("", prep)

    def test_13_whitespace_only_query_rejected(self) -> None:
        """Whitespace-only query raises VisionInputValidationError."""
        prep = _make_prepared()
        with pytest.raises(VisionInputValidationError, match="empty"):
            build_vision_input("   \t\n  ", prep)

    def test_14_invalid_query_type_rejected(self) -> None:
        """Non-string query raises VisionInputValidationError."""
        prep = _make_prepared()

        with pytest.raises(VisionInputValidationError, match="string"):
            build_vision_input(None, prep)  # type: ignore[arg-type]

        with pytest.raises(VisionInputValidationError, match="string"):
            build_vision_input(42, prep)  # type: ignore[arg-type]

        with pytest.raises(VisionInputValidationError, match="string"):
            build_vision_input(["query"], prep)  # type: ignore[arg-type]

    def test_14b_none_query_rejected(self) -> None:
        """None query raises VisionInputValidationError (not NoneType confusion)."""
        prep = _make_prepared()
        with pytest.raises(VisionInputValidationError):
            build_vision_input(None, prep)  # type: ignore[arg-type]


# ===========================================================================
# 3. Test class: Evidence validation
# ===========================================================================


class TestEvidenceValidation:
    """Tests 15-19: Invalid or wrong-type evidence is rejected."""

    def test_15_none_evidence_rejected(self) -> None:
        """None evidence raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError, match="cannot be None"):
            build_vision_input("Query", None)  # type: ignore[arg-type]

    def test_16_invalid_evidence_type_rejected(self) -> None:
        """Non-PreparedImageEvidence raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError):
            build_vision_input("Query", "not-evidence")  # type: ignore[arg-type]

        with pytest.raises(VisionEvidenceError):
            build_vision_input("Query", {"document_id": "x"})  # type: ignore[arg-type]

        with pytest.raises(VisionEvidenceError):
            build_vision_input("Query", 12345)  # type: ignore[arg-type]

    def test_17_raw_visual_evidence_rejected(self) -> None:
        """Raw VisualEvidence (unprepared) raises VisionEvidenceError with helpful message."""
        raw = _make_visual_evidence()
        with pytest.raises(VisionEvidenceError, match="Raw VisualEvidence"):
            build_vision_input("Query", raw)  # type: ignore[arg-type]

    def test_18_text_only_evidence_blocked_at_construction(self) -> None:
        """VisualEvidence with content_type='text' cannot be constructed (Day 32 guard)."""
        with pytest.raises(Exception):  # VisionEvidenceError from VisualEvidence.__post_init__
            VisualEvidence(
                document_id="d1",
                filename="f.pdf",
                chunk_id="c1",
                content_type="text",
            )

    def test_19_table_content_type_blocked_at_construction(self) -> None:
        """VisualEvidence with content_type='table' cannot be constructed (Day 32 guard)."""
        with pytest.raises(Exception):
            VisualEvidence(
                document_id="d1",
                filename="f.pdf",
                chunk_id="c1",
                content_type="table",
            )


# ===========================================================================
# 4. Test class: Determinism and repeatability
# ===========================================================================


class TestDeterminismAndRepeatability:
    """Tests 20-21: Builder is deterministic and consistent."""

    def test_20_deterministic_construction_same_inputs(self) -> None:
        """Same query + same PreparedImageEvidence -> identical to_dict() output."""
        prep = _make_prepared(
            document_id="doc-det-01",
            filename="chart.pdf",
            chunk_id="chk-det-01",
            page_number=5,
            chunk_index=2,
            content_type="diagram",
            metadata={"series": [1, 2, 3]},
        )
        r1 = build_vision_input("Explain the diagram.", prep)
        r2 = build_vision_input("Explain the diagram.", prep)
        assert r1.to_dict() == r2.to_dict()

    def test_21_repeated_construction_consistency(self) -> None:
        """Multiple separate calls with equivalent evidence produce consistent outputs."""
        png_bytes = _make_png_bytes(32, 32)

        def make() -> VisionModelInput:
            prep = _make_prepared(
                document_id="doc-rep-01",
                chunk_id="chk-rep-01",
                image_bytes=png_bytes,
            )
            return build_vision_input("Describe the image.", prep)

        results = [make() for _ in range(5)]
        first = results[0].to_dict()
        for r in results[1:]:
            assert r.to_dict() == first


# ===========================================================================
# 5. Test class: No fake lineage guarantee
# ===========================================================================


class TestNoFakeLineageGuarantee:
    """Test 22: No lineage fields are generated or synthesised."""

    def test_22_all_lineage_fields_match_source_evidence(self) -> None:
        """Every lineage field on VisionModelInput exactly matches the PreparedImageEvidence."""
        prep = _make_prepared(
            document_id="doc-nosyn-01",
            filename="original.pdf",
            chunk_id="chk-nosyn-01",
            page_number=7,
            chunk_index=3,
            content_type="image",
            metadata={"tag": "real"},
        )
        result = build_vision_input("Query", prep)

        assert result.document_id == prep.document_id
        assert result.filename == prep.filename
        assert result.chunk_id == prep.chunk_id
        assert result.page_number == prep.page_number
        assert result.chunk_index == prep.chunk_index
        assert result.content_type == prep.content_type
        assert result.evidence_metadata == prep.metadata

    def test_22b_technical_fields_match_source_evidence(self) -> None:
        """Technical image fields match PreparedImageEvidence exactly."""
        prep = _make_prepared(image_bytes=_make_jpeg_bytes(80, 60))
        result = build_vision_input("Query", prep)

        assert result.image_format == prep.image_format
        assert result.width == prep.width
        assert result.height == prep.height
        assert result.mode == prep.mode
        assert result.size_bytes == prep.size_bytes
        assert result.is_oversized == prep.is_oversized


# ===========================================================================
# 6. Test class: Immutability (frozen dataclass)
# ===========================================================================


class TestImmutability:
    """Test 23: VisionModelInput is frozen -- mutation raises FrozenInstanceError."""

    def test_23_frozen_model_cannot_mutate_query(self) -> None:
        """Setting query on VisionModelInput after construction raises FrozenInstanceError."""
        from dataclasses import FrozenInstanceError

        prep = _make_prepared()
        result = build_vision_input("Original query.", prep)
        with pytest.raises(FrozenInstanceError):
            result.query = "Changed"  # type: ignore[misc]

    def test_23b_frozen_model_cannot_mutate_document_id(self) -> None:
        """Setting document_id raises FrozenInstanceError."""
        from dataclasses import FrozenInstanceError

        prep = _make_prepared()
        result = build_vision_input("Query", prep)
        with pytest.raises(FrozenInstanceError):
            result.document_id = "tampered-id"  # type: ignore[misc]

    def test_23c_frozen_model_cannot_mutate_lineage(self) -> None:
        """All lineage fields are protected from direct attribute assignment."""
        from dataclasses import FrozenInstanceError

        prep = _make_prepared()
        result = build_vision_input("Query", prep)

        with pytest.raises(FrozenInstanceError):
            result.filename = "tampered"  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            result.chunk_id = "tampered"  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            result.content_type = "tampered"  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            result.image_format = "tampered"  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            result.width = 999  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            result.height = 999  # type: ignore[misc]

    def test_23d_evidence_metadata_is_shallow_copy(self) -> None:
        """Mutating the original metadata dict after building does not affect result."""
        original_meta = {"key": "value"}
        prep = _make_prepared(metadata=original_meta)
        result = build_vision_input("Query", prep)
        original_meta["key"] = "mutated"
        assert result.evidence_metadata.get("key") == "value"


# ===========================================================================
# 7. Test class: Metadata handling
# ===========================================================================


class TestMetadataHandling:
    """Tests 24: Metadata validation and handling."""

    def test_24_malformed_builder_metadata_rejected(self) -> None:
        """Non-dict builder_metadata raises VisionInputValidationError."""
        prep = _make_prepared()
        with pytest.raises(VisionInputValidationError, match="builder_metadata"):
            build_vision_input("Query", prep, builder_metadata="bad")  # type: ignore[arg-type]

        with pytest.raises(VisionInputValidationError, match="builder_metadata"):
            build_vision_input("Query", prep, builder_metadata=42)  # type: ignore[arg-type]

    def test_24b_none_builder_metadata_becomes_empty_dict(self) -> None:
        """None builder_metadata results in empty dict on VisionModelInput."""
        prep = _make_prepared()
        result = build_vision_input("Query", prep, builder_metadata=None)
        assert result.builder_metadata == {}

    def test_24c_builder_metadata_preserved(self) -> None:
        """Supplied builder_metadata is carried through to VisionModelInput."""
        meta = {"route": "vision-gpu-cluster", "priority": 1}
        prep = _make_prepared()
        result = build_vision_input("Query", prep, builder_metadata=meta)
        assert result.builder_metadata == meta

    def test_24d_empty_evidence_metadata_preserved(self) -> None:
        """Empty evidence metadata remains empty in the output."""
        prep = _make_prepared(metadata={})
        result = build_vision_input("Query", prep)
        assert result.evidence_metadata == {}


# ===========================================================================
# 8. Test class: Correct error types
# ===========================================================================


class TestCorrectErrorTypes:
    """Tests 25-26: Errors must be raised as the correct exception types."""

    def test_25_query_error_raises_vision_input_validation_error(self) -> None:
        """Query validation failures raise VisionInputValidationError."""
        prep = _make_prepared()
        with pytest.raises(VisionInputValidationError):
            build_vision_input("", prep)

        with pytest.raises(VisionInputValidationError):
            build_vision_input(None, prep)  # type: ignore[arg-type]

    def test_26_evidence_error_raises_vision_evidence_error(self) -> None:
        """Evidence validation failures raise VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError):
            build_vision_input("Query", None)  # type: ignore[arg-type]

        raw = _make_visual_evidence()
        with pytest.raises(VisionEvidenceError):
            build_vision_input("Query", raw)  # type: ignore[arg-type]


# ===========================================================================
# 9. Test class: No network, no LLM, no external API
# ===========================================================================


class TestNoExternalCalls:
    """Tests 27-29: The builder is fully deterministic and offline."""

    def test_27_no_network_or_async_in_builder_source(self) -> None:
        """Builder source code contains no socket, httpx, requests, or aiohttp usage."""
        import inspect
        from vision import input_builder

        source = inspect.getsource(input_builder)
        forbidden_patterns = [
            "import socket",
            "import httpx",
            "import requests",
            "import aiohttp",
            "urllib.request",
            "http.client",
        ]
        for pattern in forbidden_patterns:
            assert pattern not in source, (
                f"input_builder.py contains forbidden network import: '{pattern}'"
            )

    def test_28_no_llm_or_api_imports_in_builder(self) -> None:
        """Builder source code contains no LLM client imports."""
        import inspect
        from vision import input_builder

        source = inspect.getsource(input_builder)
        forbidden = [
            "openai",
            "anthropic",
            "google.generativeai",
            "transformers",
            "langgraph",
            "langchain",
            "fastapi",
            "streamlit",
        ]
        for token in forbidden:
            assert token not in source, (
                f"input_builder.py contains forbidden import: '{token}'"
            )

    def test_29_no_external_vision_api_in_builder(self) -> None:
        """Builder source code contains no vision API calls (scanning code lines only)."""
        import inspect
        from vision import input_builder

        source = inspect.getsource(input_builder)
        code_lines = []
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'"):
                continue
            code_lines.append(line)
        code_only = "\n".join(code_lines)

        forbidden_imports = [
            "vision_api",
            "VisionAI",
            "chat.completions",
            "generate_content",
        ]
        for token in forbidden_imports:
            assert token not in code_only, (
                f"input_builder.py code contains forbidden vision API reference: '{token}'"
            )


# ===========================================================================
# 10. Test class: Module-level convenience function
# ===========================================================================


class TestConvenienceFunction:
    """Test 30: build_vision_input module-level function works correctly."""

    def test_30_convenience_function_produces_correct_output(self) -> None:
        """build_vision_input() delegates correctly to VisionInputBuilder."""
        prep = _make_prepared(
            document_id="doc-fn-01",
            filename="fig.pdf",
            chunk_id="chk-fn-01",
            content_type="chart",
        )
        result = build_vision_input("Describe the chart.", prep)
        assert isinstance(result, VisionModelInput)
        assert result.query == "Describe the chart."
        assert result.document_id == "doc-fn-01"
        assert result.content_type == "chart"

    def test_30b_convenience_function_matches_builder_output(self) -> None:
        """build_vision_input() produces identical output to VisionInputBuilder.build()."""
        prep = _make_prepared()
        fn_result = build_vision_input("Query A", prep)
        cls_result = VisionInputBuilder().build("Query A", prep)
        assert fn_result.to_dict() == cls_result.to_dict()


# ===========================================================================
# 11. Test class: Additional field preservation
# ===========================================================================


class TestAdditionalFieldPreservation:
    """Tests 31-35: Additional field invariants and edge cases."""

    def test_31_builder_metadata_preserved_in_output(self) -> None:
        """builder_metadata supplied to build() appears in VisionModelInput.builder_metadata."""
        meta = {"experiment": "ablation-v2", "run_id": 7}
        prep = _make_prepared()
        result = build_vision_input("Query", prep, builder_metadata=meta)
        assert result.builder_metadata["experiment"] == "ablation-v2"
        assert result.builder_metadata["run_id"] == 7

    def test_32_none_builder_metadata_stored_as_empty_dict(self) -> None:
        """None builder_metadata -> empty dict in VisionModelInput.builder_metadata."""
        prep = _make_prepared()
        result = build_vision_input("Query", prep, builder_metadata=None)
        assert result.builder_metadata == {}
        assert isinstance(result.builder_metadata, dict)

    def test_33_page_number_none_preserved(self) -> None:
        """page_number=None is preserved faithfully."""
        prep = _make_prepared(page_number=None)
        result = build_vision_input("Query", prep)
        assert result.page_number is None

    def test_34_is_oversized_false_preserved(self) -> None:
        """is_oversized=False (normal image) is preserved."""
        prep = _make_prepared()
        result = build_vision_input("Query", prep)
        assert result.is_oversized is False

    def test_34b_is_oversized_true_preserved_when_policy_allows(self) -> None:
        """is_oversized=True is preserved when policy.reject=False allows it."""
        policy = OversizedImagePolicy(max_dimension=10, max_pixels=100, reject=False)
        # A 64x48 image will exceed max_dimension=10 -> is_oversized=True
        ev = _make_visual_evidence(image_bytes=_make_png_bytes(64, 48))
        prep = prepare_image_evidence(ev, oversized_policy=policy)
        assert prep.is_oversized is True

        result = build_vision_input("Query", prep)
        assert result.is_oversized is True

    def test_35_to_dict_schema_has_exactly_documented_keys(self) -> None:
        """to_dict() contains exactly the 15 documented schema keys -- no extras."""
        prep = _make_prepared()
        result = build_vision_input("Query", prep)
        d = result.to_dict()

        expected_keys = {
            "query",
            "document_id",
            "filename",
            "page_number",
            "chunk_id",
            "chunk_index",
            "content_type",
            "image_format",
            "width",
            "height",
            "mode",
            "size_bytes",
            "is_oversized",
            "evidence_metadata",
            "builder_metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_35b_jpeg_format_preserved(self) -> None:
        """JPEG format is correctly carried through the full pipeline."""
        ev = _make_visual_evidence(image_bytes=_make_jpeg_bytes(60, 40))
        prep = prepare_image_evidence(ev)
        result = build_vision_input("Describe the image.", prep)
        assert result.image_format == "jpeg"
        assert result.width == 60
        assert result.height == 40

    def test_35c_webp_format_preserved(self) -> None:
        """WEBP format is correctly carried through the full pipeline."""
        ev = _make_visual_evidence(image_bytes=_make_webp_bytes(32, 32))
        prep = prepare_image_evidence(ev)
        result = build_vision_input("Describe the image.", prep)
        assert result.image_format == "webp"
        assert result.width == 32
        assert result.height == 32


# ===========================================================================
# 12. Test class: VisionModelInput constructor guards
# ===========================================================================


class TestVisionModelInputConstructorGuards:
    """Validate that VisionModelInput itself enforces field integrity."""

    def _base_prep(self) -> PreparedImageEvidence:
        return _make_prepared()

    def _base_result(self) -> VisionModelInput:
        return build_vision_input("Query", self._base_prep())

    def test_invalid_content_type_in_model_input_raises(self) -> None:
        """Constructing VisionModelInput with non-visual content_type raises VisionEvidenceError."""
        prep = self._base_prep()
        with pytest.raises(VisionEvidenceError, match="content_type"):
            VisionModelInput(
                query="Q",
                document_id=prep.document_id,
                filename=prep.filename,
                page_number=prep.page_number,
                chunk_id=prep.chunk_id,
                chunk_index=prep.chunk_index,
                content_type="text",  # invalid
                image_format=prep.image_format,
                width=prep.width,
                height=prep.height,
                mode=prep.mode,
                size_bytes=prep.size_bytes,
                is_oversized=prep.is_oversized,
                evidence_metadata=prep.metadata,
            )

    def test_empty_query_in_direct_construction_raises(self) -> None:
        """Direct construction with empty query raises VisionInputValidationError."""
        prep = self._base_prep()
        with pytest.raises(VisionInputValidationError, match="query"):
            VisionModelInput(
                query="",  # invalid
                document_id=prep.document_id,
                filename=prep.filename,
                page_number=prep.page_number,
                chunk_id=prep.chunk_id,
                chunk_index=prep.chunk_index,
                content_type=prep.content_type,
                image_format=prep.image_format,
                width=prep.width,
                height=prep.height,
                mode=prep.mode,
                size_bytes=prep.size_bytes,
                is_oversized=prep.is_oversized,
                evidence_metadata=prep.metadata,
            )

    def test_invalid_width_in_direct_construction_raises(self) -> None:
        """Direct construction with width=0 raises VisionInputValidationError."""
        prep = self._base_prep()
        with pytest.raises(VisionInputValidationError, match="width"):
            VisionModelInput(
                query="Q",
                document_id=prep.document_id,
                filename=prep.filename,
                page_number=prep.page_number,
                chunk_id=prep.chunk_id,
                chunk_index=prep.chunk_index,
                content_type=prep.content_type,
                image_format=prep.image_format,
                width=0,  # invalid
                height=prep.height,
                mode=prep.mode,
                size_bytes=prep.size_bytes,
                is_oversized=prep.is_oversized,
                evidence_metadata=prep.metadata,
            )


# ===========================================================================
# 13. Test class: Pipeline integration Day33 -> 34 -> 35
# ===========================================================================


class TestPipelineIntegration:
    """Validate the full Day 33 -> 34 -> 35 pipeline without any shortcuts."""

    def test_full_pipeline_image(self) -> None:
        """Full pipeline: VisualEvidence -> PreparedImageEvidence -> VisionModelInput."""
        # Day 33: VisualEvidence construction
        ev = VisualEvidence(
            document_id="doc-pipeline-01",
            filename="source.pdf",
            chunk_id="chk-pipeline-01",
            page_number=3,
            chunk_index=1,
            content_type="image",
            image_bytes=_make_png_bytes(64, 64),
            metadata={"origin": "pipeline-test"},
        )

        # Day 34: Image preparation
        prep = prepare_image_evidence(ev)
        assert isinstance(prep, PreparedImageEvidence)
        assert prep.image_format == "png"
        assert prep.width == 64

        # Day 35: Input building
        result = build_vision_input("Explain the image.", prep)
        assert isinstance(result, VisionModelInput)
        assert result.query == "Explain the image."
        assert result.document_id == "doc-pipeline-01"
        assert result.filename == "source.pdf"
        assert result.chunk_id == "chk-pipeline-01"
        assert result.page_number == 3
        assert result.chunk_index == 1
        assert result.content_type == "image"
        assert result.image_format == "png"
        assert result.width == 64
        assert result.height == 64
        assert result.evidence_metadata == {"origin": "pipeline-test"}

    def test_full_pipeline_chart(self) -> None:
        """Full pipeline with chart modality."""
        ev = VisualEvidence(
            document_id="doc-chart-01",
            filename="chart.pdf",
            chunk_id="chk-chart-01",
            page_number=None,
            chunk_index=0,
            content_type="chart",
            image_bytes=_make_jpeg_bytes(100, 75),
        )
        prep = prepare_image_evidence(ev)
        result = build_vision_input("Describe the chart trend.", prep)

        assert result.content_type == "chart"
        assert result.image_format == "jpeg"
        assert result.page_number is None

    def test_full_pipeline_diagram(self) -> None:
        """Full pipeline with diagram modality."""
        ev = VisualEvidence(
            document_id="doc-diag-01",
            filename="diagram.pdf",
            chunk_id="chk-diag-01",
            page_number=10,
            chunk_index=2,
            content_type="diagram",
            image_bytes=_make_webp_bytes(32, 32),
        )
        prep = prepare_image_evidence(ev)
        result = build_vision_input("Explain the architecture.", prep)

        assert result.content_type == "diagram"
        assert result.image_format == "webp"
        assert result.page_number == 10
