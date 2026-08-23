"""
Comprehensive tests for Day 34: Image Evidence Preparation and Validation.

Tests cover:
  1.  Valid PNG preparation
  2.  Valid JPEG preparation
  3.  Valid WEBP preparation
  4.  Unsupported format rejection
  5.  Missing image source rejection
  6.  Empty image_bytes rejection
  7.  Corrupted image_bytes rejection
  8.  Unreadable image file (missing path)
  9.  Zero-width image rejection
  10. Zero-height image rejection
  11. Negative-dimension image (not representable as valid Pillow image)
  12. Oversized image rejection (default policy)
  13. Oversized image accepted (reject=False policy)
  14. Valid technical metadata capture
  15. Missing metadata preserved as empty dict
  16. document_id lineage preservation
  17. filename lineage preservation
  18. page_number lineage preservation
  19. chunk_id lineage preservation
  20. chunk_index lineage preservation
  21. content_type lineage preservation
  22. metadata lineage preservation
  23. image_format detection from bytes
  24. width detection
  25. height detection
  26. Deterministic preparation — same input, same output
  27. Repeated preparation consistency (to_dict equality)
  28. Non-visual evidence rejection (text content_type blocked at VisualEvidence level)
  29. No fake image data in production code
  30. No fake metadata or lineage synthesis

Helper utilities (in-memory image generation) are confined to this test file only.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Any

import pytest

from vision.exceptions import (
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
)
from vision.image_preparation import (
    DEFAULT_MAX_DIMENSION,
    DEFAULT_MAX_PIXELS,
    SUPPORTED_IMAGE_FORMATS,
    ImageEvidencePreparator,
    OversizedImagePolicy,
    PreparedImageEvidence,
    prepare_image_evidence,
)
from vision.models import VALID_VISUAL_CONTENT_TYPES, VisualEvidence


# ===========================================================================
# Test helpers — confined to this test module only
# ===========================================================================


def _make_png_bytes(width: int = 64, height: int = 48) -> bytes:
    """Generate minimal in-memory PNG image bytes using Pillow."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width: int = 64, height: int = 48) -> bytes:
    """Generate minimal in-memory JPEG image bytes using Pillow."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_webp_bytes(width: int = 64, height: int = 48) -> bytes:
    """Generate minimal in-memory WEBP image bytes using Pillow."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(50, 200, 100))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


def _make_visual_evidence(
    content_type: str = "image",
    image_bytes: bytes | None = None,
    image_path: str | None = None,
    image_format: str | None = None,
    document_id: str = "doc-001",
    filename: str = "test.pdf",
    chunk_id: str = "chunk-001",
    page_number: int | None = 1,
    chunk_index: int = 0,
    metadata: dict[str, Any] | None = None,
) -> VisualEvidence:
    """Construct a VisualEvidence for testing."""
    return VisualEvidence(
        document_id=document_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        chunk_index=chunk_index,
        content_type=content_type,
        image_bytes=image_bytes,
        image_path=image_path,
        image_format=image_format,
        metadata=metadata or {},
    )


# ===========================================================================
# 1. Test class: Valid format preparation
# ===========================================================================


class TestValidFormatPreparation:
    """Tests 1-3: Valid PNG, JPEG, and WEBP preparation succeeds."""

    def test_01_valid_png_preparation(self) -> None:
        """Valid PNG bytes produce a PreparedImageEvidence with format='png'."""
        ev = _make_visual_evidence(image_bytes=_make_png_bytes(64, 48))
        result = prepare_image_evidence(ev)

        assert isinstance(result, PreparedImageEvidence)
        assert result.image_format == "png"
        assert result.width == 64
        assert result.height == 48
        assert result.mode in ("RGB", "RGBA", "P", "L")
        assert result.size_bytes > 0
        assert result.is_oversized is False

    def test_02_valid_jpeg_preparation(self) -> None:
        """Valid JPEG bytes produce a PreparedImageEvidence with format='jpeg'."""
        ev = _make_visual_evidence(image_bytes=_make_jpeg_bytes(100, 75))
        result = prepare_image_evidence(ev)

        assert result.image_format == "jpeg"
        assert result.width == 100
        assert result.height == 75
        assert result.size_bytes > 0

    def test_03_valid_webp_preparation(self) -> None:
        """Valid WEBP bytes produce a PreparedImageEvidence with format='webp'."""
        ev = _make_visual_evidence(image_bytes=_make_webp_bytes(32, 32))
        result = prepare_image_evidence(ev)

        assert result.image_format == "webp"
        assert result.width == 32
        assert result.height == 32
        assert result.size_bytes > 0

    def test_03b_all_visual_content_types_accepted(self) -> None:
        """All supported visual modalities (image, chart, diagram) are accepted."""
        png_bytes = _make_png_bytes()
        for ct in VALID_VISUAL_CONTENT_TYPES:
            ev = _make_visual_evidence(content_type=ct, image_bytes=png_bytes)
            result = prepare_image_evidence(ev)
            assert result.content_type == ct


# ===========================================================================
# 2. Test class: Unsupported format rejection
# ===========================================================================


class TestUnsupportedFormat:
    """Test 4: Unsupported image formats are rejected."""

    def test_04_unsupported_gif_format_rejected(self) -> None:
        """GIF images are rejected with VisionEvidenceError."""
        from PIL import Image

        img = Image.new("P", (10, 10))
        buf = io.BytesIO()
        img.save(buf, format="GIF")
        gif_bytes = buf.getvalue()

        ev = _make_visual_evidence(image_bytes=gif_bytes)
        with pytest.raises(VisionEvidenceError, match="Unsupported image format"):
            prepare_image_evidence(ev)

    def test_04b_unsupported_bmp_format_rejected(self) -> None:
        """BMP images are rejected with VisionEvidenceError."""
        from PIL import Image

        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        bmp_bytes = buf.getvalue()

        ev = _make_visual_evidence(image_bytes=bmp_bytes)
        with pytest.raises(VisionEvidenceError, match="Unsupported image format"):
            prepare_image_evidence(ev)

    def test_04c_supported_formats_constant_correct(self) -> None:
        """SUPPORTED_IMAGE_FORMATS contains exactly png, jpeg, webp."""
        assert SUPPORTED_IMAGE_FORMATS == frozenset({"png", "jpeg", "webp"})


# ===========================================================================
# 3. Test class: Missing / empty image source
# ===========================================================================


class TestMissingImageSource:
    """Tests 5-6: No image source or empty bytes raises VisionEvidenceError."""

    def test_05_no_image_source_raises_error(self) -> None:
        """VisualEvidence with no image_bytes and no image_path raises VisionEvidenceError."""
        ev = _make_visual_evidence(image_bytes=None, image_path=None)
        with pytest.raises(VisionEvidenceError, match="no image source"):
            prepare_image_evidence(ev)

    def test_06_empty_image_bytes_raises_error(self) -> None:
        """Empty image_bytes (b'') raises VisionEvidenceError."""
        ev = _make_visual_evidence(image_bytes=b"")
        with pytest.raises(VisionEvidenceError, match="empty"):
            prepare_image_evidence(ev)


# ===========================================================================
# 4. Test class: Corrupted / unreadable image
# ===========================================================================


class TestCorruptedImage:
    """Tests 7-8: Corrupted bytes and unreadable file paths raise appropriate errors."""

    def test_07_corrupted_image_bytes_raises_processing_error(self) -> None:
        """Random/corrupted bytes that are not a valid image raise VisionProcessingError."""
        corrupted = b"\x00\x01\x02\x03corrupted_garbage_not_an_image"
        ev = _make_visual_evidence(image_bytes=corrupted)
        with pytest.raises(VisionProcessingError):
            prepare_image_evidence(ev)

    def test_08_missing_image_path_raises_error(self) -> None:
        """A non-existent image_path raises VisionEvidenceError."""
        ev = _make_visual_evidence(
            image_bytes=None,
            image_path="/non/existent/path/to/image.png",
        )
        with pytest.raises(VisionEvidenceError, match="not found"):
            prepare_image_evidence(ev)

    def test_08b_empty_path_file_raises_error(self) -> None:
        """An image_path pointing to a 0-byte file raises VisionEvidenceError."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            empty_path = f.name
        try:
            # File exists but is empty
            ev = _make_visual_evidence(image_bytes=None, image_path=empty_path)
            with pytest.raises(VisionEvidenceError, match="empty"):
                prepare_image_evidence(ev)
        finally:
            os.unlink(empty_path)

    def test_08c_valid_png_from_path(self) -> None:
        """A valid PNG image loaded via image_path succeeds."""
        png_bytes = _make_png_bytes(32, 32)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            path = f.name
        try:
            ev = _make_visual_evidence(image_bytes=None, image_path=path)
            result = prepare_image_evidence(ev)
            assert result.image_format == "png"
            assert result.width == 32
            assert result.height == 32
        finally:
            os.unlink(path)

    def test_08d_image_bytes_takes_priority_over_path(self) -> None:
        """When both image_bytes and image_path are supplied, bytes takes priority."""
        png_bytes = _make_png_bytes(16, 16)
        # Path points nowhere — if path took priority, this would raise
        ev = _make_visual_evidence(
            image_bytes=png_bytes,
            image_path="/no/such/path.png",
        )
        result = prepare_image_evidence(ev)
        assert result.image_format == "png"
        assert result.width == 16


# ===========================================================================
# 5. Test class: Dimension validation
# ===========================================================================


class TestDimensionValidation:
    """Tests 9-11: Zero/invalid dimensions are caught."""

    def test_09_zero_width_rejected(self) -> None:
        """The preparation layer correctly surfaces a zero/invalid-width image error.

        Pillow raises an error when trying to save a 0-width image, so we inject
        a corrupted header that cannot be decoded — this triggers VisionProcessingError
        rather than VisionEvidenceError for zero-width specifically, because Pillow
        itself rejects degenerate dimensions during open/load.
        """
        # We can't construct a valid image with zero width — Pillow won't allow it.
        # A 0x10 image cannot be saved. We verify the guard via a corrupted stream.
        corrupted = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # plausible-looking but invalid PNG
        ev = _make_visual_evidence(image_bytes=corrupted)
        with pytest.raises((VisionEvidenceError, VisionProcessingError)):
            prepare_image_evidence(ev)

    def test_10_zero_height_rejected(self) -> None:
        """Similarly, degenerate height data raises an appropriate vision error."""
        corrupted = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 50  # malformed WEBP header
        ev = _make_visual_evidence(image_bytes=corrupted)
        with pytest.raises((VisionEvidenceError, VisionProcessingError)):
            prepare_image_evidence(ev)

    def test_11_negative_dimensions_not_representable_in_pillow(self) -> None:
        """Pillow cannot open an image with negative dimensions — only valid images pass."""
        # Verify that a valid 1x1 image (minimum valid) succeeds
        from PIL import Image

        img = Image.new("RGB", (1, 1), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        ev = _make_visual_evidence(image_bytes=buf.getvalue())
        result = prepare_image_evidence(ev)
        assert result.width == 1
        assert result.height == 1

    def test_11b_dimension_validation_guards_in_preparator(self) -> None:
        """_validate_dimensions raises for explicit width=0 / height=0 when called directly."""
        from unittest.mock import MagicMock

        prep = ImageEvidencePreparator()
        ev = _make_visual_evidence(image_bytes=_make_png_bytes())

        # Simulate a zero-width Pillow image object
        mock_img_w = MagicMock()
        mock_img_w.size = (0, 100)
        with pytest.raises(VisionEvidenceError, match="width"):
            prep._validate_dimensions(mock_img_w, ev)

        # Simulate a zero-height Pillow image object
        mock_img_h = MagicMock()
        mock_img_h.size = (100, 0)
        with pytest.raises(VisionEvidenceError, match="height"):
            prep._validate_dimensions(mock_img_h, ev)


# ===========================================================================
# 6. Test class: Oversized image policy
# ===========================================================================


class TestOversizedImagePolicy:
    """Tests 12-13: Oversized images are rejected or flagged per policy."""

    def test_12_oversized_image_rejected_by_default(self) -> None:
        """An image exceeding the default max_dimension is rejected."""
        # Create a very narrow but tall image that exceeds DEFAULT_MAX_DIMENSION
        too_tall_dim = DEFAULT_MAX_DIMENSION + 1
        # Rather than actually creating a massive image, use the policy mock approach
        from unittest.mock import MagicMock

        prep = ImageEvidencePreparator()
        ev = _make_visual_evidence(image_bytes=_make_png_bytes())

        with pytest.raises(VisionEvidenceError, match="oversized"):
            prep._apply_oversized_policy(too_tall_dim, 100, ev)

    def test_12b_oversized_by_pixel_count_rejected(self) -> None:
        """An image exceeding the default max_pixels threshold is rejected."""
        from unittest.mock import MagicMock

        prep = ImageEvidencePreparator()
        ev = _make_visual_evidence(image_bytes=_make_png_bytes())

        # A 10001 x 10001 image = 100,020,001 pixels > DEFAULT_MAX_PIXELS
        with pytest.raises(VisionEvidenceError, match="oversized"):
            prep._apply_oversized_policy(10001, 10001, ev)

    def test_13_oversized_image_accepted_when_reject_false(self) -> None:
        """An oversized image is accepted and flagged when policy.reject=False."""
        from unittest.mock import MagicMock

        policy = OversizedImagePolicy(
            max_dimension=100,
            max_pixels=10_000,
            reject=False,
        )
        prep = ImageEvidencePreparator(oversized_policy=policy)
        ev = _make_visual_evidence(image_bytes=_make_png_bytes())

        # 200x200 exceeds both thresholds
        is_oversized = prep._apply_oversized_policy(200, 200, ev)
        assert is_oversized is True

    def test_13b_normal_image_not_flagged_as_oversized(self) -> None:
        """A normal-sized image returns is_oversized=False."""
        ev = _make_visual_evidence(image_bytes=_make_png_bytes(64, 48))
        result = prepare_image_evidence(ev)
        assert result.is_oversized is False

    def test_13c_oversized_policy_invalid_max_dimension_raises(self) -> None:
        """OversizedImagePolicy rejects invalid max_dimension values."""
        with pytest.raises(VisionInputValidationError, match="max_dimension"):
            OversizedImagePolicy(max_dimension=0)

    def test_13d_oversized_policy_invalid_max_pixels_raises(self) -> None:
        """OversizedImagePolicy rejects invalid max_pixels values."""
        with pytest.raises(VisionInputValidationError, match="max_pixels"):
            OversizedImagePolicy(max_pixels=-1)

    def test_13e_oversized_policy_invalid_reject_type_raises(self) -> None:
        """OversizedImagePolicy rejects non-boolean reject values."""
        with pytest.raises(VisionInputValidationError, match="reject"):
            OversizedImagePolicy(reject="yes")  # type: ignore[arg-type]


# ===========================================================================
# 7. Test class: Technical metadata capture
# ===========================================================================


class TestTechnicalMetadataCapture:
    """Tests 14-15: Technical metadata is captured accurately."""

    def test_14_valid_technical_metadata_captured(self) -> None:
        """PreparedImageEvidence carries accurate format, width, height, mode, size_bytes."""
        png_bytes = _make_png_bytes(128, 96)
        ev = _make_visual_evidence(image_bytes=png_bytes, metadata={"source": "pdf-extractor"})
        result = prepare_image_evidence(ev)

        assert result.image_format == "png"
        assert result.width == 128
        assert result.height == 96
        assert isinstance(result.mode, str) and len(result.mode) > 0
        assert result.size_bytes == len(png_bytes)

    def test_15_missing_metadata_preserved_as_empty_dict(self) -> None:
        """Evidence with no metadata results in empty dict on PreparedImageEvidence."""
        ev = _make_visual_evidence(image_bytes=_make_png_bytes(), metadata={})
        result = prepare_image_evidence(ev)
        assert result.metadata == {}
        assert isinstance(result.metadata, dict)


# ===========================================================================
# 8. Test class: Full lineage preservation
# ===========================================================================


class TestLineagePreservation:
    """Tests 16-22: All source lineage fields are exactly preserved."""

    def _prepared(self, **overrides: Any) -> PreparedImageEvidence:
        defaults: dict[str, Any] = dict(
            document_id="doc-lineage-01",
            filename="source.pdf",
            chunk_id="chk-lineage-01",
            page_number=3,
            chunk_index=7,
            content_type="chart",
            metadata={"origin": "test"},
            image_bytes=_make_png_bytes(),
        )
        defaults.update(overrides)
        ev = _make_visual_evidence(**defaults)
        return prepare_image_evidence(ev)

    def test_16_document_id_preserved(self) -> None:
        result = self._prepared(document_id="unique-doc-id-XYZ")
        assert result.document_id == "unique-doc-id-XYZ"
        assert result.source.document_id == "unique-doc-id-XYZ"

    def test_17_filename_preserved(self) -> None:
        result = self._prepared(filename="annual_report_2025.pdf")
        assert result.filename == "annual_report_2025.pdf"
        assert result.source.filename == "annual_report_2025.pdf"

    def test_18_page_number_preserved(self) -> None:
        result = self._prepared(page_number=42)
        assert result.page_number == 42
        assert result.source.page_number == 42

    def test_18b_none_page_number_preserved(self) -> None:
        result = self._prepared(page_number=None)
        assert result.page_number is None
        assert result.source.page_number is None

    def test_19_chunk_id_preserved(self) -> None:
        result = self._prepared(chunk_id="chunk-ABC-007")
        assert result.chunk_id == "chunk-ABC-007"
        assert result.source.chunk_id == "chunk-ABC-007"

    def test_20_chunk_index_preserved(self) -> None:
        result = self._prepared(chunk_index=15)
        assert result.chunk_index == 15
        assert result.source.chunk_index == 15

    def test_21_content_type_preserved(self) -> None:
        for ct in VALID_VISUAL_CONTENT_TYPES:
            result = self._prepared(content_type=ct)
            assert result.content_type == ct
            assert result.source.content_type == ct

    def test_22_metadata_preserved(self) -> None:
        meta = {"author": "test", "page_label": "IV", "nested": {"key": 1}}
        result = self._prepared(metadata=meta)
        assert result.metadata == meta
        assert result.source.metadata == meta

    def test_22b_metadata_is_shallow_copy(self) -> None:
        """Mutating the original metadata dict does not corrupt PreparedImageEvidence."""
        original_meta = {"key": "value"}
        ev = _make_visual_evidence(
            image_bytes=_make_png_bytes(),
            metadata=original_meta,
        )
        result = prepare_image_evidence(ev)
        original_meta["key"] = "mutated"
        # PreparedImageEvidence metadata must be unaffected
        assert result.metadata.get("key") == "value"


# ===========================================================================
# 9. Test class: Format and dimension detection
# ===========================================================================


class TestFormatAndDimensionDetection:
    """Tests 23-25: Format, width, and height are correctly detected."""

    def test_23_image_format_detected_from_bytes(self) -> None:
        """Format is detected from actual image bytes, not from the evidence hint."""
        ev = _make_visual_evidence(
            image_bytes=_make_jpeg_bytes(),
            image_format=None,  # no hint supplied
        )
        result = prepare_image_evidence(ev)
        assert result.image_format == "jpeg"

    def test_24_width_detected(self) -> None:
        """Width is read from actual pixel data."""
        ev = _make_visual_evidence(image_bytes=_make_png_bytes(200, 50))
        result = prepare_image_evidence(ev)
        assert result.width == 200

    def test_25_height_detected(self) -> None:
        """Height is read from actual pixel data."""
        ev = _make_visual_evidence(image_bytes=_make_png_bytes(50, 300))
        result = prepare_image_evidence(ev)
        assert result.height == 300

    def test_25b_format_hint_used_when_pillow_format_is_none(self) -> None:
        """When Pillow returns no format (synthetic image), the image_format hint is used."""
        from PIL import Image
        from unittest.mock import patch, MagicMock

        # Create a real PNG, but patch pil_image.format to None and use a hint
        png_bytes = _make_png_bytes(10, 10)
        ev = _make_visual_evidence(image_bytes=png_bytes, image_format="png")

        prep = ImageEvidencePreparator()
        raw_data, size_bytes = prep._load_image_source(ev)
        pil_img = prep._open_image(raw_data, ev)

        # Patch format to None to simulate the case
        mock_img = MagicMock(wraps=pil_img)
        mock_img.format = None
        mock_img.size = pil_img.size
        mock_img.mode = pil_img.mode

        fmt = prep._validate_format(mock_img, ev)
        assert fmt == "png"

    def test_25c_no_format_no_hint_raises_error(self) -> None:
        """When Pillow returns no format and no hint is available, VisionEvidenceError is raised."""
        from unittest.mock import MagicMock

        prep = ImageEvidencePreparator()
        ev = _make_visual_evidence(image_bytes=_make_png_bytes(), image_format=None)

        mock_img = MagicMock()
        mock_img.format = None

        with pytest.raises(VisionEvidenceError, match="Cannot detect image format"):
            prep._validate_format(mock_img, ev)


# ===========================================================================
# 10. Test class: Determinism and repeated preparation
# ===========================================================================


class TestDeterminismAndRepeatability:
    """Tests 26-27: Preparation is deterministic and consistent across repeated calls."""

    def test_26_deterministic_preparation_same_input(self) -> None:
        """Preparing the same VisualEvidence twice produces identical to_dict() output."""
        ev = _make_visual_evidence(
            document_id="doc-det-01",
            filename="chart.pdf",
            chunk_id="chk-det-01",
            page_number=5,
            chunk_index=2,
            content_type="diagram",
            metadata={"series": [1, 2, 3]},
            image_bytes=_make_webp_bytes(64, 64),
        )
        r1 = prepare_image_evidence(ev)
        r2 = prepare_image_evidence(ev)

        assert r1.to_dict() == r2.to_dict()

    def test_27_repeated_preparation_consistency(self) -> None:
        """Multiple preparations of distinct evidence objects with same data are consistent."""
        png_bytes = _make_png_bytes(32, 32)

        def make() -> PreparedImageEvidence:
            ev = _make_visual_evidence(
                document_id="doc-rep-01",
                chunk_id="chk-rep-01",
                image_bytes=png_bytes,
            )
            return prepare_image_evidence(ev)

        results = [make() for _ in range(5)]
        first_dict = results[0].to_dict()
        for r in results[1:]:
            assert r.to_dict() == first_dict


# ===========================================================================
# 11. Test class: Non-visual evidence rejection
# ===========================================================================


class TestNonVisualEvidenceRejection:
    """Test 28: Non-visual content types cannot reach the preparator."""

    def test_28_text_content_type_blocked_by_visual_evidence_model(self) -> None:
        """VisualEvidence itself rejects non-visual content_types at construction."""
        from vision.exceptions import VisionEvidenceError as VEE

        with pytest.raises(VEE, match="Invalid visual content_type"):
            VisualEvidence(
                document_id="d1",
                filename="f.pdf",
                chunk_id="c1",
                content_type="text",
            )

    def test_28b_table_content_type_blocked_by_visual_evidence_model(self) -> None:
        """Table content_type is also blocked at VisualEvidence construction."""
        with pytest.raises(VisionEvidenceError, match="Invalid visual content_type"):
            VisualEvidence(
                document_id="d1",
                filename="f.pdf",
                chunk_id="c1",
                content_type="table",
            )

    def test_28c_none_evidence_raises_input_validation_error(self) -> None:
        """Passing None as evidence raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="cannot be None"):
            prepare_image_evidence(None)  # type: ignore[arg-type]

    def test_28d_wrong_type_raises_input_validation_error(self) -> None:
        """Passing a non-VisualEvidence object raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="VisualEvidence"):
            prepare_image_evidence("not an evidence object")  # type: ignore[arg-type]


# ===========================================================================
# 12. Test class: No fake data guarantees
# ===========================================================================


class TestNoFakeDataGuarantees:
    """Tests 29-30: Production code must not synthesise image data, metadata, or lineage."""

    def test_29_no_fake_image_data_generated_on_missing_source(self) -> None:
        """When no image source is present, preparation fails — it never invents bytes."""
        ev = _make_visual_evidence(image_bytes=None, image_path=None)
        with pytest.raises(VisionEvidenceError):
            prepare_image_evidence(ev)
        # The error proves no synthetic image was created — otherwise it would succeed

    def test_30_no_lineage_fields_modified_or_synthesised(self) -> None:
        """PreparedImageEvidence carries identical lineage values to the source."""
        ev = _make_visual_evidence(
            document_id="doc-nosyn-01",
            filename="original.pdf",
            chunk_id="chk-nosyn-01",
            page_number=7,
            chunk_index=3,
            content_type="image",
            metadata={"tag": "real"},
            image_bytes=_make_png_bytes(),
        )
        result = prepare_image_evidence(ev)

        assert result.document_id == ev.document_id
        assert result.filename == ev.filename
        assert result.chunk_id == ev.chunk_id
        assert result.page_number == ev.page_number
        assert result.chunk_index == ev.chunk_index
        assert result.content_type == ev.content_type
        assert result.metadata == ev.metadata
        assert result.source is ev

    def test_30b_to_dict_contains_no_extra_synthesised_keys(self) -> None:
        """to_dict() keys are exactly the documented schema — no extra synthesised fields."""
        ev = _make_visual_evidence(image_bytes=_make_png_bytes())
        result = prepare_image_evidence(ev)
        d = result.to_dict()

        expected_keys = {
            "document_id", "filename", "chunk_id", "page_number", "chunk_index",
            "content_type", "metadata", "image_format", "width", "height",
            "mode", "size_bytes", "is_oversized",
        }
        assert set(d.keys()) == expected_keys


# ===========================================================================
# 13. Test class: PreparedImageEvidence constructor guards
# ===========================================================================


class TestPreparedImageEvidenceConstructorGuards:
    """Validate that PreparedImageEvidence enforces lineage integrity itself."""

    def _base_ev(self) -> VisualEvidence:
        return _make_visual_evidence(image_bytes=_make_png_bytes())

    def _base_result(self, ev: VisualEvidence) -> PreparedImageEvidence:
        return prepare_image_evidence(ev)

    def test_lineage_mismatch_document_id_raises(self) -> None:
        """Constructing PreparedImageEvidence with mismatched document_id raises."""
        ev = self._base_ev()
        res = self._base_result(ev)
        with pytest.raises(VisionEvidenceError, match="document_id"):
            PreparedImageEvidence(
                source=ev,
                document_id="wrong-id",  # mismatch
                filename=res.filename,
                chunk_id=res.chunk_id,
                page_number=res.page_number,
                chunk_index=res.chunk_index,
                content_type=res.content_type,
                metadata=res.metadata,
                image_format=res.image_format,
                width=res.width,
                height=res.height,
                mode=res.mode,
                size_bytes=res.size_bytes,
            )

    def test_lineage_mismatch_chunk_id_raises(self) -> None:
        """Constructing PreparedImageEvidence with mismatched chunk_id raises."""
        ev = self._base_ev()
        res = self._base_result(ev)
        with pytest.raises(VisionEvidenceError, match="chunk_id"):
            PreparedImageEvidence(
                source=ev,
                document_id=res.document_id,
                filename=res.filename,
                chunk_id="wrong-chunk",  # mismatch
                page_number=res.page_number,
                chunk_index=res.chunk_index,
                content_type=res.content_type,
                metadata=res.metadata,
                image_format=res.image_format,
                width=res.width,
                height=res.height,
                mode=res.mode,
                size_bytes=res.size_bytes,
            )

    def test_wrong_source_type_raises(self) -> None:
        """Constructing PreparedImageEvidence with non-VisualEvidence source raises."""
        ev = self._base_ev()
        res = self._base_result(ev)
        with pytest.raises(VisionInputValidationError, match="VisualEvidence"):
            PreparedImageEvidence(
                source="not-an-evidence",  # type: ignore[arg-type]
                document_id=res.document_id,
                filename=res.filename,
                chunk_id=res.chunk_id,
                page_number=res.page_number,
                chunk_index=res.chunk_index,
                content_type=res.content_type,
                metadata=res.metadata,
                image_format=res.image_format,
                width=res.width,
                height=res.height,
                mode=res.mode,
                size_bytes=res.size_bytes,
            )


# ===========================================================================
# 14. Test class: ImageEvidencePreparator configuration
# ===========================================================================


class TestImageEvidencePreparatorConfiguration:
    """Validate preparator construction and invalid policy rejection."""

    def test_default_preparator_uses_reject_true_policy(self) -> None:
        """Default preparator has reject=True policy."""
        prep = ImageEvidencePreparator()
        assert prep.oversized_policy.reject is True
        assert prep.oversized_policy.max_dimension == DEFAULT_MAX_DIMENSION
        assert prep.oversized_policy.max_pixels == DEFAULT_MAX_PIXELS

    def test_custom_policy_accepted(self) -> None:
        """Custom policy is accepted and applied."""
        policy = OversizedImagePolicy(max_dimension=500, max_pixels=250_000, reject=False)
        prep = ImageEvidencePreparator(oversized_policy=policy)
        assert prep.oversized_policy.max_dimension == 500
        assert prep.oversized_policy.reject is False

    def test_invalid_policy_type_raises(self) -> None:
        """Passing a non-OversizedImagePolicy raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="OversizedImagePolicy"):
            ImageEvidencePreparator(oversized_policy={"max_dimension": 100})  # type: ignore[arg-type]
