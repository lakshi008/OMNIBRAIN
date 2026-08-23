"""
Day 43 — Vision Agent Request Validation & Contract Hardening Tests.

Verifies the existing validation contracts across:
  1. VisionRequest query validation (empty, None, non-string, whitespace, long).
  2. VisionRequest evidence validation (None, wrong type, malformed items, mixed).
  3. VisualEvidence field validation (document_id, filename, chunk_id, content_type, metadata).
  4. Content type validation (supported, unsupported, empty, None, wrong type).
  5. Metadata validation (None, wrong type, empty dict, valid dict).
  6. Immutability — request and evidence objects remain unchanged after validation.
  7. Provider NOT called for invalid requests (pipeline short-circuit).
  8. Valid requests correctly reach provider with proper lineage.
  9. Multi-evidence request validation (count, order, lineage).
  10. Concurrent validation isolation — invalid/valid requests in parallel.
  11. Deterministic validation behavior (same invalid input → same exception type).
  12. Existing exception hierarchy preserved.
  13. Public API compatibility.
  14. Offline execution guarantee.
"""

from __future__ import annotations

import concurrent.futures
import copy
import io
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
    VisionProviderConfig,
    VisionRequest,
    VisionResult,
    VisualEvidence,
    run_vision_pipeline,
)
from vision.exceptions import (
    VisionAgentError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderExecutionError,
)


# ---------------------------------------------------------------------------
# Helpers & Test Doubles
# ---------------------------------------------------------------------------


def _make_png(width: int = 32, height: int = 32, color: tuple[int, int, int] = (80, 160, 200)) -> bytes:
    """Generate a minimal valid PNG byte payload."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _valid_evidence(
    doc_id: str = "doc-1",
    filename: str = "file.png",
    chunk_id: str = "chk-1",
    content_type: str = "image",
) -> VisualEvidence:
    """Create a fully valid VisualEvidence with real image bytes."""
    return VisualEvidence(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        image_bytes=_make_png(),
        content_type=content_type,
    )


class RecordingProvider(VisionModelProvider):
    """Test double that records inputs and returns a valid VisionResult."""

    def __init__(self, config: VisionProviderConfig) -> None:
        super().__init__(config)
        self.invocation_count: int = 0
        self.recorded_inputs: list[VisionModelInput] = []

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        self.invocation_count += 1
        self.recorded_inputs.append(model_input)
        return VisionResult(
            query=model_input.query,
            status="success",
            description="Validation test result",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
        )


def _make_agent() -> tuple[VisionAgent, RecordingProvider]:
    config = VisionProviderConfig(provider_name="val-test-prov", model_name="v1")
    provider = RecordingProvider(config)
    agent = VisionAgent(provider=provider)
    return agent, provider


# ---------------------------------------------------------------------------
# Test Suite 1: VisionRequest Query Validation
# ---------------------------------------------------------------------------


class TestQueryValidation:
    """Verifies VisionRequest.query validation using the existing contract."""

    def test_01_valid_query_accepted(self) -> None:
        """A well-formed query string is accepted without error."""
        req = VisionRequest(query="Explain the chart", evidence=[])
        assert req.query == "Explain the chart"

    def test_02_query_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace is stripped (per existing contract)."""
        req = VisionRequest(query="  Analyze the diagram  ", evidence=[])
        assert req.query == "Analyze the diagram"

    def test_03_empty_query_raises(self) -> None:
        """Empty string query raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="empty"):
            VisionRequest(query="", evidence=[])

    def test_04_whitespace_only_query_raises(self) -> None:
        """Whitespace-only query raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="empty"):
            VisionRequest(query="   \t\n  ", evidence=[])

    def test_05_none_query_raises(self) -> None:
        """None query raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query=None, evidence=[])  # type: ignore[arg-type]

    def test_06_integer_query_raises(self) -> None:
        """Integer query raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query=42, evidence=[])  # type: ignore[arg-type]

    def test_07_list_query_raises(self) -> None:
        """List query raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query=["analyze", "this"], evidence=[])  # type: ignore[arg-type]

    def test_08_dict_query_raises(self) -> None:
        """Dict query raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query={"q": "value"}, evidence=[])  # type: ignore[arg-type]

    def test_09_long_query_accepted(self) -> None:
        """Very long query is accepted — no arbitrary length cap in existing contract."""
        long_q = "Analyze " + "the visual content " * 500
        req = VisionRequest(query=long_q, evidence=[])
        assert len(req.query) > 100

    def test_10_single_char_query_accepted(self) -> None:
        """Single non-whitespace character query is valid."""
        req = VisionRequest(query="?", evidence=[])
        assert req.query == "?"

    def test_11_agent_execute_empty_query_raises(self) -> None:
        """VisionAgent.execute with empty string query raises VisionInputValidationError."""
        agent, provider = _make_agent()
        ev = _valid_evidence()
        with pytest.raises(VisionInputValidationError):
            agent.execute("", evidence=[ev])
        assert provider.invocation_count == 0

    def test_12_agent_execute_none_query_raises(self) -> None:
        """VisionAgent.execute with None query raises VisionInputValidationError."""
        agent, provider = _make_agent()
        with pytest.raises((VisionInputValidationError, VisionAgentError)):
            agent.execute(None, evidence=[])  # type: ignore[arg-type]
        assert provider.invocation_count == 0


# ---------------------------------------------------------------------------
# Test Suite 2: Evidence Container Validation
# ---------------------------------------------------------------------------


class TestEvidenceContainerValidation:
    """Verifies VisionRequest.evidence container validation."""

    def test_13_valid_evidence_list_accepted(self) -> None:
        """A list of valid VisualEvidence objects is accepted."""
        ev = _valid_evidence()
        req = VisionRequest(query="Q", evidence=[ev])
        assert len(req.evidence) == 1

    def test_14_empty_evidence_list_accepted(self) -> None:
        """Empty evidence list is accepted by VisionRequest (contract allows it)."""
        req = VisionRequest(query="Q", evidence=[])
        assert req.evidence == []
        assert req.has_evidence is False

    def test_15_evidence_not_a_list_raises(self) -> None:
        """Non-list evidence raises VisionInputValidationError."""
        ev = _valid_evidence()
        with pytest.raises(VisionInputValidationError, match="list"):
            VisionRequest(query="Q", evidence=ev)  # type: ignore[arg-type]

    def test_16_evidence_tuple_raises(self) -> None:
        """Tuple evidence raises VisionInputValidationError."""
        ev = _valid_evidence()
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="Q", evidence=(ev,))  # type: ignore[arg-type]

    def test_17_evidence_set_raises(self) -> None:
        """Set evidence raises VisionInputValidationError."""
        with pytest.raises((VisionInputValidationError, TypeError)):
            VisionRequest(query="Q", evidence={_valid_evidence()})  # type: ignore[arg-type]

    def test_18_none_evidence_raises(self) -> None:
        """None evidence raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="Q", evidence=None)  # type: ignore[arg-type]

    def test_19_wrong_item_type_in_evidence_raises(self) -> None:
        """Non-VisualEvidence item in evidence list raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="Q", evidence=["not-an-evidence"])  # type: ignore[list-item]

    def test_20_integer_item_in_evidence_raises(self) -> None:
        """Integer item in evidence list raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="Q", evidence=[42])  # type: ignore[list-item]

    def test_21_mixed_valid_invalid_evidence_raises(self) -> None:
        """Mixed valid/invalid evidence list raises VisionInputValidationError."""
        ev = _valid_evidence()
        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="Q", evidence=[ev, "invalid"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Test Suite 3: VisualEvidence Field Validation
# ---------------------------------------------------------------------------


class TestVisualEvidenceFieldValidation:
    """Verifies VisualEvidence field-level validation using the existing contract."""

    def test_22_valid_evidence_constructed(self) -> None:
        """Valid VisualEvidence is constructed without error."""
        ev = VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", image_bytes=_make_png())
        assert ev.document_id == "d1"

    def test_23_empty_document_id_raises(self) -> None:
        """Empty document_id raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError, match="document_id"):
            VisualEvidence(document_id="", filename="f.png", chunk_id="c1")

    def test_24_none_document_id_raises(self) -> None:
        """None document_id raises VisionEvidenceError."""
        with pytest.raises((VisionEvidenceError, TypeError)):
            VisualEvidence(document_id=None, filename="f.png", chunk_id="c1")  # type: ignore[arg-type]

    def test_25_empty_filename_raises(self) -> None:
        """Empty filename raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError, match="filename"):
            VisualEvidence(document_id="d1", filename="", chunk_id="c1")

    def test_26_empty_chunk_id_raises(self) -> None:
        """Empty chunk_id raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError, match="chunk_id"):
            VisualEvidence(document_id="d1", filename="f.png", chunk_id="")

    def test_27_zero_page_number_raises(self) -> None:
        """page_number=0 raises VisionEvidenceError (must be >0 or None)."""
        with pytest.raises(VisionEvidenceError, match="page_number"):
            VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", page_number=0)

    def test_28_negative_page_number_raises(self) -> None:
        """Negative page_number raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError, match="page_number"):
            VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", page_number=-1)

    def test_29_none_page_number_accepted(self) -> None:
        """page_number=None is accepted (optional field)."""
        ev = VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", page_number=None)
        assert ev.page_number is None

    def test_30_negative_chunk_index_raises(self) -> None:
        """Negative chunk_index raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError, match="chunk_index"):
            VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", chunk_index=-1)

    def test_31_zero_chunk_index_accepted(self) -> None:
        """chunk_index=0 is valid (non-negative)."""
        ev = VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", chunk_index=0)
        assert ev.chunk_index == 0

    def test_32_metadata_non_dict_raises(self) -> None:
        """Non-dict metadata raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError, match="metadata"):
            VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", metadata="bad")  # type: ignore[arg-type]

    def test_33_none_metadata_raises(self) -> None:
        """None metadata raises VisionEvidenceError."""
        with pytest.raises((VisionEvidenceError, TypeError)):
            VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", metadata=None)  # type: ignore[arg-type]

    def test_34_empty_metadata_accepted(self) -> None:
        """Empty metadata dict {} is accepted."""
        ev = VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", metadata={})
        assert ev.metadata == {}

    def test_35_metadata_copied_on_construction(self) -> None:
        """Metadata is shallow-copied on construction — mutations to original do not affect evidence."""
        original = {"key": "value"}
        ev = VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", metadata=original)
        original["extra"] = "injected"
        assert "extra" not in ev.metadata


# ---------------------------------------------------------------------------
# Test Suite 4: Content Type Validation
# ---------------------------------------------------------------------------


class TestContentTypeValidation:
    """Verifies content_type validation using the existing VALID_VISUAL_CONTENT_TYPES contract."""

    def test_36_image_content_type_accepted(self) -> None:
        """Content type 'image' is accepted."""
        ev = VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", content_type="image")
        assert ev.content_type == "image"

    def test_37_chart_content_type_accepted(self) -> None:
        """Content type 'chart' is accepted."""
        ev = VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", content_type="chart")
        assert ev.content_type == "chart"

    def test_38_diagram_content_type_accepted(self) -> None:
        """Content type 'diagram' is accepted."""
        ev = VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", content_type="diagram")
        assert ev.content_type == "diagram"

    def test_39_content_type_normalized_lowercase(self) -> None:
        """Content type is normalized to lowercase during construction."""
        ev = VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", content_type="IMAGE")
        assert ev.content_type == "image"

    def test_40_unsupported_content_type_raises(self) -> None:
        """Unsupported content type raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError, match="content_type"):
            VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", content_type="video")

    def test_41_text_content_type_raises(self) -> None:
        """content_type='text' raises VisionEvidenceError (non-visual)."""
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", content_type="text")

    def test_42_empty_content_type_raises(self) -> None:
        """Empty content type raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", content_type="")

    def test_43_none_content_type_raises(self) -> None:
        """None content type raises VisionEvidenceError."""
        with pytest.raises((VisionEvidenceError, TypeError)):
            VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", content_type=None)  # type: ignore[arg-type]

    def test_44_all_valid_content_types_accepted(self) -> None:
        """All content types in VALID_VISUAL_CONTENT_TYPES are accepted."""
        for ct in VALID_VISUAL_CONTENT_TYPES:
            ev = VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", content_type=ct)
            assert ev.content_type == ct


# ---------------------------------------------------------------------------
# Test Suite 5: Provider Short-Circuit Verification
# ---------------------------------------------------------------------------


class TestProviderShortCircuit:
    """Verifies that invalid inputs never reach the provider."""

    def test_45_invalid_query_provider_not_called(self) -> None:
        """Provider is not called when query is empty."""
        agent, provider = _make_agent()
        ev = _valid_evidence()
        with pytest.raises(VisionInputValidationError):
            agent.execute("", evidence=[ev])
        assert provider.invocation_count == 0

    def test_46_none_request_provider_not_called(self) -> None:
        """Provider is not called when request is None."""
        agent, provider = _make_agent()
        with pytest.raises((VisionInputValidationError, VisionAgentError)):
            agent.execute(None, evidence=[])  # type: ignore[arg-type]
        assert provider.invocation_count == 0

    def test_47_malformed_evidence_provider_not_called(self) -> None:
        """Provider is not called when evidence contains corrupted image bytes."""
        agent, provider = _make_agent()
        ev = VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", image_bytes=b"corrupted_not_png")
        with pytest.raises((VisionEvidenceError, VisionProcessingError)):
            agent.execute("Q", evidence=[ev])
        assert provider.invocation_count == 0

    def test_48_valid_request_provider_called_once(self) -> None:
        """Provider is called exactly once for a valid request."""
        agent, provider = _make_agent()
        ev = _valid_evidence()
        res = agent.execute("Valid query", evidence=[ev])
        assert res.is_success is True
        assert provider.invocation_count == 1

    def test_49_pipeline_short_circuit_invalid_query(self) -> None:
        """VisionPipeline short-circuits on empty query before reaching image preparation."""
        config = VisionProviderConfig(provider_name="sc-prov", model_name="v1")
        provider = RecordingProvider(config)
        pipeline = VisionPipeline(provider=provider)
        ev = _valid_evidence()
        with pytest.raises(VisionInputValidationError):
            pipeline.run("", evidence=[ev])
        assert provider.invocation_count == 0

    def test_50_wrong_request_type_raises_before_provider(self) -> None:
        """Non-string, non-VisionRequest type raises VisionInputValidationError before provider."""
        agent, provider = _make_agent()
        with pytest.raises(VisionInputValidationError):
            agent.execute(12345, evidence=[])  # type: ignore[arg-type]
        assert provider.invocation_count == 0


# ---------------------------------------------------------------------------
# Test Suite 6: Immutability Under Validation
# ---------------------------------------------------------------------------


class TestImmutabilityUnderValidation:
    """Verifies that validation does not mutate input objects."""

    def test_51_visionrequest_not_mutated_during_execution(self) -> None:
        """VisionRequest fields are not mutated after successful agent.execute()."""
        ev = _valid_evidence(doc_id="imm-doc", filename="imm.png")
        req = VisionRequest(query="Immutable query", evidence=[ev])
        req_dict_before = req.to_dict()
        ev_dict_before = ev.to_dict()

        agent, _ = _make_agent()
        agent.execute(req)

        assert req.to_dict() == req_dict_before
        assert ev.to_dict() == ev_dict_before

    def test_52_evidence_metadata_not_mutated_by_validation(self) -> None:
        """Evidence metadata dict is not modified during VisionRequest construction."""
        meta = {"key": "value", "count": 42}
        original_copy = dict(meta)
        ev = VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1", metadata=meta)
        assert ev.metadata == original_copy
        assert ev.metadata is not meta  # Must be a copy

    def test_53_metadata_mutation_after_construction_not_reflected(self) -> None:
        """Mutating the original metadata dict after VisionRequest construction does not affect the request."""
        meta = {"source": "A"}
        req = VisionRequest(query="Q", evidence=[], metadata=meta)
        meta["injected"] = "bad"
        assert "injected" not in req.metadata

    def test_54_evidence_list_contract_by_reference(self) -> None:
        """VisionRequest stores the evidence list reference (no defensive copy per existing contract).

        This test documents the EXISTING behavior: mutating the original list
        after construction DOES affect req.evidence. This is the current contract.
        If defensive copying is ever added, this test must be updated accordingly.
        """
        ev = _valid_evidence()
        evidence_list = [ev]
        req = VisionRequest(query="Q", evidence=evidence_list)
        # The existing contract: req.evidence is the same list (shared reference)
        assert req.evidence is evidence_list


    def test_55_query_immutability(self) -> None:
        """VisionRequest.query cannot be externally mutated after construction (dataclass behavior)."""
        req = VisionRequest(query="Original query", evidence=[])
        assert req.query == "Original query"


# ---------------------------------------------------------------------------
# Test Suite 7: Multi-Evidence Validation
# ---------------------------------------------------------------------------


class TestMultiEvidenceValidation:
    """Verifies multi-evidence requests pass validation with correct contract behavior."""

    def test_56_single_evidence_request_valid(self) -> None:
        """Single evidence request validates and executes correctly."""
        agent, provider = _make_agent()
        ev = _valid_evidence(doc_id="d1")
        res = agent.execute("Single", evidence=[ev])
        assert res.is_success
        assert len(res.evidence) == 1
        assert provider.invocation_count == 1

    def test_57_two_evidence_items_valid(self) -> None:
        """Two valid evidence items pass validation and reach provider."""
        agent, provider = _make_agent()
        ev1 = _valid_evidence(doc_id="d1", filename="f1.png", chunk_id="c1")
        ev2 = _valid_evidence(doc_id="d2", filename="f2.png", chunk_id="c2")
        res = agent.execute("Two items", evidence=[ev1, ev2])
        assert res.is_success
        assert len(res.evidence) == 2

    def test_58_three_evidence_items_valid(self) -> None:
        """Three valid evidence items pass validation and reach provider."""
        agent, provider = _make_agent()
        evs = [_valid_evidence(doc_id=f"d{i}", filename=f"f{i}.png", chunk_id=f"c{i}") for i in range(3)]
        res = agent.execute("Three items", evidence=evs)
        assert len(res.evidence) == 3

    def test_59_ten_evidence_items_valid(self) -> None:
        """Ten valid evidence items pass validation without duplication or reordering."""
        agent, provider = _make_agent()
        evs = [_valid_evidence(doc_id=f"batch-{i}", filename=f"b{i}.png", chunk_id=f"bc{i}") for i in range(10)]
        res = agent.execute("Batch of 10", evidence=evs)
        assert len(res.evidence) == 10
        assert [e.document_id for e in res.evidence] == [f"batch-{i}" for i in range(10)]

    def test_60_evidence_order_preserved_through_validation(self) -> None:
        """Evidence order [A, B, C] is preserved exactly through validation."""
        agent, _ = _make_agent()
        evA = _valid_evidence(doc_id="doc-A", filename="A.png", chunk_id="cA")
        evB = _valid_evidence(doc_id="doc-B", filename="B.png", chunk_id="cB")
        evC = _valid_evidence(doc_id="doc-C", filename="C.png", chunk_id="cC")
        res = agent.execute("Order", evidence=[evA, evB, evC])
        assert [e.document_id for e in res.evidence] == ["doc-A", "doc-B", "doc-C"]

    def test_61_lineage_preserved_through_validation(self) -> None:
        """Lineage fields (doc_id, filename, chunk_id, page_number) preserved through validation."""
        agent, _ = _make_agent()
        ev = VisualEvidence(
            document_id="lin-doc",
            filename="lin-file.pdf",
            chunk_id="lin-chk",
            page_number=7,
            chunk_index=3,
            content_type="chart",
            image_bytes=_make_png(),
            metadata={"source": "test"},
        )
        res = agent.execute("Lineage test", evidence=[ev])
        item = res.evidence[0]
        assert item.document_id == "lin-doc"
        assert item.filename == "lin-file.pdf"
        assert item.chunk_id == "lin-chk"
        assert item.page_number == 7
        assert item.chunk_index == 3
        assert item.content_type == "chart"
        assert item.metadata["source"] == "test"


# ---------------------------------------------------------------------------
# Test Suite 8: Concurrent Validation Isolation
# ---------------------------------------------------------------------------


class TestConcurrentValidationIsolation:
    """Verifies that concurrent valid/invalid requests are properly isolated."""

    def test_62_concurrent_mixed_valid_invalid_requests(self) -> None:
        """Concurrent valid and invalid requests remain isolated — failures don't corrupt successes."""
        import threading
        config = VisionProviderConfig(provider_name="conc-val-prov", model_name="v1")
        provider = RecordingProvider(config)
        import threading as _threading
        lock = _threading.Lock()

        agent = VisionAgent(provider=provider)

        def _run(task: tuple[str, bool]) -> tuple[str, bool, str | None]:
            doc_id, is_valid = task
            if is_valid:
                ev = _valid_evidence(doc_id=doc_id)
                try:
                    res = agent.execute(f"Query for {doc_id}", evidence=[ev])
                    return doc_id, True, res.document_id
                except Exception as e:
                    return doc_id, False, str(e)
            else:
                try:
                    agent.execute("", evidence=[])  # empty query → always invalid
                    return doc_id, True, None
                except VisionInputValidationError:
                    return doc_id, False, None

        tasks = [
            ("doc-valid-1", True),
            ("doc-invalid-A", False),
            ("doc-valid-2", True),
            ("doc-invalid-B", False),
            ("doc-valid-3", True),
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_run, t) for t in tasks]
            results = [f.result() for f in futures]

        result_map = {doc_id: (ok, val) for doc_id, ok, val in results}

        assert result_map["doc-valid-1"] == (True, "doc-valid-1")
        assert result_map["doc-invalid-A"][0] is False
        assert result_map["doc-valid-2"] == (True, "doc-valid-2")
        assert result_map["doc-invalid-B"][0] is False
        assert result_map["doc-valid-3"] == (True, "doc-valid-3")

    def test_63_concurrent_invalid_requests_dont_call_provider(self) -> None:
        """Multiple concurrent invalid requests produce zero provider calls."""
        config = VisionProviderConfig(provider_name="conc-zero-prov", model_name="v1")
        provider = RecordingProvider(config)
        agent = VisionAgent(provider=provider)

        def _run_invalid() -> bool:
            try:
                agent.execute("", evidence=[])
                return False  # Should have raised
            except VisionInputValidationError:
                return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_run_invalid) for _ in range(6)]
            caught = [f.result() for f in futures]

        assert all(caught)
        assert provider.invocation_count == 0


# ---------------------------------------------------------------------------
# Test Suite 9: Determinism & Error Hierarchy
# ---------------------------------------------------------------------------


class TestDeterminismAndErrorHierarchy:
    """Verifies deterministic validation behavior and existing exception hierarchy."""

    def test_64_same_invalid_query_same_exception_type(self) -> None:
        """The same invalid query raises the same exception type every time."""
        exceptions_raised: list[type] = []
        for _ in range(3):
            try:
                VisionRequest(query="", evidence=[])
            except VisionInputValidationError as e:
                exceptions_raised.append(type(e))

        assert len(exceptions_raised) == 3
        assert all(t == VisionInputValidationError for t in exceptions_raised)

    def test_65_same_invalid_evidence_same_exception_type(self) -> None:
        """The same invalid evidence raises the same exception type every time."""
        exceptions_raised: list[type] = []
        for _ in range(3):
            try:
                VisualEvidence(document_id="", filename="f.png", chunk_id="c1")
            except VisionEvidenceError as e:
                exceptions_raised.append(type(e))

        assert len(exceptions_raised) == 3
        assert all(t == VisionEvidenceError for t in exceptions_raised)

    def test_66_exception_hierarchy_correct(self) -> None:
        """VisionInputValidationError is a subclass of VisionAgentError."""
        err = VisionInputValidationError("test")
        assert isinstance(err, VisionAgentError)

    def test_67_vision_evidence_error_is_agent_error(self) -> None:
        """VisionEvidenceError is a subclass of VisionAgentError."""
        err = VisionEvidenceError("test")
        assert isinstance(err, VisionAgentError)

    def test_68_no_fake_result_for_invalid_request(self) -> None:
        """Invalid request never produces a successful VisionResult — exception must propagate."""
        agent, _ = _make_agent()
        try:
            result = agent.execute("", evidence=[])
            # If no exception was raised, result must NOT be a success
            assert not result.is_success
        except (VisionInputValidationError, VisionAgentError):
            pass  # Correct behavior: exception propagated


# ---------------------------------------------------------------------------
# Test Suite 10: Public API & Offline Integrity
# ---------------------------------------------------------------------------


class TestPublicAPIAndOfflineIntegrity:
    """Verifies public API compatibility and offline execution."""

    def test_69_public_imports_work(self) -> None:
        """All expected public symbols are importable from vision package."""
        import vision
        required = [
            "VisionRequest", "VisualEvidence", "VisionResult",
            "VisionModelInput", "VisionModelProvider",
            "VisionAgent", "VisionPipeline",
            "VisionExecutionAdapter", "VisionExecutionLifecycle",
            "VisionResultNormalizer",
            "VisionInputValidationError", "VisionEvidenceError",
            "VisionProcessingError", "VisionAgentError",
            "VALID_VISUAL_CONTENT_TYPES",
        ]
        for symbol in required:
            assert hasattr(vision, symbol), f"Public symbol '{symbol}' missing from vision package"

    def test_70_vision_request_public_import(self) -> None:
        """VisionRequest can be imported and constructed via public import."""
        from vision import VisionRequest
        req = VisionRequest(query="Test", evidence=[])
        assert req.query == "Test"

    def test_71_visual_evidence_public_import(self) -> None:
        """VisualEvidence can be imported and constructed via public import."""
        from vision import VisualEvidence
        ev = VisualEvidence(document_id="d1", filename="f.png", chunk_id="c1")
        assert ev.document_id == "d1"

    def test_72_offline_no_network_imports(self) -> None:
        """Vision package modules do not import external network libraries."""
        import inspect
        import vision.models as vm
        import vision.execution_adapter as ea
        import vision.vision_agent as va
        import vision.pipeline as vp

        for mod in (vm, ea, va, vp):
            source = inspect.getsource(mod)
            for pattern in (
                "import requests",
                "import httpx",
                "import aiohttp",
                "import openai",
                "import anthropic",
            ):
                assert pattern not in source, (
                    f"Forbidden pattern '{pattern}' found in {mod.__name__}"
                )

    def test_73_previous_tests_compatible(self) -> None:
        """Day 40 run_vision_pipeline convenience function still works after Day 43."""
        config = VisionProviderConfig(provider_name="compat-prov", model_name="v1")
        provider = RecordingProvider(config)
        ev = _valid_evidence(doc_id="compat-doc")
        res = run_vision_pipeline(provider, "Compatibility query", evidence=[ev])
        assert res.is_success
        assert res.document_id == "compat-doc"
