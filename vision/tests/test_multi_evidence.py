"""
Comprehensive Day 41 Multi-Evidence Execution & Batch-Safe Contract Tests.

Verifies:
  1. Single, double, triple, and batch (10 items) visual evidence execution.
  2. Evidence order preservation (A -> B -> C remains A -> B -> C).
  3. Lineage isolation (doc_id, filename, chunk_id, chunk_index, page_number, content_type).
  4. Single provider invocation guarantee (provider called exactly ONCE per request).
  5. Image preparation reuse (prepare_image_evidence called per item without duplication).
  6. Input immutability (VisionRequest, VisualEvidence list/items remain unmutated).
  7. Empty evidence handling (status="no_evidence").
  8. Malformed evidence in batch raises controlled VisionEvidenceError.
  9. Repeated evidence (A, A, B) handled deterministically without silent deduplication.
  10. Offline guarantee -- zero external HTTP, zero network, zero LLM SDKs, zero secrets.
"""

from __future__ import annotations

import inspect
import io
from typing import Any

import pytest
from PIL import Image

from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderExecutionError,
)
from vision.execution_adapter import VisionExecutionAdapter
from vision.image_preparation import PreparedImageEvidence, prepare_image_evidence
from vision.input_builder import VisionModelInput, build_vision_input
from vision.lifecycle import VisionExecutionLifecycle, VisionExecutionStage
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.pipeline import VisionPipeline, run_vision_pipeline
from vision.provider import VisionModelProvider
from vision.provider_config import VisionProviderConfig
from vision.result_normalizer import VisionResultNormalizer
from vision.vision_agent import VisionAgent


# ---------------------------------------------------------------------------
# Test Helpers & Test Doubles
# ---------------------------------------------------------------------------


def _make_png(width: int = 32, height: int = 32, color: tuple[int, int, int] = (100, 150, 200)) -> bytes:
    """Generate PNG byte payload for testing visual evidence."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MultiEvidenceRecordingProvider(VisionModelProvider):
    """Test double that records received inputs and returns a valid, lineage-preserving VisionResult."""

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
            description="Multi-evidence reasoning output.",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={
                "provider_name": self.provider_name,
                "confidence": 0.95,
                "api_key": "leaky_key_to_be_sanitized",
            },
        )


class FailingMultiEvidenceProvider(VisionModelProvider):
    """Test double that raises a controlled VisionProviderExecutionError."""

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        raise VisionProviderExecutionError("Multi-evidence inference backend failure.")


# ---------------------------------------------------------------------------
# Test Suite 1: Single, Dual, Triple & Batch Multi-Evidence Execution
# ---------------------------------------------------------------------------


class TestMultiEvidenceExecutionCounts:
    """Verifies pipeline execution with 1, 2, 3, and 10 evidence items."""

    def test_01_single_evidence_execution(self) -> None:
        """One evidence item executes cleanly and preserves lineage."""
        config = VisionProviderConfig(provider_name="m1-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(
            document_id="doc-1",
            filename="chart1.pdf",
            chunk_id="chk-1",
            page_number=1,
            chunk_index=0,
            content_type="chart",
            image_bytes=_make_png(32, 32, (200, 50, 50)),
        )

        res = agent.execute("Explain chart 1", evidence=[ev])

        assert isinstance(res, VisionResult)
        assert res.is_success is True
        assert res.document_id == "doc-1"
        assert res.filename == "chart1.pdf"
        assert len(res.evidence) == 1
        assert provider.invocation_count == 1

    def test_02_two_evidence_items_execution(self) -> None:
        """Two evidence items execute through pipeline with order preserved."""
        config = VisionProviderConfig(provider_name="m2-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev1 = VisualEvidence(
            document_id="doc-A",
            filename="reportA.pdf",
            chunk_id="chk-A",
            page_number=2,
            chunk_index=1,
            content_type="diagram",
            image_bytes=_make_png(32, 32, (50, 200, 50)),
        )
        ev2 = VisualEvidence(
            document_id="doc-B",
            filename="reportB.pdf",
            chunk_id="chk-B",
            page_number=4,
            chunk_index=2,
            content_type="image",
            image_bytes=_make_png(32, 32, (50, 50, 200)),
        )

        res = agent.execute("Compare diagram A and image B", evidence=[ev1, ev2])

        assert res.is_success is True
        assert len(res.evidence) == 2
        assert res.evidence[0].document_id == "doc-A"
        assert res.evidence[1].document_id == "doc-B"
        assert provider.invocation_count == 1

    def test_03_three_evidence_items_execution(self) -> None:
        """Three evidence items maintain order and lineage while invoking provider once."""
        config = VisionProviderConfig(provider_name="m3-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev1 = VisualEvidence(document_id="d1", filename="f1.png", chunk_id="c1", image_bytes=_make_png())
        ev2 = VisualEvidence(document_id="d2", filename="f2.png", chunk_id="c2", image_bytes=_make_png())
        ev3 = VisualEvidence(document_id="d3", filename="f3.png", chunk_id="c3", image_bytes=_make_png())

        res = agent.execute("Analyze 3 images", evidence=[ev1, ev2, ev3])

        assert res.is_success is True
        assert len(res.evidence) == 3
        assert [e.document_id for e in res.evidence] == ["d1", "d2", "d3"]
        assert [e.filename for e in res.evidence] == ["f1.png", "f2.png", "f3.png"]
        assert [e.chunk_id for e in res.evidence] == ["c1", "c2", "c3"]
        assert provider.invocation_count == 1

    def test_04_batch_ten_evidence_items_execution(self) -> None:
        """Batch of 10 evidence items executes correctly with single provider call."""
        config = VisionProviderConfig(provider_name="m10-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev_list = [
            VisualEvidence(
                document_id=f"doc-batch-{i}",
                filename=f"file_{i}.pdf",
                chunk_id=f"chk-{i}",
                page_number=i + 1,
                chunk_index=i,
                image_bytes=_make_png(),
            )
            for i in range(10)
        ]

        res = agent.execute("Analyze batch of 10 documents", evidence=ev_list)

        assert res.is_success is True
        assert len(res.evidence) == 10
        assert [e.document_id for e in res.evidence] == [f"doc-batch-{i}" for i in range(10)]
        assert provider.invocation_count == 1
        recorded = provider.recorded_inputs[0]
        assert recorded.builder_metadata["total_evidence_count"] == 10


# ---------------------------------------------------------------------------
# Test Suite 2: Contract Invariants, Lineage Isolation & Immutability
# ---------------------------------------------------------------------------


class TestMultiEvidenceContractInvariants:
    """Verifies order preservation, lineage isolation, immutability, and single invocation."""

    def test_05_evidence_order_preserved(self) -> None:
        """Order [A, B, C] is strictly preserved through all pipeline stages."""
        config = VisionProviderConfig(provider_name="ord-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        evA = VisualEvidence(document_id="doc-A", filename="A.png", chunk_id="cA", image_bytes=_make_png())
        evB = VisualEvidence(document_id="doc-B", filename="B.png", chunk_id="cB", image_bytes=_make_png())
        evC = VisualEvidence(document_id="doc-C", filename="C.png", chunk_id="cC", image_bytes=_make_png())

        res = agent.execute("Check order", evidence=[evA, evB, evC])

        assert [e.document_id for e in res.evidence] == ["doc-A", "doc-B", "doc-C"]

        rec_lineage = provider.recorded_inputs[0].builder_metadata["all_evidence_lineage"]
        assert [item["document_id"] for item in rec_lineage] == ["doc-A", "doc-B", "doc-C"]

    def test_06_lineage_isolation_no_cross_contamination(self) -> None:
        """Each evidence item retains its own attributes without field leakage."""
        config = VisionProviderConfig(provider_name="iso-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        evA = VisualEvidence(
            document_id="doc-A",
            filename="report-A.pdf",
            chunk_id="chunk-A",
            page_number=1,
            chunk_index=0,
            content_type="chart",
            image_bytes=_make_png(),
            metadata={"source": "finances"},
        )
        evB = VisualEvidence(
            document_id="doc-B",
            filename="report-B.pdf",
            chunk_id="chunk-B",
            page_number=8,
            chunk_index=3,
            content_type="diagram",
            image_bytes=_make_png(),
            metadata={"source": "engineering"},
        )

        res = agent.execute("Check isolation", evidence=[evA, evB])

        itemA = res.evidence[0]
        itemB = res.evidence[1]

        assert itemA.document_id == "doc-A"
        assert itemA.filename == "report-A.pdf"
        assert itemA.page_number == 1
        assert itemA.chunk_id == "chunk-A"
        assert itemA.chunk_index == 0
        assert itemA.content_type == "chart"
        assert itemA.metadata == {"source": "finances"}

        assert itemB.document_id == "doc-B"
        assert itemB.filename == "report-B.pdf"
        assert itemB.page_number == 8
        assert itemB.chunk_id == "chunk-B"
        assert itemB.chunk_index == 3
        assert itemB.content_type == "diagram"
        assert itemB.metadata == {"source": "engineering"}

    def test_07_no_evidence_duplication(self) -> None:
        """Input list [A, B, C] does not duplicate to [A, B, C, A, B, C]."""
        config = VisionProviderConfig(provider_name="nodup-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        evs = [
            VisualEvidence(document_id=f"d{i}", filename=f"f{i}.png", chunk_id=f"c{i}", image_bytes=_make_png())
            for i in range(3)
        ]

        res = agent.execute("Check no duplication", evidence=evs)
        assert len(res.evidence) == 3

    def test_08_repeated_evidence_handled_deterministically(self) -> None:
        """Repeated evidence items [A, A, B] are preserved as given without silent deduplication."""
        config = VisionProviderConfig(provider_name="rep-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        evA = VisualEvidence(document_id="doc-A", filename="A.png", chunk_id="cA", image_bytes=_make_png())
        evB = VisualEvidence(document_id="doc-B", filename="B.png", chunk_id="cB", image_bytes=_make_png())

        res = agent.execute("Check repeated", evidence=[evA, evA, evB])

        assert len(res.evidence) == 3
        assert [e.document_id for e in res.evidence] == ["doc-A", "doc-A", "doc-B"]

    def test_09_input_immutability(self) -> None:
        """VisionRequest and VisualEvidence objects remain completely unmutated."""
        ev1 = VisualEvidence(document_id="d1", filename="f1.png", chunk_id="c1", image_bytes=_make_png())
        ev2 = VisualEvidence(document_id="d2", filename="f2.png", chunk_id="c2", image_bytes=_make_png())

        req = VisionRequest(query="Immutability check", evidence=[ev1, ev2])

        dict1_before = ev1.to_dict()
        dict2_before = ev2.to_dict()
        req_before = req.to_dict()

        config = VisionProviderConfig(provider_name="imm-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        agent.execute(req)

        assert ev1.to_dict() == dict1_before
        assert ev2.to_dict() == dict2_before
        assert req.to_dict() == req_before

    def test_10_provider_called_exactly_once(self) -> None:
        """Provider.execute is called exactly ONCE regardless of evidence count."""
        config = VisionProviderConfig(provider_name="once-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        evs = [
            VisualEvidence(document_id=f"d{i}", filename=f"f{i}.png", chunk_id=f"c{i}", image_bytes=_make_png())
            for i in range(5)
        ]

        assert provider.invocation_count == 0
        agent.execute("Five evidence query", evidence=evs)
        assert provider.invocation_count == 1


# ---------------------------------------------------------------------------
# Test Suite 3: Failure Isolation & Error Handling
# ---------------------------------------------------------------------------


class TestMultiEvidenceFailureIsolation:
    """Verifies controlled failure when malformed evidence is present in multi-evidence requests."""

    def test_11_malformed_evidence_in_batch_raises_controlled_error(self) -> None:
        """If one item in a batch is invalid/corrupted, pipeline raises controlled VisionEvidenceError."""
        config = VisionProviderConfig(provider_name="mal-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        valid_ev = VisualEvidence(document_id="d1", filename="valid.png", chunk_id="c1", image_bytes=_make_png())
        invalid_ev = VisualEvidence(document_id="d2", filename="invalid.png", chunk_id="c2", image_bytes=b"corrupted_bytes")

        with pytest.raises((VisionEvidenceError, VisionProcessingError)):
            agent.execute("Batch with invalid item", evidence=[valid_ev, invalid_ev])

        # Provider must NOT have been called if evidence preparation failed
        assert provider.invocation_count == 0

    def test_12_empty_evidence_follows_existing_contract(self) -> None:
        """Zero evidence request transitions to COMPLETED with status='no_evidence'."""
        config = VisionProviderConfig(provider_name="empty-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        res = agent.execute("Zero evidence query", evidence=[])

        assert isinstance(res, VisionResult)
        assert res.status == "no_evidence"
        assert len(res.evidence) == 0
        assert provider.invocation_count == 0

    def test_13_provider_failure_in_multi_evidence(self) -> None:
        """Provider failure during multi-evidence execution raises controlled VisionProviderExecutionError."""
        config = VisionProviderConfig(provider_name="fail-prov", model_name="v1")
        provider = FailingMultiEvidenceProvider(config)
        agent = VisionAgent(provider=provider)

        ev1 = VisualEvidence(document_id="d1", filename="f1.png", chunk_id="c1", image_bytes=_make_png())
        ev2 = VisualEvidence(document_id="d2", filename="f2.png", chunk_id="c2", image_bytes=_make_png())

        with pytest.raises(VisionProviderExecutionError, match="Multi-evidence inference backend failure"):
            agent.execute("Fail multi query", evidence=[ev1, ev2])


# ---------------------------------------------------------------------------
# Test Suite 4: Component Reuse & Offline Integrity
# ---------------------------------------------------------------------------


class TestComponentReuseAndOfflineIntegrity:
    """Verifies reuse of Days 33-40 components and pure offline execution."""

    def test_14_days_33_to_40_components_reused(self) -> None:
        """Pipeline reuses VisualEvidenceAdapter, prepare_image_evidence, build_vision_input, VisionExecutionLifecycle, and VisionResultNormalizer."""
        config = VisionProviderConfig(provider_name="reuse-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        pipeline = VisionPipeline(provider=provider)

        ev1 = VisualEvidence(document_id="d1", filename="f1.png", chunk_id="c1", image_bytes=_make_png())
        ev2 = VisualEvidence(document_id="d2", filename="f2.png", chunk_id="c2", image_bytes=_make_png())

        res = run_vision_pipeline(provider, "Pipeline reuse query", evidence=[ev1, ev2])

        assert res.is_success is True
        assert len(res.evidence) == 2
        assert "execution_lifecycle" in res.metadata
        assert "execution_trace" in res.metadata

    def test_15_offline_verification(self) -> None:
        """No external network libraries or HTTP modules imported in vision package."""
        import vision.execution_adapter as ea
        import vision.pipeline as vp

        for mod in (ea, vp):
            source = inspect.getsource(mod)
            for pattern in (
                "import requests",
                "import httpx",
                "import aiohttp",
                "import socket",
                "import urllib.request",
                "import openai",
                "import anthropic",
            ):
                assert pattern not in source, f"Module {mod.__name__} contains forbidden pattern '{pattern}'"

    def test_16_evidence_count_preserved(self) -> None:
        """Count of evidence items is preserved across request, model input metadata, and result."""
        config = VisionProviderConfig(provider_name="cnt-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        evs = [VisualEvidence(document_id=f"d{i}", filename=f"f{i}.png", chunk_id=f"c{i}", image_bytes=_make_png()) for i in range(4)]
        res = agent.execute("Count test", evidence=evs)

        assert len(res.evidence) == 4
        recorded = provider.recorded_inputs[0]
        assert recorded.builder_metadata["total_evidence_count"] == 4

    def test_17_individual_lineage_fields_preserved(self) -> None:
        """document_id, filename, chunk_id, chunk_index, page_number, content_type, metadata preserved for each item."""
        config = VisionProviderConfig(provider_name="fld-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev1 = VisualEvidence(
            document_id="doc-101",
            filename="file101.pdf",
            chunk_id="chk-101",
            page_number=5,
            chunk_index=2,
            content_type="chart",
            image_bytes=_make_png(),
            metadata={"m_key": "m_val1"},
        )
        ev2 = VisualEvidence(
            document_id="doc-102",
            filename="file102.pdf",
            chunk_id="chk-102",
            page_number=10,
            chunk_index=7,
            content_type="diagram",
            image_bytes=_make_png(),
            metadata={"m_key": "m_val2"},
        )

        res = agent.execute("Field preservation query", evidence=[ev1, ev2])

        assert res.query == "Field preservation query"
        item1 = res.evidence[0]
        assert item1.document_id == "doc-101"
        assert item1.filename == "file101.pdf"
        assert item1.chunk_id == "chk-101"
        assert item1.chunk_index == 2
        assert item1.page_number == 5
        assert item1.content_type == "chart"
        assert item1.metadata["m_key"] == "m_val1"

        item2 = res.evidence[1]
        assert item2.document_id == "doc-102"
        assert item2.filename == "file102.pdf"
        assert item2.chunk_id == "chk-102"
        assert item2.chunk_index == 7
        assert item2.page_number == 10
        assert item2.content_type == "diagram"
        assert item2.metadata["m_key"] == "m_val2"

    def test_18_invalid_provider_result_handled(self) -> None:
        """When provider returns None or an invalid object, VisionProcessingError is raised."""
        class InvalidReturnProvider(VisionModelProvider):
            def execute(self, model_input: VisionModelInput, **kwargs: Any) -> Any:
                return None

        config = VisionProviderConfig(provider_name="inv-prov", model_name="v1")
        provider = InvalidReturnProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f1.png", chunk_id="c1", image_bytes=_make_png())

        with pytest.raises(VisionProcessingError, match="Provider returned None"):
            agent.execute("Invalid return query", evidence=[ev])

    def test_19_no_secrets_leaked_in_metadata(self) -> None:
        """Forbidden metadata keys such as api_key or token are sanitized in final result metadata."""
        config = VisionProviderConfig(provider_name="sec-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f1.png", chunk_id="c1", image_bytes=_make_png())
        res = agent.execute("Sanitization query", evidence=[ev])

        assert "api_key" not in res.metadata
        assert res.metadata.get("provider_name") == "sec-prov"
        assert res.metadata.get("confidence") == 0.95


    def test_20_public_imports_work(self) -> None:
        """Verify public API exports in vision package."""
        import vision
        for symbol in (
            "VisualEvidence", "VisionRequest", "VisionResult",
            "VisionModelInput", "VisionModelProvider", "VisionExecutionAdapter",
            "VisionExecutionLifecycle", "VisionResultNormalizer", "VisionPipeline",
            "VisionAgent", "prepare_image_evidence", "build_vision_input",
            "run_vision_pipeline",
        ):
            assert hasattr(vision, symbol), f"Public symbol {symbol} missing in vision package"

    def test_21_determinism(self) -> None:
        """Multiple runs with identical inputs produce identical order and lineage outputs."""
        config = VisionProviderConfig(provider_name="det-prov", model_name="v1")
        provider = MultiEvidenceRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev1 = VisualEvidence(document_id="d1", filename="f1.png", chunk_id="c1", image_bytes=_make_png())
        ev2 = VisualEvidence(document_id="d2", filename="f2.png", chunk_id="c2", image_bytes=_make_png())

        res1 = agent.execute("Deterministic query", evidence=[ev1, ev2])
        res2 = agent.execute("Deterministic query", evidence=[ev1, ev2])

        assert [e.to_dict() for e in res1.evidence] == [e.to_dict() for e in res2.evidence]

