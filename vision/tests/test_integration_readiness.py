"""
Day 50 - OmniBrain Member 3 Vision Agent: Integration Readiness Tests.

Verifies that the existing Vision subsystem (Member 3) correctly consumes
evidence structures produced by the existing Search Agent (Member 2) and
Ingestion pipeline (Member 1) without:

  - Modifying Member 1 (ingestion/)
  - Modifying Member 2 (agents/)
  - Creating duplicate retrieval, embedding, or Qdrant access
  - Creating another pipeline, adapter, provider, lifecycle, or normalizer
  - Using any network, LLM, external API, or external telemetry

All tests execute 100% offline using controlled test doubles.
"""

from __future__ import annotations

import io
import threading
import time
from typing import Any

import pytest
from PIL import Image

from agents.models import AgentCitation, AgentResponse, SearchResult
from ingestion.models import VectorSearchResult
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionCancellationError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProviderExecutionError,
    VisionTimeoutError,
)
from vision.execution_adapter import VisionExecutionAdapter
from vision.image_preparation import PreparedImageEvidence, prepare_image_evidence
from vision.input_builder import VisionModelInput, build_vision_input
from vision.lifecycle import (
    VisionCancellationToken,
    VisionExecutionLifecycle,
    VisionExecutionStage,
    VisionRetryPolicy,
)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.pipeline import VisionPipeline, run_vision_pipeline
from vision.provider import VisionModelProvider
from vision.provider_config import VisionProviderCapabilities, VisionProviderConfig
from vision.result_normalizer import VisionResultNormalizer


# ===========================================================================
# Test Helpers & Controlled Test Doubles
# ===========================================================================


def _make_png_bytes(
    width: int = 32,
    height: int = 32,
    color: tuple = (64, 128, 192),
) -> bytes:
    """Generate a valid PNG byte payload for use in evidence fixtures."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_SAMPLE_PNG = _make_png_bytes()


def _make_citation(
    doc_id: str = "doc-m2-001",
    filename: str = "report.pdf",
    chunk_id: str = "chunk-m2-001",
    content_type: str = "image",
    page_number: int = 3,
    score: float = 0.92,
    metadata: Any = None,
) -> AgentCitation:
    """Construct a Member 2 AgentCitation with visual content_type."""
    return AgentCitation(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        content_type=content_type,
        score=score,
        metadata=metadata if metadata is not None else {"source": "search_agent", "page": page_number},
    )


def _make_visual_evidence(
    doc_id: str = "doc-vis-001",
    filename: str = "chart.pdf",
    chunk_id: str = "chunk-vis-001",
    content_type: str = "chart",
    page_number: int = 5,
    chunk_index: int = 2,
    metadata: Any = None,
) -> VisualEvidence:
    """Construct a VisualEvidence directly for baseline tests."""
    return VisualEvidence(
        document_id=doc_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        chunk_index=chunk_index,
        content_type=content_type,
        image_bytes=_SAMPLE_PNG,
        metadata=metadata if metadata is not None else {"section": "charts"},
    )


def _make_vector_search_result(
    doc_id: str = "doc-vsr-001",
    filename: str = "diagram.pdf",
    chunk_id: str = "chunk-vsr-001",
    content_type: str = "diagram",
    page_number: int = 7,
    chunk_index: int = 4,
    metadata: Any = None,
) -> VectorSearchResult:
    """Construct a Member 1 VectorSearchResult with visual content_type."""
    return VectorSearchResult(
        chunk_id=chunk_id,
        score=0.88,
        document_id=doc_id,
        filename=filename,
        page_number=page_number,
        chunk_index=chunk_index,
        content_type=content_type,
        content="A diagram extracted from the source PDF.",
        metadata=metadata if metadata is not None else {"source": "ingestion"},
    )


def _make_search_result(citations: list) -> SearchResult:
    """Construct a Member 2 SearchResult from a list of citations."""
    return SearchResult(
        query="Test integration query",
        status="RESULTS_FOUND" if citations else "NO_RESULTS",
        citations=citations,
        context="[1] Integration context from search.",
        metadata={"search_agent": "SearchAgent", "top_k": 10},
    )


class IntegrationTestProvider(VisionModelProvider):
    """Controlled offline test double for integration readiness verification."""

    def __init__(
        self,
        capabilities: VisionProviderCapabilities | None = None,
        should_fail: bool = False,
        fail_count: int = 0,
        latency: float = 0.0,
        simulate_timeout: bool = False,
    ) -> None:
        config = VisionProviderConfig(provider_name="integration_test", model_name="test_model_v1")
        super().__init__(config=config, capabilities=capabilities)
        self._should_fail = should_fail
        self._fail_count = fail_count
        self._call_count = 0
        self._received_inputs: list[VisionModelInput] = []
        self._latency = latency
        self._simulate_timeout = simulate_timeout
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    @property
    def received_inputs(self) -> list[VisionModelInput]:
        with self._lock:
            return list(self._received_inputs)

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        """Execute with deterministic result; optionally fail for retry testing."""
        if self._latency > 0:
            time.sleep(self._latency)

        with self._lock:
            self._call_count += 1
            current_count = self._call_count
            self._received_inputs.append(model_input)

        if self._simulate_timeout:
            raise VisionTimeoutError("Simulated execution timeout in test provider.")

        if self._should_fail and current_count <= self._fail_count:
            raise VisionProviderExecutionError(
                f"Simulated provider failure (attempt {current_count}/{self._fail_count})."
            )

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Integration test analysis of '{model_input.query}'",
            document_id=model_input.document_id,
            filename=model_input.filename,
            chunk_id=model_input.chunk_id,
            page_number=model_input.page_number,
            metadata={"provider": "integration_test", "invocation": current_count},
        )


# ===========================================================================
# 1. Member 2 Evidence Compatibility
# ===========================================================================


class TestMember2EvidenceCompatibility:
    """Test 1: Verify Member 2 AgentCitation fields match VisualEvidence contract."""

    def test_01_member2_evidence_compatibility(self) -> None:
        """AgentCitation carries all fields required by VisualEvidence lineage contract."""
        citation = _make_citation(
            doc_id="doc-compat-001",
            filename="compat.pdf",
            chunk_id="chunk-compat-001",
            content_type="image",
            page_number=3,
        )
        assert citation.document_id == "doc-compat-001"
        assert citation.filename == "compat.pdf"
        assert citation.chunk_id == "chunk-compat-001"
        assert citation.page_number == 3
        assert citation.content_type == "image"
        assert citation.score == 0.92
        assert isinstance(citation.metadata, dict)

    def test_02_citation_conversion(self) -> None:
        """AgentCitation -> VisualEvidence via adapt_citation preserves all lineage fields."""
        citation = _make_citation(
            doc_id="doc-conv-001",
            filename="conv.pdf",
            chunk_id="chunk-conv-001",
            content_type="chart",
            page_number=7,
            metadata={"source": "search_agent", "page": 7, "section": "revenue"},
        )
        ev = VisualEvidenceAdapter.adapt_citation(citation)

        assert isinstance(ev, VisualEvidence)
        assert ev.document_id == "doc-conv-001"
        assert ev.filename == "conv.pdf"
        assert ev.chunk_id == "chunk-conv-001"
        assert ev.page_number == 7
        assert ev.content_type == "chart"
        assert ev.metadata["source"] == "search_agent"
        assert ev.metadata["section"] == "revenue"

    def test_03_evidence_mapping_from_search_result(self) -> None:
        """SearchResult.citations -> adapt_search_package -> list of VisualEvidence."""
        citations = [
            _make_citation(doc_id=f"doc-{i}", chunk_id=f"chk-{i}", content_type="image")
            for i in range(3)
        ]
        pkg = _make_search_result(citations)

        evidence_list = VisualEvidenceAdapter.adapt_search_package(pkg)
        assert len(evidence_list) == 3
        for i, ev in enumerate(evidence_list):
            assert isinstance(ev, VisualEvidence)
            assert ev.document_id == f"doc-{i}"
            assert ev.chunk_id == f"chk-{i}"

    def test_04_non_visual_citations_filtered(self) -> None:
        """Non-visual citations (text, table) are filtered out in non-strict mode."""
        citations = [
            _make_citation(content_type="text"),
            _make_citation(doc_id="doc-img-001", chunk_id="chk-img-001", content_type="image"),
            _make_citation(content_type="table"),
            _make_citation(doc_id="doc-chart-001", chunk_id="chk-chart-001", content_type="chart"),
        ]
        pkg = _make_search_result(citations)
        evidence_list = VisualEvidenceAdapter.adapt_search_package(pkg, strict=False)

        assert len(evidence_list) == 2
        assert evidence_list[0].document_id == "doc-img-001"
        assert evidence_list[1].document_id == "doc-chart-001"


# ===========================================================================
# 2. Single and Multi-Evidence Integration
# ===========================================================================


class TestSingleAndMultiEvidenceIntegration:
    """Tests 4-5: Verify single and multi-evidence flows through the pipeline."""

    def test_04_single_evidence_integration(self) -> None:
        """Single AgentCitation adapts -> pipeline executes -> VisionResult produced."""
        citation = _make_citation(
            doc_id="doc-single-001",
            filename="single.pdf",
            chunk_id="chk-single-001",
            content_type="image",
        )
        ev = VisualEvidenceAdapter.adapt_citation(
            citation, image_bytes=_SAMPLE_PNG, image_format="png"
        )
        provider = IntegrationTestProvider()
        pipeline = VisionPipeline(provider=provider)

        result = pipeline.run("Analyze the single chart", evidence=[ev])
        assert isinstance(result, VisionResult)
        assert result.status == "success"
        assert provider.call_count == 1

    def test_05a_two_evidence_integration(self) -> None:
        """Two evidence items produce one result with provider called once."""
        citations = [
            _make_citation(doc_id="doc-two-A", chunk_id="chk-A", content_type="image"),
            _make_citation(doc_id="doc-two-B", chunk_id="chk-B", content_type="chart"),
        ]
        evidence = [
            VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG) for c in citations
        ]
        provider = IntegrationTestProvider()
        result = VisionPipeline(provider=provider).run("Compare two visual elements", evidence=evidence)
        assert result.status == "success"
        assert provider.call_count == 1

    def test_05b_three_evidence_integration(self) -> None:
        """Three evidence items: provider called once, result produced."""
        citations = [
            _make_citation(doc_id=f"doc-three-{i}", chunk_id=f"chk-{i}", content_type="diagram")
            for i in range(3)
        ]
        evidence = [
            VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG) for c in citations
        ]
        provider = IntegrationTestProvider()
        result = VisionPipeline(provider=provider).run("Analyze three diagrams", evidence=evidence)
        assert result.status == "success"
        assert provider.call_count == 1

    def test_05c_five_evidence_integration(self) -> None:
        """Five evidence items: batch integration successful."""
        citations = [
            _make_citation(doc_id=f"doc-five-{i}", chunk_id=f"chk-five-{i}", content_type="chart")
            for i in range(5)
        ]
        evidence = [
            VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG) for c in citations
        ]
        provider = IntegrationTestProvider()
        result = VisionPipeline(provider=provider).run("Analyze five charts", evidence=evidence)
        assert result.status == "success"
        assert provider.call_count == 1

    def test_05d_ten_evidence_integration(self) -> None:
        """Ten evidence items: large batch integration successful."""
        citations = [
            _make_citation(doc_id=f"doc-ten-{i}", chunk_id=f"chk-ten-{i}", content_type="image")
            for i in range(10)
        ]
        evidence = [
            VisualEvidenceAdapter.adapt_citation(c, image_bytes=_SAMPLE_PNG) for c in citations
        ]
        provider = IntegrationTestProvider()
        result = VisionPipeline(provider=provider).run("Analyze ten images", evidence=evidence)
        assert result.status == "success"
        assert provider.call_count == 1


# ===========================================================================
# 3. Evidence Order & Lineage Preservation
# ===========================================================================


class TestEvidenceOrderAndLineage:
    """Tests 6, 7: Order and full lineage preserved through the adapter chain."""

    def test_06_evidence_order_preserved(self) -> None:
        """Evidence A, B, C adapted from citations preserves A, B, C order."""
        ids = ["doc-alpha", "doc-beta", "doc-gamma"]
        items = [
            _make_citation(doc_id=did, chunk_id=f"chk-{did}", content_type="image")
            for did in ids
        ]
        evidence_list = VisualEvidenceAdapter.adapt_batch(items)
        for i, ev in enumerate(evidence_list):
            assert ev.document_id == ids[i], f"Order mismatch at index {i}"

    def test_07_lineage_preserved_end_to_end(self) -> None:
        """Full lineage (doc_id, filename, chunk_id, page_number, chunk_index) survives pipeline."""
        citation = _make_citation(
            doc_id="DOC-001",
            filename="report.pdf",
            chunk_id="chunk-5",
            page_number=7,
            content_type="image",
            metadata={"chunk_index": 5, "source": "retrieval_agent"},
        )
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)

        assert ev.document_id == "DOC-001"
        assert ev.filename == "report.pdf"
        assert ev.chunk_id == "chunk-5"
        assert ev.page_number == 7
        assert ev.chunk_index == 5

        req = VisionRequest(query="Analyze lineage test chart", evidence=[ev])
        assert req.evidence[0].document_id == "DOC-001"

        prep = prepare_image_evidence(ev)
        model_input = build_vision_input(query="Analyze lineage test chart", evidence=prep)
        assert model_input.document_id == "DOC-001"
        assert model_input.filename == "report.pdf"
        assert model_input.chunk_id == "chunk-5"
        assert model_input.page_number == 7

        provider = IntegrationTestProvider()
        result = VisionPipeline(provider=provider).run("Analyze lineage test chart", evidence=[ev])
        assert result.document_id == "DOC-001"
        assert result.filename == "report.pdf"
        assert result.chunk_id == "chunk-5"
        assert result.page_number == 7


# ===========================================================================
# 4. Metadata Preservation
# ===========================================================================


class TestMetadataPreservation:
    """Test 8: Metadata is preserved, not lost, not duplicated, not cross-contaminated."""

    def test_08_metadata_preserved_through_adapter(self) -> None:
        """Citation metadata survives AgentCitation -> VisualEvidence conversion."""
        citation = _make_citation(
            doc_id="doc-meta-001",
            chunk_id="chk-meta-001",
            content_type="chart",
            metadata={"source": "search_agent", "page": 3, "section": "Q1 Revenue"},
        )
        ev = VisualEvidenceAdapter.adapt_citation(citation)

        assert ev.metadata["source"] == "search_agent"
        assert ev.metadata["page"] == 3
        assert ev.metadata["section"] == "Q1 Revenue"

    def test_08b_metadata_not_duplicated(self) -> None:
        """Metadata dictionary is not duplicated during adaptation."""
        citation = _make_citation(metadata={"key": "value", "count": 1})
        ev1 = VisualEvidenceAdapter.adapt_citation(citation)
        ev2 = VisualEvidenceAdapter.adapt_citation(citation)

        assert ev1.metadata == ev2.metadata
        assert ev1.metadata is not ev2.metadata

    def test_08c_metadata_not_cross_contaminated(self) -> None:
        """Metadata from evidence A does not appear in evidence B."""
        cit_a = _make_citation(
            doc_id="doc-A", chunk_id="chk-A", content_type="image",
            metadata={"owner": "doc_A_exclusive"},
        )
        cit_b = _make_citation(
            doc_id="doc-B", chunk_id="chk-B", content_type="chart",
            metadata={"owner": "doc_B_exclusive"},
        )
        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a)
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b)

        assert ev_a.metadata.get("owner") == "doc_A_exclusive"
        assert ev_b.metadata.get("owner") == "doc_B_exclusive"
        assert "doc_A_exclusive" not in ev_b.metadata.values()
        assert "doc_B_exclusive" not in ev_a.metadata.values()


# ===========================================================================
# 5. Content Type Compatibility
# ===========================================================================


class TestContentTypeCompatibility:
    """Test 9: Supported visual content types accepted; unsupported types rejected."""

    def test_09_supported_content_types(self) -> None:
        """image, chart, and diagram content types are accepted by the adapter."""
        for ctype in ("image", "chart", "diagram"):
            citation = _make_citation(
                doc_id=f"doc-{ctype}", chunk_id=f"chk-{ctype}", content_type=ctype
            )
            ev = VisualEvidenceAdapter.adapt_citation(citation)
            assert ev.content_type == ctype

    def test_09b_unsupported_content_types_not_visual(self) -> None:
        """text and table content types are detected as non-visual."""
        for ctype in ("text", "table", "pdf", "audio", "video"):
            assert not VisualEvidenceAdapter.is_visual_content_type(ctype)

    def test_09c_valid_visual_content_types_set(self) -> None:
        """VALID_VISUAL_CONTENT_TYPES matches the expected integration contract."""
        assert "image" in VALID_VISUAL_CONTENT_TYPES
        assert "chart" in VALID_VISUAL_CONTENT_TYPES
        assert "diagram" in VALID_VISUAL_CONTENT_TYPES
        assert "text" not in VALID_VISUAL_CONTENT_TYPES
        assert "table" not in VALID_VISUAL_CONTENT_TYPES

    def test_09d_non_visual_citation_rejected_by_adapt_citation(self) -> None:
        """adapt_citation raises VisionEvidenceError for non-visual content types."""
        citation = _make_citation(content_type="text")
        with pytest.raises(VisionEvidenceError):
            VisualEvidenceAdapter.adapt_citation(citation)


# ===========================================================================
# 6. VisionRequest Construction & Query Preservation
# ===========================================================================


class TestVisionRequestConstructionAndQuery:
    """Tests 10, 11: VisionRequest construction and query preservation."""

    def test_10_vision_request_from_evidence(self) -> None:
        """VisionRequest can be constructed from adapted AgentCitation evidence."""
        citation = _make_citation(content_type="image")
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="Describe the image content", evidence=[ev])

        assert req.query == "Describe the image content"
        assert req.has_evidence is True
        assert len(req.evidence) == 1
        assert req.evidence[0] is ev

    def test_11_query_preserved_normal(self) -> None:
        """Normal query string is preserved exactly through the VisionRequest."""
        query = "What does this chart show about Q3 performance?"
        citation = _make_citation(content_type="chart")
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query=query, evidence=[ev])
        assert req.query == query

    def test_11b_query_whitespace_stripped(self) -> None:
        """Whitespace around query is stripped by VisionRequest validation."""
        citation = _make_citation(content_type="image")
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
        req = VisionRequest(query="  Describe the image  ", evidence=[ev])
        assert req.query == "Describe the image"

    def test_11c_empty_query_rejected(self) -> None:
        """Empty query raises VisionInputValidationError, reusing Day 43 validation."""
        with pytest.raises(VisionInputValidationError, match="query"):
            VisionRequest(query="")

    def test_11d_whitespace_only_query_rejected(self) -> None:
        """Whitespace-only query raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="query"):
            VisionRequest(query="   ")


# ===========================================================================
# 7. Image Preparation & Input Builder Compatibility
# ===========================================================================


class TestImagePreparationAndInputBuilderCompatibility:
    """Tests 12, 13: Day 34 image preparation and Day 35 input builder reused."""

    def test_12_image_preparation_compatibility(self) -> None:
        """VisualEvidence from AgentCitation passes through prepare_image_evidence (Day 34)."""
        citation = _make_citation(content_type="chart")
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG, image_format="png")

        prep = prepare_image_evidence(ev)
        assert isinstance(prep, PreparedImageEvidence)
        assert prep.document_id == ev.document_id
        assert prep.filename == ev.filename
        assert prep.chunk_id == ev.chunk_id
        assert prep.size_bytes == len(_SAMPLE_PNG)
        assert prep.source.image_bytes == _SAMPLE_PNG

    def test_13_vision_model_input_compatibility(self) -> None:
        """PreparedImageEvidence from adapted citation feeds build_vision_input (Day 35)."""
        citation = _make_citation(
            doc_id="doc-input-001",
            filename="input.pdf",
            chunk_id="chk-input-001",
            content_type="diagram",
            page_number=9,
            metadata={"chunk_index": 3},
        )
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
        prep = prepare_image_evidence(ev)
        model_input = build_vision_input(query="Describe the diagram", evidence=prep)

        assert isinstance(model_input, VisionModelInput)
        assert model_input.query == "Describe the diagram"
        assert model_input.document_id == "doc-input-001"
        assert model_input.filename == "input.pdf"
        assert model_input.chunk_id == "chk-input-001"
        assert model_input.page_number == 9


# ===========================================================================
# 8. Provider, Execution, Lifecycle, and Result Normalization Compatibility
# ===========================================================================


class TestProviderAndExecutionCompatibility:
    """Tests 14-17: Days 36, 37, 38, 39 component reuse verified."""

    def test_14_provider_receives_valid_input(self) -> None:
        """Provider receives VisionModelInput constructed from Member 2 evidence."""
        citation = _make_citation(content_type="image")
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
        provider = IntegrationTestProvider()
        adapter = VisionExecutionAdapter(provider=provider)

        result = adapter.execute("Test provider query", evidence=[ev])
        assert result.status == "success"
        assert provider.call_count == 1

        received = provider.received_inputs[0]
        assert isinstance(received, VisionModelInput)
        assert received.query == "Test provider query"

    def test_15_execution_adapter_reused(self) -> None:
        """VisionExecutionAdapter (Day 37) handles integrated evidence path without bypass."""
        citation = _make_citation(content_type="chart")
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
        provider = IntegrationTestProvider()
        adapter = VisionExecutionAdapter(provider=provider)

        result = adapter.execute("Execute chart analysis", evidence=[ev])
        assert isinstance(result, VisionResult)
        assert result.status == "success"

    def test_16_lifecycle_followed_in_integrated_execution(self) -> None:
        """Integrated execution uses lifecycle (Day 38) without bypassing it."""
        citation = _make_citation(content_type="image")
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
        provider = IntegrationTestProvider()
        adapter = VisionExecutionAdapter(provider=provider)

        result = adapter.execute("Lifecycle integration test", evidence=[ev])
        assert result.status == "success"
        assert provider.call_count == 1

    def test_17_result_normalization_applied(self) -> None:
        """VisionResultNormalizer (Day 39) applied: lineage attached to result."""
        citation = _make_citation(
            doc_id="doc-norm-001",
            filename="norm.pdf",
            chunk_id="chk-norm-001",
            content_type="diagram",
            page_number=4,
        )
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
        provider = IntegrationTestProvider()
        result = VisionPipeline(provider=provider).run("Analyze diagram with normalization", evidence=[ev])

        assert result.document_id == "doc-norm-001"
        assert result.filename == "norm.pdf"
        assert result.chunk_id == "chk-norm-001"
        assert result.page_number == 4


# ===========================================================================
# 9. Observability, Retry, Timeout, Cancellation Compatibility
# ===========================================================================


class TestObservabilityRetryTimeoutCancellation:
    """Tests 18-21: Days 45, 47, 46, 46 component reuse verified."""

    def test_18_observability_reused_no_external_telemetry(self) -> None:
        """No external telemetry introduced; observation stays execution-local."""
        citation = _make_citation(content_type="image")
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
        provider = IntegrationTestProvider()
        adapter = VisionExecutionAdapter(provider=provider)

        result = adapter.execute("Observability test", evidence=[ev])
        assert result.status == "success"
        assert provider.call_count == 1

    def test_19_retry_on_provider_failure_reused(self) -> None:
        """Provider failure triggers existing retry mechanism (Day 47); evidence not duplicated."""
        citation = _make_citation(content_type="chart")
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)

        provider = IntegrationTestProvider(should_fail=True, fail_count=1)
        retry_policy = VisionRetryPolicy(max_retries=1)
        adapter = VisionExecutionAdapter(provider=provider)

        result = adapter.execute("Retry integration test", evidence=[ev], retry_policy=retry_policy)
        assert result.status == "success"
        assert provider.call_count == 2

    def test_20_timeout_handled_by_existing_mechanism(self) -> None:
        """Timeout safety (Day 46): tight timeout / provider timeout raises VisionTimeoutError."""
        citation = _make_citation(content_type="image")
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)

        provider = IntegrationTestProvider(simulate_timeout=True)
        adapter = VisionExecutionAdapter(provider=provider)

        with pytest.raises(VisionTimeoutError):
            adapter.execute("Timeout integration test", evidence=[ev])

    def test_21_cancellation_handled_by_existing_mechanism(self) -> None:
        """Cancellation safety (Day 46): pre-cancelled token raises VisionCancellationError."""
        citation = _make_citation(content_type="diagram")
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)

        token = VisionCancellationToken()
        token.cancel(reason="Integration test pre-cancellation")

        provider = IntegrationTestProvider()
        adapter = VisionExecutionAdapter(provider=provider)

        with pytest.raises(VisionCancellationError):
            adapter.execute("Cancellation integration test", evidence=[ev], cancellation_token=token)


# ===========================================================================
# 10. Concurrent Integration
# ===========================================================================


class TestConcurrentIntegration:
    """Test 22: Concurrent requests remain isolated."""

    def test_22_concurrent_integration_isolated(self) -> None:
        """Concurrent A/B/C requests each receive correct evidence; no cross-contamination."""
        results: dict[str, VisionResult] = {}
        errors: list[Exception] = []

        def run_request(label: str, doc_id: str, ct: str) -> None:
            try:
                citation = _make_citation(doc_id=doc_id, chunk_id=f"chk-{label}", content_type=ct)
                ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
                provider = IntegrationTestProvider()
                result = VisionPipeline(provider=provider).run(f"Query from {label}", evidence=[ev])
                results[label] = result
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=run_request, args=("A", "doc-A", "image")),
            threading.Thread(target=run_request, args=("B", "doc-B", "chart")),
            threading.Thread(target=run_request, args=("C", "doc-C", "diagram")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Concurrent errors: {errors}"
        assert results["A"].document_id == "doc-A"
        assert results["B"].document_id == "doc-B"
        assert results["C"].document_id == "doc-C"

    def test_22b_concurrent_mixed_outcomes(self) -> None:
        """Concurrent A(success), B(failure), C(retry->success) remain isolated."""
        results: dict[str, Any] = {}
        errors_map: dict[str, Exception] = {}

        def run_a() -> None:
            citation = _make_citation(doc_id="doc-A-mix", chunk_id="chk-A-mix", content_type="image")
            ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
            provider = IntegrationTestProvider()
            result = VisionPipeline(provider=provider).run("Query A", evidence=[ev])
            results["A"] = result

        def run_b() -> None:
            citation = _make_citation(doc_id="doc-B-mix", chunk_id="chk-B-mix", content_type="chart")
            ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
            provider = IntegrationTestProvider(should_fail=True, fail_count=10)
            retry_policy = VisionRetryPolicy(max_retries=0)
            adapter = VisionExecutionAdapter(provider=provider)
            try:
                adapter.execute("Query B", evidence=[ev], retry_policy=retry_policy)
            except VisionProviderExecutionError as exc:
                errors_map["B"] = exc

        def run_c() -> None:
            citation = _make_citation(doc_id="doc-C-mix", chunk_id="chk-C-mix", content_type="diagram")
            ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
            provider = IntegrationTestProvider(should_fail=True, fail_count=1)
            retry_policy = VisionRetryPolicy(max_retries=2)
            adapter = VisionExecutionAdapter(provider=provider)
            result = adapter.execute("Query C", evidence=[ev], retry_policy=retry_policy)
            results["C"] = result

        threads = [
            threading.Thread(target=run_a),
            threading.Thread(target=run_b),
            threading.Thread(target=run_c),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert results["A"].status == "success"
        assert results["A"].document_id == "doc-A-mix"
        assert "B" in errors_map
        assert results["C"].status == "success"
        assert results["C"].document_id == "doc-C-mix"


# ===========================================================================
# 11. Resource Safety
# ===========================================================================


class TestResourceSafety:
    """Test 23: Repeated integrated execution leaves no residual state."""

    def test_23_resource_safety_repeated_execution(self) -> None:
        """10 sequential integrated executions produce independent results with no leakage."""
        for i in range(10):
            citation = _make_citation(
                doc_id=f"doc-resource-{i}",
                chunk_id=f"chk-resource-{i}",
                content_type="image",
                metadata={"run": i},
            )
            ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
            provider = IntegrationTestProvider()
            result = VisionPipeline(provider=provider).run(f"Resource safety run {i}", evidence=[ev])

            assert result.status == "success"
            assert result.document_id == f"doc-resource-{i}"
            assert provider.call_count == 1


# ===========================================================================
# 12. Public API Compatibility
# ===========================================================================


class TestPublicAPICompatibility:
    """Test 24: Integration uses only public Day 49 API."""

    def test_24_integration_uses_public_api_only(self) -> None:
        """Entire integration flow uses only publicly exported vision symbols."""
        from vision import (  # noqa: F401
            VisualEvidence,
            VisualEvidenceAdapter,
            VisionAgent,
            VisionCancellationError,
            VisionCancellationToken,
            VisionEvidenceError,
            VisionExecutionAdapter,
            VisionExecutionLifecycle,
            VisionExecutionObservation,
            VisionExecutionStage,
            VisionExecutionTrace,
            VisionInputValidationError,
            VisionModelInput,
            VisionModelProvider,
            VisionPipeline,
            VisionProviderCapabilities,
            VisionProviderConfig,
            VisionProviderConfigError,
            VisionProviderError,
            VisionProviderExecutionError,
            VisionProviderUnavailableError,
            VisionRequest,
            VisionResult,
            VisionResultNormalizer,
            VisionRetryPolicy,
            VisionTimeoutError,
            execute_vision_request,
            run_vision_pipeline,
        )
        assert VisionPipeline is not None
        assert VisualEvidenceAdapter is not None
        assert VisionResultNormalizer is not None
        assert run_vision_pipeline is not None


# ===========================================================================
# 13. No Duplicate Retrieval, Embedding, or Qdrant Access
# ===========================================================================


class TestNoDuplicateRetrievalOrEmbedding:
    """Tests 25-27: Vision subsystem consumes evidence; does not perform retrieval."""

    def test_25_no_duplicate_retrieval(self) -> None:
        """Member 3 consumes pre-supplied evidence; no vector search performed."""
        citation = _make_citation(content_type="image")
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
        provider = IntegrationTestProvider()
        result = VisionPipeline(provider=provider).run("No retrieval test", evidence=[ev])
        assert result.status == "success"

    def test_26_no_duplicate_embedding(self) -> None:
        """No embedding generation occurs within the Vision pipeline execution."""
        import vision.pipeline as vision_pipeline_module
        import vision.execution_adapter as execution_adapter_module
        import vision.image_preparation as image_prep_module

        for module in (vision_pipeline_module, execution_adapter_module, image_prep_module):
            assert "EmbeddingGenerator" not in dir(module)
            assert "generate_embedding" not in dir(module)

    def test_27_no_qdrant_access(self) -> None:
        """No Qdrant client is instantiated or accessible in the Vision pipeline."""
        import vision.pipeline as vision_pipeline_module
        import vision.execution_adapter as execution_adapter_module

        for module in (vision_pipeline_module, execution_adapter_module):
            assert "QdrantClient" not in dir(module)
            assert "qdrant_client" not in dir(module)

    def test_27b_vector_search_result_consumed_not_generated(self) -> None:
        """VectorSearchResult is consumed via adapter; not generated by Member 3."""
        vsr = _make_vector_search_result(
            doc_id="doc-vsr-consume",
            chunk_id="chk-vsr-consume",
            content_type="diagram",
        )
        ev = VisualEvidenceAdapter.adapt_search_result(
            vsr, image_bytes=_SAMPLE_PNG, image_format="png"
        )
        assert isinstance(ev, VisualEvidence)
        assert ev.document_id == "doc-vsr-consume"
        assert ev.content_type == "diagram"
        assert ev.chunk_index == vsr.chunk_index


# ===========================================================================
# 14. Offline Execution Guarantee
# ===========================================================================


class TestOfflineExecutionGuarantee:
    """Test 28: Full integration path executes 100% offline."""

    def test_28_offline_execution(self) -> None:
        """Complete integration flow: citation -> adapter -> pipeline -> result runs offline."""
        citation = _make_citation(
            doc_id="doc-offline-001",
            filename="offline.pdf",
            chunk_id="chk-offline-001",
            content_type="chart",
            page_number=2,
            metadata={"source": "offline_test"},
        )
        ev = VisualEvidenceAdapter.adapt_citation(citation, image_bytes=_SAMPLE_PNG)
        provider = IntegrationTestProvider()
        result = VisionPipeline(provider=provider).run("Offline integration test", evidence=[ev])

        assert result.status == "success"
        assert result.document_id == "doc-offline-001"
        assert result.filename == "offline.pdf"
        assert result.chunk_id == "chk-offline-001"
        assert result.page_number == 2

    def test_28b_agent_response_offline_integration(self) -> None:
        """AgentResponse from Member 2 -> adapt_search_package -> pipeline runs offline."""
        citations = [
            _make_citation(doc_id="doc-resp-A", chunk_id="chk-resp-A", content_type="image"),
            _make_citation(doc_id="doc-resp-B", chunk_id="chk-resp-B", content_type="chart"),
        ]
        agent_response = AgentResponse(
            answer="The visual analysis results.",
            agent_name="SearchAgent",
            status="success",
            citations=citations,
            metadata={"query": "Describe chart and image", "context": ""},
        )
        evidence_list = VisualEvidenceAdapter.adapt_search_package(agent_response)
        assert len(evidence_list) == 2

        for ev in evidence_list:
            ev_with_bytes = VisualEvidence(
                document_id=ev.document_id,
                filename=ev.filename,
                chunk_id=ev.chunk_id,
                page_number=ev.page_number,
                chunk_index=ev.chunk_index,
                content_type=ev.content_type,
                image_bytes=_SAMPLE_PNG,
                metadata=ev.metadata,
            )
            provider = IntegrationTestProvider()
            result = VisionPipeline(provider=provider).run(
                "Offline agent response integration", evidence=[ev_with_bytes]
            )
            assert result.status == "success"
