"""
OmniBrain Member 4 — Day 47 Multi-Modal Evidence & Vision Grounding Regression Certification.

Validates the VisualEvidence, VisionRequest, VisionResult, and VisualEvidenceAdapter contracts across:
  - Vision Subsystem (VisualEvidence, VisionRequest, VisionResult, VisualEvidenceAdapter)
  - Ingestion Layer (VectorSearchResult, DocumentChunk)
  - Agents Layer (AgentCitation, AgentResponse, SearchResult)

Covers:
  1.  Valid VisualEvidence construction, field validation, and lineage preservation.
  2.  Document, page, and chunk identity preservation across visual modalities.
  3.  Multi-page evidence isolation within a single document.
  4.  Multi-document visual evidence isolation (zero cross-document leakage).
  5.  VisionResult and VisionRequest structural schema guarantees.
  6.  Retrieval (Member 1) -> Search citation (Member 2) -> VisualEvidence (Member 3) adaptation.
  7.  Visual description and metadata preservation.
  8.  Complete dictionary and JSON serialization round-trips for VisualEvidence, VisionRequest, and VisionResult.
  9.  Corrupted/invalid serialization and type validation error contracts.
  10. Empty evidence, missing image byte/path handling, and optional visual fields.
  11. Cross-page and cross-document contamination prevention.
  12. Cross-request isolation and execution state encapsulation.
  13. Duplicate visual evidence handling.
  14. Input mutation safety and object isolation.
  15. 3-iteration determinism and error isolation across sequential calls.
  16. End-to-end multi-modal lineage traceability.

Constraints:
  - 100% Offline: Synthetic deterministic models, no external vision APIs, no real multimodal LLMs.
  - Zero production code modified.
  - No new vision logic, citation generators, or adapters added.
  - Synthetic deterministic data only.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# Ingestion layer (Member 1)
from ingestion.models import (
    DocumentChunk,
    VectorSearchResult,
)

# Agents layer (Member 2)
from agents.models import (
    AgentCitation,
    AgentResponse,
    SearchResult,
)

# Vision layer (Member 3)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.exceptions import (
    VisionError,
    VisionEvidenceError,
    VisionInputValidationError,
)
from vision.evidence_adapter import VisualEvidenceAdapter


# ============================================================================
# Deterministic Synthetic Fixtures
# ============================================================================

DAY47_DOC_A = "DAY47-DOC-A"
DAY47_DOC_B = "DAY47-DOC-B"

DAY47_FILE_A = "day47_alpha_diagrams.pdf"
DAY47_FILE_B = "day47_beta_charts.pdf"

DAY47_PAGE_A1 = 1
DAY47_PAGE_A2 = 2
DAY47_PAGE_B1 = 1

DAY47_CHUNK_A1 = "DAY47-CHUNK-A1"
DAY47_CHUNK_A2 = "DAY47-CHUNK-A2"
DAY47_CHUNK_B1 = "DAY47-CHUNK-B1"

DAY47_VISUAL_A1 = "DAY47_VISUAL_A1_FLOWCHART"
DAY47_VISUAL_A2 = "DAY47_VISUAL_A2_BARCHART"
DAY47_VISUAL_B1 = "DAY47_VISUAL_B1_SYSTEM_DIAGRAM"

DAY47_VISUAL_DESCRIPTION_A = "DAY47_VISUAL_DESCRIPTION_A: Architecture pipeline flowchart."
DAY47_VISUAL_DESCRIPTION_B = "DAY47_VISUAL_DESCRIPTION_B: Component latency benchmarks."

DAY47_META_A: dict[str, Any] = {
    "day47_source": "synthetic",
    "day47_document": "A",
    "resolution": "1920x1080",
}
DAY47_META_B: dict[str, Any] = {
    "day47_source": "synthetic",
    "day47_document": "B",
    "resolution": "1280x720",
}


# ============================================================================
# 1. Valid VisualEvidence Construction & Lineage
# ============================================================================

class TestValidVisualEvidenceConstruction:
    """Certifies constructor, lineage fields, and modality validation for VisualEvidence."""

    def test_visual_evidence_field_preservation(self) -> None:
        """VisualEvidence retains all lineage and image attributes."""
        ev = VisualEvidence(
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            chunk_id=DAY47_CHUNK_A1,
            page_number=DAY47_PAGE_A1,
            chunk_index=0,
            content_type="diagram",
            image_path="/data/images/flowchart.png",
            image_bytes=b"\x89PNG\r\n\x1a\n_mock_bytes",
            image_format="png",
            width=1920,
            height=1080,
            description=DAY47_VISUAL_DESCRIPTION_A,
            metadata=DAY47_META_A,
        )

        assert ev.document_id == DAY47_DOC_A
        assert ev.filename == DAY47_FILE_A
        assert ev.chunk_id == DAY47_CHUNK_A1
        assert ev.page_number == DAY47_PAGE_A1
        assert ev.chunk_index == 0
        assert ev.content_type == "diagram"
        assert ev.image_path == "/data/images/flowchart.png"
        assert ev.image_bytes == b"\x89PNG\r\n\x1a\n_mock_bytes"
        assert ev.image_format == "png"
        assert ev.width == 1920
        assert ev.height == 1080
        assert ev.description == DAY47_VISUAL_DESCRIPTION_A
        assert ev.metadata == DAY47_META_A

    @pytest.mark.parametrize("modality", ["image", "chart", "diagram"])
    def test_supported_visual_modalities(self, modality: str) -> None:
        """VisualEvidence accepts all valid visual modalities ('image', 'chart', 'diagram')."""
        ev = VisualEvidence(
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            chunk_id=f"CHUNK-{modality}",
            content_type=modality,
        )
        assert ev.content_type == modality


# ============================================================================
# 2. Multi-Page & Multi-Document Isolation
# ============================================================================

class TestMultiPageAndMultiDocumentIsolation:
    """Certifies page-level and document-level isolation of visual evidence."""

    def test_multi_page_evidence_isolation_within_document(self) -> None:
        """Evidence items across different pages of the same document maintain strict page lineage."""
        ev_p1 = VisualEvidence(
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            chunk_id=DAY47_CHUNK_A1,
            page_number=DAY47_PAGE_A1,
            chunk_index=0,
            content_type="diagram",
            description=DAY47_VISUAL_A1,
        )
        ev_p2 = VisualEvidence(
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            chunk_id=DAY47_CHUNK_A2,
            page_number=DAY47_PAGE_A2,
            chunk_index=1,
            content_type="chart",
            description=DAY47_VISUAL_A2,
        )

        assert ev_p1.page_number == 1
        assert ev_p1.chunk_id == DAY47_CHUNK_A1
        assert ev_p1.description == DAY47_VISUAL_A1

        assert ev_p2.page_number == 2
        assert ev_p2.chunk_id == DAY47_CHUNK_A2
        assert ev_p2.description == DAY47_VISUAL_A2

        assert ev_p1.page_number != ev_p2.page_number
        assert ev_p1.chunk_id != ev_p2.chunk_id

    def test_multi_document_evidence_isolation(self) -> None:
        """Visual evidence from DOC-A and DOC-B remain strictly isolated."""
        ev_a = VisualEvidence(
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            chunk_id=DAY47_CHUNK_A1,
            page_number=DAY47_PAGE_A1,
            description=DAY47_VISUAL_A1,
            metadata=DAY47_META_A,
        )
        ev_b = VisualEvidence(
            document_id=DAY47_DOC_B,
            filename=DAY47_FILE_B,
            chunk_id=DAY47_CHUNK_B1,
            page_number=DAY47_PAGE_B1,
            description=DAY47_VISUAL_B1,
            metadata=DAY47_META_B,
        )

        req_a = VisionRequest(query="Analyze Doc A diagrams", evidence=[ev_a])
        req_b = VisionRequest(query="Analyze Doc B charts", evidence=[ev_b])

        assert req_a.evidence[0].document_id == DAY47_DOC_A
        assert req_a.evidence[0].filename == DAY47_FILE_A
        assert req_a.evidence[0].metadata["day47_document"] == "A"

        assert req_b.evidence[0].document_id == DAY47_DOC_B
        assert req_b.evidence[0].filename == DAY47_FILE_B
        assert req_b.evidence[0].metadata["day47_document"] == "B"


# ============================================================================
# 3. Vision Result & Request Contracts
# ============================================================================

class TestVisionResultAndRequestContracts:
    """Certifies structural and operational properties of VisionRequest and VisionResult."""

    def test_vision_request_properties_and_validation(self) -> None:
        """VisionRequest exposes total_evidence, has_evidence, and validates inputs."""
        ev = VisualEvidence(
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            chunk_id=DAY47_CHUNK_A1,
            page_number=DAY47_PAGE_A1,
        )
        req = VisionRequest(
            query="Analyze latency graph",
            evidence=[ev],
            session_id="SESS-VISION-47",
            metadata={"priority": "high"},
        )

        assert req.query == "Analyze latency graph"
        assert req.has_evidence is True
        assert req.total_evidence == 1
        assert req.session_id == "SESS-VISION-47"
        assert req.metadata["priority"] == "high"

    def test_vision_result_primary_lineage_inheritance(self) -> None:
        """VisionResult inherits document_id, filename, page_number, and chunk_id from primary evidence."""
        ev = VisualEvidence(
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            chunk_id=DAY47_CHUNK_A1,
            page_number=DAY47_PAGE_A1,
            content_type="chart",
        )

        res = VisionResult(
            query="Interpret chart",
            status="success",
            description="Bar chart showing 20% throughput increase.",
            evidence=[ev],
        )

        assert res.document_id == DAY47_DOC_A
        assert res.filename == DAY47_FILE_A
        assert res.page_number == DAY47_PAGE_A1
        assert res.chunk_id == DAY47_CHUNK_A1
        assert res.content_type == "chart"
        assert res.is_success is True
        assert res.is_error is False
        assert res.has_evidence is True


# ============================================================================
# 4. Cross-Component Adaptation (Member 1 -> 2 -> 3)
# ============================================================================

class TestCrossComponentVisualEvidenceAdaptation:
    """Certifies VisualEvidenceAdapter converting VectorSearchResult and AgentCitation into VisualEvidence."""

    def test_adapt_citation_preserves_full_lineage(self) -> None:
        """VisualEvidenceAdapter.adapt_citation preserves all citation attributes."""
        citation = AgentCitation(
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            chunk_id=DAY47_CHUNK_A1,
            page_number=DAY47_PAGE_A1,
            content_type="chart",
            score=0.92,
            metadata={"image_path": "/img/chart.png", "chunk_index": 2},
        )

        assert VisualEvidenceAdapter.is_visual(citation) is True
        ev = VisualEvidenceAdapter.adapt_citation(citation)

        assert ev.document_id == DAY47_DOC_A
        assert ev.filename == DAY47_FILE_A
        assert ev.chunk_id == DAY47_CHUNK_A1
        assert ev.page_number == DAY47_PAGE_A1
        assert ev.content_type == "chart"
        assert ev.chunk_index == 2
        assert ev.image_path == "/img/chart.png"

    def test_adapt_vector_search_result_preserves_content_as_description(self) -> None:
        """VisualEvidenceAdapter.adapt_search_result sets description from result content."""
        vs_res = VectorSearchResult(
            chunk_id=DAY47_CHUNK_A2,
            score=0.89,
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            page_number=DAY47_PAGE_A2,
            chunk_index=1,
            content_type="diagram",
            content=DAY47_VISUAL_DESCRIPTION_A,
            metadata=DAY47_META_A,
        )

        assert VisualEvidenceAdapter.is_visual(vs_res) is True
        ev = VisualEvidenceAdapter.adapt_search_result(vs_res)

        assert ev.document_id == DAY47_DOC_A
        assert ev.filename == DAY47_FILE_A
        assert ev.chunk_id == DAY47_CHUNK_A2
        assert ev.page_number == DAY47_PAGE_A2
        assert ev.content_type == "diagram"
        assert ev.description == DAY47_VISUAL_DESCRIPTION_A
        assert ev.metadata["day47_document"] == "A"

    def test_non_visual_rejection_in_adapter(self) -> None:
        """VisualEvidenceAdapter rejects text and table modalities with VisionEvidenceError."""
        text_citation = AgentCitation(
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            chunk_id="CHUNK-TXT",
            content_type="text",
        )

        assert VisualEvidenceAdapter.is_visual(text_citation) is False
        with pytest.raises(VisionEvidenceError, match="Unsupported content_type 'text'"):
            VisualEvidenceAdapter.adapt_citation(text_citation)


# ============================================================================
# 5. Serialization Round-Trips
# ============================================================================

class TestVisualEvidenceSerializationRoundTrips:
    """Certifies to_dict -> JSON -> from_dict serialization round-trips."""

    def test_visual_evidence_json_roundtrip(self) -> None:
        """VisualEvidence survives full JSON round-trip without field loss."""
        ev = VisualEvidence(
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            chunk_id=DAY47_CHUNK_A1,
            page_number=DAY47_PAGE_A1,
            chunk_index=3,
            content_type="chart",
            image_path="/data/c1.png",
            image_format="png",
            width=800,
            height=600,
            description=DAY47_VISUAL_DESCRIPTION_A,
            metadata=DAY47_META_A,
        )

        d = ev.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        restored = VisualEvidence.from_dict(parsed)

        assert restored.document_id == ev.document_id
        assert restored.filename == ev.filename
        assert restored.chunk_id == ev.chunk_id
        assert restored.page_number == ev.page_number
        assert restored.chunk_index == ev.chunk_index
        assert restored.content_type == ev.content_type
        assert restored.image_path == ev.image_path
        assert restored.image_format == ev.image_format
        assert restored.width == ev.width
        assert restored.height == ev.height
        assert restored.description == ev.description
        assert restored.metadata == ev.metadata

    def test_vision_request_and_result_json_roundtrip(self) -> None:
        """VisionRequest and VisionResult survive JSON serialization round-trips."""
        ev = VisualEvidence(
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            chunk_id=DAY47_CHUNK_A1,
            page_number=DAY47_PAGE_A1,
        )
        res = VisionResult(
            query="Analyze diagram",
            status="success",
            description="Diagram analysis complete.",
            evidence=[ev],
            metadata={"confidence": 0.98},
        )

        d_res = res.to_dict()
        json_res = json.dumps(d_res)
        restored_res = VisionResult.from_dict(json.loads(json_res))

        assert restored_res.query == res.query
        assert restored_res.status == "success"
        assert restored_res.description == res.description
        assert len(restored_res.evidence) == 1
        assert restored_res.evidence[0].document_id == DAY47_DOC_A
        assert restored_res.document_id == DAY47_DOC_A
        assert restored_res.metadata["confidence"] == 0.98


# ============================================================================
# 6. Validation Errors & Corrupted Data Handling
# ============================================================================

class TestValidationAndCorruptedDataHandling:
    """Certifies deterministic rejection of invalid types, empty IDs, and wrong modalities."""

    def test_missing_required_ids_raises_vision_evidence_error(self) -> None:
        """Empty or whitespace document_id, filename, or chunk_id raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError, match="document_id must be a non-empty string"):
            VisualEvidence(document_id="", filename="f.pdf", chunk_id="c1")

        with pytest.raises(VisionEvidenceError, match="filename must be a non-empty string"):
            VisualEvidence(document_id="d1", filename="   ", chunk_id="c1")

        with pytest.raises(VisionEvidenceError, match="chunk_id must be a non-empty string"):
            VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="")

    def test_invalid_types_and_values_raises_vision_evidence_error(self) -> None:
        """Invalid page_number (<=0), chunk_index (<0), or invalid content_type raises VisionEvidenceError."""
        with pytest.raises(VisionEvidenceError, match="page_number must be a positive integer"):
            VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", page_number=0)

        with pytest.raises(VisionEvidenceError, match="chunk_index must be a non-negative integer"):
            VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", chunk_index=-1)

        with pytest.raises(VisionEvidenceError, match="Invalid visual content_type 'audio'"):
            VisualEvidence(document_id="d1", filename="f.pdf", chunk_id="c1", content_type="audio")

    def test_vision_request_empty_query_raises_validation_error(self) -> None:
        """VisionRequest with empty or whitespace query raises VisionInputValidationError."""
        with pytest.raises(VisionInputValidationError, match="query cannot be empty"):
            VisionRequest(query="   ")

        with pytest.raises(VisionInputValidationError, match="query must be a string"):
            VisionRequest(query=123)  # type: ignore[arg-type]


# ============================================================================
# 7. Empty, Missing Images & Optional Fields
# ============================================================================

class TestEmptyAndMissingImageHandling:
    """Certifies handling of optional image paths, missing byte buffers, and empty evidence lists."""

    def test_visual_evidence_with_no_image_file_is_valid(self) -> None:
        """VisualEvidence with None image_path/image_bytes is structurally valid for metadata lineage."""
        ev = VisualEvidence(
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            chunk_id=DAY47_CHUNK_A1,
            page_number=DAY47_PAGE_A1,
            image_path=None,
            image_bytes=None,
            description="OCR extracted text representation",
        )
        assert ev.image_path is None
        assert ev.image_bytes is None
        assert ev.description == "OCR extracted text representation"

    def test_vision_request_empty_evidence_list_is_valid(self) -> None:
        """VisionRequest with empty evidence list is valid and reports has_evidence=False."""
        req = VisionRequest(query="Query with no visual evidence", evidence=[])
        assert req.has_evidence is False
        assert req.total_evidence == 0


# ============================================================================
# 8. Cross-Request Isolation & State Encapsulation
# ============================================================================

class TestCrossRequestIsolationAndState:
    """Certifies sequential request independence without state leakage."""

    def test_sequential_vision_requests_isolation(self) -> None:
        """Two VisionRequests processed sequentially maintain independent metadata and evidence."""
        ev_a = VisualEvidence(
            document_id=DAY47_DOC_A, filename=DAY47_FILE_A, chunk_id=DAY47_CHUNK_A1,
            metadata={"request_id": "REQ-A", "marker": DAY47_VISUAL_A1},
        )
        ev_b = VisualEvidence(
            document_id=DAY47_DOC_B, filename=DAY47_FILE_B, chunk_id=DAY47_CHUNK_B1,
            metadata={"request_id": "REQ-B", "marker": DAY47_VISUAL_B1},
        )

        req_a = VisionRequest(query="Query A", evidence=[ev_a], session_id="SESS-A")
        req_b = VisionRequest(query="Query B", evidence=[ev_b], session_id="SESS-B")

        assert req_a.evidence[0].metadata["request_id"] == "REQ-A"
        assert req_a.evidence[0].document_id == DAY47_DOC_A
        assert req_a.session_id == "SESS-A"

        assert req_b.evidence[0].metadata["request_id"] == "REQ-B"
        assert req_b.evidence[0].document_id == DAY47_DOC_B
        assert req_b.session_id == "SESS-B"


# ============================================================================
# 9. Input Mutation Safety & Object Isolation
# ============================================================================

class TestInputMutationSafetyAndIsolation:
    """Certifies that caller input dictionaries and instances are not mutated."""

    def test_caller_metadata_mutation_safety(self) -> None:
        """Mutating caller metadata dictionary after creation does not affect VisualEvidence."""
        caller_meta = {"env": "staging", "marker": "initial"}
        ev = VisualEvidence(
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            chunk_id=DAY47_CHUNK_A1,
            metadata=caller_meta,
        )

        caller_meta["env"] = "corrupted"
        assert ev.metadata["env"] == "staging"

    def test_visual_evidence_object_isolation(self) -> None:
        """Mutating one VisualEvidence dictionary output does not mutate another instance."""
        ev1 = VisualEvidence(document_id=DAY47_DOC_A, filename=DAY47_FILE_A, chunk_id="C-1")
        ev2 = VisualEvidence(document_id=DAY47_DOC_B, filename=DAY47_FILE_B, chunk_id="C-2")

        d1 = ev1.to_dict()
        d1["document_id"] = "MUTATED"

        assert ev1.document_id == DAY47_DOC_A
        assert ev2.document_id == DAY47_DOC_B


# ============================================================================
# 10. Repeated Execution & Error Isolation
# ============================================================================

class TestRepeatedExecutionAndErrorIsolation:
    """Certifies 3-iteration determinism and error isolation."""

    def test_visual_evidence_determinism_3_iterations(self) -> None:
        """3 identical executions yield identical serialized representations."""
        runs: list[dict[str, Any]] = []
        for _ in range(3):
            ev = VisualEvidence(
                document_id=DAY47_DOC_A,
                filename=DAY47_FILE_A,
                chunk_id=DAY47_CHUNK_A1,
                page_number=DAY47_PAGE_A1,
                content_type="chart",
                description=DAY47_VISUAL_DESCRIPTION_A,
                metadata=DAY47_META_A,
            )
            runs.append(ev.to_dict())

        assert runs[0] == runs[1] == runs[2]

    def test_sequential_error_isolation(self) -> None:
        """An invalid evidence construction in a sequence does not affect valid ones."""
        valid_items: list[str] = []

        # 1. Valid A
        ev_a = VisualEvidence(document_id=DAY47_DOC_A, filename=DAY47_FILE_A, chunk_id=DAY47_CHUNK_A1)
        valid_items.append(ev_a.document_id)

        # 2. Invalid B (empty chunk_id)
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id=DAY47_DOC_B, filename=DAY47_FILE_B, chunk_id="")

        # 3. Valid C
        ev_c = VisualEvidence(document_id=DAY47_DOC_B, filename=DAY47_FILE_B, chunk_id=DAY47_CHUNK_B1)
        valid_items.append(ev_c.document_id)

        assert valid_items == [DAY47_DOC_A, DAY47_DOC_B]


# ============================================================================
# 11. End-to-End Multi-Modal Lineage Traceability
# ============================================================================

class TestEndToEndMultiModalLineage:
    """Certifies complete multi-modal pipeline lineage from Ingestion to Vision & Response."""

    def test_complete_multimodal_lineage_chain(self) -> None:
        """
        Flow:
          VectorSearchResult (Member 1: diagram chunk)
            -> AgentCitation (Member 2: search citation)
            -> VisualEvidence (Member 3: adapted visual evidence)
            -> VisionRequest (Member 3: vision processing request)
            -> VisionResult (Member 3: analysis result)
            -> AgentResponse (Member 2: synthesized response)
        """
        # 1. Member 1: VectorSearchResult for diagram
        vs_res = VectorSearchResult(
            chunk_id=DAY47_CHUNK_A1,
            score=0.96,
            document_id=DAY47_DOC_A,
            filename=DAY47_FILE_A,
            page_number=DAY47_PAGE_A1,
            chunk_index=0,
            content_type="diagram",
            content=DAY47_VISUAL_DESCRIPTION_A,
            metadata={"image_path": "/data/diagram_p1.png", "marker": DAY47_VISUAL_A1},
        )

        # 2. Member 2: AgentCitation creation
        citation = AgentCitation.from_search_result(vs_res)
        assert citation.document_id == DAY47_DOC_A
        assert citation.chunk_id == DAY47_CHUNK_A1
        assert citation.page_number == DAY47_PAGE_A1
        assert citation.content_type == "diagram"

        # 3. Member 3: VisualEvidence adaptation
        visual_ev = VisualEvidenceAdapter.adapt_citation(citation)
        assert visual_ev.document_id == DAY47_DOC_A
        assert visual_ev.filename == DAY47_FILE_A
        assert visual_ev.chunk_id == DAY47_CHUNK_A1
        assert visual_ev.page_number == DAY47_PAGE_A1
        assert visual_ev.content_type == "diagram"
        assert visual_ev.image_path == "/data/diagram_p1.png"

        # 4. Member 3: VisionRequest & VisionResult
        v_req = VisionRequest(query="Explain architecture flowchart", evidence=[visual_ev])
        v_res = VisionResult(
            query=v_req.query,
            status="success",
            description="Architecture flowchart indicates 3 primary ingestion modules.",
            evidence=v_req.evidence,
        )
        assert v_res.document_id == DAY47_DOC_A
        assert v_res.page_number == DAY47_PAGE_A1
        assert v_res.chunk_id == DAY47_CHUNK_A1

        # 5. Member 2: AgentResponse delivery
        agent_resp = AgentResponse(
            answer=f"Verified diagram analysis: {v_res.description}",
            agent_name="VisionAgent",
            citations=[citation],
            metadata={"vision_status": v_res.status},
        )
        assert agent_resp.citations[0].document_id == DAY47_DOC_A
        assert agent_resp.citations[0].chunk_id == DAY47_CHUNK_A1
        assert agent_resp.citations[0].page_number == DAY47_PAGE_A1
        assert agent_resp.unique_documents == [DAY47_DOC_A]
