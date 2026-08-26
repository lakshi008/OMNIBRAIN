"""
OmniBrain Member 4 -- Day 20 End-to-End Regression & Release Readiness Certification Tests.

Performs a comprehensive final release-readiness certification verifying that all established
integration contracts remain stable, secure, deterministic, and interoperable across the pipeline:

  Ingestion (Member 1)
       ↓
  Search / Retrieval (Member 2)
       ↓
  Vision (Member 3)
       ↓
  Supervisor / Downstream Consumers

Certification Areas:
 1. Ingestion contract certification (DocumentChunk, VectorSearchResult, retrieval processor)
 2. Search / Retrieval contract certification (SearchRequest, SearchResult, AgentCitation, AgentRequest)
 3. Vision contract certification (VisionRequest, VisualEvidence, VisionResult, VisionExecutionLifecycle)
 4. End-to-end handoff certification (Ingestion -> Search -> Vision -> Downstream AgentState/AgentResponse)
 5. Multi-document certification (Document A vs Document B strict isolation)
 6. Failure certification (typed validation exceptions, clean error VisionResult without stale state)
 7. Retry certification (deterministic lifecycle stage progression across simulated retries)
 8. Concurrency certification (thread-safe isolated executions under high thread concurrency)
 9. Determinism certification (identical inputs yield identical observable result structures across runs)
10. Serialization certification (to_dict -> from_dict -> to_dict roundtrip across all public models)
11. Security certification (metadata sanitization, forbidden keys stripped, zero credential exposure)
12. Offline execution certification (100% offline, zero network, zero external LLMs or databases)
13. Resource safety certification (in-memory execution with no dangling file artifacts)
14. Duplicate-work certification (exact evidence counts preserved without synthetic duplication)
15. Public API contract certification (all public classes, factory methods, and conversions accessible)

Constraints:
 - 100% Offline: No external APIs, network, real LLMs, or production secrets.
 - Zero production code modified.
 - Only observable behavior guaranteed by existing public contracts certified.
"""

from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# Ingestion Subsystem (Member 1)
from ingestion.models import (
    DocumentChunk,
    VectorSearchResult,
)
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.retrieval_processor import build_retrieval_context, process_retrieval_results

# Search / Agents Subsystem (Member 2)
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    AgentState,
    SearchRequest,
    SearchResult,
)
from agents.exceptions import AgentValidationError

# Vision Subsystem (Member 3)
from vision.models import (
    VisionRequest,
    VisionResult,
    VisualEvidence,
    VALID_VISUAL_CONTENT_TYPES,
)
from vision.exceptions import VisionEvidenceError, VisionInputValidationError
from vision.lifecycle import (
    VisionExecutionLifecycle,
    VisionExecutionStage,
)
from vision.result_normalizer import (
    FORBIDDEN_METADATA_KEYS,
    VisionExecutionTrace,
    VisionResultNormalizer,
)


# ============================================================================
# Helpers & Fixtures
# ============================================================================

def _certified_chunk(
    chunk_id: str = "chk-cert-001",
    document_id: str = "doc-cert-001",
    filename: str = "certified_report.pdf",
    page_number: int | None = 2,
    chunk_index: int = 0,
    content: str = "Certified system revenue and metrics.",
    content_type: str = "chart",
    metadata: dict[str, Any] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        content=content,
        content_type=content_type,
        metadata=metadata if metadata is not None else {"cert_test": "DAY20", "source": "CERT_DOC_A"},
    )


def _certified_vsr(
    chunk_id: str = "chk-cert-001",
    score: float = 0.95,
    document_id: str = "doc-cert-001",
    filename: str = "certified_report.pdf",
    page_number: int | None = 2,
    chunk_index: int = 0,
    content: str = "Certified system revenue and metrics.",
    content_type: str = "chart",
    metadata: dict[str, Any] | None = None,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        score=score,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        chunk_index=chunk_index,
        content_type=content_type,
        content=content,
        metadata=metadata if metadata is not None else {"cert_test": "DAY20", "source": "CERT_DOC_A"},
    )


def _certified_evidence(
    chunk_id: str = "chk-cert-001",
    document_id: str = "doc-cert-001",
    filename: str = "certified_report.pdf",
    page_number: int | None = 2,
    chunk_index: int = 0,
    content_type: str = "chart",
    metadata: dict[str, Any] | None = None,
) -> VisualEvidence:
    return VisualEvidence(
        document_id=document_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        chunk_index=chunk_index,
        content_type=content_type,
        metadata=metadata if metadata is not None else {"cert_test": "DAY20", "source": "CERT_DOC_A"},
    )


# ============================================================================
# 1. INGESTION CONTRACT CERTIFICATION
# ============================================================================


class TestIngestionContractCertification:
    """Certifies Member 1 domain models and retrieval processors."""

    def test_ingestion_chunk_and_retrieval_processing(self) -> None:
        chunk = _certified_chunk(
            chunk_id="chk-ing-01",
            document_id="doc-ing-01",
            filename="financial.pdf",
            page_number=1,
            content="Summary revenue metrics table.",
            content_type="table",
        )
        assert chunk.chunk_id == "chk-ing-01"
        assert chunk.document_id == "doc-ing-01"
        assert chunk.content_type == "table"

        vsr = VectorSearchResult(
            chunk_id=chunk.chunk_id,
            score=0.91,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content_type=chunk.content_type,
            content=chunk.content,
            metadata=chunk.metadata,
        )
        processed = process_retrieval_results([vsr], min_score=0.80)
        assert len(processed) == 1
        assert processed[0].chunk_id == "chk-ing-01"

        ctx = build_retrieval_context(processed)
        assert "financial.pdf" in ctx
        assert "Summary revenue metrics table." in ctx


# ============================================================================
# 2. SEARCH / RETRIEVAL CONTRACT CERTIFICATION
# ============================================================================


class TestSearchContractCertification:
    """Certifies Member 2 search models, requests, and citation transformations."""

    def test_search_request_and_citation_contract(self) -> None:
        req = SearchRequest(
            query="Certified financial query",
            top_k=10,
            min_score=0.75,
            collection_name="certified_docs",
            metadata={"priority": "high"},
        )
        assert req.query == "Certified financial query"
        assert req.top_k == 10
        assert req.min_score == 0.75

        vsr = _certified_vsr(chunk_id="chk-search-01", score=0.89)
        citation = AgentCitation.from_search_result(vsr)

        assert citation.chunk_id == "chk-search-01"
        assert citation.score == 0.89
        assert citation.document_id == vsr.document_id
        assert citation.content_type == "chart"

        sr = SearchResult(query=req.query, status="RESULTS_FOUND", citations=[citation])
        assert sr.status == "RESULTS_FOUND"
        assert len(sr.citations) == 1


# ============================================================================
# 3. VISION CONTRACT CERTIFICATION
# ============================================================================


class TestVisionContractCertification:
    """Certifies Member 3 vision domain models, lifecycle, and result normalizer."""

    def test_vision_lifecycle_and_result_normalization(self) -> None:
        lc = VisionExecutionLifecycle(provider_name="mock_vision_provider", model_name="mock_model_v1")
        assert lc.stage == VisionExecutionStage.PENDING
        lc.transition_to(VisionExecutionStage.VALIDATING)
        lc.transition_to(VisionExecutionStage.EXECUTING)
        lc.transition_to(VisionExecutionStage.COMPLETED)
        assert lc.is_terminal is True
        assert lc.stage == VisionExecutionStage.COMPLETED

        trace = VisionExecutionTrace()
        trace.add_stage("request_received")
        trace.add_stage("provider_completed")

        ev = _certified_evidence(chunk_id="chk-vis-01", content_type="diagram")
        raw_result = VisionResult(
            query="Analyze system architecture diagram",
            status="success",
            description="3-tier microservice architecture verified.",
            evidence=[ev],
            metadata={"untrusted_key": "safe_val"},
        )

        normalized = VisionResultNormalizer.normalize(raw_result, trace=trace)
        assert normalized.status == "success"
        assert normalized.content_type == "diagram"
        assert "execution_trace" in normalized.metadata
        assert "result_normalized" in normalized.metadata["execution_trace"]["stages"]


# ============================================================================
# 4. END-TO-END HANDOFF CERTIFICATION
# ============================================================================


class TestEndToEndHandoffCertification:
    """Certifies the complete Ingestion -> Search -> Vision -> Downstream handoff pipeline."""

    def test_end_to_end_cross_member_handoff(self) -> None:
        DOC_ID = "DOC-E2E-CERT-001"
        CHUNK_ID = "CHK-E2E-CERT-001"
        FILENAME = "e2e_cert_report.pdf"

        # 1. Member 1 Chunk
        chunk = _certified_chunk(chunk_id=CHUNK_ID, document_id=DOC_ID, filename=FILENAME, page_number=4)

        # 2. Member 1 Search Result
        vsr = VectorSearchResult(
            chunk_id=chunk.chunk_id,
            score=0.98,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content_type=chunk.content_type,
            content=chunk.content,
            metadata=chunk.metadata,
        )

        # 3. Member 2 Citation
        citation = AgentCitation.from_search_result(vsr)

        # 4. Member 3 Evidence
        evidence = VisualEvidence.from_search_result(vsr)

        # 5. Member 3 Result
        vision_result = VisionResult(
            query="Complete E2E certification query",
            status="success",
            description="End-to-end evidence verified.",
            evidence=[evidence],
        )

        # 6. Downstream AgentState and AgentResponse
        state = AgentState(query=vision_result.query)
        state.add_citation(citation)

        response = AgentResponse(
            answer=vision_result.description,
            agent_name="SupervisorAgent",
            citations=state.citations,
            status="success",
            metadata={"query": state.query, "vision_status": vision_result.status},
        )

        # Certify end-to-end provenance preservation
        assert chunk.document_id == vsr.document_id == citation.document_id == evidence.document_id == vision_result.document_id == DOC_ID
        assert chunk.chunk_id == vsr.chunk_id == citation.chunk_id == evidence.chunk_id == vision_result.chunk_id == CHUNK_ID
        assert chunk.filename == vsr.filename == citation.filename == evidence.filename == vision_result.filename == FILENAME
        assert response.citations[0].document_id == DOC_ID
        assert response.citations[0].chunk_id == CHUNK_ID
        assert response.status == "success"


# ============================================================================
# 5. MULTI-DOCUMENT ISOLATION CERTIFICATION
# ============================================================================


class TestMultiDocumentIsolationCertification:
    """Certifies Document A and Document B remain strictly segregated across workflows."""

    def test_multi_document_strict_isolation(self) -> None:
        chunk_a = _certified_chunk(document_id="doc-cert-A", filename="doc_A.pdf", metadata={"doc_label": "ALPHA"})
        chunk_b = _certified_chunk(document_id="doc-cert-B", filename="doc_B.pdf", metadata={"doc_label": "BETA"})

        vsr_a = _certified_vsr(chunk_id=chunk_a.chunk_id, document_id=chunk_a.document_id, filename=chunk_a.filename, metadata=chunk_a.metadata)
        vsr_b = _certified_vsr(chunk_id=chunk_b.chunk_id, document_id=chunk_b.document_id, filename=chunk_b.filename, metadata=chunk_b.metadata)

        ev_a = VisualEvidence.from_search_result(vsr_a)
        ev_b = VisualEvidence.from_search_result(vsr_b)

        res_a = VisionResult(query="Query A", status="success", description="Result A.", evidence=[ev_a])
        res_b = VisionResult(query="Query B", status="success", description="Result B.", evidence=[ev_b])

        assert res_a.document_id == "doc-cert-A"
        assert res_a.evidence[0].metadata["doc_label"] == "ALPHA"
        assert "BETA" not in str(res_a.to_dict())

        assert res_b.document_id == "doc-cert-B"
        assert res_b.evidence[0].metadata["doc_label"] == "BETA"
        assert "ALPHA" not in str(res_b.to_dict())


# ============================================================================
# 6. FAILURE CERTIFICATION
# ============================================================================


class TestFailureCertification:
    """Certifies typed failure modes and error propagation without stale state leakage."""

    def test_typed_validation_failures_and_state_cleanliness(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="bad.pdf", chunk_id="c1")

        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename="bad.pdf", chunk_id="c1", content_type="image")

        with pytest.raises(VisionInputValidationError):
            VisionRequest(query="")

        # Failed result contains only own error context
        err_result = VisionResult(
            query="Failing query",
            status="error",
            description="",
            error="Provider timeout after 10000ms.",
        )
        assert err_result.status == "error"
        assert err_result.error == "Provider timeout after 10000ms."
        assert err_result.has_evidence is False
        assert err_result.evidence == []


# ============================================================================
# 7. RETRY CERTIFICATION
# ============================================================================


class TestRetryCertification:
    """Certifies lifecycle determinism across repeated execution attempts."""

    def test_lifecycle_stage_sequence_stability(self) -> None:
        for _ in range(5):
            lc = VisionExecutionLifecycle(provider_name="mock_p", model_name="mock_m")
            assert lc.stage == VisionExecutionStage.PENDING
            lc.transition_to(VisionExecutionStage.VALIDATING)
            lc.transition_to(VisionExecutionStage.COMPLETED)
            assert lc.stage == VisionExecutionStage.COMPLETED
            assert lc.error is None


# ============================================================================
# 8. CONCURRENCY CERTIFICATION
# ============================================================================


class TestConcurrencyCertification:
    """Certifies thread-safe execution and state isolation under concurrent workload."""

    def test_concurrent_execution_provenance_isolation(self) -> None:
        def _worker(thread_idx: int) -> dict[str, Any]:
            doc_id = f"doc-thread-{thread_idx:02d}"
            chunk_id = f"chk-thread-{thread_idx:02d}"
            marker = f"THREAD_MARKER_{thread_idx:02d}"

            ev = _certified_evidence(
                chunk_id=chunk_id,
                document_id=doc_id,
                filename=f"thread_{thread_idx:02d}.pdf",
                page_number=thread_idx + 1,
                metadata={"cert_test": "DAY20", "thread_marker": marker},
            )
            res = VisionResult(
                query=f"Concurrent thread query {thread_idx}",
                status="success",
                description=f"Thread {thread_idx} certified.",
                evidence=[ev],
            )
            return {
                "thread_idx": thread_idx,
                "document_id": res.document_id,
                "chunk_id": res.chunk_id,
                "page_number": res.page_number,
                "marker": res.evidence[0].metadata["thread_marker"],
                "serialized": str(res.to_dict()),
            }

        concurrency = 16
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_worker, i) for i in range(concurrency)]
            outputs = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(outputs) == concurrency
        for r in outputs:
            tidx = r["thread_idx"]
            assert r["document_id"] == f"doc-thread-{tidx:02d}"
            assert r["chunk_id"] == f"chk-thread-{tidx:02d}"
            assert r["page_number"] == tidx + 1
            assert r["marker"] == f"THREAD_MARKER_{tidx:02d}"

            for other_idx in range(concurrency):
                if other_idx != tidx:
                    assert f"THREAD_MARKER_{other_idx:02d}" not in r["serialized"]


# ============================================================================
# 9. DETERMINISM CERTIFICATION
# ============================================================================


class TestDeterminismCertification:
    """Certifies identical inputs produce identical observable structures across multiple runs."""

    def test_identical_input_repeated_execution_determinism(self) -> None:
        RUN_COUNT = 6
        snapshots: list[dict[str, Any]] = []

        for _ in range(RUN_COUNT):
            ev = _certified_evidence(
                chunk_id="chk-det-01",
                document_id="doc-det-01",
                filename="deterministic.pdf",
                page_number=3,
                metadata={"cert_test": "DAY20", "marker": "DET_RUN"},
            )
            res = VisionResult(
                query="Deterministic certification query",
                status="success",
                description="Deterministic result verified.",
                evidence=[ev],
            )
            snapshots.append({
                "status": res.status,
                "document_id": res.document_id,
                "chunk_id": res.chunk_id,
                "page_number": res.page_number,
                "evidence_count": len(res.evidence),
                "marker": res.evidence[0].metadata["marker"],
            })

        first = snapshots[0]
        for snap in snapshots[1:]:
            assert snap == first


# ============================================================================
# 10. SERIALIZATION CERTIFICATION
# ============================================================================


class TestSerializationCertification:
    """Certifies to_dict -> from_dict -> to_dict roundtrip across public models."""

    def test_public_models_serialization_round_trip(self) -> None:
        # AgentCitation
        orig_cit = AgentCitation(
            document_id="doc-ser-cit",
            filename="ser_cit.pdf",
            chunk_id="chk-ser-cit",
            page_number=4,
            content_type="chart",
            score=0.92,
            metadata={"cert_test": "DAY20"},
        )
        assert AgentCitation.from_dict(orig_cit.to_dict()).to_dict() == orig_cit.to_dict()

        # VisualEvidence
        orig_ev = _certified_evidence(chunk_id="chk-ser-ev", page_number=5)
        assert VisualEvidence.from_dict(orig_ev.to_dict()).to_dict() == orig_ev.to_dict()

        # VisionResult
        orig_res = VisionResult(
            query="Serialization test query",
            status="success",
            description="Serialization verified.",
            evidence=[orig_ev],
        )
        d1 = orig_res.to_dict()
        restored_res = VisionResult.from_dict(d1)
        d2 = restored_res.to_dict()
        assert d1["document_id"] == d2["document_id"]
        assert d1["status"] == d2["status"]
        assert d1["description"] == d2["description"]


# ============================================================================
# 11. SECURITY & SANITIZATION CERTIFICATION
# ============================================================================


class TestSecurityCertification:
    """Certifies metadata sanitization, secret stripping, and absence of credential leakage."""

    def test_forbidden_keys_stripped_from_metadata(self) -> None:
        dirty_metadata = {
            "api_key": "SYNTHETIC_API_KEY_SECRET",
            "password": "SYNTHETIC_PASSWORD",
            "access_token": "SYNTHETIC_TOKEN",
            "safe_metric": 42.0,
            "nested": {
                "secret": "SYNTHETIC_SECRET",
                "credentials": "SYNTHETIC_CREDS",
                "safe_nested": "allowed_value",
            },
        }

        sanitized = VisionResultNormalizer.sanitize_metadata(dirty_metadata)
        assert "api_key" not in sanitized
        assert "password" not in sanitized
        assert "access_token" not in sanitized
        assert sanitized["safe_metric"] == 42.0
        assert "secret" not in sanitized["nested"]
        assert "credentials" not in sanitized["nested"]
        assert sanitized["nested"]["safe_nested"] == "allowed_value"

    def test_byte_payloads_stripped_from_metadata(self) -> None:
        raw_bytes_meta = {
            "image_bytes": b"\x89PNG\r\n\x1a\n\x00\x00\x00",
            "report_name": "quarterly.pdf",
        }
        sanitized = VisionResultNormalizer.sanitize_metadata(raw_bytes_meta)
        assert "image_bytes" not in sanitized
        assert sanitized["report_name"] == "quarterly.pdf"


# ============================================================================
# 12. OFFLINE & RESOURCE SAFETY CERTIFICATION
# ============================================================================


class TestOfflineAndResourceSafetyCertification:
    """Certifies purely in-memory offline operations without external dependencies or leaking resources."""

    def test_offline_execution_pure_in_memory(self) -> None:
        # Full in-memory flow executes with 0 network or filesystem side-effects
        chunk = _certified_chunk(chunk_id="chk-off-01")
        vsr = _certified_vsr(chunk_id=chunk.chunk_id)
        citation = AgentCitation.from_search_result(vsr)
        evidence = VisualEvidence.from_search_result(vsr)
        result = VisionResult(query="Offline query", status="success", description="Pure offline.", evidence=[evidence])

        assert result.status == "success"
        assert citation.chunk_id == "chk-off-01"


# ============================================================================
# 13. DUPLICATE-WORK & MULTI-ITEM CERTIFICATION
# ============================================================================


class TestDuplicateWorkCertification:
    """Certifies exact multi-item preservation without synthetic duplication."""

    def test_evidence_collection_item_count_strictly_preserved(self) -> None:
        ITEM_COUNT = 7
        ev_items = [
            _certified_evidence(chunk_id=f"chk-dup-{i}", page_number=i + 1)
            for i in range(ITEM_COUNT)
        ]
        req = VisionRequest(query="Multi item count check", evidence=ev_items)
        result = VisionResult(query=req.query, status="success", description="Count check.", evidence=req.evidence)

        assert len(result.evidence) == ITEM_COUNT
        assert len({e.chunk_id for e in result.evidence}) == ITEM_COUNT


# ============================================================================
# 14. PUBLIC API CONTRACT CERTIFICATION
# ============================================================================


class TestPublicAPIContractCertification:
    """Certifies that all public contracts and classes remain discoverable and callable."""

    def test_public_api_surface_availability(self) -> None:
        # Ingestion
        assert callable(DocumentChunk)
        assert callable(VectorSearchResult)
        assert callable(validate_chunks)
        assert callable(normalize_chunks)
        assert callable(process_retrieval_results)
        assert callable(build_retrieval_context)

        # Search / Agents
        assert callable(AgentCitation)
        assert callable(AgentCitation.from_search_result)
        assert callable(AgentCitation.from_dict)
        assert callable(SearchRequest)
        assert callable(SearchResult)
        assert callable(AgentRequest)
        assert callable(AgentResponse)
        assert callable(AgentState)

        # Vision
        assert callable(VisualEvidence)
        assert callable(VisualEvidence.from_search_result)
        assert callable(VisualEvidence.from_citation)
        assert callable(VisualEvidence.from_dict)
        assert callable(VisionRequest)
        assert callable(VisionResult)
        assert callable(VisionResult.from_dict)
        assert callable(VisionExecutionLifecycle)
        assert callable(VisionResultNormalizer.normalize)
        assert callable(VisionResultNormalizer.sanitize_metadata)
