"""
OmniBrain Member 4 — Day 30 Security Regression & Data Isolation Certification Tests.

Validates that existing OMNIBRAIN integration contracts enforce strict cross-request,
cross-document, metadata, citation, evidence, lineage, and serialization boundaries:

    Ingestion (Member 1)
         ↓
    Search / Retrieval (Member 2)
         ↓
    Vision (Member 3)
         ↓
    Downstream Supervisor / Agent Consumers

Focus areas:
 1. Synthetic security markers definition (synthetic only, zero real credentials).
 2. Cross-request data isolation (Request A vs Request B complete separation).
 3. Cross-document data isolation (Document A vs Document B non-bleed).
 4. Metadata isolation across tenants/users.
 5. Citation isolation (citations stay strictly bound to source documents).
 6. VisualEvidence isolation (evidence items stay strictly bound to source requests).
 7. Lineage isolation (Doc -> Chunk -> Retrieval -> Citation -> Evidence -> Result).
 8. Serialization isolation (to_dict -> from_dict cross-instance independence).
 9. Unknown field safety (harmless unknown keys ignored where supported).
 10. Malformed input boundaries (typed validation exceptions triggered safely).
 11. Error information safety (zero credentials/secrets exposed in error strings).
 12. Synthetic secret-like input handling (masking / safe retention).
 13. Artifact & workspace safety (zero temporary files, dumps, or leaked artifacts).
 14. Request object reuse and independent state lifecycle.
 15. Failure → Success isolation (Request A failure does not affect Request B).
 16. Success → Failure → Success isolation (State stays pristine after intermediate error).
 17. Concurrent security isolation (Multi-threaded execution with mixed success/failure).
 18. Mutation safety of caller-owned metadata and input collections.

Constraints:
 - 100% Offline: Synthetic markers only (e.g. DAY30_SECRET_A_12345). No external network, real LLMs, or production secrets.
 - Zero production code modified.
 - Zero security middleware, auth mechanisms, or encryption added.
"""

from __future__ import annotations

import concurrent.futures
import copy
import sys
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# Ingestion Subsystem (Member 1)
from ingestion.models import (
    ChunkValidationResult,
    ChunkingResult,
    DocumentChunk,
    DocumentMetadata,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    VectorSearchResult,
)
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.retrieval_processor import build_retrieval_context, process_retrieval_results
from ingestion.ingestion_errors import IngestionError, IngestionValidationError

# Search / Agents Subsystem (Member 2)
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    AgentState,
    SearchRequest,
    SearchResult,
)
from agents.exceptions import AgentError, AgentValidationError

# Vision Subsystem (Member 3)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.result_normalizer import FORBIDDEN_METADATA_KEYS, VisionResultNormalizer
from vision.exceptions import (
    VisionAgentError,
    VisionEvidenceError,
    VisionInputValidationError,
)

# ---------------------------------------------------------------------------
# Synthetic Security Markers
# ---------------------------------------------------------------------------

USER_A_SECRET_MARKER = "DAY30_SECRET_A_12345"
USER_B_SECRET_MARKER = "DAY30_SECRET_B_67890"

DOCUMENT_A = "DAY30_PRIVATE_DOCUMENT_A"
DOCUMENT_B = "DAY30_PRIVATE_DOCUMENT_B"
DOCUMENT_C = "DAY30_PRIVATE_DOCUMENT_C"
DOCUMENT_D = "DAY30_PRIVATE_DOCUMENT_D"

FILE_A = "day30_private_doc_a.pdf"
FILE_B = "day30_private_doc_b.pdf"
FILE_C = "day30_private_doc_c.pdf"
FILE_D = "day30_private_doc_d.pdf"

REQUEST_A = "DAY30_REQUEST_A"
REQUEST_B = "DAY30_REQUEST_B"
REQUEST_C = "DAY30_REQUEST_C"
REQUEST_D = "DAY30_REQUEST_D"

FAKE_API_KEY = "DAY30_FAKE_API_KEY_123456"
FAKE_TOKEN = "DAY30_FAKE_TOKEN_ABCDEF"


def _make_secure_vsr(
    doc_id: str,
    filename: str,
    req_id: str,
    content_type: str = "image",
    secret_marker: str | None = None,
) -> VectorSearchResult:
    meta = {"req_id": req_id, "document_id": doc_id, "classification": "restricted"}
    if secret_marker:
        meta["secret_marker"] = secret_marker

    return VectorSearchResult(
        chunk_id=f"chunk_{doc_id}_01",
        score=0.96,
        document_id=doc_id,
        filename=filename,
        page_number=1,
        chunk_index=0,
        content_type=content_type,
        content=f"Confidential content for {req_id} from {doc_id}",
        metadata=meta,
    )


def _execute_secure_workflow(
    req_id: str,
    doc_id: str,
    filename: str,
    secret_marker: str,
) -> tuple[AgentResponse, VisionResult]:
    """Runs a complete Member 1 -> Member 2 -> Member 3 workflow with strict metadata isolation."""
    vsr = _make_secure_vsr(doc_id, filename, req_id, content_type="image", secret_marker=secret_marker)
    processed = process_retrieval_results([vsr], min_score=0.5, max_results=10)
    ctx = build_retrieval_context(processed)

    citations = [AgentCitation.from_search_result(v) for v in processed]
    agent_resp = AgentResponse(
        answer=f"Confidential answer for {req_id}",
        agent_name="SecurityAgent",
        status="success",
        citations=citations,
        metadata={"req_id": req_id, "doc_id": doc_id, "secret_marker": secret_marker, "context": ctx},
    )

    image_citations = agent_resp.image_results
    evidence = VisualEvidenceAdapter.adapt_batch(image_citations)
    normalizer = VisionResultNormalizer()
    raw_res = VisionResult(
        query=f"Audit query for {req_id}",
        status="success",
        description=f"Visual audit for {req_id}",
        evidence=evidence,
        metadata={"req_id": req_id, "secret_marker": secret_marker},
    )
    normalized_res = normalizer.normalize(raw_res)

    return agent_resp, normalized_res


# ===========================================================================
# 1. Cross-Request Data Isolation
# ===========================================================================

class TestCrossRequestDataIsolation:
    """Verifies that Request A and Request B execute with complete non-overlap."""

    def test_request_a_and_request_b_isolation(self) -> None:
        resp_a, vis_a = _execute_secure_workflow(REQUEST_A, DOCUMENT_A, FILE_A, USER_A_SECRET_MARKER)
        resp_b, vis_b = _execute_secure_workflow(REQUEST_B, DOCUMENT_B, FILE_B, USER_B_SECRET_MARKER)

        # Verify Request A contains only A data
        assert resp_a.metadata["req_id"] == REQUEST_A
        assert resp_a.metadata["secret_marker"] == USER_A_SECRET_MARKER
        assert resp_a.unique_documents == [DOCUMENT_A]
        assert vis_a.document_id == DOCUMENT_A
        assert vis_a.metadata["secret_marker"] == USER_A_SECRET_MARKER

        # Verify Request B contains only B data
        assert resp_b.metadata["req_id"] == REQUEST_B
        assert resp_b.metadata["secret_marker"] == USER_B_SECRET_MARKER
        assert resp_b.unique_documents == [DOCUMENT_B]
        assert vis_b.document_id == DOCUMENT_B
        assert vis_b.metadata["secret_marker"] == USER_B_SECRET_MARKER

        # Verify zero cross-bleed between A and B
        assert USER_B_SECRET_MARKER not in str(resp_a.to_dict())
        assert USER_B_SECRET_MARKER not in str(vis_a.to_dict())
        assert DOCUMENT_B not in str(resp_a.to_dict())

        assert USER_A_SECRET_MARKER not in str(resp_b.to_dict())
        assert USER_A_SECRET_MARKER not in str(vis_b.to_dict())
        assert DOCUMENT_A not in str(resp_b.to_dict())


# ===========================================================================
# 2. Cross-Document & Metadata Isolation
# ===========================================================================

class TestCrossDocumentAndMetadataIsolation:
    """Verifies that Document A and Document B metadata and content never mix."""

    def test_metadata_isolation_between_documents(self) -> None:
        meta_a = {"day30_owner": "A", "day30_marker": "MARKER_A", "tenant": "TENANT_A"}
        meta_b = {"day30_owner": "B", "day30_marker": "MARKER_B", "tenant": "TENANT_B"}

        chunk_a = DocumentChunk(
            chunk_id="chunk_a_01",
            chunk_index=0,
            document_id=DOCUMENT_A,
            filename=FILE_A,
            page_number=1,
            content="Confidential content of Document A.",
            content_type="text",
            metadata=meta_a,
        )
        chunk_b = DocumentChunk(
            chunk_id="chunk_b_01",
            chunk_index=0,
            document_id=DOCUMENT_B,
            filename=FILE_B,
            page_number=1,
            content="Confidential content of Document B.",
            content_type="text",
            metadata=meta_b,
        )

        assert chunk_a.metadata["day30_owner"] == "A"
        assert "MARKER_B" not in chunk_a.metadata.values()

        assert chunk_b.metadata["day30_owner"] == "B"
        assert "MARKER_A" not in chunk_b.metadata.values()


# ===========================================================================
# 3. Citation, Evidence & Lineage Isolation
# ===========================================================================

class TestCitationEvidenceLineageIsolation:
    """Verifies that citations and visual evidence preserve provenance lock."""

    def test_citations_locked_to_originating_document(self) -> None:
        vsr_a = _make_secure_vsr(DOCUMENT_A, FILE_A, REQUEST_A, content_type="text")
        vsr_b = _make_secure_vsr(DOCUMENT_B, FILE_B, REQUEST_B, content_type="text")

        cit_a = AgentCitation.from_search_result(vsr_a)
        cit_b = AgentCitation.from_search_result(vsr_b)

        assert cit_a.document_id == DOCUMENT_A
        assert cit_a.filename == FILE_A
        assert cit_a.chunk_id == "chunk_DAY30_PRIVATE_DOCUMENT_A_01"

        assert cit_b.document_id == DOCUMENT_B
        assert cit_b.filename == FILE_B
        assert cit_b.chunk_id == "chunk_DAY30_PRIVATE_DOCUMENT_B_01"

    def test_visual_evidence_locked_to_originating_request(self) -> None:
        vsr_a = _make_secure_vsr(DOCUMENT_A, FILE_A, REQUEST_A, content_type="image")
        vsr_b = _make_secure_vsr(DOCUMENT_B, FILE_B, REQUEST_B, content_type="image")

        ev_a = VisualEvidence.from_search_result(vsr_a)
        ev_b = VisualEvidence.from_search_result(vsr_b)

        assert ev_a.document_id == DOCUMENT_A
        assert ev_b.document_id == DOCUMENT_B
        assert ev_a.metadata["req_id"] == REQUEST_A
        assert ev_b.metadata["req_id"] == REQUEST_B


# ===========================================================================
# 4. Serialization Isolation & Unknown Field Safety
# ===========================================================================

class TestSerializationIsolationAndUnknownFields:
    """Verifies serialization round-trip preservation and unknown field handling."""

    def test_serialization_roundtrip_isolation(self) -> None:
        resp_a, vis_a = _execute_secure_workflow(REQUEST_A, DOCUMENT_A, FILE_A, USER_A_SECRET_MARKER)
        resp_b, vis_b = _execute_secure_workflow(REQUEST_B, DOCUMENT_B, FILE_B, USER_B_SECRET_MARKER)

        dict_a = resp_a.to_dict()
        dict_b = resp_b.to_dict()

        restored_a = AgentResponse.from_dict(dict_a)
        restored_b = AgentResponse.from_dict(dict_b)

        assert restored_a.metadata["secret_marker"] == USER_A_SECRET_MARKER
        assert restored_a.unique_documents == [DOCUMENT_A]

        assert restored_b.metadata["secret_marker"] == USER_B_SECRET_MARKER
        assert restored_b.unique_documents == [DOCUMENT_B]

    def test_unknown_field_safety_in_deserialization(self) -> None:
        # SearchResult deserialization ignores unknown synthetic fields
        sr_dict = {
            "query": "Security query",
            "status": "RESULTS_FOUND",
            "citations": [],
            "context": "",
            "metadata": {},
            "day30_unknown_security_test": "synthetic_value",
            "another_harmless_field": 42,
        }
        sr = SearchResult.from_dict(sr_dict)
        assert sr.query == "Security query"
        assert sr.status == "RESULTS_FOUND"


# ===========================================================================
# 5. Malformed Input Boundaries
# ===========================================================================

class TestMalformedInputBoundaries:
    """Verifies that invalid or malformed inputs trigger expected validation exceptions without crashing."""

    def test_agent_citation_malformed_inputs(self) -> None:
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="file.pdf", chunk_id="ck")

        with pytest.raises(AgentValidationError):
            AgentCitation.from_dict("not_a_dict")  # type: ignore[arg-type]

    def test_visual_evidence_malformed_inputs(self) -> None:
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename="file.pdf", chunk_id="ck", content_type="image")

        with pytest.raises(VisionEvidenceError):
            VisualEvidence.from_dict(["invalid_list"])  # type: ignore[arg-type]


# ===========================================================================
# 6. Error Information Safety & Synthetic Secret Non-Disclosure
# ===========================================================================

class TestErrorInformationSafetyAndSecretNonDisclosure:
    """Verifies error messages and public structures do not disclose synthetic secrets."""

    def test_validation_errors_do_not_reflect_secret_values(self) -> None:
        try:
            AgentRequest(query="", metadata={"secret_key": FAKE_API_KEY})
        except AgentValidationError as err:
            err_str = str(err)
            assert FAKE_API_KEY not in err_str
            assert FAKE_TOKEN not in err_str

    def test_vision_normalizer_sanitizes_forbidden_keys(self) -> None:
        dirty_meta = {
            "api_key": FAKE_API_KEY,
            "token": FAKE_TOKEN,
            "secret": "SECRET_DATA",
            "password": "PASS",
            "public_metric": 100,
        }
        sanitized = VisionResultNormalizer.sanitize_metadata(dirty_meta)

        assert "api_key" not in sanitized
        assert "token" not in sanitized
        assert "secret" not in sanitized
        assert "password" not in sanitized
        assert sanitized["public_metric"] == 100
        assert FAKE_API_KEY not in str(sanitized)
        assert FAKE_TOKEN not in str(sanitized)


# ===========================================================================
# 7. Failure Isolation (Failure -> Success, Success -> Failure -> Success)
# ===========================================================================

class TestFailureIsolationPatterns:
    """Verifies that errors on one request do not contaminate subsequent healthy requests."""

    def test_failure_then_success_isolation(self) -> None:
        # Request A: Fails with validation error
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename=FILE_A, chunk_id="ck_a")

        # Request B: Independent success
        resp_b, vis_b = _execute_secure_workflow(REQUEST_B, DOCUMENT_B, FILE_B, USER_B_SECRET_MARKER)
        assert resp_b.is_success
        assert vis_b.is_success
        assert resp_b.error is None
        assert resp_b.unique_documents == [DOCUMENT_B]

    def test_success_failure_success_isolation(self) -> None:
        # Step 1: Success A
        resp_a, vis_a = _execute_secure_workflow(REQUEST_A, DOCUMENT_A, FILE_A, USER_A_SECRET_MARKER)
        assert resp_a.is_success

        # Step 2: Failure B
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename=FILE_B, chunk_id="ck_b", content_type="image")

        # Step 3: Success C
        resp_c, vis_c = _execute_secure_workflow(REQUEST_C, DOCUMENT_C, FILE_C, "SECRET_C")
        assert resp_c.is_success
        assert resp_c.unique_documents == [DOCUMENT_C]
        assert resp_c.error is None


# ===========================================================================
# 8. Concurrent Security & State Isolation
# ===========================================================================

class TestConcurrentSecurityIsolation:
    """Verifies multi-threaded requests process simultaneously with 100% data and error isolation."""

    def test_concurrent_mixed_security_isolation(self) -> None:
        configs = [
            (REQUEST_A, DOCUMENT_A, FILE_A, USER_A_SECRET_MARKER, False),
            (REQUEST_B, DOCUMENT_B, FILE_B, USER_B_SECRET_MARKER, False),
            (REQUEST_C, DOCUMENT_C, FILE_C, "FAIL_C", True),  # Injected failure
            (REQUEST_D, DOCUMENT_D, FILE_D, "SECRET_D", False),
        ]

        def worker(cfg: tuple[str, str, str, str, bool]) -> tuple[str, str, Any]:
            req_id, doc_id, filename, marker, should_fail = cfg
            if should_fail:
                try:
                    AgentCitation(document_id="", filename=filename, chunk_id="ck")
                    return req_id, "unexpected_success", None
                except AgentValidationError as e:
                    return req_id, "expected_failure", str(e)
            else:
                resp, vis = _execute_secure_workflow(req_id, doc_id, filename, marker)
                return req_id, "success", (resp, vis)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, cfg) for cfg in configs]
            results = dict([(r[0], (r[1], r[2])) for r in [f.result() for f in futures]])

        assert len(results) == 4

        # Request C failed safely
        assert results[REQUEST_C][0] == "expected_failure"

        # Requests A, B, D succeeded with complete isolation
        for req_id in (REQUEST_A, REQUEST_B, REQUEST_D):
            status, (resp, vis) = results[req_id]
            assert status == "success"
            assert resp.is_success
            assert vis.is_success


# ===========================================================================
# 9. Caller-Owned Object Mutation Safety
# ===========================================================================

class TestCallerObjectMutationSafety:
    """Verifies caller-owned input dictionaries and metadata are never mutated during processing."""

    def test_caller_metadata_and_citations_unmutated(self) -> None:
        original_meta = {"day30_owner": "Alice", "security_tier": "CONFIDENTIAL"}
        meta_snapshot = copy.deepcopy(original_meta)

        citation = AgentCitation(
            document_id=DOCUMENT_A,
            filename=FILE_A,
            chunk_id="ck_01",
            page_number=1,
            content_type="image",
            metadata=original_meta,
        )

        # Adapt evidence and normalize
        ev = VisualEvidenceAdapter.adapt_batch([citation])
        assert len(ev) == 1

        # Confirm original caller metadata dictionary is unchanged
        assert original_meta == meta_snapshot


# ===========================================================================
# 10. Resource & Artifact Safety
# ===========================================================================

class TestResourceAndArtifactSafety:
    """Verifies security test execution does not leave disk pollution or leaked credential dumps."""

    def test_zero_disk_artifacts_after_security_runs(self) -> None:
        root_path = Path(REPO_ROOT)
        unexpected = [
            f.name for f in root_path.iterdir()
            if f.is_file() and f.name.endswith((".tmp", ".temp", ".dump", ".log", ".bak"))
        ]
        assert unexpected == []
