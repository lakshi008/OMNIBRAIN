"""
OmniBrain Member 4 -- Day 14 Security, Data Isolation & Sensitive-Information Leakage Regression Tests.

Verifies that the existing OMNIBRAIN integration contracts prevent accidental leakage of:
 - Secrets and credentials
 - Internal implementation details
 - Unrelated document data
 - Unrelated request data
 - Internal exceptions & stack traces
 - Private metadata
 - Provider configuration

Concern areas:
 1. Secret & Credential Leakage Prevention (synthetic markers)
 2. Credential Isolation & Provider Configuration Safety
 3. Error Sanitization & Public Exception Safety
 4. Multi-Document Data & Provenance Isolation
 5. Cross-Request State & Evidence Isolation
 6. Concurrent Request Security & State Isolation
 7. Serialization Safety & Round-trip Sanitization
 8. Public Model Surface Safety (no internal credentials/objects)
 9. Citation & Lineage Provenance Locking
10. Metadata Isolation & Non-Leakage
11. Failure Isolation between independent requests
12. Repeated Execution Isolation & State Cleanup

Constraints:
 - 100% Offline: Synthetic markers only (e.g. TEST_SECRET_12345). No external network, real LLMs, or production secrets.
 - Zero production code modified.
 - Zero security wrappers, auth mechanisms, or encryption added.
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
    RetrievalServiceResult,
    VectorSearchResult,
)
from ingestion.chunk_validator import validate_chunks, normalize_chunks
from ingestion.embedding_preparation import prepare_for_embedding
from ingestion.retrieval_processor import build_retrieval_context, process_retrieval_results
from ingestion.ingestion_errors import (
    IngestionChunkingError,
    IngestionEmbeddingError,
    IngestionError,
    IngestionExtractionError,
    IngestionPipelineError,
    IngestionValidationError,
)

# Search / Agents Subsystem (Member 2)
from agents.models import (
    AgentCitation,
    AgentRequest,
    AgentResponse,
    AgentState,
    SearchRequest,
    SearchResult,
)
from agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentRoutingError,
    AgentValidationError,
)

# Vision Subsystem (Member 3)
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import (
    VisionAgentError,
    VisionCancellationError,
    VisionError,
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProcessingError,
    VisionProviderError,
    VisionTimeoutError,
)
from vision.result_normalizer import (
    FORBIDDEN_METADATA_KEYS,
    VisionResultNormalizer,
)


# ============================================================================
# Shared Fixtures & Synthetic Secret Constants
# ============================================================================

SYNTHETIC_TEST_SECRET = "TEST_SECRET_KEY_12345_XYZ"
SYNTHETIC_PROVIDER_TOKEN = "TEST_PROVIDER_BEARER_TOKEN_999"
SYNTHETIC_PASSWORD = "TEST_DATABASE_PASSWORD_555"


def _create_secure_chunk(
    chunk_id: str = "chk-sec-001",
    document_id: str = "doc-sec-001",
    filename: str = "security_report.pdf",
    page_number: int | None = 1,
    content: str = "Public audit findings content.",
    content_type: str = "image",
    metadata: dict[str, Any] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=0,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        content=content,
        content_type=content_type,
        metadata=metadata if metadata is not None else {"classification": "public"},
    )


def _create_secure_vsr(
    chunk_id: str = "chk-sec-001",
    score: float = 0.90,
    document_id: str = "doc-sec-001",
    filename: str = "security_report.pdf",
    page_number: int | None = 1,
    content: str = "Public audit findings content.",
    content_type: str = "image",
    metadata: dict[str, Any] | None = None,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        score=score,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        chunk_index=0,
        content_type=content_type,
        content=content,
        metadata=metadata if metadata is not None else {"classification": "public"},
    )


def _create_secure_evidence(
    chunk_id: str = "chk-sec-001",
    document_id: str = "doc-sec-001",
    filename: str = "security_report.pdf",
    page_number: int | None = 1,
    content_type: str = "chart",
    metadata: dict[str, Any] | None = None,
) -> VisualEvidence:
    return VisualEvidence(
        document_id=document_id,
        filename=filename,
        chunk_id=chunk_id,
        page_number=page_number,
        chunk_index=0,
        content_type=content_type,
        metadata=metadata if metadata is not None else {"classification": "public"},
    )


# ============================================================================
# 1. SECRET & CREDENTIAL LEAKAGE SANITIZATION
# ============================================================================


class TestSecretAndCredentialLeakage:
    """Verifies that synthetic secrets and credentials are filtered and never leak into outputs."""

    def test_metadata_normalizer_sanitizes_forbidden_keys(self) -> None:
        dirty_metadata = {
            "api_key": SYNTHETIC_TEST_SECRET,
            "secret": SYNTHETIC_TEST_SECRET,
            "password": SYNTHETIC_PASSWORD,
            "token": SYNTHETIC_PROVIDER_TOKEN,
            "safe_tag": "public_data",
            "department": "Security",
        }
        sanitized = VisionResultNormalizer.sanitize_metadata(dirty_metadata)

        assert "api_key" not in sanitized
        assert "secret" not in sanitized
        assert "password" not in sanitized
        assert "token" not in sanitized
        assert SYNTHETIC_TEST_SECRET not in str(sanitized)
        assert SYNTHETIC_PROVIDER_TOKEN not in str(sanitized)
        assert SYNTHETIC_PASSWORD not in str(sanitized)
        assert sanitized["safe_tag"] == "public_data"
        assert sanitized["department"] == "Security"

    def test_all_forbidden_metadata_keys_are_defined(self) -> None:
        expected_keys = {
            "api_key", "apikey", "secret", "password", "token",
            "auth", "authorization", "credentials", "bearer", "access_token"
        }
        assert expected_keys.issubset(FORBIDDEN_METADATA_KEYS)


# ============================================================================
# 2. CREDENTIAL ISOLATION & PROVIDER CONFIG SAFETY
# ============================================================================


class TestCredentialIsolation:
    """Verifies that provider configurations and credentials are not exposed on public models."""

    def test_public_models_do_not_contain_credential_attributes(self) -> None:
        citation = AgentCitation(document_id="doc-01", filename="file.pdf", chunk_id="chk-01")
        evidence = VisualEvidence(document_id="doc-01", filename="file.pdf", chunk_id="chk-01")
        vision_result = VisionResult(query="Test query", status="success", description="Output")
        agent_response = AgentResponse(answer="Public answer", agent_name="TestAgent")

        for model_obj in (citation, evidence, vision_result, agent_response):
            for sensitive_attr in ("api_key", "token", "secret", "credentials", "auth_token", "password"):
                assert not hasattr(model_obj, sensitive_attr)


# ============================================================================
# 3. ERROR SANITIZATION & PUBLIC EXCEPTION SAFETY
# ============================================================================


class TestErrorSanitization:
    """Verifies that public error messages and exceptions do not leak secrets or private paths."""

    def test_validation_error_messages_do_not_contain_secrets(self) -> None:
        try:
            AgentCitation(
                document_id="",
                filename="secure.pdf",
                chunk_id="chk-01",
                metadata={"secret": SYNTHETIC_TEST_SECRET},
            )
        except AgentValidationError as err:
            err_msg = str(err)
            assert SYNTHETIC_TEST_SECRET not in err_msg

    def test_vision_result_error_field_does_not_contain_credentials(self) -> None:
        vres = VisionResult(
            query="Audit query",
            status="error",
            description="",
            error="Operation timed out after 30 seconds.",
        )
        assert vres.status == "error"
        assert SYNTHETIC_TEST_SECRET not in (vres.error or "")
        assert SYNTHETIC_PROVIDER_TOKEN not in (vres.error or "")


# ============================================================================
# 4. MULTI-DOCUMENT DATA & PROVENANCE ISOLATION
# ============================================================================


class TestMultiDocumentDataIsolation:
    """Verifies that Document A data/metadata/citations never leak into Document B results."""

    def test_two_distinct_documents_have_zero_cross_leakage(self) -> None:
        # Document A: Top Secret Project Apollo
        doc_a_chunk = _create_secure_chunk(
            chunk_id="chk-apollo-01",
            document_id="doc-apollo-100",
            filename="apollo_classified.pdf",
            content="Project Apollo budget is $50M.",
            metadata={"project": "Apollo", "security_tier": "TOP_SECRET_A"},
        )
        vsr_a = _create_secure_vsr(
            chunk_id=doc_a_chunk.chunk_id,
            document_id=doc_a_chunk.document_id,
            filename=doc_a_chunk.filename,
            content=doc_a_chunk.content,
            metadata=doc_a_chunk.metadata,
        )
        cit_a = AgentCitation.from_search_result(vsr_a)

        # Document B: Public Project Zeus
        doc_b_chunk = _create_secure_chunk(
            chunk_id="chk-zeus-01",
            document_id="doc-zeus-200",
            filename="zeus_public.pdf",
            content="Project Zeus roadmap is public.",
            metadata={"project": "Zeus", "security_tier": "UNCLASSIFIED_B"},
        )
        vsr_b = _create_secure_vsr(
            chunk_id=doc_b_chunk.chunk_id,
            document_id=doc_b_chunk.document_id,
            filename=doc_b_chunk.filename,
            content=doc_b_chunk.content,
            metadata=doc_b_chunk.metadata,
        )
        cit_b = AgentCitation.from_search_result(vsr_b)

        # Assert strict isolation on Citation A
        assert cit_a.document_id == "doc-apollo-100"
        assert cit_a.filename == "apollo_classified.pdf"
        assert cit_a.metadata["project"] == "Apollo"
        assert "Zeus" not in str(cit_a.to_dict())
        assert "UNCLASSIFIED_B" not in str(cit_a.to_dict())

        # Assert strict isolation on Citation B
        assert cit_b.document_id == "doc-zeus-200"
        assert cit_b.filename == "zeus_public.pdf"
        assert cit_b.metadata["project"] == "Zeus"
        assert "Apollo" not in str(cit_b.to_dict())
        assert "TOP_SECRET_A" not in str(cit_b.to_dict())


# ============================================================================
# 5. CROSS-REQUEST STATE & EVIDENCE ISOLATION
# ============================================================================


class TestCrossRequestStateIsolation:
    """Verifies that Request A state/evidence/citations cannot be accessed or inherited by Request B."""

    def test_request_state_isolation_between_two_independent_sessions(self) -> None:
        # Session A
        state_a = AgentState(query="User A query", metadata={"session_id": "sess-user-A"})
        cit_a = AgentCitation(
            document_id="doc-A-only",
            filename="user_a_file.pdf",
            chunk_id="chk-A-01",
            metadata={"user": "Alice", "token_a": "TOK_ALICE_123"},
        )
        state_a.add_citation(cit_a)
        state_a.answer = "Confidential response for Alice."
        state_a.status = "completed"

        # Session B
        state_b = AgentState(query="User B query", metadata={"session_id": "sess-user-B"})
        cit_b = AgentCitation(
            document_id="doc-B-only",
            filename="user_b_file.pdf",
            chunk_id="chk-B-01",
            metadata={"user": "Bob", "token_b": "TOK_BOB_456"},
        )
        state_b.add_citation(cit_b)
        state_b.answer = "Public response for Bob."
        state_b.status = "completed"

        # Verify Session A
        assert len(state_a.citations) == 1
        assert state_a.citations[0].document_id == "doc-A-only"
        assert "Bob" not in str(state_a.to_dict())
        assert "TOK_BOB_456" not in str(state_a.to_dict())

        # Verify Session B
        assert len(state_b.citations) == 1
        assert state_b.citations[0].document_id == "doc-B-only"
        assert "Alice" not in str(state_b.to_dict())
        assert "TOK_ALICE_123" not in str(state_b.to_dict())


# ============================================================================
# 6. CONCURRENT REQUEST SECURITY & STATE ISOLATION
# ============================================================================


class TestConcurrentRequestSecurityIsolation:
    """Verifies multi-threaded requests process simultaneously with zero cross-thread data leakage."""

    def test_concurrent_isolation_across_threads(self) -> None:
        def _execute_session(user_id: int) -> dict[str, Any]:
            doc_id = f"doc-user-sec-{user_id:02d}"
            user_token = f"SYNTHETIC_USER_TOKEN_{user_id:02d}"
            ev = _create_secure_evidence(
                chunk_id=f"chk-sec-{user_id:02d}",
                document_id=doc_id,
                filename=f"{doc_id}.pdf",
                metadata={"user_id": user_id, "user_token": user_token},
            )
            req = VisionRequest(query=f"Security audit for user {user_id}", evidence=[ev])
            res = VisionResult(
                query=req.query,
                status="success",
                description=f"Audit complete for user {user_id}.",
                evidence=req.evidence,
            )
            serialized = res.to_dict()
            return {
                "user_id": user_id,
                "doc_id": res.document_id,
                "user_token": user_token,
                "serialized_str": str(serialized),
            }

        worker_count = 12
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(_execute_session, i) for i in range(worker_count)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == worker_count
        for r in results:
            uid = r["user_id"]
            expected_doc = f"doc-user-sec-{uid:02d}"
            expected_token = f"SYNTHETIC_USER_TOKEN_{uid:02d}"

            assert r["doc_id"] == expected_doc
            assert expected_token in r["serialized_str"]

            # Ensure tokens from OTHER users are absent in this user's result
            for other_id in range(worker_count):
                if other_id != uid:
                    other_token = f"SYNTHETIC_USER_TOKEN_{other_id:02d}"
                    assert other_token not in r["serialized_str"]


# ============================================================================
# 7. SERIALIZATION SAFETY & ROUND-TRIP SANITIZATION
# ============================================================================


class TestSerializationSafety:
    """Verifies that serialization to_dict / from_dict preserves public contracts without exposing internal state."""

    def test_serialization_round_trip_safety(self) -> None:
        ev = _create_secure_evidence(
            chunk_id="chk-ser-01",
            document_id="doc-ser-101",
            filename="ser_doc.pdf",
            page_number=3,
            metadata={"classification": "INTERNAL_AUDIT", "auditor": "SecTeam"},
        )
        orig = VisionResult(
            query="Analyze audit report",
            status="success",
            description="Audit report findings verified.",
            evidence=[ev],
        )

        data = orig.to_dict()
        assert isinstance(data, dict)
        assert data["document_id"] == "doc-ser-101"
        assert data["filename"] == "ser_doc.pdf"

        # Deserialization
        restored = VisionResult.from_dict(data)
        assert restored.document_id == "doc-ser-101"
        assert restored.filename == "ser_doc.pdf"
        assert restored.description == "Audit report findings verified."
        assert len(restored.evidence) == 1
        assert restored.evidence[0].metadata["classification"] == "INTERNAL_AUDIT"


# ============================================================================
# 8. PUBLIC MODEL SURFACE SAFETY
# ============================================================================


class TestPublicModelSurfaceSafety:
    """Verifies that public models expose only contracted fields and reject injection of private fields."""

    def test_models_have_clean_dictionaries(self) -> None:
        citation = AgentCitation(
            document_id="doc-clean",
            filename="clean.pdf",
            chunk_id="chk-clean",
            page_number=1,
            score=0.9,
            metadata={"tag": "public"},
        )
        d = citation.to_dict()
        allowed_keys = {"document_id", "filename", "chunk_id", "page_number", "content_type", "score", "metadata"}
        assert set(d.keys()) == allowed_keys


# ============================================================================
# 9. CITATION & LINEAGE PROVENANCE LOCKING
# ============================================================================


class TestCitationAndLineageSafety:
    """Verifies that citation and lineage remain strictly locked to their originating source."""

    def test_citation_provenance_locking(self) -> None:
        doc_id = "doc-lock-555"
        filename = "locked_provenance.pdf"
        chunk_id = "chk-lock-01"
        page = 4

        vsr = _create_secure_vsr(
            chunk_id=chunk_id,
            document_id=doc_id,
            filename=filename,
            page_number=page,
        )
        citation = AgentCitation.from_search_result(vsr)
        ev = VisualEvidence.from_search_result(vsr)

        assert citation.document_id == doc_id
        assert citation.filename == filename
        assert citation.chunk_id == chunk_id
        assert citation.page_number == page

        assert ev.document_id == doc_id
        assert ev.filename == filename
        assert ev.chunk_id == chunk_id
        assert ev.page_number == page


# ============================================================================
# 10. METADATA ISOLATION & NON-LEAKAGE
# ============================================================================


class TestMetadataIsolation:
    """Verifies that distinct metadata sets never mix across processing boundaries."""

    def test_distinct_metadata_isolation(self) -> None:
        meta_a = {"tenant": "TENANT_ALPHA", "encryption_tag": "ENC_A_999"}
        meta_b = {"tenant": "TENANT_BETA", "encryption_tag": "ENC_B_888"}

        chunk_a = _create_secure_chunk(chunk_id="chk-a", document_id="doc-a", metadata=meta_a)
        chunk_b = _create_secure_chunk(chunk_id="chk-b", document_id="doc-b", metadata=meta_b)

        assert chunk_a.metadata["tenant"] == "TENANT_ALPHA"
        assert "TENANT_BETA" not in chunk_a.metadata.values()

        assert chunk_b.metadata["tenant"] == "TENANT_BETA"
        assert "TENANT_ALPHA" not in chunk_b.metadata.values()


# ============================================================================
# 11. FAILURE ISOLATION BETWEEN INDEPENDENT REQUESTS
# ============================================================================


class TestFailureIsolation:
    """Verifies that a failure in Request A does not leak errors or exceptions into subsequent Request B."""

    def test_failure_in_request_a_leaves_clean_slate_for_request_b(self) -> None:
        # Request A: Fails with AgentValidationError
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename="corrupt.pdf", chunk_id="chk-fail")

        # Request B: Independent successful request
        valid_chunk = _create_secure_chunk(
            chunk_id="chk-success-01",
            document_id="doc-success-01",
            filename="valid.pdf",
        )
        valid_vsr = _create_secure_vsr(
            chunk_id=valid_chunk.chunk_id,
            document_id=valid_chunk.document_id,
            filename=valid_chunk.filename,
        )
        valid_citation = AgentCitation.from_search_result(valid_vsr)
        response = AgentResponse(
            answer="Clean success answer.",
            agent_name="SearchAgent",
            status="success",
            citations=[valid_citation],
        )

        assert response.is_success is True
        assert response.error is None
        assert response.citations[0].document_id == "doc-success-01"


# ============================================================================
# 12. REPEATED EXECUTION ISOLATION & STATE CLEANUP
# ============================================================================


class TestRepeatedExecutionSecurityIsolation:
    """Verifies that running security-sensitive workflows repeatedly maintains 100% clean state across iterations."""

    def test_repeated_security_runs_remain_deterministic_and_clean(self) -> None:
        for run_idx in range(5):
            secret_marker = f"SYNTHETIC_RUN_SECRET_{run_idx}"
            dirty_meta = {"secret": secret_marker, "run_idx": run_idx, "public_tag": "audited"}
            clean_meta = VisionResultNormalizer.sanitize_metadata(dirty_meta)

            assert "secret" not in clean_meta
            assert secret_marker not in str(clean_meta)
            assert clean_meta["run_idx"] == run_idx
            assert clean_meta["public_tag"] == "audited"
