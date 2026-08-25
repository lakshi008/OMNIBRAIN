"""
OmniBrain Member 4 — Day 9 Security, Data Leakage & Contract Safety Tests.

Verifies that the existing OMNIBRAIN integration contracts do not leak sensitive
information, credentials, secrets, or internal implementation details across
Ingestion, Search, Vision, and Supervisor layers.

Focus areas:
1. Error information & exception safety (no secret or unhandled internal state leakage).
2. Secret & credential sanitization (enforcement of FORBIDDEN_METADATA_KEYS stripping).
3. Cross-request data leakage prevention (Request A data never bleeds into Request B).
4. Cross-document provenance isolation (Document A evidence never appears in Document B).
5. Serialization safety (roundtrip to_dict() and from_dict() without private state leaks).
6. Public API contract enforcement (typed, validated dataclass surfaces).
7. Lineage & citation provenance locking.
8. 100% offline, deterministic, side-effect-free execution using test doubles.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path for test runners executing this file directly
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

from agents.exceptions import AgentValidationError
from agents.models import AgentCitation, AgentRequest, AgentResponse, AgentState, SearchResult
from ingestion.models import DocumentChunk, VectorSearchResult
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import VisionEvidenceError, VisionInputValidationError
from vision.models import VisionRequest, VisionResult, VisualEvidence
from vision.result_normalizer import (
    FORBIDDEN_METADATA_KEYS,
    VisionResultNormalizer,
)


# ============================================================================
# 1. SECRET & CREDENTIAL LEAKAGE SANITIZATION
# ============================================================================


class TestSecretAndCredentialSanitization:
    """Verifies that sensitive credentials and forbidden keys are sanitized across all boundaries."""

    def test_forbidden_keys_fully_stripped_from_metadata(self) -> None:
        """Verify VisionResultNormalizer.sanitize_metadata strips all forbidden key categories."""
        raw_dirty_metadata = {
            "api_key": "TEST_KEY_VALUE_12345",
            "apikey": "TEST_KEY_VALUE_67890",
            "secret": "SUPER_SECRET_CLIENT_SECRET",
            "password": "DATABASE_PASSWORD_TEST",
            "token": "SESSION_AUTH_TOKEN_TEST",
            "auth": "BEARER_AUTH_TEST",
            "authorization": "Bearer TEST_TOKEN",
            "credentials": {"user": "admin", "pass": "secret"},
            "bearer": "BEARER_TOKEN_VALUE",
            "image_bytes": b"\x89PNG\r\n\x1a\n_RAW_BYTES_",
            "base64": "data:image/png;base64,iVBORw0KGgo...",
            "raw_request_headers": {"Authorization": "Bearer 123"},
            "access_token": "ACCESS_TOKEN_123",
            "safe_metric": "latency_ms",
            "latency_ms": 42,
            "nested_clean": {"status": "ok", "nested_dirty": {"password": "leak"}},
        }

        sanitized = VisionResultNormalizer.sanitize_metadata(raw_dirty_metadata)

        for forbidden_key in FORBIDDEN_METADATA_KEYS:
            assert forbidden_key not in sanitized, f"Forbidden key '{forbidden_key}' leaked in metadata!"

        # Verify safe keys preserved
        assert sanitized["safe_metric"] == "latency_ms"
        assert sanitized["latency_ms"] == 42
        assert sanitized["nested_clean"]["status"] == "ok"
        assert "password" not in sanitized["nested_clean"]["nested_dirty"]

    def test_vision_result_normalization_strips_forbidden_keys(self) -> None:
        """Verify VisionResultNormalizer.normalize removes secrets when producing VisionResult."""
        candidate_result = {
            "query": "What is the chart trend?",
            "status": "success",
            "description": "Upward revenue trend.",
            "metadata": {
                "api_key": "TEST_API_KEY_LEAK",
                "secret": "SECRET_LEAK",
                "model_version": "v1.2",
            },
        }

        normalized = VisionResultNormalizer.normalize(candidate_result)
        assert "api_key" not in normalized.metadata
        assert "secret" not in normalized.metadata
        assert normalized.metadata["model_version"] == "v1.2"
        assert normalized.description == "Upward revenue trend."


# ============================================================================
# 2. ERROR & EXCEPTION INFORMATION SAFETY
# ============================================================================


class TestErrorAndExceptionInformationSafety:
    """Verifies that domain exceptions and error strings do not leak raw credentials."""

    def test_validation_error_messages_are_safe_and_structured(self) -> None:
        """Verify AgentValidationError produces clear error messages without raw system dumps."""
        with pytest.raises(AgentValidationError) as excinfo:
            AgentCitation(
                document_id="",  # empty
                filename="file.pdf",
                chunk_id="c1",
                content_type="image",
            )
        err_msg = str(excinfo.value)
        assert "document_id must be a non-empty string" in err_msg
        assert "password" not in err_msg.lower()
        assert "token" not in err_msg.lower()

    def test_vision_result_error_status_clean_propagation(self) -> None:
        """Verify VisionResult error status records safe error string for supervisor."""
        safe_error_msg = "Vision provider connection timed out after 30s."
        err_result = VisionResult(
            query="Analyze degraded figure",
            status="error",
            description="",
            error=safe_error_msg,
            metadata={"attempt": 3},
        )

        assert err_result.is_error is True
        assert err_result.error == safe_error_msg
        assert err_result.description == ""


# ============================================================================
# 3. CROSS-REQUEST DATA ISOLATION
# ============================================================================


class TestCrossRequestDataIsolation:
    """Verifies that separate search, vision, and supervisor requests have zero cross-talk."""

    def test_cross_request_content_and_metadata_isolation(self) -> None:
        """Verify Request A and Request B maintain completely disjoint state."""
        # Request A
        cit_a = AgentCitation(
            document_id="doc-confidential-A",
            filename="confidential_A.pdf",
            chunk_id="chk-A-001",
            page_number=1,
            content_type="chart",
            metadata={"secret_project": "Project-Alpha", "classification": "TopSecret"},
        )
        ev_a = VisualEvidenceAdapter.adapt_citation(cit_a)
        res_a = VisionResult(
            query="Summarize Project Alpha chart",
            status="success",
            description="Alpha chart summary.",
            evidence=[ev_a],
            metadata={"request_id": "REQ-A"},
        )

        # Request B
        cit_b = AgentCitation(
            document_id="doc-public-B",
            filename="public_B.pdf",
            chunk_id="chk-B-002",
            page_number=2,
            content_type="diagram",
            metadata={"public_info": "GeneralOverview", "classification": "Public"},
        )
        ev_b = VisualEvidenceAdapter.adapt_citation(cit_b)
        res_b = VisionResult(
            query="Summarize Public diagram",
            status="success",
            description="Public diagram summary.",
            evidence=[ev_b],
            metadata={"request_id": "REQ-B"},
        )

        # Verify A contains zero B data
        assert "Public" not in str(res_a.to_dict())
        assert "public_B.pdf" not in str(res_a.to_dict())
        assert "chk-B-002" not in str(res_a.to_dict())

        # Verify B contains zero A data
        assert "TopSecret" not in str(res_b.to_dict())
        assert "Project-Alpha" not in str(res_b.to_dict())
        assert "confidential_A.pdf" not in str(res_b.to_dict())
        assert "chk-A-001" not in str(res_b.to_dict())


# ============================================================================
# 4. CROSS-DOCUMENT PROVENANCE ISOLATION
# ============================================================================


class TestCrossDocumentProvenanceIsolation:
    """Verifies that evidence from multiple documents maintains strict provenance boundaries."""

    def test_multi_document_lineage_locking(self) -> None:
        """Verify citations from Doc 101 and Doc 202 cannot cross-attribute page or chunk identity."""
        results = [
            VectorSearchResult(
                chunk_id="c-doc101-p1",
                score=0.95,
                document_id="DOC-101",
                filename="financials_101.pdf",
                page_number=1,
                chunk_index=0,
                content_type="chart",
                content="DOC 101 Page 1 Chart",
                metadata={"client": "Client 101"},
            ),
            VectorSearchResult(
                chunk_id="c-doc202-p5",
                score=0.89,
                document_id="DOC-202",
                filename="engineering_202.pdf",
                page_number=5,
                chunk_index=3,
                content_type="diagram",
                content="DOC 202 Page 5 Diagram",
                metadata={"client": "Client 202"},
            ),
        ]

        citations = [AgentCitation.from_search_result(r) for r in results]
        search_pkg = SearchResult(
            query="Compare clients 101 and 202",
            status="RESULTS_FOUND",
            citations=citations,
            context="Combined context",
        )

        adapted = VisualEvidenceAdapter.adapt_search_package(search_pkg)
        assert len(adapted) == 2

        # Verify item 0 strictly DOC-101
        assert adapted[0].document_id == "DOC-101"
        assert adapted[0].filename == "financials_101.pdf"
        assert adapted[0].chunk_id == "c-doc101-p1"
        assert adapted[0].page_number == 1
        assert adapted[0].metadata["client"] == "Client 101"

        # Verify item 1 strictly DOC-202
        assert adapted[1].document_id == "DOC-202"
        assert adapted[1].filename == "engineering_202.pdf"
        assert adapted[1].chunk_id == "c-doc202-p5"
        assert adapted[1].page_number == 5
        assert adapted[1].metadata["client"] == "Client 202"


# ============================================================================
# 5. SERIALIZATION & CONTRACT INTEGRITY SAFETY
# ============================================================================


class TestSerializationAndContractIntegritySafety:
    """Verifies that serialization methods produce clean, typed dictionaries without private state."""

    def test_vision_result_serialization_roundtrip_safety(self) -> None:
        """Verify VisionResult.to_dict() and from_dict() roundtrip without data loss or extra fields."""
        evidence = [
            VisualEvidence(
                document_id="doc-sec-01",
                filename="sec_report.pdf",
                chunk_id="chk-sec-1",
                page_number=2,
                chunk_index=0,
                content_type="chart",
                image_path="B:/tmp/safe_image.png",
                description="Security plot",
                metadata={"safety_score": 100},
            )
        ]

        original = VisionResult(
            query="Analyze security plot",
            status="success",
            description="Security metrics are 100% compliant.",
            evidence=evidence,
            metadata={"audit_id": "AUD-999"},
        )

        d = original.to_dict()
        expected_keys = {
            "query",
            "status",
            "description",
            "evidence",
            "document_id",
            "filename",
            "page_number",
            "chunk_id",
            "content_type",
            "metadata",
            "error",
        }
        assert set(d.keys()) == expected_keys

        restored = VisionResult.from_dict(d)
        assert restored.to_dict() == d
        assert restored.document_id == "doc-sec-01"
        assert restored.evidence[0].chunk_id == "chk-sec-1"

    def test_agent_citation_serialization_safety(self) -> None:
        """Verify AgentCitation.to_dict() matches expected public schema."""
        cit = AgentCitation(
            document_id="doc-cit-01",
            filename="cit.pdf",
            chunk_id="chk-cit-01",
            page_number=3,
            content_type="diagram",
            score=0.94,
            metadata={"meta_key": "meta_val"},
        )

        d = cit.to_dict()
        expected_cit_keys = {
            "document_id",
            "filename",
            "chunk_id",
            "page_number",
            "content_type",
            "score",
            "metadata",
        }
        assert set(d.keys()) == expected_cit_keys
        assert d["document_id"] == "doc-cit-01"
        assert d["page_number"] == 3
        assert d["score"] == 0.94
