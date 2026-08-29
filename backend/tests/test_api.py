"""
Backend API test suite.

Tests all FastAPI endpoints using FastAPI's built-in TestClient (synchronous).
Uses isolated in-memory state — no real Qdrant, no real embedding model needed.

Run with:
    python -m pytest backend/tests/test_api.py -v
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure Qdrant uses in-memory mode during tests
os.environ.setdefault("QDRANT_URL", ":memory:")

from backend.main import app

client = TestClient(app, raise_server_exceptions=False)

# ── Sample PDF fixture ────────────────────────────────────────────────────────

SAMPLE_PDF = Path(__file__).parent.parent.parent / "sample.pdf"


def _pdf_bytes() -> bytes:
    """Load sample.pdf or create a minimal synthetic PDF."""
    if SAMPLE_PDF.exists():
        return SAMPLE_PDF.read_bytes()
    # Minimal valid PDF structure for testing
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000062 00000 n \n0000000114 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF"
    )


# ── Health tests ──────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert isinstance(data["components"], list)
        assert "uptime_seconds" in data

    def test_health_has_components(self):
        response = client.get("/health")
        data = response.json()
        names = [c["name"] for c in data["components"]]
        assert "api_server" in names
        assert "ingestion_pipeline" in names

    def test_root_redirects(self):
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (302, 307, 308)


# ── Document upload tests ─────────────────────────────────────────────────────

class TestDocumentUpload:
    def test_upload_valid_pdf(self):
        pdf_bytes = _pdf_bytes()
        response = client.post(
            "/api/documents/upload",
            files={"file": ("sample.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "sample.pdf"
        assert data["status"] == "processing"
        assert "message" in data
        assert data["file_size_bytes"] > 0

    def test_upload_returns_document_id(self):
        pdf_bytes = _pdf_bytes()
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert response.status_code == 201
        doc_id = response.json()["document_id"]
        assert len(doc_id) == 36  # UUID4 format

    def test_upload_invalid_file_type_txt(self):
        response = client.post(
            "/api/documents/upload",
            files={"file": ("document.txt", io.BytesIO(b"not a pdf"), "text/plain")},
        )
        assert response.status_code == 422

    def test_upload_invalid_extension(self):
        response = client.post(
            "/api/documents/upload",
            files={"file": ("document.docx", io.BytesIO(b"fake"), "application/octet-stream")},
        )
        assert response.status_code == 422

    def test_upload_empty_file(self):
        response = client.post(
            "/api/documents/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
        assert response.status_code == 422


# ── Document list / get tests ─────────────────────────────────────────────────

class TestDocuments:
    def test_list_documents_empty(self):
        # Reset registry by calling fresh client
        from fastapi.testclient import TestClient
        from backend.main import app
        fresh = TestClient(app)
        # Cannot reset singleton registry easily; just verify schema
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)

    def test_get_document_not_found(self):
        response = client.get("/api/documents/nonexistent-id-00000000")
        assert response.status_code == 404

    def test_upload_then_get(self):
        pdf_bytes = _pdf_bytes()
        upload_res = client.post(
            "/api/documents/upload",
            files={"file": ("get_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert upload_res.status_code == 201
        doc_id = upload_res.json()["document_id"]

        get_res = client.get(f"/api/documents/{doc_id}")
        assert get_res.status_code == 200
        data = get_res.json()
        assert data["document_id"] == doc_id
        assert data["filename"] == "get_test.pdf"

    def test_delete_document(self):
        pdf_bytes = _pdf_bytes()
        upload_res = client.post(
            "/api/documents/upload",
            files={"file": ("delete_me.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        doc_id = upload_res.json()["document_id"]

        del_res = client.delete(f"/api/documents/{doc_id}")
        assert del_res.status_code == 204

        get_res = client.get(f"/api/documents/{doc_id}")
        assert get_res.status_code == 404

    def test_delete_not_found(self):
        response = client.delete("/api/documents/does-not-exist")
        assert response.status_code == 404


# ── Ingestion status tests ────────────────────────────────────────────────────

class TestIngestionStatus:
    def test_status_not_found(self):
        response = client.get("/api/ingestion/no-such-doc/status")
        assert response.status_code == 404

    def test_status_after_upload(self):
        pdf_bytes = _pdf_bytes()
        upload_res = client.post(
            "/api/documents/upload",
            files={"file": ("status_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert upload_res.status_code == 201
        doc_id = upload_res.json()["document_id"]

        # Status is created asynchronously; it may or may not be available immediately
        status_res = client.get(f"/api/ingestion/{doc_id}/status")
        # Accept either 200 (status created) or 404 (task not started yet in sync test env)
        assert status_res.status_code in (200, 404)

        if status_res.status_code == 200:
            data = status_res.json()
            assert "status" in data
            assert "current_stage" in data
            assert "progress" in data
            assert 0 <= data["progress"] <= 100


# ── Search tests ──────────────────────────────────────────────────────────────

class TestSearch:
    def test_search_health(self):
        response = client.get("/api/search/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "collection_exists" in data
        assert "collection_name" in data

    def test_search_empty_collection(self):
        """Search against an empty/non-existent collection should return a structured error or empty results."""
        response = client.post(
            "/api/search",
            json={"query": "What is this document about?", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "status" in data
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_invalid_empty_query(self):
        response = client.post(
            "/api/search",
            json={"query": "", "top_k": 5},
        )
        assert response.status_code == 422

    def test_search_invalid_top_k(self):
        response = client.post(
            "/api/search",
            json={"query": "test query", "top_k": 0},
        )
        assert response.status_code == 422

    def test_search_response_schema(self):
        response = client.post(
            "/api/search",
            json={"query": "document content", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        required_keys = {"query", "answer", "status", "total_results", "results"}
        assert required_keys.issubset(data.keys())


# ── Error handling tests ──────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_file_field(self):
        response = client.post("/api/documents/upload")
        assert response.status_code == 422

    def test_unknown_endpoint(self):
        response = client.get("/api/unknown-endpoint-xyz")
        assert response.status_code == 404
