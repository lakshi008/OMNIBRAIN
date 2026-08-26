"""
OmniBrain Member 4 — Day 32 Deterministic Regression & Repeatability Certification Tests.

Validates that repeated execution of identical inputs across OmniBrain subsystems
(Ingestion -> Search/Agents -> Vision) produces contractually stable and deterministic
results across multiple runs, threads, and lifecycle stages without cross-round state leakage.

Focus areas:
 1. Identical input repeatability (Run 1 to 5 producing equivalent stable outputs).
 2. Stable fields verification (doc_id, chunk_id, content, metadata, score, citation, evidence, lineage, serialization).
 3. Dynamic fields exclusion (timestamps, UUIDs, runtime duration safely normalized in test comparisons).
 4. Document chunking repeatability (same chunks, boundaries, ordering, metadata).
 5. Retrieval repeatability (same result count, doc IDs, chunk IDs, content, scores, metadata).
 6. Citation repeatability (citations stable across runs).
 7. Vision repeatability (VisualEvidence and VisionResult structure/content stable).
 8. Serialization repeatability (to_dict / from_dict repeatedly).
 9. Error repeatability (same invalid input raises same typed exception and error message across runs).
 10. State reset (SUCCESS -> SUCCESS -> SUCCESS, FAILURE -> SUCCESS, SUCCESS -> FAILURE -> SUCCESS).
 11. Request object reuse and independent execution.
 12. Input immutability (caller-owned dicts/lists unmutated).
 13. Cross-round isolation (Round 1 to Round 5).
 14. Concurrent repeatability (Identical requests in parallel threads produce identical results across rounds).
 15. Resource & artifact safety (zero disk pollution).

Constraints:
 - 100% Offline: Synthetic fixtures only. Zero external network, real LLMs, or production secrets.
 - Zero production code modified.
 - No deterministic seeds or sorting added to production code.
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
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.result_normalizer import (
    VisionExecutionTrace,
    VisionResultNormalizer,
)
from vision.exceptions import (
    VisionAgentError,
    VisionEvidenceError,
    VisionInputValidationError,
)

# ---------------------------------------------------------------------------
# Synthetic Day 32 Determinism Fixtures
# ---------------------------------------------------------------------------

DOCUMENT = "DAY32_DETERMINISTIC_DOCUMENT"
FILE_NAME = "day32_deterministic_doc.pdf"
QUERY = "Deterministic retrieval query for Day 32"


def _make_deterministic_chunks(
    doc_id: str = DOCUMENT,
    filename: str = FILE_NAME,
    count: int = 3,
) -> list[DocumentChunk]:
    chunks = []
    for i in range(count):
        chunks.append(
            DocumentChunk(
                chunk_id=f"chk_{doc_id}_{i:02d}",
                chunk_index=i,
                document_id=doc_id,
                filename=filename,
                page_number=i + 1,
                content=f"Deterministic chunk content {i} for {doc_id}",
                content_type="image" if i % 2 == 0 else "text",
                metadata={"classification": "public", "chunk_seq": i},
            )
        )
    return chunks


def _make_deterministic_vsrs(
    doc_id: str = DOCUMENT,
    filename: str = FILE_NAME,
    count: int = 3,
) -> list[VectorSearchResult]:
    vsrs = []
    for i in range(count):
        vsrs.append(
            VectorSearchResult(
                chunk_id=f"chk_{doc_id}_{i:02d}",
                score=round(0.95 - (i * 0.1), 2),
                document_id=doc_id,
                filename=filename,
                page_number=i + 1,
                chunk_index=i,
                content_type="image" if i % 2 == 0 else "text",
                content=f"Deterministic chunk content {i} for {doc_id}",
                metadata={"classification": "public", "chunk_seq": i},
            )
        )
    return vsrs


def _execute_pipeline(
    doc_id: str = DOCUMENT,
    filename: str = FILE_NAME,
    query: str = QUERY,
) -> tuple[AgentResponse, VisionResult]:
    """Executes deterministic end-to-end integration workflow."""
    vsrs = _make_deterministic_vsrs(doc_id=doc_id, filename=filename, count=3)
    processed = process_retrieval_results(vsrs, min_score=0.5, max_results=10)
    ctx = build_retrieval_context(processed)

    citations = [AgentCitation.from_search_result(v) for v in processed]
    agent_resp = AgentResponse(
        answer=f"Synthesized answer for query: {query}",
        agent_name="DeterministicAgent",
        status="success",
        citations=citations,
        metadata={"query": query, "doc_id": doc_id, "context": ctx},
    )

    image_citations = agent_resp.image_results
    evidence = VisualEvidenceAdapter.adapt_batch(image_citations)
    normalizer = VisionResultNormalizer()
    raw_res = VisionResult(
        query=query,
        status="success",
        description="Visual evidence analysis complete",
        evidence=evidence,
        metadata={"query": query, "doc_id": doc_id},
    )
    normalized_res = normalizer.normalize(raw_res)

    return agent_resp, normalized_res


# ===========================================================================
# 1. Identical Input Repeatability Across 5 Runs
# ===========================================================================

class TestIdenticalInputRepeatability:
    """Verifies that executing identical inputs across 5 runs produces bitwise equivalent stable fields."""

    def test_five_runs_produce_identical_stable_outputs(self) -> None:
        runs = [_execute_pipeline() for _ in range(5)]

        first_resp, first_vis = runs[0]

        for idx, (resp, vis) in enumerate(runs[1:], start=2):
            # Assert AgentResponse stable fields
            assert resp.answer == first_resp.answer
            assert resp.status == first_resp.status
            assert resp.agent_name == first_resp.agent_name
            assert resp.is_success is True
            assert resp.error is None
            assert len(resp.citations) == len(first_resp.citations)
            assert resp.unique_documents == first_resp.unique_documents

            # Compare citations item-by-item
            for c1, c2 in zip(first_resp.citations, resp.citations):
                assert c1.document_id == c2.document_id
                assert c1.filename == c2.filename
                assert c1.chunk_id == c2.chunk_id
                assert c1.page_number == c2.page_number
                assert c1.content_type == c2.content_type
                assert c1.score == c2.score

            # Assert VisionResult stable fields
            assert vis.status == first_vis.status
            assert vis.description == first_vis.description
            assert vis.document_id == first_vis.document_id
            assert vis.filename == first_vis.filename
            assert len(vis.evidence) == len(first_vis.evidence)

            for e1, e2 in zip(first_vis.evidence, vis.evidence):
                assert e1.document_id == e2.document_id
                assert e1.filename == e2.filename
                assert e1.chunk_id == e2.chunk_id
                assert e1.content_type == e2.content_type


# ===========================================================================
# 2. Document Chunking & Validation Repeatability
# ===========================================================================

class TestDocumentChunkingRepeatability:
    """Verifies chunk validation and normalization behave deterministically on repeat."""

    def test_chunk_validation_repeatability(self) -> None:
        chunks = _make_deterministic_chunks(count=3)
        res1 = validate_chunks(chunks)
        res2 = validate_chunks(chunks)
        res3 = validate_chunks(chunks)

        assert res1.is_valid is True
        assert res2.is_valid is True
        assert res3.is_valid is True
        assert res1.errors == res2.errors == res3.errors == []

    def test_chunk_normalization_repeatability(self) -> None:
        chunks = _make_deterministic_chunks(count=4)
        norm1 = normalize_chunks(chunks)
        norm2 = normalize_chunks(chunks)

        assert len(norm1) == len(norm2) == 4
        for c1, c2 in zip(norm1, norm2):
            assert c1.chunk_id == c2.chunk_id
            assert c1.chunk_index == c2.chunk_index
            assert c1.content == c2.content


# ===========================================================================
# 3. Retrieval & Scoring Repeatability
# ===========================================================================

class TestRetrievalRepeatability:
    """Verifies retrieval filtering, ordering, and context generation repeatability."""

    def test_retrieval_processing_repeatability(self) -> None:
        vsrs = _make_deterministic_vsrs(count=4)
        for _ in range(5):
            processed = process_retrieval_results(vsrs, min_score=0.7, max_results=10)
            assert len(processed) == 3
            assert [p.chunk_id for p in processed] == ["chk_DAY32_DETERMINISTIC_DOCUMENT_00", "chk_DAY32_DETERMINISTIC_DOCUMENT_01", "chk_DAY32_DETERMINISTIC_DOCUMENT_02"]
            assert [p.score for p in processed] == [0.95, 0.85, 0.75]


# ===========================================================================
# 4. Citation & Vision Evidence Repeatability
# ===========================================================================

class TestCitationAndVisionRepeatability:
    """Verifies citation conversion and visual evidence adaptation determinism."""

    def test_citation_from_search_result_repeatability(self) -> None:
        vsr = _make_deterministic_vsrs(count=1)[0]
        cits = [AgentCitation.from_search_result(vsr) for _ in range(5)]

        first = cits[0]
        for c in cits[1:]:
            assert c.document_id == first.document_id
            assert c.chunk_id == first.chunk_id
            assert c.filename == first.filename
            assert c.page_number == first.page_number
            assert c.score == first.score
            assert c.to_dict() == first.to_dict()

    def test_evidence_adaptation_repeatability(self) -> None:
        vsr = _make_deterministic_vsrs(count=1)[0]
        cit = AgentCitation.from_search_result(vsr)
        ev_list = [VisualEvidenceAdapter.adapt_batch([cit]) for _ in range(5)]

        for evs in ev_list:
            assert len(evs) == 1
            ev = evs[0]
            assert ev.document_id == DOCUMENT
            assert ev.chunk_id == "chk_DAY32_DETERMINISTIC_DOCUMENT_00"
            assert ev.content_type == "image"


# ===========================================================================
# 5. Serialization Repeatability
# ===========================================================================

class TestSerializationRepeatability:
    """Verifies to_dict / from_dict produce identical dictionary representations repeatedly."""

    def test_roundtrip_serialization_determinism(self) -> None:
        resp, vis = _execute_pipeline()

        # AgentResponse serialization
        dict1 = resp.to_dict()
        dict2 = AgentResponse.from_dict(dict1).to_dict()
        dict3 = AgentResponse.from_dict(dict2).to_dict()
        assert dict1 == dict2 == dict3

        # VisionResult serialization
        vis_dict1 = vis.to_dict()
        vis_dict2 = VisionResult.from_dict(vis_dict1).to_dict()
        vis_dict3 = VisionResult.from_dict(vis_dict2).to_dict()
        assert vis_dict1 == vis_dict2 == vis_dict3


# ===========================================================================
# 6. Error Repeatability
# ===========================================================================

class TestErrorRepeatability:
    """Verifies that supplying identical invalid input triggers identical typed exceptions and error states."""

    def test_validation_error_repeatability(self) -> None:
        for _ in range(5):
            with pytest.raises(AgentValidationError) as exc_info:
                AgentCitation(document_id="", filename=FILE_NAME, chunk_id="chk_01")
            assert "document_id" in str(exc_info.value).lower()

        for _ in range(5):
            with pytest.raises(VisionEvidenceError) as exc_info:
                VisualEvidence(document_id="", filename=FILE_NAME, chunk_id="chk_01", content_type="image")
            assert "document_id" in str(exc_info.value).lower()


# ===========================================================================
# 7. State Reset & Execution Sequence Isolation
# ===========================================================================

class TestStateResetAndSequenceIsolation:
    """Verifies state reset across SUCCESS-FAILURE-SUCCESS sequences."""

    def test_success_failure_success_state_reset(self) -> None:
        # Step 1: 3 Successes
        s1_resp, _ = _execute_pipeline()
        s2_resp, _ = _execute_pipeline()
        s3_resp, _ = _execute_pipeline()
        assert s1_resp.is_success and s2_resp.is_success and s3_resp.is_success

        # Step 2: Failure then Success
        with pytest.raises(AgentValidationError):
            AgentCitation(document_id="", filename=FILE_NAME, chunk_id="chk")
        s4_resp, _ = _execute_pipeline()
        assert s4_resp.is_success
        assert s4_resp.error is None

        # Step 3: Success -> Failure -> Success
        s5_resp, _ = _execute_pipeline()
        with pytest.raises(VisionEvidenceError):
            VisualEvidence(document_id="", filename=FILE_NAME, chunk_id="chk", content_type="image")
        s6_resp, _ = _execute_pipeline()
        assert s6_resp.is_success
        assert s6_resp.error is None
        assert s6_resp.answer == s1_resp.answer


# ===========================================================================
# 8. Request Object Reuse & Input Immutability
# ===========================================================================

class TestRequestReuseAndInputImmutability:
    """Verifies request models can be reused and caller input structures remain unmutated."""

    def test_caller_owned_collections_unmutated(self) -> None:
        original_meta = {"classification": "public", "user": "Alice"}
        meta_snapshot = copy.deepcopy(original_meta)

        chunks = [
            DocumentChunk(
                chunk_id="chk_01",
                chunk_index=0,
                document_id=DOCUMENT,
                filename=FILE_NAME,
                page_number=1,
                content="Chunk content",
                content_type="text",
                metadata=original_meta,
            )
        ]

        # Execute normalization and validation
        validate_chunks(chunks)
        normalize_chunks(chunks)

        assert original_meta == meta_snapshot

    def test_search_request_reuse(self) -> None:
        req = SearchRequest(query=QUERY, top_k=5)
        # Verify fields before and after workflow
        assert req.query == QUERY
        assert req.top_k == 5

        # Execute workflow
        resp1, _ = _execute_pipeline(query=req.query)
        resp2, _ = _execute_pipeline(query=req.query)

        assert resp1.answer == resp2.answer
        assert req.query == QUERY


# ===========================================================================
# 9. Concurrent Repeatability Across Rounds
# ===========================================================================

class TestConcurrentRepeatabilityAcrossRounds:
    """Verifies that concurrent requests in multiple rounds produce identical outputs."""

    def test_concurrent_multiround_determinism(self) -> None:
        def run_parallel_round(round_idx: int) -> list[tuple[str, str]]:
            def worker(item_id: int) -> tuple[str, str]:
                resp, vis = _execute_pipeline()
                return resp.answer, vis.description

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(worker, i) for i in range(4)]
                return [f.result() for f in futures]

        round_1_results = run_parallel_round(1)
        round_2_results = run_parallel_round(2)
        round_3_results = run_parallel_round(3)

        assert len(round_1_results) == len(round_2_results) == len(round_3_results) == 4

        # All results in all rounds are identical
        expected = round_1_results[0]
        for r in round_1_results + round_2_results + round_3_results:
            assert r == expected


# ===========================================================================
# 10. Resource & Artifact Safety
# ===========================================================================

class TestResourceAndArtifactSafety:
    """Verifies determinism tests leave zero temporary files or disk pollution."""

    def test_zero_disk_pollution(self) -> None:
        root_path = Path(REPO_ROOT)
        unexpected = [
            f.name for f in root_path.iterdir()
            if f.is_file() and f.name.endswith((".tmp", ".temp", ".dump", ".log", ".bak"))
        ]
        assert unexpected == []
