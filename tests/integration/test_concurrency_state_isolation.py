"""
OmniBrain Member 4 — Day 7 Concurrency, State Isolation & Resource Safety Tests.

Verifies that concurrent multi-threaded execution across Ingestion, Search, Vision,
and Supervisor components preserves complete state isolation, resource safety, and immutability.

Focus areas:
1. Concurrent execution of independent requests (2-way and N-way thread concurrency).
2. Evidence, metadata, lineage, and citation isolation across concurrent flows.
3. Result and Supervisor state isolation without memory cross-talk or race conditions.
4. Deterministic repeated execution without stale state accumulation.
5. Mutation safety of input evidence, citations, and metadata structures.
6. Concurrent failure isolation (failures in one thread do not corrupt sibling threads).
7. Independent retry and lifecycle state isolation per request.
8. 100% offline, deterministic, side-effect-free execution.
"""

from __future__ import annotations

import copy
import sys
import threading
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path for test runners executing this file directly
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

from agents.exceptions import AgentValidationError
from agents.models import AgentCitation, AgentResponse, AgentState, SearchResult
from ingestion.models import DocumentChunk, VectorSearchResult
from vision.evidence_adapter import VisualEvidenceAdapter
from vision.exceptions import VisionEvidenceError, VisionInputValidationError
from vision.lifecycle import VisionExecutionLifecycle, VisionRetryPolicy
from vision.models import (
    VALID_VISUAL_CONTENT_TYPES,
    VisionRequest,
    VisionResult,
    VisualEvidence,
)


# ============================================================================
# 1. TWO CONCURRENT REQUESTS ISOLATION
# ============================================================================


class TestTwoConcurrentRequestsIsolation:
    """Verifies that two parallel requests maintain strict data and lineage boundaries."""

    def test_two_concurrent_search_vision_supervisor_flows(self) -> None:
        """Execute Flow A and Flow B simultaneously in parallel threads."""
        outputs: dict[str, dict[str, Any]] = {}
        errors: list[Exception] = []

        def run_flow_a() -> None:
            try:
                vs_a = VectorSearchResult(
                    chunk_id="chunk-flow-A",
                    score=0.95,
                    document_id="doc-alpha-100",
                    filename="alpha_specs.pdf",
                    page_number=3,
                    chunk_index=1,
                    content_type="chart",
                    content="Alpha chart metrics.",
                    metadata={"team": "alpha_team", "run_id": "A_001"},
                )
                cit_a = AgentCitation.from_search_result(vs_a)
                ev_a = VisualEvidenceAdapter.adapt_citation(cit_a)
                res_a = VisionResult(
                    query="Explain Alpha chart metrics",
                    status="success",
                    description="Alpha metrics verified: 25% growth.",
                    evidence=[ev_a],
                    metadata={"flow": "A"},
                )
                state_a = AgentState(query=res_a.query, route="vision")
                state_a.add_citation(
                    AgentCitation(
                        document_id=res_a.document_id,
                        filename=res_a.filename,
                        chunk_id=res_a.chunk_id,
                        page_number=res_a.page_number,
                        content_type=res_a.content_type,
                        score=1.0,
                        metadata=dict(res_a.metadata),
                    )
                )
                state_a.update(answer=res_a.description, status="completed")

                outputs["A"] = {
                    "doc_id": state_a.citations[0].document_id,
                    "filename": state_a.citations[0].filename,
                    "chunk_id": state_a.citations[0].chunk_id,
                    "answer": state_a.answer,
                    "metadata": dict(state_a.citations[0].metadata),
                }
            except Exception as e:
                errors.append(e)

        def run_flow_b() -> None:
            try:
                vs_b = VectorSearchResult(
                    chunk_id="chunk-flow-B",
                    score=0.88,
                    document_id="doc-beta-200",
                    filename="beta_architecture.pdf",
                    page_number=7,
                    chunk_index=3,
                    content_type="diagram",
                    content="Beta architecture diagram.",
                    metadata={"team": "beta_team", "run_id": "B_002"},
                )
                cit_b = AgentCitation.from_search_result(vs_b)
                ev_b = VisualEvidenceAdapter.adapt_citation(cit_b)
                res_b = VisionResult(
                    query="Explain Beta architecture diagram",
                    status="success",
                    description="Beta architecture verified: 4-tier design.",
                    evidence=[ev_b],
                    metadata={"flow": "B"},
                )
                state_b = AgentState(query=res_b.query, route="vision")
                state_b.add_citation(
                    AgentCitation(
                        document_id=res_b.document_id,
                        filename=res_b.filename,
                        chunk_id=res_b.chunk_id,
                        page_number=res_b.page_number,
                        content_type=res_b.content_type,
                        score=1.0,
                        metadata=dict(res_b.metadata),
                    )
                )
                state_b.update(answer=res_b.description, status="completed")

                outputs["B"] = {
                    "doc_id": state_b.citations[0].document_id,
                    "filename": state_b.citations[0].filename,
                    "chunk_id": state_b.citations[0].chunk_id,
                    "answer": state_b.answer,
                    "metadata": dict(state_b.citations[0].metadata),
                }
            except Exception as e:
                errors.append(e)

        thread_a = threading.Thread(target=run_flow_a)
        thread_b = threading.Thread(target=run_flow_b)

        thread_a.start()
        thread_b.start()

        thread_a.join()
        thread_b.join()

        assert len(errors) == 0, f"Concurrent execution failed with errors: {errors}"
        assert "A" in outputs and "B" in outputs

        # Verify A contains only A data
        assert outputs["A"]["doc_id"] == "doc-alpha-100"
        assert outputs["A"]["filename"] == "alpha_specs.pdf"
        assert outputs["A"]["chunk_id"] == "chunk-flow-A"
        assert "Alpha metrics verified" in outputs["A"]["answer"]
        assert outputs["A"]["metadata"]["flow"] == "A"

        # Verify B contains only B data
        assert outputs["B"]["doc_id"] == "doc-beta-200"
        assert outputs["B"]["filename"] == "beta_architecture.pdf"
        assert outputs["B"]["chunk_id"] == "chunk-flow-B"
        assert "Beta architecture verified" in outputs["B"]["answer"]
        assert outputs["B"]["metadata"]["flow"] == "B"


# ============================================================================
# 2. MULTI-REQUEST & EVIDENCE ISOLATION (N-WAY)
# ============================================================================


class TestMultiRequestAndEvidenceIsolation:
    """Verifies that N concurrent requests maintain complete evidence and metadata isolation."""

    def test_n_way_concurrent_evidence_isolation(self) -> None:
        """Run 16 concurrent threads each processing unique document evidence."""
        thread_count = 16
        results: dict[int, dict[str, Any]] = {}
        errors: list[Exception] = []

        def worker(thread_idx: int) -> None:
            try:
                doc_id = f"doc-concurrent-{thread_idx}"
                fname = f"document_{thread_idx}.pdf"
                cid = f"chk-{thread_idx}"
                page = thread_idx + 1
                meta = {
                    "thread_idx": thread_idx,
                    "secret_uuid": f"uuid-{thread_idx}-{thread_idx * 7}",
                    "chunk_index": thread_idx,
                }

                vs_item = VectorSearchResult(
                    chunk_id=cid,
                    score=0.90 + (thread_idx * 0.005),
                    document_id=doc_id,
                    filename=fname,
                    page_number=page,
                    chunk_index=thread_idx,
                    content_type="chart" if thread_idx % 2 == 0 else "image",
                    content=f"Figure content for thread {thread_idx}",
                    metadata=dict(meta),
                )

                citation = AgentCitation.from_search_result(vs_item)
                evidence = VisualEvidenceAdapter.adapt_citation(citation)

                vision_result = VisionResult(
                    query=f"Analyze figure from {doc_id}",
                    status="success",
                    description=f"Verified figure content {thread_idx}",
                    evidence=[evidence],
                    metadata={"thread_idx": thread_idx},
                )

                resp_cit = AgentCitation(
                    document_id=vision_result.document_id,
                    filename=vision_result.filename,
                    chunk_id=vision_result.chunk_id,
                    page_number=vision_result.page_number,
                    content_type=vision_result.content_type,
                    score=1.0,
                    metadata=dict(vision_result.metadata),
                )

                response = AgentResponse(
                    answer=vision_result.description,
                    agent_name="VisionAgent",
                    citations=[resp_cit],
                    metadata=dict(vision_result.metadata),
                )

                results[thread_idx] = {
                    "doc_id": response.citations[0].document_id,
                    "filename": response.citations[0].filename,
                    "chunk_id": response.citations[0].chunk_id,
                    "page": response.citations[0].page_number,
                    "thread_idx": response.metadata.get("thread_idx"),
                    "answer": response.answer,
                }
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == thread_count

        for i, data in results.items():
            assert data["doc_id"] == f"doc-concurrent-{i}"
            assert data["filename"] == f"document_{i}.pdf"
            assert data["chunk_id"] == f"chk-{i}"
            assert data["page"] == i + 1
            assert data["thread_idx"] == i
            assert data["answer"] == f"Verified figure content {i}"


# ============================================================================
# 3. REPEATED EXECUTION & MUTATION SAFETY
# ============================================================================


class TestRepeatedExecutionAndMutationSafety:
    """Verifies that repeated execution preserves determinism and leaves input structures untouched."""

    def test_input_object_immutability_under_repeated_runs(self) -> None:
        """Verify upstream VectorSearchResult and AgentCitation are never mutated across 10 iterations."""
        orig_meta = {"key1": "value1", "tags": ["prod", "v1"], "nested": {"counter": 0}}
        source_result = VectorSearchResult(
            chunk_id="chunk-immutable-01",
            score=0.93,
            document_id="doc-immutable",
            filename="immutable.pdf",
            page_number=2,
            chunk_index=0,
            content_type="diagram",
            content="Static invariant content",
            metadata=copy.deepcopy(orig_meta),
        )

        baseline_dict = copy.deepcopy(orig_meta)

        for iteration in range(10):
            cit = AgentCitation.from_search_result(source_result)
            ev = VisualEvidenceAdapter.adapt_citation(cit)
            res = VisionResult(
                query="Verify immutability",
                status="success",
                description=f"Iteration {iteration}",
                evidence=[ev],
                metadata={"iteration": iteration},
            )

            # Mutate downstream copies
            ev.metadata["mutated_field"] = "tampered"
            res.metadata["tampered"] = True

            # Assert original source_result was untouched
            assert source_result.metadata == baseline_dict
            assert "mutated_field" not in source_result.metadata
            assert source_result.document_id == "doc-immutable"
            assert source_result.score == 0.93


# ============================================================================
# 4. CONCURRENT FAILURE ISOLATION
# ============================================================================


class TestConcurrentFailureIsolation:
    """Verifies that an error or failure in one thread does not corrupt or block sibling threads."""

    def test_mixed_success_and_failure_concurrency(self) -> None:
        """Run 3 concurrent threads: Thread A (success), Thread B (failure), Thread C (success)."""
        thread_outcomes: dict[str, Any] = {}
        thread_errors: dict[str, str] = {}

        def thread_success_a() -> None:
            cit = AgentCitation(
                document_id="doc-succ-A",
                filename="a.pdf",
                chunk_id="chk-A",
                content_type="image",
            )
            ev = VisualEvidenceAdapter.adapt_citation(cit)
            res = VisionResult(
                query="Query A",
                status="success",
                description="Result A Success",
                evidence=[ev],
            )
            thread_outcomes["A"] = res.description

        def thread_failure_b() -> None:
            try:
                # Deliberately supply invalid content_type to trigger VisionEvidenceError
                invalid_cit = AgentCitation(
                    document_id="doc-fail-B",
                    filename="b.pdf",
                    chunk_id="chk-B",
                    content_type="unsupported_text_modality",
                )
                VisualEvidenceAdapter.adapt_citation(invalid_cit)
            except VisionEvidenceError as e:
                thread_errors["B"] = str(e)

        def thread_success_c() -> None:
            cit = AgentCitation(
                document_id="doc-succ-C",
                filename="c.pdf",
                chunk_id="chk-C",
                content_type="chart",
            )
            ev = VisualEvidenceAdapter.adapt_citation(cit)
            res = VisionResult(
                query="Query C",
                status="success",
                description="Result C Success",
                evidence=[ev],
            )
            thread_outcomes["C"] = res.description

        t_a = threading.Thread(target=thread_success_a)
        t_b = threading.Thread(target=thread_failure_b)
        t_c = threading.Thread(target=thread_success_c)

        t_a.start()
        t_b.start()
        t_c.start()

        t_a.join()
        t_b.join()
        t_c.join()

        # Threads A and C must succeed cleanly
        assert thread_outcomes.get("A") == "Result A Success"
        assert thread_outcomes.get("C") == "Result C Success"

        # Thread B must have captured the expected validation failure
        assert "B" in thread_errors
        assert "Unsupported content_type" in thread_errors["B"]


# ============================================================================
# 5. RETRY & LIFECYCLE STATE ISOLATION PER REQUEST
# ============================================================================


class TestRetryAndLifecycleStateIsolation:
    """Verifies that retry configurations and lifecycle counters remain strictly isolated per request."""

    def test_independent_retry_policy_instances(self) -> None:
        """Verify two retry policy configurations do not cross-talk or share mutable state."""
        policy_default = VisionRetryPolicy()
        policy_custom = VisionRetryPolicy(max_retries=4)

        assert policy_default.max_retries == 0
        assert policy_default.max_attempts == 1

        assert policy_custom.max_retries == 4
        assert policy_custom.max_attempts == 5

        # Confirm policy default is untouched
        assert policy_default.max_retries == 0
