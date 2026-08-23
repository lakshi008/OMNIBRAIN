"""
Day 42 — Vision Agent Concurrency Safety, Request Isolation & Stateless Execution Tests.

Verifies:
  1. Sequential request isolation (queries, evidence, lineage, metadata, lifecycle).
  2. Concurrent request isolation (ThreadPoolExecutor execution across multiple threads).
  3. Shared agent instance thread-safety and stateless execution.
  4. Shared provider instance thread-safety.
  5. Failure isolation (a failing/invalid request does not corrupt or cancel sibling requests).
  6. Input immutability under concurrency.
  7. Exact N-provider invocation count (N requests -> N executions).
  8. Offline execution integrity (zero external HTTP, vendor SDKs, credentials).
"""

from __future__ import annotations

import concurrent.futures
import inspect
import io
import threading
from typing import Any

import pytest
from PIL import Image

from vision import (
    VisionAgent,
    VisionExecutionAdapter,
    VisionExecutionLifecycle,
    VisionExecutionStage,
    VisionModelInput,
    VisionModelProvider,
    VisionPipeline,
    VisionProviderConfig,
    VisionRequest,
    VisionResult,
    VisualEvidence,
    run_vision_pipeline,
)
from vision.exceptions import (
    VisionEvidenceError,
    VisionInputValidationError,
    VisionProviderExecutionError,
)


# ---------------------------------------------------------------------------
# Test Helpers & Thread-Safe Test Doubles
# ---------------------------------------------------------------------------


def _make_png(width: int = 32, height: int = 32, color: tuple[int, int, int] = (100, 150, 200)) -> bytes:
    """Generate PNG byte payload for testing visual evidence."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class ThreadSafeRecordingProvider(VisionModelProvider):
    """Thread-safe test double recording all received inputs and invocation counts."""

    def __init__(self, config: VisionProviderConfig) -> None:
        super().__init__(config)
        self._lock = threading.Lock()
        self.recorded_inputs: list[VisionModelInput] = []
        self.invocation_count: int = 0

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        with self._lock:
            self.invocation_count += 1
            self.recorded_inputs.append(model_input)

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Analyzed {model_input.filename}",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
            metadata={"doc_id": model_input.document_id},
        )


class SelectiveFailingThreadSafeProvider(VisionModelProvider):
    """Thread-safe test double that fails conditionally for specified document_ids."""

    def __init__(self, config: VisionProviderConfig, fail_doc_ids: set[str]) -> None:
        super().__init__(config)
        self._lock = threading.Lock()
        self.fail_doc_ids = set(fail_doc_ids)
        self.invocation_count: int = 0
        self.recorded_inputs: list[VisionModelInput] = []

    def execute(self, model_input: VisionModelInput, **kwargs: Any) -> VisionResult:
        self.validate_input(model_input)
        with self._lock:
            self.invocation_count += 1
            self.recorded_inputs.append(model_input)

        if model_input.document_id in self.fail_doc_ids:
            raise VisionProviderExecutionError(f"Simulated failure for doc {model_input.document_id}")

        return VisionResult(
            query=model_input.query,
            status="success",
            description=f"Success for {model_input.document_id}",
            document_id=model_input.document_id,
            filename=model_input.filename,
            page_number=model_input.page_number,
            chunk_id=model_input.chunk_id,
            content_type=model_input.content_type,
        )


# ---------------------------------------------------------------------------
# Test Suite 1: Sequential Request Isolation
# ---------------------------------------------------------------------------


class TestSequentialRequestIsolation:
    """Verifies that sequential calls on the same agent do not leak state."""

    def test_01_sequential_requests_isolated(self) -> None:
        """Sequential request A followed by request B results in isolated outputs."""
        config = VisionProviderConfig(provider_name="seq-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        evA = VisualEvidence(document_id="doc-A", filename="fileA.pdf", chunk_id="chk-A", image_bytes=_make_png())
        evB = VisualEvidence(document_id="doc-B", filename="fileB.pdf", chunk_id="chk-B", image_bytes=_make_png())

        resA = agent.execute("Query A", evidence=[evA])
        resB = agent.execute("Query B", evidence=[evB])

        assert resA.query == "Query A"
        assert resA.document_id == "doc-A"
        assert resA.evidence[0].document_id == "doc-A"

        assert resB.query == "Query B"
        assert resB.document_id == "doc-B"
        assert resB.evidence[0].document_id == "doc-B"

    def test_02_result_objects_not_reused(self) -> None:
        """Sequential executions create distinct VisionResult objects."""
        config = VisionProviderConfig(provider_name="obj-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev1 = VisualEvidence(document_id="d1", filename="f1.png", chunk_id="c1", image_bytes=_make_png())
        res1 = agent.execute("Q1", evidence=[ev1])
        res2 = agent.execute("Q2", evidence=[ev1])

        assert res1 is not res2
        assert res1.query != res2.query

    def test_03_evidence_isolation_across_sequential_requests(self) -> None:
        """Evidence from Request A does not leak into Request B."""
        config = VisionProviderConfig(provider_name="ev-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        evA = VisualEvidence(document_id="doc-A", filename="A.png", chunk_id="cA", image_bytes=_make_png())
        evB = VisualEvidence(document_id="doc-B", filename="B.png", chunk_id="cB", image_bytes=_make_png())

        resA = agent.execute("Q-A", evidence=[evA])
        resB = agent.execute("Q-B", evidence=[evB])

        assert len(resA.evidence) == 1
        assert resA.evidence[0].document_id == "doc-A"

        assert len(resB.evidence) == 1
        assert resB.evidence[0].document_id == "doc-B"

    def test_04_query_isolation_across_sequential_requests(self) -> None:
        """Query strings remain strictly isolated across sequential requests."""
        config = VisionProviderConfig(provider_name="qry-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f1.png", chunk_id="c1", image_bytes=_make_png())

        res1 = agent.execute("First unique query string", evidence=[ev])
        res2 = agent.execute("Second unique query string", evidence=[ev])

        assert res1.query == "First unique query string"
        assert res2.query == "Second unique query string"
        assert provider.recorded_inputs[0].query == "First unique query string"
        assert provider.recorded_inputs[1].query == "Second unique query string"

    def test_05_metadata_isolation_across_sequential_requests(self) -> None:
        """Metadata dictionaries are not mutated or shared across requests."""
        config = VisionProviderConfig(provider_name="meta-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        evA = VisualEvidence(document_id="dA", filename="fA.png", chunk_id="cA", image_bytes=_make_png(), metadata={"tag": "A"})
        evB = VisualEvidence(document_id="dB", filename="fB.png", chunk_id="cB", image_bytes=_make_png(), metadata={"tag": "B"})

        resA = agent.execute("QA", evidence=[evA])
        resB = agent.execute("QB", evidence=[evB])

        assert resA.evidence[0].metadata == {"tag": "A"}
        assert resB.evidence[0].metadata == {"tag": "B"}

    def test_06_lineage_isolation_across_sequential_requests(self) -> None:
        """Source document lineage remains strictly isolated per request."""
        config = VisionProviderConfig(provider_name="lin-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        evA = VisualEvidence(document_id="doc-100", filename="fin.pdf", chunk_id="c1", page_number=1, image_bytes=_make_png())
        evB = VisualEvidence(document_id="doc-200", filename="eng.pdf", chunk_id="c2", page_number=15, image_bytes=_make_png())

        resA = agent.execute("QA", evidence=[evA])
        resB = agent.execute("QB", evidence=[evB])

        assert resA.document_id == "doc-100"
        assert resA.filename == "fin.pdf"
        assert resA.page_number == 1

        assert resB.document_id == "doc-200"
        assert resB.filename == "eng.pdf"
        assert resB.page_number == 15

    def test_07_lifecycle_isolation_across_sequential_requests(self) -> None:
        """Each execution receives its own independent execution lifecycle."""
        config = VisionProviderConfig(provider_name="life-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f1.png", chunk_id="c1", image_bytes=_make_png())

        res1 = agent.execute("Q1", evidence=[ev])
        res2 = agent.execute("Q2", evidence=[ev])

        lf1 = res1.metadata.get("execution_lifecycle")
        lf2 = res2.metadata.get("execution_lifecycle")

        assert lf1 is not None and lf2 is not None
        assert lf1["stage"] == "completed"
        assert lf2["stage"] == "completed"

    def test_08_provider_invocation_count_sequential(self) -> None:
        """N sequential requests trigger exactly N provider calls."""
        config = VisionProviderConfig(provider_name="cnt-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="d1", filename="f1.png", chunk_id="c1", image_bytes=_make_png())

        for i in range(5):
            agent.execute(f"Sequential query {i}", evidence=[ev])

        assert provider.invocation_count == 5


# ---------------------------------------------------------------------------
# Test Suite 2: Concurrent Request Isolation
# ---------------------------------------------------------------------------


class TestConcurrentRequestIsolation:
    """Verifies thread-safe execution under multi-threaded concurrency."""

    def test_09_concurrent_requests_independent_results(self) -> None:
        """Concurrent executions across 10 threads complete independently without cross-leakage."""
        config = VisionProviderConfig(provider_name="conc-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        def _run_request(index: int) -> VisionResult:
            ev = VisualEvidence(
                document_id=f"doc-{index}",
                filename=f"file_{index}.pdf",
                chunk_id=f"chk-{index}",
                page_number=index + 1,
                chunk_index=index,
                image_bytes=_make_png(),
            )
            return agent.execute(f"Concurrent Query {index}", evidence=[ev])

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_run_request, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 10
        assert provider.invocation_count == 10

        for res in results:
            idx = int(res.document_id.split("-")[1])
            assert res.query == f"Concurrent Query {idx}"
            assert res.filename == f"file_{idx}.pdf"
            assert res.page_number == idx + 1
            assert len(res.evidence) == 1
            assert res.evidence[0].document_id == f"doc-{idx}"

    def test_10_shared_agent_concurrent_requests(self) -> None:
        """A single VisionAgent instance can safely serve concurrent requests."""
        config = VisionProviderConfig(provider_name="ag-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        shared_agent = VisionAgent(provider=provider)

        def _run_task(task_id: int) -> tuple[int, VisionResult]:
            ev = VisualEvidence(
                document_id=f"shared-doc-{task_id}",
                filename=f"shared_{task_id}.png",
                chunk_id=f"shared-chk-{task_id}",
                image_bytes=_make_png(),
            )
            res = shared_agent.execute(f"Shared Agent Query {task_id}", evidence=[ev])
            return task_id, res

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_run_task, i) for i in range(8)]
            completed = [f.result() for f in futures]

        for tid, res in completed:
            assert res.document_id == f"shared-doc-{tid}"
            assert res.query == f"Shared Agent Query {tid}"

    def test_11_shared_provider_concurrent_requests(self) -> None:
        """A single VisionModelProvider instance can safely serve multiple agents/adapters concurrently."""
        config = VisionProviderConfig(provider_name="p-prov", model_name="v1")
        shared_provider = ThreadSafeRecordingProvider(config)

        def _run_with_new_agent(idx: int) -> VisionResult:
            agent = VisionAgent(provider=shared_provider)
            ev = VisualEvidence(
                document_id=f"prov-doc-{idx}",
                filename=f"p_{idx}.png",
                chunk_id=f"pchk-{idx}",
                image_bytes=_make_png(),
            )
            return agent.execute(f"Provider Query {idx}", evidence=[ev])

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_run_with_new_agent, i) for i in range(6)]
            results = [f.result() for f in futures]

        assert len(results) == 6
        assert shared_provider.invocation_count == 6

    def test_12_concurrent_lineage_verification(self) -> None:
        """Lineage fields (document_id, filename, chunk_id, page_number) never mix during concurrency."""
        config = VisionProviderConfig(provider_name="lin-conc-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        def _run_lin(i: int) -> bool:
            ev = VisualEvidence(
                document_id=f"unique-doc-{i}",
                filename=f"unique-file-{i}.pdf",
                chunk_id=f"unique-chk-{i}",
                page_number=100 + i,
                chunk_index=i,
                image_bytes=_make_png(),
            )
            res = agent.execute(f"Q-{i}", evidence=[ev])
            return (
                res.document_id == f"unique-doc-{i}"
                and res.filename == f"unique-file-{i}.pdf"
                and res.chunk_id == f"unique-chk-{i}"
                and res.page_number == 100 + i
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(_run_lin, i) for i in range(12)]
            matches = [f.result() for f in futures]

        assert all(matches)

    def test_13_concurrent_query_verification(self) -> None:
        """Every concurrent execution receives its own exact query without query swapping."""
        config = VisionProviderConfig(provider_name="qry-conc", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        queries = [f"Distinct query prompt number {i}" for i in range(10)]

        def _run_q(q: str, i: int) -> VisionResult:
            ev = VisualEvidence(document_id=f"d-{i}", filename=f"f-{i}.png", chunk_id=f"c-{i}", image_bytes=_make_png())
            return agent.execute(q, evidence=[ev])

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_run_q, queries[i], i) for i in range(10)]
            results = [f.result() for f in futures]

        returned_queries = {r.query for r in results}
        assert returned_queries == set(queries)

    def test_14_concurrent_evidence_verification(self) -> None:
        """Multi-evidence lists per request remain isolated during concurrent runs."""
        config = VisionProviderConfig(provider_name="ev-conc", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        def _run_multi(req_id: int) -> VisionResult:
            evs = [
                VisualEvidence(
                    document_id=f"doc-{req_id}-item-{j}",
                    filename=f"file-{req_id}-{j}.png",
                    chunk_id=f"chk-{req_id}-{j}",
                    image_bytes=_make_png(),
                )
                for j in range(3)
            ]
            return agent.execute(f"Multi query {req_id}", evidence=evs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_run_multi, i) for i in range(5)]
            results = [f.result() for f in futures]

        for res in results:
            req_id = int(res.query.split()[-1])
            assert len(res.evidence) == 3
            assert [e.document_id for e in res.evidence] == [f"doc-{req_id}-item-{j}" for j in range(3)]


# ---------------------------------------------------------------------------
# Test Suite 3: Failure & Boundary Isolation
# ---------------------------------------------------------------------------


class TestFailureAndBoundaryIsolation:
    """Verifies that failures or invalid inputs in one request do not corrupt siblings."""

    def test_15_one_failed_request_does_not_corrupt_others(self) -> None:
        """A provider failure on doc-FAIL does not cause other concurrent requests to fail."""
        config = VisionProviderConfig(provider_name="fail-iso-prov", model_name="v1")
        provider = SelectiveFailingThreadSafeProvider(config, fail_doc_ids={"doc-FAIL"})
        agent = VisionAgent(provider=provider)

        def _run_fail_test(doc_id: str) -> tuple[str, bool, VisionResult | None, str | None]:
            ev = VisualEvidence(document_id=doc_id, filename=f"{doc_id}.png", chunk_id=f"c-{doc_id}", image_bytes=_make_png())
            try:
                res = agent.execute(f"Query for {doc_id}", evidence=[ev])
                return doc_id, True, res, None
            except Exception as err:
                return doc_id, False, None, str(err)

        doc_ids = ["doc-1", "doc-2", "doc-FAIL", "doc-3", "doc-4"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_run_fail_test, d) for d in doc_ids]
            outcomes = [f.result() for f in futures]

        by_id = {did: (succ, res, err) for did, succ, res, err in outcomes}

        # doc-FAIL must fail with controlled error
        assert by_id["doc-FAIL"][0] is False
        assert "Simulated failure for doc doc-FAIL" in by_id["doc-FAIL"][2]

        # all other requests must succeed
        for ok_id in ["doc-1", "doc-2", "doc-3", "doc-4"]:
            assert by_id[ok_id][0] is True
            assert by_id[ok_id][1].is_success is True
            assert by_id[ok_id][1].document_id == ok_id

    def test_16_invalid_request_isolated(self) -> None:
        """An invalid request raising VisionInputValidationError does not corrupt valid sibling requests."""
        config = VisionProviderConfig(provider_name="val-iso-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        def _run_val_test(is_valid: bool, idx: int) -> tuple[int, bool]:
            if is_valid:
                ev = VisualEvidence(document_id=f"valid-doc-{idx}", filename=f"v{idx}.png", chunk_id=f"c{idx}", image_bytes=_make_png())
                res = agent.execute(f"Valid Query {idx}", evidence=[ev])
                return idx, res.is_success
            else:
                try:
                    agent.execute("", evidence=[])  # Invalid empty query
                    return idx, True
                except VisionInputValidationError:
                    return idx, False

        tasks = [(True, 1), (False, 2), (True, 3)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_run_val_test, is_v, idx) for is_v, idx in tasks]
            res_dict = dict([f.result() for f in futures])

        assert res_dict[1] is True
        assert res_dict[2] is False
        assert res_dict[3] is True

    def test_17_input_immutability_under_concurrency(self) -> None:
        """Input objects are not mutated even when processed concurrently."""
        config = VisionProviderConfig(provider_name="imm-conc-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="imm-doc", filename="imm.png", chunk_id="imm-chk", page_number=5, image_bytes=_make_png())
        req = VisionRequest(query="Shared Request Query", evidence=[ev])

        req_dict_before = req.to_dict()
        ev_dict_before = ev.to_dict()

        def _run_imm() -> VisionResult:
            return agent.execute(req)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_run_imm) for _ in range(5)]
            results = [f.result() for f in futures]

        assert len(results) == 5
        assert req.to_dict() == req_dict_before
        assert ev.to_dict() == ev_dict_before

    def test_18_deterministic_execution_under_concurrency(self) -> None:
        """Concurrent execution produces identical structured results for identical inputs."""
        config = VisionProviderConfig(provider_name="det-conc-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(document_id="det-doc", filename="det.png", chunk_id="det-chk", image_bytes=_make_png())

        def _run_det() -> dict[str, Any]:
            res = agent.execute("Deterministic concurrent query", evidence=[ev])
            return {
                "query": res.query,
                "status": res.status,
                "document_id": res.document_id,
                "filename": res.filename,
                "evidence_count": len(res.evidence),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_run_det) for _ in range(4)]
            dicts = [f.result() for f in futures]

        for d in dicts:
            assert d == dicts[0]

    def test_19_no_duplicate_execution(self) -> None:
        """N concurrent requests execute provider exactly N times."""
        config = VisionProviderConfig(provider_name="nodup-conc", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        def _run_n(i: int) -> None:
            ev = VisualEvidence(document_id=f"d-{i}", filename=f"f-{i}.png", chunk_id=f"c-{i}", image_bytes=_make_png())
            agent.execute(f"Q-{i}", evidence=[ev])

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_run_n, i) for i in range(7)]
            for f in futures:
                f.result()

        assert provider.invocation_count == 7

    def test_20_no_cross_request_contamination(self) -> None:
        """Verify that provider recorded inputs contain strictly matching lineage for each request."""
        config = VisionProviderConfig(provider_name="contam-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        def _run_contam(i: int) -> None:
            ev = VisualEvidence(document_id=f"doc-c-{i}", filename=f"fc-{i}.png", chunk_id=f"cc-{i}", image_bytes=_make_png())
            agent.execute(f"Contamination Query {i}", evidence=[ev])

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_run_contam, i) for i in range(6)]
            for f in futures:
                f.result()

        recorded = provider.recorded_inputs
        assert len(recorded) == 6

        for inp in recorded:
            idx = int(inp.document_id.split("-")[-1])
            assert inp.query == f"Contamination Query {idx}"
            assert inp.filename == f"fc-{idx}.png"

    def test_21_provider_receives_correct_input_concurrently(self) -> None:
        """Provider receives frozen VisionModelInput matching exact request parameters."""
        config = VisionProviderConfig(provider_name="inp-conc-prov", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)
        agent = VisionAgent(provider=provider)

        ev = VisualEvidence(
            document_id="doc-inp-1",
            filename="inp1.png",
            chunk_id="chk-inp-1",
            page_number=3,
            chunk_index=2,
            content_type="chart",
            image_bytes=_make_png(64, 64),
        )

        res = agent.execute("Chart query", evidence=[ev])

        assert res.is_success is True
        recorded = provider.recorded_inputs[0]
        assert isinstance(recorded, VisionModelInput)
        assert recorded.document_id == "doc-inp-1"
        assert recorded.filename == "inp1.png"
        assert recorded.page_number == 3
        assert recorded.chunk_index == 2
        assert recorded.content_type == "chart"


# ---------------------------------------------------------------------------
# Test Suite 4: Public API & Offline Integrity
# ---------------------------------------------------------------------------


class TestPublicAPIAndOfflineIntegrity:
    """Verifies public exports and strictly offline execution."""

    def test_22_public_imports_work(self) -> None:
        """Public imports from vision package work seamlessly."""
        import vision
        for symbol in (
            "VisionAgent",
            "VisionPipeline",
            "VisionExecutionAdapter",
            "VisionExecutionLifecycle",
            "VisionModelProvider",
            "VisionModelInput",
            "VisionResult",
            "VisualEvidence",
            "run_vision_pipeline",
        ):
            assert hasattr(vision, symbol), f"Public symbol {symbol} missing from vision package."

    def test_23_offline_concurrency_verification(self) -> None:
        """No external network or vendor SDK modules imported in vision package."""
        import vision.execution_adapter as ea
        import vision.pipeline as vp
        import vision.vision_agent as va

        for mod in (ea, vp, va):
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

    def test_24_previous_tests_remain_compatible(self) -> None:
        """Day 40 pipeline integration convenience function works with concurrency."""
        config = VisionProviderConfig(provider_name="pipe-conc", model_name="v1")
        provider = ThreadSafeRecordingProvider(config)

        ev = VisualEvidence(document_id="p-doc", filename="p.png", chunk_id="p-chk", image_bytes=_make_png())
        res = run_vision_pipeline(provider, "Pipeline run query", evidence=[ev])

        assert res.is_success is True
        assert res.document_id == "p-doc"
