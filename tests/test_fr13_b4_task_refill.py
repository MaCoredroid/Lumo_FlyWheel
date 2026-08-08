"""Continuous task-pool refill for the B4 SWE campaign.

Nominal B4 co-residency is 4, but a pool of exactly 4 tasks decays
4 -> 3 -> 2 -> 1 as sessions end minutes apart; the alignment study attributes
54.7% of the co-residency shortfall to that task-end skew.  FR13_B4_TASK_REFILL
holds the slot count full from a larger pool.  It is DEFAULT OFF, and OFF must
leave admission exactly as every banked contract measured it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

DRIVER = SCRIPTS / "fr13_b4_campaign_driver.sh"
SERVE_VARIANT = SCRIPTS / "fr13_bigdenom_swe_serve_variant.sh"
ORCHESTRATOR = SCRIPTS / "run_swe_bench_q36_a.py"


def _load_runner() -> Any:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "fr13_task_refill_runner_test", ORCHESTRATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# --------------------------------------------------------------------------- #
# the env gate                                                                 #
# --------------------------------------------------------------------------- #
def test_refill_is_off_by_default(monkeypatch):
    monkeypatch.delenv(runner.FR13_TASK_REFILL_ENV, raising=False)
    assert runner._task_refill_enabled() is False


def test_refill_env_is_strictly_binary(monkeypatch):
    monkeypatch.setenv(runner.FR13_TASK_REFILL_ENV, "1")
    assert runner._task_refill_enabled() is True
    monkeypatch.setenv(runner.FR13_TASK_REFILL_ENV, "0")
    assert runner._task_refill_enabled() is False
    for bad in ("true", "yes", "2", ""):
        monkeypatch.setenv(runner.FR13_TASK_REFILL_ENV, bad)
        with pytest.raises(runner.Fixed32BoundaryError, match="exactly 0 or 1"):
            runner._task_refill_enabled()


# --------------------------------------------------------------------------- #
# the admitter                                                                 #
# --------------------------------------------------------------------------- #
class _DepthProbe:
    """Records the true concurrent depth seen inside the job body."""

    def __init__(self, hold_s: float = 0.02) -> None:
        self.lock = threading.Lock()
        self.depth = 0
        self.peak = 0
        self.started: list[str] = []
        self.hold_s = hold_s

    def job(self, instance_id: str) -> dict[str, Any]:
        with self.lock:
            self.depth += 1
            self.peak = max(self.peak, self.depth)
            self.started.append(instance_id)
        time.sleep(self.hold_s)
        with self.lock:
            self.depth -= 1
        return {"instance_id": instance_id}


def test_pool_drains_at_full_width_without_exceeding_the_slot_count(tmp_path):
    probe = _DepthProbe()
    ids = [f"t{i:02d}" for i in range(12)]

    results = runner._run_task_pool_with_refill(
        job=probe.job,
        instance_ids=ids,
        concurrency=4,
        ledger_path=tmp_path / "ledger.jsonl",
        summary_path=tmp_path / "summary.json",
    )

    assert sorted(r["instance_id"] for r in results) == ids
    assert probe.peak == 4, "the pool must run full width"
    assert probe.depth == 0
    # Every task was admitted exactly once.
    assert sorted(probe.started) == ids


def test_every_completion_is_immediately_backfilled(tmp_path):
    """The refill property itself: while the pool has work left, an admit
    follows every completion, so depth returns to the slot count."""
    ids = [f"t{i:02d}" for i in range(10)]
    runner._run_task_pool_with_refill(
        job=lambda iid: {"instance_id": iid},
        instance_ids=ids,
        concurrency=3,
        ledger_path=tmp_path / "ledger.jsonl",
        summary_path=tmp_path / "summary.json",
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
    ]
    assert [e["event"] for e in events].count("admit") == len(ids)
    assert [e["event"] for e in events].count("complete") == len(ids)

    # Walk the ledger: depth may never exceed the slots, and any completion that
    # leaves work pending must be followed by an admit before the next wait.
    depth = 0
    for index, event in enumerate(events):
        depth = event["depth"]
        assert depth <= 3
        if event["event"] == "complete" and event["pending"] > 0:
            tail = events[index + 1 :]
            assert tail, "a completion with work pending must be backfilled"
            assert any(e["event"] == "admit" for e in tail[:3])
    assert depth == 0


def test_the_first_admissions_fill_every_slot_before_any_completion(tmp_path):
    ids = [f"t{i:02d}" for i in range(8)]
    runner._run_task_pool_with_refill(
        job=lambda iid: {"instance_id": iid},
        instance_ids=ids,
        concurrency=4,
        ledger_path=tmp_path / "ledger.jsonl",
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
    ]
    assert [e["event"] for e in events[:4]] == ["admit"] * 4
    assert [e["depth"] for e in events[:4]] == [1, 2, 3, 4]


def test_results_are_collected_in_completion_order_not_submission_order(tmp_path):
    """ex.map blocks on the FIRST task; the refill admitter must not."""
    release = threading.Event()

    def job(instance_id: str) -> dict[str, Any]:
        if instance_id == "slow":
            release.wait(timeout=5.0)
        return {"instance_id": instance_id}

    ids = ["slow", "a", "b", "c", "d", "e"]

    finished: list[str] = []

    def watcher() -> None:
        # Let the fast tasks land, then release the straggler.
        time.sleep(0.15)
        release.set()

    thread = threading.Thread(target=watcher)
    thread.start()
    results = runner._run_task_pool_with_refill(
        job=job, instance_ids=ids, concurrency=3
    )
    thread.join()
    finished = [r["instance_id"] for r in results]

    assert sorted(finished) == sorted(ids)
    # The straggler was submitted first but must not be collected first.
    assert finished[0] != "slow"


def test_a_failing_task_stops_admission_and_propagates(tmp_path):
    admitted: list[str] = []
    lock = threading.Lock()

    def job(instance_id: str) -> dict[str, Any]:
        with lock:
            admitted.append(instance_id)
        if instance_id == "t01":
            raise SystemExit("FR13 CIRCUIT BREAKER: endpoint is dead")
        time.sleep(0.05)
        return {"instance_id": instance_id}

    ids = [f"t{i:02d}" for i in range(20)]

    with pytest.raises(SystemExit, match="CIRCUIT BREAKER"):
        runner._run_task_pool_with_refill(
            job=job,
            instance_ids=ids,
            concurrency=2,
            ledger_path=tmp_path / "ledger.jsonl",
            summary_path=tmp_path / "summary.json",
        )

    # The pool was dropped: the campaign must not burn all 20 tasks.
    assert len(admitted) < len(ids)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["aborted"] is True


def test_summary_reduces_the_ledger_to_citable_occupancy(tmp_path):
    ids = [f"t{i:02d}" for i in range(9)]
    runner._run_task_pool_with_refill(
        job=lambda iid: (time.sleep(0.02), {"instance_id": iid})[1],
        instance_ids=ids,
        concurrency=3,
        ledger_path=tmp_path / "ledger.jsonl",
        summary_path=tmp_path / "summary.json",
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["schema"] == "fr13.task_refill.summary.v1"
    assert summary["task_count"] == 9
    assert summary["slots"] == 3
    assert summary["admissions"] == 9
    assert summary["completed"] == 9
    assert summary["peak_depth"] == 3
    assert summary["aborted"] is False
    assert 0.0 < summary["time_weighted_mean_depth"] <= 3.0
    assert 0.0 <= summary["full_width_fraction"] <= 1.0
    # The depth is a worker-thread depth, and the artifact must say so rather
    # than let a reader mistake it for measured decode co-residency.
    assert "UPPER BOUND" in summary["note"]


def test_reducer_time_weights_depth_rather_than_counting_events():
    # Depth 2 for 10s then depth 1 for 90s: an event-count mean would say 1.5.
    events = [
        {"event": "admit", "t_s": 0.0, "depth": 1, "pending": 1},
        {"event": "admit", "t_s": 0.0, "depth": 2, "pending": 0},
        {"event": "complete", "t_s": 10.0, "depth": 1, "pending": 0},
        {"event": "complete", "t_s": 100.0, "depth": 0, "pending": 0},
    ]
    summary = runner._reduce_task_refill_ledger(events, concurrency=2, task_count=2)
    assert summary["arm_wall_s"] == pytest.approx(100.0)
    assert summary["time_weighted_mean_depth"] == pytest.approx(
        (2 * 10.0 + 1 * 90.0) / 100.0
    )
    assert summary["full_width_fraction"] == pytest.approx(0.10)


def test_zero_slots_is_rejected():
    with pytest.raises(runner.Fixed32BoundaryError, match="concurrency >= 1"):
        runner._run_task_pool_with_refill(
            job=lambda iid: {"instance_id": iid},
            instance_ids=["a"],
            concurrency=0,
        )


# --------------------------------------------------------------------------- #
# wiring: default OFF, and the contract caveat is written down                 #
# --------------------------------------------------------------------------- #
def test_orchestrator_keeps_the_unmodified_admission_path_for_default_off():
    text = ORCHESTRATOR.read_text(encoding="utf-8")
    # OFF must still be the historical ThreadPoolExecutor.map over the subset.
    assert "for res in ex.map(_job, instance_ids):" in text
    assert "elif task_refill:" in text
    # A pool run needs a pool.
    assert "needs a task pool LARGER than the slot" in text
    assert "is meaningless at concurrency 1" in text


def test_orchestrator_documents_that_depth_is_an_upper_bound():
    text = ORCHESTRATOR.read_text(encoding="utf-8")
    assert "WHAT IT DOES NOT FIX" in text
    assert "UPPER BOUND" in text


def test_shell_layers_default_the_refill_off_and_validate_it():
    driver = DRIVER.read_text(encoding="utf-8")
    variant = SERVE_VARIANT.read_text(encoding="utf-8")
    for text in (driver, variant):
        assert "FR13_B4_TASK_REFILL=${FR13_B4_TASK_REFILL:-0}" in text
        assert "FR13_B4_TASK_REFILL must be exactly 0 or 1" in text
    # The driver hands it to both arm vehicles.
    assert driver.count('FR13_B4_TASK_REFILL="$FR13_B4_TASK_REFILL"') == 2
    # And the serve variant exports it to the orchestrator process.
    assert "export FR13_B4_TASK_REFILL" in variant


def test_the_citability_caveat_is_recorded_where_it_is_turned_on():
    driver = DRIVER.read_text(encoding="utf-8")
    variant = SERVE_VARIANT.read_text(encoding="utf-8")
    assert "NOT exact4-citable without a contract update" in driver
    assert "requires its own contract" in variant
