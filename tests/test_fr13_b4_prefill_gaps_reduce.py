"""What the out-of-decode decomposition is allowed to claim.

The reducer's whole load-bearing claim is that time outside an
`fr13.fixed32.step` NVTX range is not idle -- it is a chunked-prefill/mixed
forward pass, because that range is pushed only on a pure-decode step. These
tests build synthetic traces in which the answer is known in closed form, so a
misclassification shows up as a WRONG NUMBER and not merely a wrong shape:

  1. a class of kernel that never appears inside a decode step is prefill by
     construction, and all of its out-of-step time lands in class (a);
  2. a class that appears in decode steps contributes its own in-step per-pass
     rate to class (d) and only the EXCESS to class (a) -- so a mixed batch's
     shared GEMM is not laundered wholesale into either bucket;
  3. real GPU idle in a gap lands in class (b/c) and nowhere else;
  4. the pass identity is fail-closed: a trace whose `sample_readback`
     instances do not reconcile against the step ranges raises rather than
     emitting a partial decomposition.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]

MS = 1_000_000
S = 1_000_000_000


def _load(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gaps = _load("fr13_b4_prefill_gaps_reduce", "scripts/fr13_b4_prefill_gaps_reduce.py")


class _TraceBuilder:
    """Minimal Nsight-shaped sqlite: StringIds, NVTX_EVENTS, KERNEL."""

    def __init__(self, path: Path) -> None:
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript("""
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, textId INTEGER, text TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, gridX INTEGER, shortName INTEGER);
        """)
        self._ids: dict[str, int] = {}

    def sid(self, value: str) -> int:
        if value not in self._ids:
            i = len(self._ids) + 1
            self._ids[value] = i
            self.conn.execute("INSERT INTO StringIds VALUES (?,?)", (i, value))
        return self._ids[value]

    def nvtx(self, name: str, start: int, end: int) -> None:
        self.conn.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,NULL)",
            (start, end, self.sid(name)),
        )

    def kernel(self, name: str, start: int, dur: int, grid_x: int = 1) -> None:
        self.conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)",
            (start, start + dur, grid_x, self.sid(name)),
        )

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def _closed_form_trace(tmp_path: Path, *, gap_idle_ms: int = 0) -> Path:
    """Two decode steps, one mixed pass between them, everything exact.

    decode step   100 ms wall, filled by 60 ms `decode_only` + 40 ms `shared`
    gap          1000 ms wall, one mixed pass carrying
                    600 ms `prefill_only`   -> class (a), exclusively
                    400 - idle ms `shared`  -> 40 ms is the in-step rate and
                                               therefore class (d); the rest is
                                               class (a) excess
    """
    path = tmp_path / f"trace_idle{gap_idle_ms}.sqlite"
    tb = _TraceBuilder(path)
    step_ns = 100 * MS
    gap_ns = 1000 * MS
    t = 0
    for i in range(2):
        tb.nvtx("fr13.fixed32.step", t, t + step_ns)
        tb.nvtx("fr13.fixed32.sample_readback", t + 90 * MS, t + 95 * MS)
        tb.kernel("decode_only", t, 60 * MS, grid_x=2)
        tb.kernel("shared", t + 60 * MS, 40 * MS, grid_x=2)
        t += step_ns
        if i == 0:
            gap_start = t
            tb.nvtx("fr13.fixed32.sample_readback",
                    gap_start + 950 * MS, gap_start + 960 * MS)
            tb.kernel("prefill_only", gap_start, 600 * MS, grid_x=9)
            tb.kernel("shared", gap_start + 600 * MS,
                      (400 - gap_idle_ms) * MS, grid_x=2)
            t += gap_ns
    tb.close()
    return path


def test_prefill_exclusive_class_is_attributed_whole(tmp_path: Path) -> None:
    doc = gaps.reduce_capture(str(_closed_form_trace(tmp_path)))
    ev = doc["kernel_evidence"]
    names = {r["kernel"] for r in ev["prefill_exclusive_classes"]}
    assert names == {"prefill_only"}, names
    assert ev["prefill_exclusive_plain_sum_s"] == pytest.approx(0.600, abs=1e-6)


def test_shared_class_splits_at_the_in_step_rate(tmp_path: Path) -> None:
    doc = gaps.reduce_capture(str(_closed_form_trace(tmp_path)))
    ev = doc["kernel_evidence"]
    # `shared` runs 40 ms in each of 2 decode steps -> 40 ms/pass is the decode
    # rate; the mixed pass ran it for 400 ms, so 40 ms is class (d) and the
    # remaining 360 ms is prefill excess.
    assert ev["cobatched_decode_plain_sum_s"] == pytest.approx(0.040, abs=1e-6)
    assert ev["prefill_excess_on_shared_plain_sum_s"] == pytest.approx(0.360, abs=1e-6)
    assert doc["classes"]["a_chunked_prefill_compute_s"] == pytest.approx(0.960, abs=1e-3)
    assert doc["classes"]["d_cobatched_decode_compute_s"] == pytest.approx(0.040, abs=1e-3)


def test_gap_idle_is_the_only_home_of_class_b(tmp_path: Path) -> None:
    doc = gaps.reduce_capture(str(_closed_form_trace(tmp_path, gap_idle_ms=250)))
    cl = doc["classes"]
    assert cl["b_gpu_idle_awaiting_demand_s"] == pytest.approx(0.250, abs=1e-6)
    assert doc["idle"]["longest_out_of_step_idle_interval_ms"] == pytest.approx(250.0, abs=1e-3)
    # and it is NOT silently added to either compute class
    total = (cl["a_chunked_prefill_compute_s"] + cl["d_cobatched_decode_compute_s"]
             + cl["b_gpu_idle_awaiting_demand_s"])
    assert total == pytest.approx(doc["window"]["outside_decode_s"], abs=1e-3)


def test_window_shares_are_exact(tmp_path: Path) -> None:
    doc = gaps.reduce_capture(str(_closed_form_trace(tmp_path)))
    w = doc["window"]
    assert w["pure_decode_step_ranges"] == 2
    assert w["mixed_forward_passes"] == 1
    assert w["forward_passes_total"] == 3
    assert w["wall_s"] == pytest.approx(1.200, abs=1e-6)
    assert w["inside_decode_pct"] == pytest.approx(100 * 0.2 / 1.2, abs=1e-6)
    assert w["outside_decode_pct"] == pytest.approx(100 * 1.0 / 1.2, abs=1e-6)
    assert w["gaps_over_threshold"] == 1


def test_pass_identity_is_fail_closed(tmp_path: Path) -> None:
    """A sample_readback that vanishes inside a step range must raise."""
    path = tmp_path / "broken.sqlite"
    tb = _TraceBuilder(path)
    tb.nvtx("fr13.fixed32.step", 0, 100 * MS)
    tb.nvtx("fr13.fixed32.step", 1100 * MS, 1200 * MS)
    # only ONE readback, and it is in the gap: pure count 0 != 2 step ranges
    tb.nvtx("fr13.fixed32.sample_readback", 500 * MS, 510 * MS)
    tb.kernel("decode_only", 0, 60 * MS, grid_x=2)
    tb.kernel("prefill_only", 200 * MS, 600 * MS, grid_x=9)
    tb.kernel("decode_only", 1100 * MS, 60 * MS, grid_x=2)
    tb.close()
    with pytest.raises(SystemExit, match="pass identity broken"):
        gaps.reduce_capture(str(path))


def test_a_mixed_pass_outside_every_gap_is_fail_closed(tmp_path: Path) -> None:
    """A mixed pass in a SUB-threshold gap is not silently dropped."""
    path = tmp_path / "stray.sqlite"
    tb = _TraceBuilder(path)
    tb.nvtx("fr13.fixed32.step", 0, 100 * MS)
    tb.nvtx("fr13.fixed32.sample_readback", 90 * MS, 95 * MS)
    # 50 ms gap -- below the 1000 ms threshold -- but it carries a mixed pass
    tb.nvtx("fr13.fixed32.sample_readback", 120 * MS, 125 * MS)
    tb.nvtx("fr13.fixed32.step", 150 * MS, 250 * MS)
    tb.nvtx("fr13.fixed32.sample_readback", 240 * MS, 245 * MS)
    tb.kernel("decode_only", 0, 60 * MS, grid_x=2)
    tb.kernel("prefill_only", 110 * MS, 30 * MS, grid_x=9)
    tb.kernel("decode_only", 150 * MS, 60 * MS, grid_x=2)
    tb.close()
    with pytest.raises(SystemExit, match="outside every gap"):
        gaps.reduce_capture(str(path))


def test_bogus_boundary_range_allowance_is_bounded(tmp_path: Path) -> None:
    """One capture-boundary artifact is tolerated; a pile of them is not."""
    def build(n_bogus: int) -> Path:
        path = tmp_path / f"bogus{n_bogus}.sqlite"
        tb = _TraceBuilder(path)
        t = 0
        for _ in range(2):
            tb.nvtx("fr13.fixed32.step", t, t + 100 * MS)
            tb.nvtx("fr13.fixed32.sample_readback", t + 90 * MS, t + 95 * MS)
            tb.kernel("decode_only", t, 60 * MS, grid_x=2)
            t += 1100 * MS
        tb.nvtx("fr13.fixed32.sample_readback", 300 * MS, 310 * MS)
        tb.kernel("prefill_only", 200 * MS, 600 * MS, grid_x=9)
        for i in range(n_bogus):
            far = 780_000_000_000_000 + i * S
            tb.nvtx("fr13.fixed32.step", far, far + 100 * MS)
        tb.close()
        return path

    doc = gaps.reduce_capture(str(build(1)))
    assert doc["window"]["bogus_step_ranges_dropped"] == 1
    assert doc["window"]["pure_decode_step_ranges"] == 2
    with pytest.raises(SystemExit, match="Refusing to decompose"):
        gaps.reduce_capture(str(build(3)))


def test_union_never_double_counts_concurrent_streams(tmp_path: Path) -> None:
    """Gap GPU busy is a union: two overlapping kernels are not 2x the wall."""
    path = tmp_path / "overlap.sqlite"
    tb = _TraceBuilder(path)
    tb.nvtx("fr13.fixed32.step", 0, 100 * MS)
    tb.nvtx("fr13.fixed32.sample_readback", 90 * MS, 95 * MS)
    tb.kernel("decode_only", 0, 60 * MS, grid_x=2)
    gap = 100 * MS
    tb.nvtx("fr13.fixed32.sample_readback", gap + 900 * MS, gap + 910 * MS)
    tb.kernel("prefill_only", gap, 500 * MS, grid_x=9)
    tb.kernel("prefill_only", gap + 250 * MS, 500 * MS, grid_x=9)
    tb.nvtx("fr13.fixed32.step", gap + 1000 * MS, gap + 1100 * MS)
    tb.nvtx("fr13.fixed32.sample_readback", gap + 1090 * MS, gap + 1095 * MS)
    tb.kernel("decode_only", gap + 1000 * MS, 60 * MS, grid_x=2)
    tb.close()
    doc = gaps.reduce_capture(str(path))
    g = doc["gaps"][0]
    assert g["gpu_busy_s"] == pytest.approx(0.750, abs=1e-6)   # union, not 1.0
    assert g["gpu_idle_s"] == pytest.approx(0.250, abs=1e-6)
    assert doc["kernel_evidence"]["plain_sum_vs_union_ratio"] == pytest.approx(
        1.0 / 0.75, abs=1e-6)
