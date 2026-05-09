"""Tests for ``scripts/build_track_b_round2_delta.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_track_b_round2_delta as delta


def _baseline_report() -> dict:
    return {
        "totals": {
            "turns": 84,
            "prefill_sum_s_total": 1642.3,
            "decode_sum_s_total": 295.1,
            "wallclock_s_total": 1948.1,
            "prompt_tokens_total": 100000,
            "completion_tokens_total": 5000,
        },
        "regimes": {
            "tool-call": {"turns": 79, "prefill_sum_s": 1638.8, "decode_sum_s": 280.3},
            "reasoning": {"turns": 5, "prefill_sum_s": 3.4, "decode_sum_s": 14.7},
        },
        "techniques": {
            "T1_cross_turn_ngram": {
                "turns_covered": 84,
                "decode_sum_s_covered": 295.1,
                "decode_sum_s_fraction_of_corpus": 1.0,
                "decode_reduction_ceiling_s": 98.4,
            },
            "T3_schema_aware_tool_drafter": {
                "turns_covered": 79,
                "decode_sum_s_covered": 280.3,
                "decode_sum_s_fraction_of_corpus": 0.95,
                "decode_reduction_ceiling_s": 186.9,
            },
        },
        "files_read_count": 52,
    }


def _patched_report(decode_reduction_s: float = 100.0) -> dict:
    base = _baseline_report()
    base["totals"]["decode_sum_s_total"] -= decode_reduction_s
    base["totals"]["wallclock_s_total"] -= decode_reduction_s
    base["regimes"]["tool-call"]["decode_sum_s"] -= decode_reduction_s
    base["techniques"]["T3_schema_aware_tool_drafter"]["decode_sum_s_covered"] -= decode_reduction_s
    base["techniques"]["T1_cross_turn_ngram"]["decode_sum_s_covered"] -= decode_reduction_s
    base["files_read_count"] = 52
    return base


def test_build_delta_headline_reflects_decode_reduction() -> None:
    report = delta.build_delta(_baseline_report(), _patched_report(100.0))
    headline = report["headline"]
    assert headline["corpus_decode_reduction_s"] == pytest.approx(100.0)
    assert headline["corpus_decode_reduction_pct"] == pytest.approx(100.0 / 295.1)


def test_build_delta_per_technique_measured_vs_ceiling() -> None:
    report = delta.build_delta(_baseline_report(), _patched_report(100.0))
    t3 = report["techniques_delta"]["T3_schema_aware_tool_drafter"]
    assert t3["measured_decode_reduction_s"] == pytest.approx(100.0)
    # Ceiling 186.9; measured 100; ratio = 100/186.9 ≈ 0.535
    assert t3["measured_vs_ceiling_ratio"] == pytest.approx(100.0 / 186.9)


def test_build_delta_handles_no_change() -> None:
    same = _baseline_report()
    report = delta.build_delta(same, same)
    assert report["headline"]["corpus_decode_reduction_s"] == pytest.approx(0.0)
    assert report["headline"]["corpus_decode_reduction_pct"] == pytest.approx(0.0)


def test_build_delta_per_regime_includes_before_after() -> None:
    report = delta.build_delta(_baseline_report(), _patched_report(50.0))
    tc = report["regimes_delta"]["tool-call"]
    assert tc["decode_sum_s_before"] == pytest.approx(280.3)
    assert tc["decode_sum_s_after"] == pytest.approx(280.3 - 50.0)
    assert tc["decode_sum_s_delta"] == pytest.approx(-50.0)
    # Reasoning regime unchanged.
    rs = report["regimes_delta"]["reasoning"]
    assert rs["decode_sum_s_delta"] == pytest.approx(0.0)


def test_build_delta_handles_techniques_added_in_patched_run() -> None:
    """A technique that fired in the patched run but not the baseline
    (e.g., T4 once a plan emitter ships) should still appear in the
    delta report."""

    baseline = _baseline_report()
    patched = _baseline_report()
    patched["techniques"]["T4_plan_structure"] = {
        "turns_covered": 5,
        "decode_sum_s_covered": 20.0,
        "decode_reduction_ceiling_s": 10.0,
    }
    report = delta.build_delta(baseline, patched)
    t4 = report["techniques_delta"]["T4_plan_structure"]
    assert t4["turns_covered_before"] is None
    assert t4["turns_covered_after"] == 5


def test_main_writes_delta_file(tmp_path: Path) -> None:
    bp = tmp_path / "baseline.json"
    pp = tmp_path / "patched.json"
    bp.write_text(json.dumps(_baseline_report()))
    pp.write_text(json.dumps(_patched_report(40.0)))
    out = tmp_path / "delta.json"
    rc = delta.main(["--baseline", str(bp), "--patched", str(pp), "--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["schema"] == delta.SCHEMA
    assert payload["headline"]["corpus_decode_reduction_s"] == pytest.approx(40.0)


def test_main_returns_nonzero_when_baseline_missing(tmp_path: Path) -> None:
    bp = tmp_path / "missing.json"
    pp = tmp_path / "patched.json"
    pp.write_text(json.dumps(_patched_report()))
    out = tmp_path / "delta.json"
    rc = delta.main(["--baseline", str(bp), "--patched", str(pp), "--output", str(out)])
    assert rc == 2
