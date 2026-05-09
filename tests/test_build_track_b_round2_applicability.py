"""Tests for ``scripts/build_track_b_round2_applicability.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_track_b_round2_applicability as analyzer


def _row(**kwargs: object) -> dict[str, object]:
    base = {
        "schema": "lumo.track_b.vllm_request_metrics.v1",
        "upstream_status": 200,
        "saw_response_completed": True,
        "regime": "tool-call",
        "tool_call_observed": True,
        "prefill_sum_s": 1.0,
        "decode_sum_s": 0.5,
        "wallclock_s": 1.6,
        "prompt_tokens": 100,
        "completion_tokens": 20,
    }
    base.update(kwargs)
    return base


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_build_report_groups_per_regime(tmp_path: Path) -> None:
    rows = [
        _row(regime="tool-call", decode_sum_s=10.0),
        _row(regime="tool-call", decode_sum_s=10.0),
        _row(regime="reasoning", decode_sum_s=2.0, tool_call_observed=False),
    ]
    _write_jsonl(tmp_path / "vllm_request_metrics.jsonl", rows)
    report = analyzer.build_report([tmp_path / "vllm_request_metrics.jsonl"])
    assert report["totals"]["turns"] == 3
    assert report["regimes"]["tool-call"]["turns"] == 2
    assert report["regimes"]["reasoning"]["turns"] == 1
    assert report["regimes"]["tool-call"]["decode_sum_s"] == pytest.approx(20.0)


def test_build_report_t3_covers_only_tool_call(tmp_path: Path) -> None:
    rows = [
        _row(regime="tool-call", decode_sum_s=8.0),
        _row(regime="reasoning", decode_sum_s=2.0, tool_call_observed=False),
    ]
    _write_jsonl(tmp_path / "vllm_request_metrics.jsonl", rows)
    report = analyzer.build_report([tmp_path / "vllm_request_metrics.jsonl"])
    t3 = report["techniques"]["T3_schema_aware_tool_drafter"]
    assert t3["turns_covered"] == 1
    assert t3["decode_sum_s_covered"] == pytest.approx(8.0)
    # 3.0x target on 8s of 10s total decode -> reduction = 8 * (1 - 1/3) = 5.333s
    assert t3["decode_reduction_ceiling_s"] == pytest.approx(8.0 * 2 / 3)
    # Reduction as fraction of corpus decode (10s)
    assert t3["decode_reduction_ceiling_fraction_of_corpus"] == pytest.approx(8.0 * 2 / 3 / 10.0)


def test_build_report_t1_fires_on_every_turn(tmp_path: Path) -> None:
    rows = [
        _row(regime="tool-call", decode_sum_s=5.0),
        _row(regime="reasoning", decode_sum_s=5.0, tool_call_observed=False),
    ]
    _write_jsonl(tmp_path / "vllm_request_metrics.jsonl", rows)
    report = analyzer.build_report([tmp_path / "vllm_request_metrics.jsonl"])
    t1 = report["techniques"]["T1_cross_turn_ngram"]
    assert t1["turns_covered_fraction"] == pytest.approx(1.0)
    assert t1["decode_sum_s_covered"] == pytest.approx(10.0)


def test_build_report_t5_speedup_zero_reduction(tmp_path: Path) -> None:
    rows = [_row()]
    _write_jsonl(tmp_path / "vllm_request_metrics.jsonl", rows)
    report = analyzer.build_report([tmp_path / "vllm_request_metrics.jsonl"])
    t5 = report["techniques"]["T5_lifecycle"]
    assert t5["decode_speedup_target_x"] == pytest.approx(1.0)
    assert t5["decode_reduction_ceiling_s"] == pytest.approx(0.0)


def test_build_report_skips_failed_or_unfinished_rows(tmp_path: Path) -> None:
    rows = [
        _row(),
        _row(upstream_status=429),
        _row(saw_response_completed=False),
        _row(schema="some.other.schema"),
    ]
    _write_jsonl(tmp_path / "vllm_request_metrics.jsonl", rows)
    report = analyzer.build_report([tmp_path / "vllm_request_metrics.jsonl"])
    assert report["totals"]["turns"] == 1


def test_build_report_recursively_walks_directories(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "task-a/run_01/vllm_request_metrics.jsonl", [_row()])
    _write_jsonl(tmp_path / "task-b/run_02/vllm_request_metrics.jsonl", [_row()])
    paths = analyzer._iter_jsonl_files([tmp_path])
    assert len(paths) == 2
    report = analyzer.build_report(paths)
    assert report["totals"]["turns"] == 2


def test_build_report_t4_disabled_by_default(tmp_path: Path) -> None:
    rows = [_row(regime="reasoning", tool_call_observed=False)]
    _write_jsonl(tmp_path / "vllm_request_metrics.jsonl", rows)
    report = analyzer.build_report([tmp_path / "vllm_request_metrics.jsonl"])
    assert "T4_plan_structure_predrafting" not in report["techniques"]


def test_main_writes_report_file(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "vllm_request_metrics.jsonl", [_row()])
    out = tmp_path / "report.json"
    rc = analyzer.main(["--input", str(tmp_path), "--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["schema"] == analyzer.SCHEMA
    assert payload["totals"]["turns"] == 1
