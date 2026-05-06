from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from lumo_flywheel_serving.track_b import (
    TrackBRoundManager,
    collect_cutlass_round_memory,
    evaluate_b1_metrics,
    evaluate_b2_metrics,
    evaluate_b3_metrics,
)


def _write_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"prompt_tokens": 4096, "output_tokens": 256, "turn_index": 0, "task_type": "code"},
        {"prompt_tokens": 8192, "output_tokens": 128, "turn_index": 1, "task_type": "edit"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_cutlass_round(root: Path) -> Path:
    round_dir = root / "qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z"
    round_dir.mkdir(parents=True)
    (round_dir / "run_log.json").write_text(
        json.dumps({"outcome": "ROUND_BLOCKED", "terminal_condition": "compile_failures_3x"}),
        encoding="utf-8",
    )
    (round_dir / "mutations_rejected.tsv").write_text(
        "iteration\tcandidate_uuid\tmutation_hash\trejection_reason\tfirst_diverging_probe_index\ttolerance_overshoot\n"
        "020\tabc\tdeadbeef\tl0c_generation_speed_gate\t\t\n",
        encoding="utf-8",
    )
    (round_dir / "research_memory.tsv").write_text(
        "iteration\tfailure_class\tnext_implication\n020\tperformance\tdeprioritize_until_speed_gate_hypothesis_changes\n",
        encoding="utf-8",
    )
    return round_dir


def test_track_b_launch_extends_cutlass_memory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    trace = repo / "benchmark_blueprints" / "families" / "responses-sdk-adapter-cutover" / "seed_trace_v5.jsonl"
    _write_trace(trace)
    round_root = repo / "output" / "auto_research"
    _write_cutlass_round(round_root)
    report_dir = repo / "docs" / "reports" / "auto_research"
    report_dir.mkdir(parents=True)
    (report_dir / "l0c-fp8-cutlass-round-20260505-closeout.md").write_text("closeout", encoding="utf-8")

    result = TrackBRoundManager(repo_root=repo).launch(
        round_root=round_root,
        workload_trace=trace,
        dry_run=True,
        round_id="track-b-test",
    )

    spec = yaml.safe_load((result.round_dir / "round_spec.yaml").read_text(encoding="utf-8"))
    assert spec["round_type"] == "track_b_quality_bounded_mutation"
    assert spec["extends_round_type"] == "l0c_mutation"
    assert spec["target_decode_tps"] == 37.5
    assert spec["success_criteria"]["candidate_acceptance_incremental_speedup_at_least"] == 1.2
    assert spec["success_criteria"]["measurement_harness"] == "real_vllm_workload_first_five"
    assert spec["success_criteria"]["real_workload_windows"]["warm_completions_measured"] == 4
    assert spec["prior_cutlass_memory"]["round_count_indexed"] == 1
    assert (result.round_dir / "prior_cutlass_memory.md").is_file()
    prior_rejections = (result.round_dir / "prior_cutlass_rejections.tsv").read_text(encoding="utf-8")
    assert "l0c_generation_speed_gate" in prior_rejections
    strategy = (result.round_dir / "strategy_brief.md").read_text(encoding="utf-8")
    assert "prior memory at `prior_cutlass_memory.md`" in strategy
    assert "Do not retry schedule/tile/stage/caller mutations" in strategy


def test_collect_cutlass_round_memory_indexes_recent_rounds(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    round_root = repo / "output" / "auto_research"
    _write_cutlass_round(round_root)

    memory = collect_cutlass_round_memory(repo, round_root)

    assert memory["round_count_indexed"] == 1
    assert memory["recent_rounds"][0]["terminal_condition"] == "compile_failures_3x"
    assert memory["rejected_rows_sample"][0]["rejection_reason"] == "l0c_generation_speed_gate"
    assert memory["summary"]["track_a_surface_status"] == "exhausted_for_2x_target"


def test_track_b_quality_metric_evaluators() -> None:
    assert evaluate_b1_metrics(
        {"mean_kl": 0.01, "p95_kl": 0.1, "top1_agreement": 0.99, "entropy_delta": 0.01}
    )["pass"]
    assert not evaluate_b1_metrics(
        {"mean_kl": 0.2, "p95_kl": 0.1, "top1_agreement": 0.99, "entropy_delta": 0.01}
    )["pass"]
    assert evaluate_b2_metrics(
        {
            "benchmark_deltas_pp": {"mmlu": -0.2, "gsm8k": -0.5},
            "workload_behavioral_judge_score_delta": 0.0,
            "needle_recall_ratio": 0.99,
        }
    )["pass"]
    assert evaluate_b3_metrics(
        {
            "benchmark_deltas_pp": {"mmlu": -0.2, "gsm8k": -0.4},
            "aggregate_quality_score_delta_pp": -0.2,
            "mauve": 0.96,
            "perplexity_ratio_delta": 0.001,
        }
    )["pass"]


def test_run_b1_distributional_script(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.yaml"
    fixture.write_text(
        yaml.safe_dump(
            {
                "thresholds": {
                    "mean_kl": 0.05,
                    "p95_kl": 0.25,
                    "top1_agreement": 0.98,
                    "entropy_delta_abs": 0.05,
                }
            }
        ),
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {"mean_kl": 0.01, "p95_kl": 0.1, "top1_agreement": 0.99, "entropy_delta": 0.01}
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "run_b1_distributional.py"),
            "--candidate-metrics",
            str(metrics),
            "--fixture",
            str(fixture),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["pass"] is True
