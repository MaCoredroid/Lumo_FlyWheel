from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_track_b_e2e_task as runner  # noqa: E402


def test_runner_cli_accepts_documented_ncu_mode_flag() -> None:
    script = SCRIPTS / "run_track_b_e2e_task.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert "--ncu-mode" in result.stdout


def test_runner_accepts_family_variant_task_id_form() -> None:
    assert runner._resolve_task_args("sqlalchemy-2-session-modernization/v1-clean-baseline", "v1-clean-baseline") == (
        "sqlalchemy-2-session-modernization",
        "v1-clean-baseline",
    )
    assert runner._resolve_task_args("sqlalchemy-2-session-modernization/v2", "v1-clean-baseline") == (
        "sqlalchemy-2-session-modernization",
        "v2",
    )


def test_runner_rejects_conflicting_task_id_variant() -> None:
    with pytest.raises(ValueError, match="conflict"):
        runner._resolve_task_args("sqlalchemy-2-session-modernization/v2", "v3")


def test_runner_requires_trace_out_in_command_template() -> None:
    runner._validate_codex_command_template("codex exec --trace-out {trace_out} --cwd {workspace}")
    with pytest.raises(ValueError, match="trace_out"):
        runner._validate_codex_command_template("codex exec --cwd {workspace}")


def test_runner_normalizes_vllm_request_metrics_jsonl(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    source = tmp_path / "vllm_requests.jsonl"
    source.write_text(
        json.dumps(
            {
                "request_id": "req-1",
                "prompt_tokens": 50,
                "generation_tokens": 12,
                "prefill_s": 0.2,
                "decode_s": 1.0,
                "spec_decode_num_accepted_tokens": 3,
                "spec_decode_num_draft_tokens": 12,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    runner._write_vllm_per_turn_from_jsonl(task_dir, source)

    payload = json.loads((task_dir / "vllm_per_turn.json").read_text(encoding="utf-8"))
    metrics = payload["requests"]["req-1"]
    assert metrics["completion_tokens"] == 12
    assert metrics["prefill_sum_s"] == 0.2
    assert metrics["decode_sum_s"] == 1.0
    assert metrics["decode_tps"] == 12.0
    assert metrics["accepted_per_draft_token"] == 0.25
    assert (task_dir / "vllm_request_metrics.jsonl").is_file()


def test_runner_rejects_incomplete_vllm_request_metrics_jsonl(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    source = tmp_path / "vllm_requests.jsonl"
    source.write_text(json.dumps({"request_id": "req-1", "prompt_tokens": 50}) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing numeric fields"):
        runner._write_vllm_per_turn_from_jsonl(task_dir, source)


def test_runner_rejects_empty_prometheus_request_join(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    metric_names = [
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:request_prefill_kv_computed_tokens_sum",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_decode_time_seconds_sum",
        "vllm:inter_token_latency_seconds_sum",
    ]
    metrics_before = "\n".join(f'{name}{{engine="0"}} 10' for name in metric_names)
    metrics_after = "\n".join(f'{name}{{engine="0"}} 20' for name in metric_names)

    with pytest.raises(RuntimeError, match="request-keyed vLLM rows"):
        runner._write_vllm_per_turn(task_dir, metrics_before, metrics_after)
