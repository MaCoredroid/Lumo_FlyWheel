from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
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
    assert "--runtime-config-hash" in result.stdout


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
    runner._validate_codex_command_template("codex exec --json", require_trace_out=False)
    with pytest.raises(ValueError, match="trace_out"):
        runner._validate_codex_command_template("codex exec --cwd {workspace}")


def test_runner_writes_deferred_vllm_metrics(tmp_path: Path) -> None:
    runner._write_deferred_vllm_per_turn(
        tmp_path,
        runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        reason="vllm_request_metrics_join_available",
    )

    payload = json.loads((tmp_path / "vllm_per_turn.json").read_text(encoding="utf-8"))
    assert payload["deferred"] is True
    assert payload["requests"] == {}
    assert payload["deferred_reason"] == "vllm_request_metrics_join_available"


def test_runner_rejects_unstamped_runtime_hash() -> None:
    runner._validate_runtime_config_hash("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    with pytest.raises(ValueError, match="runtime-config-hash"):
        runner._validate_runtime_config_hash("not-a-runtime-hash")
    with pytest.raises(ValueError, match="runtime-config-hash"):
        runner._validate_runtime_config_hash("sha256:")


def test_run_one_rejects_unstamped_runtime_hash_before_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="runtime-config-hash"):
        runner.run_one(
            Namespace(
                runtime_config_hash="not-a-runtime-hash",
                out_root=str(tmp_path),
            ),
            "missing-family",
            "v1-clean-baseline",
        )

    assert not any(tmp_path.iterdir())


def test_run_one_can_record_missing_workspace_diagnostic(tmp_path: Path) -> None:
    args = Namespace(
        runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        out_root=str(tmp_path),
        round=0,
        attempt=2,
        codex_command_template="codex exec --json",
        vllm_request_metrics_jsonl="",
        ncu_mode=False,
        defer_codex_trace_out=True,
        defer_vllm_request_metrics_join=True,
        defer_dcgm_profile_fields=True,
        allow_missing_workspace_diagnostic=True,
    )

    rc = runner.run_one(args, "missing-family", "v1-clean-baseline")

    task_dir = tmp_path / "round_0" / "missing-family__v1-clean-baseline" / "run_02"
    assert rc == 0
    assert json.loads((task_dir / "runner_metadata.json").read_text(encoding="utf-8"))["workspace_missing"] is True
    assert json.loads((task_dir / "vllm_per_turn.json").read_text(encoding="utf-8"))["deferred"] is True
    trace_rows = [
        json.loads(line)
        for line in (task_dir / "codex_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in trace_rows] == ["task_start", "task_end"]


def test_runner_normalizes_vllm_request_metrics_jsonl(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    source = tmp_path / "vllm_requests.jsonl"
    source.write_text(
        json.dumps(
            {
                "request_id": "req-1",
                "schema": "lumo.track_b.vllm_request_metrics.v1",
                "producer": "track_b_vllm_request_metrics_patch",
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

    capture = runner._write_vllm_per_turn_from_jsonl(task_dir, source, runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    payload = json.loads((task_dir / "vllm_per_turn.json").read_text(encoding="utf-8"))
    assert payload["runtime_config_hash"] == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    metrics = payload["requests"]["req-1"]
    assert metrics["completion_tokens"] == 12
    assert metrics["prefill_sum_s"] == 0.2
    assert metrics["decode_sum_s"] == 1.0
    assert metrics["decode_tps"] == 12.0
    assert metrics["accepted_per_draft_token"] == 0.25
    raw_rows = [
        json.loads(line)
        for line in (task_dir / "vllm_request_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert raw_rows[0]["runtime_config_hash"] == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert capture == {
        "source": str(source),
        "start_offset": 0,
        "end_offset": source.stat().st_size,
        "captured_row_count": 1,
        "normalized_request_count": 1,
        "request_ids": ["req-1"],
    }


def test_runner_normalizes_only_new_vllm_request_metrics_jsonl_rows(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    source = tmp_path / "vllm_requests.jsonl"
    stale = (
        json.dumps(
            {
                "request_id": "stale",
                "schema": "lumo.track_b.vllm_request_metrics.v1",
                "producer": "track_b_vllm_request_metrics_patch",
                "prompt_tokens": 10,
                "generation_tokens": 1,
                "spec_decode_num_accepted_tokens": 0,
                "spec_decode_num_draft_tokens": 1,
            }
        )
        + "\n"
    )
    fresh = (
        json.dumps(
            {
                "request_id": "fresh",
                "schema": "lumo.track_b.vllm_request_metrics.v1",
                "producer": "track_b_vllm_request_metrics_patch",
                "prompt_tokens": 50,
                "generation_tokens": 12,
                "spec_decode_num_accepted_tokens": 3,
                "spec_decode_num_draft_tokens": 12,
            }
        )
        + "\n"
    )
    source.write_text(stale + fresh, encoding="utf-8")

    capture = runner._write_vllm_per_turn_from_jsonl(
        task_dir,
        source,
        start_offset=len(stale.encode("utf-8")),
        runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )

    payload = json.loads((task_dir / "vllm_per_turn.json").read_text(encoding="utf-8"))
    assert sorted(payload["requests"]) == ["fresh"]
    raw_rows = [
        json.loads(line)
        for line in (task_dir / "vllm_request_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["request_id"] for row in raw_rows] == ["fresh"]
    assert raw_rows[0]["runtime_config_hash"] == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert capture["start_offset"] == len(stale.encode("utf-8"))
    assert capture["end_offset"] == source.stat().st_size
    assert capture["captured_row_count"] == 1
    assert capture["normalized_request_count"] == 1
    assert capture["request_ids"] == ["fresh"]


def test_runner_rejects_incomplete_vllm_request_metrics_jsonl(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    source = tmp_path / "vllm_requests.jsonl"
    source.write_text(
        json.dumps(
            {
                "request_id": "req-1",
                "schema": "lumo.track_b.vllm_request_metrics.v1",
                "producer": "track_b_vllm_request_metrics_patch",
                "prompt_tokens": 50,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing numeric fields"):
        runner._write_vllm_per_turn_from_jsonl(task_dir, source)


def test_runner_rejects_nonfinite_vllm_request_metrics_jsonl(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    source = tmp_path / "vllm_requests.jsonl"
    source.write_text(
        json.dumps(
            {
                "request_id": "req-1",
                "schema": "lumo.track_b.vllm_request_metrics.v1",
                "producer": "track_b_vllm_request_metrics_patch",
                "prompt_tokens": 50,
                "generation_tokens": float("nan"),
                "spec_decode_num_accepted_tokens": 3,
                "spec_decode_num_draft_tokens": 12,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing numeric fields"):
        runner._write_vllm_per_turn_from_jsonl(task_dir, source)


def test_runner_rejects_vllm_request_metrics_without_producer_metadata(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    source = tmp_path / "vllm_requests.jsonl"
    source.write_text(
        json.dumps(
            {
                "request_id": "req-1",
                "prompt_tokens": 50,
                "generation_tokens": 12,
                "spec_decode_num_accepted_tokens": 3,
                "spec_decode_num_draft_tokens": 12,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="producer metadata"):
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


def test_runner_stamps_runtime_hash_into_dcgm_sampler_command(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class FakeProcess:
        pass

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        seen["command"] = command
        seen["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    process = runner._run_sampler(
        Namespace(no_dcgm=False, gpu=0, dcgm_interval_s=0.01, runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        tmp_path,
    )

    assert isinstance(process, FakeProcess)
    command = seen["command"]
    assert isinstance(command, list)
    assert command[command.index("--runtime-config-hash") + 1] == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
