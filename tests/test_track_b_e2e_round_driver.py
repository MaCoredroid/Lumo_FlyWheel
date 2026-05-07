from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_track_b_e2e_round as round_driver  # noqa: E402


def _completed(command: list[str], returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout="", stderr="")


def _trace_text(
    duration_s: float,
    completion_tokens: int,
    *,
    task_id: str = "transcript-merge-regression/v1-clean-baseline",
    runtime_config_hash: str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
) -> str:
    start = datetime(2026, 5, 7, 20, 0, 0, tzinfo=UTC)
    end = start + timedelta(seconds=duration_s)
    rows = [
        {
            "event": "task_start",
            "ts": start.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "task_id": task_id,
            "runtime_config_hash": runtime_config_hash,
        },
        {"event": "turn_end", "completion_tokens": completion_tokens},
        {
            "event": "task_end",
            "ts": end.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "task_id": task_id,
        },
    ]
    return "\n".join(json.dumps(row) for row in rows) + "\n"


def _write_trace_correctness_artifact(path: Path, *, verified_at: str = "2026-05-07T00:00:00Z") -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.codex_trace_correctness.v1",
                "verified_at": verified_at,
                "trace_out_supported": True,
                "tasks": [
                    {
                        "task_id": f"task-{index}",
                        "trace_out_enabled_exit_code": 0,
                        "trace_out_disabled_exit_code": 0,
                        "model_outputs_byte_identical": True,
                        "tool_call_sequences_byte_identical": True,
                        "milestone_scores_identical": True,
                        "trace_schema_valid": True,
                    }
                    for index in range(3)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_round_driver_blocks_before_measurement_when_preflight_fails(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        preflight_out = Path(command[command.index("--out") + 1])
        preflight_out.write_text(
            json.dumps({"blocking_reasons": ["codex_trace_out_supported"]}) + "\n",
            encoding="utf-8",
        )
        return _completed(command, returncode=1)

    monkeypatch.setattr(round_driver, "_run", fake_run)

    rc = round_driver.main(
        [
            "--round",
            "0",
            "--runtime-config-hash",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--codex-command-template",
            "codex exec --trace-out {trace_out}",
            "--clock-skew-ms-p99",
            "10",
            "--trace-emitter-correctness-verified-at",
            "2026-05-07T00:00:00Z",
            "--protocol-hash-match",
            "--out-root",
            str(tmp_path),
        ]
    )

    assert rc == 1
    assert len(calls) == 1
    assert calls[0][1].endswith("preflight_track_b_e2e.py")


def test_round_driver_passes_vllm_side_channel_to_preflight(tmp_path: Path) -> None:
    side_channel = tmp_path / "vllm_request_metrics.jsonl"
    command = round_driver._preflight_command(
        argparse.Namespace(
            python=sys.executable,
            health_url="http://127.0.0.1:9950/health",
            metrics_url="http://127.0.0.1:9950/metrics",
            vllm_request_metrics_jsonl=str(side_channel),
        ),
        tmp_path / "preflight.json",
    )

    assert "--vllm-request-metrics-jsonl" in command
    assert command[command.index("--vllm-request-metrics-jsonl") + 1] == str(side_channel)


def test_round_driver_blocks_after_preflight_without_trace_correctness_artifact(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        preflight_out = Path(command[command.index("--out") + 1])
        preflight_out.write_text(json.dumps({"blocking_reasons": []}) + "\n", encoding="utf-8")
        return _completed(command)

    monkeypatch.setattr(round_driver, "_run", fake_run)

    rc = round_driver.main(
        [
            "--round",
            "0",
            "--runtime-config-hash",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--codex-command-template",
            "codex exec --trace-out {trace_out}",
            "--clock-skew-ms-p99",
            "10",
            "--trace-emitter-correctness-verified-at",
            "2026-05-07T00:00:00Z",
            "--protocol-hash-match",
            "--out-root",
            str(tmp_path),
            "--trace-correctness-artifact",
            str(tmp_path / "missing_trace_correctness.json"),
        ]
    )

    assert rc == 2
    assert len(calls) == 1
    assert calls[0][1].endswith("preflight_track_b_e2e.py")


def test_round_driver_summarizes_only_canonical_attempt_with_all_wallclocks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tasks = ["transcript-merge-regression/v1-clean-baseline", "dead-flag-reachability-audit/v1-clean-baseline"]
    monkeypatch.setattr(round_driver, "_tasks", lambda: tasks)
    commands: list[list[str]] = []
    trace_correctness_artifact = tmp_path / "trace_correctness.json"
    _write_trace_correctness_artifact(trace_correctness_artifact)

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        script = Path(command[1]).name
        if script == "preflight_track_b_e2e.py":
            preflight_out = Path(command[command.index("--out") + 1])
            preflight_out.write_text(json.dumps({"blocking_reasons": []}) + "\n", encoding="utf-8")
        if script == "run_track_b_e2e_task.py":
            round_index = int(command[command.index("--round") + 1])
            out_root = Path(command[command.index("--out-root") + 1])
            for task_id in tasks:
                family, variant = task_id.split("/", 1)
                for attempt in range(1, 5):
                    task_dir = out_root / f"round_{round_index}" / f"{family}__{variant}" / f"run_{attempt:02d}"
                    task_dir.mkdir(parents=True)
                    (task_dir / "runner_metadata.json").write_text(
                        json.dumps({"elapsed_s": 900.0 + attempt}) + "\n",
                        encoding="utf-8",
                    )
                    (task_dir / "codex_trace.jsonl").write_text(
                        _trace_text(10.0 + attempt, 100 + attempt, task_id=task_id),
                        encoding="utf-8",
                    )
        return _completed(command)

    monkeypatch.setattr(round_driver, "_run", fake_run)

    rc = round_driver.main(
        [
            "--round",
            "0",
            "--runtime-config-hash",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--codex-command-template",
            "codex exec --trace-out {trace_out}",
            "--clock-skew-ms-p99",
            "10",
            "--trace-emitter-correctness-verified-at",
            "2026-05-07T00:00:00Z",
            "--protocol-hash-match",
            "--out-root",
            str(tmp_path),
            "--trace-correctness-artifact",
            str(trace_correctness_artifact),
        ]
    )

    assert rc == 0
    task_summary_commands = [
        command for command in commands if Path(command[1]).name == "build_track_b_e2e_summary.py" and "task" in command
    ]
    runner_commands = [command for command in commands if Path(command[1]).name == "run_track_b_e2e_task.py"]
    assert len(runner_commands) == 1
    assert runner_commands[0][runner_commands[0].index("--runtime-config-hash") + 1] == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert len(task_summary_commands) == 2
    for command, task_id in zip(task_summary_commands, tasks, strict=True):
        family, variant = task_id.split("/", 1)
        assert command[command.index("--task-dir") + 1].endswith(f"{family}__{variant}/run_02")
        assert json.loads(command[command.index("--run-wallclocks-json") + 1]) == [12.0, 13.0, 14.0]
        assert "--protocol-hash-match" in command
        assert "--generation-volume-within-band" in command

    round_summary_commands = [
        command for command in commands if Path(command[1]).name == "build_track_b_e2e_summary.py" and "round" in command
    ]
    assert len(round_summary_commands) == 1


def test_round_driver_rejects_generation_volume_outlier(monkeypatch, tmp_path: Path) -> None:
    tasks = ["transcript-merge-regression/v1-clean-baseline"]
    monkeypatch.setattr(round_driver, "_tasks", lambda: tasks)
    trace_correctness_artifact = tmp_path / "trace_correctness.json"
    _write_trace_correctness_artifact(trace_correctness_artifact)

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        script = Path(command[1]).name
        if script == "preflight_track_b_e2e.py":
            preflight_out = Path(command[command.index("--out") + 1])
            preflight_out.write_text(json.dumps({"blocking_reasons": []}) + "\n", encoding="utf-8")
        if script == "run_track_b_e2e_task.py":
            out_root = Path(command[command.index("--out-root") + 1])
            family, variant = tasks[0].split("/", 1)
            for attempt, tokens in [(1, 100), (2, 100), (3, 100), (4, 1000)]:
                task_dir = out_root / "round_0" / f"{family}__{variant}" / f"run_{attempt:02d}"
                task_dir.mkdir(parents=True)
                (task_dir / "runner_metadata.json").write_text(
                    json.dumps({"elapsed_s": 100.0 + attempt}) + "\n",
                    encoding="utf-8",
                )
                (task_dir / "codex_trace.jsonl").write_text(
                    _trace_text(10.0 + attempt, tokens, task_id=tasks[0]),
                    encoding="utf-8",
                )
        return _completed(command)

    monkeypatch.setattr(round_driver, "_run", fake_run)

    rc = round_driver.main(
        [
            "--round",
            "0",
            "--runtime-config-hash",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--codex-command-template",
            "codex exec --trace-out {trace_out}",
            "--clock-skew-ms-p99",
            "10",
            "--trace-emitter-correctness-verified-at",
            "2026-05-07T00:00:00Z",
            "--protocol-hash-match",
            "--out-root",
            str(tmp_path),
            "--trace-correctness-artifact",
            str(trace_correctness_artifact),
        ]
    )

    assert rc == 2


def test_round_driver_rejects_measured_trace_runtime_hash_mismatch(monkeypatch, tmp_path: Path) -> None:
    tasks = ["transcript-merge-regression/v1-clean-baseline"]
    monkeypatch.setattr(round_driver, "_tasks", lambda: tasks)
    trace_correctness_artifact = tmp_path / "trace_correctness.json"
    _write_trace_correctness_artifact(trace_correctness_artifact)

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        script = Path(command[1]).name
        if script == "preflight_track_b_e2e.py":
            preflight_out = Path(command[command.index("--out") + 1])
            preflight_out.write_text(json.dumps({"blocking_reasons": []}) + "\n", encoding="utf-8")
        if script == "run_track_b_e2e_task.py":
            out_root = Path(command[command.index("--out-root") + 1])
            family, variant = tasks[0].split("/", 1)
            for attempt in range(1, 5):
                task_dir = out_root / "round_0" / f"{family}__{variant}" / f"run_{attempt:02d}"
                task_dir.mkdir(parents=True)
                runtime_hash = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" if attempt == 3 else "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                (task_dir / "codex_trace.jsonl").write_text(
                    _trace_text(10.0 + attempt, 100, task_id=tasks[0], runtime_config_hash=runtime_hash),
                    encoding="utf-8",
                )
        return _completed(command)

    monkeypatch.setattr(round_driver, "_run", fake_run)

    rc = round_driver.main(
        [
            "--round",
            "0",
            "--runtime-config-hash",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--codex-command-template",
            "codex exec --trace-out {trace_out}",
            "--clock-skew-ms-p99",
            "10",
            "--trace-emitter-correctness-verified-at",
            "2026-05-07T00:00:00Z",
            "--protocol-hash-match",
            "--out-root",
            str(tmp_path),
            "--trace-correctness-artifact",
            str(trace_correctness_artifact),
        ]
    )

    assert rc == 2


def test_round_driver_rejects_existing_measurement_outputs(tmp_path: Path) -> None:
    stale_trace = tmp_path / "round_0" / "task" / "run_02" / "codex_trace.jsonl"
    stale_trace.parent.mkdir(parents=True)
    stale_trace.write_text("{}\n", encoding="utf-8")

    rc = round_driver.main(
        [
            "--round",
            "0",
            "--runtime-config-hash",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--codex-command-template",
            "codex exec --trace-out {trace_out}",
            "--clock-skew-ms-p99",
            "10",
            "--trace-emitter-correctness-verified-at",
            "2026-05-07T00:00:00Z",
            "--protocol-hash-match",
            "--out-root",
            str(tmp_path),
        ]
    )

    assert rc == 2


def test_round_driver_rejects_unstamped_runtime_hash(tmp_path: Path) -> None:
    rc = round_driver.main(
        [
            "--round",
            "0",
            "--runtime-config-hash",
            "not-a-runtime-hash",
            "--codex-command-template",
            "codex exec --trace-out {trace_out}",
            "--clock-skew-ms-p99",
            "10",
            "--trace-emitter-correctness-verified-at",
            "2026-05-07T00:00:00Z",
            "--protocol-hash-match",
            "--out-root",
            str(tmp_path),
        ]
    )

    assert rc == 2
