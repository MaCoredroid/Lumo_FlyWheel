from __future__ import annotations

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


def _trace_text(duration_s: float, completion_tokens: int) -> str:
    start = datetime(2026, 5, 7, 20, 0, 0, tzinfo=UTC)
    end = start + timedelta(seconds=duration_s)
    rows = [
        {"event": "task_start", "ts": start.isoformat(timespec="milliseconds").replace("+00:00", "Z")},
        {"event": "turn_end", "completion_tokens": completion_tokens},
        {"event": "task_end", "ts": end.isoformat(timespec="milliseconds").replace("+00:00", "Z")},
    ]
    return "\n".join(json.dumps(row) for row in rows) + "\n"


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
            "sha256:test",
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


def test_round_driver_summarizes_only_canonical_attempt_with_all_wallclocks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tasks = ["transcript-merge-regression/v1-clean-baseline", "dead-flag-reachability-audit/v1-clean-baseline"]
    monkeypatch.setattr(round_driver, "_tasks", lambda: tasks)
    commands: list[list[str]] = []

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
                        _trace_text(10.0 + attempt, 100 + attempt),
                        encoding="utf-8",
                    )
        return _completed(command)

    monkeypatch.setattr(round_driver, "_run", fake_run)

    rc = round_driver.main(
        [
            "--round",
            "0",
            "--runtime-config-hash",
            "sha256:test",
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

    assert rc == 0
    task_summary_commands = [
        command for command in commands if Path(command[1]).name == "build_track_b_e2e_summary.py" and "task" in command
    ]
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
                (task_dir / "codex_trace.jsonl").write_text(_trace_text(10.0 + attempt, tokens), encoding="utf-8")
        return _completed(command)

    monkeypatch.setattr(round_driver, "_run", fake_run)

    rc = round_driver.main(
        [
            "--round",
            "0",
            "--runtime-config-hash",
            "sha256:test",
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
