from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from scripts import run_codex_experiment as experiment


def _completed(cmd: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")


def test_apply_config_forwards_fb_independent_spines(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    commands: list[list[str]] = []
    timeouts: list[object] = []

    def fake_sh(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        timeouts.append(kwargs.get("timeout"))
        if cmd[:1] == ["curl"]:
            return _completed(cmd, stdout="vllm:up\n")
        if "/tmp/relaunch_qwen36_round.py" in cmd:
            return _completed(cmd, stdout="READY config=Fb mtp=2 row_mode=independent spines=3\n")
        return _completed(cmd)

    monkeypatch.setenv("LUMO_SUDO_PASSWORD", "test-password")
    monkeypatch.setattr(experiment, "sh", fake_sh)

    experiment.apply_config("Fb", mtp=2, row_mode="independent", spines=3)

    assert [
        str(experiment.REPO / ".venv/bin/python"),
        "/tmp/relaunch_qwen36_round.py",
        "--config",
        "Fb",
        "--mtp",
        "2",
        "--row-mode",
        "independent",
        "--spines",
        "3",
    ] in commands
    assert experiment.VLLM_RELAUNCH_TIMEOUT_S in timeouts
    assert "row_mode=independent spines=3" in capsys.readouterr().out


def test_apply_config_preserves_non_fb_relaunch_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    commands: list[list[str]] = []

    def fake_sh(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        if cmd[:1] == ["curl"]:
            return _completed(cmd, stdout="vllm:up\n")
        if "/tmp/relaunch_qwen36_round.py" in cmd:
            return _completed(cmd, stdout="READY config=D\n")
        return _completed(cmd)

    monkeypatch.setenv("LUMO_SUDO_PASSWORD", "test-password")
    monkeypatch.setattr(experiment, "sh", fake_sh)

    experiment.apply_config("D", mtp=5, row_mode="independent", spines=9)

    assert [
        str(experiment.REPO / ".venv/bin/python"),
        "/tmp/relaunch_qwen36_round.py",
        "--config",
        "D",
    ] in commands
    relaunch_commands = [cmd for cmd in commands if "/tmp/relaunch_qwen36_round.py" in cmd]
    assert relaunch_commands
    assert "--row-mode" not in relaunch_commands[0]
    assert "--spines" not in relaunch_commands[0]
    assert "row_mode=- spines=-" in capsys.readouterr().out


@pytest.mark.parametrize("spines", [0, 11])
def test_apply_config_rejects_invalid_fb_spines(monkeypatch: pytest.MonkeyPatch, spines: int) -> None:
    calls: list[list[str]] = []

    monkeypatch.setenv("LUMO_SUDO_PASSWORD", "test-password")
    monkeypatch.setattr(experiment, "sh", lambda cmd, **kwargs: calls.append(cmd) or _completed(cmd))

    with pytest.raises(ValueError, match=rf"--spines must be in \[1, 10\], got {spines}"):
        experiment.apply_config("Fb", mtp=2, row_mode="independent", spines=spines)

    assert calls == []


def test_launch_suite_does_not_skip_existing_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_sh(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        return _completed(cmd, stdout="launched pid=123\n")

    monkeypatch.setattr(experiment, "sh", fake_sh)

    experiment.launch_suite(
        argparse.Namespace(
            suite="swe",
            exp_tag="tag",
            subset="subset.json",
            limit=0,
            agent_wall_s=1800,
            eval_timeout_s=1800,
            concurrency=4,
            skip_existing=False,
        )
    )

    launch_command = commands[0][-1]
    assert "--skip-existing" not in launch_command


def test_launch_suite_can_explicitly_skip_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_sh(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        return _completed(cmd, stdout="launched pid=123\n")

    monkeypatch.setattr(experiment, "sh", fake_sh)

    experiment.launch_suite(
        argparse.Namespace(
            suite="swe",
            exp_tag="tag",
            subset="subset.json",
            limit=0,
            agent_wall_s=1800,
            eval_timeout_s=1800,
            concurrency=4,
            skip_existing=True,
        )
    )

    launch_command = commands[0][-1]
    assert "--skip-existing" in launch_command


def test_request_metrics_smoke_uses_x86_tunnel(monkeypatch: pytest.MonkeyPatch) -> None:
    ssh_commands: list[str] = []

    def fake_ssh(command: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        ssh_commands.append(command)
        if "Path(" in command:
            return _completed(["ssh"], stdout="200\n")
        return _completed(["ssh"], stdout='{"id":"resp-smoke"}\n')

    local_sizes = iter([100, 200])
    remote_sizes = iter([100, 200])

    monkeypatch.setattr(experiment, "ssh", fake_ssh)
    monkeypatch.setattr(experiment, "_local_file_size", lambda path: next(local_sizes))
    monkeypatch.setattr(experiment, "_remote_file_size", lambda path: next(remote_sizes))

    experiment.require_request_metrics_live()

    smoke_commands = [cmd for cmd in ssh_commands if "/v1/responses" in cmd]
    assert smoke_commands
    assert "http://127.0.0.1:8022/v1/responses" in smoke_commands[0]
    assert "Reply with exactly: OK" in smoke_commands[0]


def test_codex_protocol_marker_scanner_flags_agent_text_and_commands(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "codex_trace.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}}),
                json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "</think>\n\n<|host|>"}}),
                json.dumps({"type": "item.started", "item": {"type": "command_execution", "command": "printf '<think>'"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    hits = experiment._codex_protocol_marker_hits(task_dir)

    assert [(trace, line) for trace, line, _snippet in hits] == [
        ("codex_trace.jsonl", 2),
        ("codex_trace.jsonl", 3),
    ]


def test_stream_capture_script_does_not_truncate_remote_mirror() -> None:
    script = Path("scripts/swe_x86_helpers/stream_capture_to_alienware.sh").read_text()

    assert "touch $DST" in script
    assert ": > $DST" not in script


def test_required_speed_win_failure_writes_comparison_before_reporting_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(experiment, "REPO", tmp_path)
    local = tmp_path / "output" / "tag"
    local.mkdir(parents=True)
    current = local / "agentic_summary.json"
    baseline = tmp_path / "output" / "baseline" / "agentic_summary.json"
    baseline.parent.mkdir(parents=True)
    current.write_text(json.dumps({"steptrace": {"decode_tps": 10.0}}), encoding="utf-8")
    baseline.write_text(json.dumps({"steptrace": {"decode_tps": 20.0}}), encoding="utf-8")

    failed = experiment.finalize_speed_comparison(
        argparse.Namespace(
            speed_baseline_agentic_summary=str(baseline),
            require_speed_win=True,
        ),
        local,
        current,
    )

    payload = json.loads((local / "speed_comparison.json").read_text())
    assert failed is True
    assert payload["speedup"] == 0.5
    assert payload["speed_win"] is False
    assert payload["require_speed_win"] is True
