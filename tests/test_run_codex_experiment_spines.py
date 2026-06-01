from __future__ import annotations

import subprocess

import pytest

from scripts import run_codex_experiment as experiment


def _completed(cmd: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")


def test_apply_config_forwards_fb_independent_spines(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    commands: list[list[str]] = []

    def fake_sh(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
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
