from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_track_b_e2e_ncu_profiles as ncu_profiles  # noqa: E402


def _completed(command: list[str], returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout="", stderr="")


def _metric_csv() -> str:
    return "Metric Name,Metric Value\n" + "\n".join(f"{metric},1" for metric in ncu_profiles.NCU_REQUIRED_METRICS) + "\n"


def test_ncu_profile_driver_builds_named_archetype_command(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ncu_profiles.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        profile_path = Path(command[command.index("--log-file") + 1])
        profile_path.write_text(_metric_csv(), encoding="utf-8")
        return _completed(command)

    monkeypatch.setattr(ncu_profiles, "_run", fake_run)

    rc = ncu_profiles.main(
        [
            "--archetype",
            "tool-call-frame",
            "--out-root",
            str(tmp_path),
            "--task-out-root",
            str(tmp_path / "ncu_task_runs"),
            "--codex-command-template",
            "codex exec --trace-out {trace_out}",
            "--runtime-config-hash",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ]
    )

    assert rc == 0
    assert len(commands) == 1
    command = commands[0]
    assert command[0] == "ncu"
    assert command[command.index("--log-file") + 1] == str(tmp_path / "ncu_tool-call-frame.csv")
    assert command[command.index("--out-root") + 1] == str(tmp_path / "ncu_task_runs")
    assert command[command.index("--runtime-config-hash") + 1] == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert "policy-aware-request-resolution/v1-clean-baseline" in command
    assert "--no-dcgm" in command
    assert "--ncu-mode" in command
    assert command[command.index("--metrics") + 1] == ",".join(ncu_profiles.NCU_REQUIRED_METRICS)
    metadata = json.loads((tmp_path / "ncu_tool-call-frame.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == "lumo.track_b.ncu_archetype_profile.v1"
    assert metadata["runtime_config_hash"] == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert metadata["archetype"] == "tool-call-frame"
    assert metadata["task_id"] == "policy-aware-request-resolution/v1-clean-baseline"


def test_ncu_profile_driver_propagates_deferred_instrumentation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ncu_profiles.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        profile_path = Path(command[command.index("--log-file") + 1])
        profile_path.write_text(_metric_csv(), encoding="utf-8")
        return _completed(command)

    monkeypatch.setattr(ncu_profiles, "_run", fake_run)

    rc = ncu_profiles.main(
        [
            "--archetype",
            "tool-call-frame",
            "--out-root",
            str(tmp_path),
            "--task-out-root",
            str(tmp_path / "ncu_task_runs"),
            "--codex-command-template",
            "codex exec --json",
            "--runtime-config-hash",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--defer-codex-trace-out",
            "--defer-vllm-request-metrics-join",
            "--defer-dcgm-profile-fields",
        ]
    )

    assert rc == 0
    command = commands[0]
    assert "--defer-codex-trace-out" in command
    assert "--defer-vllm-request-metrics-join" in command
    assert "--defer-dcgm-profile-fields" in command
    metadata = json.loads((tmp_path / "ncu_tool-call-frame.json").read_text(encoding="utf-8"))
    assert metadata["deferred_instrumentation_checks"] == [
        "codex_trace_out_supported",
        "dcgm_profile_fields_available",
        "vllm_request_metrics_join_available",
    ]


def test_ncu_profile_driver_rejects_missing_metric(tmp_path: Path) -> None:
    profile = tmp_path / "ncu_long-text.csv"
    profile.write_text("Metric Name,Metric Value\ngpu__time_duration.sum,1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing required metrics"):
        ncu_profiles._validate_profile(profile)


def test_ncu_profile_driver_rejects_no_kernel_warning(tmp_path: Path) -> None:
    profile = tmp_path / "ncu_long-text.csv"
    profile.write_text("==WARNING== No kernels were profiled.\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no profiled kernels"):
        ncu_profiles._validate_profile(profile)


def test_ncu_profile_driver_rejects_nonfinite_metric_value(tmp_path: Path) -> None:
    profile = tmp_path / "ncu_long-text.csv"
    text = _metric_csv().replace("gpu__time_duration.sum,1", "gpu__time_duration.sum,NaN")
    profile.write_text(text, encoding="utf-8")

    with pytest.raises(RuntimeError, match="no finite values"):
        ncu_profiles._validate_profile(profile)


def test_ncu_profile_driver_rejects_unstamped_runtime_hash(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ncu_profiles.shutil, "which", lambda binary: f"/usr/bin/{binary}")

    rc = ncu_profiles.main(
        [
            "--archetype",
            "tool-call-frame",
            "--out-root",
            str(tmp_path),
            "--task-out-root",
            str(tmp_path / "ncu_task_runs"),
            "--codex-command-template",
            "codex exec --trace-out {trace_out}",
            "--runtime-config-hash",
            "not-a-runtime-hash",
        ]
    )

    assert rc == 2
