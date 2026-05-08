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
    assert command[command.index("--reset-prefix-cache-url") + 1] == "http://127.0.0.1:9950/reset_prefix_cache"
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


def test_ncu_profile_driver_normalizes_relative_output_roots(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ncu_profiles.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        profile_path = Path(command[command.index("--log-file") + 1])
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(_metric_csv(), encoding="utf-8")
        return _completed(command)

    monkeypatch.setattr(ncu_profiles, "_run", fake_run)

    rc = ncu_profiles.main(
        [
            "--archetype",
            "tool-call-frame",
            "--out-root",
            "profiles",
            "--task-out-root",
            "task-runs",
            "--codex-command-template",
            "codex exec --trace-out {trace_out}",
            "--runtime-config-hash",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ]
    )

    assert rc == 0
    command = commands[0]
    assert command[command.index("--log-file") + 1] == str(tmp_path / "profiles" / "ncu_tool-call-frame.csv")
    assert command[command.index("--out-root") + 1] == str(tmp_path / "task-runs")


def test_ncu_profile_driver_runs_server_launch_target(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ncu_profiles.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    server_commands: list[list[str]] = []
    task_commands: list[list[str]] = []
    ready_urls: list[str] = []

    class FakeServer:
        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def wait(self, timeout: float) -> int:
            self.returncode = -15 if self.returncode is None else self.returncode
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    def fake_popen(
        command: list[str],
        *,
        stdout_path: Path,
        stderr_path: Path,
    ) -> FakeServer:
        server_commands.append(command)
        profile_path = Path(command[command.index("--log-file") + 1])
        profile_path.write_text(_metric_csv(), encoding="utf-8")
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return FakeServer()

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        task_commands.append(command)
        return _completed(command)

    def fake_wait(url: str, process: FakeServer, *, stderr_path: Path, timeout_s: float) -> None:
        assert process is not None
        assert stderr_path == tmp_path / "ncu_long-text_server_stderr.log"
        ready_urls.append(url)

    monkeypatch.setattr(ncu_profiles, "_popen", fake_popen)
    monkeypatch.setattr(ncu_profiles, "_run", fake_run)
    monkeypatch.setattr(ncu_profiles, "_wait_for_server_ready", fake_wait)

    rc = ncu_profiles.main(
        [
            "--profile-target",
            "server-launch",
            "--archetype",
            "long-text",
            "--out-root",
            str(tmp_path),
            "--task-out-root",
            str(tmp_path / "ncu_task_runs"),
            "--server-launch-command",
            "python serve.py --model {model}",
            "--server-ready-url",
            "http://127.0.0.1:9951/health",
            "--health-url",
            "http://127.0.0.1:9951/health",
            "--metrics-url",
            "http://127.0.0.1:9951/metrics",
            "--reset-prefix-cache-url",
            "http://127.0.0.1:9951/reset_prefix_cache",
            "--endpoint",
            "http://127.0.0.1:8023/v1",
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
    assert ready_urls == ["http://127.0.0.1:9951/health"]
    server_command = server_commands[0]
    assert server_command[0] == "ncu"
    assert server_command[-3:] == ["bash", "-lc", "python serve.py --model qwen3.5-27b"]
    task_command = task_commands[0]
    assert task_command[0].endswith("python")
    assert "--ncu-mode" in task_command
    assert task_command[task_command.index("--health-url") + 1] == "http://127.0.0.1:9951/health"
    assert task_command[task_command.index("--metrics-url") + 1] == "http://127.0.0.1:9951/metrics"
    assert task_command[task_command.index("--reset-prefix-cache-url") + 1] == "http://127.0.0.1:9951/reset_prefix_cache"
    assert task_command[task_command.index("--endpoint") + 1] == "http://127.0.0.1:8023/v1"
    metadata = json.loads((tmp_path / "ncu_long-text.json").read_text(encoding="utf-8"))
    assert metadata["profile_target"] == "server-launch"
    assert metadata["server_ready_url"] == "http://127.0.0.1:9951/health"


def test_ncu_profile_driver_reports_server_launch_exit_before_ready(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ncu_profiles.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    task_commands: list[list[str]] = []

    class ExitedServer:
        returncode = 127

        def poll(self) -> int:
            return self.returncode

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            return self.returncode

    def fake_popen(
        command: list[str],
        *,
        stdout_path: Path,
        stderr_path: Path,
    ) -> ExitedServer:
        stderr_path.write_text("/usr/bin/bash: line 1: vllm: command not found\n", encoding="utf-8")
        return ExitedServer()

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        task_commands.append(command)
        return _completed(command)

    monkeypatch.setattr(ncu_profiles, "_popen", fake_popen)
    monkeypatch.setattr(ncu_profiles, "_run", fake_run)

    rc = ncu_profiles.main(
        [
            "--profile-target",
            "server-launch",
            "--archetype",
            "long-text",
            "--out-root",
            str(tmp_path),
            "--task-out-root",
            str(tmp_path / "ncu_task_runs"),
            "--server-launch-command",
            "vllm serve",
            "--server-ready-url",
            "http://127.0.0.1:9951/health",
            "--server-ready-timeout-s",
            "30",
            "--codex-command-template",
            "codex exec --trace-out {trace_out}",
            "--runtime-config-hash",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert not task_commands
    assert "server exited before ready" in captured.err
    assert "vllm: command not found" in captured.err


def test_ncu_profile_driver_runs_container_server_launch_target(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ncu_profiles.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    server_commands: list[list[str]] = []
    run_commands: list[list[str]] = []
    ready_urls: list[str] = []

    class FakeServer:
        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def wait(self, timeout: float) -> int:
            self.returncode = -15 if self.returncode is None else self.returncode
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    def fake_popen(
        command: list[str],
        *,
        stdout_path: Path,
        stderr_path: Path,
    ) -> FakeServer:
        server_commands.append(command)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return FakeServer()

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        run_commands.append(command)
        if command[:2] == ["docker", "cp"]:
            Path(command[3]).write_text(_metric_csv(), encoding="utf-8")
        return _completed(command)

    def fake_wait(url: str, process: FakeServer, *, stderr_path: Path, timeout_s: float) -> None:
        assert process is not None
        ready_urls.append(url)

    monkeypatch.setattr(ncu_profiles, "_popen", fake_popen)
    monkeypatch.setattr(ncu_profiles, "_run", fake_run)
    monkeypatch.setattr(ncu_profiles, "_wait_for_server_ready", fake_wait)

    rc = ncu_profiles.main(
        [
            "--profile-target",
            "container-server-launch",
            "--archetype",
            "long-text",
            "--out-root",
            str(tmp_path),
            "--task-out-root",
            str(tmp_path / "ncu_task_runs"),
            "--container-name",
            "lumo-vllm-l0c-fp8-cutlass-run30",
            "--container-profile-csv",
            "/tmp/track_b_ncu_{archetype}.csv",
            "--container-server-stop-command",
            "pkill -f 'vllm serve .*--port 9951' || true",
            "--server-launch-command",
            "vllm serve --served-model-name {model} --tag {archetype}",
            "--server-ready-url",
            "http://127.0.0.1:9951/health",
            "--health-url",
            "http://127.0.0.1:9951/health",
            "--metrics-url",
            "http://127.0.0.1:9951/metrics",
            "--reset-prefix-cache-url",
            "http://127.0.0.1:9951/reset_prefix_cache",
            "--endpoint",
            "http://127.0.0.1:8023/v1",
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
    assert ready_urls == ["http://127.0.0.1:9951/health"]
    server_command = server_commands[0]
    assert server_command[:4] == ["docker", "exec", "lumo-vllm-l0c-fp8-cutlass-run30", "bash"]
    shell = server_command[-1]
    assert "--log-file /tmp/track_b_ncu_long-text.csv" in shell
    assert "vllm serve --served-model-name qwen3.5-27b --tag long-text" in shell
    task_command = next(command for command in run_commands if command[0].endswith("python"))
    assert task_command[task_command.index("--endpoint") + 1] == "http://127.0.0.1:8023/v1"
    assert ["docker", "exec", "lumo-vllm-l0c-fp8-cutlass-run30", "bash", "-lc", "pkill -f 'vllm serve .*--port 9951' || true"] in run_commands
    assert [
        "docker",
        "cp",
        "lumo-vllm-l0c-fp8-cutlass-run30:/tmp/track_b_ncu_long-text.csv",
        str(tmp_path / "ncu_long-text.csv"),
    ] in run_commands
    metadata = json.loads((tmp_path / "ncu_long-text.json").read_text(encoding="utf-8"))
    assert metadata["profile_target"] == "container-server-launch"
    assert metadata["container_name"] == "lumo-vllm-l0c-fp8-cutlass-run30"
    assert metadata["container_profile_csv"] == "/tmp/track_b_ncu_long-text.csv"


def test_ncu_profile_driver_requires_container_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ncu_profiles.shutil, "which", lambda binary: f"/usr/bin/{binary}")

    rc = ncu_profiles.main(
        [
            "--profile-target",
            "container-server-launch",
            "--archetype",
            "long-text",
            "--out-root",
            str(tmp_path),
            "--task-out-root",
            str(tmp_path / "ncu_task_runs"),
            "--server-launch-command",
            "vllm serve",
            "--server-ready-url",
            "http://127.0.0.1:9951/health",
            "--codex-command-template",
            "codex exec --trace-out {trace_out}",
            "--runtime-config-hash",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ]
    )

    assert rc == 2


def test_server_launch_command_formatter_preserves_json_braces() -> None:
    args = ncu_profiles.argparse.Namespace(
        model="qwen3.5-27b",
        runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )

    rendered = ncu_profiles._format_server_launch_command(
        (
            "vllm serve --served-model-name {model} "
            "--default-chat-template-kwargs '{\"enable_thinking\": false}' "
            "--tag {archetype} --hash {runtime_config_hash}"
        ),
        args,
        "long-text",
    )

    assert "--served-model-name qwen3.5-27b" in rendered
    assert "--tag long-text" in rendered
    assert "--hash sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in rendered
    assert "'{\"enable_thinking\": false}'" in rendered


def test_ncu_profile_driver_requires_server_launch_command(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ncu_profiles.shutil, "which", lambda binary: f"/usr/bin/{binary}")

    rc = ncu_profiles.main(
        [
            "--profile-target",
            "server-launch",
            "--archetype",
            "long-text",
            "--out-root",
            str(tmp_path),
            "--task-out-root",
            str(tmp_path / "ncu_task_runs"),
            "--server-ready-url",
            "http://127.0.0.1:9951/health",
            "--codex-command-template",
            "codex exec --trace-out {trace_out}",
            "--runtime-config-hash",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ]
    )

    assert rc == 2


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


def test_ncu_profile_driver_rejects_application_error(tmp_path: Path) -> None:
    profile = tmp_path / "ncu_long-text.csv"
    profile.write_text("==ERROR== The application returned an error code (1).\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="application error"):
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
