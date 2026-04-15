from pathlib import Path
import subprocess

import pytest
import requests

from lumo_flywheel_serving.model_server import (
    DEFAULT_VLLM_BASE_IMAGE,
    DEFAULT_VLLM_DOCKERFILE,
    DEFAULT_VLLM_IMAGE,
    ModelServer,
)
from lumo_flywheel_serving.registry import load_registry


def test_build_run_command_includes_required_flags(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(
        registry_path=registry,
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
    )
    cmd = server._build_run_command(
        "qwen3.5-27b",
        server.registry["qwen3.5-27b"],
        enable_request_logging=True,
        kv_cache_dtype="fp8_e5m2",
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )
    command = " ".join(cmd)
    assert "--network host" in command
    assert "--gpus all" in command
    assert "--ulimit memlock=-1" in command
    assert "--ulimit stack=67108864" in command
    assert "vllm serve /models/qwen3.5-27b-fp8" in command
    assert "--chat-template /opt/lumo/chat_templates/qwen3-openai-codex.jinja" in command
    assert "docker/chat_templates/qwen3-openai-codex.jinja:/opt/lumo/chat_templates/qwen3-openai-codex.jinja:ro" in command
    assert "--enable-prefix-caching" in command
    assert "--enable-chunked-prefill" in command
    assert "--enable-log-requests" in command


def test_build_run_command_can_force_eager_mode(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(
        registry_path=registry,
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
    )
    cmd = server._build_run_command(
        "qwen3.5-27b",
        server.registry["qwen3.5-27b"],
        enable_request_logging=False,
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.65,
        enforce_eager=True,
    )
    command = " ".join(cmd)
    assert "--enforce-eager" in command


def test_fp8_checkpoints_default_initial_kv_cache_dtype_to_auto(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(registry_path=registry)

    assert server._initial_kv_cache_dtype(server.registry["qwen3.5-27b"]) == "auto"


def test_non_qwen_models_do_not_override_chat_template(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  glm-4.7:
    hf_repo: zai-org/GLM-4.7
    local_path: /models/glm-4.7
    quantization: fp8
    dtype: auto
    kv_cache_dtype: auto
    max_model_len: 65536
    gpu_memory_utilization: 0.85
    max_num_batched_tokens: 4096
    max_num_seqs: 2
"""
    )
    server = ModelServer(
        registry_path=registry,
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
    )
    cmd = server._build_run_command(
        "glm-4.7",
        server.registry["glm-4.7"],
        enable_request_logging=False,
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.85,
        enforce_eager=False,
    )

    assert "--chat-template" not in " ".join(cmd)
    assert server._chat_template_container_path(server.registry["glm-4.7"]) is None


def test_next_gpu_memory_utilization_uses_reported_free_memory() -> None:
    error_text = (
        "ValueError: Free memory on device cuda:0 (25.22/117.51 GiB) on startup "
        "is less than desired GPU memory utilization (0.9, 105.76 GiB). "
        "Decrease GPU memory utilization or reduce GPU memory used by other processes."
    )
    assert ModelServer._next_gpu_memory_utilization(error_text, current=0.9) == 0.2


def test_next_gpu_memory_utilization_keeps_legacy_steps_for_unparsed_errors() -> None:
    assert ModelServer._next_gpu_memory_utilization("Free memory on device cuda:0", current=0.9) == 0.85
    assert ModelServer._next_gpu_memory_utilization("Free memory on device cuda:0", current=0.65) == 0.6


def test_low_free_memory_prefers_grace_retries_before_degrading() -> None:
    error_text = (
        "ValueError: Free memory on device cuda:0 (25.06/117.51 GiB) on startup "
        "is less than desired GPU memory utilization (0.9, 105.76 GiB). "
        "Decrease GPU memory utilization or reduce GPU memory used by other processes."
    )
    assert ModelServer._should_retry_low_free_memory(error_text, retries=0, current=0.9) is True
    assert ModelServer._should_retry_low_free_memory(error_text, retries=1, current=0.9) is True
    assert ModelServer._should_retry_low_free_memory(error_text, retries=2, current=0.9) is False


def test_low_free_memory_skips_grace_when_even_minimum_floor_cannot_fit() -> None:
    error_text = (
        "ValueError: Free memory on device cuda:0 (7.06/117.51 GiB) on startup "
        "is less than desired GPU memory utilization (0.15, 17.63 GiB). "
        "Decrease GPU memory utilization or reduce GPU memory used by other processes."
    )
    assert ModelServer._should_retry_low_free_memory(error_text, retries=0, current=0.15) is False


def test_next_gpu_memory_utilization_stops_before_context_floor() -> None:
    assert ModelServer._next_gpu_memory_utilization("Free memory on device cuda:0", current=0.15) is None


def test_wait_vram_free_uses_grace_period_when_nvidia_smi_is_unsupported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(registry_path=registry)
    now = 0.0
    sleeps: list[float] = []

    def fake_run(cmd: list[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="[Not Supported]\n", stderr="")

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    monkeypatch.setattr(server, "_run", fake_run)
    monkeypatch.setattr(server, "_cuda_mem_get_info_gib", lambda: None)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.time", lambda: now)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.sleep", fake_sleep)

    server._wait_vram_free(timeout_s=5)

    assert sleeps == [5]


def test_wait_vram_free_retries_until_memory_probe_recovers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(registry_path=registry)
    now = 0.0
    sleeps: list[float] = []
    snapshots = iter([None, None, (110.0, 117.51)])

    def fake_run(cmd: list[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="[Not Supported]\n", stderr="")

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    monkeypatch.setattr(server, "_run", fake_run)
    monkeypatch.setattr(server, "_cuda_mem_get_info_gib", lambda: next(snapshots))
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.time", lambda: now)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.sleep", fake_sleep)

    server._wait_vram_free(timeout_s=50, required_utilization=0.9)

    assert sleeps == [20, 20]


def test_wait_vram_free_waits_for_cuda_memory_recovery_without_compute_pids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(registry_path=registry)
    sleeps: list[float] = []
    snapshots = iter([(8.0, 117.51), (110.0, 117.51)])

    def fake_run(cmd: list[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(server, "_run", fake_run)
    monkeypatch.setattr(server, "_cuda_mem_get_info_gib", lambda: next(snapshots))
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.sleep", lambda seconds: sleeps.append(seconds))

    server._wait_vram_free(timeout_s=30, required_utilization=0.9)

    assert sleeps == [5]


def test_wait_vram_free_fails_fast_when_host_never_reaches_minimum_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(registry_path=registry)
    now = 0.0

    def fake_run(cmd: list[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="[Not Supported]\n", stderr="")

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    monkeypatch.setattr(server, "_run", fake_run)
    monkeypatch.setattr(server, "_cuda_mem_get_info_gib", lambda: (10.89, 117.51))
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.sleep", fake_sleep)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.time", lambda: now)

    with pytest.raises(RuntimeError, match="Timed out waiting for minimum free VRAM"):
        server._wait_vram_free(timeout_s=40, required_utilization=0.9)


def test_start_resets_previous_log_before_retry_cycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(
        registry_path=registry,
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
    )
    log_path = server.logs_path("qwen3.5-27b")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("stale log text", encoding="utf-8")

    monkeypatch.setattr(server, "_ensure_image_present", lambda: None)
    monkeypatch.setattr(server, "stop", lambda missing_ok=False: None)
    monkeypatch.setattr(server, "_wait_vram_free", lambda timeout_s=120, required_utilization=None: None)
    monkeypatch.setattr(
        server,
        "_run",
        lambda cmd, capture_output=False, check=True: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )

    def fake_wait_ready(model_id: str, timeout_s: int = 900) -> None:
        assert log_path.read_text(encoding="utf-8") == ""

    monkeypatch.setattr(server, "_wait_ready", fake_wait_ready)

    server.start("qwen3.5-27b")


def test_start_waits_for_vram_release_before_first_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(
        registry_path=registry,
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
    )
    events: list[str] = []

    monkeypatch.setattr(server, "_ensure_image_present", lambda: None)
    monkeypatch.setattr(server, "_reset_log", lambda model_id: events.append(f"reset:{model_id}"))
    monkeypatch.setattr(server, "stop", lambda missing_ok=False: events.append(f"stop:{missing_ok}"))
    monkeypatch.setattr(
        server,
        "_wait_vram_free",
        lambda timeout_s=120, required_utilization=None: events.append(
            f"wait:{timeout_s}:{required_utilization}"
        ),
    )
    monkeypatch.setattr(
        server,
        "_run",
        lambda cmd, capture_output=False, check=True: events.append("run")
        or subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(server, "_wait_ready", lambda model_id, timeout_s=900: events.append(f"ready:{model_id}"))

    server.start("qwen3.5-27b")

    assert events[:4] == ["reset:qwen3.5-27b", "stop:True", "wait:120:0.9", "run"]


def test_start_uses_valid_initial_kv_cache_dtype_for_fp8_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(
        registry_path=registry,
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
    )
    launched_kv_dtypes: list[str] = []

    monkeypatch.setattr(server, "_ensure_image_present", lambda: None)
    monkeypatch.setattr(server, "_reset_log", lambda model_id: None)
    monkeypatch.setattr(server, "stop", lambda missing_ok=False: None)
    monkeypatch.setattr(server, "_wait_vram_free", lambda timeout_s=120, required_utilization=None: None)
    monkeypatch.setattr(
        server,
        "_build_run_command",
        lambda model_id, config, enable_request_logging, kv_cache_dtype, gpu_memory_utilization, enforce_eager: (
            launched_kv_dtypes.append(kv_cache_dtype) or ["docker", "run"]
        ),
    )
    monkeypatch.setattr(
        server,
        "_run",
        lambda cmd, capture_output=False, check=True: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(server, "_wait_ready", lambda model_id, timeout_s=900: None)

    server.start("qwen3.5-27b")

    assert launched_kv_dtypes == ["auto"]


def test_start_waits_against_lowered_gpu_memory_target_before_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(
        registry_path=registry,
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
    )
    waits: list[float | None] = []
    launch_targets: list[float] = []
    ready_attempts = 0

    monkeypatch.setattr(server, "_ensure_image_present", lambda: None)
    monkeypatch.setattr(server, "_reset_log", lambda model_id: None)
    monkeypatch.setattr(server, "stop", lambda missing_ok=False: None)
    monkeypatch.setattr(
        server,
        "_wait_vram_free",
        lambda timeout_s=120, required_utilization=None: waits.append(required_utilization),
    )
    monkeypatch.setattr(
        server,
        "_build_run_command",
        lambda model_id, config, enable_request_logging, kv_cache_dtype, gpu_memory_utilization, enforce_eager: (
            launch_targets.append(gpu_memory_utilization) or ["docker", "run"]
        ),
    )
    monkeypatch.setattr(
        server,
        "_run",
        lambda cmd, capture_output=False, check=True: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )

    def fake_wait_ready(model_id: str, timeout_s: int = 900) -> None:
        nonlocal ready_attempts
        ready_attempts += 1
        if ready_attempts == 1:
            raise RuntimeError(
                "ValueError: Free memory on device cuda:0 (105.2/117.51 GiB) on startup "
                "is less than desired GPU memory utilization (0.9, 105.76 GiB). "
                "Decrease GPU memory utilization or reduce GPU memory used by other processes."
            )

    monkeypatch.setattr(server, "_wait_ready", fake_wait_ready)

    server.start("qwen3.5-27b")

    assert waits == [0.9, 0.88]
    assert launch_targets == [0.9, 0.88]


def test_reset_log_uses_docker_fallback_for_root_owned_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(
        registry_path=registry,
        image="lumo-flywheel-vllm:test",
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
    )
    log_path = server.logs_path("qwen3.5-27b")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    def fake_write_text(*args: object, **kwargs: object) -> str:
        raise PermissionError("root owned")

    def fake_subprocess_run(
        cmd: list[str], check: bool, capture_output: bool, text: bool, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(Path, "write_text", fake_write_text)
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    server._reset_log("qwen3.5-27b")

    assert captured["cmd"] == [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{server.logs_root}:{server.logs_root}",
        "--entrypoint",
        "python3",
        "lumo-flywheel-vllm:test",
        "-c",
        "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('', encoding='utf-8')",
        str(log_path),
    ]


def test_models_uses_auth_header(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(registry_path=registry)
    captured: dict[str, object] = {}

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, list[dict[str, str]]]:
            return {"data": [{"id": "qwen3.5-27b"}]}

    def fake_get(url: str, timeout: int, headers: dict[str, str]) -> Response:
        captured["url"] = url
        captured["headers"] = headers
        return Response()

    monkeypatch.setenv("VLLM_API_KEY", "test-token")
    monkeypatch.setattr(requests, "get", fake_get)

    response = server.models()

    assert response.json()["data"][0]["id"] == "qwen3.5-27b"
    assert captured["url"] == "http://127.0.0.1:8000/v1/models"
    assert captured["headers"] == {"Authorization": "Bearer test-token"}


def test_wait_ready_requires_target_model_in_v1_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(
        registry_path=registry,
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
    )
    now = 0.0
    sleeps: list[float] = []
    log_lines: list[str] = []
    responses = iter(
        [
            {"health": 200, "models": {"data": [{"id": "bootstrap"}]}},
            {"health": 200, "models": {"data": [{"id": "qwen3.5-27b"}]}},
        ]
    )

    class Response:
        def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    current = {"health": 200, "models": {"data": [{"id": "bootstrap"}]}}

    def fake_run(cmd: list[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="running\n", stderr="")

    def fake_get(url: str, headers: dict[str, str], timeout: int) -> Response:
        nonlocal current
        if url.endswith("/health"):
            current = next(responses)
            return Response(status_code=current["health"])
        if url.endswith("/v1/models"):
            return Response(payload=current["models"])
        raise AssertionError(url)

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    monkeypatch.setattr(server, "_run", fake_run)
    monkeypatch.setattr(server, "_append_log_text", lambda model_id, text: log_lines.append(text))
    monkeypatch.setattr("lumo_flywheel_serving.model_server.requests.get", fake_get)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.sleep", fake_sleep)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.time", lambda: now)

    server._wait_ready("qwen3.5-27b", timeout_s=15)

    assert sleeps == [5]
    assert "served_models=qwen3.5-27b" in log_lines[0]


def test_wait_ready_times_out_when_target_model_never_appears(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_num_batched_tokens: 8192
    max_num_seqs: 4
"""
    )
    server = ModelServer(registry_path=registry)
    now = 0.0

    class Response:
        def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    def fake_run(cmd: list[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="running\n", stderr="")

    def fake_get(url: str, headers: dict[str, str], timeout: int) -> Response:
        if url.endswith("/health"):
            return Response(status_code=200)
        if url.endswith("/v1/models"):
            return Response(payload={"data": [{"id": "wrong-model"}]})
        raise AssertionError(url)

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    monkeypatch.setattr(server, "_run", fake_run)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.requests.get", fake_get)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.sleep", fake_sleep)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.time", lambda: now)

    with pytest.raises(TimeoutError, match="not ready within 10s"):
        server._wait_ready("qwen3.5-27b", timeout_s=10)


def test_defaults_pin_repo_owned_nvidia_image() -> None:
    assert DEFAULT_VLLM_BASE_IMAGE == "nvcr.io/nvidia/pytorch:26.01-py3"
    assert DEFAULT_VLLM_IMAGE == "lumo-flywheel-vllm:26.01-py3-v0.19.0"
    assert DEFAULT_VLLM_DOCKERFILE.name == "Dockerfile.nvidia-vllm"


def test_registry_rejects_unresolved_placeholder_entries(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  comparison-slot:
    hf_repo: ""
    local_path: ""
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
"""
    )
    with pytest.raises(ValueError, match="missing local_path"):
        load_registry(registry)
