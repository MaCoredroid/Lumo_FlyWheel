import os
from pathlib import Path
import subprocess

import pytest
import requests

from lumo_flywheel_serving.model_server import (
    DEFAULT_INFERENCE_PROXY_PORT,
    DEFAULT_VLLM_BASE_IMAGE,
    DEFAULT_VLLM_DOCKERFILE,
    DEFAULT_VLLM_IMAGE,
    MIN_GPU_MEMORY_UTILIZATION,
    ModelServer,
)
from lumo_flywheel_serving.registry import ModelConfig, load_registry


VALID_METRICS_TEXT = "\n".join(
    [
        "vllm:prompt_tokens_total 10",
        "vllm:generation_tokens_total 4",
        "vllm:prefix_cache_queries_total 3",
        "vllm:prefix_cache_hits_total 1",
        "vllm:request_prefill_kv_computed_tokens_sum 11",
        "vllm:time_to_first_token_seconds_sum 1.8",
        "vllm:time_to_first_token_seconds_count 2",
        "vllm:request_prefill_time_seconds_sum 2.5",
        "vllm:request_decode_time_seconds_sum 1.2",
    ]
)


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
    assert "--enable-auto-tool-choice" in command
    assert "--tool-call-parser qwen3_xml" in command
    assert "--reasoning-parser qwen3" in command
    assert "--enable-prefix-caching" in command
    assert "--enable-chunked-prefill" in command
    assert "--enable-log-requests" in command
    assert "VLLM_ENABLE_RESPONSES_API_STORE=1" in command


def test_build_run_command_includes_lora_adapter_flags(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    served_model_name: qwen3.5-27b
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.88
    max_num_batched_tokens: 8192
    max_num_seqs: 4
    max_lora_rank: 32
    lora_modules:
      codex-sft-all: /models/adapters/codex-sft-all
      codex-dapo: /models/adapters/codex-dapo
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
        gpu_memory_utilization=0.88,
        enforce_eager=False,
    )
    command = " ".join(cmd)

    assert "--served-model-name qwen3.5-27b" in command
    assert "--enable-lora" in command
    assert "--max-lora-rank 32" in command
    assert "--lora-modules codex-sft-all=/models/adapters/codex-sft-all codex-dapo=/models/adapters/codex-dapo" in command


def test_build_run_command_rejects_lora_paths_outside_models_mount(tmp_path: Path) -> None:
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
    gpu_memory_utilization: 0.88
    max_num_batched_tokens: 8192
    max_num_seqs: 4
    max_lora_rank: 32
    lora_modules:
      codex-sft-all: /tmp/adapters/codex-sft-all
"""
    )
    with pytest.raises(ValueError, match="under /models"):
        ModelServer(
            registry_path=registry,
            logs_root=tmp_path / "logs",
            triton_cache_root=tmp_path / "triton",
        )


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


def test_proxy_base_url_uses_default_inference_proxy_port(tmp_path: Path) -> None:
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

    assert server.proxy_base_url() == f"http://127.0.0.1:{DEFAULT_INFERENCE_PROXY_PORT}"


def test_record_launch_metadata_appends_sorted_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    lines: list[str] = []
    monkeypatch.setattr(server, "_append_log_text", lambda model_id, text: lines.append(f"{model_id}:{text}"))

    server.record_launch_metadata(
        "qwen3.5-27b",
        metric_schema_variant="openmetrics_total",
        gate1_responses_status="pass",
        sleep_mode=False,
    )

    assert lines == [
        "qwen3.5-27b:[VLLM-META] gate1_responses_status=pass metric_schema_variant=openmetrics_total sleep_mode=false\n"
    ]


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


def test_next_gpu_memory_utilization_for_kv_cache_startup_raises_floor() -> None:
    assert ModelServer._next_gpu_memory_utilization_for_kv_cache_startup(current=0.08) == 0.30
    assert ModelServer._next_gpu_memory_utilization_for_kv_cache_startup(current=0.30) == 0.35
    assert ModelServer._next_gpu_memory_utilization_for_kv_cache_startup(current=0.95) is None


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


def test_wait_vram_free_rejects_persistent_busy_compute_pids_without_memory_probe(
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
        return subprocess.CompletedProcess(cmd, 0, stdout="1234\n", stderr="")

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    monkeypatch.setattr(server, "_run", fake_run)
    monkeypatch.setattr(server, "_cuda_mem_get_info_gib", lambda: None)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.sleep", fake_sleep)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.time", lambda: now)

    with pytest.raises(RuntimeError, match="active GPU compute processes"):
        server._wait_vram_free(timeout_s=5, required_utilization=0.9)


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
    monkeypatch.setattr(server, "_recover_host_memory", lambda: None)
    monkeypatch.setattr(server, "_wait_vram_free", lambda timeout_s=120, required_utilization=None: None)
    monkeypatch.setattr(
        server,
        "_run",
        lambda cmd, capture_output=False, check=True: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(server, "_start_proxy", lambda: None)

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
    monkeypatch.setattr(server, "_recover_host_memory", lambda: events.append("recover"))
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
    monkeypatch.setattr(server, "_start_proxy", lambda: events.append("proxy"))

    server.start("qwen3.5-27b")

    assert events[:5] == [
        "reset:qwen3.5-27b",
        "stop:True",
        "recover",
        f"wait:120:{MIN_GPU_MEMORY_UTILIZATION}",
        "run",
    ]


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
    monkeypatch.setattr(server, "_recover_host_memory", lambda: None)
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
    monkeypatch.setattr(server, "_start_proxy", lambda: None)

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
    monkeypatch.setattr(server, "_recover_host_memory", lambda: None)
    monkeypatch.setattr(
        server,
        "_wait_vram_free",
        lambda timeout_s=120, required_utilization=None: waits.append(required_utilization),
    )
    monkeypatch.setattr(
        server,
        "_build_run_command",
        lambda model_id, config, enable_request_logging, kv_cache_dtype, gpu_memory_utilization, enforce_eager, **kwargs: (
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
    monkeypatch.setattr(server, "_start_proxy", lambda: None)

    server.start("qwen3.5-27b")

    assert waits == [MIN_GPU_MEMORY_UTILIZATION, 0.88]
    assert launch_targets == [0.9, 0.88]


def test_start_raises_gpu_memory_target_after_kv_cache_startup_failure(
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
    max_model_len: 32768
    gpu_memory_utilization: 0.08
    max_num_batched_tokens: 6144
    max_num_seqs: 1
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
    monkeypatch.setattr(server, "_recover_host_memory", lambda: None)
    monkeypatch.setattr(
        server,
        "_wait_vram_free",
        lambda timeout_s=120, required_utilization=None: waits.append(required_utilization),
    )
    monkeypatch.setattr(
        server,
        "_build_run_command",
        lambda model_id, config, enable_request_logging, kv_cache_dtype, gpu_memory_utilization, enforce_eager, **kwargs: (
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
                "Available KV cache memory: -27.97 GiB\n"
                "ValueError: No available memory for the cache blocks. "
                "Try increasing `gpu_memory_utilization` when initializing the engine."
            )

    monkeypatch.setattr(server, "_wait_ready", fake_wait_ready)
    monkeypatch.setattr(server, "_start_proxy", lambda: None)

    server.start("qwen3.5-27b")

    assert waits == [MIN_GPU_MEMORY_UTILIZATION, 0.30]
    assert launch_targets == [0.08, 0.30]


def test_switch_model_restarts_when_sleep_mode_state_is_stale(
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
    server = ModelServer(registry_path=registry, use_sleep_mode=True)
    server.current_model = "qwen3.5-27b"
    events: list[str] = []

    monkeypatch.setattr(server, "_is_serving_model", lambda model_id: False)
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
        "start",
        lambda model_id, enable_request_logging=False: events.append(
            f"start:{model_id}:{enable_request_logging}"
        ),
    )

    server.switch_model("qwen3.5-27b", enable_request_logging=True)

    assert events == ["stop:True", f"wait:120:{MIN_GPU_MEMORY_UTILIZATION}", "start:qwen3.5-27b:True"]


def test_switch_model_keeps_sleep_mode_fast_path_when_model_is_healthy(
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
    server = ModelServer(registry_path=registry, use_sleep_mode=True)
    server.current_model = "qwen3.5-27b"
    events: list[str] = []

    monkeypatch.setattr(server, "_is_serving_model", lambda model_id: True)
    monkeypatch.setattr(server, "stop", lambda missing_ok=False: events.append(f"stop:{missing_ok}"))
    monkeypatch.setattr(
        server,
        "start",
        lambda model_id, enable_request_logging=False: events.append(
            f"start:{model_id}:{enable_request_logging}"
        ),
    )

    server.switch_model("qwen3.5-27b", enable_request_logging=True)

    assert events == []


def test_stop_runs_host_memory_recovery_when_container_missing(
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
    events: list[str] = []

    monkeypatch.setattr(
        server,
        "_run",
        lambda cmd, capture_output=True, check=True: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(server, "_recover_host_memory", lambda: events.append("recover"))

    server.stop(missing_ok=True)

    assert events == ["recover"]


def test_recover_host_memory_uses_configured_sudo_password(
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
    seen: list[list[str]] = []

    monkeypatch.setenv("LUMO_SUDO_PASSWORD", "secret")

    def fake_subprocess_run(
        cmd: list[str], capture_output: bool, text: bool, check: bool, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("lumo_flywheel_serving.model_server.subprocess.run", fake_subprocess_run)

    server._recover_host_memory()

    assert seen
    assert seen[0][:2] == ["bash", "-lc"]
    assert "sudo -S bash -lc" in seen[0][2]


def test_recover_host_memory_ignores_missing_sudo_credentials(
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

    monkeypatch.delenv("LUMO_SUDO_PASSWORD", raising=False)
    monkeypatch.setattr(
        "lumo_flywheel_serving.model_server.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, stdout="", stderr="sudo: a password is required\n"
        ),
    )

    server._recover_host_memory()


def test_model_server_loads_local_runtime_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    (tmp_path / ".lumo.local.env").write_text("export LUMO_SUDO_PASSWORD='from-file'\n", encoding="utf-8")
    monkeypatch.delenv("LUMO_SUDO_PASSWORD", raising=False)

    ModelServer(registry_path=registry)

    assert os.environ["LUMO_SUDO_PASSWORD"] == "from-file"


def test_model_server_does_not_override_existing_runtime_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    (tmp_path / ".lumo.local.env").write_text("export LUMO_SUDO_PASSWORD='from-file'\n", encoding="utf-8")
    monkeypatch.setenv("LUMO_SUDO_PASSWORD", "from-env")

    ModelServer(registry_path=registry)

    assert os.environ["LUMO_SUDO_PASSWORD"] == "from-env"


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


def test_build_run_command_propagates_host_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    monkeypatch.setenv("VLLM_API_KEY", "test-token")

    cmd = server._build_run_command(
        "qwen3.5-27b",
        server.registry["qwen3.5-27b"],
        enable_request_logging=False,
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )

    assert "VLLM_API_KEY=test-token" in cmd
    assert "VLLM_ENABLE_RESPONSES_API_STORE=1" in cmd


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
            self.text = VALID_METRICS_TEXT

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
        if url.endswith("/metrics"):
            return Response()
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
    assert "expected_models=qwen3.5-27b" in log_lines[0]
    assert "served_models=qwen3.5-27b" in log_lines[0]


def test_wait_ready_uses_served_model_name_override_and_lora_adapter_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  sprint3-qwen:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    served_model_name: qwen3.5-27b
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.88
    max_num_batched_tokens: 8192
    max_num_seqs: 4
    max_lora_rank: 32
    lora_modules:
      codex-sft-all: /models/adapters/codex-sft-all
"""
    )
    server = ModelServer(registry_path=registry)
    now = 0.0
    sleeps: list[float] = []
    log_lines: list[str] = []
    responses = iter(
        [
            {"health": 200, "models": {"data": [{"id": "qwen3.5-27b"}]}},
            {"health": 200, "models": {"data": [{"id": "qwen3.5-27b"}, {"id": "codex-sft-all"}]}},
        ]
    )

    class Response:
        def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or {}
            self.text = VALID_METRICS_TEXT

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    current = {"health": 200, "models": {"data": [{"id": "qwen3.5-27b"}]}}

    def fake_run(cmd: list[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="running\n", stderr="")

    def fake_get(url: str, headers: dict[str, str], timeout: int) -> Response:
        nonlocal current
        if url.endswith("/health"):
            current = next(responses)
            return Response(status_code=current["health"])
        if url.endswith("/v1/models"):
            return Response(payload=current["models"])
        if url.endswith("/metrics"):
            return Response()
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

    server._wait_ready("sprint3-qwen", timeout_s=15)

    assert sleeps == [5]
    assert "expected_models=codex-sft-all,qwen3.5-27b" in log_lines[0]
    assert "served_models=qwen3.5-27b,codex-sft-all" in log_lines[0]


def test_is_serving_model_requires_all_expected_ids_for_lora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  sprint3-qwen:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    served_model_name: qwen3.5-27b
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.88
    max_num_batched_tokens: 8192
    max_num_seqs: 4
    max_lora_rank: 32
    lora_modules:
      codex-sft-all: /models/adapters/codex-sft-all
"""
    )
    server = ModelServer(registry_path=registry)
    monkeypatch.setattr(server, "health", lambda: None)
    monkeypatch.setattr(server, "_served_model_ids", lambda: ["qwen3.5-27b"])

    assert server._is_serving_model("sprint3-qwen") is False


def test_is_serving_model_rejects_unexpected_extra_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(server, "health", lambda: None)
    monkeypatch.setattr(server, "_served_model_ids", lambda: ["qwen3.5-27b", "unexpected-shadow"])

    assert server._is_serving_model("qwen3.5-27b") is False


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
        if url.endswith("/metrics"):
            return Response()
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


def test_wait_ready_requires_metrics_endpoint_before_success(
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
    metric_status_codes = iter([404, 200])

    class Response:
        def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = VALID_METRICS_TEXT) -> None:
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error")

        def json(self) -> dict:
            return self._payload

    def fake_run(cmd: list[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="running\n", stderr="")

    def fake_get(url: str, headers: dict[str, str], timeout: int) -> Response:
        if url.endswith("/health"):
            return Response(status_code=200)
        if url.endswith("/v1/models"):
            return Response(payload={"data": [{"id": "qwen3.5-27b"}]})
        if url.endswith("/metrics"):
            return Response(status_code=next(metric_status_codes))
        raise AssertionError(url)

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    monkeypatch.setattr(server, "_run", fake_run)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.requests.get", fake_get)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.sleep", fake_sleep)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.time", lambda: now)

    server._wait_ready("qwen3.5-27b", timeout_s=15)

    assert sleeps == [5]


def test_wait_ready_rejects_unexpected_extra_served_ids(
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
    responses = iter(
        [
            {"health": 200, "models": {"data": [{"id": "qwen3.5-27b"}, {"id": "unexpected-shadow"}]}},
            {"health": 200, "models": {"data": [{"id": "qwen3.5-27b"}]}},
        ]
    )

    class Response:
        def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = VALID_METRICS_TEXT) -> None:
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    current = {"health": 200, "models": {"data": [{"id": "qwen3.5-27b"}, {"id": "unexpected-shadow"}]}}

    def fake_run(cmd: list[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="running\n", stderr="")

    def fake_get(url: str, headers: dict[str, str], timeout: int) -> Response:
        nonlocal current
        if url.endswith("/health"):
            current = next(responses)
            return Response(status_code=current["health"])
        if url.endswith("/v1/models"):
            return Response(payload=current["models"])
        if url.endswith("/metrics"):
            return Response()
        raise AssertionError(url)

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    monkeypatch.setattr(server, "_run", fake_run)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.requests.get", fake_get)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.sleep", fake_sleep)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.time", lambda: now)

    server._wait_ready("qwen3.5-27b", timeout_s=15)

    assert sleeps == [5]


def test_wait_ready_rejects_malformed_v1_models_entries(
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
    responses = iter(
        [
            {"health": 200, "models": {"data": [{"id": "qwen3.5-27b"}, {"shadow": "missing-id"}]}},
            {"health": 200, "models": {"data": [{"id": "qwen3.5-27b"}]}},
        ]
    )

    class Response:
        def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = VALID_METRICS_TEXT) -> None:
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    current = {"health": 200, "models": {"data": [{"id": "qwen3.5-27b"}, {"shadow": "missing-id"}]}}

    def fake_run(cmd: list[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="running\n", stderr="")

    def fake_get(url: str, headers: dict[str, str], timeout: int) -> Response:
        nonlocal current
        if url.endswith("/health"):
            current = next(responses)
            return Response(status_code=current["health"])
        if url.endswith("/v1/models"):
            return Response(payload=current["models"])
        if url.endswith("/metrics"):
            return Response()
        raise AssertionError(url)

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    monkeypatch.setattr(server, "_run", fake_run)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.requests.get", fake_get)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.sleep", fake_sleep)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.time", lambda: now)

    server._wait_ready("qwen3.5-27b", timeout_s=15)

    assert sleeps == [5]


def test_wait_ready_requires_metrics_schema_before_success(
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
    metric_payloads = iter(
        [
            "# HELP missing metrics\nmetric_without_required_fields 1\n",
            VALID_METRICS_TEXT,
        ]
    )

    class Response:
        def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error")

        def json(self) -> dict:
            return self._payload

    def fake_run(cmd: list[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="running\n", stderr="")

    def fake_get(url: str, headers: dict[str, str], timeout: int) -> Response:
        if url.endswith("/health"):
            return Response(status_code=200)
        if url.endswith("/v1/models"):
            return Response(payload={"data": [{"id": "qwen3.5-27b"}]})
        if url.endswith("/metrics"):
            return Response(text=next(metric_payloads))
        raise AssertionError(url)

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    monkeypatch.setattr(server, "_run", fake_run)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.requests.get", fake_get)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.sleep", fake_sleep)
    monkeypatch.setattr("lumo_flywheel_serving.model_server.time.time", lambda: now)

    server._wait_ready("qwen3.5-27b", timeout_s=15)

    assert sleeps == [5]


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


def test_registry_rejects_local_paths_outside_models_mount(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /srv/models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
"""
    )

    with pytest.raises(ValueError, match="under /models"):
        load_registry(registry)


def test_registry_rejects_non_normalized_models_mount_paths(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/../escape
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
"""
    )

    with pytest.raises(ValueError, match="normalized container path under /models"):
        load_registry(registry)

    lora_registry = tmp_path / "lora_model_registry.yaml"
    lora_registry.write_text(
        """
models:
  sprint3-qwen:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_lora_rank: 32
    lora_modules:
      codex-sft-all: /models/adapters/../escape
"""
    )

    with pytest.raises(ValueError, match=r"lora_modules\[codex-sft-all\].*normalized container path under /models"):
        load_registry(lora_registry)


def test_registry_rejects_malformed_model_surface_ids(tmp_path: Path) -> None:
    registry = tmp_path / "bad_model_id.yaml"
    registry.write_text(
        """
models:
  BadModel:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
"""
    )

    with pytest.raises(ValueError, match="Registry model_id must be a lowercase slug"):
        load_registry(registry)

    served_name_registry = tmp_path / "bad_served_name.yaml"
    served_name_registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    served_model_name: " qwen3.5-27b "
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
"""
    )

    with pytest.raises(ValueError, match="served_model_name must not contain leading or trailing whitespace"):
        load_registry(served_name_registry)

    adapter_name_registry = tmp_path / "bad_adapter_name.yaml"
    adapter_name_registry.write_text(
        """
models:
  sprint3-qwen:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_lora_rank: 32
    lora_modules:
      codex/sft/all: /models/adapters/codex-sft-all
"""
    )

    with pytest.raises(ValueError, match="adapter_name must be a lowercase slug"):
        load_registry(adapter_name_registry)


def test_registry_rejects_blank_served_model_name(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    served_model_name: ""
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
"""
    )

    with pytest.raises(ValueError, match="served_model_name"):
        load_registry(registry)


def test_registry_rejects_lora_adapter_name_collision_with_served_model_name(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  sprint3-qwen:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    served_model_name: qwen3.5-27b
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
    max_lora_rank: 32
    lora_modules:
      qwen3.5-27b: /models/adapters/codex-sft-all
"""
    )

    with pytest.raises(ValueError, match="collides with served_model_name"):
        load_registry(registry)


def test_registry_rejects_duplicate_exposed_model_ids_across_models(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  base-qwen:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    served_model_name: qwen3.5-27b
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
  lora-qwen:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    served_model_name: qwen3.5-27b-sprint3
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.88
    max_lora_rank: 32
    lora_modules:
      qwen3.5-27b: /models/adapters/codex-sft-all
"""
    )

    with pytest.raises(ValueError, match="globally unique"):
        load_registry(registry)


def test_registry_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/first
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/second
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate YAML key 'qwen3.5-27b'"):
        load_registry(registry)


def test_registry_rejects_non_fp8_precision_baseline(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B
    local_path: /models/qwen3.5-27b
    quantization: awq
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
"""
    )

    with pytest.raises(ValueError, match="quantization must be 'fp8'"):
        load_registry(registry)


def test_registry_allows_missing_hf_revision_for_local_serving_configs(tmp_path: Path) -> None:
    registry = tmp_path / "missing_revision.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
"""
    )

    registry_data = load_registry(registry)
    assert registry_data["qwen3.5-27b"].hf_revision is None

    gated_registry = tmp_path / "gated_missing_revision.yaml"
    gated_registry.write_text(
        """
models:
  qwen3-coder-next-80b-a3b:
    hf_repo: Qwen/Qwen3-Coder-Next-80B-A3B
    local_path: /models/qwen3-coder-next-80b-a3b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.93
    sprint0_gate: "Confirm upstream FP8 checkpoint identity before Dev-Bench"
"""
    )

    registry_data = load_registry(gated_registry)
    assert registry_data["qwen3-coder-next-80b-a3b"].hf_revision is None
    assert registry_data["qwen3-coder-next-80b-a3b"].sprint0_gate == (
        "Confirm upstream FP8 checkpoint identity before Dev-Bench"
    )


def test_registry_rejects_malformed_hf_revision_and_blank_sprint0_gate(tmp_path: Path) -> None:
    bad_revision_registry = tmp_path / "bad_revision.yaml"
    bad_revision_registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    hf_revision: latest
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
"""
    )

    with pytest.raises(ValueError, match="40-character git commit hash"):
        load_registry(bad_revision_registry)

    blank_gate_registry = tmp_path / "blank_gate.yaml"
    blank_gate_registry.write_text(
        """
models:
  qwen3-coder-next-80b-a3b:
    hf_repo: Qwen/Qwen3-Coder-Next-80B-A3B
    local_path: /models/qwen3-coder-next-80b-a3b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.93
    sprint0_gate: ""
"""
    )

    with pytest.raises(ValueError, match="sprint0_gate must be a non-empty string"):
        load_registry(blank_gate_registry)


def test_registry_rejects_non_auto_dtype_or_unknown_kv_cache_dtype(tmp_path: Path) -> None:
    dtype_registry = tmp_path / "dtype_registry.yaml"
    dtype_registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B
    local_path: /models/qwen3.5-27b
    quantization: fp8
    dtype: bfloat16
    max_model_len: 131072
    gpu_memory_utilization: 0.9
"""
    )
    with pytest.raises(ValueError, match="dtype must be 'auto'"):
        load_registry(dtype_registry)

    kv_registry = tmp_path / "kv_registry.yaml"
    kv_registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B
    local_path: /models/qwen3.5-27b
    quantization: fp8
    dtype: auto
    kv_cache_dtype: bf16
    max_model_len: 131072
    gpu_memory_utilization: 0.9
"""
    )
    with pytest.raises(ValueError, match="kv_cache_dtype must be 'fp8_e5m2' or 'auto'"):
        load_registry(kv_registry)


@pytest.mark.parametrize(
    ("field_name", "field_value", "expected_error"),
    [
        ("max_model_len", "0", "max_model_len"),
        ("gpu_memory_utilization", "1.0", "gpu_memory_utilization"),
        ("max_num_batched_tokens", "0", "max_num_batched_tokens"),
        ("max_num_seqs", "0", "max_num_seqs"),
        ("max_lora_rank", "0", "max_lora_rank"),
    ],
)
def test_registry_rejects_non_positive_or_out_of_range_launch_parameters(
    tmp_path: Path, field_name: str, field_value: str, expected_error: str
) -> None:
    registry = tmp_path / f"{field_name}.yaml"
    max_model_len = "131072"
    gpu_memory_utilization = "0.9"
    max_num_batched_tokens = "8192"
    max_num_seqs = "4"
    max_lora_rank = ""
    lora_modules = ""
    if field_name == "max_model_len":
        max_model_len = field_value
    elif field_name == "gpu_memory_utilization":
        gpu_memory_utilization = field_value
    elif field_name == "max_num_batched_tokens":
        max_num_batched_tokens = field_value
    elif field_name == "max_num_seqs":
        max_num_seqs = field_value
    elif field_name == "max_lora_rank":
        max_lora_rank = f"\n    max_lora_rank: {field_value}"
        lora_modules = "\n    lora_modules:\n      codex-sft-all: /models/adapters/codex-sft-all"
    registry.write_text(
        f"""
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: {max_model_len}
    gpu_memory_utilization: {gpu_memory_utilization}
    max_num_batched_tokens: {max_num_batched_tokens}
    max_num_seqs: {max_num_seqs}{max_lora_rank}{lora_modules}
"""
    )

    with pytest.raises(ValueError, match=expected_error):
        load_registry(registry)


def test_format_lora_modules_rejects_non_normalized_container_paths(tmp_path: Path) -> None:
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    max_model_len: 131072
    gpu_memory_utilization: 0.9
"""
    )
    server = ModelServer(registry_path=registry)
    config = ModelConfig(
        model_id="qwen3.5-27b",
        hf_repo="Qwen/Qwen3.5-27B-FP8",
        local_path=Path("/models/qwen3.5-27b-fp8"),
        served_model_name="qwen3.5-27b",
        quantization="fp8",
        dtype="auto",
        kv_cache_dtype="fp8_e5m2",
        max_model_len=131072,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=8192,
        max_num_seqs=4,
        max_lora_rank=32,
        lora_modules=(("codex-sft-all", Path("/models/adapters/../escape")),),
    )

    with pytest.raises(ValueError, match="must use a container path under /models"):
        server._format_lora_modules(config)
