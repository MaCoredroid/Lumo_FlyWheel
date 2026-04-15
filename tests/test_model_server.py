from pathlib import Path

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


def test_next_gpu_memory_utilization_stops_before_context_floor() -> None:
    assert ModelServer._next_gpu_memory_utilization("Free memory on device cuda:0", current=0.15) is None


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
