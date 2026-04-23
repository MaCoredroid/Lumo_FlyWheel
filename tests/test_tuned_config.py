from __future__ import annotations

from pathlib import Path

from lumo_flywheel_serving.model_server import ModelServer
from lumo_flywheel_serving.tuned_config import (
    RuntimeStateStore,
    load_tuned_config_bundle,
    make_tuned_config_bundle,
    persist_tuned_config_bundle,
)


def _write_registry(path: Path) -> None:
    path.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    hf_revision: 2e1b21350ce589fcaafbb3c7d7eac526a7aed582
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.90
    max_num_batched_tokens: 8192
    max_num_seqs: 4
""",
        encoding="utf-8",
    )


def test_tuned_config_round_trip_and_runtime_override(tmp_path: Path) -> None:
    registry_path = tmp_path / "model_registry.yaml"
    _write_registry(registry_path)
    bundle = make_tuned_config_bundle(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        weight_version_id="2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
        workload_distribution_id="prmj-v1-live",
        vllm_config={
            "max_num_seqs": 8,
            "max_num_batched_tokens": 12288,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.93,
            "max_model_len": 65536,
            "kv_cache_dtype": "fp8_e5m2",
        },
        objective={"metric": "sustained_concurrent_eval_threads_at_L_ceiling", "value": 8},
        measurement_trace_ref=str(tmp_path / "measurement.json"),
        search_trace_ref=str(tmp_path / "search.json"),
        baseline_bundle_id=None,
        regression_guard={"baseline_value": 4, "delta": 4},
        safety_rails={"regression_guard_passed": True},
    )
    bundle_path = persist_tuned_config_bundle(bundle, tmp_path / "bundles")

    loaded = load_tuned_config_bundle(bundle_path)
    assert loaded.family_id == "proposal-ranking-manager-judgment"

    store = RuntimeStateStore(tmp_path / "state")
    state = store.activate_bundle(bundle_path, loaded)
    assert state.active_tuned_config_path == str(bundle_path)

    server = ModelServer(registry_path=registry_path, state_root=tmp_path / "state")
    resolved_config, resolved_path, resolved_bundle = server.resolved_model_config("qwen3.5-27b")

    assert resolved_path == str(bundle_path)
    assert resolved_bundle is not None
    assert resolved_bundle.bundle_id == loaded.bundle_id
    assert resolved_config.max_num_seqs == 8
    assert resolved_config.max_num_batched_tokens == 12288
    assert resolved_config.gpu_memory_utilization == 0.93
    assert resolved_config.max_model_len == 65536


def test_launch_command_respects_loaded_prefix_caching_flag(tmp_path: Path) -> None:
    registry_path = tmp_path / "model_registry.yaml"
    _write_registry(registry_path)
    bundle = make_tuned_config_bundle(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        weight_version_id="2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
        workload_distribution_id="prmj-v1-live",
        vllm_config={
            "max_num_seqs": 1,
            "max_num_batched_tokens": 6144,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": 0.08,
            "max_model_len": 32768,
            "kv_cache_dtype": "fp8_e5m2",
        },
        objective={"metric": "sustained_concurrent_eval_threads_at_L_ceiling", "value": 1},
        measurement_trace_ref=str(tmp_path / "measurement.json"),
        search_trace_ref=str(tmp_path / "search.json"),
        baseline_bundle_id=None,
        regression_guard={"baseline_value": 0, "delta": 1},
        safety_rails={"regression_guard_passed": True},
    )
    bundle_path = persist_tuned_config_bundle(bundle, tmp_path / "bundles")

    store = RuntimeStateStore(tmp_path / "state")
    store.activate_bundle(bundle_path, bundle)
    server = ModelServer(
        registry_path=registry_path,
        state_root=tmp_path / "state",
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
    )
    resolved_config, _resolved_path, resolved_bundle = server.resolved_model_config("qwen3.5-27b")

    command = server._build_run_command(
        "qwen3.5-27b",
        resolved_config,
        False,
        kv_cache_dtype="auto",
        gpu_memory_utilization=resolved_config.gpu_memory_utilization,
        enforce_eager=False,
        tuned_config_id=resolved_bundle.bundle_id if resolved_bundle is not None else None,
        weight_version_id=resolved_bundle.weight_version_id if resolved_bundle is not None else None,
    )
    shell_command = command[-1]

    assert "--enable-chunked-prefill" in shell_command
    assert "--enable-prefix-caching" not in shell_command
