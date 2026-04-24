from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lumo_flywheel_serving.model_server import ModelServer
from lumo_flywheel_serving.tuned_config import (
    RuntimeStateStore,
    StructuredValidationError,
    compute_workload_distribution_id,
    load_tuned_config_bundle,
    make_tuned_config_bundle,
    persist_tuned_config_bundle,
    validate_bundle_load_policy,
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


def test_bundle_confidence_policy_warns_strict_rejects_and_pins_workload_id(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    seed = tmp_path / "seed_trace.jsonl"
    holdout = tmp_path / "holdout_trace.jsonl"
    seed.write_text('{"turn_index": 0, "prompt_tokens": 1, "output_tokens": 1}\n', encoding="utf-8")
    holdout.write_text('{"turn_index": 1, "prompt_tokens": 1, "output_tokens": 1}\n', encoding="utf-8")
    descriptor = tmp_path / "workload.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "family_id": "proposal-ranking-manager-judgment",
                "workload_distribution_id": None,
                "workload_distribution_id_hardening_version": "v1-thinking-realistic",
                "seed_trace_ref": "seed_trace.jsonl",
                "holdout_trace_ref": "holdout_trace.jsonl",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    workload_id = compute_workload_distribution_id(descriptor)
    payload = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    payload["workload_distribution_id"] = workload_id
    descriptor.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    bundle = make_tuned_config_bundle(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        weight_version_id="2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
        workload_distribution_id=workload_id,
        vllm_config={
            "max_num_seqs": 1,
            "max_num_batched_tokens": 6144,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": 0.08,
            "max_model_len": 32768,
            "kv_cache_dtype": "fp8_e5m2",
        },
        objective={"metric": "eval_throughput", "value": 1},
        measurement_trace_ref="measurement.json",
        search_trace_ref="search.json",
        baseline_bundle_id=None,
        regression_guard={},
        safety_rails={},
        round_provenance={
            "confidence": "within_noise_floor",
            "latency_above_slo": True,
            "workload_descriptor_path": str(descriptor),
        },
    )

    with caplog.at_level("WARNING"):
        warnings = validate_bundle_load_policy(bundle, bundle_confidence_policy="warn")
    assert {warning["code"] for warning in warnings} == {
        "bundle_confidence_not_defensible",
        "bundle_latency_above_slo",
    }
    assert "non_defensible_tuned_config_bundle" in caplog.text
    with pytest.raises(StructuredValidationError, match="bundle-validity: refused"):
        validate_bundle_load_policy(bundle, bundle_confidence_policy="strict")

    mismatched = make_tuned_config_bundle(
        model_id=bundle.model_id,
        family_id=bundle.family_id,
        weight_version_id=bundle.weight_version_id,
        workload_distribution_id="not-canonical",
        vllm_config=bundle.vllm_config,
        objective=bundle.objective,
        measurement_trace_ref=bundle.measurement_trace_ref,
        search_trace_ref=bundle.search_trace_ref,
        baseline_bundle_id=None,
        regression_guard={},
        safety_rails={},
        round_provenance={"confidence": "defensible", "workload_descriptor_path": str(descriptor)},
    )
    with pytest.raises(StructuredValidationError, match="bundle-validity: refused"):
        validate_bundle_load_policy(mismatched, bundle_confidence_policy="passthrough")
