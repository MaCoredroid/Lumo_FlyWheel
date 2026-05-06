from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import yaml

from lumo_flywheel_serving.tuned_config import load_tuned_config_bundle


def _load_loop_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_track_b_loop.py"
    spec = importlib.util.spec_from_file_location("run_track_b_loop", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_registry(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "qwen3.5-27b": {
                        "hf_repo": "Qwen/Qwen3.5-27B-FP8",
                        "hf_revision": "2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
                        "local_path": "/models/qwen3.5-27b-fp8",
                        "quantization": "fp8",
                        "dtype": "auto",
                        "kv_cache_dtype": "fp8_e5m2",
                        "max_model_len": 131072,
                        "gpu_memory_utilization": 0.9,
                        "max_num_batched_tokens": 8192,
                        "max_num_seqs": 4,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _write_workload(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "seed.jsonl").write_text('{"prompt_tokens": 128, "output_tokens": 16}\n', encoding="utf-8")
    (path.parent / "holdout.jsonl").write_text('{"prompt_tokens": 256, "output_tokens": 16}\n', encoding="utf-8")
    path.write_text(
        yaml.safe_dump(
            {
                "workload_distribution_id": None,
                "seed_trace_ref": "seed.jsonl",
                "holdout_trace_ref": "holdout.jsonl",
            }
        ),
        encoding="utf-8",
    )


def test_runtime_vllm_config_candidate_defaults_concurrency_from_max_num_seqs(tmp_path: Path) -> None:
    loop = _load_loop_module()
    candidate_dir = tmp_path / "candidates" / "000"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "serve_config.yaml").write_text(
        yaml.safe_dump({"vllm_config": {"max_num_seqs": 6}}),
        encoding="utf-8",
    )

    config, error = loop._load_serve_config(candidate_dir)
    overrides, override_error = loop._parse_vllm_config_overrides(config)
    concurrency = loop._parse_target_concurrency_from_config(
        config,
        default_from_runtime_config=overrides["max_num_seqs"],
    )

    assert error is None
    assert override_error is None
    assert concurrency == 6


def test_runtime_tuned_config_bundle_merges_candidate_overrides(tmp_path: Path) -> None:
    loop = _load_loop_module()
    registry = tmp_path / "model_registry.yaml"
    workload = tmp_path / "workload" / "workload.yaml"
    round_dir = tmp_path / "round"
    candidate_dir = round_dir / "candidates" / "000"
    candidate_dir.mkdir(parents=True)
    _write_registry(registry)
    _write_workload(workload)

    args = argparse.Namespace(
        model="qwen3.5-27b",
        registry_path=registry,
        port=9950,
        runtime_container_name="test-vllm",
        runtime_logs_root=tmp_path / "logs",
        runtime_triton_cache_root=tmp_path / "triton",
        state_root=tmp_path / "state",
        runtime_ready_timeout_s=1,
    )
    bundle_path = loop._write_runtime_tuned_config_bundle(
        args,
        round_dir=round_dir,
        candidate_dir=candidate_dir,
        candidate_id="000",
        candidate_config={"request_shaping": {"target_concurrency": 8}},
        vllm_config_overrides={"max_num_seqs": 8, "max_num_batched_tokens": 16384},
        workload_file=workload,
        target_tps=37.5,
        candidate_accept_tps=9.0,
    )

    bundle = load_tuned_config_bundle(bundle_path)

    assert bundle.vllm_config["max_num_seqs"] == 8
    assert bundle.vllm_config["max_num_batched_tokens"] == 16384
    assert bundle.vllm_config["max_model_len"] == 131072
    assert bundle.request_shaping["target_concurrency"] == 8
    assert bundle.objective["candidate_accept_decode_tps"] == 9.0
    assert bundle.round_provenance["round_type"] == "track_b_auto_research_runtime_config"
