from __future__ import annotations

import argparse
import json
import importlib.util
from pathlib import Path

import yaml

from lumo_flywheel_serving.tuned_config import load_tuned_config_bundle
from lumo_flywheel_serving.model_server import ModelServer


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


def test_runtime_vllm_config_rejects_unsupported_kv_dtype(tmp_path: Path) -> None:
    loop = _load_loop_module()
    candidate_dir = tmp_path / "candidates" / "000"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "serve_config.yaml").write_text(
        yaml.safe_dump({"vllm_config": {"kv_cache_dtype": "fp8_e4m3", "max_num_seqs": 8}}),
        encoding="utf-8",
    )

    config, error = loop._load_serve_config(candidate_dir)
    overrides, override_error = loop._parse_vllm_config_overrides(config)

    assert error is None
    assert overrides is None
    assert override_error == "invalid_vllm_config_kv_cache_dtype:must_be_fp8_e5m2_or_auto"


def test_runtime_vllm_config_without_max_num_seqs_uses_default_concurrency(tmp_path: Path) -> None:
    loop = _load_loop_module()
    config = {"vllm_config": {"kv_cache_dtype": "auto"}}
    overrides, override_error = loop._parse_vllm_config_overrides(config)
    default_concurrency = 4 if overrides is not None else None
    concurrency = loop._parse_target_concurrency_from_config(
        config,
        default_from_runtime_config=default_concurrency,
    )

    assert override_error is None
    assert concurrency == 4


def test_spec_decode_candidate_parses_ngram_config() -> None:
    loop = _load_loop_module()
    parsed, error = loop._parse_spec_decode_config(
        {
            "spec_decode": {
                "method": "ngram",
                "num_speculative_tokens": 6,
                "prompt_lookup_min": 2,
                "prompt_lookup_max": 8,
            }
        }
    )

    assert error is None
    assert parsed == {
        "method": "ngram",
        "num_speculative_tokens": 6,
        "prompt_lookup_min": 2,
        "prompt_lookup_max": 8,
    }


def test_spec_decode_candidate_rejects_unsupported_method() -> None:
    loop = _load_loop_module()
    parsed, error = loop._parse_spec_decode_config(
        {"spec_decode": {"method": "draft_model", "num_speculative_tokens": 4}}
    )

    assert parsed is None
    assert error == "invalid_spec_decode_method:must_be_ngram"


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
        runtime_proxy_port=8011,
        state_root=tmp_path / "state",
        runtime_ready_timeout_s=1,
    )
    bundle_path = loop._write_runtime_tuned_config_bundle(
        args,
        round_dir=round_dir,
        candidate_dir=candidate_dir,
        candidate_id="000",
        candidate_config={
            "request_shaping": {"target_concurrency": 8},
            "spec_decode": {"method": "ngram", "num_speculative_tokens": 4},
        },
        vllm_config_overrides={"max_num_seqs": 8, "max_num_batched_tokens": 16384},
        spec_decode_config={"method": "ngram", "num_speculative_tokens": 4},
        workload_file=workload,
        target_tps=37.5,
        candidate_accept_tps=9.0,
    )

    bundle = load_tuned_config_bundle(bundle_path)

    assert bundle.vllm_config["max_num_seqs"] == 8
    assert bundle.vllm_config["max_num_batched_tokens"] == 16384
    assert bundle.vllm_config["max_model_len"] == 131072
    assert bundle.request_shaping["target_concurrency"] == 8
    assert bundle.spec_decode["method"] == "ngram"
    assert bundle.objective["candidate_accept_decode_tps"] == 9.0
    assert bundle.round_provenance["round_type"] == "track_b_auto_research_runtime_config"


def test_stale_active_tuned_config_state_is_tolerated(tmp_path: Path) -> None:
    loop = _load_loop_module()
    registry = tmp_path / "model_registry.yaml"
    state_root = tmp_path / "state"
    state_root.mkdir()
    _write_registry(registry)
    (state_root / "serving_runtime_state.json").write_text(
        json.dumps(
            {
                "current_model_id": "qwen3.5-27b",
                "active_tuned_config_path": str(tmp_path / "missing-bundle.yaml"),
                "active_tuned_config_id": "stale",
                "status": "READY",
            }
        ),
        encoding="utf-8",
    )
    server = ModelServer(
        registry_path=registry,
        port=9950,
        container_name="test-vllm",
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
        state_root=state_root,
        ready_timeout_s=1,
    )

    bundle_path, bundle, warning = loop._active_tuned_config_bundle_safe(server, "qwen3.5-27b")

    assert bundle_path is None
    assert bundle is None
    assert warning == "Invalid tuned-config bundle"


def test_duplicate_serving_surface_detects_prior_candidate(tmp_path: Path) -> None:
    loop = _load_loop_module()
    round_dir = tmp_path / "round"
    prior = round_dir / "candidates" / "001"
    current = round_dir / "candidates" / "002"
    prior.mkdir(parents=True)
    current.mkdir(parents=True)
    config = {"request_shaping": {"target_concurrency": 4}}
    (prior / "serve_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (prior / "controller_result.json").write_text(
        json.dumps({"status": "rejected", "reason": "speed_below_candidate_acceptance"}),
        encoding="utf-8",
    )
    (current / "serve_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    signature = loop._surface_signature(config)

    assert loop._has_prior_surface_signature(round_dir, signature, current_candidate_id="002")


def test_surface_history_marks_request_shaping_only_as_exhausted(tmp_path: Path) -> None:
    loop = _load_loop_module()
    round_dir = tmp_path / "round"
    for index, concurrency in enumerate((2, 4, 6, 8), start=1):
        candidate = round_dir / "candidates" / f"{index:03d}"
        candidate.mkdir(parents=True)
        (candidate / "serve_config.yaml").write_text(
            yaml.safe_dump({"request_shaping": {"target_concurrency": concurrency}}),
            encoding="utf-8",
        )
        (candidate / "controller_result.json").write_text(
            json.dumps(
                {
                    "status": "rejected",
                    "reason": "speed_below_candidate_acceptance",
                    "decode_tps": 7.3,
                }
            ),
            encoding="utf-8",
        )

    history = loop._candidate_surface_history(round_dir)
    brief = loop._render_exhausted_surface_brief(history)

    assert loop._request_shaping_only_exhausted(history)
    assert "request_shaping-only candidates are exhausted" in brief
    assert "prefer a vllm_config runtime candidate" in brief
    assert "vLLM ngram spec_decode is supported and unmeasured" in brief
