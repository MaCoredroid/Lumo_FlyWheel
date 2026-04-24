from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import requests

from lumo_flywheel_serving.inference_proxy import build_proxy_handler
from lumo_flywheel_serving.tuned_config import (
    compute_workload_distribution_id,
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


def test_admin_endpoints_return_structured_validation_errors_and_accept_valid_payloads(tmp_path: Path) -> None:
    registry_path = tmp_path / "model_registry.yaml"
    _write_registry(registry_path)
    workload_path = tmp_path / "workload.yaml"
    (tmp_path / "seed.jsonl").write_text('{"prompt_tokens": 16, "output_tokens": 8}\n', encoding="utf-8")
    (tmp_path / "holdout.jsonl").write_text('{"prompt_tokens": 16, "output_tokens": 8}\n', encoding="utf-8")
    workload_path.write_text(
        """
family_id: proposal-ranking-manager-judgment
workload_distribution_id: null
seed_trace_ref: seed.jsonl
holdout_trace_ref: holdout.jsonl
""",
        encoding="utf-8",
    )
    workload_distribution_id = compute_workload_distribution_id(workload_path)
    bundle = make_tuned_config_bundle(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        weight_version_id="2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
        workload_distribution_id=workload_distribution_id,
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
        round_provenance={
            "confidence": "defensible",
            "latency_above_slo": False,
            "workload_descriptor_path": str(workload_path),
        },
    )
    bundle_path = persist_tuned_config_bundle(bundle, tmp_path / "bundles")

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_proxy_handler(
            "http://127.0.0.1:9",
            state_root=tmp_path / "state",
            registry_path=registry_path,
        ),
    )
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"

        invalid_load = requests.post(f"{base_url}/admin/load_tuned_config", json={"bundle_path": ""}, timeout=10)
        assert invalid_load.status_code == 400
        invalid_load_payload = invalid_load.json()
        assert invalid_load_payload["error"]["code"] == "validation_error"
        assert invalid_load_payload["error"]["details"][0]["field"] == "bundle_path"

        valid_load = requests.post(
            f"{base_url}/admin/load_tuned_config",
            json={"bundle_path": str(bundle_path)},
            timeout=10,
        )
        assert valid_load.status_code == 200
        assert valid_load.json()["bundle_id"] == bundle.bundle_id

        invalid_invalidate = requests.post(f"{base_url}/admin/invalidate", json={}, timeout=10)
        assert invalid_invalidate.status_code == 400
        invalid_invalidate_payload = invalid_invalidate.json()
        assert invalid_invalidate_payload["error"]["code"] == "validation_error"
        assert invalid_invalidate_payload["error"]["details"][0]["field"] == "weight_version_id"

        valid_invalidate = requests.post(
            f"{base_url}/admin/invalidate",
            json={"weight_version_id": "new-weight-version"},
            timeout=10,
        )
        assert valid_invalidate.status_code == 200
        assert valid_invalidate.json()["weight_version_id"] == "new-weight-version"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
