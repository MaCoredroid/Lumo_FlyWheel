from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_track_b_runtime_config_hash.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_track_b_runtime_config_hash", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parses_compact_init_lines() -> None:
    module = _load_module()
    log = (
        "[VLLM-INIT] timestamp=2026-05-07T08:19:12.273177+00:00\n"
        "[VLLM-INIT] model_id=qwen3.5-27b served_model_name=qwen3.5-27b vllm_version=0.19.0 git_hash=unknown\n"
        "[VLLM-INIT] quantization=fp8 kv_cache_dtype=auto\n"
        "[VLLM-INIT] max_model_len=131072 gpu_memory_utilization=0.9\n"
        "[VLLM-INIT] enforce_eager=false\n"
        "[VLLM-INIT] tuned_config_id=712fd011 weight_version_id=2e1b21\n"
        "[VLLM-INIT] kernel_runtime_activation={\"resolved\":{\"attention_backend\":\"vllm-auto\"}}\n"
        "[VLLM-INIT] speculative_config={\"method\":\"suffix\",\"num_speculative_tokens\":12}\n"
        "[VLLM-INIT] wire_api=responses\n"
    )
    fields = module.parse_init_log(log)
    assert fields["model_id"] == "qwen3.5-27b"
    assert fields["served_model_name"] == "qwen3.5-27b"
    assert fields["vllm_version"] == "0.19.0"
    assert fields["quantization"] == "fp8"
    assert fields["kv_cache_dtype"] == "auto"
    assert fields["max_model_len"] == 131072
    assert fields["gpu_memory_utilization"] == 0.9
    assert fields["enforce_eager"] is False
    assert fields["wire_api"] == "responses"
    assert fields["speculative_config"] == {"method": "suffix", "num_speculative_tokens": 12}
    assert fields["kernel_runtime_activation"] == {"resolved": {"attention_backend": "vllm-auto"}}


def test_canonical_payload_picks_only_hash_fields() -> None:
    module = _load_module()
    fields = {
        "model_id": "qwen3.5-27b",
        "vllm_version": "0.19.0",
        "timestamp": "2026-05-07T08:19:12Z",  # not in HASH_FIELDS
        "speculative_config": {"method": "suffix"},
    }
    payload = module.canonical_payload(fields)
    assert "timestamp" not in payload
    assert payload["model_id"] == "qwen3.5-27b"
    assert payload["vllm_version"] == "0.19.0"
    assert payload["speculative_config"] == {"method": "suffix"}


def test_compute_hash_is_deterministic_and_field_order_independent() -> None:
    module = _load_module()
    h1 = module.compute_hash({"a": 1, "b": 2, "c": [3, 4]})
    h2 = module.compute_hash({"c": [3, 4], "b": 2, "a": 1})
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert len(h1) == len("sha256:") + 64
