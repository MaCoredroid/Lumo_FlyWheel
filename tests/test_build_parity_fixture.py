from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lumo_flywheel_serving.parity_fixture import (
    DEFAULT_WEIGHT_VERSION_ID,
    DELTANET_STATE_DEBUG_KIND,
    LOGITS_DEBUG_KIND,
    KERNEL_TARGETS,
    P2B_FAMILY_PROBE_COUNTS,
    validate_p2b_fixture_set,
)
import scripts.build_parity_fixture as build_parity_fixture


def test_request_completion_makes_required_state_checkpoint_non_terminal(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"id": "cmpl-test"}

    def fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: float) -> Response:
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(build_parity_fixture.requests, "post", fake_post)

    result = build_parity_fixture._request_completion(
        endpoint="http://127.0.0.1:8100/v1",
        api_key="EMPTY",
        model="qwen3.5-27b",
        probe={"prompt": "probe", "output_token_count": 1024},
        timeout_s=30.0,
        minimum_completion_tokens=1025,
    )

    assert result == {"id": "cmpl-test"}
    assert captured["json"]["max_tokens"] == 1025
    assert captured["json"]["min_tokens"] == 1025
    assert captured["json"]["ignore_eos"] is True


def test_request_completion_does_not_extend_gatedattn_only_probe(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"id": "cmpl-test"}

    def fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: float) -> Response:
        captured.update({"json": json})
        return Response()

    monkeypatch.setattr(build_parity_fixture.requests, "post", fake_post)

    build_parity_fixture._request_completion(
        endpoint="http://127.0.0.1:8100/v1",
        api_key="EMPTY",
        model="qwen3.5-27b",
        probe={"prompt": "probe", "output_token_count": 16},
        timeout_s=30.0,
    )

    assert captured["json"]["max_tokens"] == 16
    assert "min_tokens" not in captured["json"]
    assert "ignore_eos" not in captured["json"]


def _write_seed_trace(repo: Path, family_id: str) -> None:
    family_dir = repo / "benchmark_blueprints" / "families" / family_id
    family_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "turn_index": 0,
            "prompt_tokens": 64,
            "request_max_output_tokens": 16,
            "capture_prompt_label": "short",
            "family_id": family_id,
        }
    ]
    (family_dir / "seed_trace_v5.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_debug_logits(path: Path, torch_module: object, *, request_id: str) -> None:
    torch_module.save(
        {
            "kind": LOGITS_DEBUG_KIND,
            "request_id": request_id,
            "generated_token_index": 1,
            "source_shape": (4,),
            "saved_shape": (4,),
            "logits_is_truncated": False,
            "logits": torch_module.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch_module.float32),
        },
        path,
    )


def _write_debug_state(path: Path, torch_module: object, *, request_id: str, token: int) -> None:
    torch_module.save(
        {
            "kind": DELTANET_STATE_DEBUG_KIND,
            "request_id": request_id,
            "generated_token_index": token,
            "layers": {
                "model.layers.0.linear_attn": [
                    {
                        "state_index": 1,
                        "state_role": "recurrent_ssm_state",
                        "saved_shape": (2,),
                        "saved_dtype": "torch.float32",
                        "tensor": torch_module.tensor([float(token), 1.0], dtype=torch_module.float32),
                    }
                ]
            },
        },
        path,
    )


def test_convert_debug_artifacts_writes_full_p2b_fixture_set(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    for family_id in P2B_FAMILY_PROBE_COUNTS:
        _write_seed_trace(tmp_path, family_id)

    source_root = tmp_path / "output" / "logs"
    source_root.mkdir(parents=True)
    request_id = "cmpl-real-capture-0"
    _write_debug_logits(source_root / f"logits_req_{request_id}_tok_000001.pt", torch, request_id=request_id)
    _write_debug_state(source_root / f"state_req_{request_id}_tok_000001.pt", torch, request_id=request_id, token=1)
    _write_debug_state(source_root / f"state_req_{request_id}_tok_001024.pt", torch, request_id=request_id, token=1024)

    result = build_parity_fixture._convert_debug_artifacts_to_p2b_fixture_set(
        repo_root=tmp_path,
        source_debug_root=source_root,
        weight_version_id=DEFAULT_WEIGHT_VERSION_ID,
        vllm_version="0.19.0-test",
        generated_at="2026-04-26T00:00:00Z",
    )

    assert result["validation_pass"], result["validation_errors"]
    assert result["source_counts"] == {"logits": 1, "state_token_1": 1, "state_token_1024": 1}
    yaml_paths = [path for path in result["written"] if path.endswith("_v1.yaml")]
    assert len(yaml_paths) == len(P2B_FAMILY_PROBE_COUNTS) * len(KERNEL_TARGETS)
    validation = validate_p2b_fixture_set(tmp_path, expected_weight_version_id=DEFAULT_WEIGHT_VERSION_ID)
    assert validation["pass"], validation["errors"]
