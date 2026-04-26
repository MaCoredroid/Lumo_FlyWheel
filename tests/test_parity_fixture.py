from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml

from lumo_flywheel_serving.parity_fixture import (
    DEFAULT_WEIGHT_VERSION_ID,
    DELTANET_STATE_DEBUG_KIND,
    KERNEL_TARGETS,
    LOGITS_DEBUG_KIND,
    P2B_FAMILY_PROBE_COUNTS,
    REFERENCE_BASELINE,
    SYNTHETIC_TEST_ARTIFACT_PURPOSE,
    DebugProbeArtifacts,
    assert_debug_capture_runs_reproduce,
    fetch_endpoint_capabilities,
    fixture_content_hash,
    fixture_payload,
    fixture_yaml_path,
    summarize_debug_export_pt,
    p2b_blocked_payload,
    validate_fixture,
    validate_p2b_fixture_set,
    write_debug_export_npz_companions,
)


def _write_seed_trace(repo: Path, family_id: str) -> None:
    family_dir = repo / "benchmark_blueprints" / "families" / family_id
    family_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "turn_index": 0,
            "prompt_tokens": 64,
            "output_tokens": 16,
            "request_max_output_tokens": 16,
            "thinking_tokens": 16,
            "capture_prompt_label": "short",
            "family_id": family_id,
        }
    ]
    (family_dir / "seed_trace_v5.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_fixture(repo: Path, family_id: str, kernel_target: str, probe_count: int) -> Path:
    fixture_dir = repo / "benchmark_blueprints" / "families" / family_id / "parity_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    probes = [
        {
            "probe_index": index,
            "family_id": family_id,
            "prompt": f"probe {index}",
            "output_token_count": 16,
        }
        for index in range(probe_count)
    ]
    (fixture_dir / "probes_input.jsonl").write_text(
        "\n".join(json.dumps(probe, sort_keys=True) for probe in probes) + "\n",
        encoding="utf-8",
    )
    _write_npz_like(fixture_dir / f"{kernel_target}_reference_logits.npz", ["logits", "probe_index"])
    payload = fixture_payload(
        family_id=family_id,
        kernel_target=kernel_target,
        probe_count=probe_count,
        weight_version_id=DEFAULT_WEIGHT_VERSION_ID,
        vllm_version="0.19.0",
        generated_at="2026-04-26T00:00:00Z",
    )
    if kernel_target == "deltanet":
        _write_npz_like(fixture_dir / "deltanet_reference_state.npz", ["state_token_1", "state_token_1024", "probe_index"])
    path = fixture_dir / f"{kernel_target}_v1.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_npz_like(path: Path, members: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(f"{member}.npy", b"npz-member-placeholder")


def test_fixture_schema_and_weight_version_binding(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, "responses-sdk-adapter-cutover", "deltanet", 64)

    validation = validate_fixture(
        path,
        repo_root=tmp_path,
        expected_family_id="responses-sdk-adapter-cutover",
        expected_kernel_target="deltanet",
        expected_probe_count=64,
        expected_weight_version_id=DEFAULT_WEIGHT_VERSION_ID,
    )

    assert validation.pass_
    assert validation.content_hash == fixture_content_hash(path)

    wrong_weight = validate_fixture(
        path,
        repo_root=tmp_path,
        expected_family_id="responses-sdk-adapter-cutover",
        expected_kernel_target="deltanet",
        expected_probe_count=64,
        expected_weight_version_id="rotated-weight",
    )
    assert "weight_version_id_mismatch" in wrong_weight.errors


def test_fixture_schema_rejects_reference_baseline_drift(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, "responses-sdk-adapter-cutover", "gatedattn", 64)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["generated_against"]["reference_baseline"] = {**REFERENCE_BASELINE, "attention_backend": "flashinfer"}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    validation = validate_fixture(
        path,
        repo_root=tmp_path,
        expected_family_id="responses-sdk-adapter-cutover",
        expected_kernel_target="gatedattn",
        expected_probe_count=64,
        expected_weight_version_id=DEFAULT_WEIGHT_VERSION_ID,
    )

    assert "reference_baseline_mismatch" in validation.errors


def test_fixture_content_hash_binds_every_referenced_blob(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, "codex-provider-rollover", "deltanet", 16)
    original_hash = fixture_content_hash(path)

    (path.parent / "probes_input.jsonl").write_text(
        json.dumps({"probe_index": 0, "family_id": "codex-provider-rollover", "prompt": "changed"}) + "\n",
        encoding="utf-8",
    )
    assert fixture_content_hash(path) != original_hash

    (path.parent / "deltanet_reference_state.npz").unlink()
    with pytest.raises(FileNotFoundError):
        fixture_content_hash(path)


def test_validate_fixture_reports_missing_referenced_blob(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, "codex-provider-rollover", "gatedattn", 16)
    (path.parent / "gatedattn_reference_logits.npz").unlink()

    validation = validate_fixture(
        path,
        repo_root=tmp_path,
        expected_family_id="codex-provider-rollover",
        expected_kernel_target="gatedattn",
        expected_probe_count=16,
        expected_weight_version_id=DEFAULT_WEIGHT_VERSION_ID,
    )

    assert any(error.startswith("referenced_blob_missing:") for error in validation.errors)


def test_validate_fixture_rejects_synthetic_production_artifacts(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, "responses-sdk-adapter-cutover", "deltanet", 64)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["artifact_purpose"] = SYNTHETIC_TEST_ARTIFACT_PURPOSE
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _write_npz_like(path.parent / "deltanet_reference_logits.npz", ["logits", "probe_index", "synthetic_test_placeholder"])

    validation = validate_fixture(
        path,
        repo_root=tmp_path,
        expected_family_id="responses-sdk-adapter-cutover",
        expected_kernel_target="deltanet",
        expected_probe_count=64,
        expected_weight_version_id=DEFAULT_WEIGHT_VERSION_ID,
    )

    assert "production_fixture_declares_synthetic_test_placeholder" in validation.errors
    assert any(error.startswith("reference_blob_contains_synthetic_test_placeholder:") for error in validation.errors)


def test_validate_fixture_rejects_non_npz_reference_blob(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, "codex-provider-rollover", "gatedattn", 16)
    (path.parent / "gatedattn_reference_logits.npz").write_bytes(b"not-a-zip")

    validation = validate_fixture(
        path,
        repo_root=tmp_path,
        expected_family_id="codex-provider-rollover",
        expected_kernel_target="gatedattn",
        expected_probe_count=16,
        expected_weight_version_id=DEFAULT_WEIGHT_VERSION_ID,
    )

    assert "reference_blob_not_zip_npz:gatedattn_reference_logits.npz" in validation.errors


def test_fetch_endpoint_capabilities_records_missing_full_logits_and_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def json(self) -> dict:
            return self._payload

    def fake_get(url: str, headers: dict, timeout: float) -> _Response:
        if url.endswith("/health"):
            return _Response(200)
        if url.endswith("/v1/models"):
            return _Response(200, {"data": [{"id": "qwen3.5-27b"}]})
        if url.endswith("/version"):
            return _Response(200, {"version": "0.19.0"})
        if url.endswith("/server_info?config_format=json"):
            return _Response(
                200,
                {"vllm_config": {"model_config": {"max_logprobs": 20, "logprobs_mode": "raw_logprobs"}}},
            )
        if url.endswith("/openapi.json"):
            return _Response(200, {"paths": {"/collective_rpc": {}}})
        return _Response(404)

    def fake_post(url: str, headers: dict, json: dict, timeout: float) -> _Response:
        assert json["logprobs"] == 100000
        return _Response(
            400,
            {"error": {"message": "Requested sample logprobs of 100000, which is greater than max allowed: 20"}},
        )

    import lumo_flywheel_serving.parity_fixture as parity_fixture

    monkeypatch.setattr(parity_fixture.requests, "get", fake_get)
    monkeypatch.setattr(parity_fixture.requests, "post", fake_post)

    capabilities = fetch_endpoint_capabilities("http://127.0.0.1:8100/v1", api_key="EMPTY", model="qwen3.5-27b")

    assert capabilities["health_ok"]
    assert capabilities["models_ok"]
    assert capabilities["max_logprobs"] == 20
    assert not capabilities["openai_logprobs_full_vocab_available"]
    assert not capabilities["deltanet_state_snapshots_available"]
    assert capabilities["dev_collective_rpc_available"]
    assert "GatedDeltaNetAttention" in p2b_blocked_payload(
        family_id="responses-sdk-adapter-cutover",
        probe_count=64,
        weight_version_id=DEFAULT_WEIGHT_VERSION_ID,
        capabilities=capabilities,
    )["required_vllm_or_model_change"]


def test_p2b_presence_checks_all_families_and_kernels(tmp_path: Path) -> None:
    missing = validate_p2b_fixture_set(tmp_path, expected_weight_version_id=DEFAULT_WEIGHT_VERSION_ID)
    assert not missing["pass"]
    assert len(missing["fixtures"]) == len(P2B_FAMILY_PROBE_COUNTS) * len(KERNEL_TARGETS)

    for family_id, probe_count in P2B_FAMILY_PROBE_COUNTS.items():
        _write_seed_trace(tmp_path, family_id)
        for kernel_target in KERNEL_TARGETS:
            _write_fixture(tmp_path, family_id, kernel_target, probe_count)

    result = validate_p2b_fixture_set(tmp_path, expected_weight_version_id=DEFAULT_WEIGHT_VERSION_ID)
    assert result["pass"], result["errors"]
    assert not result["errors"]
    assert fixture_yaml_path(tmp_path, "responses-sdk-adapter-cutover", "deltanet").is_file()


def _write_debug_logits(path: Path, torch_module: object, *, request_id: str, token: int, values: list[float]) -> None:
    torch_module.save(
        {
            "kind": LOGITS_DEBUG_KIND,
            "request_id": request_id,
            "generated_token_index": token,
            "source_shape": (len(values),),
            "saved_shape": (len(values),),
            "logits_is_truncated": False,
            "logits": torch_module.tensor(values, dtype=torch_module.float32),
        },
        path,
    )


def _write_debug_state(path: Path, torch_module: object, *, request_id: str, token: int, values: list[float]) -> None:
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
                        "saved_shape": (len(values),),
                        "saved_dtype": "torch.float32",
                        "tensor": torch_module.tensor(values, dtype=torch_module.float32),
                    }
                ]
            },
        },
        path,
    )


def test_aggregate_real_debug_exports_into_npz_companions(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    artifacts: list[DebugProbeArtifacts] = []
    for probe_index in range(2):
        request_id = f"cmpl-live-{probe_index}"
        probe_dir = tmp_path / f"probe_{probe_index:06d}"
        probe_dir.mkdir()
        logits_path = probe_dir / f"logits_req_{request_id}_tok_000001.pt"
        state_1 = probe_dir / f"state_req_{request_id}_tok_000001.pt"
        state_1024 = probe_dir / f"state_req_{request_id}_tok_001024.pt"
        _write_debug_logits(logits_path, torch, request_id=request_id, token=1, values=[probe_index, 2.0, 3.0])
        _write_debug_state(state_1, torch, request_id=request_id, token=1, values=[1.0, float(probe_index)])
        _write_debug_state(state_1024, torch, request_id=request_id, token=1024, values=[1024.0, float(probe_index)])
        artifacts.append(
            DebugProbeArtifacts(
                probe_index=probe_index,
                request_id=request_id,
                logits_paths=(logits_path,),
                state_paths=(state_1, state_1024),
            )
        )

    result = write_debug_export_npz_companions(
        fixture_dir=tmp_path / "parity_fixture",
        kernel_target="deltanet",
        probe_artifacts=artifacts,
        expected_probe_count=2,
    )

    import numpy as np

    logits = np.load(result.logits_path)
    state = np.load(result.state_path)
    assert logits["probe_index"].tolist() == [0, 1]
    assert logits["probe_000000_logits"].shape == (1, 3)
    assert state["state_token_1"].shape == (2, 2)
    assert state["state_token_1024"].shape == (2, 2)
    summary = summarize_debug_export_pt(artifacts[0].logits_paths[0])
    assert summary["kind"] == LOGITS_DEBUG_KIND
    assert "synthetic_test_placeholder" not in logits.files


def test_reproducibility_check_rejects_debug_tensor_drift(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    runs: list[list[DebugProbeArtifacts]] = []
    for run_index, final_logit in enumerate([1.0, 2.0, 9.0], start=1):
        request_id = f"cmpl-run-{run_index}"
        probe_dir = tmp_path / f"run_{run_index:02d}" / "probe_000000"
        probe_dir.mkdir(parents=True)
        logits_path = probe_dir / f"logits_req_{request_id}_tok_000001.pt"
        _write_debug_logits(logits_path, torch, request_id=request_id, token=1, values=[0.0, final_logit])
        runs.append(
            [
                DebugProbeArtifacts(
                    probe_index=0,
                    request_id=request_id,
                    logits_paths=(logits_path,),
                )
            ]
        )

    with pytest.raises(ValueError, match="fixture_reference_nondeterministic"):
        assert_debug_capture_runs_reproduce(runs=runs, kernel_target="gatedattn")
