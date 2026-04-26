from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lumo_flywheel_serving.parity_fixture import (
    DEFAULT_WEIGHT_VERSION_ID,
    KERNEL_TARGETS,
    P2B_FAMILY_PROBE_COUNTS,
    REFERENCE_BASELINE,
    fixture_content_hash,
    fixture_payload,
    fixture_yaml_path,
    validate_fixture,
    validate_p2b_fixture_set,
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
    (fixture_dir / f"{kernel_target}_reference_logits.npz").write_bytes(f"{kernel_target}-logits".encode("ascii"))
    payload = fixture_payload(
        family_id=family_id,
        kernel_target=kernel_target,
        probe_count=probe_count,
        weight_version_id=DEFAULT_WEIGHT_VERSION_ID,
        vllm_version="0.19.0",
        generated_at="2026-04-26T00:00:00Z",
    )
    if kernel_target == "deltanet":
        (fixture_dir / "deltanet_reference_state.npz").write_bytes(b"deltanet-state")
    path = fixture_dir / f"{kernel_target}_v1.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


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

