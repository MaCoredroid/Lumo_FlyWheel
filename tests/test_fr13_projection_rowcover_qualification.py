from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load("fr13_projection_rowcover_b1_pass")
    candidate_bytes = b"projection-rowcover-pair\n"
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(candidate_bytes)
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    monkeypatch.setattr(
        module.binary, "STATIC_PERSISTENT_CANDIDATE_SIZE", len(candidate_bytes)
    )
    monkeypatch.setattr(
        module.binary, "STATIC_PERSISTENT_CANDIDATE_SHA256", candidate_sha256
    )
    patch = tmp_path / "patch.py"
    patch.write_bytes(b"projection patch\n")
    monkeypatch.setattr(
        module,
        "PATCH_SOURCE_SHA256",
        hashlib.sha256(patch.read_bytes()).hexdigest(),
    )
    blocks = tmp_path / "blocks.json"
    blocks.write_bytes(b"{}\n")
    block_sha256 = hashlib.sha256(blocks.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "DRAFT_VOCAB_BLOCKS_SHA256", block_sha256)
    live = tmp_path / "live.json"
    payload = {
        "schema": module.LIVE_SCHEMA,
        "status": "pass",
        "run_classification": "one_real_swe_verified_b1_k64_root_byte_diagnostic",
        "acceptance_valid": False,
        "task_count": 1,
        "task_ids": list(module.EXPECTED_TASK_IDS),
        "task_marker": module.EXPECTED_TASK_MARKER,
        "qualification_profile": "k64_root",
        "draft_vocab_root": 1,
        "draft_vocab_k": 65_536,
        "draft_vocab_blocks": module.DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
        "draft_vocab_blocks_sha256": block_sha256,
        "mandatory_weight_bytes": module.floor.FIXED32_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": module.floor.FIXED32_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": module.EXPECTED_SLO_CAP_MS,
        "comparator_timing_eligible": False,
        "batch_size": 1,
        "concurrency": 1,
        "fixed_rows": 32,
        "candidate": module.CANDIDATE_SELECTOR,
        "diagnostic_selector": module.DIAGNOSTIC_SELECTOR,
        "served_result": "stock",
        "production_enabled": False,
        "comparisons": 320,
        "observed_m_values": [32],
        "observed_projection_nk": [
            list(shape) for shape in module.EXPECTED_PROJECTION_NK
        ],
        "mismatching_comparisons": 0,
        "differing_bytes": 0,
        "candidate_family": "projection_rowcover_pair",
        "candidate_sha256": candidate_sha256,
        "candidate_bytes": len(candidate_bytes),
        "patch_source_sha256": module.PATCH_SOURCE_SHA256,
        "vllm_base_commit": module.VLLM_BASE_COMMIT,
        "patched_dispatch_sha256": module.PATCHED_DISPATCH_SHA256,
        "source_commit": "a" * 40,
        "binary_attestation_sha256": "b" * 64,
        "real_task_arm_sha256": "c" * 64,
        "container_env_sha256": "d" * 64,
        "errors": [],
    }
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    return module, candidate, patch, blocks, live, payload


def test_b1_k64_pass_issues_and_verifies_exact_rowcover_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch, blocks, live, _ = _fixture(tmp_path, monkeypatch)
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()
    sidecar = tmp_path / "sidecar.json"

    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch,
        "a" * 40,
        draft_vocab_blocks=blocks,
    )
    verified = module.verify_sidecar(
        sidecar,
        hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        candidate,
        patch,
        draft_vocab_blocks=blocks,
    )

    assert verified == issued
    assert issued["candidate_family"] == "projection_rowcover_pair"
    assert issued["qualification_profile"] == "k64_root"
    assert issued["qualified_fixed_rows"] == 32
    assert issued["qualified_draft_vocab_root"] == 1
    assert issued["qualified_draft_vocab_k"] == 65_536


def test_b1_k64_pass_rejects_profile_or_shape_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch, blocks, live, payload = _fixture(tmp_path, monkeypatch)
    for field, value in (
        ("qualification_profile", "full_vocab"),
        ("draft_vocab_root", 0),
        ("draft_vocab_k", 0),
        ("fixed_rows", 128),
        ("observed_m_values", [128]),
    ):
        tampered = dict(payload)
        tampered[field] = value
        live.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="ascii")
        with pytest.raises(module.QualificationError, match=field):
            module.validate_live_result(
                live,
                hashlib.sha256(live.read_bytes()).hexdigest(),
                candidate,
                patch,
                draft_vocab_blocks=blocks,
            )


def test_static_qualifier_binds_both_selectors_and_zero_stack_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("fr13_projection_rowcover_static_qualify")
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(b"pair\n")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    for name in (
        "B4_M128_CANDIDATE_SHA256",
        "STATIC_PERSISTENT_CANDIDATE_SHA256",
    ):
        monkeypatch.setattr(module.binary, name, digest)
    for name in ("B4_M128_CANDIDATE_SIZE", "STATIC_PERSISTENT_CANDIDATE_SIZE"):
        monkeypatch.setattr(module.binary, name, candidate.stat().st_size)
    patch = tmp_path / "patch.py"
    patch.write_bytes(b"patch\n")
    patch_digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    monkeypatch.setattr(module.b1_qualification, "PATCH_SOURCE_SHA256", patch_digest)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    resource_header = (
        "candidate\toutput_type\ttile_m\ttile_n\ttile_k\tschedule\tregisters\t"
        "stack_bytes\tlocal_bytes\tshared_bytes\tconstant_0_bytes\n"
    )
    resource_rows = (
        "b1_static_persistent\tfp16\t128\t32\t128\t"
        "cooperative_static_persistent\t168\t0\t0\t1024\t2688\n"
        "b1_static_persistent\tbf16\t128\t32\t128\t"
        "cooperative_static_persistent\t168\t0\t0\t1024\t2688\n"
        "b4_persistent_m128\tfp16\t128\t128\t128\t"
        "cooperative_persistent\t168\t0\t0\t1024\t2560\n"
        "b4_persistent_m128\tbf16\t128\t128\t128\t"
        "cooperative_persistent\t168\t0\t0\t1024\t2560\n"
    )
    (evidence / "kernel_resources.tsv").write_text(
        resource_header + resource_rows,
        encoding="ascii",
    )
    (evidence / "stock_equivalence.txt").write_text(
        "status=pass\n"
        f"candidate_binary_sha256={digest}\n"
        "reference_stock_record_count=6\n"
        "candidate_stock_record_count=6\n"
        "matched_stock_record_count=6\n"
        "missing_stock_record_count=0\n"
        "strong_dynamic_reference_count=873\n"
        "strong_dynamic_candidate_count=873\n"
        "strong_dynamic_comparison=exact\n",
        encoding="ascii",
    )
    (evidence / "candidate.json").write_text(
        json.dumps(
            {
                "source": {
                    "patch_sha256": patch_digest,
                    "patched_dispatch_sha256": (
                        module.b1_qualification.PATCHED_DISPATCH_SHA256
                    ),
                },
                "build": {
                    "binary_sha256": digest,
                    "binary_bytes": candidate.stat().st_size,
                    "binary_mode": "0555",
                },
            }
        ),
        encoding="ascii",
    )

    result = module.qualify(candidate, patch, evidence)

    assert result["status"] == "pass"
    assert result["resource_records"] == 4
    assert result["zero_stack_records"] == 4
    assert result["zero_local_records"] == 4
    assert result["requires_fresh_b1_k64_byte_gate"] is True
    assert result["requires_fresh_b4_k64_exact4_byte_gate"] is True


def test_projection_rowcover_qualifiers_are_runtime_manifest_inputs() -> None:
    manifest = (SCRIPTS / "fr13_runtime_manifest.py").read_text(encoding="utf-8")

    assert '"scripts/fr13_projection_rowcover_b1_pass.py"' in manifest
    assert '"scripts/fr13_projection_rowcover_static_qualify.py"' in manifest
