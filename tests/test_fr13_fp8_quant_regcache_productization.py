from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path("scripts").resolve()))
runtime = importlib.import_module("fr13_fp8_quant_regcache_runtime")
qualification = importlib.import_module("fr13_fp8_quant_regcache_pass")


def _candidate(path: Path) -> tuple[str, int]:
    raw = bytearray(b"\x7fELF")
    for token in runtime.REQUIRED_BINARY_TOKENS:
        raw.extend(token + b"\0")
    raw.extend(b"x" * (70 * 1024 - len(raw)))
    path.write_bytes(raw)
    os.chmod(path, 0o755)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _record(invocation: int) -> dict[str, object]:
    return {
        "schema": qualification.RECORD_SCHEMA,
        "invocation": invocation,
        "target_forward_ordinal": invocation // 128,
        "call_in_target_forward": invocation % 128,
        "task_marker": qualification.TASK_MARKER,
        "rows": 32,
        "k": 5120,
        "group_size": 128,
        "groups": 1280,
        "groups_per_cta": 16,
        "ctas": 80,
        "threads_per_cta": 256,
        "output_bytes": 163840,
        "output_byte_equal": True,
        "output_mismatch_count": 0,
        "output_first_mismatch": None,
        "scale_bytes": 5120,
        "scale_byte_equal": True,
        "scale_mismatch_count": 0,
        "scale_first_mismatch": None,
        "scale_layout": "column_major_fp32_32x40_stride_1_32",
        "stock_served": True,
        "comparison_sampled": False,
    }


def _records(path: Path, count: int = 128) -> None:
    path.write_text(
        "".join(
            json.dumps(_record(index), separators=(",", ":"), sort_keys=True)
            + "\n"
            for index in range(count)
        ),
        encoding="ascii",
    )


def test_runtime_binary_is_pinned_and_default_off_install_is_distinct(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.so"
    digest, size = _candidate(candidate)
    assert runtime.validate_binary(candidate, digest, size)["bytes"] == size
    patch_source = tmp_path / "patch.py"
    patch_source.write_text("candidate patch\n", encoding="ascii")
    destination = tmp_path / "vllm" / "_C_stable_libtorch.abi3.so"
    attestation = tmp_path / "binary.json"
    payload = runtime.install_binary(
        source=candidate,
        destination=destination,
        attestation=attestation,
        selector="0",
        expected_sha256=digest,
        patch_source=patch_source,
        source_commit="a" * 40,
    )
    assert payload["selector"] == "0"
    assert payload["production_enabled"] is False
    assert payload["diagnostic_enabled"] is False
    assert payload["smoke_load_passed"] is False
    assert payload["production_sidecar_sha256"] is None
    assert destination.read_bytes() == candidate.read_bytes()
    assert oct(destination.stat().st_mode & 0o777) == "0o555"


def test_record_gate_requires_every_complete_target_forward(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    _records(records)
    loaded, raw = qualification.load_records(records)
    assert len(loaded) == 128
    assert raw.endswith(b"\n")

    incomplete = tmp_path / "incomplete.jsonl"
    _records(incomplete, 127)
    with pytest.raises(ValueError, match="complete 128-call target forwards"):
        qualification.load_records(incomplete)

    mismatched = tmp_path / "mismatched.jsonl"
    rows = [_record(index) for index in range(128)]
    rows[91]["scale_byte_equal"] = False
    mismatched.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="ascii"
    )
    with pytest.raises(ValueError, match="record 91 contract drifted"):
        qualification.load_records(mismatched)


def test_pass_sidecar_binds_binary_source_and_complete_raw_gate(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.so"
    digest, size = _candidate(candidate)
    patch_source = tmp_path / "patch.py"
    patch_source.write_text("candidate patch\n", encoding="ascii")
    source_commit = "b" * 40
    live = {
        "schema": qualification.LIVE_SCHEMA,
        "status": "PASS",
        "candidate_sha256": digest,
        "candidate_bytes": size,
        "source_commit": source_commit,
        "task_ids": [qualification.INSTANCE_ID],
        "comparisons": 128,
        "target_forwards": 1,
        "output_mismatching_bytes": 0,
        "scale_mismatching_bytes": 0,
        "stock_served": True,
        "comparison_sampled": False,
    }
    live_path = tmp_path / "live.json"
    live_path.write_bytes(qualification.canonical_bytes(live) + b"\n")
    sidecar_path = tmp_path / "pass.json"
    issued = qualification.issue_sidecar(
        live_result=live_path,
        expected_live_sha256=hashlib.sha256(live_path.read_bytes()).hexdigest(),
        candidate_so=candidate,
        expected_candidate_sha256=digest,
        patch_source=patch_source,
        qualified_source_commit=source_commit,
        out=sidecar_path,
    )
    sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    verified = qualification.verify_sidecar(
        sidecar_path=sidecar_path,
        expected_sidecar_sha256=sidecar_sha,
        candidate_so=candidate,
        expected_candidate_sha256=digest,
        patch_source=patch_source,
    )
    assert issued == verified
    assert verified["qualified_invocations"] == 128
    patch_source.write_text("changed patch\n", encoding="ascii")
    with pytest.raises(ValueError, match="contract drifted"):
        qualification.verify_sidecar(
            sidecar_path=sidecar_path,
            expected_sidecar_sha256=sidecar_sha,
            candidate_so=candidate,
            expected_candidate_sha256=digest,
            patch_source=patch_source,
        )


def test_launch_route_installs_the_same_binary_for_control_and_candidate() -> None:
    launcher = Path("scripts/fr13_launch_forked_fa2_tree_server.sh").read_text()
    bigdenom = Path("scripts/fr13_bigdenom_swe_serve_variant.sh").read_text()
    gate = Path("scripts/fr13_run_b1_fp8_quant_regcache_live_gate.sh").read_text()
    timing = Path("scripts/fr13_run_b1_fp8_quant_regcache_timing.sh").read_text()

    assert "0|byte_ab|1" in launcher
    assert "/tmp/fr13_fp8_quant_regcache.abi3.so" in launcher
    assert "fr13_fp8_quant_regcache_runtime.py install" in launcher
    assert launcher.count("--smoke-load") == 2
    assert "fr13_fp8_quant_regcache_pass.py verify" in launcher
    assert "requires isolated Hydra27 physical32 K64/root1 B1" in launcher
    assert "FR13_FIXED32_B1_FP8_QUANT_REGCACHE:-0}" in bigdenom
    assert "--fixed32-cutlass-real-event-arm" in bigdenom

    assert "subset_b1_diagnostic_one.json" not in timing
    assert "subset_b4_four.json" in timing
    assert 'run_arm "$STOCK_ARM" 0' in timing
    assert 'run_arm "$CANDIDATE_ARM" 1' in timing
    assert "FR13_SFWD_GPU_TIMER=1" in timing
    assert "FR13_DFWD_GPU_TIMER=1" in timing
    assert "FR13_CFWD_GPU_TIMER=1" in timing
    assert "measured_tps_fullstep_wall" in timing
    assert "formal_floor_acceptance_eligible\": False" in timing

    assert "subset_b1_diagnostic_one.json" in gate
    assert "FR13_FIXED32_B1_FP8_QUANT_REGCACHE=byte_ab" in gate
    assert "fr13_fp8_quant_regcache_pass.py qualify" in gate
    assert "FP8_QUANT_PASS" not in gate
    assert "spec_speed_probe" not in gate + timing

    builder = Path("scripts/fr13_build_fp8_quant_regcache_sm121a.sh").read_text()
    assert "VLLM_SOURCE_COMMIT" in builder
    assert "fe9c3d6c5f66c873d196800384ed6880687b9e52" in builder
    assert "d655c46ab6ba497f83a62d1498ca3affb7344b163a3044754f404009ba00ae16" in builder
    assert "arch=compute_121a,code=sm_121a" in builder
    assert "BASE_OBJECT_ROOT" in builder
    assert "CUTLASS_SOURCE_COMMIT" in builder
    assert "PINNED_BASE_OBJECTS" in builder
    assert builder.count(" -c ") == 2
    assert "--gpus" not in builder
