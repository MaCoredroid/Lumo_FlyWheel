from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_b1_m128_cooperative_target_sfwd_real_gate_a8a904ed6_20260805"
)


def _json(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text())


def test_both_cooperative_target_passes_are_exact_and_non_timing() -> None:
    manifest = _json("manifest.json")
    assert manifest["source_commit"] == (
        "a8a904ed6c27a6338d43151038c155ebb76e3656"
    )
    assert manifest["packaging_base_commit"] != manifest["source_commit"]

    for name in ("target_standalone_pass.json", "target_combined_pass.json"):
        target = _json(name)
        assert target["schema"] == (
            "fr13.fixed32.cutlass_identity_wide256_fullgrid_b1_k64_root_live_gate.v1"
        )
        assert target["status"] == "pass"
        assert target["source_commit"] == manifest["source_commit"]
        assert target["candidate"] == "identity_wide256_fullgrid_b1"
        assert target["comparisons"] == target["comparison_call_limit"] == 320
        assert target["mismatching_comparisons"] == 0
        assert target["differing_bytes"] == 0
        assert target["errors"] == []
        assert target["fixed_rows"] == 32
        assert (target["draft_vocab_k"], target["draft_vocab_root"]) == (
            65536,
            1,
        )
        assert target["task_ids"] == ["astropy__astropy-12907"]
        assert target["served_result"] == "stock"
        assert target["production_enabled"] is False
        assert target["timing_eligible"] is False
        assert target["acceptance_valid"] is False


def test_combined_qrow16_and_sfwd_contracts_are_exact() -> None:
    manifest = _json("manifest.json")
    target = _json("target_combined_pass.json")
    qrow_pass = _json("qrow16_production_pass.json")
    qrow = _json("qrow16_engagement.json")
    sfwd = _json("sfwd_live_pass.json")
    summary = _json("sfwd_gate_summary.json")

    assert qrow_pass["schema"] == (
        "fr13.fixed32.fa2_qrow16_production_pass.v1"
    )
    assert qrow_pass["status"] == "PASS"
    assert qrow_pass["instance_id"] == "astropy__astropy-12907"
    assert qrow["status"] == "ENGAGED"
    assert qrow["runtime_mode"] == "EAGER"
    assert qrow["calls_observed"] == qrow["layer_count"] == 16
    assert qrow["sfwd_conv_postprep_byte_ab"] is True
    assert qrow["sfwd_state_fusion_production"] is False

    assert sfwd["schema"] == "fr13.fixed32.sfwd_conv_postprep.live_pass.v1"
    assert sfwd["status"] == "byte_pass_source_only"
    assert sfwd["source_commit"] == manifest["source_commit"]
    assert sfwd["comparisons"] == 48 * 7 == 336
    assert sfwd["layer_count"] == len(sfwd["layers"]) == 48
    assert sfwd["mismatches"] == sfwd["differing_bytes"] == 0
    assert sfwd["candidate_decision"] == "shadow_only"
    assert sfwd["candidate_returned"] is False
    assert sfwd["reference_always_served"] is True
    assert sfwd["reference_decision"] == "serve_incumbent"
    assert sfwd["timing_eligible"] is False

    raw = manifest["normalized_copies"]
    assert summary["combined_target_live_pass_sha256"] == raw[
        "target_combined_pass.json"
    ]["source_raw_sha256"]
    assert summary["qrow16_sidecar_sha256"] == raw[
        "qrow16_production_pass.json"
    ]["source_raw_sha256"]
    assert summary["qrow16_capture_sha256"] == raw[
        "qrow16_engagement.json"
    ]["source_raw_sha256"]
    assert summary["live_pass_sha256"] == raw["sfwd_live_pass.json"][
        "source_raw_sha256"
    ]
    assert summary["source_manifest_sha256"] == raw[
        "sfwd_source_manifest.json"
    ]["source_raw_sha256"]
    assert summary["status"] == "pass"
    assert summary["no_fallback"] is True
    assert summary["reference_returned"] is True
    assert summary["production_enabled"] is False
    assert summary["timing_eligible"] is False
    assert target["status"] == "pass"


def test_sfwd_native_and_derived_record_counts_are_not_conflated() -> None:
    manifest = _json("manifest.json")
    records = _json("sfwd_records_summary.json")
    summary = _json("sfwd_gate_summary.json")
    native = records["source_native"]
    window = records["derived_final_40_attempt_window"]

    assert records["source_commit"] == manifest["source_commit"]
    assert records["source_raw_jsonl_sha256"] == summary["records_sha256"]
    assert native == {
        "attempt_first": 1,
        "attempt_last": 417,
        "attempts": 417,
        "records": 20016,
        "layer_count": 48,
        "records_per_attempt_min": 48,
        "records_per_attempt_max": 48,
        "surfaces_per_record_min": 7,
        "surfaces_per_record_max": 7,
        "surface_comparisons": 140112,
        "bad_records": 0,
    }
    assert window["derived"] is True
    assert window["attempts"] == 40
    assert window["attempt_last"] - window["attempt_first"] + 1 == 40
    assert window["expected_records"] == 40 * 48 == 1920
    assert window["observed_records"] == 1920
    assert window["zero_mismatch_records"] == 1920
    assert window["bad_records"] == 0
    assert records["live_pass_native"]["comparisons"] == 336


def test_both_real_tasks_resolved_and_manifests_stayed_fixed() -> None:
    for prefix in ("standalone", "combined"):
        health = _json(f"{prefix}_health.json")
        task = _json(f"{prefix}_task_summary.json")
        assert health["swe_orchestrator_rc"] == 0
        assert health["tasks"] == [
            {
                "instance_id": "astropy__astropy-12907",
                "codex_elapsed_s": health["tasks"][0]["codex_elapsed_s"],
                "codex_timed_out": False,
                "patch_bytes": 504,
                "verdict": "resolved",
            }
        ]
        assert task["instances_total"] == 1
        assert task["verdict_counts"] == {"resolved": 1}
        assert task["failure_mode_counts"] == {"tests_passed": 1}
        assert task["resolved_rate"] == 1.0
        assert task["fixed32_run_classification"] == {
            "run_classification": "b1_diagnostic",
            "gate_eligible": False,
            "floor_acceptance_eligible": False,
        }

    digests = _json("launch_end_manifest_digests.json")
    for run in ("standalone_target", "combined_qrow16_target_sfwd"):
        for kind in ("runtime", "external"):
            identity = digests[run][kind]
            assert identity["byte_identical"] is True
            assert identity["launch_sha256"] == identity["end_sha256"]
    sfwd_source = digests["combined_qrow16_target_sfwd"]["sfwd_source"]
    assert sfwd_source["byte_identical"] is True
    assert len(
        {
            sfwd_source["launch_sha256"],
            sfwd_source["installed_sha256"],
            sfwd_source["end_sha256"],
        }
    ) == 1


def test_normalized_copies_and_historical_source_are_bound() -> None:
    manifest = _json("manifest.json")
    for name, identity in manifest["normalized_copies"].items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == (
            identity["normalized_sha256"]
        )

    source = _json("sfwd_source_manifest.json")
    assert source["source_commit"] == manifest["source_commit"]
    for relative, identity in source["files"].items():
        historical = subprocess.check_output(
            ["git", "show", f"{source['source_commit']}:{relative}"], cwd=ROOT
        )
        assert len(historical) == identity["bytes"]
        assert hashlib.sha256(historical).hexdigest() == identity["sha256"]


def test_artifact_is_sanitized_small_and_checksum_complete() -> None:
    expected = {
        "README.md",
        "SHA256SUMS",
        "combined_health.json",
        "combined_task_summary.json",
        "launch_end_manifest_digests.json",
        "manifest.json",
        "qrow16_engagement.json",
        "qrow16_production_pass.json",
        "sfwd_gate_summary.json",
        "sfwd_live_pass.json",
        "sfwd_records_summary.json",
        "sfwd_source_manifest.json",
        "standalone_health.json",
        "standalone_task_summary.json",
        "target_combined_pass.json",
        "target_standalone_pass.json",
    }
    assert {path.name for path in ARTIFACT.iterdir()} == expected
    assert not any(path.suffix in {".jsonl", ".log", ".diff"} for path in ARTIFACT.iterdir())
    assert sum(path.stat().st_size for path in ARTIFACT.iterdir()) < 100_000

    published = "\n".join(
        path.read_text()
        for path in ARTIFACT.iterdir()
        if path.name != "SHA256SUMS"
    )
    for forbidden in (
        "/home/",
        "/tmp/",
        '"prompt":',
        '"response":',
        '"patch":',
        '"container_id":',
        '"pid":',
    ):
        assert forbidden not in published

    checksums = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    assert set(checksums) == expected - {"SHA256SUMS"}
    for name, digest in checksums.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
