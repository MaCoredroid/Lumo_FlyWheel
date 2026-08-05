from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "results" / "fr13_b1_gate_b_m128_directgrid_pass_20260805"
)


def _json(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text())


def test_target_and_sfwd_gates_pass_exactly() -> None:
    manifest = _json("manifest.json")
    target = _json("target_gate.json")
    comparator = _json("comparator_summary.json")
    sfwd = _json("sfwd_gate.json")

    assert manifest["status"] == "PASS"
    assert target["status"] == "pass"
    assert target["source_commit"] == manifest["source_commit"]
    assert target["comparisons"] == 320
    assert target["mismatching_comparisons"] == 0
    assert target["errors"] == []
    assert comparator["records"] == 320
    assert comparator["unique_invocations"] == 320
    assert comparator["all_byte_equal"] is True
    assert comparator["mismatch_sum"] == 0
    assert sum(shape["comparisons"] for shape in comparator["shapes"]) == 320
    assert sfwd["status"] == "pass"
    assert sfwd["source_commit"] == manifest["source_commit"]
    assert sfwd["layer_count"] == 48
    assert sfwd["decision_exact"] is True
    assert sfwd["no_fallback"] is True
    assert sfwd["candidate_returned"] is False
    assert sfwd["reference_returned"] is True


def test_qrow_engaged_and_real_task_resolved() -> None:
    manifest = _json("manifest.json")
    qrow = _json("qrow_engagement.json")
    task = _json("task_summary.json")

    assert qrow["status"] == "ENGAGED"
    assert qrow["runtime_mode"] == "EAGER"
    assert qrow["batch_size"] == 1
    assert qrow["draft_vocab_k"] == 65536
    assert qrow["draft_vocab_root"] == 1
    assert qrow["layer_count"] == 16
    assert qrow["calls_observed"] == 16
    assert qrow["sfwd_state_fusion_production"] is False
    assert qrow["sfwd_conv_postprep_byte_ab"] is True
    assert task["instances_total"] == 1
    assert task["verdict_counts"] == {"resolved": 1}
    assert task["resolved_rate"] == 1.0
    assert manifest["claims"]["timing_eligible"] is False
    assert manifest["claims"]["hardware_floor_evidence"] is False


def test_source_manifest_is_bound_to_frozen_source() -> None:
    manifest = _json("manifest.json")
    source = _json("source_manifest.json")
    source_path = ARTIFACT / manifest["source_identity"]["path"]

    assert source["source_commit"] == manifest["source_commit"]
    assert source["schema"] == manifest["source_identity"]["schema"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        manifest["source_identity"]["sha256"]
    )
    for relative, identity in source["files"].items():
        content = subprocess.check_output(
            ["git", "show", f"{source['source_commit']}:{relative}"], cwd=ROOT
        )
        assert len(content) == identity["bytes"]
        assert hashlib.sha256(content).hexdigest() == identity["sha256"]


def test_artifact_is_sanitized_and_checksum_complete() -> None:
    expected = {
        "README.md",
        "SHA256SUMS",
        "comparator_summary.json",
        "manifest.json",
        "qrow_engagement.json",
        "sfwd_gate.json",
        "source_manifest.json",
        "target_gate.json",
        "task_summary.json",
    }
    assert {path.name for path in ARTIFACT.iterdir()} == expected

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
