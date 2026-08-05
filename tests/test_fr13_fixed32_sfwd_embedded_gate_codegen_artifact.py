from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/fr13_fixed32_sfwd_embedded_gate_cta_sm121a_20260805"
SUMMARY = json.loads((ARTIFACT / "codegen_summary.json").read_text())


def test_embedded_gate_artifact_checksums_and_verifier_are_exact() -> None:
    for line in (ARTIFACT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == expected
    subprocess.run(
        ["python3", str(ARTIFACT / "verify_codegen_summary.py")],
        check=True,
        capture_output=True,
        text=True,
    )


def test_embedded_gate_artifact_binds_historical_candidate_sources() -> None:
    candidate = SUMMARY["revisions"]["candidate"]
    assert candidate == "086da781207322601fc4876f9f6d69292a4a71a1"
    for relative, expected in SUMMARY["source_hashes"]["candidate"].items():
        historical = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{candidate}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == expected


def test_embedded_gate_artifact_records_work_reduction_without_runtime_claim() -> None:
    assert SUMMARY["status"] == "PASS"
    assert SUMMARY["static_gate_pass"] is True
    assert SUMMARY["offline_only"] is True
    assert SUMMARY["gpu_api_used"] is False
    assert SUMMARY["runtime_byte_correctness"] is False
    assert SUMMARY["timing_claim"] is False
    assert SUMMARY["performance_claim"] is False
    assert SUMMARY["floor_acceptance_eligible"] is False
    assert SUMMARY["work_deltas"]["b1"]["ctas_whole_batch_all_48_layers"] == -192
    assert SUMMARY["work_deltas"]["b4"]["ctas_whole_batch_all_48_layers"] == -768
    assert SUMMARY["work_deltas"]["b1"]["requested_gate_bytes_whole_batch_all_48_layers"] == 0
    assert SUMMARY["work_deltas"]["b4"]["requested_gate_bytes_whole_batch_all_48_layers"] == 0
    assert "not_measured_dram_or_hbm" in SUMMARY["traffic_classification"]
