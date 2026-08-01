from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
ARTIFACT = (
    REPO
    / "results"
    / "fr13_fixed32_taw_current_b1_live_pass_20260731T164000Z"
)
LIVE_PASS_SHA256 = (
    "4bfb971f4e9808069d67c4896d9664ecee19542767867157a21f66b0c22f79e5"
)

SPEC = importlib.util.spec_from_file_location(
    "fr13_taw_current_b1_artifact_kernel",
    REPO / "scripts" / "fr13_device_multidraft_kernel.py",
)
assert SPEC is not None and SPEC.loader is not None
kernel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kernel)


def _json(name: str) -> dict[str, object]:
    return json.loads((ARTIFACT / name).read_text(encoding="ascii"))


def test_pre_tail23_live_pass_is_exact_but_current_validator_rejects() -> None:
    path = ARTIFACT / "live_pass.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == LIVE_PASS_SHA256

    with pytest.raises(RuntimeError, match="different candidate/source"):
        kernel._fr13_fixed32_taw_native_production_pass(
            path=str(path),
            expected_mode="hydra27_fixed32",
            expected_batch=1,
        )


def test_artifact_is_honest_about_correctness_only_scope() -> None:
    manifest = _json("manifest.json")
    verification = _json("verification.json")

    assert manifest["status"] == "pass"
    assert manifest["executed_source_commit"] == (
        "c8d8bda914af632741d3f2bd9ff0980256b3e897"
    )
    assert manifest["classification"] == {
        "floor_acceptance_eligible": False,
        "gate_eligible": False,
        "performance_evidence": False,
        "run_classification": "b1_diagnostic",
    }
    assert manifest["coverage"]["batches"] == [1]
    assert manifest["production_contract"]["separate_sidecar_issued"] is False
    assert manifest["production_contract"]["separate_sidecar_supported"] is False

    checks = verification["checks"]
    assert checks["complete_event_count"] == 968
    assert checks["root_check_pass_count"] == 968
    assert checks["nonzero_mismatch_log_records"] == 0
    assert checks["production_validator_b1"] is True
    assert checks["production_validator_b4"] is False
    assert verification["arm_provenance"]["archived_marker_still_present"] is True


def test_internal_checksums_cover_every_non_checksum_artifact_file() -> None:
    checksum_path = ARTIFACT / "SHA256SUMS"
    expected = {}
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest

    actual_names = sorted(
        path.name for path in ARTIFACT.iterdir() if path.name != "SHA256SUMS"
    )
    assert sorted(expected) == actual_names
    for name, digest in expected.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
