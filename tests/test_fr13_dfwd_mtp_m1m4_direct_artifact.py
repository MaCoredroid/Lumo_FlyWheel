from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_dfwd_mtp_m1m4_direct_sm121a_20260803"
)


def _json(name: str) -> dict[str, object]:
    return json.loads((ARTIFACT / name).read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_at_revision(revision: str, relative: str) -> str:
    historical = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{revision}:{relative}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return hashlib.sha256(historical).hexdigest()


def _binary_module():
    path = ROOT / "scripts" / "fr13_cutlass_wave_binary.py"
    spec = importlib.util.spec_from_file_location("fr13_mtp_binary", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_binds_source_and_diagnostic_binary() -> None:
    manifest = _json("manifest.json")
    source = manifest["source"]
    outputs = manifest["outputs"]
    assert isinstance(source, dict)
    assert isinstance(outputs, dict)

    revision = source["source_commit"]
    assert isinstance(revision, str)
    assert _sha256_at_revision(
        revision, "scripts/fr13_patch_cutlass_fixed32_wave.py"
    ) == source["patch_source_sha256"]
    assert _sha256_at_revision(
        revision, "scripts/fr13_cutlass_wave_binary.py"
    ) == source["binary_verifier_sha256"]

    binary = outputs["candidate_binary"]
    assert isinstance(binary, dict)
    module = _binary_module()
    assert binary["sha256"] == module.MTP_M1M4_DIRECT_CANDIDATE_SHA256
    assert binary["bytes"] == module.MTP_M1M4_DIRECT_CANDIDATE_SIZE
    assert module.MTP_M1M4_DIRECT_SELECTORS == {"mtp_m1m4_direct_byte_ab"}
    assert "mtp_m1m4_direct_byte_ab" not in module.PRODUCTION_SELECTORS


def test_artifact_is_default_off_stock_served_and_non_accepting() -> None:
    manifest = _json("manifest.json")
    candidate = manifest["candidate"]
    assert isinstance(candidate, dict)

    assert manifest["acceptance_valid"] is False
    assert manifest["performance_claim"] is False
    assert manifest["gpu_runtime_used"] is False
    assert manifest["docker_used"] is False
    assert candidate["diagnostic_selector"] == "mtp_m1m4_direct_byte_ab"
    assert candidate["production_selector"] is None
    assert candidate["default_enabled"] is False
    assert candidate["qualification_profile"] == "k64_root"
    assert candidate["served_output"] == "stock"
    assert candidate["unarmed_behavior"] == "stock_only"


def test_corrected_five_pass_launch_and_floor_ledger() -> None:
    audit = _json("offline_audit.json")
    launches = audit["shape_and_launch_ledger"]
    floor = audit["dfwd_floor_ledger"]
    assert isinstance(launches, dict)
    assert isinstance(floor, dict)

    shapes = launches["shapes"]
    assert isinstance(shapes, list)
    assert len(shapes) == 5
    assert sum(shape["weight_bytes"] for shape in shapes) == 456_130_560
    assert launches["initial_passes_per_event"] == 1
    assert launches["post_root_graph_passes_per_event"] == 4
    assert launches["total_passes_per_event"] == 5
    assert launches["candidate_projection_launches_per_event"] == 25
    assert launches["projection_weight_bytes_per_event"] == 2_280_652_800
    assert launches["complete_mtp_weight_bytes_per_event"] == 2_385_998_720

    assert floor["mtp_bytes_per_event"] == 2_385_998_720
    assert floor["five_k64_bf16_head_bytes_per_event"] == 3_355_443_200
    assert floor["total_dfwd_bytes_per_event"] == 5_741_441_920
    assert abs(floor["total_dfwd_floor_ms_per_event"] - 21.030922784) < 1e-9
    assert abs(floor["valid_b1_hydra_dfwd_gap_ms"] - 15.782445350) < 1e-9


def test_checksum_inventory_is_complete() -> None:
    expected = {"README.md", "manifest.json", "offline_audit.json", "SHA256SUMS"}
    assert {path.name for path in ARTIFACT.iterdir()} == expected

    checksums: dict[str, str] = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    assert set(checksums) == expected - {"SHA256SUMS"}
    for name, digest in checksums.items():
        assert _sha256(ARTIFACT / name) == digest
