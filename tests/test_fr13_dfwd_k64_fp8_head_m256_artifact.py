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
    / "fr13_fixed32_dfwd_k64_fp8_head_m256_sm121a_20260805"
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
    spec = importlib.util.spec_from_file_location("fr13_k64_m256_binary", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_binds_source_tests_and_diagnostic_binary() -> None:
    manifest = _json("manifest.json")
    source = manifest["source"]
    outputs = manifest["outputs"]
    assert isinstance(source, dict)
    assert isinstance(outputs, dict)

    revision = source["source_commit"]
    assert isinstance(revision, str)
    expected = {
        "scripts/fr13_patch_cutlass_fixed32_wave.py": "patch_source_sha256",
        "scripts/fr13_cutlass_wave_binary.py": "binary_verifier_sha256",
        "scripts/fr13_build_dfwd_k64_fp8_head_m256_sm121a.sh": (
            "build_script_sha256"
        ),
        "tests/test_fr13_cutlass_fixed32_wave_patch.py": "patch_test_sha256",
        "tests/test_fr13_cutlass_wave_binary.py": "binary_test_sha256",
    }
    for relative, field in expected.items():
        assert _sha256_at_revision(revision, relative) == source[field]

    binary = outputs["candidate_binary"]
    assert isinstance(binary, dict)
    module = _binary_module()
    assert binary["sha256"] == module.K64_HEAD_M256_CANDIDATE_SHA256
    assert binary["bytes"] == module.K64_HEAD_M256_CANDIDATE_SIZE
    assert module.K64_HEAD_M256_SELECTORS == {"k64_head_m256_byte_ab"}
    assert "k64_head_m256_byte_ab" not in module.PRODUCTION_SELECTORS


def test_artifact_is_default_off_stock_served_and_non_accepting() -> None:
    manifest = _json("manifest.json")
    candidate = manifest["candidate"]
    assert isinstance(candidate, dict)

    assert manifest["acceptance_valid"] is False
    assert manifest["performance_claim"] is False
    assert manifest["gpu_runtime_used"] is False
    assert manifest["docker_used"] is False
    assert candidate["diagnostic_selector"] == "k64_head_m256_byte_ab"
    assert candidate["production_selector"] is None
    assert candidate["default_enabled"] is False
    assert candidate["qualification_profile"] == "k64_root"
    assert candidate["served_output"] == "stock"
    assert candidate["unarmed_behavior"] == "stock_only"


def test_candidate_reduces_launch_waves_without_register_regression() -> None:
    audit = _json("offline_audit.json")
    resources = audit["resource_audit"]
    launches = audit["shape_and_launch_ledger"]
    assert isinstance(resources, dict)
    assert isinstance(launches, dict)

    assert resources["threads_per_cta"] == 384
    assert resources["registers_per_thread"] == 168
    assert resources["registers_per_cta"] == 64_512
    assert resources["stock_registers_per_cta"] == 64_512
    assert resources["same_register_occupancy_envelope"] is True
    assert resources["detected_spills"] is False

    stock = launches["stock"]
    candidate = launches["candidate"]
    assert isinstance(stock, dict)
    assert isinstance(candidate, dict)
    assert stock["logical_output_tiles"] == 512
    assert stock["physical_grid_ctas"] == 512
    assert stock["nominal_cta_waves_at_48_sms"] == 11
    assert candidate["logical_output_tiles"] == 256
    assert candidate["balanced_physical_grid_ctas"] == 32
    assert candidate["logical_tiles_per_cta"] == 8
    assert candidate["nominal_cta_waves_at_48_sms"] == 1
    assert launches["b1_b4_geometry_equal"] is True


def test_sass_and_corrected_fp8_floor_ledgers() -> None:
    audit = _json("offline_audit.json")
    sass = audit["sass_audit"]
    floor = audit["traffic_and_floor_ledger"]
    assert isinstance(sass, dict)
    assert isinstance(floor, dict)

    assert sass["candidate"]["instructions"] == 832
    assert sass["candidate"]["qmma"] == 64
    assert sass["candidate"]["ldl"] == 0
    assert sass["candidate"]["stl"] == 0
    assert sass["candidate"]["call"] == 0
    assert sass["stock"]["instructions"] == 1176
    assert sass["performance_inference_allowed"] is False

    assert floor["mandatory_bytes_per_call"] == 335_626_240
    assert floor["five_call_mandatory_bytes"] == 1_678_131_200
    assert floor["candidate_full_step_mandatory_bytes"] == 30_989_326_208
    assert abs(floor["candidate_mandatory_weight_floor_ms"] - 113.514015414) < 1e-9
    assert abs(floor["one_sided_u95_cap_ms"] - 130.541117726) < 1e-9
    assert floor["nonweight_costs_included"] is False


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
