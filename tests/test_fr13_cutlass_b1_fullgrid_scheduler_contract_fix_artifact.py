from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_cutlass_b1_fullgrid_scheduler_contract_fix_20260803"
)


def _load(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contract_fix_artifact_is_default_off_and_non_accepting() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())

    assert manifest["status"] == (
        "host_compile_link_import_pass_fixed_binary_default_off_"
        "gpu_requalification_required"
    )
    assert manifest["acceptance_valid"] is False
    assert manifest["performance_claim"] is False
    assert manifest["candidate"]["default_enabled"] is False
    assert manifest["liveness_failure"]["valid_gate_credential"] is False
    assert manifest["liveness_failure"]["valid_byte_comparison"] is False
    assert manifest["liveness_failure"]["valid_timing"] is False


def test_contract_fix_artifact_binds_source_and_binary_selectors() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    source = manifest["source"]
    candidate = manifest["candidate"]

    patch_path = ROOT / "scripts/fr13_patch_cutlass_fixed32_wave.py"
    assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == source[
        "patch_source_sha256"
    ]

    streamk = _load(ROOT / "scripts/fr13_cutlass_streamk_pass.py")
    binary = _load(ROOT / "scripts/fr13_cutlass_wave_binary.py")
    selector = candidate["selector"]
    assert streamk.SOURCE_CONTRACTS[selector]["patch_source_sha256"] == source[
        "patch_source_sha256"
    ]
    assert binary.IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SHA256 == (
        candidate["sha256"]
    )
    assert binary.IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SIZE == candidate[
        "bytes"
    ]


def test_contract_fix_artifact_pins_exact_five_shape_routes() -> None:
    rows = (ARTIFACT / "shape_contract.tsv").read_text().splitlines()
    assert rows[1:] == [
        "32\t34816\t5120\ttrue\t272\t1\t48\t1\tpingpong\t"
        "initialized_static_persistent\trepaired",
        "32\t16384\t5120\ttrue\t128\t1\t48\t1\tpingpong\t"
        "initialized_static_persistent\trepaired",
        "32\t14336\t5120\ttrue\t112\t1\t48\t1\tpingpong\t"
        "initialized_static_persistent\trepaired",
        "32\t5120\t17408\ttrue\t40\t1\t40\t1\tcooperative\t"
        "exact_single_tile\tunchanged",
        "32\t5120\t6144\ttrue\t40\t1\t40\t1\tcooperative\t"
        "exact_single_tile\tunchanged",
    ]


def test_contract_fix_artifact_is_sanitized_and_reproducible() -> None:
    expected = {
        "README.md",
        "SHA256SUMS",
        "kernel_resources.tsv",
        "manifest.json",
        "sass_delta.tsv",
        "shape_contract.tsv",
        "verification.txt",
    }
    assert {path.name for path in ARTIFACT.iterdir()} == expected
    assert not any(
        path.suffix in {".cubin", ".ptx", ".sass", ".log"}
        for path in ARTIFACT.rglob("*")
    )

    published_text = "\n".join(
        path.read_text()
        for path in ARTIFACT.iterdir()
        if path.name != "SHA256SUMS"
    )
    for forbidden in ("/home/", "CUDA_VISIBLE_DEVICES=", "container_id", "pid="):
        assert forbidden not in published_text

    checksums = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    assert set(checksums) == expected - {"SHA256SUMS"}
    for name, expected_digest in checksums.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == (
            expected_digest
        )
