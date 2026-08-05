import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_cutlass_b1_wide256_directgrid_f81_build_20260805"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_manifest_binds_f81_source_and_binary_registries() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    assert manifest["acceptance_valid"] is False
    assert manifest["measurement_valid"] is False
    assert manifest["performance_claim"] is False
    assert manifest["source"]["main_base_commit"] == (
        "f81a1c774b55a7f76d30d30ed0fac2be73665be9"
    )
    assert manifest["source"]["implementation_commit"] == (
        "6898009ccf5f557a3ca8a1bc9f3e19b1a16c2467"
    )
    assert manifest["source"]["integration_parent_commit"] == (
        manifest["source"]["main_base_commit"]
    )
    assert manifest["source"]["patch_source_sha256"] == (
        "7f0d6e37e12898e7a4f747d980747146b7a4fd05361502b5307988cd2948ec11"
    )
    assert manifest["source"]["patch_source_sha256"] != _sha256(
        ROOT / "scripts/fr13_patch_cutlass_fixed32_wave.py"
    )
    assert manifest["source"]["binary_registry_sha256"] != _sha256(
        ROOT / "scripts/fr13_cutlass_wave_binary.py"
    )
    assert manifest["source"]["qualification_registry_sha256"] != _sha256(
        ROOT / "scripts/fr13_cutlass_streamk_pass.py"
    )


def test_failed_binary_and_source_contract_are_superseded() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    binary = _load(
        ROOT / "scripts/fr13_cutlass_wave_binary.py", "directgrid_binary_registry"
    )
    qualification = _load(
        ROOT / "scripts/fr13_cutlass_streamk_pass.py",
        "directgrid_qualification_registry",
    )

    assert binary.candidate_identity(
        manifest["binary"]["diagnostic_selector"]
    ) != (
        manifest["binary"]["sha256"],
        manifest["binary"]["bytes"],
        manifest["binary"]["candidate_family"],
    )
    source_contract = qualification._source_contract(
        manifest["binary"]["candidate_family"]
    )
    assert source_contract["patch_source_sha256"] != manifest["source"][
        "patch_source_sha256"
    ]
    assert source_contract["patched_dispatch_sha256"] != manifest["source"][
        "patched_dispatch_sha256"
    ]
    assert binary.IDENTITY_FULLTILE_CANDIDATE_SHA256 == manifest["incumbent"][
        "sha256"
    ]
    assert (
        qualification._source_contract("identity_onen_n5120_fullgrid_b1")
        != source_contract
    )


def test_link_closure_and_deferred_gate_are_complete() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    rows = list(
        csv.DictReader(
            (ARTIFACT / "object_closure.tsv").read_text().splitlines(),
            delimiter="\t",
        )
    )
    assert len(rows) == manifest["build"]["object_count"] == 19
    target = [
        row
        for row in rows
        if row["logical_path"].endswith("scaled_mm_blockwise_sm120_fp8.cu.o")
    ]
    assert len(target) == 1
    assert target[0]["sha256"] == manifest["build"][
        "candidate_translation_unit_sha256"
    ]

    launch = json.loads((ARTIFACT / "launch_contract.json").read_text())
    assert launch["launch_state"] == "deferred_until_serial_gpu_campaign_idle"
    assert launch["outer_wrapper"].endswith(
        "fr13_run_b1_target_sfwd_conv_postprep_live_gate.sh"
    )
    assert launch["required_postconditions"]["comparisons"] == 320
    assert launch["required_postconditions"]["mismatching_comparisons"] == 0
    assert launch["required_postconditions"]["differing_bytes"] == 0
    assert len(launch["required_postconditions"]["observed_projection_nk"]) == 5
    assert launch["inputs"]["target_sha256"] == manifest["binary"]["sha256"]


def test_build_artifact_checksums() -> None:
    expected = {
        "README.md",
        "manifest.json",
        "object_closure.tsv",
        "binary_verify.json",
        "launch_contract.json",
        "deferred_gate_command.txt",
        "host_verification.txt",
        "SHA256SUMS",
    }
    assert {path.name for path in ARTIFACT.iterdir()} == expected
    checksums = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    assert set(checksums) == expected - {"SHA256SUMS"}
    for name, digest in checksums.items():
        assert _sha256(ARTIFACT / name) == digest
