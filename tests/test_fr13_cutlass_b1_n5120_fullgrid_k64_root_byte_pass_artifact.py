from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_cutlass_b1_n5120_fullgrid_k64_root_byte_pass_20260803"
)
GATE = ARTIFACT / "cutlass_identity_onen_n5120_fullgrid_b1_k64_root_byte_gate.json"
GATE_SHA256 = "9e50e63635cb57a126dbe5621ae8ffde5c156bf59186ea0a5a73de0afa1e8083"
SOURCE_COMMIT = "c49c8eb5370e4d4035aceffaa8476aea31f921f5"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_gate_is_bound_to_source_profile_and_candidate() -> None:
    gate = _json(GATE)

    assert hashlib.sha256(GATE.read_bytes()).hexdigest() == GATE_SHA256
    assert gate["schema"] == (
        "fr13.fixed32.cutlass_identity_onen_n5120_fullgrid_b1_"
        "k64_root_live_gate.v1"
    )
    assert gate["status"] == "pass"
    assert gate["source_commit"] == SOURCE_COMMIT
    assert gate["qualification_profile"] == "k64_root"
    assert gate["diagnostic_task_profile"] == "astropy13236"
    assert gate["candidate"] == "identity_onen_n5120_fullgrid_b1"
    assert gate["task_ids"] == ["astropy__astropy-13236"]
    assert gate["errors"] == []

    binary = _load(ROOT / "scripts/fr13_cutlass_wave_binary.py")
    assert gate["candidate_sha256"] == (
        binary.IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SHA256
    )
    assert gate["candidate_bytes"] == (
        binary.IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SIZE
    )


def test_gate_records_all_five_m32_shapes_with_exact_bytes() -> None:
    gate = _json(GATE)

    assert gate["comparison_call_limit"] == 320
    assert gate["comparisons"] == 320
    assert gate["fixed_rows"] == 32
    assert gate["observed_m_values"] == [32]
    assert gate["observed_projection_nk"] == [
        [5120, 6144],
        [5120, 17408],
        [14336, 5120],
        [16384, 5120],
        [34816, 5120],
    ]
    assert gate["mismatching_comparisons"] == 0
    assert gate["differing_bytes"] == 0


def test_gate_is_stock_served_default_off_and_non_accepting() -> None:
    gate = _json(GATE)
    manifest = _json(ARTIFACT / "manifest.json")

    assert gate["served_result"] == "stock"
    assert gate["production_enabled"] is False
    assert gate["acceptance_valid"] is False
    assert gate["timing_eligible"] is False
    assert gate["comparator_timing_eligible"] is False
    assert gate["floor_acceptance_eligible"] is False
    assert manifest["task"]["outcome"] == "resolved"
    assert manifest["candidate"]["default_enabled"] is False
    assert manifest["claims"] == {
        "real_task_resolved": True,
        "byte_correctness": True,
        "acceptance_valid": False,
        "acceptance_claim": False,
        "timing_eligible": False,
        "timing_claim": False,
        "performance_claim": False,
        "hardware_floor_evidence": False,
        "candidate_served": False,
    }


def test_source_identity_matches_the_pinned_repository_files() -> None:
    gate = _json(GATE)

    assert gate["source_identity"]["source_commit"] == SOURCE_COMMIT
    for relative, identity in gate["source_identity"]["files"].items():
        content = (ROOT / relative).read_bytes()
        assert len(content) == identity["bytes"]
        assert hashlib.sha256(content).hexdigest() == identity["sha256"]


def test_reduced_artifact_is_sanitized_and_checksum_complete() -> None:
    expected = {
        "README.md",
        "SHA256SUMS",
        "cutlass_identity_onen_n5120_fullgrid_b1_k64_root_byte_gate.json",
        "manifest.json",
    }
    assert {path.name for path in ARTIFACT.iterdir()} == expected
    assert not any(
        path.suffix in {".jsonl", ".log", ".patch"}
        for path in ARTIFACT.rglob("*")
    )

    published_text = "\n".join(
        path.read_text()
        for path in ARTIFACT.iterdir()
        if path.name != "SHA256SUMS"
    )
    for forbidden in (
        "/home/",
        "/tmp/",
        '"container_id":',
        '"pid":',
        '"prompt":',
        '"response":',
        '"task_output":',
    ):
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
