from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "results/fr13_fixed32_cutlass_b1_m128_exactshape_sm121a_20260805"
)


def _tsv(name: str) -> list[dict[str, str]]:
    with (ARTIFACT / name).open(newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def test_exactshape_artifact_records_offline_boundary() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    assert manifest["schema"] == (
        "fr13.fixed32.cutlass_b1_m128_exactshape_sm121a.v1"
    )
    assert manifest["status"] == "double_compile_codegen_pass_live_gate_pending"
    assert manifest["acceptance_valid"] is False
    assert manifest["measurement_valid"] is False
    assert manifest["performance_claim"] is False
    assert manifest["production_available"] is False
    assert manifest["production_default_enabled"] is False
    assert manifest["campaign_contract"] == {
        "batch_size": 1,
        "physical_rows": 32,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "selector": "identity_wide256_fullgrid_b1",
        "selector_default_off": True,
        "production_credential_remains_historical": True,
    }
    assert manifest["verification"]["gpu_runtime_used"] is False
    assert manifest["verification"]["docker_used"] is False
    assert manifest["verification"]["real_swe_verified_run"] is False


def test_exactshape_codegen_removes_loads_without_resource_or_math_regression() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    codegen = manifest["codegen"]
    assert codegen["exact_kernel_count"] == 6
    assert codegen["generic_encoded_slots"] == codegen["exact_encoded_slots"] == 560
    assert codegen["generic_ldc"] + codegen["generic_ldcu"] == 20
    assert codegen["exact_ldc"] + codegen["exact_ldcu"] == 16
    assert codegen["constant_load_delta"] == -4
    assert codegen["registers_per_thread"] == 168
    assert codegen["stack_bytes_per_thread"] == 0
    assert codegen["local_bytes_per_thread"] == 0
    assert codegen["full_sass_byte_identical_across_compiles"] is True
    assert codegen["resource_report_byte_identical_across_compiles"] is True
    assert codegen["exact_target_sass_byte_identical_across_compiles"] is True

    sass = _tsv("sass_summary.tsv")
    resources = _tsv("kernel_resources.tsv")
    assert len(sass) == len(resources) == 8
    generic = [row for row in sass if row["variant"] == "generic"]
    exact = [row for row in sass if row["variant"] == "exact"]
    assert len(generic) == 2
    assert len(exact) == 6
    for row in generic:
        assert int(row["constant_loads"]) == 20
    for row in exact:
        assert int(row["constant_loads"]) == 16
        assert int(row["slots"]) == 560
        assert int(row["branches"]) == 29
        assert (int(row["qmma"]), int(row["ffma"])) == (32, 32)
        assert (int(row["ldsm"]), int(row["stsm"])) == (24, 4)
    for row in resources:
        assert int(row["registers"]) == 168
        assert int(row["stack_bytes"]) == 0
        assert int(row["local_bytes"]) == 0
        assert int(row["static_shared_bytes"]) == 1024
        assert int(row["ldl"]) == int(row["stl"]) == int(row["call"]) == 0


def test_exactshape_projection_routes_cover_real_histogram_once() -> None:
    schedules = _tsv("projection_schedule.tsv")
    assert sum(int(row["calls_per_16"]) for row in schedules) == 16
    exact = [row for row in schedules if row["route"].startswith("exact_")]
    retained = [row for row in schedules if row["route"].startswith("retained_")]
    assert sum(int(row["calls_per_16"]) for row in exact) == 8
    assert sum(int(row["calls_per_16"]) for row in retained) == 8

    for row in exact:
        problem_tiles = int(row["logical_tiles"])
        grid_ctas = int(row["grid_ctas"])
        assignments = [
            tile
            for cta in range(grid_ctas)
            for tile in range(cta, problem_tiles, grid_ctas)
        ]
        assert sorted(assignments) == list(range(problem_tiles))
        assert len(assignments) == len(set(assignments))
        tile_counts = [
            len(range(cta, problem_tiles, grid_ctas))
            for cta in range(grid_ctas)
        ]
        assert min(tile_counts) == int(row["tiles_per_cta_min"])
        assert max(tile_counts) == int(row["tiles_per_cta_max"])


def test_exactshape_artifact_is_source_bound_sanitized_and_integral() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    forbidden = {".o", ".cubin", ".ptx", ".sass", ".log", ".resources"}
    assert not any(path.suffix in forbidden for path in ARTIFACT.rglob("*"))
    for path in ARTIFACT.rglob("*"):
        if path.is_file():
            assert "/home/" not in path.read_text()

    implementation = manifest["source"]["implementation_commit"]
    for line in (ARTIFACT / "source_checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        historical = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{implementation}:{relative}"],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == expected

    sums = (ARTIFACT / "SHA256SUMS").read_text().splitlines()
    assert sums
    for line in sums:
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((ARTIFACT / relative).read_bytes()).hexdigest() == expected
