import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_cutlass_b1_wide256_directgrid_sm121a_20260805"
)
PATCHER = ROOT / "scripts" / "fr13_patch_cutlass_fixed32_wave.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_wide256_directgrid_artifact_binds_source_and_scope() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())

    assert manifest["schema"] == (
        "fr13.fixed32.cutlass_b1_wide256_directgrid_sm121a.v1"
    )
    assert manifest["acceptance_valid"] is False
    assert manifest["measurement_valid"] is False
    assert manifest["performance_claim"] is False
    assert manifest["production_default_enabled"] is False
    assert manifest["source"]["patch_source_sha256"] == (
        "7f0d6e37e12898e7a4f747d980747146b7a4fd05361502b5307988cd2948ec11"
    )
    assert manifest["source"]["patch_source_sha256"] != _sha256(PATCHER)

    candidate = manifest["candidate"]
    assert candidate["scheduler_tag"] in PATCHER.read_text(encoding="ascii")
    assert candidate["tile_shape_mnk"] == [256, 32, 128]
    assert candidate["cluster_shape_mnk"] == [1, 1, 1]
    assert candidate["physical_grid_ctas"] == 48
    for invariant in (
        "arithmetic_changed",
        "reduction_order_changed",
        "launch_count_changed",
        "requested_traffic_changed",
        "split_k_workspace",
    ):
        assert candidate[invariant] is False


def test_wide256_directgrid_codegen_and_work_model_are_consistent() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    codegen = manifest["codegen"]

    assert codegen["incumbent_encoded_instructions"] == 1080
    assert codegen["candidate_encoded_instructions"] == 784
    assert codegen["encoded_instruction_delta"] == -296
    assert codegen["registers_per_thread"] == 168
    assert codegen["stack_bytes_per_thread"] == 0
    assert codegen["local_bytes_per_thread"] == 0
    assert codegen["detected_local_load_store_or_call"] is False
    assert codegen["m128_neighbor_sass_byte_identical"] is True

    shapes = manifest["work_model"]["shapes"]
    assert [shape["tiles_per_call"] for shape in shapes] == [56, 64, 136]
    assert sum(
        shape["tile_assignments_per_target_step"] for shape in shapes
    ) == manifest["work_model"]["wide_tile_assignments_per_target_step"]
    assert manifest["work_model"]["wide_tile_assignments_per_target_step"] == 12672

    rows = list(
        csv.DictReader(
            (ARTIFACT / "sass_delta.tsv").read_text().splitlines(),
            delimiter="\t",
        )
    )
    neighbors = [row for row in rows if row["kernel"] == "m128_neighbor"]
    assert len(neighbors) == 2
    assert all(
        row["incumbent_sass_sha256"] == row["candidate_sass_sha256"]
        for row in neighbors
    )


def test_wide256_directgrid_artifact_checksums() -> None:
    expected = {
        "README.md",
        "manifest.json",
        "kernel_resources.tsv",
        "sass_delta.tsv",
        "scheduler_work_model.tsv",
        "verification.txt",
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
