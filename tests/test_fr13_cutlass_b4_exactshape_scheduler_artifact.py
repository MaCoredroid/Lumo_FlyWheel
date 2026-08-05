from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/fr13_fixed32_cutlass_b4_exactshape_scheduler_sm121a_20260805"
)


def _tsv(name: str) -> list[dict[str, str]]:
    with (ARTIFACT / name).open(newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def test_exactshape_codegen_reduces_scheduler_work_without_spills() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    assert manifest["status"] == "RETAIN_OFFLINE_CODEGEN_WIN_RUNTIME_PENDING"
    assert manifest["acceptance_valid"] is False
    assert manifest["measurement_valid"] is False
    assert manifest["kernel_contract"]["physical_rows"] == 128
    assert manifest["kernel_contract"]["candidate_projection_calls"] == 8
    assert manifest["kernel_contract"]["total_projection_calls"] == 16

    resources = _tsv("kernel_resources.tsv")
    candidates = [row for row in resources if row["variant"] == "exactshape_candidate"]
    comparators = [row for row in resources if row["variant"] == "generic_in_object"]
    assert len(candidates) == 6
    assert len(comparators) == 2
    for row in resources:
        assert int(row["registers_per_thread"]) == 168
        assert int(row["stack_bytes_per_thread"]) == 0
        assert int(row["local_bytes_per_thread"]) == 0
        assert int(row["static_shared_bytes_per_cta"]) == 1024

    sass = _tsv("sass_summary.tsv")
    for dtype in ("fp16", "bf16"):
        baseline = next(
            row
            for row in sass
            if row["variant"] == "generic_in_object" and row["dtype"] == dtype
        )
        exact_rows = [
            row
            for row in sass
            if row["variant"] == "exactshape_candidate" and row["dtype"] == dtype
        ]
        assert len(exact_rows) == 3
        for row in exact_rows:
            assert int(row["sass_slots"]) == int(baseline["sass_slots"]) - 8
            assert int(row["ldcu"]) == int(baseline["ldcu"]) - 6
            assert int(row["ldc"]) == int(baseline["ldc"]) - 1
            for field in ("branches", "qmma", "ffma", "ldsm", "stsm"):
                assert row[field] == baseline[field]
            assert row["ldl"] == row["stl"] == row["calls"] == "0"


def test_exactshape_pingpong_assignment_covers_each_tile_once() -> None:
    schedules = _tsv("projection_schedule.tsv")
    assert sum(int(row["real_calls"]) for row in schedules) == 16
    exact = [row for row in schedules if row["route"] == "exactshape_candidate"]
    assert sum(int(row["real_calls"]) for row in exact) == 8

    for row in exact:
        grid_ctas = int(row["grid_ctas"])
        n_tiles = int(row["n_tiles"])
        n_stride = int(row["n_grid_stride"])
        assert n_stride == grid_ctas // 2
        coordinates: list[tuple[int, int]] = []
        for cta in range(grid_ctas):
            for consumer_group in range(2):
                n_idx = (cta >> 1) + consumer_group * n_stride
                while n_idx < n_tiles:
                    coordinates.append((cta & 1, n_idx))
                    n_idx += 2 * n_stride
        expected = [(m, n_idx) for m in range(2) for n_idx in range(n_tiles)]
        assert sorted(coordinates) == expected
        assert len(coordinates) == len(set(coordinates))


def test_exactshape_artifact_is_sanitized_and_integral() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    forbidden = {".o", ".cubin", ".ptx", ".sass", ".log", ".resources"}
    assert not any(path.suffix in forbidden for path in ARTIFACT.rglob("*"))
    for path in ARTIFACT.rglob("*"):
        if path.is_file():
            assert "/home/" not in path.read_text()

    for line in (ARTIFACT / "source_checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        historical = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{manifest['source']['candidate_commit']}:{relative}",
            ],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == expected

    for line in (ARTIFACT / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((ARTIFACT / relative).read_bytes()).hexdigest() == expected
