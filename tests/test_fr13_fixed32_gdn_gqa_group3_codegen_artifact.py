from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "results/fr13_fixed32_gdn_gqa_group3_sm121a_codegen_20260803"
)
REVISION = "936dd110c01d34f8c1c5c64676dde5739d0d2fa3"
PROFILE_PAIRS = (
    (
        "incumbent_base_production",
        "candidate_gqa_group3_base_production",
        120,
        None,
    ),
    (
        "incumbent_committer_stack_production",
        "candidate_gqa_group3_committer_stack_production",
        126,
        128,
    ),
)


def _summary() -> dict[str, object]:
    return json.loads((ARTIFACT / "codegen_summary.json").read_text())


def _build(summary: dict[str, object], variant: str, batch: str):
    return summary["variants"][variant]["builds"][batch]


def test_exact_b1_b4_profiles_are_spill_free_and_launch_viable() -> None:
    summary = _summary()
    assert summary["schema"] == (
        "fr13.fixed32.gdn_gqa_group3.sm121a.codegen.v1"
    )
    assert summary["revision"] == REVISION
    assert summary["compile_contract"]["target"] == "sm_121a"
    assert summary["compile_contract"]["batches"] == [1, 4]
    assert summary["compile_contract"]["committer_stack_candidate_maxnreg"] == 128
    for batch, z in (("b1", 1), ("b4", 4)):
        for incumbent_name, candidate_name, registers, maxnreg in PROFILE_PAIRS:
            incumbent = _build(summary, incumbent_name, batch)
            candidate = _build(summary, candidate_name, batch)
            assert incumbent["grid"] == [48, 16, z]
            assert candidate["grid"] == [16, 16, z]
            assert incumbent["registers_per_thread"] == 80
            assert candidate["registers_per_thread"] == registers
            assert candidate["resolved_maxnreg"] == maxnreg
            assert candidate["threads_per_cta"] == 256
            assert candidate["launch_shared_bytes_per_cta"] == 16
            assert candidate["elf_shared_bytes_per_cta"] == 1024
            for row in (incumbent, candidate):
                assert row["stack_bytes_per_thread"] == 0
                assert row["local_bytes_per_thread"] == 0
                assert row["ldl"] == 0
                assert row["stl"] == 0
                assert row["calls"] == 0
                assert row["gpu_execution"] is False


def test_resource_regression_and_static_work_proxies_are_explicit() -> None:
    summary = _summary()
    for batch in ("b1", "b4"):
        for incumbent_name, candidate_name, _registers, _maxnreg in PROFILE_PAIRS:
            incumbent = _build(summary, incumbent_name, batch)
            candidate = _build(summary, candidate_name, batch)
            assert candidate["register_bytes_per_cta"] > incumbent[
                "register_bytes_per_cta"
            ]
            assert candidate["programs_per_layer_event"] * 3 == incumbent[
                "programs_per_layer_event"
            ]
            for metric in ("static_sass_instructions", "ldg", "stg"):
                assert candidate[metric] * candidate[
                    "programs_per_layer_event"
                ] < incumbent[metric] * incumbent["programs_per_layer_event"]


def test_artifact_source_hashes_and_sanitized_scope() -> None:
    for line in (ARTIFACT / "source_checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if relative.startswith("src/"):
            raw = subprocess.run(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "show",
                    f"{CANDIDATE_REVISION}:{relative}",
                ],
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
        else:
            raw = (ROOT / relative).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
    forbidden = {".cubin", ".ptx", ".sass", ".ttir", ".ttgir", ".llir"}
    assert not any(path.suffix in forbidden for path in ARTIFACT.rglob("*"))
    readme = (ARTIFACT / "README.md").read_text()
    assert "offline resource gate PASS" in readme
    assert "not performance-promoted" in readme
    assert "No GPU kernel was launched" in readme
    assert "real SWE-Verified B1/B4 byte gate" in readme


def test_artifact_package_checksums_match() -> None:
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        entries[relative] = expected
    files = {
        path.name
        for path in ARTIFACT.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert set(entries) == files
    for relative, expected in entries.items():
        observed = hashlib.sha256((ARTIFACT / relative).read_bytes()).hexdigest()
        assert observed == expected
