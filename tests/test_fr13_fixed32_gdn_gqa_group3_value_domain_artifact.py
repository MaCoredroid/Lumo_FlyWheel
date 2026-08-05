from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/fr13_fixed32_gdn_gqa_group3_value_domain_sm121a_20260805"
)
CANDIDATE_REVISION = "1d08d3952d806306816de12988e5aa1258620566"


def test_value_domain_codegen_resources_are_exact_and_spill_free() -> None:
    summary = json.loads((ARTIFACT / "codegen_summary.json").read_text())
    assert summary["schema"] == (
        "fr13.fixed32.gdn_gqa_group3_value_domain.sm121a.codegen.v1"
    )
    contract = summary["compile_contract"]
    assert contract["batches"] == [1, 4]
    assert contract["physical_rows_per_request"] == 32
    assert contract["trusted_value_domain"] == [0, 127]
    assert contract["value_domain_masks_removed_per_cta"] == 291
    assert contract["invocation_atomics_removed_per_event"] == {
        "b1": 0,
        "b4": 3,
    }

    profiles = (
        (
            "baseline_static_schedule_base",
            "candidate_value_domain_base",
            116,
            108,
            2012,
            1972,
            54,
        ),
        (
            "baseline_static_schedule_committer_stack",
            "candidate_value_domain_committer_stack",
            118,
            118,
            2119,
            2078,
            82,
        ),
    )
    for baseline_name, candidate_name, br, cr, bs, cs, stg in profiles:
        for batch in ("b1", "b4"):
            baseline = summary["variants"][baseline_name]["builds"][batch]
            candidate = summary["variants"][candidate_name]["builds"][batch]
            assert baseline["registers_per_thread"] == br
            assert candidate["registers_per_thread"] == cr
            assert baseline["static_sass_instructions"] == bs
            assert candidate["static_sass_instructions"] == cs
            assert baseline["ldg"] == candidate["ldg"] == 74
            assert baseline["stg"] == candidate["stg"] == stg
            for row in (baseline, candidate):
                assert row["stack_bytes_per_thread"] == 0
                assert row["local_bytes_per_thread"] == 0
                assert row["ldl"] == row["stl"] == row["calls"] == 0


def test_value_domain_artifact_is_verified_and_sanitized() -> None:
    verification = json.loads((ARTIFACT / "verification.json").read_text())
    assert verification["status"] == "PASS"
    assert verification["builds_verified"] == 8
    assert verification["fresh_cache_byte_identity"] is True
    assert verification["gpu_execution"] is False

    forbidden = {".cubin", ".ptx", ".sass", ".ttir", ".ttgir", ".llir"}
    assert not any(path.suffix in forbidden for path in ARTIFACT.rglob("*"))
    for manifest in ("source_checksums.sha256", "SHA256SUMS"):
        for line in (ARTIFACT / manifest).read_text().splitlines():
            expected, relative = line.split("  ", 1)
            if manifest == "source_checksums.sha256" and relative.startswith(
                "src/"
            ):
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
                source = (
                    ROOT / relative
                    if manifest == "source_checksums.sha256"
                    else ARTIFACT / relative
                )
                raw = source.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == expected
