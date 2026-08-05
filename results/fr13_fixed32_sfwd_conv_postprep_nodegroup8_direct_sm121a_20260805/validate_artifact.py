#!/usr/bin/env python3
"""Validate the reduced direct-nodegroup8 source/codegen artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = Path(__file__).resolve().parent


def main() -> int:
    manifest = json.loads((ARTIFACT / "source_manifest.json").read_text())
    commit = manifest["source_commit"]
    assert manifest["schema"].endswith("source_manifest.v1")
    for relative, expected in manifest["files"].items():
        raw = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        assert len(raw) == expected["bytes"], relative
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"], relative

    summary = json.loads((ARTIFACT / "codegen_summary.json").read_text())
    assert summary["schema"].endswith("sm121a_codegen.v1")
    assert summary["source_commit"] == commit
    assert summary["offline_only"] is True
    assert summary["gpu_visible"] is False
    assert summary["timing_claim"] is False
    assert summary["acceptance_claim"] is False
    guard = summary["memory_guard"]
    assert guard["samples"]
    assert all(
        sample["mem_available_kib"] >= guard["minimum_mem_available_kib"]
        for sample in guard["samples"]
    )
    assert set(summary["builds"]) == {"incumbent", "nodegroup8_direct"}
    expected_profiles = {
        "b1_standalone",
        "b1_embedded",
        "b4_standalone",
        "b4_embedded",
    }
    for label, profiles in summary["builds"].items():
        assert set(profiles) == expected_profiles, label
        for profile in profiles.values():
            assert profile["stack_bytes"] == 0
            assert profile["local_bytes"] == 0
            assert profile["elf_shared_bytes"] == 0
            assert profile["launch_shared_bytes"] == 0
            assert profile["calls"] == 0
    for profile in expected_profiles:
        incumbent = summary["builds"]["incumbent"][profile]
        direct = summary["builds"]["nodegroup8_direct"][profile]
        assert incumbent["registers_per_thread"] == 56
        assert direct["registers_per_thread"] in (46, 48)
        assert direct["ldg"] == 131
        assert incumbent["ldg"] == 85
        assert direct["stg"] == incumbent["stg"] == 336
    prior = summary["prior_rowgroup8_real_b1_regression"]
    assert prior["equivalent_design"] is False
    assert prior["delta_ms_per_step"] == 2.282206918
    assert "no source descriptor" in prior["distinction"]
    print("artifact validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
