#!/usr/bin/env python3
"""Verify the checked-in SFWD 8/16-row rejection evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY_BYTES = (ROOT / "prototype_codegen_summary.json").read_bytes()
SUMMARY = json.loads(SUMMARY_BYTES)
EXPECTED_SUMMARY_SHA256 = (
    "e30877e4ab979f7d57e16d9048438fdedb8240688234386057ebf4ed1f3c329e"
)
EXPECTED_SOURCE_SHA256 = (
    "e7a927bcd6c1da3f98403ee2cb23e55038f1cb4a28fdf906e8208cf65d45fcf9"
)
EXPECTED = {
    "b1": (64, 3416, 3255, 93, 344),
    "b4": (64, 3424, 3268, 93, 344),
}
GATEPACK2 = {
    "b1": (56, 3032, 2881, 85, 336),
    "b4": (56, 3040, 2889, 85, 336),
}


def main() -> int:
    assert hashlib.sha256(SUMMARY_BYTES).hexdigest() == EXPECTED_SUMMARY_SHA256
    for profile, expected in EXPECTED.items():
        build = SUMMARY[profile]
        assert build["source_sha256"] == EXPECTED_SOURCE_SHA256
        assert build["revision"] == "gatepack3_worktree"
        observed = (
            build["registers"],
            build["encoded_sass_instructions"],
            build["static_sass_instructions"],
            build["ldg"],
            build["stg"],
        )
        assert observed == expected
        assert build["stack_bytes"] == build["local_bytes"] == 0
        assert build["elf_shared_bytes"] == build["launch_shared_bytes"] == 0
        assert build["ldl"] == build["stl"] == build["calls"] == 0
        prior = GATEPACK2[profile]
        assert observed[0] - prior[0] == 8
        assert observed[1] - prior[1] == 384
        assert observed[2] - prior[2] in (374, 379)
    readme = " ".join((ROOT / "README.md").read_text().split())
    assert "rejected" in readme.lower()
    assert "not proposed for merge" in readme
    assert "no runtime performance claim" in readme
    print("PASS: fixed32 SFWD 8/16-row rejection evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
