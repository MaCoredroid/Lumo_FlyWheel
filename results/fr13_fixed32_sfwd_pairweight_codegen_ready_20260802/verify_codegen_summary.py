#!/usr/bin/env python3
"""Verify the reduced paired-weight SFWD codegen summary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    summary = json.loads((ROOT / "codegen_summary.json").read_text())
    assert summary["status"] == "offline_codegen_pass_real_byte_gate_required"
    assert summary["target"] == "sm_121a"
    codegen = summary["candidate_codegen"]
    assert codegen["batches"] == [1, 4]
    assert codegen["b1_b4_binary_identity"] is True
    assert codegen["fresh_cache_rebuild_identity"] is True
    assert codegen["registers"] == 55
    assert codegen["launch_shared_bytes"] == 4096
    for key in ("stack_bytes", "local_bytes", "ldl", "stl", "calls"):
        assert codegen[key] == 0
    delta = summary["delta_candidate_minus_baseline"]
    for key in ("bar", "ldg", "lds", "sts", "static_sass_instructions"):
        assert delta[key] < 0


if __name__ == "__main__":
    main()
