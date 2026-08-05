#!/usr/bin/env python3
"""Reproduce the rejected 8/16-row SFWD SM121a codegen without a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")


BASE_REVISION = "1a86df82dbe6e704e472d2a770d3290917ca57e2"
COMPILER_REVISION = "ee72339c39a83282bbd86298ea4796f71020d334"
SOURCE_PATH = "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion_kernel.py"
COMPILER_PATH = (
    "results/fr13_fixed32_sfwd_conv_postprep_gatepack2_sm121a_20260805/"
    "offline_codegen_audit.py"
)
OLD_GEOMETRY = "GATE_ROWS: tl.constexpr = 2 * BLOCK_C // GATE_BLOCK"
NEW_GEOMETRY = "GATE_ROWS: tl.constexpr = 4 * BLOCK_C // GATE_BLOCK"
EXPECTED_SOURCE_SHA256 = (
    "e7a927bcd6c1da3f98403ee2cb23e55038f1cb4a28fdf906e8208cf65d45fcf9"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{BASE_REVISION}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert base == BASE_REVISION
    source = subprocess.run(
        ["git", "-C", str(repo), "show", f"{base}:{SOURCE_PATH}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert source.count(OLD_GEOMETRY) == 1
    source = source.replace(OLD_GEOMETRY, NEW_GEOMETRY, 1)
    assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED_SOURCE_SHA256

    compiler_source = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "show",
            f"{COMPILER_REVISION}:{COMPILER_PATH}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    compiler = {
        "__name__": "fr13_gatepack3_rejection_compiler",
        "__file__": f"{COMPILER_PATH}@{COMPILER_REVISION}",
    }
    exec(compile(compiler_source, compiler["__file__"], "exec"), compiler)
    builds = {
        profile: compiler["compile_one"](
            source=source,
            revision="gatepack3_worktree",
            output=output / profile,
            profile=profile,
        )
        for profile in ("b1", "b4")
    }
    output.mkdir(parents=True, exist_ok=True)
    summary = json.dumps(builds, indent=2, sort_keys=True) + "\n"
    (output / "summary.json").write_text(summary)
    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
