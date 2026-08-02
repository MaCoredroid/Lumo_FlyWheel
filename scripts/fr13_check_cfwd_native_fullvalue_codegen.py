#!/usr/bin/env python3
"""Check the pinned SM121 native CFWD object resource and SASS contract."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


KERNEL_MARKER = "fixed32_cfwd_native_fullvalue_kernel"
EXPECTED_SASS_COUNTS = {
    "MUFU.EX2": 3,
    "MUFU.RSQ": 1,
    "MUFU.RCP": 2,
    "SHFL.BFLY": 174,
    "SHFL.IDX": 16,
    "FFMA": 78,
}


def _opcode_count(sass: str, opcode: str) -> int:
    return len(re.findall(rf"\b{re.escape(opcode)}\b", sass))


def check_codegen(resource_report: str, sass: str) -> dict[str, object]:
    arch_match = re.search(r"^arch = (\S+)$", resource_report, re.MULTILINE)
    if arch_match is None or arch_match.group(1) != "sm_121a":
        raise RuntimeError("native full-value CFWD object must target sm_121a")

    resource_match = re.search(
        rf"^ Function [^\n]*{KERNEL_MARKER}[^\n]*:\n"
        r"\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        resource_report,
        re.MULTILINE,
    )
    if resource_match is None:
        raise RuntimeError("native full-value CFWD kernel resources not found")
    registers, stack, shared, local = map(int, resource_match.groups())
    resources = {
        "registers_per_thread": registers,
        "stack_bytes": stack,
        "cuobjdump_shared_bytes": shared,
        "local_bytes": local,
    }
    expected_resources = {
        "registers_per_thread": 64,
        "stack_bytes": 0,
        "cuobjdump_shared_bytes": 1572,
        "local_bytes": 0,
    }
    if resources != expected_resources:
        raise RuntimeError(
            f"native full-value CFWD resource drift: {resources}"
        )

    forbidden_counts = {
        "LDL": len(re.findall(r"\bLDL(?:\.|\s)", sass)),
        "STL": len(re.findall(r"\bSTL(?:\.|\s)", sass)),
        "CALL": _opcode_count(sass, "CALL"),
    }
    if any(forbidden_counts.values()):
        raise RuntimeError(
            f"native full-value CFWD local/call drift: {forbidden_counts}"
        )

    sass_counts = {
        opcode: _opcode_count(sass, opcode)
        for opcode in EXPECTED_SASS_COUNTS
    }
    if sass_counts != EXPECTED_SASS_COUNTS:
        raise RuntimeError(
            f"native full-value CFWD SASS shape drift: {sass_counts}"
        )

    return {
        "architecture": "sm_121a",
        "resources": resources,
        "forbidden_sass_counts": forbidden_counts,
        "sass_counts": sass_counts,
        "contract_pass": True,
    }


def _cuobjdump(cuobjdump: Path, option: str, object_path: Path) -> str:
    completed = subprocess.run(
        [str(cuobjdump), option, str(object_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("object", type=Path)
    parser.add_argument(
        "--cuobjdump",
        type=Path,
        default=Path("/usr/local/cuda/bin/cuobjdump"),
    )
    args = parser.parse_args()
    receipt = check_codegen(
        _cuobjdump(args.cuobjdump, "--dump-resource-usage", args.object),
        _cuobjdump(args.cuobjdump, "--dump-sass", args.object),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
