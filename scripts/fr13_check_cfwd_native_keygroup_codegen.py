#!/usr/bin/env python3
"""Check the pinned SM121 key-group precompute CFWD object contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


KERNEL_MARKER = "fixed32_cfwd_native_fullvalue_kernel"
EXPECTED_RESOURCES = {
    "registers_per_thread": 64,
    "stack_bytes": 0,
    "cuobjdump_shared_bytes": 7_576,
    "local_bytes": 0,
}
EXPECTED_SASS_COUNTS = {
    "MUFU.EX2": 3,
    "MUFU.RSQ": 3,
    "MUFU.RCP": 2,
    "SHFL.BFLY": 202,
    "SHFL.IDX": 16,
    "SHFL.DOWN": 0,
    "FFMA": 82,
}


def _opcode_count(sass: str, opcode: str) -> int:
    return len(re.findall(rf"\b{re.escape(opcode)}\b", sass))


def check_codegen(resource_report: str, sass: str) -> dict[str, object]:
    arch_match = re.search(r"^arch = (\S+)$", resource_report, re.MULTILINE)
    if arch_match is None or arch_match.group(1) != "sm_121a":
        raise RuntimeError(
            "native key-group precompute CFWD object must target sm_121a"
        )

    resource_match = re.search(
        rf"^ Function [^\n]*{KERNEL_MARKER}[^\n]*:\n"
        r"\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        resource_report,
        re.MULTILINE,
    )
    if resource_match is None:
        raise RuntimeError(
            "native key-group precompute CFWD kernel resources not found"
        )
    registers, stack, shared, local = map(int, resource_match.groups())
    resources = {
        "registers_per_thread": registers,
        "stack_bytes": stack,
        "cuobjdump_shared_bytes": shared,
        "local_bytes": local,
    }
    if resources != EXPECTED_RESOURCES:
        raise RuntimeError(
            f"native key-group precompute CFWD resource drift: {resources}"
        )

    forbidden_counts = {
        "LDL": len(re.findall(r"\bLDL(?:\.|\s)", sass)),
        "STL": len(re.findall(r"\bSTL(?:\.|\s)", sass)),
        "CALL": _opcode_count(sass, "CALL"),
    }
    if any(forbidden_counts.values()):
        raise RuntimeError(
            "native key-group precompute CFWD local/call drift: "
            f"{forbidden_counts}"
        )

    sass_counts = {
        opcode: _opcode_count(sass, opcode)
        for opcode in EXPECTED_SASS_COUNTS
    }
    if sass_counts != EXPECTED_SASS_COUNTS:
        raise RuntimeError(
            f"native key-group precompute CFWD SASS shape drift: {sass_counts}"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("object", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "native/fr13_fixed32_cfwd_native_fullvalue.cu"
        ),
    )
    parser.add_argument("--compile-command", required=True)
    parser.add_argument(
        "--nvcc",
        type=Path,
        default=Path("/usr/local/cuda/bin/nvcc"),
    )
    parser.add_argument(
        "--cuobjdump",
        type=Path,
        default=Path("/usr/local/cuda/bin/cuobjdump"),
    )
    args = parser.parse_args()
    if not args.object.is_file() or not args.source.is_file():
        raise RuntimeError(
            "native key-group precompute CFWD source/object binding is absent"
        )
    if "arch=compute_121a,code=sm_121a" not in args.compile_command:
        raise RuntimeError(
            "native key-group precompute CFWD compile architecture is unbound"
        )
    if args.source.name not in args.compile_command:
        raise RuntimeError(
            "native key-group precompute CFWD compile source is unbound"
        )

    receipt = check_codegen(
        _cuobjdump(args.cuobjdump, "--dump-resource-usage", args.object),
        _cuobjdump(args.cuobjdump, "--dump-sass", args.object),
    )
    receipt["binding"] = {
        "compile_command": args.compile_command,
        "compile_command_sha256": hashlib.sha256(
            args.compile_command.encode("utf-8")
        ).hexdigest(),
        "object_name": args.object.name,
        "object_sha256": _sha256(args.object),
        "source_name": args.source.name,
        "source_sha256": _sha256(args.source),
    }
    receipt["toolchain"] = {
        "cuobjdump_version": _tool_version(args.cuobjdump),
        "nvcc_version": _tool_version(args.nvcc),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
