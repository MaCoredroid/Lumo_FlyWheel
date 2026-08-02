#!/usr/bin/env python3
"""Verify two isolated row32/C64 prior-reuse offline-codegen result trees."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess


INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@[!]?P\d+\s+)?([A-Z][A-Z0-9.]*)\b"
)
RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
CONTROL_OR_PADDING = {"NOP", "BAR", "BRA", "EXIT"}
EXPECTED_COMMON = {
    "backend_producer": {
        "target": "sm_121a",
        "tool_name": "ptxas-blackwell",
        "tool_version": "Cuda compilation tools, release 12.9, V12.9.86",
        "toolkit_version": "12.9",
    },
    "bar": 0,
    "block_c": 64,
    "bra": 1,
    "calls": 0,
    "cubin_bytes": 69088,
    "cubin_sha256": (
        "8ed5af3a8efaf7eff45a048a9689c09f5d5440223a63dcd2220218573c453adc"
    ),
    "elf_shared_bytes": 0,
    "encoded_sass_instructions": 1008,
    "exit": 2,
    "launch_shared_bytes": 0,
    "ldg": 64,
    "ldl": 0,
    "lds": 0,
    "local_bytes": 0,
    "nop": 12,
    "num_warps": 8,
    "ptx_sha256": (
        "c18444ff35fdb36346c224bb9fd1cb168c68d9199cc2c58718ff621aab392d06"
    ),
    "registers": 62,
    "rows_per_program": 32,
    "sass_sha256": (
        "388c936ff7cb1df40e990fdfc6779631285a476615deb393166a1a1ea5bdadf2"
    ),
    "source_function_sha256": (
        "f91aa4712aee1984561bb8f68b56d68d610e0c82d66a885cd899cb053dcf2529"
    ),
    "stack_bytes": 0,
    "state_len": 34,
    "static_sass_instructions": 993,
    "stg": 20,
    "stl": 0,
    "sts": 0,
    "toolchain": {
        "num_stages": 3,
        "target": "sm_121a",
        "torch": "2.10.0+cu130",
        "torch_cuda": "13.0",
        "triton": "3.6.0",
    },
}
EXPECTED_BY_BATCH = {
    1: {
        "compile_hash": (
            "ce7d9c5744afa320617ba6f8ab7755cc05294f6e5b62a89a0bd00274062c26fb"
        ),
        "ctas_per_launch": 160,
        "ctas_per_request": 160,
    },
    4: {
        "compile_hash": (
            "f02a49734f3874eb10cf2e5b26df3223c506ff3d16a6a8000e8af0076d44b1ba"
        ),
        "ctas_per_launch": 640,
        "ctas_per_request": 160,
    },
}
ROW32_C64_FINALTAP_BASELINE = {
    "cubin_bytes": 74152,
    "encoded_sass_instructions": 1088,
    "static_sass_instructions": 1071,
    "ldg": 91,
    "lds": 3,
    "launch_shared_bytes": 4096,
    "elf_shared_bytes": 1024,
    "num_warps": 8,
    "registers": 106,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def allocated_registers_per_thread(registers: int) -> int:
    """Round a warp allocation to the 256-register hardware quantum."""
    registers_per_warp = registers * 32
    return ((registers_per_warp + 255) // 256 * 256) // 32


def run_bytes(argv: list[str]) -> bytes:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def operations(sass: str) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            counts[match.group(1).split(".", 1)[0]] += 1
    return counts


def verify_variant(root: Path, batch: int) -> tuple[dict[str, object], dict[str, bytes]]:
    variant = root / f"b{batch}"
    summary = json.loads((variant / "summary.json").read_text())
    expected = dict(EXPECTED_COMMON)
    expected.update(EXPECTED_BY_BATCH[batch], batch=batch)
    require(summary == expected, f"b{batch} summary does not match expected metrics")

    blobs = {
        suffix: (variant / f"kernel.{suffix}").read_bytes()
        for suffix in ("cubin", "sass", "ptx")
    }
    resource = (variant / "resource.txt").read_bytes()
    elf = (variant / "elf.txt").read_bytes()
    require(len(blobs["cubin"]) == summary["cubin_bytes"], f"b{batch} cubin size")
    for suffix in ("cubin", "sass", "ptx"):
        require(
            sha256(blobs[suffix]) == summary[f"{suffix}_sha256"],
            f"b{batch} {suffix} hash",
        )

    cubin_path = str(variant / "kernel.cubin")
    require(
        run_bytes(["/usr/local/cuda/bin/nvdisasm", "-c", cubin_path])
        == blobs["sass"],
        f"b{batch} fresh disassembly differs",
    )
    require(
        run_bytes(
            ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", cubin_path]
        )
        == resource,
        f"b{batch} fresh resource report differs",
    )
    require(
        run_bytes(["/usr/local/cuda/bin/cuobjdump", "--dump-elf", cubin_path])
        == elf,
        f"b{batch} fresh ELF report differs",
    )

    ptx = blobs["ptx"].decode()
    require(re.search(r"(?m)^\.target sm_121a$", ptx) is not None, f"b{batch} target")
    require(
        re.search(r"(?m)^\s*\.reqntid\s+256\s*$", ptx) is not None,
        f"b{batch} thread count",
    )
    elf_text = elf.decode()
    for fragment in (
        "Tool Kit Version: 12.9",
        "Tool Name: ptxas-blackwell",
        "Tool Version: Cuda compilation tools, release 12.9, V12.9.86",
        "-arch sm_121a",
    ):
        require(fragment in elf_text, f"b{batch} backend producer {fragment}")

    sass_counts = operations(blobs["sass"].decode())
    encoded = sum(sass_counts.values())
    static = sum(
        count
        for operation, count in sass_counts.items()
        if operation not in CONTROL_OR_PADDING
    )
    observed_operations = {
        "encoded_sass_instructions": encoded,
        "static_sass_instructions": static,
        "ldg": sass_counts["LDG"],
        "stg": sass_counts["STG"],
        "lds": sass_counts["LDS"],
        "sts": sass_counts["STS"],
        "ldl": sass_counts["LDL"],
        "stl": sass_counts["STL"],
        "calls": sum(
            count
            for operation, count in sass_counts.items()
            if operation.startswith("CALL")
        ),
        "bar": sass_counts["BAR"],
        "bra": sass_counts["BRA"],
        "exit": sass_counts["EXIT"],
        "nop": sass_counts["NOP"],
    }
    for name, value in observed_operations.items():
        require(value == summary[name], f"b{batch} independent {name} count")

    resource_match = RESOURCE_RE.search(resource.decode())
    require(resource_match is not None, f"b{batch} resource report parse")
    registers, stack_bytes, elf_shared_bytes, local_bytes = map(
        int, resource_match.groups()
    )
    require(registers == summary["registers"], f"b{batch} register count")
    require(stack_bytes == summary["stack_bytes"], f"b{batch} stack bytes")
    require(local_bytes == summary["local_bytes"], f"b{batch} local bytes")
    require(elf_shared_bytes == summary["elf_shared_bytes"], f"b{batch} shared bytes")

    # These conservative per-CTA gates establish compile viability, not occupancy.
    require(registers <= 255, f"b{batch} unsafe registers per thread")
    require(8 * 32 <= 1024, f"b{batch} unsafe threads per CTA")
    require(summary["launch_shared_bytes"] <= 48 * 1024, f"b{batch} shared memory")
    require(stack_bytes == local_bytes == 0, f"b{batch} stack/local usage")
    require(summary["ldl"] == summary["stl"] == 0, f"b{batch} spill operations")
    require(summary["calls"] == 0, f"b{batch} calls")
    for name in (
        "cubin_bytes",
        "encoded_sass_instructions",
        "static_sass_instructions",
        "ldg",
        "lds",
        "launch_shared_bytes",
        "elf_shared_bytes",
    ):
        require(
            summary[name] < ROW32_C64_FINALTAP_BASELINE[name],
            f"b{batch} {name} does not improve row32/C64 finaltap",
        )
    require(
        allocated_registers_per_thread(summary["registers"])
        == 64
        < allocated_registers_per_thread(ROW32_C64_FINALTAP_BASELINE["registers"])
        == 112,
        f"b{batch} allocated register quantum did not improve",
    )
    return summary, {**blobs, "resource": resource, "elf": elf}


def normalized(summary: dict[str, object]) -> dict[str, object]:
    result = dict(summary)
    for key in ("batch", "compile_hash", "ctas_per_launch"):
        result.pop(key)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--rebuild", type=Path, required=True)
    args = parser.parse_args()

    verified: dict[str, dict[int, tuple[dict[str, object], dict[str, bytes]]]] = {}
    for label, root in (("primary", args.primary), ("rebuild", args.rebuild)):
        variants = {batch: verify_variant(root, batch) for batch in (1, 4)}
        require(
            normalized(variants[1][0]) == normalized(variants[4][0]),
            f"{label} B1/B4 non-launch metrics differ",
        )
        for suffix in ("cubin", "sass", "ptx", "resource", "elf"):
            require(
                variants[1][1][suffix] == variants[4][1][suffix],
                f"{label} B1/B4 {suffix} identity",
            )
        verified[label] = variants

    for batch in (1, 4):
        require(
            verified["primary"][batch][0] == verified["rebuild"][batch][0],
            f"b{batch} fresh-cache summary identity",
        )
        for suffix in ("cubin", "sass", "ptx", "resource", "elf"):
            require(
                verified["primary"][batch][1][suffix]
                == verified["rebuild"][batch][1][suffix],
                f"b{batch} fresh-cache {suffix} identity",
            )

    print(
        json.dumps(
            {
                "status": "pass",
                "target": "sm_121a",
                "backend_producer": EXPECTED_COMMON["backend_producer"],
                "batches": [1, 4],
                "ctas_per_request": 160,
                "ctas_per_launch_by_batch": {"1": 160, "4": 640},
                "b1_b4_binary_identity": True,
                "fresh_cache_binary_identity": True,
                "fresh_disassembly_identity": True,
                "improvement_gate": {
                    "baseline": "row32/C64 finaltap live34",
                    "cubin_bytes": 69088,
                    "cubin_bytes_baseline": 74152,
                    "encoded_sass_instructions": 1008,
                    "encoded_sass_instructions_baseline": 1088,
                    "static_sass_instructions": 993,
                    "static_sass_instructions_baseline": 1071,
                    "ldg": 64,
                    "ldg_baseline": 91,
                    "lds": 0,
                    "lds_baseline": 3,
                    "launch_shared_bytes": 0,
                    "launch_shared_bytes_baseline": 4096,
                    "warp_weighted_encoded_sass": 8064,
                    "warp_weighted_encoded_sass_baseline": 8704,
                    "warp_weighted_static_sass": 7944,
                    "warp_weighted_static_sass_baseline": 8568,
                    "warp_weighted_ldg": 512,
                    "warp_weighted_ldg_baseline": 728,
                    "pass": True,
                },
                "resource_gate": {
                    "registers_per_thread": 62,
                    "registers_per_thread_baseline": 106,
                    "allocated_registers_per_thread": 64,
                    "allocated_registers_per_thread_baseline": 112,
                    "threads_per_cta": 256,
                    "registers_per_cta_reported": 15872,
                    "allocated_registers_per_cta": 16384,
                    "allocated_registers_per_cta_baseline": 28672,
                    "launch_shared_bytes": 0,
                    "stack_bytes": 0,
                    "local_bytes": 0,
                    "spill_loads": 0,
                    "spill_stores": 0,
                    "calls": 0,
                    "viable_offline": True,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
