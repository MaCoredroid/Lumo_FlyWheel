#!/usr/bin/env python3
"""Verify two isolated fixed-stride SFWD offline-codegen result trees."""

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
    "cubin_bytes": 56968,
    "cubin_sha256": (
        "e2e32ca2dd14748e26bccfd64059479d7bdb7af875714f3ec1833321f68da835"
    ),
    "elf_shared_bytes": 0,
    "encoded_sass_instructions": 792,
    "exit": 2,
    "launch_shared_bytes": 0,
    "ldg": 64,
    "ldl": 0,
    "lds": 0,
    "local_bytes": 0,
    "nop": 11,
    "num_warps": 8,
    "ptx_sha256": (
        "4d5c50435aeff0e1bac7c5a7c876a3aa60bd879949685f57030115ce5ec1fead"
    ),
    "registers": 42,
    "rows_per_program": 32,
    "sass_sha256": (
        "2f47aaf2b515ebf2716acf38323312117ae1f9a0c7e6686443cc9ebf6bb88259"
    ),
    "source_function_sha256": (
        "796c193b9cfb167119cc4675ee34241b3bb27261e98f1cd6a9d69cb71c6008a4"
    ),
    "stack_bytes": 0,
    "state_len": 34,
    "static_sass_instructions": 778,
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
            "5cc8adf2a870a27300b61668870eacba1eaff78f25d86583af632a11bc635e42"
        ),
        "ctas_per_launch": 160,
        "ctas_per_request": 160,
    },
    4: {
        "compile_hash": (
            "e013ea5248d051a2608b43f38944cb6f86f1420a90674dc35103685f7b497a58"
        ),
        "ctas_per_launch": 640,
        "ctas_per_request": 160,
    },
}
I32_ADDRESS_BASELINE = {
    "cubin_bytes": 64040,
    "encoded_sass_instructions": 928,
    "static_sass_instructions": 912,
    "ldg": 64,
    "stg": 20,
    "lds": 0,
    "launch_shared_bytes": 0,
    "elf_shared_bytes": 0,
    "num_warps": 8,
    "registers": 56,
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
        "registers",
    ):
        require(
            summary[name] < I32_ADDRESS_BASELINE[name],
            f"b{batch} {name} does not improve int32-address baseline",
        )
    for name in ("ldg", "stg", "lds", "launch_shared_bytes", "elf_shared_bytes"):
        require(
            summary[name] <= I32_ADDRESS_BASELINE[name],
            f"b{batch} {name} regresses int32-address baseline",
        )
    require(
        allocated_registers_per_thread(summary["registers"])
        == 48
        < allocated_registers_per_thread(I32_ADDRESS_BASELINE["registers"])
        == 56,
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
                    "baseline": "int32-address row32/C64 prior reuse",
                    "cubin_bytes": 56968,
                    "cubin_bytes_baseline": 64040,
                    "encoded_sass_instructions": 792,
                    "encoded_sass_instructions_baseline": 928,
                    "static_sass_instructions": 778,
                    "static_sass_instructions_baseline": 912,
                    "ldg": 64,
                    "ldg_baseline": 64,
                    "stg": 20,
                    "stg_baseline": 20,
                    "lds": 0,
                    "lds_baseline": 0,
                    "launch_shared_bytes": 0,
                    "launch_shared_bytes_baseline": 0,
                    "warp_weighted_encoded_sass": 6336,
                    "warp_weighted_encoded_sass_baseline": 7424,
                    "warp_weighted_static_sass": 6224,
                    "warp_weighted_static_sass_baseline": 7296,
                    "warp_weighted_ldg": 512,
                    "warp_weighted_ldg_baseline": 512,
                    "pass": True,
                },
                "resource_gate": {
                    "registers_per_thread": 42,
                    "registers_per_thread_baseline": 56,
                    "allocated_registers_per_thread": 48,
                    "allocated_registers_per_thread_baseline": 56,
                    "threads_per_cta": 256,
                    "registers_per_cta_reported": 10752,
                    "allocated_registers_per_cta": 12288,
                    "allocated_registers_per_cta_baseline": 14336,
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
