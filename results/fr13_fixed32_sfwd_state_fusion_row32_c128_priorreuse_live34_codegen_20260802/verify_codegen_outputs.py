#!/usr/bin/env python3
"""Verify two isolated row32/C128 prior-reuse offline-codegen result trees."""

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
    "block_c": 128,
    "bra": 1,
    "calls": 0,
    "cubin_bytes": 125072,
    "cubin_sha256": (
        "be2b9bf13d3e0763334db397f562c566e1e08d42f522ff66a42ba5b1ebb64d7c"
    ),
    "elf_shared_bytes": 0,
    "encoded_sass_instructions": 1872,
    "exit": 2,
    "launch_shared_bytes": 0,
    "ldg": 120,
    "ldl": 0,
    "lds": 0,
    "local_bytes": 0,
    "nop": 11,
    "num_warps": 8,
    "ptx_sha256": (
        "dc3330fd35d1637b0dc9a404e06025e8556aa3398a5fc293e27cbb4fdbee30c7"
    ),
    "registers": 80,
    "rows_per_program": 32,
    "sass_sha256": (
        "d9046acd19cb8c421c639145a625494ac0c99d826e8f53ea86a16ff8fcd85aa1"
    ),
    "source_function_sha256": (
        "957310fadf0fc995b54689e2e95c6c602bf9099660c98480fe6b68e189a3da71"
    ),
    "stack_bytes": 0,
    "state_len": 34,
    "static_sass_instructions": 1858,
    "stg": 36,
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
            "7a8741510bd87626497cf0895d1ebe112036ca0d9f3b0df19f7b3df46ccbe286"
        ),
        "ctas_per_launch": 80,
        "ctas_per_request": 80,
    },
    4: {
        "compile_hash": (
            "ae602686d19352daab4ae253ef74a07a337f7d300cebacb1ec10753310d7d7ea"
        ),
        "ctas_per_launch": 320,
        "ctas_per_request": 80,
    },
}
ROW32_C64_PRIORREUSE_BASELINE = {
    "ctas_per_request": 160,
    "encoded_sass_instructions": 1008,
    "static_sass_instructions": 993,
    "ldg": 64,
    "stg": 20,
    "launch_shared_bytes": 0,
    "num_warps": 8,
    "registers": 62,
}
GB10_SM_COUNT = 48
BLACKWELL_REGISTERS_PER_SM = 65536


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
    for name in ("encoded_sass_instructions", "static_sass_instructions", "ldg", "stg"):
        candidate_total = (
            summary[name] * summary["num_warps"] * summary["ctas_per_request"]
        )
        baseline_total = (
            ROW32_C64_PRIORREUSE_BASELINE[name]
            * ROW32_C64_PRIORREUSE_BASELINE["num_warps"]
            * ROW32_C64_PRIORREUSE_BASELINE["ctas_per_request"]
        )
        require(
            candidate_total < baseline_total,
            f"b{batch} launch-total {name} does not improve row32/C64 prior-reuse",
        )
    candidate_allocated = allocated_registers_per_thread(summary["registers"])
    baseline_allocated = allocated_registers_per_thread(
        ROW32_C64_PRIORREUSE_BASELINE["registers"]
    )
    require(
        candidate_allocated == 80 and baseline_allocated == 64,
        f"b{batch} allocated register accounting drift",
    )
    require(
        BLACKWELL_REGISTERS_PER_SM // (candidate_allocated * 256) == 3
        < BLACKWELL_REGISTERS_PER_SM // (baseline_allocated * 256)
        == 4,
        f"b{batch} register residency accounting drift",
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
                "ctas_per_request": 80,
                "ctas_per_launch_by_batch": {"1": 80, "4": 320},
                "b1_b4_binary_identity": True,
                "fresh_cache_binary_identity": True,
                "fresh_disassembly_identity": True,
                "improvement_gate": {
                    "baseline": "row32/C64 prior-reuse live34",
                    "ctas_per_request": 80,
                    "ctas_per_request_baseline": 160,
                    "cubin_bytes": 125072,
                    "cubin_bytes_baseline": 69640,
                    "encoded_sass_instructions": 1872,
                    "encoded_sass_instructions_baseline": 1008,
                    "static_sass_instructions": 1858,
                    "static_sass_instructions_baseline": 993,
                    "ldg": 120,
                    "ldg_baseline": 64,
                    "stg": 36,
                    "stg_baseline": 20,
                    "lds": 0,
                    "lds_baseline": 0,
                    "launch_shared_bytes": 0,
                    "launch_shared_bytes_baseline": 0,
                    "launch_total_warp_weighted_encoded_sass": 1198080,
                    "launch_total_warp_weighted_encoded_sass_baseline": 1290240,
                    "launch_total_warp_weighted_static_sass": 1189120,
                    "launch_total_warp_weighted_static_sass_baseline": 1271040,
                    "launch_total_warp_weighted_ldg": 76800,
                    "launch_total_warp_weighted_ldg_baseline": 81920,
                    "launch_total_warp_weighted_stg": 23040,
                    "launch_total_warp_weighted_stg_baseline": 25600,
                    "pass": True,
                },
                "resource_gate": {
                    "registers_per_thread": 80,
                    "registers_per_thread_baseline": 62,
                    "allocated_registers_per_thread": 80,
                    "allocated_registers_per_thread_baseline": 64,
                    "threads_per_cta": 256,
                    "registers_per_cta_reported": 20480,
                    "allocated_registers_per_cta": 20480,
                    "allocated_registers_per_cta_baseline": 16384,
                    "launch_shared_bytes": 0,
                    "stack_bytes": 0,
                    "local_bytes": 0,
                    "spill_loads": 0,
                    "spill_stores": 0,
                    "calls": 0,
                    "viable_offline": True,
                },
                "occupancy_risk": {
                    "gb10_sms": GB10_SM_COUNT,
                    "registers_per_sm": BLACKWELL_REGISTERS_PER_SM,
                    "register_limited_ctas_per_sm": 3,
                    "register_limited_ctas_per_sm_baseline": 4,
                    "register_limited_warps_per_sm": 24,
                    "register_limited_warps_per_sm_baseline": 32,
                    "b1_grid_ctas_per_sm_mean": 80 / GB10_SM_COUNT,
                    "b1_grid_ctas_per_sm_mean_baseline": 160 / GB10_SM_COUNT,
                    "runtime_required": True,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
