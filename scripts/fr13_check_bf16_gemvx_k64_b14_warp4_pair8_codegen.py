#!/usr/bin/env python3
"""Fail-closed SM121a audit for the fixed32 B1/B4 warp4-pair8 heads."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


INSTRUCTION = re.compile(
    r"\s*/\*[0-9a-f]+\*/(?:\s+@[!P0-9]+)?\s+([A-Z][A-Z0-9_.]+)"
)
TEXT_SECTION = re.compile(r"^\.text\.(.+):$", re.MULTILINE)
RESOURCE = re.compile(
    r"Function properties for .*?(m[14]_warp4_pair8_kernel).*?"
    r"Used (\d+) registers, used (\d+) barriers",
    re.DOTALL,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _section(sass: str, needle: str) -> str:
    matches = list(TEXT_SECTION.finditer(sass))
    selected = [index for index, match in enumerate(matches) if needle in match.group(1)]
    require(len(selected) == 1, f"expected one {needle} text section")
    index = selected[0]
    start = matches[index].start()
    end = matches[index + 1].start() if index + 1 < len(matches) else len(sass)
    return sass[start:end]


def _counts(section: str) -> Counter[str]:
    return Counter(
        match.group(1)
        for line in section.splitlines()
        if (match := INSTRUCTION.match(line)) is not None
    )


def _resources(ptxas: str) -> dict[str, dict[str, int]]:
    records: dict[str, dict[str, int]] = {}
    for name, registers, barriers in RESOURCE.findall(ptxas):
        records[name] = {
            "registers_per_thread": int(registers),
            "barriers": int(barriers),
            "stack_bytes_per_thread": 0,
            "spill_stores": 0,
            "spill_loads": 0,
        }
    require(
        set(records) == {"m1_warp4_pair8_kernel", "m4_warp4_pair8_kernel"},
        "kernel resource records drifted",
    )
    require(ptxas.count("0 bytes stack frame") == 2, "stack usage drifted")
    require(ptxas.count("0 bytes spill stores") == 2, "spill stores drifted")
    require(ptxas.count("0 bytes spill loads") == 2, "spill loads drifted")
    return records


def _audit_counts(counts: Counter[str], batch: int) -> None:
    expected = {
        1: {
            "instructions": 176,
            "NOP": 10,
            "LDG.E.128.CONSTANT": 5,
            "STG.E.U16": 4,
            "FFMA": 32,
            "SHFL.DOWN": 20,
            "FADD": 20,
            "SHF.L.U32": 17,
        },
        4: {
            "instructions": 472,
            "NOP": 8,
            "LDG.E.128.CONSTANT": 8,
            "STG.E.U16": 16,
            "FFMA": 128,
            "SHFL.DOWN": 80,
            "FADD": 80,
            "SHF.L.U32": 33,
        },
    }[batch]
    require(sum(counts.values()) == expected["instructions"], f"B{batch} instruction count drifted")
    for operation, count in expected.items():
        if operation != "instructions":
            require(counts[operation] == count, f"B{batch} {operation} count drifted")
    for forbidden in ("BAR", "LDL", "STL", "ATOM", "CALL"):
        require(
            not any(operation.startswith(forbidden) for operation in counts),
            f"B{batch} contains forbidden {forbidden} instruction",
        )


def audit(sass: str, ptxas: str) -> dict[str, object]:
    m1_counts = _counts(_section(sass, "m1_warp4_pair8_kernel"))
    m4_counts = _counts(_section(sass, "m4_warp4_pair8_kernel"))
    _audit_counts(m1_counts, 1)
    _audit_counts(m4_counts, 4)
    resources = _resources(ptxas)
    require(
        resources["m1_warp4_pair8_kernel"]["registers_per_thread"] == 40,
        "B1 register count drifted",
    )
    require(
        resources["m4_warp4_pair8_kernel"]["registers_per_thread"] == 80,
        "B4 register count drifted",
    )
    require(
        all(record["barriers"] == 0 for record in resources.values()),
        "barrier usage drifted",
    )

    ctas = 2048
    warps_per_cta = 8
    candidate_warps = ctas * warps_per_cta
    iterations = 20
    vector_bytes_per_warp = 32 * 16
    candidate_m1_load_instructions = candidate_warps * iterations * 5
    candidate_m4_load_instructions = candidate_warps * iterations * 8
    candidate_m1_requested_bytes = candidate_m1_load_instructions * vector_bytes_per_warp
    candidate_m4_requested_bytes = candidate_m4_load_instructions * vector_bytes_per_warp

    incumbent_warps = 1024 * 32
    incumbent_iterations = 40
    scalar_bytes_per_warp = 32 * 2
    incumbent_m1_load_instructions = incumbent_warps * incumbent_iterations * 16
    incumbent_m4_load_instructions = incumbent_warps * incumbent_iterations * 40
    incumbent_m1_requested_bytes = incumbent_m1_load_instructions * scalar_bytes_per_warp
    incumbent_m4_requested_bytes = incumbent_m4_load_instructions * scalar_bytes_per_warp

    return {
        "schema": "fr13.fixed32.dfwd_k64_b14_warp4_pair8_codegen_audit.v1",
        "status": "STATIC_CODEGEN_PASS_UNQUALIFIED",
        "acceptance_valid": False,
        "performance_measurement": False,
        "gpu_used": False,
        "resources": resources,
        "sass": {
            "b1_static_instructions": sum(m1_counts.values()),
            "b4_static_instructions": sum(m4_counts.values()),
            "b1_packed_loads": m1_counts["LDG.E.128.CONSTANT"],
            "b4_packed_loads": m4_counts["LDG.E.128.CONSTANT"],
        },
        "logical_global_load_model": {
            "scope": "warp_request_bytes_before_cache_coalescing; not measured DRAM traffic",
            "weight_bytes_per_call": 65536 * 5120 * 2,
            "b1": {
                "incumbent_r64_u8_load_instructions": incumbent_m1_load_instructions,
                "candidate_load_instructions": candidate_m1_load_instructions,
                "load_instruction_reduction_fraction": 1.0 - candidate_m1_load_instructions / incumbent_m1_load_instructions,
                "incumbent_r64_u8_requested_bytes": incumbent_m1_requested_bytes,
                "candidate_requested_bytes": candidate_m1_requested_bytes,
                "requested_byte_reduction_fraction": 1.0 - candidate_m1_requested_bytes / incumbent_m1_requested_bytes,
            },
            "b4": {
                "incumbent_r64_u8_load_instructions": incumbent_m4_load_instructions,
                "candidate_load_instructions": candidate_m4_load_instructions,
                "load_instruction_reduction_fraction": 1.0 - candidate_m4_load_instructions / incumbent_m4_load_instructions,
                "incumbent_r64_u8_requested_bytes": incumbent_m4_requested_bytes,
                "candidate_requested_bytes": candidate_m4_requested_bytes,
                "requested_byte_reduction_fraction": 1.0 - candidate_m4_requested_bytes / incumbent_m4_requested_bytes,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--ptxas", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(
        args.sass.read_text(encoding="ascii"),
        args.ptxas.read_text(encoding="ascii"),
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="ascii")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
