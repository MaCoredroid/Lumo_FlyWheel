#!/usr/bin/env python3
"""Fail-closed static audit for the fixed32 K64 B4 reused-weight head."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


INSTRUCTION = re.compile(
    r"\s*/\*[0-9a-f]+\*/(?:\s+@[!P0-9]+)?\s+([A-Z][A-Z0-9_.]+)"
)
RESOURCE = re.compile(
    r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+) "
    r"CONSTANT\[0\]:(\d+)"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _counts(sass: str) -> Counter[str]:
    return Counter(
        match.group(1)
        for line in sass.splitlines()
        if (match := INSTRUCTION.match(line)) is not None
    )


def _loop_counts(sass: str) -> Counter[str]:
    inside = False
    instructions: list[str] = []
    for line in sass.splitlines():
        if line.strip() == ".L_x_1:":
            require(not inside, "multiple steady-loop labels found")
            inside = True
            continue
        match = INSTRUCTION.match(line)
        if inside and match is not None:
            instructions.append(match.group(1))
        if inside and "BRA `(.L_x_1)" in line:
            inside = False
            break
    require(instructions and not inside, "steady-loop backedge not found")
    return Counter(instructions)


def _resources(raw: str) -> dict[str, int]:
    match = RESOURCE.search(raw)
    require(match is not None, "kernel resource record not found")
    registers, stack, shared, local, constant0 = map(int, match.groups())
    return {
        "registers_per_thread": registers,
        "stack_bytes_per_thread": stack,
        "shared_bytes_per_cta": shared,
        "local_bytes_per_thread": local,
        "constant0_bytes": constant0,
    }


def audit(sass: str, resource: str) -> dict[str, object]:
    counts = _counts(sass)
    loop = _loop_counts(sass)
    resources = _resources(resource)
    require(sum(counts.values()) == 200, "static instruction count drifted")
    require(counts["SHFL.DOWN"] == 16, "four reduction trees drifted")
    require(counts["FADD"] == 16, "four reduction add trees drifted")
    require(counts["STG.E.U16"] == 4, "B4 output stores drifted")
    require(sum(loop.values()) == 119, "steady-loop body drifted")
    require(loop["LDG.E.U16.CONSTANT"] == 40, "U8 load body drifted")
    require(loop["SHF.L.U32"] == 40, "BF16 conversion body drifted")
    require(loop["FFMA"] == 32, "four ordered FMA chains drifted")
    require(loop["BRA"] == 1, "U8 loop backedge drifted")
    for forbidden in ("BAR", "LDL", "STL", "CALL", "ATOM"):
        require(
            not any(op.startswith(forbidden) for op in counts),
            f"candidate contains forbidden {forbidden} instruction",
        )
    require(
        resources
        == {
            "registers_per_thread": 56,
            "stack_bytes_per_thread": 0,
            "shared_bytes_per_cta": 0,
            "local_bytes_per_thread": 0,
            "constant0_bytes": 928,
        },
        "candidate resources drifted",
    )
    candidate_dynamic = 40 * sum(loop.values())
    four_m1_dynamic = 4 * 1920
    require(candidate_dynamic == 4760, "candidate dynamic loop model drifted")
    return {
        "schema": "fr13.fixed32.dfwd_k64_m4_r64_u8_codegen_audit.v1",
        "status": "STATIC_CODEGEN_PASS_UNQUALIFIED",
        "performance_measurement": False,
        "gpu_used": False,
        "candidate_resources": resources,
        "candidate_static_instructions": sum(counts.values()),
        "candidate_steady_loop_instructions": sum(loop.values()),
        "candidate_dynamic_loop_instructions_per_four_rows": candidate_dynamic,
        "four_m1_u8_dynamic_loop_instructions": four_m1_dynamic,
        "dynamic_loop_instruction_reduction_fraction": (
            1.0 - candidate_dynamic / four_m1_dynamic
        ),
        "candidate_loop_loads_per_four_rows": loop["LDG.E.U16.CONSTANT"],
        "four_m1_u8_loop_loads_per_four_rows": 64,
        "weight_reuse_batch": 4,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(
        args.sass.read_text(encoding="ascii"),
        args.resource.read_text(encoding="ascii"),
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="ascii")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
