#!/usr/bin/env python3
"""Fail-closed static audit for the fixed32 K64 M1 R64-U8 head."""

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


def read_ascii(path: Path) -> str:
    return path.read_text(encoding="ascii")


def instruction_counts(sass: str) -> Counter[str]:
    return Counter(
        match.group(1)
        for line in sass.splitlines()
        if (match := INSTRUCTION.match(line)) is not None
    )


def steady_loop_counts(sass: str) -> Counter[str]:
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


def resource_record(raw: str) -> dict[str, int]:
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


def audit(
    candidate_sass: str,
    candidate_resource: str,
    baseline_sass: str,
    baseline_resource: str,
) -> dict[str, object]:
    candidate = instruction_counts(candidate_sass)
    baseline = instruction_counts(baseline_sass)
    candidate_loop = steady_loop_counts(candidate_sass)
    baseline_loop = steady_loop_counts(baseline_sass)
    candidate_resources = resource_record(candidate_resource)
    baseline_resources = resource_record(baseline_resource)

    require(sum(candidate.values()) == 96, "candidate instruction count drifted")
    require(candidate["NOP"] == 9, "candidate NOP count drifted")
    require(candidate["LDG.E.U16.CONSTANT"] == 16, "candidate loads drifted")
    require(candidate["FFMA"] == 9, "candidate FFMA count drifted")
    require(candidate["SHFL.DOWN"] == 4, "candidate shuffle tree drifted")
    require(candidate["FADD"] == 4, "candidate reduction adds drifted")
    require(candidate["STG.E.U16"] == 1, "candidate output store drifted")
    require(sum(baseline.values()) == 64, "baseline instruction count drifted")
    require(baseline["NOP"] == 14, "baseline NOP count drifted")

    require(sum(candidate_loop.values()) == 48, "candidate loop body drifted")
    require(candidate_loop["LDG.E.U16.CONSTANT"] == 16, "U8 loads drifted")
    require(candidate_loop["FFMA"] == 8, "U8 FMA order/body drifted")
    require(candidate_loop["BRA"] == 1, "U8 backedge count drifted")
    require(sum(baseline_loop.values()) == 11, "baseline loop body drifted")
    require(baseline_loop["LDG.E.U16.CONSTANT"] == 2, "baseline loads drifted")
    require(baseline_loop["FFMA"] == 1, "baseline FMA body drifted")
    require(baseline_loop["BRA"] == 1, "baseline backedge count drifted")

    for forbidden in ("BAR", "LDL", "STL", "CALL", "ATOM"):
        require(
            not any(op.startswith(forbidden) for op in candidate),
            f"candidate contains forbidden {forbidden} instruction",
        )
    require(
        candidate_resources
        == {
            "registers_per_thread": 29,
            "stack_bytes_per_thread": 0,
            "shared_bytes_per_cta": 0,
            "local_bytes_per_thread": 0,
            "constant0_bytes": 928,
        },
        "candidate resources drifted",
    )
    require(
        baseline_resources
        == {
            "registers_per_thread": 18,
            "stack_bytes_per_thread": 0,
            "shared_bytes_per_cta": 0,
            "local_bytes_per_thread": 0,
            "constant0_bytes": 928,
        },
        "baseline resources drifted",
    )

    baseline_dynamic = 320 * sum(baseline_loop.values())
    candidate_dynamic = 40 * sum(candidate_loop.values())
    require(baseline_dynamic == 3520, "baseline dynamic loop model drifted")
    require(candidate_dynamic == 1920, "candidate dynamic loop model drifted")
    return {
        "schema": "fr13.fixed32.dfwd_k64_m1_r64_u8_codegen_audit.v1",
        "status": "STATIC_CODEGEN_PASS_UNQUALIFIED",
        "performance_measurement": False,
        "gpu_used": False,
        "candidate_resources": candidate_resources,
        "baseline_resources": baseline_resources,
        "candidate_static_instructions": sum(candidate.values()),
        "baseline_static_instructions": sum(baseline.values()),
        "candidate_steady_loop_instructions": sum(candidate_loop.values()),
        "baseline_steady_loop_instructions": sum(baseline_loop.values()),
        "candidate_dynamic_loop_instructions_per_row": candidate_dynamic,
        "baseline_dynamic_loop_instructions_per_row": baseline_dynamic,
        "dynamic_loop_instruction_delta": candidate_dynamic - baseline_dynamic,
        "dynamic_loop_instruction_reduction_fraction": (
            1.0 - candidate_dynamic / baseline_dynamic
        ),
        "candidate_weight_load_window": 8,
        "baseline_weight_load_window": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-sass", type=Path, required=True)
    parser.add_argument("--candidate-resource", type=Path, required=True)
    parser.add_argument("--baseline-sass", type=Path, required=True)
    parser.add_argument("--baseline-resource", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(
        read_ascii(args.candidate_sass),
        read_ascii(args.candidate_resource),
        read_ascii(args.baseline_sass),
        read_ascii(args.baseline_resource),
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="ascii")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
