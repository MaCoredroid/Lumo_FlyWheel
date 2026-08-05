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


def _require_exact_b4_dataflow(sass: str) -> None:
    lines = sass.splitlines()
    shuffle_lines = [line for line in lines if " SHFL.DOWN " in line]
    for stride in (8, 4, 2, 1):
        require(
            sum(f", 0x{stride:x}, 0x101f ;" in line for line in shuffle_lines) == 4,
            f"width-16 stride-{stride} reduction group drifted",
        )

    store_lines = [line for line in lines if " STG.E.U16 " in line]
    for address in (
        "desc[UR4][R2.64]",
        "desc[UR4][R2.64+0x20000]",
        "desc[UR4][R2.64+0x40000]",
        "desc[UR4][R2.64+0x60000]",
    ):
        require(
            sum(address in line for line in store_lines) == 1,
            f"exact-B4 output address {address} drifted",
        )

    loop_start = sass.index(".L_x_1:")
    loop_end = sass.index("BRA `(.L_x_1)", loop_start)
    loop = sass[loop_start:loop_end]
    require(
        loop.count("ISETP.GE.U32.AND P0, PT, R11.reuse, 0x1380, PT") == 1,
        "K=5120 loop bound drifted",
    )
    require(
        loop.count("IADD3 R11, PT, PT, R11, 0x80, RZ") == 1,
        "U8 K-loop increment drifted",
    )
    require(
        loop.count("IADD3 R8, PT, PT, R8, 0x80, RZ") == 1,
        "U8 weight-row increment drifted",
    )
    require(
        loop.count("IADD.64 R2, R2, 0x100") == 1,
        "U8 input-pointer increment drifted",
    )

    for step in range(8):
        byte_step = step * 0x20
        weight_offset = "" if byte_step == 0 else f"+0x{byte_step:x}"
        require(
            loop.count(f"desc[UR4][R4.64{weight_offset}]") == 1,
            f"shared weight load step {step} drifted",
        )
        for batch_base in (-0x5000, -0x2800, 0, 0x2800):
            byte_offset = batch_base + byte_step
            if byte_offset < 0:
                input_offset = f"+-0x{-byte_offset:x}"
            elif byte_offset > 0:
                input_offset = f"+0x{byte_offset:x}"
            else:
                input_offset = ""
            require(
                loop.count(f"desc[UR4][R2.64{input_offset}]") == 1,
                f"B4 input load batch offset {batch_base:#x} step {step} drifted",
            )


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
    _require_exact_b4_dataflow(sass)
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
