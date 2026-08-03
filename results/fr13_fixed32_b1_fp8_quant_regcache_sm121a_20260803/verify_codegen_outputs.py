#!/usr/bin/env python3
"""Verify the sanitized paired SM121a FP8-quant codegen claims."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path


STOCK_FUNCTION = (
    "_Z33per_token_group_quant_8bit_kernelIN3c108BFloat16E13__nv_fp8_e4m3"
    "Lb1ELb0EfEvPKT_PvPT3_iiifffii"
)
CANDIDATE_FUNCTION = (
    "_Z50fr13_fixed32_b1_fp8_quant_regcache_r32k5120_kernel"
    "IN3c108BFloat16E13__nv_fp8_e4m3EvPKT_PvPffff"
)
INSTRUCTION = re.compile(
    r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?P[0-9]+\s+)?([A-Z][A-Z0-9_.]+)",
    re.MULTILINE,
)


def _run(*args: str) -> str:
    result = subprocess.run(args, check=True, text=True, capture_output=True)
    return result.stdout


def _resource(text: str, function: str) -> dict[str, int]:
    marker = f"Function {function}:"
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"missing resource record for {function}")
    line = text[start:].splitlines()[1]
    return {
        key: int(value)
        for key, value in re.findall(r"(REG|STACK|SHARED|LOCAL):(\d+)", line)
    }


def _opcodes(text: str) -> list[str]:
    return INSTRUCTION.findall(text)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
    parser.add_argument("--baseline-object", type=Path, required=True)
    parser.add_argument("--candidate-object", type=Path, required=True)
    args = parser.parse_args()

    baseline_resource = _run(
        args.cuobjdump, "--dump-resource-usage", str(args.baseline_object)
    )
    candidate_resource = _run(
        args.cuobjdump, "--dump-resource-usage", str(args.candidate_object)
    )
    stock = _resource(baseline_resource, STOCK_FUNCTION)
    candidate = _resource(candidate_resource, CANDIDATE_FUNCTION)
    if stock != {"REG": 48, "STACK": 0, "SHARED": 1024, "LOCAL": 0}:
        raise RuntimeError(f"stock resources drifted: {stock}")
    if candidate != {"REG": 26, "STACK": 0, "SHARED": 1024, "LOCAL": 0}:
        raise RuntimeError(f"candidate resources drifted: {candidate}")

    stock_sass = _run(
        args.cuobjdump,
        "--dump-sass",
        "--function",
        STOCK_FUNCTION,
        str(args.baseline_object),
    )
    candidate_stock_sass = _run(
        args.cuobjdump,
        "--dump-sass",
        "--function",
        STOCK_FUNCTION,
        str(args.candidate_object),
    )
    candidate_sass = _run(
        args.cuobjdump,
        "--dump-sass",
        "--function",
        CANDIDATE_FUNCTION,
        str(args.candidate_object),
    )
    if stock_sass != candidate_stock_sass:
        raise RuntimeError("untouched stock SASS differs between paired objects")

    stock_ops = _opcodes(stock_sass)
    candidate_ops = _opcodes(candidate_sass)
    if (len(stock_ops), stock_ops.count("NOP")) != (2120, 11):
        raise RuntimeError("stock instruction image drifted")
    if (len(candidate_ops), candidate_ops.count("NOP")) != (408, 10):
        raise RuntimeError("candidate instruction image drifted")
    forbidden = tuple(
        opcode
        for opcode in candidate_ops
        if opcode.startswith(("BAR", "LDS", "STS", "LDL", "STL"))
    )
    if forbidden:
        raise RuntimeError(f"candidate contains forbidden work: {forbidden}")
    if stock_ops.count("BAR.SYNC.DEFER_BLOCKING") != 1:
        raise RuntimeError("stock barrier count drifted")
    if sum(opcode.startswith("LDS") for opcode in stock_ops) != 12:
        raise RuntimeError("stock shared-load count drifted")
    if sum(opcode.startswith("STS") for opcode in stock_ops) != 40:
        raise RuntimeError("stock shared-store count drifted")

    print(
        "PASS",
        f"baseline_sha256={_sha(args.baseline_object)}",
        f"candidate_sha256={_sha(args.candidate_object)}",
        f"stock_instructions={len(stock_ops)}",
        f"candidate_instructions={len(candidate_ops)}",
        "candidate_bar_lds_sts=0",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
