#!/usr/bin/env python3
"""Fail-closed static admission gate for the fixed32 FA2 qrow32 kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fr13_patch_fa2_tree_bias import (  # noqa: E402
    FIXED32_QUERY_TILE32_TRANSLATION_UNIT,
)


TARGET_ARCH = "sm_121a"
TARGET_KERNEL = (
    "_ZN5flash36fr13_flash_fwd_fixed32_qrow32_kernel"
    "ENS_16Flash_fwd_paramsE"
)
QROW_LAUNCHER_MANGLED_FRAGMENT = "fr13_run_mha_fwd_fixed32_qrow32"
EXPECTED_STATIC_SHARED_BYTES = 1024
EXPECTED_DYNAMIC_SHARED_BYTES = 80 * 1024
MAX_REGISTERS = 254


class GateError(RuntimeError):
    pass


def _regular(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise GateError(f"{label} is missing, non-regular, or a symlink: {path}")
    return path


def _read_ascii(path: Path, label: str) -> str:
    try:
        return _regular(path, label).read_text(encoding="ascii")
    except UnicodeDecodeError as error:
        raise GateError(f"{label} is not ASCII: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_records(path: Path, label: str) -> tuple[str, ...]:
    text = _read_ascii(path, label)
    records = tuple(line for line in text.splitlines() if line)
    if not records:
        raise GateError(f"{label} is empty")
    if records != tuple(sorted(set(records))):
        raise GateError(f"{label} is not sorted and unique")
    return records


def _matching_abi_records(
    stock_path: Path,
    candidate_path: Path,
    label: str,
) -> dict[str, Any]:
    stock = _canonical_records(stock_path, f"stock {label}")
    candidate = _canonical_records(candidate_path, f"candidate {label}")
    if stock != candidate:
        stock_set = set(stock)
        candidate_set = set(candidate)
        raise GateError(
            f"{label} ABI drifted: "
            f"removed={len(stock_set - candidate_set)} "
            f"added={len(candidate_set - stock_set)}"
        )
    return {
        "records": len(stock),
        "sha256": _sha256(stock_path),
    }


def _parse_resources(text: str) -> dict[str, int]:
    if text.count(f"arch = {TARGET_ARCH}") != 1:
        raise GateError(f"resource usage is not exactly one {TARGET_ARCH} cubin")
    marker = f" Function {TARGET_KERNEL}:"
    if text.count(marker) != 1:
        raise GateError("resource usage does not contain exactly one target kernel")
    suffix = text.split(marker, 1)[1]
    match = re.search(
        r"^\s*REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)\b",
        suffix,
        flags=re.MULTILINE,
    )
    if match is None:
        raise GateError("target resource record is missing or malformed")
    resources = {
        "registers": int(match.group(1)),
        "stack_bytes": int(match.group(2)),
        "static_shared_bytes": int(match.group(3)),
        "static_local_bytes": int(match.group(4)),
    }
    if not 1 <= resources["registers"] <= MAX_REGISTERS:
        raise GateError("target register count exceeds the admitted ceiling")
    if resources["stack_bytes"] != 0:
        raise GateError("target kernel has a nonzero stack frame")
    if resources["static_local_bytes"] != 0:
        raise GateError("target kernel has static local memory")
    if resources["static_shared_bytes"] != EXPECTED_STATIC_SHARED_BYTES:
        raise GateError("target static shared memory drifted")
    return resources


def _verify_ptxas(text: str, registers: int) -> dict[str, int]:
    entry = f"Compiling entry function '{TARGET_KERNEL}' for '{TARGET_ARCH}'"
    if text.count(entry) != 1:
        raise GateError("ptxas log does not identify exactly one target SM121a kernel")
    properties = f"Function properties for {TARGET_KERNEL}"
    if text.count(properties) != 1:
        raise GateError("ptxas target function properties are missing or duplicated")
    suffix = text.split(properties, 1)[1]
    spill = re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
        r"(\d+) bytes spill loads",
        suffix,
    )
    used = re.search(r"Used (\d+) registers, used (\d+) barriers", suffix)
    if spill is None or used is None:
        raise GateError("ptxas resource summary is incomplete")
    ptxas = {
        "stack_bytes": int(spill.group(1)),
        "spill_store_bytes": int(spill.group(2)),
        "spill_load_bytes": int(spill.group(3)),
        "registers": int(used.group(1)),
        "barriers": int(used.group(2)),
    }
    if any(
        ptxas[key] != 0
        for key in ("stack_bytes", "spill_store_bytes", "spill_load_bytes")
    ):
        raise GateError("ptxas reports a stack frame or spills")
    if ptxas["registers"] != registers:
        raise GateError("ptxas and cuobjdump register counts differ")
    return ptxas


def _sass_opcode_count(text: str, opcode: str) -> int:
    instruction = re.compile(
        rf"/\*[0-9a-f]+\*/\s+(?:@[!A-Z0-9.]+\s+)?{opcode}(?:\.|\s)",
        flags=re.IGNORECASE,
    )
    return len(instruction.findall(text))


def _verify_sass(text: str) -> dict[str, int]:
    if text.count(f"Function : {TARGET_KERNEL}") != 1:
        raise GateError("SASS does not contain exactly one target kernel")
    counts = {
        opcode.lower(): _sass_opcode_count(text, opcode)
        for opcode in ("LDL", "STL", "CALL")
    }
    if any(counts.values()):
        raise GateError("target SASS contains local-memory or call instructions")
    return counts


def verify_static(args: argparse.Namespace) -> dict[str, Any]:
    source = _regular(args.qrow_source, "qrow32 translation unit")
    expected_source = FIXED32_QUERY_TILE32_TRANSLATION_UNIT.encode("ascii")
    if source.read_bytes() != expected_source:
        raise GateError("qrow32 translation unit differs from the gated generator")
    if expected_source.count(b"static_assert(smem_size == 80 * 1024);") != 1:
        raise GateError("qrow32 dynamic shared-memory contract drifted")

    qrow_object = _regular(args.qrow_object, "qrow32 SM121a object")
    stock_so = _regular(args.stock_so, "stock FA2 shared object")
    candidate_so = _regular(args.candidate_so, "candidate FA2 shared object")

    elf_list = _read_ascii(args.elf_list, "cuobjdump ELF list")
    elf_records = tuple(
        line.strip() for line in elf_list.splitlines() if line.startswith("ELF file")
    )
    if len(elf_records) != 1 or not elf_records[0].endswith(
        f".1.{TARGET_ARCH}.cubin"
    ):
        raise GateError("qrow32 object is not exactly one SM121a cubin")

    resources = _parse_resources(
        _read_ascii(args.resource_usage, "cuobjdump resource usage")
    )
    resources["dynamic_shared_bytes"] = EXPECTED_DYNAMIC_SHARED_BYTES
    ptxas = _verify_ptxas(
        _read_ascii(args.ptxas_log, "ptxas log"),
        resources["registers"],
    )
    sass = _verify_sass(_read_ascii(args.sass, "target SASS"))

    abi = {
        "defined_dynamic": _matching_abi_records(
            args.stock_defined,
            args.candidate_defined,
            "defined dynamic symbols",
        ),
        "undefined_dynamic": _matching_abi_records(
            args.stock_undefined,
            args.candidate_undefined,
            "undefined dynamic symbols",
        ),
        "dt_needed": _matching_abi_records(
            args.stock_needed,
            args.candidate_needed,
            "DT_NEEDED",
        ),
    }
    candidate_defined = _read_ascii(
        args.candidate_defined,
        "candidate defined dynamic symbols",
    )
    if QROW_LAUNCHER_MANGLED_FRAGMENT in candidate_defined:
        raise GateError("qrow32 launcher leaked into the dynamic symbol table")

    result = {
        "schema": "fr13.fixed32.fa2_qrow32_static_admission.v1",
        "status": "PASS",
        "target": {
            "arch": TARGET_ARCH,
            "head_dim": 256,
            "block_m": 32,
            "block_n": 64,
            "warps": 2,
            "threads": 64,
            "split_k": False,
        },
        "source_sha256": hashlib.sha256(expected_source).hexdigest(),
        "object_sha256": _sha256(qrow_object),
        "stock_so_sha256": _sha256(stock_so),
        "candidate_so_sha256": _sha256(candidate_so),
        "resources": resources,
        "ptxas": ptxas,
        "sass": sass,
        "abi": abi,
        "gpu_used": False,
        "performance_measurement": False,
        "production_eligible": False,
        "timing_eligible": False,
        "default_off": True,
        "required_next_gate": (
            "canonical real SWE-Verified exact4 B4 raw-byte A/B for Tail23 "
            "and Hydra27"
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrow-source", type=Path, required=True)
    parser.add_argument("--qrow-object", type=Path, required=True)
    parser.add_argument("--stock-so", type=Path, required=True)
    parser.add_argument("--candidate-so", type=Path, required=True)
    parser.add_argument("--elf-list", type=Path, required=True)
    parser.add_argument("--resource-usage", type=Path, required=True)
    parser.add_argument("--ptxas-log", type=Path, required=True)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--stock-defined", type=Path, required=True)
    parser.add_argument("--candidate-defined", type=Path, required=True)
    parser.add_argument("--stock-undefined", type=Path, required=True)
    parser.add_argument("--candidate-undefined", type=Path, required=True)
    parser.add_argument("--stock-needed", type=Path, required=True)
    parser.add_argument("--candidate-needed", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = verify_static(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
