#!/usr/bin/env python3
"""Offline-only Triton SM121a compile and static cubin/SASS audit."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import linecache
import os
from pathlib import Path
import re
import subprocess


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("refusing to run unless CUDA_VISIBLE_DEVICES is explicitly empty")

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


KERNEL_NAME = "_fr13_fixed32_sfwd_prior_reuse_kernel"
SIGNATURE = {
    "x": "*bf16",
    "conv_state": "*bf16",
    "spec_state_indices": "*i32",
    "source_flat": "*i64",
    "conv_weights": "*bf16",
    "bias": "*bf16",
    "out": "*bf16",
    "source_stage": "*bf16",
    "x_stride_row": "i32",
    "conv_stride_row": "i32",
    "conv_stride_c": "i32",
    "conv_stride_l": "i32",
    "ssi_stride_b": "i32",
    "ssi_stride_s": "i32",
    "weight_stride_c": "i32",
    "weight_stride_w": "i32",
}
BASE_CONSTANTS = {
    "N": 32,
    "C": 10240,
    "WIDTH": 4,
    "SOURCE_ROWS": 36,
    "HAS_BIAS": False,
}
INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@[!]?P\d+\s+)?([A-Z][A-Z0-9.]*)\b"
)
RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
TOOLKIT_RE = re.compile(r"Tool Kit Version:\s*([^\s]+)")
TOOL_NAME_RE = re.compile(r"Tool Name:\s*(\S+)")
TOOL_VERSION_RE = re.compile(r"Tool Version:\s*(.+)")
CONTROL_OR_PADDING = {"NOP", "BAR", "BRA", "EXIT"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_at_revision(repo: Path, revision: str, relative_path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{relative_path}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout


def jit_function_from_source(source: str, canonical_path: str):
    tree = ast.parse(source, filename=canonical_path)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == KERNEL_NAME
    )
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    end = node.end_lineno
    lines = source.splitlines(keepends=True)
    synthetic = "\n" * (start - 1) + "".join(lines[start - 1 : end])
    linecache.cache[canonical_path] = (
        len(synthetic),
        None,
        synthetic.splitlines(keepends=True),
        canonical_path,
    )
    namespace = {
        "__name__": "fr13_sfwd_offline_codegen",
        "__file__": canonical_path,
        "triton": triton,
        "tl": tl,
    }
    exec(compile(synthetic, canonical_path, "exec"), namespace)
    return namespace[KERNEL_NAME], "".join(lines[start - 1 : end])


def run_text(argv: list[str]) -> str:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout


def compile_one(
    *,
    source: str,
    canonical_path: str,
    out_dir: Path,
    batch: int,
    rows_per_program: int,
    block_c: int,
    state_len: int,
    num_warps: int,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fn, function_source = jit_function_from_source(source, canonical_path)
    constants = dict(BASE_CONSTANTS)
    constants.update(
        B=batch,
        ROWS_PER_PROGRAM=rows_per_program,
        BLOCK_C=block_c,
        STATE_LEN=state_len,
    )
    target = GPUTarget("cuda", 121, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": num_warps, "num_stages": 3})
    compiled = triton.compile(
        ASTSource(fn=fn, signature=SIGNATURE, constexprs=constants),
        target=target,
        options=options.__dict__,
    )

    for name, value in compiled.asm.items():
        output = out_dir / f"kernel.{name}"
        if isinstance(value, bytes):
            output.write_bytes(value)
        elif isinstance(value, str):
            output.write_text(value)

    cubin = compiled.asm["cubin"]
    cubin_path = out_dir / "kernel.cubin"
    sass = run_text(["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin_path)])
    resource = run_text(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(cubin_path)]
    )
    elf = run_text(["/usr/local/cuda/bin/cuobjdump", "--dump-elf", str(cubin_path)])
    (out_dir / "kernel.sass").write_text(sass)
    (out_dir / "resource.txt").write_text(resource)
    (out_dir / "elf.txt").write_text(elf)

    resource_match = RESOURCE_RE.search(resource)
    if resource_match is None:
        raise RuntimeError(f"cannot parse resource usage:\n{resource}")
    registers, stack_bytes, elf_shared_bytes, local_bytes = map(
        int, resource_match.groups()
    )
    toolkit_match = TOOLKIT_RE.search(elf)
    tool_name_match = TOOL_NAME_RE.search(elf)
    tool_version_match = TOOL_VERSION_RE.search(elf)
    if None in (toolkit_match, tool_name_match, tool_version_match):
        raise RuntimeError("cannot parse embedded backend producer from cubin ELF")
    operations: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            operations[match.group(1).split(".", 1)[0]] += 1
    encoded_instructions = sum(operations.values())
    static_body_instructions = sum(
        count
        for operation, count in operations.items()
        if operation not in CONTROL_OR_PADDING
    )
    calls = sum(
        count for operation, count in operations.items() if operation.startswith("CALL")
    )
    metadata = (
        compiled.metadata._asdict()
        if hasattr(compiled.metadata, "_asdict")
        else vars(compiled.metadata)
    )
    result = {
        "batch": batch,
        "rows_per_program": rows_per_program,
        "block_c": block_c,
        "state_len": state_len,
        "num_warps": num_warps,
        "ctas_per_request": (32 // rows_per_program)
        * ((10240 + block_c - 1) // block_c),
        "ctas_per_launch": batch
        * (32 // rows_per_program)
        * ((10240 + block_c - 1) // block_c),
        "compile_hash": metadata["hash"],
        "cubin_sha256": sha256(cubin),
        "cubin_bytes": len(cubin),
        "source_function_sha256": sha256(function_source.encode()),
        "registers": registers,
        "stack_bytes": stack_bytes,
        "local_bytes": local_bytes,
        "elf_shared_bytes": elf_shared_bytes,
        "launch_shared_bytes": int(metadata["shared"]),
        "encoded_sass_instructions": encoded_instructions,
        "static_sass_instructions": static_body_instructions,
        "ldg": operations["LDG"],
        "stg": operations["STG"],
        "lds": operations["LDS"],
        "sts": operations["STS"],
        "ldl": operations["LDL"],
        "stl": operations["STL"],
        "calls": calls,
        "bar": operations["BAR"],
        "bra": operations["BRA"],
        "exit": operations["EXIT"],
        "nop": operations["NOP"],
        "sass_sha256": sha256(sass.encode()),
        "ptx_sha256": sha256(compiled.asm["ptx"].encode()),
        "backend_producer": {
            "toolkit_version": toolkit_match.group(1),
            "tool_name": tool_name_match.group(1),
            "tool_version": tool_version_match.group(1).strip(),
            "target": "sm_121a",
        },
        "toolchain": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__,
            "target": "sm_121a",
            "num_stages": 3,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--canonical-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-program", type=int, required=True)
    parser.add_argument("--block-c", type=int, default=64)
    parser.add_argument("--state-len", type=int, default=34)
    parser.add_argument("--num-warps", type=int, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()
    if 32 % args.rows_per_program != 0:
        raise SystemExit("rows-per-program must divide the fixed 32 rows")
    if 10240 % args.block_c != 0:
        raise SystemExit("block-c must divide the fixed 10240 channels")
    relative_path = "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse.py"
    source = source_at_revision(args.repo, args.revision, relative_path)
    results = []
    for batch in args.batches:
        result = compile_one(
            source=source,
            canonical_path=args.canonical_path,
            out_dir=args.output / f"b{batch}",
            batch=batch,
            rows_per_program=args.rows_per_program,
            block_c=args.block_c,
            state_len=args.state_len,
            num_warps=args.num_warps,
        )
        results.append(result)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
