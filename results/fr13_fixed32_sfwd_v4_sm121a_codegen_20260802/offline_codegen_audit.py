#!/usr/bin/env python3
"""Compile the fixed32 SFWD v3/v4 kernels for SM121a without a GPU."""

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
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


SOURCE_PATH = (
    "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py"
)
KERNEL_NAME = "_fr13_fixed32_sfwd_channel_serial_kernel"
SIGNATURE = {
    "x": "*bf16",
    "conv_state": "*bf16",
    "spec_state_indices": "*i32",
    "conv_weights": "*bf16",
    "bias": "*bf16",
    "out": "*bf16",
    "source_stage": "*bf16",
}
BASE_CONSTANTS = {
    "CONV_STRIDE_ROW": 348160,
    "N": 32,
    "C": 10240,
    "WIDTH": 4,
    "STATE_LEN": 34,
    "SOURCE_ROWS": 36,
    "HAS_BIAS": False,
    "X_STRIDE_ROW": 16384,
}
DEPLOYMENT_CONFIGS = {
    1: {"block_c": 128, "num_warps": 2},
    4: {"block_c": 256, "num_warps": 4},
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


def run_bytes(argv: list[str]) -> bytes:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def source_at_revision(repo: Path, revision: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{SOURCE_PATH}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout


def kernel_node(source: str) -> ast.FunctionDef:
    tree = ast.parse(source, filename=SOURCE_PATH)
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == KERNEL_NAME
    )


def jit_function(source: str, revision: str):
    node = kernel_node(source)
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    lines = source.splitlines(keepends=True)
    synthetic = "\n" * (start - 1) + "".join(lines[start - 1 : node.end_lineno])
    canonical_path = f"{SOURCE_PATH}@{revision}"
    linecache.cache[canonical_path] = (
        len(synthetic),
        None,
        synthetic.splitlines(keepends=True),
        canonical_path,
    )
    namespace = {
        "__name__": f"fr13_sfwd_codegen_{revision}",
        "__file__": canonical_path,
        "triton": triton,
        "tl": tl,
    }
    exec(compile(synthetic, canonical_path, "exec"), namespace)
    return namespace[KERNEL_NAME], "".join(lines[start - 1 : node.end_lineno])


def assignment_name(node: ast.stmt) -> str | None:
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    return None


def is_tl_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == name
    )


def source_profile(source: str) -> dict[str, object]:
    kernel = kernel_node(source)
    x_load_rows: list[int] = []
    activations: list[int] = []
    saved_accumulators: list[int] = []
    product_assignments = 0
    stores = 0
    for statement in kernel.body:
        name = assignment_name(statement)
        if name and name.startswith("x_") and isinstance(statement, ast.Assign):
            if is_tl_call(statement.value, "load"):
                x_load_rows.append(int(name.removeprefix("x_")))
        if name and name.startswith("activated_"):
            activations.append(int(name.removeprefix("activated_")))
        if name and name.startswith("acc_"):
            saved_accumulators.append(int(name.removeprefix("acc_")))
        if name and name.startswith("product_"):
            product_assignments += 1
        if isinstance(statement, ast.Expr) and is_tl_call(statement.value, "store"):
            stores += 1
    return {
        "x_global_load_assignments": len(x_load_rows),
        "x_unique_rows_loaded": len(set(x_load_rows)),
        "x_reload_count": len(x_load_rows) - len(set(x_load_rows)),
        "activation_assignments": len(activations),
        "saved_accumulator_assignments": len(saved_accumulators),
        "activation_window": 2 if saved_accumulators else 1,
        "peak_live_accumulator_values": 2 if saved_accumulators else 1,
        "product_assignments": product_assignments,
        "store_calls": stores,
        "x_load_order": x_load_rows,
        "activation_order": activations,
        "saved_accumulator_nodes": saved_accumulators,
    }


def operations(sass: str) -> collections.Counter[str]:
    result: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            result[match.group(1).split(".", 1)[0]] += 1
    return result


def compile_one(
    *,
    source: str,
    revision: str,
    output: Path,
    batch: int,
) -> dict[str, object]:
    config = DEPLOYMENT_CONFIGS[batch]
    output.mkdir(parents=True, exist_ok=True)
    fn, function_source = jit_function(source, revision)
    constants = dict(BASE_CONSTANTS)
    constants.update(B=batch, BLOCK_C=config["block_c"])
    target = GPUTarget("cuda", 121, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options(
        {"num_warps": config["num_warps"], "num_stages": 3}
    )
    compiled = triton.compile(
        ASTSource(fn=fn, signature=SIGNATURE, constexprs=constants),
        target=target,
        options=options.__dict__,
    )
    for name, value in compiled.asm.items():
        path = output / f"kernel.{name}"
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value)
    cubin_path = output / "kernel.cubin"
    sass = run_bytes(["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin_path)])
    resource = run_bytes(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(cubin_path)]
    )
    elf = run_bytes(["/usr/local/cuda/bin/cuobjdump", "--dump-elf", str(cubin_path)])
    (output / "kernel.sass").write_bytes(sass)
    (output / "resource.txt").write_bytes(resource)
    (output / "elf.txt").write_bytes(elf)
    resource_match = RESOURCE_RE.search(resource.decode())
    if resource_match is None:
        raise RuntimeError("unable to parse cubin resource usage")
    registers, stack_bytes, shared_bytes, local_bytes = map(
        int, resource_match.groups()
    )
    toolkit = TOOLKIT_RE.search(elf.decode())
    tool_name = TOOL_NAME_RE.search(elf.decode())
    tool_version = TOOL_VERSION_RE.search(elf.decode())
    if None in (toolkit, tool_name, tool_version):
        raise RuntimeError("unable to parse cubin producer metadata")
    counts = operations(sass.decode())
    metadata = (
        compiled.metadata._asdict()
        if hasattr(compiled.metadata, "_asdict")
        else vars(compiled.metadata)
    )
    cubin = compiled.asm["cubin"]
    result = {
        "revision": revision,
        "batch": batch,
        "physical_rows_per_request": 32,
        "channels": 10240,
        "block_c": config["block_c"],
        "num_warps": config["num_warps"],
        "num_stages": 3,
        "ctas_per_request": (10240 + config["block_c"] - 1)
        // config["block_c"],
        "ctas_per_launch": batch
        * ((10240 + config["block_c"] - 1) // config["block_c"]),
        "compile_hash": metadata["hash"],
        "source_function_sha256": sha256(function_source.encode()),
        "cubin_sha256": sha256(cubin),
        "cubin_bytes": len(cubin),
        "ptx_sha256": sha256(compiled.asm["ptx"].encode()),
        "sass_sha256": sha256(sass),
        "registers": registers,
        "stack_bytes": stack_bytes,
        "local_bytes": local_bytes,
        "elf_shared_bytes": shared_bytes,
        "launch_shared_bytes": int(metadata["shared"]),
        "encoded_sass_instructions": sum(counts.values()),
        "static_sass_instructions": sum(
            value
            for operation, value in counts.items()
            if operation not in CONTROL_OR_PADDING
        ),
        "ldg": counts["LDG"],
        "stg": counts["STG"],
        "lds": counts["LDS"],
        "sts": counts["STS"],
        "ldl": counts["LDL"],
        "stl": counts["STL"],
        "calls": sum(
            value for operation, value in counts.items() if operation.startswith("CALL")
        ),
        "backend_producer": {
            "toolkit_version": toolkit.group(1),
            "tool_name": tool_name.group(1),
            "tool_version": tool_version.group(1).strip(),
            "target": "sm_121a",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--incumbent-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revisions = {
        "incumbent": args.incumbent_revision,
        "candidate": args.candidate_revision,
    }
    summary: dict[str, object] = {
        "schema": "fr13.fixed32.sfwd.v4.sm121a.offline_codegen.v1",
        "source_path": SOURCE_PATH,
        "compile_contract": {
            "target": "sm_121a",
            "physical_rows_per_request": 32,
            "channels": 10240,
            "conv_width": 4,
            "conv_state_len": 34,
            "x_stride_row": 16384,
            "conv_stride_row": BASE_CONSTANTS["CONV_STRIDE_ROW"],
            "has_bias": False,
            "deployment_configs": DEPLOYMENT_CONFIGS,
        },
        "revisions": revisions,
        "variants": {},
    }
    for label, revision in revisions.items():
        source = source_at_revision(args.repo, revision)
        builds = {}
        for batch in DEPLOYMENT_CONFIGS:
            builds[f"b{batch}"] = compile_one(
                source=source,
                revision=revision,
                output=args.output / label / f"b{batch}",
                batch=batch,
            )
        summary["variants"][label] = {
            "source_sha256": sha256(source.encode()),
            "source_profile": source_profile(source),
            "builds": builds,
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
