#!/usr/bin/env python3
"""Compile fixed32 direct-col0 metadata variants for SM121a without a GPU."""

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
from triton.runtime.jit import MockTensor, create_function_from_signature


SOURCE_PATH = "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
KERNELS = {
    "incumbent_metadata_copy": (
        "_fr13_fixed32_conv_direct_col0_metadata_kernel"
    ),
    "candidate_direct_input": "_fr13_fixed32_conv_direct_col0_kernel",
}
INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+"
    r"(?:@[!]?(?:U)?P\d+\s+)?([A-Z][A-Z0-9.]*)\b"
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


def jit_function(source: str, revision: str, kernel_name: str):
    tree = ast.parse(source, filename=SOURCE_PATH)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == kernel_name
    )
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    lines = source.splitlines(keepends=True)
    function_source = "".join(lines[start - 1 : node.end_lineno])
    synthetic = "\n" * (start - 1) + function_source
    canonical_path = f"{SOURCE_PATH}@{revision}:{kernel_name}"
    linecache.cache[canonical_path] = (
        len(synthetic),
        None,
        synthetic.splitlines(keepends=True),
        canonical_path,
    )
    namespace = {
        "__name__": f"fr13_committer_direct_codegen_{revision}",
        "__file__": canonical_path,
        "triton": triton,
        "tl": tl,
    }
    exec(compile(synthetic, canonical_path, "exec"), namespace)
    return namespace[kernel_name], function_source


def operations(sass: str) -> collections.Counter[str]:
    result: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            result[match.group(1).split(".", 1)[0]] += 1
    return result


def exact_arguments(label: str, batch: int) -> list[object]:
    tensors: list[object] = [
        MockTensor(torch.bfloat16, [257, 10_240, 34]),
        MockTensor(torch.int64, [48]),
        MockTensor(torch.bfloat16, [batch * 36, 10_240]),
        MockTensor(torch.int64, [48]),
        MockTensor(torch.int64, [32 * 34]),
        MockTensor(torch.int32, [48, batch, 32]),
        MockTensor(torch.int32, [batch, 16]),
        MockTensor(torch.int32, [batch]),
    ]
    if label == "incumbent_metadata_copy":
        tensors.extend(
            [
                MockTensor(torch.int32, [batch, 16]),
                MockTensor(torch.int32, [batch]),
            ]
        )
    strides = [batch * 32, 32, 1, 16, 1, 1]
    if label == "incumbent_metadata_copy":
        strides.extend([16, 1, 1])
    strides.extend([10_240 * 34, 34, 1, 10_240, 1])
    return tensors + strides


def exact_specialization(fn, backend, label: str, batch: int):
    kwargs = {
        "CONV_C": 10_240,
        "CONV_L": 34,
        "SOURCE_ROWS": 36,
        "ELEM_BYTES": 2,
        "SPEC_COLS": 32,
        "PATH_COLS": 16,
        "B": batch,
        "BLOCK_C": 1024,
        "ZERO_TAIL": True,
        "LIVE_STATE_COLS": 3,
        "num_warps": 4,
    }
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    bound, specialization, options = binder(
        *exact_arguments(label, batch), **kwargs
    )
    compile_options, signature, constexprs, attrs = fn._pack_args(
        backend,
        options,
        bound,
        specialization,
        options,
    )
    return compile_options, signature, constexprs, attrs


def compile_one(
    *,
    source: str,
    revision: str,
    output: Path,
    label: str,
    batch: int,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    kernel_name = KERNELS[label]
    fn, function_source = jit_function(source, revision, kernel_name)
    target = GPUTarget("cuda", 121, 32)
    backend = triton.compiler.make_backend(target)
    options, signature, constexprs, attrs = exact_specialization(
        fn, backend, label, batch
    )
    compiled = triton.compile(
        ASTSource(
            fn=fn,
            signature=signature,
            constexprs=constexprs,
            attrs=attrs,
        ),
        target=target,
        options=options.__dict__,
    )
    cubin_path = output / "kernel.cubin"
    cubin_path.write_bytes(compiled.asm["cubin"])
    (output / "kernel.ptx").write_text(compiled.asm["ptx"])
    sass = run_bytes(["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin_path)])
    resource = run_bytes(
        [
            "/usr/local/cuda/bin/cuobjdump",
            "--dump-resource-usage",
            str(cubin_path),
        ]
    )
    elf = run_bytes(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-elf", str(cubin_path)]
    )
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
    result = {
        "revision": revision,
        "kernel": kernel_name,
        "batch": batch,
        "compile_hash": metadata["hash"],
        "source_function_sha256": sha256(function_source.encode()),
        "cubin_sha256": sha256(compiled.asm["cubin"]),
        "cubin_bytes": len(compiled.asm["cubin"]),
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
        "bar": counts["BAR"],
        "calls": sum(
            value
            for operation, value in counts.items()
            if operation.startswith("CALL")
        ),
        "logical_work": {
            "kernel_launches_per_event": 1,
            "conv_rows_per_event": 48 * batch,
            "metadata_elements_loaded_per_event": (
                16 * batch if label == "incumbent_metadata_copy" else 0
            ),
            "metadata_elements_stored_per_event": (
                17 * batch if label == "incumbent_metadata_copy" else 0
            ),
            "metadata_intermediate_roundtrip_elements_per_event": (
                17 * batch if label == "incumbent_metadata_copy" else 0
            ),
        },
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
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = source_at_revision(args.repo, args.revision)
    summary: dict[str, object] = {
        "schema": "fr13.fixed32.committer_direct_metadata.sm121a.codegen.v1",
        "revision": args.revision,
        "source_path": SOURCE_PATH,
        "source_sha256": sha256(source.encode()),
        "compile_contract": {
            "target": "sm_121a",
            "batches": [1, 4],
            "layers": 48,
            "physical_rows": 32,
            "path_capacity": 16,
            "channels": 10_240,
            "state_columns": 34,
            "live_state_columns": 3,
            "source_rows_per_request": 36,
            "block_channels": 1024,
            "num_warps": 4,
            "zero_tail": True,
            "jit_specialization": "mock_tensor_exact_shape_stride_and_alignment",
        },
        "variants": {},
    }
    for label in KERNELS:
        builds = {}
        for batch in (1, 4):
            builds[f"b{batch}"] = compile_one(
                source=source,
                revision=args.revision,
                output=args.output / label / f"b{batch}",
                label=label,
                batch=batch,
            )
        summary["variants"][label] = {"builds": builds}
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
