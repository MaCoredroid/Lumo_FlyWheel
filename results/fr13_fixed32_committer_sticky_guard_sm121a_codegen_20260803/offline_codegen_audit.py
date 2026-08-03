#!/usr/bin/env python3
"""Compile vector-result and sticky-scalar committer guards without a GPU."""

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
KERNEL_NAMES = {
    "incumbent_vector_result": "_fr13_fixed32_conv_commit_row_guard_kernel",
    "candidate_sticky_scalar": (
        "_fr13_fixed32_conv_commit_sticky_guard_kernel"
    ),
}
BANK_ROWS_FIXTURE = 257
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


def kernel_node(source: str, kernel_name: str) -> ast.FunctionDef:
    tree = ast.parse(source, filename=SOURCE_PATH)
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name
    )


def jit_function(source: str, revision: str, kernel_name: str):
    node = kernel_node(source, kernel_name)
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    lines = source.splitlines(keepends=True)
    function_source = "".join(lines[start - 1 : node.end_lineno])
    synthetic = "\n" * (start - 1) + function_source
    canonical_path = f"{SOURCE_PATH}@{revision}"
    linecache.cache[canonical_path] = (
        len(synthetic),
        None,
        synthetic.splitlines(keepends=True),
        canonical_path,
    )
    namespace = {
        "__name__": f"fr13_committer_codegen_{revision}",
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


def logical_work(label: str, batch: int) -> dict[str, int]:
    programs = 48 * batch
    incumbent = label == "incumbent_vector_result"
    return {
        "programs_per_event": programs,
        "guard_kernel_launches_per_event": 1,
        "scalar_reduction_launches_per_event": 1 if incumbent else 0,
        "async_assert_launches_per_event": 1,
        "source_visible_guard_pipeline_launches_per_event": (
            3 if incumbent else 2
        ),
        "physical_ssi_row_values": programs * 32,
        "path_values": batch * 16,
        "accepted_length_values": batch,
        "alias_id_values": 48,
        "result_values_stored_on_valid_event": programs if incumbent else 0,
        "result_bytes_stored_on_valid_event": programs if incumbent else 0,
        "result_values_reloaded_by_reduction": programs if incumbent else 0,
        "sticky_values_stored_on_valid_event": 0,
        "sticky_failure_atomics_on_valid_event": 0,
    }


def exact_specialization(fn, backend, label: str, batch: int):
    result_tensor = (
        MockTensor(torch.bool, [48 * batch])
        if label == "incumbent_vector_result"
        else MockTensor(torch.int32, [])
    )
    args = [
        MockTensor(torch.int32, [48, batch, 32]),
        MockTensor(torch.int32, [batch, 16]),
        MockTensor(torch.int32, [batch]),
        MockTensor(torch.int64, [48]),
        MockTensor(torch.int32, [48, 3]),
        result_tensor,
        batch * 32,
        32,
        1,
        16,
        1,
        1,
        3,
        1,
    ]
    kwargs = {
        "BANK_ROWS": BANK_ROWS_FIXTURE,
        "B": batch,
        "LAYERS": 48,
        "SPEC_COLS": 32,
        "PATH_COLS": 16,
        "MAX_ACCEPTED": 11,
        "ALIAS_WIDTH": 3,
        "PEER_CAP": 16,
        "num_warps": 4,
        "num_stages": 1,
    }
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    bound, specialization, options = binder(*args, **kwargs)
    compile_options, signature, constexprs, attrs = fn._pack_args(
        backend,
        options,
        bound,
        specialization,
        options,
    )
    return compile_options, signature, constexprs, attrs


def compile_one(
    *, source: str, revision: str, output: Path, label: str, batch: int
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    kernel_name = KERNEL_NAMES[label]
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
    for name, value in compiled.asm.items():
        path = output / f"kernel.{name}"
        path.write_bytes(value) if isinstance(value, bytes) else path.write_text(
            value
        )
    cubin_path = output / "kernel.cubin"
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
        "global_atomics": sum(
            value
            for operation, value in counts.items()
            if operation.startswith("ATOM")
        ),
        "redux": counts["REDUX"],
        "calls": sum(
            value
            for operation, value in counts.items()
            if operation.startswith("CALL")
        ),
        "logical_work": logical_work(label, batch),
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
    revision = args.revision
    summary: dict[str, object] = {
        "schema": "fr13.fixed32.committer_sticky_guard.sm121a.codegen.v1",
        "source_path": SOURCE_PATH,
        "revision": revision,
        "kernels": dict(KERNEL_NAMES),
        "compile_contract": {
            "target": "sm_121a",
            "bank_rows_fixture": BANK_ROWS_FIXTURE,
            "layers": 48,
            "physical_rows": 32,
            "path_capacity": 16,
            "alias_width": 3,
            "peer_capacity": 16,
            "batches": [1, 4],
            "num_warps": 4,
            "num_stages": 1,
            "jit_specialization": "mock_tensor_exact_shape_stride_and_alignment",
        },
        "variants": {},
    }
    source = source_at_revision(args.repo, revision)
    summary["source_sha256"] = sha256(source.encode())
    for label in KERNEL_NAMES:
        builds = {}
        for batch in (1, 4):
            builds[f"b{batch}"] = compile_one(
                source=source,
                revision=revision,
                output=args.output / label / f"b{batch}",
                label=label,
                batch=batch,
            )
        summary["variants"][label] = {
            "builds": builds,
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
