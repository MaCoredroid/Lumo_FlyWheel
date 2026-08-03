#!/usr/bin/env python3
"""Compile the gate-to-decay-reuse fixed32 ladder offline."""

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
PRODUCER = "_tree_gdn_kernel_fixed32_single_launch"
COMMITTER = "_fr13_fixed32_committer_native_layer_batch_kernel"
DEPENDENCIES = {
    PRODUCER: (
        "_gdn_node_step",
        "_gdn_node_step_precomputed_decay",
        "_tree_gdn_fixed32_single_launch_node",
        PRODUCER,
    ),
    COMMITTER: (COMMITTER,),
}
VARIANTS = {
    "producer_parent_gate": ("parent", PRODUCER, False),
    "producer_current_gate": ("current", PRODUCER, False),
    "producer_candidate_decay": ("current", PRODUCER, True),
    "committer_parent_gate": ("parent", COMMITTER, False),
    "committer_current_gate": ("current", COMMITTER, False),
    "committer_candidate_decay": ("current", COMMITTER, True),
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


def function_node(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source, filename=SOURCE_PATH)
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def jit_function(source: str, revision: str, kernel_name: str):
    source_lines = source.splitlines(keepends=True)
    synthetic_lines = ["\n"] * len(source_lines)
    included = []
    for name in DEPENDENCIES[kernel_name]:
        try:
            node = function_node(source, name)
        except StopIteration:
            if name == "_gdn_node_step_precomputed_decay":
                continue
            raise
        start = min([node.lineno] + [item.lineno for item in node.decorator_list])
        synthetic_lines[start - 1 : node.end_lineno] = source_lines[
            start - 1 : node.end_lineno
        ]
        included.append("".join(source_lines[start - 1 : node.end_lineno]))
    synthetic = "".join(synthetic_lines)
    canonical_path = f"{SOURCE_PATH}@{revision}:{kernel_name}"
    linecache.cache[canonical_path] = (
        len(synthetic),
        None,
        synthetic.splitlines(keepends=True),
        canonical_path,
    )
    namespace = {
        "__name__": f"fr13_gate_codegen_{revision}",
        "__file__": canonical_path,
        "triton": triton,
        "tl": tl,
    }
    exec(compile(synthetic, canonical_path, "exec"), namespace)
    return namespace[kernel_name], "\n".join(included)


def operation_counts(sass: str) -> tuple[collections.Counter, collections.Counter]:
    base: collections.Counter[str] = collections.Counter()
    full: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            operation = match.group(1)
            full[operation] += 1
            base[operation.split(".", 1)[0]] += 1
    return base, full


def producer_specialization(fn, backend, batch: int, candidate: bool, current: bool):
    rows = batch * 32
    args = [
        MockTensor(torch.bfloat16, [rows, 16, 128]),
        MockTensor(torch.bfloat16, [rows, 16, 128]),
        MockTensor(torch.bfloat16, [rows, 48, 128]),
        MockTensor(torch.bfloat16, [rows, 48]),
        MockTensor(torch.bfloat16, [rows, 48]),
        MockTensor(torch.bfloat16, [rows, 48]),
        MockTensor(torch.bfloat16, [rows, 48]),
        MockTensor(torch.float32, [48]),
        MockTensor(torch.float32, [48]),
        MockTensor(torch.float32, [257, 48, 128, 128]),
        MockTensor(torch.int32, [batch, 32]),
        MockTensor(torch.int32, [batch]),
        MockTensor(torch.int64, [1]),
        MockTensor(torch.int32, [5]),
        MockTensor(torch.int32, [77]),
        MockTensor(torch.int32, [11]),
        MockTensor(torch.int32, [15]),
        MockTensor(torch.int32, [5]),
        MockTensor(torch.bfloat16, [rows, 48, 128]),
        MockTensor(torch.bfloat16, [rows, 16, 128]),
        MockTensor(torch.bfloat16, [rows, 48, 128]),
        MockTensor(torch.bfloat16, [rows, 48]),
        MockTensor(torch.bfloat16, [rows, 48]),
    ]
    args.append(MockTensor(torch.int32, [2]))
    args.append(MockTensor(torch.float32, [rows, 16]))
    args.append(MockTensor(torch.float32, [rows, 48, 2]))
    kwargs = {
        "N_ACTUAL": 32,
        "NUM_KH": 16,
        "NUM_VH": 48,
        "DIM_K": 128,
        "DIM_V": 128,
        "BLOCK_V": 8,
        "OUTPUT_SCALE": 128**-0.5,
        "USE_QK_L2NORM_IN_KERNEL": True,
        "H0_IS_BANK": True,
        "H0_INDEX_ROW": 0,
        "H0_INDEX_BATCH_STRIDE": 32,
        "H0_BATCH_INDEX": 0,
        "H0_ACCEPTED_BATCH_STRIDE": 1,
        "H0_BANK_STRIDE": 48 * 128 * 128,
        "H0_USE_ACCEPTED_COLUMN": False,
        "RAW_GATING": True,
        "COUNT_INVOCATION": False,
        "SCAN_ALIGN": False,
        "ROOT_STEPS": 5,
        "MAX_PATH_LEN": 7,
        "MAX_GROUP_PATHS": 3,
        "NUM_GROUPS": 5,
        "PRESCALED_PATH_BASE": True,
        "RING_EXPORT": True,
        "K_NORM_EXPORT": True,
        "FLAGS_EXPORT": True,
        "FLAGS_ROWS": batch,
        "num_warps": 8,
    }
    kwargs["GATE_EXPORT"] = True
    kwargs["maxnreg"] = 80
    if current:
        kwargs["DECAY_EXPORT"] = candidate
    return args, kwargs


def committer_specialization(fn, backend, batch: int, candidate: bool, current: bool):
    args = [
        MockTensor(torch.bfloat16, [48, batch, 32, 48]),
        MockTensor(torch.bfloat16, [48, batch, 32, 48]),
        MockTensor(torch.float32, [48, 48, 2]),
        MockTensor(torch.bfloat16, [48, batch, 32, 16, 128]),
    ]
    args.extend(
        [
            MockTensor(torch.bfloat16, [48, batch, 32, 48, 128]),
            MockTensor(torch.float32, [257, 48, 128, 128]),
            MockTensor(torch.int64, [48]),
            MockTensor(torch.int32, [batch, 16]),
            MockTensor(torch.int32, [batch]),
            MockTensor(torch.int32, [48, batch, 32]),
        ]
    )
    args.append(MockTensor(torch.float32, [48, batch, 32, 16]))
    args.append(MockTensor(torch.float32, [48, batch, 32, 48, 2]))
    kwargs = {
        "B": batch,
        "PATH_CAP": 16,
        "H": 16,
        "HV": 48,
        "K": 128,
        "V": 128,
        "BK": 128,
        "BV": 128,
        "BANK_STRIDE": 48 * 128 * 128,
        "GATE_L_STRIDE": 48 * 2,
        "RING_K_L_STRIDE": batch * 32 * 16 * 128,
        "RING_K_B_STRIDE": 32 * 16 * 128,
        "RING_K_N_STRIDE": 16 * 128,
        "RING_V_L_STRIDE": batch * 32 * 48 * 128,
        "RING_V_B_STRIDE": 32 * 48 * 128,
        "RING_V_N_STRIDE": 48 * 128,
        "RING_AB_L_STRIDE": batch * 32 * 48,
        "RING_AB_B_STRIDE": 32 * 48,
        "RING_AB_N_STRIDE": 48,
        "SPEC_L_STRIDE": batch * 32,
        "SPEC_B_STRIDE": 32,
        "BETA": 1.0,
        "THRESHOLD": 20.0,
        "USE_QK_L2NORM_IN_KERNEL": True,
        "RING_KN_L_STRIDE": batch * 32 * 16,
        "RING_KN_B_STRIDE": 32 * 16,
        "RING_KN_N_STRIDE": 16,
        "K_NORM_REUSE": True,
        "num_warps": 8,
        "num_stages": 3,
    }
    kwargs.update(
        {
            "RING_GATE_L_STRIDE": batch * 32 * 48 * 2,
            "RING_GATE_B_STRIDE": 32 * 48 * 2,
            "RING_GATE_N_STRIDE": 48 * 2,
            "GATE_REUSE": True,
        }
    )
    if current:
        kwargs["DECAY_REUSE"] = candidate
        if candidate and batch != 1:
            kwargs["maxnreg"] = 167
    return args, kwargs


def exact_specialization(fn, backend, kernel: str, batch: int, candidate: bool, current: bool):
    if kernel == PRODUCER:
        args, kwargs = producer_specialization(fn, backend, batch, candidate, current)
    else:
        args, kwargs = committer_specialization(fn, backend, batch, candidate, current)
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


def logical_work(kernel: str, candidate: bool, batch: int) -> dict[str, object]:
    if kernel == PRODUCER:
        raw_ab_values = 48 * batch * 32 * 48 * 2
        return {
            "kernel_launches_per_layer_event": 1,
            "layers": 48,
            "programs_per_layer_event": batch * 48 * 16,
            "physical_nodes_per_request": 32,
            "decay_nonlinear_evaluations_added": 0,
            "gate_scalar_values_stored": 48 * batch * 32 * 48 * 2,
            "gate_scalar_bytes_stored": 48 * batch * 32 * 48 * 2 * 4,
            "raw_ab_values_stored": raw_ab_values,
            "raw_ab_bytes_stored": raw_ab_values * 2,
            "raw_ab_bytes_eliminated": 0,
        }
    steps = {}
    programs = 48 * batch * 48
    for accepted in (0, 4, 11):
        live_steps = accepted + 1
        decay_exponentials = 0 if candidate else programs * live_steps
        loads = programs * live_steps * 2
        steps[f"accepted_{accepted}"] = {
            "live_steps": live_steps,
            "decay_exponentials": decay_exponentials,
            "gate_scalar_loads": loads,
            "decay_exponentials_removed": (
                programs * live_steps if candidate else 0
            ),
        }
    return {
        "kernel_launches_per_event": 1,
        "programs_per_event": programs,
        "dynamic_step_census": steps,
    }


def compile_one(
    *,
    source: str,
    revision: str,
    output: Path,
    label: str,
    kernel: str,
    candidate: bool,
    current: bool,
    batch: int,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    fn, included_source = jit_function(source, revision, kernel)
    target = GPUTarget("cuda", 121, 32)
    backend = triton.compiler.make_backend(target)
    options, signature, constexprs, attrs = exact_specialization(
        fn, backend, kernel, batch, candidate, current
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
        path.write_bytes(value) if isinstance(value, bytes) else path.write_text(value)
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
    registers, stack_bytes, shared_bytes, local_bytes = map(int, resource_match.groups())
    toolkit = TOOLKIT_RE.search(elf.decode())
    tool_name = TOOL_NAME_RE.search(elf.decode())
    tool_version = TOOL_VERSION_RE.search(elf.decode())
    if None in (toolkit, tool_name, tool_version):
        raise RuntimeError("unable to parse cubin producer metadata")
    base, full = operation_counts(sass.decode())
    metadata = (
        compiled.metadata._asdict()
        if hasattr(compiled.metadata, "_asdict")
        else vars(compiled.metadata)
    )
    result = {
        "label": label,
        "revision": revision,
        "kernel": kernel,
        "batch": batch,
        "candidate": candidate,
        "compile_hash": metadata["hash"],
        "included_source_sha256": sha256(included_source.encode()),
        "cubin_sha256": sha256(compiled.asm["cubin"]),
        "cubin_bytes": len(compiled.asm["cubin"]),
        "ptx_sha256": sha256(compiled.asm["ptx"].encode()),
        "sass_sha256": sha256(sass),
        "registers": registers,
        "stack_bytes": stack_bytes,
        "local_bytes": local_bytes,
        "elf_shared_bytes": shared_bytes,
        "launch_shared_bytes": int(metadata["shared"]),
        "encoded_sass_instructions": sum(base.values()),
        "static_sass_instructions": sum(
            count for operation, count in base.items() if operation not in CONTROL_OR_PADDING
        ),
        "base_operations": dict(sorted(base.items())),
        "full_operations": dict(sorted(full.items())),
        "logical_work": logical_work(kernel, candidate, batch),
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
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "current": source_at_revision(args.repo, args.revision),
        "parent": source_at_revision(args.repo, args.parent),
    }
    summary: dict[str, object] = {
        "schema": "fr13.fixed32.committer_decay_ring.sm121a.codegen.v1",
        "source_path": SOURCE_PATH,
        "revision": args.revision,
        "parent_revision": args.parent,
        "source_sha256": sha256(sources["current"].encode()),
        "parent_source_sha256": sha256(sources["parent"].encode()),
        "compile_contract": {
            "target": "sm_121a",
            "layers": 48,
            "physical_rows": 32,
            "key_heads": 16,
            "value_heads": 48,
            "dim_k": 128,
            "dim_v": 128,
            "producer_block_v": 8,
            "committer_block_v": 128,
            "batches": [1, 4],
            "num_warps": 8,
            "producer_num_stages": "triton_default",
            "producer_candidate_maxnreg": 80,
            "committer_b2_b4_candidate_maxnreg": 167,
            "committer_num_stages": 3,
            "prescaled_path_base": True,
            "jit_specialization": "mock_tensor_exact_shape_stride_and_alignment",
        },
        "variants": {},
    }
    for label, (source_key, kernel, candidate) in VARIANTS.items():
        builds = {}
        for batch in (1, 4):
            builds[f"b{batch}"] = compile_one(
                source=sources[source_key],
                revision=args.parent if source_key == "parent" else args.revision,
                output=args.output / label / f"b{batch}",
                label=label,
                kernel=kernel,
                candidate=candidate,
                current=source_key == "current",
                batch=batch,
            )
        summary["variants"][label] = {"builds": builds}
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
