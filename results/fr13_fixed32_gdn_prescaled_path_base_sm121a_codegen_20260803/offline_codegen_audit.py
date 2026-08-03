#!/usr/bin/env python3
"""Compile the fixed32 ordered GDN descriptor variants for SM121a."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


SOURCE_PATH = Path("src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py")
EXPECTED_REVISION = "8959f328ce6b5e36c5eb6bbb1cb53c3c6e5f5bbe"
KERNEL_NAME = "_tree_gdn_kernel_fixed32_single_launch"
MODULE_NAME = "lumo_flywheel_serving.fr10_gdn_tree_kernel"
VARIANTS = {
    "incumbent_index_scaled": False,
    "candidate_prescaled_path_base": True,
}
POINTER_SIGNATURE = {
    "q": "*bf16",
    "k": "*bf16",
    "v": "*bf16",
    "g": "*fp32",
    "beta": "*fp32",
    "raw_a": "*bf16",
    "raw_b": "*bf16",
    "A_log": "*fp32",
    "dt_bias": "*fp32",
    "h0": "*bf16",
    "h0_indices": "*i64",
    "h0_num_accepted_tokens": "*i32",
    "invocation_counter": "*i32",
    "root_nodes": "*i32",
    "branch_nodes": "*i32",
    "branch_lengths": "*i32",
    "group_path_indices": "*i32",
    "group_path_counts": "*i32",
    "out": "*bf16",
    "ring_k": "*bf16",
    "ring_v": "*bf16",
    "ring_a": "*bf16",
    "ring_b": "*bf16",
    "flags_ptr": "*i32",
}
BASE_CONSTANTS = {
    "N_ACTUAL": 32,
    "NUM_KH": 4,
    "NUM_VH": 12,
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
    "H0_BANK_STRIDE": 12 * 128 * 128,
    "H0_USE_ACCEPTED_COLUMN": False,
    "RAW_GATING": True,
    "COUNT_INVOCATION": False,
    "SCAN_ALIGN": True,
    "ROOT_STEPS": 5,
    "MAX_PATH_LEN": 7,
    "MAX_GROUP_PATHS": 3,
    "NUM_GROUPS": 5,
    "RING_EXPORT": True,
    "FLAGS_EXPORT": True,
}
RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+"
    r"(?:@[!]?(?:U)?P\d+\s+)?([A-Z][A-Z0-9.]*)\b"
)
SASS_SLOT_RE = re.compile(r"^\s*/\*")
TOOLKIT_RE = re.compile(r"Tool Kit Version:\s*([^\s]+)")
TOOL_NAME_RE = re.compile(r"Tool Name:\s*(\S+)")
TOOL_VERSION_RE = re.compile(r"Tool Version:\s*(.+)")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_bytes(argv: list[str]) -> bytes:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def function_sha256(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    lines = source.splitlines(keepends=True)
    return sha256("".join(lines[start - 1 : node.end_lineno]).encode())


def operations(sass: str) -> collections.Counter[str]:
    result: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            result[match.group(1).split(".", 1)[0]] += 1
    return result


def compile_one(
    *,
    fn,
    revision: str,
    source: str,
    output: Path,
    label: str,
    batch: int,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    constants = {
        **BASE_CONSTANTS,
        "FLAGS_ROWS": batch,
        "PRESCALED_PATH_BASE": VARIANTS[label],
    }
    compiled = triton.compile(
        ASTSource(fn, signature=POINTER_SIGNATURE, constexprs=constants),
        target=GPUTarget("cuda", 121, 32),
        options={"num_warps": 8},
    )
    cubin = compiled.asm["cubin"]
    ptx = compiled.asm["ptx"].encode()
    cubin_path = output / "kernel.cubin"
    cubin_path.write_bytes(cubin)
    (output / "kernel.ptx").write_bytes(ptx)
    sass = run_bytes(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(cubin_path)]
    )
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
    sass_text = sass.decode()
    counts = operations(sass_text)
    metadata = (
        compiled.metadata._asdict()
        if hasattr(compiled.metadata, "_asdict")
        else vars(compiled.metadata)
    )
    programs = 12 * (128 // 8) * batch
    result = {
        "revision": revision,
        "kernel": KERNEL_NAME,
        "batch": batch,
        "compile_hash": metadata["hash"],
        "kernel_source_sha256": function_sha256(source, KERNEL_NAME),
        "node_source_sha256": function_sha256(
            source, "_tree_gdn_fixed32_single_launch_node"
        ),
        "recurrence_source_sha256": function_sha256(
            source, "_gdn_node_step"
        ),
        "cubin_sha256": sha256(cubin),
        "cubin_bytes": len(cubin),
        "ptx_sha256": sha256(ptx),
        "ptx_bytes": len(ptx),
        "sass_sha256": sha256(sass),
        "sass_addressed_lines": sum(
            bool(SASS_SLOT_RE.match(line)) for line in sass_text.splitlines()
        ),
        "static_sass_instructions": sum(counts.values()),
        "registers": registers,
        "stack_bytes": stack_bytes,
        "local_bytes": local_bytes,
        "elf_shared_bytes": shared_bytes,
        "launch_shared_bytes": int(metadata["shared"]),
        "ldg": counts["LDG"],
        "stg": counts["STG"],
        "ldl": counts["LDL"],
        "stl": counts["STL"],
        "calls": sum(
            value
            for operation, value in counts.items()
            if operation.startswith("CALL")
        ),
        "opcodes": dict(sorted(counts.items())),
        "logical_work": {
            "programs_per_event": programs,
            "reachable_branch_paths_per_program": 11,
            "path_base_scale_operations_per_program": (
                0 if VARIANTS[label] else 11
            ),
            "path_base_scale_operations_per_event": (
                0 if VARIANTS[label] else 11 * programs
            ),
            "tree_node_updates_per_program": 32,
        },
        "backend_producer": {
            "toolkit_version": toolkit.group(1),
            "tool_name": tool_name.group(1),
            "tool_version": tool_version.group(1).strip(),
            "target": "sm_121a",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.revision != EXPECTED_REVISION:
        raise RuntimeError(f"expected source revision {EXPECTED_REVISION}")
    source_path = repo / SOURCE_PATH
    source = source_path.read_text(encoding="utf-8")
    sys.path.insert(0, str(repo / "src"))
    module = importlib.import_module(MODULE_NAME)
    if Path(module.__file__).resolve() != source_path.resolve():
        raise RuntimeError("imported GDN source path drift")
    fn = getattr(module, KERNEL_NAME)
    summary: dict[str, object] = {
        "schema": "fr13.fixed32.gdn_prescaled_path_base.sm121a.codegen.v1",
        "revision": args.revision,
        "source_path": str(SOURCE_PATH),
        "source_sha256": sha256(source.encode()),
        "compile_contract": {
            "target": "sm_121a",
            "batches": [1, 4],
            "physical_nodes": 32,
            "draft_vocab_k": 65_536,
            "draft_vocab_root": 1,
            "num_key_heads": 4,
            "num_value_heads": 12,
            "dim_k": 128,
            "dim_v": 128,
            "block_v": 8,
            "num_warps": 8,
            "root_steps": 5,
            "max_path_len": 7,
            "max_group_paths": 3,
            "groups": 5,
            "ring_export": True,
            "flags_export": True,
            "ordered_dynamic_loops": True,
            "signature": "explicit_deployed_pointer_types",
        },
        "variants": {},
    }
    for label in VARIANTS:
        builds = {}
        for batch in (1, 4):
            builds[f"b{batch}"] = compile_one(
                fn=fn,
                revision=args.revision,
                source=source,
                output=args.output / label / f"b{batch}",
                label=label,
                batch=batch,
            )
        summary["variants"][label] = {"builds": builds}
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
