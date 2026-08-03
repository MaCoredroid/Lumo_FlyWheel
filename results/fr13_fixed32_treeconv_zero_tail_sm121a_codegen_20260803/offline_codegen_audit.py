#!/usr/bin/env python3
"""Compile fixed32 direct-conv incumbent/candidate kernels without a GPU."""

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
import textwrap


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


DIRECT_SOURCE = "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
GENERIC_SOURCE = "src/lumo_flywheel_serving/fr13_tree_conv_fused.py"
DIRECT_KERNELS = {
    "direct": "_fr13_fixed32_conv_direct_col0_kernel",
    "metadata": "_fr13_fixed32_conv_direct_col0_metadata_kernel",
}
GENERIC_KERNEL = "_fr13_conv_wb_fused_batched_kernel"
BASE_CONSTANTS = {
    "CONV_C": 10240,
    "CONV_L": 34,
    "SOURCE_ROWS": 36,
    "ELEM_BYTES": 2,
    "SPEC_COLS": 32,
    "PATH_COLS": 16,
    "BLOCK_C": 1024,
}
DEPLOYMENT_CONFIGS = {
    1: {"num_warps": 4},
    4: {"num_warps": 4},
}
DEPLOYMENT_CONTEXT = {
    "fixed_physical_rows": 32,
    "drafter_vocab_k": 65536,
    "root_reduction": 1,
}
POINTER_TYPES = {
    "anchor_ptr": "*bf16",
    "bank_off16": "*i64",
    "source_anchor": "*bf16",
    "source_off16": "*i64",
    "state_src": "*i64",
    "spec_state_indices": "*i32",
    "accepted_paths": "*i32",
    "accepted_lens": "*i32",
    "committer_paths": "*i32",
    "committer_lens": "*i32",
}
SCALAR_NAMES = {
    "ssi_stride_l",
    "ssi_stride_b",
    "ssi_stride_s",
    "path_stride_b",
    "path_stride_s",
    "lens_stride_b",
    "committer_path_stride_b",
    "committer_path_stride_s",
    "committer_lens_stride_b",
    "bank_row_stride",
    "bank_c_stride",
    "bank_l_stride",
    "source_row_stride",
    "source_c_stride",
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


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_bytes(argv: list[str]) -> bytes:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def source_at_revision(repo: Path, revision: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout


def jit_function(source: str, path: str, name: str, revision: str):
    tree = ast.parse(source, filename=path)
    matches = [
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} definition, found {len(matches)}")
    node = matches[0]
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    lines = source.splitlines(keepends=True)
    function_source = textwrap.dedent(
        "".join(lines[start - 1 : node.end_lineno])
    )
    canonical_path = f"{path}@{revision}:{name}"
    synthetic = "\n" * (start - 1) + function_source
    linecache.cache[canonical_path] = (
        len(synthetic),
        None,
        synthetic.splitlines(keepends=True),
        canonical_path,
    )
    namespace = {
        "__name__": f"fr13_treeconv_codegen_{revision}_{name}",
        "__file__": canonical_path,
        "triton": triton,
        "tl": tl,
    }
    exec(compile(synthetic, canonical_path, "exec"), namespace)
    return namespace[name], function_source


def operations(sass: str) -> collections.Counter[str]:
    result: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            result[match.group(1).split(".", 1)[0]] += 1
    return result


def compile_kernel(
    *,
    fn,
    function_source: str,
    constants: dict[str, object],
    output: Path,
    batch: int,
    loaded_columns: int,
) -> dict[str, object]:
    signature = {}
    for name in fn.arg_names:
        if name in constants:
            continue
        if name in POINTER_TYPES:
            signature[name] = POINTER_TYPES[name]
        elif name in SCALAR_NAMES:
            signature[name] = "i64"
        else:
            raise RuntimeError(f"untyped direct-kernel argument {name!r}")
    target = GPUTarget("cuda", 121, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options(
        {"num_warps": DEPLOYMENT_CONFIGS[batch]["num_warps"]}
    )
    compiled = triton.compile(
        ASTSource(fn=fn, signature=signature, constexprs=constants),
        target=target,
        options=options.__dict__,
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, value in compiled.asm.items():
        destination = output / f"kernel.{name}"
        if isinstance(value, bytes):
            destination.write_bytes(value)
        else:
            destination.write_text(value)
    cubin = compiled.asm["cubin"]
    cubin_path = output / "kernel.cubin"
    sass = run_bytes(["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin_path)])
    resource = run_bytes(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(cubin_path)]
    )
    elf = run_bytes(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-elf", str(cubin_path)]
    )
    (output / "kernel.sass").write_bytes(sass)
    (output / "resource.txt").write_bytes(resource)
    (output / "elf.txt").write_bytes(elf)
    resource_match = RESOURCE_RE.search(resource.decode())
    toolkit = TOOLKIT_RE.search(elf.decode())
    tool_name = TOOL_NAME_RE.search(elf.decode())
    tool_version = TOOL_VERSION_RE.search(elf.decode())
    if None in (resource_match, toolkit, tool_name, tool_version):
        raise RuntimeError("unable to parse SM121a codegen metadata")
    registers, stack_bytes, shared_bytes, local_bytes = map(
        int, resource_match.groups()
    )
    counts = operations(sass.decode())
    metadata = (
        compiled.metadata._asdict()
        if hasattr(compiled.metadata, "_asdict")
        else vars(compiled.metadata)
    )
    channels = int(constants["CONV_C"])
    state_length = int(constants["CONV_L"])
    block_c = int(constants["BLOCK_C"])
    layers = 48
    ctas = layers * batch * ((channels + block_c - 1) // block_c)
    source_read_bytes = layers * batch * loaded_columns * channels * 2
    destination_write_bytes = layers * batch * state_length * channels * 2
    result = {
        "batch": batch,
        "physical_rows_per_request": 32,
        "channels": channels,
        "state_length": state_length,
        "block_c": block_c,
        "num_warps": DEPLOYMENT_CONFIGS[batch]["num_warps"],
        "ctas_per_event": ctas,
        "source_columns_loaded_per_row": loaded_columns,
        "destination_columns_stored_per_row": state_length,
        "source_read_bytes_per_event": source_read_bytes,
        "destination_write_bytes_per_event": destination_write_bytes,
        "modeled_global_bytes_per_event": (
            source_read_bytes + destination_write_bytes
        ),
        "roofline_ms_at_273GBps": (
            source_read_bytes + destination_write_bytes
        ) / 273_000_000_000 * 1000,
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
        "ldg": counts["LDG"],
        "stg": counts["STG"],
        "ldl": counts["LDL"],
        "stl": counts["STL"],
        "calls": sum(
            value
            for operation, value in counts.items()
            if operation.startswith("CALL")
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


def compile_direct_variants(
    *, repo: Path, revisions: dict[str, str], output: Path
) -> dict[str, object]:
    sources = {
        label: source_at_revision(repo, revision, DIRECT_SOURCE)
        for label, revision in revisions.items()
    }
    result = {}
    for kind, kernel_name in DIRECT_KERNELS.items():
        result[kind] = {}
        for label in ("incumbent", "retained_off", "candidate"):
            revision_label = "incumbent" if label == "incumbent" else "candidate"
            revision = revisions[revision_label]
            source = sources[revision_label]
            fn, function_source = jit_function(
                source, DIRECT_SOURCE, kernel_name, revision
            )
            result[kind][label] = {}
            for batch in DEPLOYMENT_CONFIGS:
                constants = dict(BASE_CONSTANTS, B=batch)
                loaded_columns = 34
                if label != "incumbent":
                    constants.update(
                        ZERO_TAIL=label == "candidate",
                        LIVE_STATE_COLS=3,
                    )
                    loaded_columns = 3 if label == "candidate" else 34
                result[kind][label][f"b{batch}"] = compile_kernel(
                    fn=fn,
                    function_source=function_source,
                    constants=constants,
                    output=output / kind / label / f"b{batch}",
                    batch=batch,
                    loaded_columns=loaded_columns,
                )
    return result


def generic_baseline(repo: Path, revision: str, output: Path) -> dict[str, object]:
    source = source_at_revision(repo, revision, GENERIC_SOURCE)
    fn, function_source = jit_function(
        source, GENERIC_SOURCE, GENERIC_KERNEL, revision
    )
    constants = {"S": 36, "N": 32, "C": 10240, "L": 34, "BLOCK_C": 1024}
    signature = {
        "source_z": "*bf16",
        "state_src": "*i64",
        "dst_rows": "*i32",
        "conv_state": "*bf16",
        "stride_cs0": "i64",
        "stride_cs1": "i64",
        "stride_cs2": "i64",
    }
    target = GPUTarget("cuda", 121, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": 4})
    compiled = triton.compile(
        ASTSource(fn=fn, signature=signature, constexprs=constants),
        target=target,
        options=options.__dict__,
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, value in compiled.asm.items():
        path = output / f"kernel.{name}"
        path.write_bytes(value) if isinstance(value, bytes) else path.write_text(value)
    cubin_path = output / "kernel.cubin"
    sass = run_bytes(["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin_path)])
    resource = run_bytes(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(cubin_path)]
    )
    elf = run_bytes(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-elf", str(cubin_path)]
    )
    (output / "kernel.sass").write_bytes(sass)
    (output / "resource.txt").write_bytes(resource)
    (output / "elf.txt").write_bytes(elf)
    counts = operations(sass.decode())
    registers, stack_bytes, shared_bytes, local_bytes = map(
        int, RESOURCE_RE.search(resource.decode()).groups()
    )
    result = {
        "source_function_sha256": sha256(function_source.encode()),
        "cubin_sha256": sha256(compiled.asm["cubin"]),
        "ptx_sha256": sha256(compiled.asm["ptx"].encode()),
        "sass_sha256": sha256(sass),
        "cubin_bytes": len(compiled.asm["cubin"]),
        "registers": registers,
        "stack_bytes": stack_bytes,
        "local_bytes": local_bytes,
        "elf_shared_bytes": shared_bytes,
        "ldg": counts["LDG"],
        "stg": counts["STG"],
        "ldl": counts["LDL"],
        "stl": counts["STL"],
        "calls": sum(v for op, v in counts.items() if op.startswith("CALL")),
        "b1_ctas_per_event": 48 * 32 * 10,
        "b4_ctas_per_event": 48 * 4 * 32 * 10,
        "b1_modeled_global_bytes_per_event": 2 * 48 * 32 * 10240 * 34 * 2,
        "b4_modeled_global_bytes_per_event": 2 * 48 * 4 * 32 * 10240 * 34 * 2,
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
        "candidate": args.candidate_revision,
        "incumbent": args.incumbent_revision,
    }
    patcher = source_at_revision(
        args.repo,
        args.candidate_revision,
        "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
    )
    direct_source = source_at_revision(
        args.repo, args.candidate_revision, DIRECT_SOURCE
    )
    summary = {
        "schema": "fr13.fixed32.treeconv.zero_tail.sm121a.codegen.v1",
        "offline_only": True,
        "timing_claim": False,
        "revisions": revisions,
        "deployment_context": DEPLOYMENT_CONTEXT,
        "compile_contract": {
            "target": "sm_121a",
            "physical_rows_per_request": 32,
            "channels": 10240,
            "conv_width": 4,
            "conv_state_len": 34,
            "source_rows_per_request": 36,
            "layers": 48,
            "block_c": 1024,
            "deployment_configs": DEPLOYMENT_CONFIGS,
        },
        "fixed32_route": {
            "generic_batched_writeback_guarded_out": (
                "and not _FR13_FIXED32_MODE" in patcher
            ),
            "full_node_writebacks_per_event": 0,
            "direct_committed_rows": {"b1": 48, "b4": 192},
            "candidate_default_off": (
                'os.environ.get("FR13_FIXED32_CONV_COMMIT_ZERO_TAIL", "0")'
                in direct_source
            ),
        },
        "direct_kernels": compile_direct_variants(
            repo=args.repo,
            revisions=revisions,
            output=args.output,
        ),
        "retained_generic": generic_baseline(
            args.repo,
            args.candidate_revision,
            args.output / "retained_generic",
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
