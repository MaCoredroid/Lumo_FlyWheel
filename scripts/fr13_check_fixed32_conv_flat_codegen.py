#!/usr/bin/env python3
"""Offline SM121a codegen audit for the fixed32 flat conv committer."""

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
import tempfile


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit(
        "refusing to run unless CUDA_VISIBLE_DEVICES is explicitly empty"
    )

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


KERNEL_NAME = "_fr13_fixed32_conv_flat_zeroelide_col0_kernel"
SIGNATURE = {
    "anchor_ptr": "*bf16",
    "bank_off16": "*i64",
    "source_anchor": "*bf16",
    "source_off16": "*i64",
    "state_src": "*i64",
    "spec_state_indices": "*i32",
    "accepted_paths": "*i32",
    "accepted_lens": "*i32",
    "ssi_stride_l": "i32",
    "ssi_stride_b": "i32",
    "ssi_stride_s": "i32",
    "path_stride_b": "i32",
    "path_stride_s": "i32",
    "lens_stride_b": "i32",
    "bank_row_stride": "i32",
    "source_row_stride": "i32",
}
BASE_CONSTANTS = {
    "CONV_C": 10_240,
    "CONV_L": 34,
    "LIVE_SOURCE_COLS": 3,
    "SOURCE_ROWS": 36,
    "ELEM_BYTES": 2,
    "SPEC_COLS": 32,
    "PATH_COLS": 16,
    "BLOCK": 1024,
}
INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@[!]?P\d+\s+)?([A-Z][A-Z0-9.]*)\b"
)
RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run_text(argv: list[str]) -> str:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout


def _jit_function(source: str, canonical_path: str):
    tree = ast.parse(source, filename=canonical_path)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == KERNEL_NAME
    )
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    lines = source.splitlines(keepends=True)
    function_source = "".join(lines[start - 1 : node.end_lineno])
    synthetic = "\n" * (start - 1) + function_source
    linecache.cache[canonical_path] = (
        len(synthetic),
        None,
        synthetic.splitlines(keepends=True),
        canonical_path,
    )
    namespace = {
        "__name__": "fr13_fixed32_conv_flat_offline_codegen",
        "__file__": canonical_path,
        "triton": triton,
        "tl": tl,
    }
    exec(compile(synthetic, canonical_path, "exec"), namespace)
    return namespace[KERNEL_NAME], function_source


def _compile_one(
    *,
    source: str,
    canonical_path: str,
    batch: int,
) -> dict[str, object]:
    function, function_source = _jit_function(source, canonical_path)
    constants = dict(BASE_CONSTANTS, B=batch)
    target = GPUTarget("cuda", 121, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": 4, "num_stages": 3})
    compiled = triton.compile(
        ASTSource(fn=function, signature=SIGNATURE, constexprs=constants),
        target=target,
        options=options.__dict__,
    )
    cubin = compiled.asm["cubin"]
    with tempfile.TemporaryDirectory(prefix="fr13-conv-flat-codegen-") as raw:
        cubin_path = Path(raw) / "kernel.cubin"
        cubin_path.write_bytes(cubin)
        sass = _run_text(["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin_path)])
        resources = _run_text(
            [
                "/usr/local/cuda/bin/cuobjdump",
                "--dump-resource-usage",
                str(cubin_path),
            ]
        )
    resource_match = RESOURCE_RE.search(resources)
    if resource_match is None:
        raise RuntimeError("cannot parse Triton resource usage")
    registers, stack_bytes, shared_bytes, local_bytes = map(
        int, resource_match.groups()
    )
    operations: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            operations[match.group(1).split(".", 1)[0]] += 1
    forbidden = {
        "CALL": operations["CALL"],
        "LDL": operations["LDL"],
        "STL": operations["STL"],
    }
    if stack_bytes or local_bytes or any(forbidden.values()):
        raise RuntimeError(
            "flat conv committer generated stack/local/call instructions"
        )
    metadata = (
        compiled.metadata._asdict()
        if hasattr(compiled.metadata, "_asdict")
        else vars(compiled.metadata)
    )
    ctas_per_layer_request = (
        BASE_CONSTANTS["CONV_C"]
        * BASE_CONSTANTS["CONV_L"]
        // BASE_CONSTANTS["BLOCK"]
    )
    return {
        "batch": batch,
        "compile_hash": metadata["hash"],
        "cubin_bytes": len(cubin),
        "cubin_sha256": _sha256(cubin),
        "function_source_sha256": _sha256(function_source.encode()),
        "registers_per_thread": registers,
        "stack_bytes": stack_bytes,
        "local_bytes": local_bytes,
        "reported_shared_bytes": shared_bytes,
        "launch_shared_bytes": int(metadata["shared"]),
        "forbidden_sass_counts": forbidden,
        "sass_counts": {
            name: operations[name]
            for name in ("BRA", "EXIT", "LDG", "STG", "ISETP", "SEL")
        },
        "ctas_per_layer_request": ctas_per_layer_request,
        "ctas_per_event": 48 * batch * ctas_per_layer_request,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
        ),
    )
    args = parser.parse_args()
    source = args.source.read_text()
    canonical_path = str(args.source.resolve())
    variants = [
        _compile_one(source=source, canonical_path=canonical_path, batch=batch)
        for batch in (1, 4)
    ]
    receipt = {
        "schema": "fr13.fixed32.conv_flat_zeroelide_codegen.v1",
        "architecture": "sm_121a",
        "kernel": KERNEL_NAME,
        "source_file_sha256": _sha256(source.encode()),
        "constants": BASE_CONSTANTS,
        "variants": variants,
        "semantic_traffic_bytes": {
            str(batch): {
                "source_reads": 48
                * batch
                * BASE_CONSTANTS["CONV_C"]
                * BASE_CONSTANTS["LIVE_SOURCE_COLS"]
                * BASE_CONSTANTS["ELEM_BYTES"],
                "destination_writes": 48
                * batch
                * BASE_CONSTANTS["CONV_C"]
                * BASE_CONSTANTS["CONV_L"]
                * BASE_CONSTANTS["ELEM_BYTES"],
            }
            for batch in (1, 4)
        },
        "contract_pass": True,
        "gpu_visible": False,
        "performance_measured": False,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
