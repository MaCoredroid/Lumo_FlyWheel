#!/usr/bin/env python3
"""Offline-only SM121a codegen audit for fixed32 fused decisions."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit(
        "refusing to compile unless CUDA_VISIBLE_DEVICES is explicitly empty"
    )

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@[!]?P\d+\s+)?([A-Z][A-Z0-9.]*)\b"
)

KERNELS = {
    "probability_block_sums": {
        "function": "_fr13_cfwd_probability_block_sums_kernel",
        "signature": {
            "self_probability": "*fp32",
            "target_probability": "*fp32",
            "block_sums": "*fp32",
            "vocab_size": "i32",
            "self_total_rows": "i32",
        },
        "constexprs": {
            "SELF_ROWS": 13,
            "TARGET_ROWS": 17,
            "BLOCK_V": 256,
            "MAX_BLOCKS": 1024,
        },
        "num_warps": 4,
        "ctas_per_request": 30 * 1024,
    },
    "parent_setup": {
        "function": "_fr13_cfwd_parent_setup_kernel",
        "signature": {
            "target_probability": "*fp32",
            "probability_block_sums": "*fp32",
            "drafts": "*i64",
            "child_table": "*i64",
            "child_counts": "*i64",
            "target_parent_slots": "*i64",
            "target_uniform_levels": "*i64",
            "uniforms": "*fp32",
            "target_totals": "*fp32",
            "kid_tokens_out": "*i64",
            "q_weights_out": "*fp32",
            "source_out": "*i64",
            "selected_token_out": "*i64",
            "accepted_out": "*i1",
            "vocab_size": "i32",
            "self_total_rows": "i32",
        },
        "constexprs": {
            "SELF_ROWS": 13,
            "TARGET_ROWS": 17,
            "PHYSICAL_DRAFTS": 31,
            "PHYSICAL_ROWS": 32,
            "FANOUT": 3,
            "WALK_CAP": 12,
            "MAX_BLOCKS": 1024,
        },
        "num_warps": 8,
        "ctas_per_request": 17,
    },
    "residual_block_sums": {
        "function": "_fr13_cfwd_residual_block_sums_kernel",
        "signature": {
            "target_probability": "*fp32",
            "target_totals": "*fp32",
            "kid_tokens": "*i64",
            "q_weights": "*fp32",
            "residual_block_sums": "*fp32",
            "vocab_size": "i32",
        },
        "constexprs": {
            "TARGET_ROWS": 17,
            "FANOUT": 3,
            "BLOCK_V": 256,
            "MAX_BLOCKS": 1024,
        },
        "num_warps": 4,
        "ctas_per_request": 17 * 1024,
    },
    "inverse_cdf": {
        "function": "_fr13_cfwd_inverse_cdf_kernel",
        "signature": {
            "self_probability": "*fp32",
            "target_probability": "*fp32",
            "probability_block_sums": "*fp32",
            "residual_block_sums": "*fp32",
            "target_totals": "*fp32",
            "kid_tokens": "*i64",
            "q_weights": "*fp32",
            "self_uniform_levels": "*i64",
            "target_uniform_levels": "*i64",
            "uniforms": "*fp32",
            "self_token_out": "*i64",
            "rejected_token_out": "*i64",
            "vocab_size": "i32",
            "self_total_rows": "i32",
        },
        "constexprs": {
            "SELF_ROWS": 13,
            "TARGET_ROWS": 17,
            "FANOUT": 3,
            "WALK_CAP": 12,
            "BLOCK_V": 256,
            "MAX_BLOCKS": 1024,
        },
        "num_warps": 8,
        "ctas_per_request": 30,
    },
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_text(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _load_source(path: Path):
    spec = importlib.util.spec_from_file_location(
        "_fr13_cfwd_fused_decision_codegen",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compile_one(module, name: str, contract: dict, output: Path) -> dict:
    function = getattr(module, contract["function"])
    target = GPUTarget("cuda", 121, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options(
        {"num_warps": contract["num_warps"], "num_stages": 3}
    )
    compiled = triton.compile(
        ASTSource(
            fn=function,
            signature=contract["signature"],
            constexprs=contract["constexprs"],
        ),
        target=target,
        options=options.__dict__,
    )
    kernel_dir = output / name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    cubin = compiled.asm["cubin"]
    cubin_path = kernel_dir / "kernel.cubin"
    cubin_path.write_bytes(cubin)
    sass = _run_text(["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin_path)])
    resource = _run_text(
        [
            "/usr/local/cuda/bin/cuobjdump",
            "--dump-resource-usage",
            str(cubin_path),
        ]
    )
    resource_match = RESOURCE_RE.search(resource)
    if resource_match is None:
        raise RuntimeError(f"cannot parse resource usage for {name}")
    registers, stack_bytes, shared_bytes, local_bytes = map(
        int, resource_match.groups()
    )
    operations: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            operations[match.group(1).split(".", 1)[0]] += 1
    metadata = (
        compiled.metadata._asdict()
        if hasattr(compiled.metadata, "_asdict")
        else vars(compiled.metadata)
    )
    return {
        "kernel": name,
        "function": contract["function"],
        "num_warps": contract["num_warps"],
        "ctas_b1": contract["ctas_per_request"],
        "ctas_b4": contract["ctas_per_request"] * 4,
        "registers": registers,
        "stack_bytes": stack_bytes,
        "shared_bytes": shared_bytes,
        "launch_shared_bytes": int(metadata["shared"]),
        "local_bytes": local_bytes,
        "spill_loads": operations["LDL"],
        "spill_stores": operations["STL"],
        "calls": sum(
            count
            for operation, count in operations.items()
            if operation.startswith("CALL")
        ),
        "ldg": operations["LDG"],
        "stg": operations["STG"],
        "sass_instructions": sum(operations.values()),
        "cubin_bytes": len(cubin),
        "cubin_sha256": _sha256(cubin),
        "ptx_sha256": _sha256(compiled.asm["ptx"].encode("utf-8")),
        "sass_sha256": _sha256(sass.encode("utf-8")),
        "compile_hash": metadata["hash"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    source_sha256 = _sha256(source_bytes)
    if source_sha256 != args.expected_source_sha256:
        raise SystemExit(
            f"source digest drift: {source_sha256} != {args.expected_source_sha256}"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    module = _load_source(args.source.resolve())
    results = [
        _compile_one(module, name, contract, args.output)
        for name, contract in KERNELS.items()
    ]
    payload = {
        "schema": "fr13.fixed32.cfwd_sparse_decisions.codegen.v1",
        "source_sha256": source_sha256,
        "candidate": module.CANDIDATE,
        "architecture": "sm_121a",
        "toolchain": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__,
            "num_stages": 3,
        },
        "kernels": results,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
