#!/usr/bin/env python3
"""Offline SM121a resource audit for fixed32 logit-direct CFWD kernels."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


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
TARGET = GPUTarget("cuda", 121, 32)
VOCAB_SIZE = 248_320
BLOCK_V = 4096
MAX_VOCAB_BLOCKS = 262_144 // BLOCK_V
ACTIVE_BLOCKS = (VOCAB_SIZE + BLOCK_V - 1) // BLOCK_V


KERNELS = {
    "block_stats": {
        "function": "_fr13_cfwd_logit_block_stats_kernel",
        "signature": {
            "self_logits": "*fp32",
            "target_logits": "*fp32",
            "self_source_indices": "*i64",
            "target_source_indices": "*i64",
            "block_maxima": "*fp32",
            "block_sums": "*fp32",
            "vocab_size": "i32",
            "self_total_rows": "i32",
        },
        "constexprs": {
            "BLOCK_V": BLOCK_V,
            "MAX_BLOCKS": MAX_VOCAB_BLOCKS,
        },
        "num_warps": 4,
        "ctas_b1": 30 * ACTIVE_BLOCKS,
        "ctas_b4": 4 * 30 * ACTIVE_BLOCKS,
    },
    "direct_decision": {
        "function": "_fr13_cfwd_logit_direct_decision_kernel",
        "signature": {
            "self_logits": "*fp32",
            "target_logits": "*fp32",
            "self_source_indices": "*i64",
            "target_source_indices": "*i64",
            "block_maxima": "*fp32",
            "block_sums": "*fp32",
            "drafts": "*i64",
            "child_table": "*i64",
            "child_counts": "*i64",
            "self_uniform_levels": "*i64",
            "target_parent_slots": "*i64",
            "target_uniform_levels": "*i64",
            "uniforms": "*fp32",
            "self_token_out": "*i64",
            "source_out": "*i64",
            "selected_token_out": "*i64",
            "rejected_token_out": "*i64",
            "accepted_out": "*i1",
            "vocab_size": "i32",
            "number_of_blocks": "i32",
            "self_total_rows": "i32",
        },
        "constexprs": {
            "SELF_ROWS": 13,
            "TARGET_ROWS": 17,
            "PHYSICAL_DRAFTS": 31,
            "PHYSICAL_ROWS": 32,
            "FANOUT": 3,
            "WALK_CAP": 12,
            "BLOCK_V": BLOCK_V,
            "MAX_BLOCKS": MAX_VOCAB_BLOCKS,
        },
        "num_warps": 8,
        "ctas_b1": 30,
        "ctas_b4": 120,
    },
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run_text(arguments: list[str]) -> str:
    return subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(
        "_fr13_cfwd_logit_direct_codegen", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compile(module, name: str, contract: dict[str, object]) -> dict[str, object]:
    backend = triton.compiler.make_backend(TARGET)
    options = backend.parse_options(
        {"num_warps": int(contract["num_warps"]), "num_stages": 3}
    )
    compiled = triton.compile(
        ASTSource(
            fn=getattr(module, str(contract["function"])),
            signature=contract["signature"],
            constexprs=contract["constexprs"],
        ),
        target=TARGET,
        options=options.__dict__,
    )
    cubin = compiled.asm["cubin"]
    if not isinstance(cubin, bytes):
        cubin = bytes(cubin)
    with tempfile.TemporaryDirectory(prefix=f"fr13_cfwd_{name}_") as scratch:
        cubin_path = Path(scratch) / "kernel.cubin"
        cubin_path.write_bytes(cubin)
        resource = _run_text(
            ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(cubin_path)]
        )
        sass = _run_text(["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin_path)])
    match = RESOURCE_RE.search(resource)
    if match is None:
        raise RuntimeError(f"cannot parse resource usage for {name}")
    registers, stack_bytes, shared_bytes, local_bytes = map(int, match.groups())
    operations: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        instruction = INSTRUCTION_RE.match(line)
        if instruction:
            operations[instruction.group(1).split(".", 1)[0]] += 1
    metadata = (
        compiled.metadata._asdict()
        if hasattr(compiled.metadata, "_asdict")
        else vars(compiled.metadata)
    )
    return {
        "kernel": name,
        "function": contract["function"],
        "shared_b1_b4_binary": True,
        "num_warps": contract["num_warps"],
        "ctas_b1": contract["ctas_b1"],
        "ctas_b4": contract["ctas_b4"],
        "registers_per_thread": registers,
        "stack_bytes": stack_bytes,
        "shared_bytes": shared_bytes,
        "launch_shared_bytes": int(metadata["shared"]),
        "local_bytes": local_bytes,
        "spill_loads": operations["LDL"],
        "spill_stores": operations["STL"],
        "calls": sum(
            count for opcode, count in operations.items() if opcode.startswith("CALL")
        ),
        "cubin_bytes": len(cubin),
        "cubin_sha256": _sha256(cubin),
        "ptx_sha256": _sha256(compiled.asm["ptx"].encode("utf-8")),
        "compile_hash": metadata["hash"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    source_raw = source.read_bytes()
    source_sha256 = _sha256(source_raw)
    if source_sha256 != args.expected_source_sha256:
        raise SystemExit(
            f"source digest drift: {source_sha256} != {args.expected_source_sha256}"
        )
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if shutil.which("cuobjdump") is None or shutil.which("nvdisasm") is None:
        raise SystemExit("cuobjdump and nvdisasm are required")
    module = _load(source)
    if module.VOCAB_SIZE != VOCAB_SIZE:
        raise SystemExit(
            f"candidate verifier vocab drift: {module.VOCAB_SIZE} != {VOCAB_SIZE}"
        )
    if module.BLOCK_V != BLOCK_V or module.MAX_BLOCKS != MAX_VOCAB_BLOCKS:
        raise SystemExit(
            "candidate block geometry drift: "
            f"block_v={module.BLOCK_V} max_blocks={module.MAX_BLOCKS}"
        )
    kernels = [_compile(module, name, contract) for name, contract in KERNELS.items()]
    payload = {
        "schema": "fr13.fixed32.cfwd_logit_direct.codegen.v1",
        "status": "pass" if all(
            item["stack_bytes"] == 0
            and item["local_bytes"] == 0
            and item["spill_loads"] == 0
            and item["spill_stores"] == 0
            and item["calls"] == 0
            for item in kernels
        ) else "rejected_resource_regression",
        "candidate": module.CANDIDATE,
        "source_sha256": source_sha256,
        "architecture": "sm_121a",
        "vocab_size": VOCAB_SIZE,
        "block_v": BLOCK_V,
        "max_blocks": MAX_VOCAB_BLOCKS,
        "active_vocab_blocks": ACTIVE_BLOCKS,
        "toolchain": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__,
            "num_stages": 3,
        },
        "kernels": kernels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
