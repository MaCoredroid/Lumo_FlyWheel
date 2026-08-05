#!/usr/bin/env python3
"""Compile exact fixed32 committer geometry and index-width cubins."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import linecache
import os
import re
import subprocess
from pathlib import Path


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.jit import MockTensor, create_function_from_signature


SOURCE_PATH = "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
KERNEL = "_fr13_fixed32_committer_native_layer_batch_kernel"
VARIANTS = {
    "incumbent_bv128_warp8": (128, 8, False),
    "control_bv64_warp4_i64": (64, 4, False),
    "candidate_bv64_warp4": (64, 4, True),
}
INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+"
    r"(?:@[!]?(?:U)?P\d+\s+)?([A-Z][A-Z0-9.]*)\b"
)
RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
CONTROL_OR_PADDING = {"NOP", "BAR", "BRA", "EXIT"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(argv: list[str]) -> bytes:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def _source_at_revision(repo: Path, revision: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{SOURCE_PATH}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout


def _jit_kernel(source: str, revision: str):
    tree = ast.parse(source, filename=SOURCE_PATH)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == KERNEL
    )
    lines = source.splitlines(keepends=True)
    synthetic = ["\n"] * len(lines)
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    synthetic[start - 1 : node.end_lineno] = lines[start - 1 : node.end_lineno]
    text = "".join(synthetic)
    canonical_path = f"{SOURCE_PATH}@{revision}:{KERNEL}"
    linecache.cache[canonical_path] = (
        len(text),
        None,
        text.splitlines(keepends=True),
        canonical_path,
    )
    namespace = {
        "__name__": f"fr13_committer_bv64_{revision}",
        "__file__": canonical_path,
        "triton": triton,
        "tl": tl,
    }
    exec(compile(text, canonical_path, "exec"), namespace)
    included = "".join(lines[start - 1 : node.end_lineno])
    return namespace[KERNEL], included


def _specialization(
    fn, backend, *, batch: int, bv: int, warps: int, physical32_i32: bool
):
    args = [
        MockTensor(torch.bfloat16, [48, batch, 32, 48]),
        MockTensor(torch.bfloat16, [48, batch, 32, 48]),
        MockTensor(torch.float32, [48, 48, 2]),
        MockTensor(torch.bfloat16, [48, batch, 32, 16, 128]),
        MockTensor(torch.bfloat16, [48, batch, 32, 48, 128]),
        MockTensor(torch.float32, [257, 48, 128, 128]),
        MockTensor(torch.int64, [48]),
        MockTensor(torch.int32, [batch, 16]),
        MockTensor(torch.int32, [batch]),
        MockTensor(torch.int32, [48, batch, 32]),
        MockTensor(torch.float32, [48, batch, 32, 16]),
        MockTensor(torch.float32, [48, batch, 32, 48, 2]),
    ]
    kwargs = {
        "B": batch,
        "PATH_CAP": 16,
        "H": 16,
        "HV": 48,
        "K": 128,
        "V": 128,
        "BK": 128,
        "BV": bv,
        "BANK_STRIDE": 48 * 128 * 128,
        "GATE_L_STRIDE": 48 * 2,
        "RING_K_L_STRIDE": batch * 32 * 16 * 128,
        "RING_K_B_STRIDE": 32 * 16 * 128,
        "RING_K_N_STRIDE": 16 * 128,
        "RING_KN_L_STRIDE": batch * 32 * 16,
        "RING_KN_B_STRIDE": 32 * 16,
        "RING_KN_N_STRIDE": 16,
        "RING_GATE_L_STRIDE": batch * 32 * 48 * 2,
        "RING_GATE_B_STRIDE": 32 * 48 * 2,
        "RING_GATE_N_STRIDE": 48 * 2,
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
        "K_NORM_REUSE": True,
        "GATE_REUSE": True,
        "DECAY_REUSE": True,
        "PHYSICAL32_I32_INDEX": physical32_i32,
        "num_warps": warps,
        "num_stages": 3,
    }
    if batch != 1:
        kwargs["maxnreg"] = 167
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


def _operations(sass: str) -> collections.Counter[str]:
    result: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            result[match.group(1).split(".", 1)[0]] += 1
    return result


def _compile(
    *, fn, source: str, backend, batch: int, label: str, bv: int, warps: int,
    physical32_i32: bool, output: Path,
) -> dict[str, object]:
    options, signature, constexprs, attrs = _specialization(
        fn,
        backend,
        batch=batch,
        bv=bv,
        warps=warps,
        physical32_i32=physical32_i32,
    )
    compiled = triton.compile(
        ASTSource(
            fn=fn,
            signature=signature,
            constexprs=constexprs,
            attrs=attrs,
        ),
        target=GPUTarget("cuda", 121, 32),
        options=options.__dict__,
    )
    directory = output / label / f"b{batch}"
    directory.mkdir(parents=True, exist_ok=True)
    cubin = compiled.asm["cubin"]
    ptx = compiled.asm["ptx"]
    cubin_path = directory / "kernel.cubin"
    cubin_path.write_bytes(cubin)
    (directory / "kernel.ptx").write_text(ptx, encoding="utf-8")
    sass_raw = _run(["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin_path)])
    resource_raw = _run(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(cubin_path)]
    )
    (directory / "kernel.sass").write_bytes(sass_raw)
    (directory / "resource.txt").write_bytes(resource_raw)
    resource = RESOURCE_RE.search(resource_raw.decode("utf-8"))
    if resource is None:
        raise RuntimeError("unable to parse cubin resource usage")
    registers, stack_bytes, shared_bytes, local_bytes = map(int, resource.groups())
    operations = _operations(sass_raw.decode("utf-8"))
    metadata = (
        compiled.metadata._asdict()
        if hasattr(compiled.metadata, "_asdict")
        else vars(compiled.metadata)
    )
    threads = warps * 32
    result = {
        "batch": batch,
        "block_k": 128,
        "block_v": bv,
        "cubin_bytes": len(cubin),
        "cubin_sha256": _sha256(cubin),
        "encoded_sass_instructions": sum(operations.values()),
        "label": label,
        "launch_shared_bytes": int(metadata["shared"]),
        "local_bytes": local_bytes,
        "num_stages": 3,
        "num_warps": warps,
        "physical32_i32_index": physical32_i32,
        "operations": dict(sorted(operations.items())),
        "programs_per_event": 48 * batch * 48 * (128 // bv),
        "ptx_sha256": _sha256(ptx.encode("utf-8")),
        "register_allocation_per_cta": registers * threads,
        "registers_per_thread": registers,
        "sass_sha256": _sha256(sass_raw),
        "shared_bytes": shared_bytes,
        "stack_bytes": stack_bytes,
        "static_sass_instructions": sum(
            count
            for operation, count in operations.items()
            if operation not in CONTROL_OR_PADDING
        ),
        "threads_per_cta": threads,
        "value_tiles_per_head": 128 // bv,
    }
    (directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = _source_at_revision(args.repo, args.revision)
    fn, included = _jit_kernel(source, args.revision)
    backend = triton.compiler.make_backend(GPUTarget("cuda", 121, 32))
    variants: dict[str, object] = {}
    for label, (bv, warps, physical32_i32) in VARIANTS.items():
        variants[label] = {
            f"b{batch}": _compile(
                fn=fn,
                source=included,
                backend=backend,
                batch=batch,
                label=label,
                bv=bv,
                warps=warps,
                physical32_i32=physical32_i32,
                output=args.output,
            )
            for batch in (1, 4)
        }
    summary = {
        "compile_contract": {
            "batches": [1, 4],
            "decay_reuse": True,
            "dim_k": 128,
            "dim_v": 128,
            "gate_reuse": True,
            "hydra27_physical_rows": 32,
            "candidate_index_bits": 32,
            "k_norm_reuse": True,
            "layers": 48,
            "target": "sm_121a",
            "value_heads": 48,
            "vocab_k": 65536
        },
        "gpu_execution": False,
        "included_kernel_source_sha256": _sha256(included.encode("utf-8")),
        "revision": args.revision,
        "schema": "fr13.fixed32.committer_bv64_warp4.sm121a.codegen.v1",
        "source_path": SOURCE_PATH,
        "source_sha256": _sha256(source.encode("utf-8")),
        "variants": variants,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
