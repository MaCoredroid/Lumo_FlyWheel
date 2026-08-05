#!/usr/bin/env python3
"""Compile the bounded active-depth packed walk for SM121a without a GPU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")

import torch
import triton
from triton.compiler import ASTSource
from triton.runtime.jit import MockTensor, create_function_from_signature


BASE_REVISION = "ed66c077bd543f90ad18a78ea974325227a21d7d"
BASE_SOURCE = "scripts/fr13_cfwd_packed_walk_node_trust_kernel.py"
CANDIDATE_SOURCE = "scripts/fr13_cfwd_packed_walk_active_depth_kernel.py"
BASE_KERNEL = "_fr13_fixed32_taw_packed_node_trust_commit_kernel"
CANDIDATE_KERNEL = (
    "_fr13_fixed32_taw_packed_active_depth_commit_kernel"
)


def _helpers():
    path = (
        Path(__file__).resolve().parents[1]
        / "fr13_fixed32_cfwd_physical_slots_sm121a_codegen_20260805"
        / "offline_codegen_audit.py"
    )
    spec = importlib.util.spec_from_file_location("fr13_active_depth_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load codegen helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _helpers()


def _arguments(batch: int) -> list[object]:
    return [
        MockTensor(torch.int64, [batch, 31]),
        MockTensor(torch.int64, [batch, 32]),
        MockTensor(torch.int64, [batch]),
        MockTensor(torch.int64, [batch, 32]),
        MockTensor(torch.int64, [batch]),
        MockTensor(torch.int64, [batch, 16]),
        MockTensor(torch.int64, [batch]),
        MockTensor(torch.int64, [batch]),
    ]


def _compile(fn, source: str, *, batch: int) -> dict[str, object]:
    backend = triton.compiler.make_backend(H.TARGET)
    keywords = {
        "PHYSICAL_DRAFTS": 31,
        "PHYSICAL_ROWS": 32,
        "WALK_CAP": 12,
        "OUTPUT_CAPACITY": 32,
        "PATH_CAPACITY": 16,
        "num_warps": 1,
    }
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    bound, specialization, options = binder(*_arguments(batch), **keywords)
    compile_options, signature, constexprs, attrs = fn._pack_args(
        backend,
        options,
        bound,
        specialization,
        options,
    )
    compiled = triton.compile(
        ASTSource(
            fn=fn,
            signature=signature,
            constexprs=constexprs,
            attrs=attrs,
        ),
        target=H.TARGET,
        options=compile_options.__dict__,
    )
    result = H.compiled_result(compiled, function_source=source)
    with tempfile.TemporaryDirectory(prefix="fr13_active_depth_") as scratch:
        cubin = Path(scratch) / "kernel.cubin"
        cubin.write_bytes(compiled.asm["cubin"])
        sass = subprocess.run(
            ["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin)],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
    counts = H.operation_counts(sass)
    result["bra"] = counts["BRA"]
    result["batch"] = batch
    result["num_warps"] = int(compile_options.num_warps)
    result["num_stages"] = int(compile_options.num_stages)
    return result


def _clean(build: dict[str, object]) -> bool:
    return all(
        int(build[name]) == 0
        for name in ("stack_bytes", "local_bytes", "ldl", "stl", "calls")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)

    base_source = H.source_at_revision(repo, BASE_REVISION, BASE_SOURCE)
    candidate_source = H.source_at_revision(repo, args.revision, CANDIDATE_SOURCE)
    base_fn, base_function = H.jit_function(
        base_source, BASE_SOURCE, BASE_REVISION, BASE_KERNEL
    )
    candidate_fn, candidate_function = H.jit_function(
        candidate_source,
        CANDIDATE_SOURCE,
        args.revision,
        CANDIDATE_KERNEL,
    )
    summary: dict[str, object] = {
        "schema": "fr13.fixed32.cfwd_packed_walk.active_depth.sm121a.v1",
        "status": "pending",
        "claim_scope": "static_sm121a_codegen_no_gpu_runtime_claim",
        "base_revision": BASE_REVISION,
        "candidate_revision": args.revision,
        "source_paths": {
            "base": BASE_SOURCE,
            "candidate": CANDIDATE_SOURCE,
        },
        "source_sha256": {
            "base": H.sha256(base_source.encode()),
            "candidate": H.sha256(candidate_source.encode()),
            "base_kernel": H.sha256(base_function.encode()),
            "candidate_kernel": H.sha256(candidate_function.encode()),
        },
        "compile_contract": {
            "target": "sm_121a",
            "batches": [1, 4],
            "physical_rows": 32,
            "maximum_walk_iterations": 12,
            "output_capacity": 32,
            "path_capacity": 16,
            "programs_per_request": 1,
            "num_warps": 1,
            "jit_specialization": "exact_mock_tensor_shape_stride_alignment",
        },
        "exact_work": {
            "base_emitted_walk_bodies": 12,
            "candidate_emitted_walk_bodies": 1,
            "candidate_dynamic_backedge": True,
            "candidate_executed_iterations": (
                "min(realized accepted drafts + 1, 12)"
            ),
            "maximum_iterations_before": 12,
            "maximum_iterations_after": 12,
            "topology_size_controls_loop_bound": False,
            "output_products_changed": 0,
        },
        "builds": {"base": {}, "candidate": {}},
    }
    for batch in (1, 4):
        summary["builds"]["base"][f"b{batch}"] = _compile(
            base_fn, base_function, batch=batch
        )
        summary["builds"]["candidate"][f"b{batch}"] = _compile(
            candidate_fn, candidate_function, batch=batch
        )

    improves = True
    for batch in ("b1", "b4"):
        base = summary["builds"]["base"][batch]
        candidate = summary["builds"]["candidate"][batch]
        improves &= (
            int(candidate["registers"]) <= int(base["registers"])
            and int(candidate["ldg"]) < int(base["ldg"])
            and int(candidate["static_noncontrol_sass_instructions"])
            < int(base["static_noncontrol_sass_instructions"])
            and int(candidate["stg"]) <= int(base["stg"])
            and _clean(base)
            and _clean(candidate)
        )
    summary["conclusion"] = {
        "candidate_static_improves_b1_b4": improves,
        "bounded_dynamic_walk": True,
        "gpu_execution": False,
        "runtime_speedup_claimed": False,
        "real_swe_verified_byte_gate_required": True,
    }
    summary["status"] = "pass" if improves else "rejected"
    (output / "codegen_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
