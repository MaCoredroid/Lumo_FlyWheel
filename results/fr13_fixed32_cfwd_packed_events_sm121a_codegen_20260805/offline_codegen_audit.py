#!/usr/bin/env python3
"""Compile fixed32 packed-event CFWD against physical-slot v2 without a GPU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")

import torch
import triton
from triton.compiler import ASTSource
from triton.runtime.jit import MockTensor, create_function_from_signature


BASE_REVISION = "1f7485ade5ec6bfacf51dde7afa514531effcbcd"
DEVICE_SOURCE = "scripts/fr13_device_multidraft_kernel.py"
PRODUCER_SOURCE = "scripts/fr13_cfwd_logit_direct_decision_kernel.py"
BASE_COMMIT = "_fr13_fixed32_taw_physical_slot_commit_kernel"
CANDIDATE_COMMIT = "_fr13_fixed32_taw_packed_physical_slot_commit_kernel"
COMPARATOR = "_fr13_cfwd_logit_direct_compare_kernel"
BLOCK_STATS = "_fr13_cfwd_logit_block_stats_kernel"
DIRECT_DECISION = "_fr13_cfwd_logit_direct_decision_kernel"


def _audit_helpers():
    path = (
        Path(__file__).resolve().parents[1]
        / "fr13_fixed32_cfwd_physical_slots_sm121a_codegen_20260805"
        / "offline_codegen_audit.py"
    )
    spec = importlib.util.spec_from_file_location("fr13_cfwd_codegen_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load codegen helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _audit_helpers()


def _compile_bound(
    fn,
    source: str,
    arguments: list[object],
    keywords: dict[str, int],
) -> dict[str, object]:
    backend = triton.compiler.make_backend(H.TARGET)
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    bound, specialization, options = binder(*arguments, **keywords)
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
    result["num_warps"] = int(compile_options.num_warps)
    result["num_stages"] = int(compile_options.num_stages)
    return result


def _candidate_decision_contract() -> dict[str, object]:
    signature = dict(H.producer_contract(DIRECT_DECISION)["signature"])
    del signature["source_out"]
    del signature["selected_token_out"]
    del signature["rejected_token_out"]
    del signature["accepted_out"]
    signature["event_out"] = "*i64"
    return {
        "signature": signature,
        "constexprs": H.producer_contract(DIRECT_DECISION)["constexprs"],
        "num_warps": 8,
        "num_stages": 3,
    }


def _candidate_commit_arguments(batch: int) -> list[object]:
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


def _candidate_comparator_arguments(batch: int) -> list[object]:
    return [
        MockTensor(torch.int32, [1]),
        MockTensor(torch.int64, [1]),
        MockTensor(torch.int64, [5]),
        MockTensor(torch.int64, [5]),
        MockTensor(torch.int64, [batch * 13]),
        MockTensor(torch.int64, [17]),
        MockTensor(torch.int64, [batch, 32, 3]),
        MockTensor(torch.int64, [batch, 32]),
        MockTensor(torch.int64, [batch, 13]),
        MockTensor(torch.int64, [batch, 31]),
        MockTensor(torch.int64, [batch, 17]),
        MockTensor(torch.int64, [batch, 17]),
        MockTensor(torch.int64, [batch, 17]),
        MockTensor(torch.bool, [batch, 17]),
        MockTensor(torch.int64, [batch, 32]),
        MockTensor(torch.int64, [batch, 32]),
        MockTensor(torch.int64, [batch, 32]),
        MockTensor(torch.int64, [batch]),
        MockTensor(torch.int64, [batch]),
        MockTensor(torch.int64, [batch, 16]),
        MockTensor(torch.int64, [batch, 16]),
        MockTensor(torch.int64, [batch]),
        MockTensor(torch.int64, [batch]),
        MockTensor(torch.int64, [batch]),
        MockTensor(torch.int64, [batch]),
    ]


def _resource_clean(build: dict[str, object]) -> bool:
    return all(
        build[name] == 0
        for name in ("stack_bytes", "local_bytes", "ldl", "stl", "calls")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    revision = args.revision
    if revision == BASE_REVISION:
        raise SystemExit("candidate revision must differ from frozen base")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)

    sources = {
        "base": {
            DEVICE_SOURCE: H.source_at_revision(repo, BASE_REVISION, DEVICE_SOURCE),
            PRODUCER_SOURCE: H.source_at_revision(
                repo, BASE_REVISION, PRODUCER_SOURCE
            ),
        },
        "candidate": {
            DEVICE_SOURCE: H.source_at_revision(repo, revision, DEVICE_SOURCE),
            PRODUCER_SOURCE: H.source_at_revision(repo, revision, PRODUCER_SOURCE),
        },
    }
    summary: dict[str, object] = {
        "schema": "fr13.fixed32.cfwd_packed_events.sm121a.codegen.v1",
        "status": "pending",
        "base_revision": BASE_REVISION,
        "candidate_revision": revision,
        "claim_scope": (
            "static_sm121a_codegen_and_exact_cpu_semantics_no_runtime_claim"
        ),
        "source_sha256": {
            label: {path: H.sha256(source.encode()) for path, source in files.items()}
            for label, files in sources.items()
        },
        "source_contracts": {
            "candidate": {
                "name": "fixed32_cfwd_logit_direct_packed_physical_slots_v3",
                "schema": (
                    "fr13.fixed32.cfwd_logit_direct_packed_physical_slots.v3"
                ),
                "sha256": (
                    "5a9107306bdc37200448a6a5add2b84dfd839dc377b11009f218662c63abcc1c"
                ),
            },
            "cfwd_integration": {
                "schema": "fr13.fixed32.cfwd_logit_direct.integration_source.v2",
                "sha256": (
                    "a82ce3f5e526792ca45bb444212e5440e8444778f174fd0650accc4bb5f8558c"
                ),
            },
            "taw": {
                "schema": "fr13-fixed32-taw-all-parent-v7",
                "sha256": (
                    "998bc6331177469d6890f97f3e066e1d07c2ca2d8ab4bff723f32d5229fef290"
                ),
                "unchanged_by_candidate": True,
            },
        },
        "packed_event_contract": {
            "accepted_node_zero_row": 1,
            "accepted_row_mask": 0x1F,
            "accepted_row_shift": 18,
            "parent_mask": 0x800000,
            "rejection_accepted_row": 0,
            "token_mask": 0x3FFFF,
            "verifier_vocab_size": 248_320,
            "verifier_vocab_fits_token_bits": True,
        },
        "exact_work": {
            "physical_rows": 32,
            "walk_levels": 12,
            "decision_programs_per_request_before": 30,
            "decision_programs_per_request_after": 30,
            "decision_values_stored_per_request_before": 81,
            "decision_values_stored_per_request_after": 30,
            "decision_workspace_bytes_per_request_before": 1048,
            "decision_workspace_bytes_per_request_after": 504,
            "tree_metadata_loads_per_request_before": 24,
            "tree_metadata_loads_per_request_after": 0,
            "commit_programs_per_request_before": 1,
            "commit_programs_per_request_after": 1,
        },
        "producer": {},
        "commit": {},
        "comparator": {},
    }

    for label, revision_label in (("base", BASE_REVISION), ("candidate", revision)):
        producer_source = sources[label][PRODUCER_SOURCE]
        builds = {}
        for name in (BLOCK_STATS, DIRECT_DECISION):
            fn, function_source = H.jit_function(
                producer_source,
                PRODUCER_SOURCE,
                revision_label,
                name,
            )
            contract = (
                _candidate_decision_contract()
                if label == "candidate" and name == DIRECT_DECISION
                else H.producer_contract(name)
            )
            builds[name] = H.compile_plain(
                fn,
                signature=contract["signature"],
                constexprs=contract["constexprs"],
                num_warps=contract["num_warps"],
                num_stages=contract["num_stages"],
                function_source=function_source,
            )
        summary["producer"][label] = builds

    base_fn, base_source = H.jit_function(
        sources["base"][DEVICE_SOURCE],
        DEVICE_SOURCE,
        BASE_REVISION,
        BASE_COMMIT,
    )
    candidate_fn, candidate_source = H.jit_function(
        sources["candidate"][DEVICE_SOURCE],
        DEVICE_SOURCE,
        revision,
        CANDIDATE_COMMIT,
    )
    summary["commit"] = {"base": {}, "candidate": {}}
    for batch in (1, 4):
        base = H.compile_commit(
            base_fn,
            candidate=True,
            batch=batch,
            function_source=base_source,
        )
        candidate = _compile_bound(
            candidate_fn,
            candidate_source,
            _candidate_commit_arguments(batch),
            {
                "PHYSICAL_DRAFTS": 31,
                "PHYSICAL_ROWS": 32,
                "WALK_CAP": 12,
                "OUTPUT_CAPACITY": 32,
                "PATH_CAPACITY": 16,
                "num_warps": 1,
            },
        )
        base["batch"] = batch
        candidate["batch"] = batch
        summary["commit"]["base"][f"b{batch}"] = base
        summary["commit"]["candidate"][f"b{batch}"] = candidate

    comparator_fn, comparator_source = H.jit_function(
        sources["candidate"][DEVICE_SOURCE],
        DEVICE_SOURCE,
        revision,
        COMPARATOR,
    )
    for batch in (1, 4):
        summary["comparator"][f"b{batch}"] = _compile_bound(
            comparator_fn,
            comparator_source,
            _candidate_comparator_arguments(batch),
            {
                "TARGET_ROWS": 17,
                "PHYSICAL_ROWS": 32,
                "SELF_N": batch * 13,
                "TARGET_N": batch * 17,
                "OUTPUT_N": batch * 32,
                "BATCH_N": batch,
                "PATH_N": batch * 16,
                "BLOCK": 128,
                "num_warps": 4,
            },
        )

    base_decision = summary["producer"]["base"][DIRECT_DECISION]
    candidate_decision = summary["producer"]["candidate"][DIRECT_DECISION]
    producer_ok = (
        candidate_decision["registers"] <= base_decision["registers"]
        and candidate_decision["stg"] < base_decision["stg"]
        and _resource_clean(candidate_decision)
    )
    committer_ok = all(
        summary["commit"]["candidate"][batch][field]
        < summary["commit"]["base"][batch][field]
        for batch in ("b1", "b4")
        for field in ("registers", "ldg", "static_noncontrol_sass_instructions")
    ) and all(
        _resource_clean(build)
        for build in summary["commit"]["candidate"].values()
    )
    comparator_ok = all(
        _resource_clean(build) for build in summary["comparator"].values()
    )
    summary["conclusion"] = {
        "producer_registers_preserved_and_stores_reduced": producer_ok,
        "committer_static_improves": committer_ok,
        "comparator_resource_clean": comparator_ok,
        "runtime_speedup_claimed": False,
        "real_swe_verified_gate_required": True,
    }
    summary["status"] = (
        "pass" if producer_ok and committer_ok and comparator_ok else "rejected"
    )
    (output / "codegen_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
