#!/usr/bin/env python3
"""Compile fixed32 CFWD physical-slot kernels for SM121a without a GPU."""

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
import textwrap


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.jit import MockTensor, create_function_from_signature


BASE_REVISION = "dfb04b3b20e246118006ab2f4cb91a4a196f2491"
DEVICE_SOURCE = "scripts/fr13_device_multidraft_kernel.py"
PRODUCER_SOURCE = "scripts/fr13_cfwd_logit_direct_decision_kernel.py"
INCUMBENT_COMMIT = "_fr13_fixed32_taw_all_parent_commit_kernel"
CANDIDATE_COMMIT = "_fr13_fixed32_taw_physical_slot_commit_kernel"
COMPARATOR = "_fr13_cfwd_logit_direct_compare_kernel"
BLOCK_STATS = "_fr13_cfwd_logit_block_stats_kernel"
DIRECT_DECISION = "_fr13_cfwd_logit_direct_decision_kernel"
TARGET = GPUTarget("cuda", 121, 32)
CONTROL_OR_PADDING = {"NOP", "BAR", "BRA", "EXIT"}
RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+"
    r"(?:@[^ ]+\s+)?([A-Z][A-Z0-9.]*)\b"
)
TOOLKIT_RE = re.compile(r"Tool Kit Version:\s*([^\s]+)")
TOOL_NAME_RE = re.compile(r"Tool Name:\s*(\S+)")
TOOL_VERSION_RE = re.compile(r"Tool Version:\s*(.+)")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_at_revision(repo: Path, revision: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout


def jit_function(source: str, path: str, revision: str, name: str):
    tree = ast.parse(source, filename=path)
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    function_source = textwrap.dedent(
        "".join(source.splitlines(keepends=True)[start - 1 : node.end_lineno])
    )
    synthetic = "\n" * (start - 1) + function_source
    canonical = f"{path}@{revision}:{name}"
    linecache.cache[canonical] = (
        len(synthetic),
        None,
        synthetic.splitlines(keepends=True),
        canonical,
    )
    namespace = {
        "__file__": canonical,
        "__name__": f"fr13_physical_slots_{revision}_{name}",
        "triton": triton,
        "tl": tl,
    }
    exec(compile(synthetic, canonical, "exec"), namespace)
    return namespace[name], function_source


def operation_counts(sass: str) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            counts[match.group(1).split(".", 1)[0]] += 1
    return counts


def compiled_result(compiled, *, function_source: str) -> dict[str, object]:
    cubin = compiled.asm["cubin"]
    with tempfile.TemporaryDirectory(prefix="fr13_physical_slots_") as scratch:
        cubin_path = Path(scratch) / "kernel.cubin"
        cubin_path.write_bytes(cubin)
        sass = subprocess.run(
            ["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin_path)],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
        resource = subprocess.run(
            [
                "/usr/local/cuda/bin/cuobjdump",
                "--dump-resource-usage",
                str(cubin_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout
        elf = subprocess.run(
            ["/usr/local/cuda/bin/cuobjdump", "--dump-elf", str(cubin_path)],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
    resource_match = RESOURCE_RE.search(resource)
    toolkit = TOOLKIT_RE.search(elf)
    tool_name = TOOL_NAME_RE.search(elf)
    tool_version = TOOL_VERSION_RE.search(elf)
    if resource_match is None or None in (toolkit, tool_name, tool_version):
        raise RuntimeError("cannot parse SM121a cubin metadata")
    registers, stack_bytes, shared_bytes, local_bytes = map(
        int, resource_match.groups()
    )
    counts = operation_counts(sass)
    metadata = (
        compiled.metadata._asdict()
        if hasattr(compiled.metadata, "_asdict")
        else vars(compiled.metadata)
    )
    return {
        "compile_hash": metadata["hash"],
        "source_function_sha256": sha256(function_source.encode()),
        "cubin_sha256": sha256(cubin),
        "cubin_bytes": len(cubin),
        "ptx_sha256": sha256(compiled.asm["ptx"].encode()),
        "registers": registers,
        "stack_bytes": stack_bytes,
        "local_bytes": local_bytes,
        "elf_shared_bytes": shared_bytes,
        "launch_shared_bytes": int(metadata["shared"]),
        "encoded_sass_instructions": sum(counts.values()),
        "static_noncontrol_sass_instructions": sum(
            value
            for operation, value in counts.items()
            if operation not in CONTROL_OR_PADDING
        ),
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
            "target": "sm_121a",
            "toolkit_version": toolkit.group(1),
            "tool_name": tool_name.group(1),
            "tool_version": tool_version.group(1).strip(),
        },
    }


def compile_plain(
    fn,
    *,
    signature: dict[str, str],
    constexprs: dict[str, int],
    num_warps: int,
    num_stages: int,
    function_source: str,
) -> dict[str, object]:
    backend = triton.compiler.make_backend(TARGET)
    options = backend.parse_options(
        {"num_warps": num_warps, "num_stages": num_stages}
    )
    compiled = triton.compile(
        ASTSource(fn=fn, signature=signature, constexprs=constexprs),
        target=TARGET,
        options=options.__dict__,
    )
    return compiled_result(compiled, function_source=function_source)


def commit_arguments(*, candidate: bool, batch: int) -> list[object]:
    arguments: list[object] = [
        MockTensor(torch.int64, [batch, 32, 3]),
        MockTensor(torch.int64, [batch, 32]),
    ]
    if not candidate:
        arguments.extend(
            [
                MockTensor(torch.int64, [31]),
                MockTensor(torch.int64, [32]),
            ]
        )
    self_rows = 31 if candidate else 13
    target_rows = 32 if candidate else 17
    arguments.extend(
        [
            MockTensor(torch.int64, [batch, self_rows]),
            MockTensor(torch.int64, [batch, target_rows]),
            MockTensor(torch.int64, [batch, target_rows]),
            MockTensor(torch.int64, [batch, target_rows]),
            MockTensor(torch.bool, [batch, target_rows]),
            MockTensor(torch.int64, [batch]),
            MockTensor(torch.int64, [batch, 32]),
            MockTensor(torch.int64, [batch]),
            MockTensor(torch.int64, [batch, 16]),
            MockTensor(torch.int64, [batch]),
            MockTensor(torch.int64, [batch]),
        ]
    )
    return arguments


def compile_commit(
    fn,
    *,
    candidate: bool,
    batch: int,
    function_source: str,
) -> dict[str, object]:
    backend = triton.compiler.make_backend(TARGET)
    keywords = {
        "PHYSICAL_DRAFTS": 31,
        "PHYSICAL_ROWS": 32,
        "FANOUT": 3,
        "WALK_CAP": 12,
        "OUTPUT_CAPACITY": 32,
        "PATH_CAPACITY": 16,
        "num_warps": 1,
    }
    if not candidate:
        keywords.update({"SELF_ROWS": 13, "TARGET_ROWS": 17})
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    bound, specialization, options = binder(
        *commit_arguments(candidate=candidate, batch=batch), **keywords
    )
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
        target=TARGET,
        options=compile_options.__dict__,
    )
    result = compiled_result(compiled, function_source=function_source)
    result["batch"] = batch
    result["num_warps"] = 1
    result["num_stages"] = int(compile_options.num_stages)
    return result


def comparator_arguments(batch: int) -> list[object]:
    arguments: list[object] = [
        MockTensor(torch.int32, [1]),
        MockTensor(torch.int64, [1]),
        MockTensor(torch.int64, [5]),
        MockTensor(torch.int64, [5]),
        MockTensor(torch.int64, [batch * 13]),
        MockTensor(torch.int64, [17]),
        MockTensor(torch.int64, [batch, 13]),
        MockTensor(torch.int64, [batch, 31]),
    ]
    for _ in range(3):
        arguments.extend(
            [
                MockTensor(torch.int64, [batch, 17]),
                MockTensor(torch.int64, [batch, 32]),
            ]
        )
    arguments.extend(
        [
            MockTensor(torch.bool, [batch, 17]),
            MockTensor(torch.bool, [batch, 32]),
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
    )
    return arguments


def compile_comparator(
    fn, *, batch: int, function_source: str
) -> dict[str, object]:
    backend = triton.compiler.make_backend(TARGET)
    keywords = {
        "TARGET_ROWS": 17,
        "PHYSICAL_ROWS": 32,
        "SELF_N": batch * 13,
        "TARGET_N": batch * 17,
        "OUTPUT_N": batch * 32,
        "BATCH_N": batch,
        "PATH_N": batch * 16,
        "BLOCK": 128,
        "num_warps": 4,
    }
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    bound, specialization, options = binder(
        *comparator_arguments(batch), **keywords
    )
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
        target=TARGET,
        options=compile_options.__dict__,
    )
    result = compiled_result(compiled, function_source=function_source)
    result["batch"] = batch
    result["num_warps"] = 4
    result["num_stages"] = int(compile_options.num_stages)
    return result


def producer_contract(name: str) -> dict[str, object]:
    if name == BLOCK_STATS:
        return {
            "signature": {
                "self_logits": "*fp32",
                "target_logits": "*fp32",
                "self_source_indices": "*i64",
                "target_source_indices": "*i64",
                "block_maxima": "*fp32",
                "block_sums": "*fp32",
                "invalid_out": "*i32",
                "vocab_size": "i32",
                "self_total_rows": "i32",
                "source_rows": "i32",
            },
            "constexprs": {"BLOCK_V": 4096, "MAX_BLOCKS": 64},
            "num_warps": 4,
            "num_stages": 3,
        }
    return {
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
            "invalid_out": "*i32",
            "vocab_size": "i32",
            "number_of_blocks": "i32",
            "self_total_rows": "i32",
            "source_rows": "i32",
        },
        "constexprs": {
            "SELF_ROWS": 13,
            "TARGET_ROWS": 17,
            "PHYSICAL_DRAFTS": 31,
            "PHYSICAL_ROWS": 32,
            "FANOUT": 3,
            "WALK_CAP": 12,
            "BLOCK_V": 4096,
            "MAX_BLOCKS": 64,
        },
        "num_warps": 8,
        "num_stages": 3,
    }


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
    output = args.output
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)

    sources = {
        "incumbent": {
            DEVICE_SOURCE: source_at_revision(repo, BASE_REVISION, DEVICE_SOURCE),
            PRODUCER_SOURCE: source_at_revision(
                repo, BASE_REVISION, PRODUCER_SOURCE
            ),
        },
        "candidate": {
            DEVICE_SOURCE: source_at_revision(repo, revision, DEVICE_SOURCE),
            PRODUCER_SOURCE: source_at_revision(repo, revision, PRODUCER_SOURCE),
        },
    }
    summary: dict[str, object] = {
        "schema": "fr13.fixed32.cfwd_physical_slots.sm121a.codegen.v1",
        "status": "pending",
        "base_revision": BASE_REVISION,
        "candidate_revision": revision,
        "source_paths": [DEVICE_SOURCE, PRODUCER_SOURCE],
        "source_sha256": {
            label: {
                path: sha256(source.encode())
                for path, source in source_map.items()
            }
            for label, source_map in sources.items()
        },
        "compile_contract": {
            "target": "sm_121a",
            "batches": [1, 4],
            "physical_drafts": 31,
            "physical_rows": 32,
            "self_rows": 13,
            "target_rows": 17,
            "walk_cap": 12,
            "fanout": 3,
            "vocab_size": 248_320,
            "vocab_blocks": 61,
            "producer_programs_per_request": {
                "block_stats": 1_830,
                "direct_decision": 30,
            },
            "commit_programs_per_request": 1,
            "jit_specialization": "exact_signature_or_mock_tensor_shape_stride_alignment",
        },
        "logical_work": {
            "decision_programs_per_request_before": 30,
            "decision_programs_per_request_after": 30,
            "decision_values_stored_per_request_before": 81,
            "decision_values_stored_per_request_after": 81,
            "commit_launches_per_event_before": 1,
            "commit_launches_per_event_after": 1,
            "commit_programs_per_request_before": 1,
            "commit_programs_per_request_after": 1,
            "topology_index_scalar_loads_per_request_before": 24,
            "topology_index_scalar_loads_per_request_after": 0,
            "decision_workspace_bytes_per_request_before": 529,
            "decision_workspace_bytes_per_request_after": 1_048,
        },
        "producer": {},
        "commit": {},
        "diagnostic_comparator": {},
        "claim_scope": (
            "static_sm121a_codegen_and_exact_work_only_no_runtime_speed_claim"
        ),
    }

    for label, revision_label in (
        ("incumbent", BASE_REVISION),
        ("candidate", revision),
    ):
        builds = {}
        producer_source = sources[label][PRODUCER_SOURCE]
        for name in (BLOCK_STATS, DIRECT_DECISION):
            fn, function_source = jit_function(
                producer_source,
                PRODUCER_SOURCE,
                revision_label,
                name,
            )
            contract = producer_contract(name)
            builds[name] = compile_plain(
                fn,
                signature=contract["signature"],
                constexprs=contract["constexprs"],
                num_warps=contract["num_warps"],
                num_stages=contract["num_stages"],
                function_source=function_source,
            )
        summary["producer"][label] = builds

    for label, revision_label, name, candidate in (
        ("incumbent", BASE_REVISION, INCUMBENT_COMMIT, False),
        ("candidate", revision, CANDIDATE_COMMIT, True),
    ):
        fn, function_source = jit_function(
            sources[label][DEVICE_SOURCE],
            DEVICE_SOURCE,
            revision_label,
            name,
        )
        summary["commit"][label] = {
            f"b{batch}": compile_commit(
                fn,
                candidate=candidate,
                batch=batch,
                function_source=function_source,
            )
            for batch in (1, 4)
        }

    comparator_fn, comparator_source = jit_function(
        sources["candidate"][DEVICE_SOURCE],
        DEVICE_SOURCE,
        revision,
        COMPARATOR,
    )
    summary["diagnostic_comparator"] = {
        f"b{batch}": compile_comparator(
            comparator_fn,
            batch=batch,
            function_source=comparator_source,
        )
        for batch in (1, 4)
    }

    candidate_decision = summary["producer"]["candidate"][DIRECT_DECISION]
    incumbent_decision = summary["producer"]["incumbent"][DIRECT_DECISION]
    resource_clean = all(
        build["stack_bytes"] == 0
        and build["local_bytes"] == 0
        and build["ldl"] == 0
        and build["stl"] == 0
        and build["calls"] == 0
        for family in (
            summary["producer"]["candidate"].values(),
            summary["commit"]["candidate"].values(),
            summary["diagnostic_comparator"].values(),
        )
        for build in family
    )
    commit_improves = all(
        summary["commit"]["candidate"][key]["registers"]
        <= summary["commit"]["incumbent"][key]["registers"]
        and summary["commit"]["candidate"][key]["ldg"]
        < summary["commit"]["incumbent"][key]["ldg"]
        and summary["commit"]["candidate"][key][
            "static_noncontrol_sass_instructions"
        ]
        < summary["commit"]["incumbent"][key][
            "static_noncontrol_sass_instructions"
        ]
        for key in ("b1", "b4")
    )
    producer_preserved = (
        candidate_decision["registers"] <= incumbent_decision["registers"]
        and candidate_decision["stack_bytes"] == 0
        and candidate_decision["local_bytes"] == 0
    )
    summary["status"] = (
        "pass"
        if resource_clean and commit_improves and producer_preserved
        else "rejected_codegen_regression"
    )
    summary["conclusion"] = {
        "resource_clean": resource_clean,
        "producer_registers_preserved": producer_preserved,
        "committer_static_improves": commit_improves,
        "runtime_speedup_claimed": False,
        "real_swe_verified_gate_required": True,
    }
    (output / "codegen_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
