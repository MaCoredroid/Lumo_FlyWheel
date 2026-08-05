#!/usr/bin/env python3
"""Compile exact physical32 K64 GQA3 BV8/BV16 SM121a resources."""

from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.jit import MockTensor, create_function_from_signature


SOURCE = Path("src/lumo_flywheel_serving/fr13_gdn_gqa_group3.py")
KERNEL = "_fr13_fixed32_gdn_gqa_group3_single_launch_kernel"
VARIANTS = {
    "baseline_bv8_base": (
        "baseline",
        False,
        True,
        True,
        False,
    ),
    "candidate_bv16_base": (
        "candidate",
        False,
        True,
        True,
        True,
    ),
    "baseline_bv8_committer_stack": (
        "baseline",
        True,
        True,
        True,
        False,
    ),
    "candidate_bv16_committer_stack": (
        "candidate",
        True,
        True,
        True,
        True,
    ),
}
RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+"
    r"(?:@[!]?(?:U)?P\d+\s+)?([A-Z][A-Z0-9.]*)\b"
)
TOOLKIT_RE = re.compile(r"Tool Kit Version:\s*([^\s]+)")
TOOL_NAME_RE = re.compile(r"Tool Name:\s*(\S+)")
TOOL_VERSION_RE = re.compile(r"Tool Version:\s*(.+)")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
CONTROL_OR_PADDING = {"NOP", "BAR", "BRA", "EXIT"}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run_bytes(argv: list[str]) -> bytes:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def git_source(repo: Path, revision: str, path: Path) -> bytes:
    return run_bytes(
        ["git", "-C", str(repo), "show", f"{revision}:{path.as_posix()}"]
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Triton source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def function_sha256(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    lines = source.splitlines(keepends=True)
    return sha256("".join(lines[start - 1 : node.end_lineno]).encode())


def operation_counts(
    sass: str,
) -> tuple[collections.Counter[str], collections.Counter[str]]:
    base: collections.Counter[str] = collections.Counter()
    full: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            operation = match.group(1)
            full[operation] += 1
            base[operation.split(".", 1)[0]] += 1
    return base, full


def production_args(
    batch: int, *, committer_stack: bool, static_schedule: bool
) -> list[MockTensor]:
    rows = batch * 32
    result = [
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
        MockTensor(torch.int32, [1]),
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
        MockTensor(torch.int32, [2]),
        MockTensor(torch.int32, [32, 32]),
        MockTensor(torch.int32, [32, 32]),
    ]
    if committer_stack:
        result[-2] = MockTensor(torch.float32, [rows, 16])
        result[-1] = MockTensor(torch.float32, [rows, 48, 2])
    if static_schedule:
        result = result[:13] + result[18:]
    return result


def production_kwargs(
    batch: int,
    *,
    trust_node_domain: bool,
    committer_stack: bool,
    static_schedule: bool,
    wide_value_tile: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "N_ACTUAL": 32,
        "NUM_KH": 16,
        "NUM_VH": 48,
        "DIM_K": 128,
        "DIM_V": 128,
        "BLOCK_V": 16 if wide_value_tile else 8,
        "OUTPUT_SCALE": 128**-0.5,
        "H0_IS_BANK": True,
        "H0_INDEX_ROW": 0,
        "H0_INDEX_BATCH_STRIDE": 32,
        "H0_BATCH_INDEX": 0,
        "H0_ACCEPTED_BATCH_STRIDE": 1,
        "H0_BANK_STRIDE": 48 * 128 * 128,
        "H0_USE_ACCEPTED_COLUMN": False,
        "USE_QK_L2NORM_IN_KERNEL": True,
        "RAW_GATING": True,
        "COUNT_INVOCATION": True,
        "SCAN_ALIGN": False,
        "ROOT_STEPS": 5,
        "MAX_PATH_LEN": 7,
        "MAX_GROUP_PATHS": 3,
        "PRESCALED_PATH_BASE": False,
        "RING_EXPORT": True,
        "K_NORM_EXPORT": False,
        "GATE_EXPORT": False,
        "DECAY_EXPORT": False,
        "FLAGS_EXPORT": True,
        "FLAGS_ROWS": batch,
        "num_warps": 8,
    }
    result["HEAD_GROUP"] = 3
    if trust_node_domain:
        result["TRUST_FIXED32_NODE_DOMAIN"] = True
    if static_schedule:
        for name in (
            "ROOT_STEPS",
            "MAX_PATH_LEN",
            "MAX_GROUP_PATHS",
            "PRESCALED_PATH_BASE",
        ):
            del result[name]
    if committer_stack:
        result.update(
            {
                "K_NORM_EXPORT": True,
                "GATE_EXPORT": True,
                "DECAY_EXPORT": True,
                "maxnreg": 128,
            }
        )
    return result


def exact_specialization(
    fn,
    backend,
    batch: int,
    *,
    trust_node_domain: bool,
    committer_stack: bool,
    static_schedule: bool,
    wide_value_tile: bool,
):
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    bound, specialization, options = binder(
        *production_args(
            batch,
            committer_stack=committer_stack,
            static_schedule=static_schedule,
        ),
        **production_kwargs(
            batch,
            trust_node_domain=trust_node_domain,
            committer_stack=committer_stack,
            static_schedule=static_schedule,
            wide_value_tile=wide_value_tile,
        ),
    )
    compile_options, signature, constexprs, attrs = fn._pack_args(
        backend,
        options,
        bound,
        specialization,
        options,
    )
    return compile_options, signature, constexprs, attrs


def compile_one(
    *,
    fn,
    source: str,
    source_path: Path,
    revision: str,
    output: Path,
    label: str,
    batch: int,
    committer_stack: bool,
    trust_node_domain: bool,
    static_schedule: bool,
    wide_value_tile: bool,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    target = GPUTarget("cuda", 121, 32)
    backend = triton.compiler.make_backend(target)
    options, signature, constexprs, attrs = exact_specialization(
        fn,
        backend,
        batch,
        trust_node_domain=trust_node_domain,
        committer_stack=committer_stack,
        static_schedule=static_schedule,
        wide_value_tile=wide_value_tile,
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
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value, encoding="utf-8")
    cubin_path = output / "kernel.cubin"
    sass = run_bytes(["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin_path)])
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
    base, full = operation_counts(sass_text)
    metadata = (
        compiled.metadata._asdict()
        if hasattr(compiled.metadata, "_asdict")
        else vars(compiled.metadata)
    )
    grid = [16, 8 if wide_value_tile else 16, batch]
    programs = grid[0] * grid[1] * grid[2]
    dependencies = (
        "_fr13_fixed32_gdn_gqa_group3_value_head_node",
        "_fr13_fixed32_gdn_gqa_group3_node",
        KERNEL,
    )
    result = {
        "label": label,
        "revision": revision,
        "source_path": source_path.as_posix(),
        "source_sha256": sha256(source.encode()),
        "function_source_sha256": {
            name: function_sha256(source, name) for name in dependencies
        },
        "kernel": fn.__name__,
        "batch": batch,
        "committer_stack": committer_stack,
        "trust_fixed32_node_domain": trust_node_domain,
        "static_physical32_schedule": static_schedule,
        "block_v": 16 if wide_value_tile else 8,
        "compile_hash": metadata["hash"],
        "cubin_sha256": sha256(compiled.asm["cubin"]),
        "cubin_bytes": len(compiled.asm["cubin"]),
        "ptx_sha256": sha256(compiled.asm["ptx"].encode()),
        "ptx_bytes": len(compiled.asm["ptx"].encode()),
        "sass_sha256": sha256(sass),
        "registers_per_thread": registers,
        "stack_bytes_per_thread": stack_bytes,
        "local_bytes_per_thread": local_bytes,
        "elf_shared_bytes_per_cta": shared_bytes,
        "launch_shared_bytes_per_cta": int(metadata["shared"]),
        "resolved_num_stages": int(options.num_stages),
        "resolved_maxnreg": options.maxnreg,
        "num_warps": int(metadata["num_warps"]),
        "threads_per_cta": int(metadata["num_warps"]) * 32,
        "grid": grid,
        "programs_per_layer_event": programs,
        "programs_per_48_layer_event": programs * 48,
        "register_values_per_cta": registers * int(metadata["num_warps"]) * 32,
        "register_bytes_per_cta": (
            registers * int(metadata["num_warps"]) * 32 * 4
        ),
        "encoded_sass_instructions": sum(base.values()),
        "static_sass_instructions": sum(
            count
            for operation, count in base.items()
            if operation not in CONTROL_OR_PADDING
        ),
        "ldg": base["LDG"],
        "stg": base["STG"],
        "ldl": base["LDL"],
        "stl": base["STL"],
        "calls": sum(
            count
            for operation, count in base.items()
            if operation.startswith("CALL")
        ),
        "base_operations": dict(sorted(base.items())),
        "full_operations": dict(sorted(full.items())),
        "backend_producer": {
            "toolkit_version": toolkit.group(1),
            "tool_name": tool_name.group(1),
            "tool_version": tool_version.group(1).strip(),
            "target": "sm_121a",
        },
        "gpu_execution": False,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revisions = {
        "baseline": args.baseline_revision,
        "candidate": args.candidate_revision,
    }
    if any(HEX40_RE.fullmatch(value) is None for value in revisions.values()):
        raise RuntimeError("revisions must be full lowercase Git object IDs")
    repo = args.repo.resolve()
    modules = {}
    sources = {}
    current = (repo / SOURCE).read_bytes()
    candidate_committed = git_source(repo, args.candidate_revision, SOURCE)
    if current != candidate_committed:
        raise RuntimeError(f"current {SOURCE} differs from candidate revision")
    source_stage = Path(tempfile.gettempdir()) / (
        "fr13_gqa3_bv16_sources_"
        f"{args.baseline_revision[:12]}_{args.candidate_revision[:12]}"
    )
    source_stage.mkdir(parents=True, exist_ok=True)
    with contextlib.nullcontext(source_stage) as temp_root:
        for label, (
            revision_key,
            _committer_stack,
            _trust,
            _static_schedule,
            _wide_value_tile,
        ) in VARIANTS.items():
            revision = revisions[revision_key]
            raw = git_source(repo, revision, SOURCE)
            source_path = temp_root / f"{label}.py"
            source_path.write_bytes(raw)
            os.utime(source_path, ns=(0, 0))
            source = raw.decode("utf-8")
            module = load_module(
                source_path,
                f"fr13_gqa3_bv16_{label}_{revision}",
            )
            modules[label] = getattr(module, KERNEL)
            sources[label] = source

        summary: dict[str, object] = {
            "schema": (
                "fr13.fixed32.gdn_gqa_group3_bv16_source_bound.sm121a.codegen.v1"
            ),
            "baseline_revision": args.baseline_revision,
            "candidate_revision": args.candidate_revision,
            "compile_contract": {
                "target": "sm_121a",
                "batches": [1, 4],
                "physical_rows_per_request": 32,
                "key_heads": 16,
                "value_heads": 48,
                "value_heads_per_key_head": 3,
                "dim_k": 128,
                "dim_v": 128,
                "block_v_variants": [8, 16],
                "num_warps": 8,
                "num_stages_request": "unset",
                "resolved_num_stages": 3,
                "base_maxnreg": None,
                "committer_stack_maxnreg": 128,
                "root_steps": 5,
                "max_path_len": 7,
                "max_group_paths": 3,
                "prescaled_path_base": False,
                "h0_is_bank": True,
                "h0_use_accepted_column": False,
                "qk_l2norm_in_kernel": True,
                "raw_gating": True,
                "count_invocation": True,
                "scan_align": False,
                "ring_export": True,
                "k_norm_export": False,
                "gate_export": False,
                "decay_export": False,
                "flags_export": True,
                "candidate_trust_fixed32_node_domain": True,
                "trusted_node_domain": [0, 31],
                "candidate_static_physical32_schedule": True,
                "candidate_trust_fixed32_value_domain": True,
                "trusted_value_domain": [0, 127],
                "value_domain_masks_removed_per_cta": 291,
                "value_domain_masks_removed_per_48_layer_event": {
                    "b1": 3575808,
                    "b4": 14303232,
                },
                "reference_invocation_atomics_per_event": {
                    "b1": 1,
                    "b4": 4,
                },
                "candidate_invocation_atomics_per_event": {
                    "b1": 1,
                    "b4": 1,
                },
                "invocation_atomics_removed_per_event": {
                    "b1": 0,
                    "b4": 3,
                },
                "device_descriptor_pointer_args_removed": 5,
                "device_descriptor_loads_removed_per_cta": 59,
                "device_descriptor_loads_removed_per_48_layer_event": {
                    "b1": 724992,
                    "b4": 2899968,
                },
                "committer_stack_profile": {
                    "k_norm_export": True,
                    "gate_export": True,
                    "decay_export": True,
                },
                "draft_vocab_k": 65536,
                "draft_vocab_root": 1,
                "selector": "gqa_group3_bv16",
                "candidate_default_off": True,
                "served_reference_block_v": 8,
                "candidate_block_v": 16,
                "programs_per_48_layer_event": {
                    "baseline_bv8": {"b1": 12288, "b4": 49152},
                    "candidate_bv16": {"b1": 6144, "b4": 24576},
                    "removed": {"b1": 6144, "b4": 24576},
                },
                "repeated_qk_bytes_removed_per_48_layer_event": {
                    "b1": 100663296,
                    "b4": 402653184,
                },
                "qk_norm_reductions_removed_per_48_layer_event": {
                    "b1": 393216,
                    "b4": 1572864,
                },
                "jit_specialization": (
                    "mock_tensor_exact_shape_stride_alignment_and_pointer_dtype"
                ),
                "gpu_execution": False,
            },
            "variants": {},
        }
        for label, (
            revision_key,
            committer_stack,
            trust,
            static_schedule,
            wide_value_tile,
        ) in VARIANTS.items():
            revision = revisions[revision_key]
            builds = {}
            for batch in (1, 4):
                batch_label = f"b{batch}"
                builds[batch_label] = compile_one(
                    fn=modules[label],
                    source=sources[label],
                    source_path=SOURCE,
                    revision=revision,
                    output=args.output / label / batch_label,
                    label=label,
                    batch=batch,
                    committer_stack=committer_stack,
                    trust_node_domain=trust,
                    static_schedule=static_schedule,
                    wide_value_tile=wide_value_tile,
                )
            summary["variants"][label] = {"builds": builds}
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
