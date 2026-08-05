#!/usr/bin/env python3
"""Compile fixed32 SFWD gate-pack revisions for SM121a without a GPU."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import linecache
import os
from pathlib import Path
import re
import subprocess


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


SOURCE_PATH = (
    "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion_kernel.py"
)
KERNEL_NAME = "_fr13_fixed32_sfwd_conv_postprep_fusion_kernel"
SIGNATURE = {
    "x": "*bf16",
    "conv_state": "*bf16",
    "spec_state_indices": "*i32",
    "sticky_guard_ok": "*i32",
    "conv_weights": "*bf16",
    "bias": "*bf16",
    "a": "*bf16",
    "b": "*bf16",
    "A_log": "*fp32",
    "dt_bias": "*bf16",
    "query": "*bf16",
    "key": "*bf16",
    "value_spec": "*bf16",
    "value_tree": "*bf16",
    "g": "*fp32",
    "beta": "*fp32",
    "source_stage": "*bf16",
    "conv_tap": "*bf16",
}
BASE_CONSTANTS = {
    "CONV_STRIDE_ROW": 2097152,
    "BANK_ROWS": 257,
    "N": 32,
    "C": 10240,
    "WIDTH": 4,
    "STATE_LEN": 34,
    "SOURCE_ROWS": 36,
    "H": 16,
    "HV": 48,
    "K": 128,
    "V": 128,
    "HAS_BIAS": False,
    "STORE_CONV_TAP": False,
    "CAPTURE_GUARD": True,
    "X_STRIDE_ROW": 16384,
    "GATE_BLOCK": 64,
    "SOFTPLUS_THRESHOLD": 20.0,
}
DEPLOYMENT_CONFIGS = {
    "b1": {"batch": 1, "block_c": 128, "num_warps": 2},
    "b4": {"batch": 4, "block_c": 256, "num_warps": 4},
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
CONTROL_OR_PADDING = {"NOP", "BAR", "BRA", "EXIT"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_bytes(argv: list[str]) -> bytes:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def resolve_revision(repo: Path, revision: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{revision}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_at_revision(repo: Path, revision: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{SOURCE_PATH}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def jit_function(source: str, revision: str):
    canonical_path = f"{SOURCE_PATH}@{revision}"
    linecache.cache[canonical_path] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        canonical_path,
    )
    namespace = {
        "__name__": f"fr13_sfwd_gatepack_codegen_{revision}",
        "__file__": canonical_path,
    }
    exec(compile(source, canonical_path, "exec"), namespace)
    return namespace[KERNEL_NAME]


def operations(sass: str) -> collections.Counter[str]:
    result: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            result[match.group(1).split(".", 1)[0]] += 1
    return result


def compile_one(
    *,
    source: str,
    revision: str,
    output: Path,
    profile: str,
) -> dict[str, object]:
    config = DEPLOYMENT_CONFIGS[profile]
    output.mkdir(parents=True, exist_ok=True)
    constants = dict(BASE_CONSTANTS)
    constants.update(B=config["batch"], BLOCK_C=config["block_c"])
    target = GPUTarget("cuda", 121, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options(
        {"num_warps": config["num_warps"], "num_stages": 3}
    )
    compiled = triton.compile(
        ASTSource(
            fn=jit_function(source, revision),
            signature=SIGNATURE,
            constexprs=constants,
        ),
        target=target,
        options=options.__dict__,
    )
    for name, value in compiled.asm.items():
        path = output / f"kernel.{name}"
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value)
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
    counts = operations(sass.decode())
    metadata = (
        compiled.metadata._asdict()
        if hasattr(compiled.metadata, "_asdict")
        else vars(compiled.metadata)
    )
    cubin = compiled.asm["cubin"]
    result = {
        "revision": revision,
        "profile": profile,
        "batch": config["batch"],
        "block_c": config["block_c"],
        "num_warps": config["num_warps"],
        "num_stages": 3,
        "compile_hash": metadata["hash"],
        "source_sha256": sha256(source.encode()),
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
        "static_sass_instructions": sum(
            value
            for operation, value in counts.items()
            if operation not in CONTROL_OR_PADDING
        ),
        "ldg": counts["LDG"],
        "stg": counts["STG"],
        "ldl": counts["LDL"],
        "stl": counts["STL"],
        "calls": counts["CALL"],
        "backend_producer": {
            "toolkit_version": toolkit.group(1),
            "tool_name": tool_name.group(1),
            "tool_version": tool_version.group(1),
            "target": "sm_121a",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def work_model(profile: str) -> dict[str, object]:
    config = DEPLOYMENT_CONFIGS[profile]
    batch = config["batch"]
    block_c = config["block_c"]
    channel_programs = 10240 // block_c
    gate_rows_per_program = block_c // 64
    incumbent_gate_programs = 32
    candidate_gate_programs = 32 // gate_rows_per_program
    incumbent_per_request = channel_programs + incumbent_gate_programs
    candidate_per_request = channel_programs + candidate_gate_programs
    row_varying_reads = 2 * 32 * 48 * 2
    row_varying_writes = 2 * 32 * 48 * 4
    invariant_bytes_per_program = 48 * (4 + 2)
    incumbent_bytes = (
        row_varying_reads
        + row_varying_writes
        + incumbent_gate_programs * invariant_bytes_per_program
    )
    candidate_bytes = (
        row_varying_reads
        + row_varying_writes
        + candidate_gate_programs * invariant_bytes_per_program
    )
    return {
        "batch": batch,
        "channel_programs_per_request": channel_programs,
        "gate_rows_per_candidate_program": gate_rows_per_program,
        "gate_programs_per_request": {
            "incumbent": incumbent_gate_programs,
            "candidate": candidate_gate_programs,
            "delta": candidate_gate_programs - incumbent_gate_programs,
        },
        "total_programs_per_request": {
            "incumbent": incumbent_per_request,
            "candidate": candidate_per_request,
            "delta": candidate_per_request - incumbent_per_request,
        },
        "total_programs_all_48_layers": {
            "incumbent": batch * incumbent_per_request * 48,
            "candidate": batch * candidate_per_request * 48,
            "delta": batch * (candidate_per_request - incumbent_per_request) * 48,
        },
        "requested_gate_bytes_per_request_layer": {
            "incumbent": incumbent_bytes,
            "candidate": candidate_bytes,
            "delta": candidate_bytes - incumbent_bytes,
        },
        "requested_gate_bytes_whole_batch_all_48_layers": {
            "incumbent": batch * incumbent_bytes * 48,
            "candidate": batch * candidate_bytes * 48,
            "delta": batch * (candidate_bytes - incumbent_bytes) * 48,
        },
        "kernel_launches_all_48_layers": {
            "incumbent": 48,
            "candidate": 48,
            "delta": 0,
        },
        "traffic_classification": (
            "exact_fixed32_source_address_requested_bytes_not_measured_dram_bytes"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    revisions = {
        "incumbent": resolve_revision(repo, args.incumbent),
        "candidate": resolve_revision(repo, args.candidate),
    }
    sources = {
        label: source_at_revision(repo, revision)
        for label, revision in revisions.items()
    }
    builds: dict[str, dict[str, dict[str, object]]] = {}
    for label in ("incumbent", "candidate"):
        builds[label] = {}
        for profile in ("b1", "b4"):
            builds[label][profile] = compile_one(
                source=sources[label],
                revision=revisions[label],
                output=output / label / profile,
                profile=profile,
            )
    deltas = {}
    for profile in ("b1", "b4"):
        incumbent = builds["incumbent"][profile]
        candidate = builds["candidate"][profile]
        deltas[profile] = {
            key: int(candidate[key]) - int(incumbent[key])
            for key in (
                "registers",
                "stack_bytes",
                "local_bytes",
                "elf_shared_bytes",
                "encoded_sass_instructions",
                "static_sass_instructions",
                "ldg",
                "stg",
                "ldl",
                "stl",
                "calls",
                "cubin_bytes",
            )
        }
    work = {profile: work_model(profile) for profile in ("b1", "b4")}
    gate_pass = all(
        builds["candidate"][profile][key] == 0
        for profile in ("b1", "b4")
        for key in (
            "stack_bytes",
            "local_bytes",
            "elf_shared_bytes",
            "launch_shared_bytes",
            "ldl",
            "stl",
            "calls",
        )
    ) and all(
        work[profile]["total_programs_all_48_layers"]["delta"] < 0
        and work[profile]["requested_gate_bytes_whole_batch_all_48_layers"][
            "delta"
        ]
        < 0
        for profile in ("b1", "b4")
    )
    summary = {
        "schema": (
            "fr13.fixed32.sfwd_conv_postprep.gatepack.sm121a.offline_codegen.v1"
        ),
        "offline_only": True,
        "gpu_api_used": False,
        "timing_claim": False,
        "runtime_correctness_claim": False,
        "revisions": revisions,
        "source_path": SOURCE_PATH,
        "source_sha256": {
            label: sha256(source.encode()) for label, source in sources.items()
        },
        "compile_contract": {
            "target": "sm_121a",
            "physical_rows_per_request": 32,
            "channels": 10240,
            "conv_state_len": 34,
            "source_rows": 36,
            "capture_guard": True,
            "bank_rows_fixture": 257,
            "store_conv_tap": False,
            "dt_bias_dtype": "bf16",
            "deployment_configs": DEPLOYMENT_CONFIGS,
        },
        "work_model": work,
        "builds": builds,
        "codegen_deltas": deltas,
        "static_gate_pass": gate_pass,
        "required_next_gate": (
            "real SWE-Verified B1 and B4 byte equality, then full-step timing"
        ),
    }
    (output / "codegen_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
