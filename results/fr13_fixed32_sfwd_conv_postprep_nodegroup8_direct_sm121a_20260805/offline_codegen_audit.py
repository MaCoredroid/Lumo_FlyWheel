#!/usr/bin/env python3
"""Compare incumbent and direct-nodegroup8 SFWD codegen without a GPU."""

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
import sys
import tempfile


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


SOURCE_COMMIT = "b73d78f681d0cea8487b97a75eaf2ac44d3bc8ec"
SOURCE_PATH = (
    "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion_kernel.py"
)
KERNELS = {
    "incumbent": "_fr13_fixed32_sfwd_conv_postprep_fusion_kernel",
    "nodegroup8_direct": (
        "_fr13_fixed32_sfwd_conv_postprep_nodegroup8_direct_kernel"
    ),
}
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
    "BLOCK_C": 256,
    "GATE_BLOCK": 64,
    "SOFTPLUS_THRESHOLD": 20.0,
}
PROFILES = {
    "b1_standalone": {"B": 1, "EMBED_GATE_CTA": False},
    "b1_embedded": {"B": 1, "EMBED_GATE_CTA": True},
    "b4_standalone": {"B": 4, "EMBED_GATE_CTA": False},
    "b4_embedded": {"B": 4, "EMBED_GATE_CTA": True},
}
MIN_MEM_AVAILABLE_KIB = 20 * 1024 * 1024
INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@[!]?P\d+\s+)?([A-Z][A-Z0-9.]*)\b"
)
RESOURCE_RE = re.compile(
    rb"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
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


def mem_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def guard_memory(stage: str, samples: list[dict[str, object]]) -> None:
    available = mem_available_kib()
    samples.append({"stage": stage, "mem_available_kib": available})
    if available < MIN_MEM_AVAILABLE_KIB:
        raise RuntimeError(
            f"memory guard failed at {stage}: {available} < "
            f"{MIN_MEM_AVAILABLE_KIB} KiB"
        )


def source_at_commit(repo: Path) -> str:
    observed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{SOURCE_COMMIT}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != SOURCE_COMMIT:
        raise RuntimeError(f"source commit drifted: {observed}")
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{SOURCE_COMMIT}:{SOURCE_PATH}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def jit_functions(source: str) -> dict[str, object]:
    canonical_path = f"{SOURCE_PATH}@{SOURCE_COMMIT}"
    linecache.cache[canonical_path] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        canonical_path,
    )
    namespace = {
        "__name__": "fr13_sfwd_nodegroup8_direct_codegen",
        "__file__": canonical_path,
    }
    exec(compile(source, canonical_path, "exec"), namespace)
    return {label: namespace[name] for label, name in KERNELS.items()}


def operations(sass: str) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            counts[match.group(1).split(".", 1)[0]] += 1
    return counts


def compile_one(
    *,
    function: object,
    label: str,
    profile: str,
    constants: dict[str, object],
) -> dict[str, object]:
    target = GPUTarget("cuda", 121, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": 4, "num_stages": 3})
    compiled = triton.compile(
        ASTSource(fn=function, signature=SIGNATURE, constexprs=constants),
        target=target,
        options=options.__dict__,
    )
    metadata = (
        compiled.metadata._asdict()
        if hasattr(compiled.metadata, "_asdict")
        else vars(compiled.metadata)
    )
    cubin = compiled.asm["cubin"]
    with tempfile.TemporaryDirectory(prefix="fr13-sfwd-codegen-") as temporary:
        cubin_path = Path(temporary) / "kernel.cubin"
        cubin_path.write_bytes(cubin)
        sass = run_bytes(["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin_path)])
        resource = run_bytes(
            [
                "/usr/local/cuda/bin/cuobjdump",
                "--dump-resource-usage",
                str(cubin_path),
            ]
        )
    resource_match = RESOURCE_RE.search(resource)
    if resource_match is None:
        raise RuntimeError(f"unable to parse resources for {label}/{profile}")
    registers, stack_bytes, shared_bytes, local_bytes = map(
        int, resource_match.groups()
    )
    counts = operations(sass.decode("ascii"))
    return {
        "kernel": label,
        "profile": profile,
        "batch": int(constants["B"]),
        "embedded_gate_cta": bool(constants["EMBED_GATE_CTA"]),
        "block_c": 256,
        "num_warps": 4,
        "num_stages": 3,
        "compile_hash": metadata["hash"],
        "cubin_sha256": sha256(cubin),
        "cubin_bytes": len(cubin),
        "ptx_sha256": sha256(compiled.asm["ptx"].encode()),
        "sass_sha256": sha256(sass),
        "registers_per_thread": registers,
        "stack_bytes": stack_bytes,
        "local_bytes": local_bytes,
        "elf_shared_bytes": shared_bytes,
        "launch_shared_bytes": int(metadata["shared"]),
        "encoded_sass_instructions": sum(counts.values()),
        "static_sass_instructions": sum(
            count
            for operation, count in counts.items()
            if operation not in CONTROL_OR_PADDING
        ),
        "ldg": counts["LDG"],
        "stg": counts["STG"],
        "ldl": counts["LDL"],
        "stl": counts["STL"],
        "calls": counts["CALL"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    memory_samples: list[dict[str, object]] = []
    guard_memory("before_source_read", memory_samples)
    source = source_at_commit(repo)
    functions = jit_functions(source)
    builds: dict[str, dict[str, dict[str, object]]] = {
        label: {} for label in KERNELS
    }
    for label, function in functions.items():
        for profile, profile_constants in PROFILES.items():
            guard_memory(f"before_{label}_{profile}", memory_samples)
            constants = {**BASE_CONSTANTS, **profile_constants}
            builds[label][profile] = compile_one(
                function=function,
                label=label,
                profile=profile,
                constants=constants,
            )
            guard_memory(f"after_{label}_{profile}", memory_samples)

    comparison = {}
    fields = (
        "registers_per_thread",
        "stack_bytes",
        "local_bytes",
        "elf_shared_bytes",
        "launch_shared_bytes",
        "cubin_bytes",
        "encoded_sass_instructions",
        "static_sass_instructions",
        "ldg",
        "stg",
        "ldl",
        "stl",
        "calls",
    )
    for profile in PROFILES:
        incumbent = builds["incumbent"][profile]
        direct = builds["nodegroup8_direct"][profile]
        comparison[profile] = {
            field: {
                "incumbent": incumbent[field],
                "nodegroup8_direct": direct[field],
                "delta": int(direct[field]) - int(incumbent[field]),
            }
            for field in fields
        }
    payload = {
        "schema": "fr13.fixed32.sfwd.nodegroup8_direct.sm121a_codegen.v1",
        "source_commit": SOURCE_COMMIT,
        "source_path": SOURCE_PATH,
        "source_sha256": sha256(source.encode()),
        "target": "sm_121a",
        "offline_only": True,
        "gpu_visible": False,
        "timing_claim": False,
        "acceptance_claim": False,
        "block_c": 256,
        "num_warps": 4,
        "num_stages": 3,
        "toolchain": {
            "python": sys.version.split()[0],
            "triton": triton.__version__,
            "nvdisasm": run_bytes(
                ["/usr/local/cuda/bin/nvdisasm", "--version"]
            ).decode("ascii").splitlines()[-1],
            "cuobjdump": run_bytes(
                ["/usr/local/cuda/bin/cuobjdump", "--version"]
            ).decode("ascii").splitlines()[-1],
        },
        "memory_guard": {
            "minimum_mem_available_kib": MIN_MEM_AVAILABLE_KIB,
            "samples": memory_samples,
        },
        "shape": {
            "incumbent_channel_programs_per_request": 40,
            "nodegroup8_channel_programs_per_request": 160,
            "node_groups": 4,
            "nodes_per_group": 8,
            "incumbent_x_loads_per_channel": 32,
            "nodegroup8_x_loads_per_channel_across_groups": 54,
        },
        "prior_rowgroup8_real_b1_regression": {
            "candidate": "fixed32_sfwd_state_fusion_rowgroup8_v3",
            "artifact": (
                "results/fr13_fixed32_sfwd_rowgroup8_k64_root_b1_"
                "timing_rejection_20260802"
            ),
            "stock_ms_per_step": 242.286839679,
            "candidate_ms_per_step": 244.569046597,
            "delta_ms_per_step": 2.282206918,
            "delta_percent": 0.941944235,
            "equivalent_design": False,
            "distinction": (
                "measured v3 used an 8x256 row tensor, dynamic source_flat "
                "descriptors, masked prior/x selection, and row-dependent "
                "addresses; this candidate uses fixed scalar-unrolled node "
                "branches and has no source descriptor, gather, shared tile, "
                "reduction, or barrier"
            ),
        },
        "builds": builds,
        "comparison": comparison,
    }
    (output / "codegen_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    columns = (
        "kernel",
        "profile",
        "batch",
        "embedded_gate_cta",
        "registers_per_thread",
        "stack_bytes",
        "local_bytes",
        "elf_shared_bytes",
        "launch_shared_bytes",
        "cubin_bytes",
        "static_sass_instructions",
        "ldg",
        "stg",
        "calls",
    )
    lines = ["\t".join(columns)]
    for label in KERNELS:
        for profile in PROFILES:
            build = builds[label][profile]
            lines.append("\t".join(str(build[column]) for column in columns))
    (output / "codegen_summary.tsv").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
