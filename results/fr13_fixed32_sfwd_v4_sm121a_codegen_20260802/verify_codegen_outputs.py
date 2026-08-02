#!/usr/bin/env python3
"""Independently verify two fixed32 SFWD SM121a offline builds."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess


INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@[!]?P\d+\s+)?([A-Z][A-Z0-9.]*)\b"
)
RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
CONTROL_OR_PADDING = {"NOP", "BAR", "BRA", "EXIT"}
EXPECTED_PROFILE_COMMON = {
    "x_global_load_assignments": 32,
    "x_unique_rows_loaded": 32,
    "x_reload_count": 0,
    "activation_assignments": 32,
    "product_assignments": 128,
    "store_calls": 68,
}
EXPECTED_PROFILE_DELTA = {
    "incumbent": {
        "saved_accumulator_assignments": 0,
        "activation_window": 1,
        "peak_live_accumulator_values": 1,
    },
    "candidate": {
        "saved_accumulator_assignments": 16,
        "activation_window": 2,
        "peak_live_accumulator_values": 2,
    },
}
EXPECTED_CONFIG = {
    "b1": {"batch": 1, "block_c": 128, "num_warps": 2, "ctas": 80},
    "b4": {"batch": 4, "block_c": 256, "num_warps": 4, "ctas": 40},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_bytes(argv: list[str]) -> bytes:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def operations(sass: str) -> collections.Counter[str]:
    result: collections.Counter[str] = collections.Counter()
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            result[match.group(1).split(".", 1)[0]] += 1
    return result


def verify_build(
    root: Path,
    label: str,
    batch_key: str,
    expected: dict[str, object],
) -> dict[str, bytes]:
    variant = root / label / batch_key
    observed = json.loads((variant / "summary.json").read_text())
    require(observed == expected, f"{label}/{batch_key} embedded summary")
    blobs = {
        name: (variant / f"kernel.{name}").read_bytes()
        for name in ("cubin", "sass", "ptx")
    }
    resource = (variant / "resource.txt").read_bytes()
    elf = (variant / "elf.txt").read_bytes()
    require(len(blobs["cubin"]) == expected["cubin_bytes"], "cubin size")
    for name in ("cubin", "sass", "ptx"):
        require(
            sha256(blobs[name]) == expected[f"{name}_sha256"],
            f"{label}/{batch_key} {name} hash",
        )
    cubin_path = str(variant / "kernel.cubin")
    require(
        run_bytes(["/usr/local/cuda/bin/nvdisasm", "-c", cubin_path])
        == blobs["sass"],
        f"{label}/{batch_key} fresh disassembly",
    )
    require(
        run_bytes(
            ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", cubin_path]
        )
        == resource,
        f"{label}/{batch_key} fresh resource report",
    )
    require(
        run_bytes(["/usr/local/cuda/bin/cuobjdump", "--dump-elf", cubin_path])
        == elf,
        f"{label}/{batch_key} fresh ELF report",
    )
    config = EXPECTED_CONFIG[batch_key]
    require(expected["batch"] == config["batch"], f"{batch_key} batch")
    require(expected["block_c"] == config["block_c"], f"{batch_key} block C")
    require(expected["num_warps"] == config["num_warps"], f"{batch_key} warps")
    require(
        expected["ctas_per_request"] == config["ctas"],
        f"{batch_key} CTAs/request",
    )
    require(
        expected["ctas_per_launch"] == config["ctas"] * config["batch"],
        f"{batch_key} CTAs/launch",
    )
    ptx = blobs["ptx"].decode()
    require(re.search(r"(?m)^\.target sm_121a$", ptx) is not None, "SM121a target")
    threads = int(config["num_warps"]) * 32
    require(
        re.search(rf"(?m)^\s*\.reqntid\s+{threads}\s*$", ptx) is not None,
        f"{batch_key} thread count",
    )
    elf_text = elf.decode()
    for fragment in ("ptxas-blackwell", "-arch sm_121a"):
        require(fragment in elf_text, f"{label}/{batch_key} producer {fragment}")
    resource_match = RESOURCE_RE.search(resource.decode())
    require(resource_match is not None, f"{label}/{batch_key} resource parse")
    resource_values = tuple(map(int, resource_match.groups()))
    require(
        resource_values
        == (
            expected["registers"],
            expected["stack_bytes"],
            expected["elf_shared_bytes"],
            expected["local_bytes"],
        ),
        f"{label}/{batch_key} resource values",
    )
    counts = operations(blobs["sass"].decode())
    recounted = {
        "encoded_sass_instructions": sum(counts.values()),
        "static_sass_instructions": sum(
            value
            for operation, value in counts.items()
            if operation not in CONTROL_OR_PADDING
        ),
        "ldg": counts["LDG"],
        "stg": counts["STG"],
        "lds": counts["LDS"],
        "sts": counts["STS"],
        "ldl": counts["LDL"],
        "stl": counts["STL"],
        "calls": sum(
            value for operation, value in counts.items() if operation.startswith("CALL")
        ),
    }
    for name, value in recounted.items():
        require(value == expected[name], f"{label}/{batch_key} {name}")
    require(expected["registers"] <= 255, f"{label}/{batch_key} register limit")
    require(
        expected["stack_bytes"]
        == expected["local_bytes"]
        == expected["ldl"]
        == expected["stl"]
        == expected["calls"]
        == 0,
        f"{label}/{batch_key} spill/stack/local/call gate",
    )
    return {**blobs, "resource": resource, "elf": elf}


def verify_tree(root: Path) -> dict[str, object]:
    summary = json.loads((root / "summary.json").read_text())
    require(
        summary["schema"] == "fr13.fixed32.sfwd.v4.sm121a.offline_codegen.v1",
        "summary schema",
    )
    contract = summary["compile_contract"]
    require(contract["target"] == "sm_121a", "compile target")
    require(contract["physical_rows_per_request"] == 32, "physical rows")
    require(contract["channels"] == 10240, "channels")
    require(contract["conv_width"] == 4, "conv width")
    require(contract["conv_state_len"] == 34, "state length")
    require(contract["x_stride_row"] == 16384, "padded x stride")
    require(contract["conv_stride_row"] == 348160, "conv-state row stride")
    require(contract["has_bias"] is False, "bias specialization")
    verified_blobs: dict[tuple[str, str], dict[str, bytes]] = {}
    for label in ("incumbent", "candidate"):
        variant = summary["variants"][label]
        profile = variant["source_profile"]
        for name, value in EXPECTED_PROFILE_COMMON.items():
            require(profile[name] == value, f"{label} source {name}")
        for name, value in EXPECTED_PROFILE_DELTA[label].items():
            require(profile[name] == value, f"{label} source {name}")
        for batch_key in ("b1", "b4"):
            verified_blobs[(label, batch_key)] = verify_build(
                root, label, batch_key, variant["builds"][batch_key]
            )
    incumbent_profile = summary["variants"]["incumbent"]["source_profile"]
    candidate_profile = summary["variants"]["candidate"]["source_profile"]
    require(
        candidate_profile["x_load_order"] == incumbent_profile["x_load_order"],
        "candidate changed load-once order",
    )
    require(
        candidate_profile["activation_order"]
        == incumbent_profile["activation_order"],
        "candidate changed activation order",
    )
    comparisons = {}
    absolute_codegen = {}
    for batch_key in ("b1", "b4"):
        incumbent = summary["variants"]["incumbent"]["builds"][batch_key]
        candidate = summary["variants"]["candidate"]["builds"][batch_key]
        comparisons[batch_key] = {
            name: int(candidate[name]) - int(incumbent[name])
            for name in (
                "registers",
                "stack_bytes",
                "local_bytes",
                "static_sass_instructions",
                "encoded_sass_instructions",
                "ldg",
                "stg",
                "ldl",
                "stl",
                "calls",
                "cubin_bytes",
            )
        }
        absolute_codegen[batch_key] = {}
        for label, build in (("incumbent", incumbent), ("candidate", candidate)):
            sass = (root / label / batch_key / "kernel.sass").read_text()
            counts = operations(sass)
            absolute_codegen[batch_key][label] = {
                name: build[name]
                for name in (
                    "block_c",
                    "num_warps",
                    "ctas_per_request",
                    "ctas_per_launch",
                    "registers",
                    "stack_bytes",
                    "local_bytes",
                    "static_sass_instructions",
                    "encoded_sass_instructions",
                    "ldg",
                    "stg",
                    "ldl",
                    "stl",
                    "calls",
                    "cubin_bytes",
                    "cubin_sha256",
                    "sass_sha256",
                    "ptx_sha256",
                    "source_function_sha256",
                    "backend_producer",
                )
            }
            absolute_codegen[batch_key][label]["key_sass_counts"] = {
                name: counts[name]
                for name in (
                    "F2FP",
                    "FADD",
                    "FMUL",
                    "FSETP",
                    "HFMA2",
                    "HMUL2",
                    "MOV",
                    "MUFU",
                    "PRMT",
                )
            }
    source_profiles = {}
    for label in ("incumbent", "candidate"):
        profile = summary["variants"][label]["source_profile"]
        source_profiles[label] = {
            name: profile[name]
            for name in (
                "x_global_load_assignments",
                "x_unique_rows_loaded",
                "x_reload_count",
                "activation_assignments",
                "saved_accumulator_assignments",
                "activation_window",
                "peak_live_accumulator_values",
                "product_assignments",
                "store_calls",
            )
        }
    return {
        "schema": "fr13.fixed32.sfwd.v4.sm121a.verification.v1",
        "verified": True,
        "offline_only": True,
        "timing_claim": False,
        "candidate_static_gate_pass": True,
        "revisions": summary["revisions"],
        "compile_contract": summary["compile_contract"],
        "source_profiles": source_profiles,
        "absolute_codegen": absolute_codegen,
        "source_profile_delta": {
            "x_loads": 0,
            "x_reloads": 0,
            "activations": 0,
            "saved_accumulators": 16,
            "peak_live_accumulators": 1,
        },
        "codegen_deltas": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--rebuild", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    primary = verify_tree(args.primary)
    rebuild = verify_tree(args.rebuild)
    require(primary == rebuild, "primary/rebuild verification report")
    for label in ("incumbent", "candidate"):
        for batch_key in ("b1", "b4"):
            for name in ("cubin", "sass", "ptx", "resource", "elf"):
                filename = (
                    f"kernel.{name}"
                    if name in {"cubin", "sass", "ptx"}
                    else f"{name}.txt"
                )
                left = args.primary / label / batch_key / filename
                right = args.rebuild / label / batch_key / filename
                require(
                    left.read_bytes() == right.read_bytes(),
                    f"rebuild {label}/{batch_key}/{name}",
                )
    rendered = json.dumps(primary, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
