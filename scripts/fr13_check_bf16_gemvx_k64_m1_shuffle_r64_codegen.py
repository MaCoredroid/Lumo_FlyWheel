#!/usr/bin/env python3
"""Fail-closed static codegen audit for the FR13 K64 M1 R64 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ascii(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise AuditError(f"cannot read ASCII audit input {path}: {error}") from error


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def arithmetic_body(source: str) -> str:
    try:
        start = source.index("  float accumulator = 0.0f;")
        end = source.index("\n  }\n}\n", start) + len("\n  }\n}")
    except ValueError as error:
        raise AuditError("scalar arithmetic body is missing") from error
    return source[start:end]


def audit_source(r64_source: str, r32_source: str) -> dict[str, object]:
    required = (
        "constexpr int kHidden = 5120;",
        "constexpr int kVocab = 65536;",
        "constexpr int kLanes = 16;",
        "constexpr int kRowsPerCta = 64;",
        "static_assert(kLanes * kRowsPerCta == 1024);",
        "static_assert(kCtas == 1024);",
        "const dim3 block(kLanes, kRowsPerCta, 1);",
        "<<<kCtas, block, 0, at::cuda::getCurrentCUDAStream()>>>",
        "gemvx_m1_shuffle_r64_out(Tensor(a!) output, Tensor input, ",
    )
    for needle in required:
        require(needle in r64_source, f"R64 source contract is missing {needle!r}")
    for forbidden in (
        "gemvx_m1_shuffle_r32_out",
        "FR13_DRAFT_HEAD_M1_R64",
        "__syncthreads",
        "extern __shared__",
        "atomicAdd",
    ):
        require(forbidden not in r64_source, f"R64 source contains {forbidden!r}")
    require(
        arithmetic_body(r64_source) == arithmetic_body(r32_source),
        "R64 per-row arithmetic differs from the exact-order R32 parent",
    )
    require(r64_source.count("__shfl_down_sync(") == 4, "shuffle count drifted")
    require(r64_source.count("__fadd_rn(") == 4, "FP32 add count drifted")
    return {
        "grid": [1024, 1, 1],
        "block": [16, 64, 1],
        "threads_per_cta": 1024,
        "rows_per_cta": 64,
        "k_partition_lanes": 16,
        "lane_k_iterations": 320,
        "reduction_strides": [8, 4, 2, 1],
        "per_row_arithmetic_matches_r32_source": True,
    }


def audit_resource_usage(resource_usage: str) -> dict[str, int]:
    require("arch = sm_121a" in resource_usage, "resource target is not sm_121a")
    match = re.search(
        r"Function [^\n]*shuffle_r64_kernel[^\n]*:\s*\n"
        r"\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+) "
        r"CONSTANT\[0\]:(\d+)",
        resource_usage,
    )
    require(match is not None, "R64 kernel resource tuple is missing")
    registers, stack, shared, local, constant0 = map(int, match.groups())
    require(registers == 18, f"R64 register count drifted to {registers}")
    require(stack == 0, f"R64 stack usage is {stack}")
    require(shared == 0, f"R64 shared usage is {shared}")
    require(local == 0, f"R64 local usage is {local}")
    require(constant0 == 928, f"R64 constant0 usage drifted to {constant0}")
    return {
        "registers_per_thread": registers,
        "stack_bytes_per_thread": stack,
        "local_bytes_per_thread": local,
        "static_shared_bytes_per_cta": shared,
        "constant0_bytes": constant0,
    }


def audit_elf(elf_list: str, elf_dump: str) -> dict[str, object]:
    elf_lines = [line.strip() for line in elf_list.splitlines() if line.strip()]
    require(len(elf_lines) == 1, f"expected one cubin, found {len(elf_lines)}")
    require(elf_lines[0].endswith("sm_121a.cubin"), "cubin target is not sm_121a")
    require("CUDA Virtual SM: sm_121" in elf_dump, "virtual SM metadata drifted")
    require("CUDA Tool Kit Version: 13.0" in elf_dump, "CUDA toolkit drifted")
    require(
        re.search(
            r"Attribute:\s*EIATTR_MAX_THREADS\s*\n"
            r"\s*Format:\s*EIFMT_SVAL\s*\n"
            r"\s*Value:\s*0x400 0x1 0x1",
            elf_dump,
        )
        is not None,
        "compiled launch bound is not exactly 1024 threads",
    )
    require("register count: 18" in elf_dump, "ELF register count drifted")
    require("frame size: 0x0" in elf_dump, "ELF frame size is nonzero")
    require("min stack size: 0x0" in elf_dump, "ELF minimum stack is nonzero")
    require(
        re.search(
            r"Attribute:\s*EIATTR_CRS_STACK_SIZE\s*\n"
            r"\s*Format:\s*EIFMT_SVAL\s*\n"
            r"\s*Value:\s*0x0",
            elf_dump,
        )
        is not None,
        "ELF CRS stack is nonzero or missing",
    )
    return {
        "cubin_count": 1,
        "target": "sm_121a",
        "virtual_sm": "sm_121",
        "cuda_toolkit": "13.0",
        "max_threads": [1024, 1, 1],
        "frame_bytes_per_thread": 0,
        "crs_stack_bytes": 0,
    }


def sass_mnemonics(sass: str) -> list[str]:
    mnemonics: list[str] = []
    instruction = re.compile(
        r"^\s*/\*[0-9a-f]+\*/\s+(?:@[!]?[A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)"
    )
    for line in sass.splitlines():
        match = instruction.match(line)
        if match is not None:
            mnemonics.append(match.group(1))
    return mnemonics


def audit_sass(sass: str) -> dict[str, int]:
    require(".target\tsm_121a" in sass, "SASS target is not sm_121a")
    require(sass.count("//--------------------- .text.") == 1, "kernel count drifted")
    mnemonics = sass_mnemonics(sass)
    counts = Counter(mnemonics)
    require(len(mnemonics) == 64, f"encoded instruction count is {len(mnemonics)}")
    require(counts["NOP"] == 14, f"NOP count drifted to {counts['NOP']}")
    require(counts["SHFL.DOWN"] == 4, "shuffle instruction count drifted")
    require(counts["FADD"] == 4, "FADD instruction count drifted")
    require(counts["FFMA"] == 2, "FFMA instruction count drifted")
    require(counts["LDG.E.U16.CONSTANT"] == 2, "global load count drifted")
    require(counts["STG.E.U16"] == 1, "global store count drifted")
    require(counts["F2FP.BF16.F32.PACK_AB"] == 1, "BF16 conversion drifted")

    forbidden_prefixes = ("BAR", "LDL", "STL", "CALL", "ATOM", "RED")
    forbidden = sorted(
        mnemonic
        for mnemonic in counts
        if any(
            mnemonic == prefix or mnemonic.startswith(prefix + ".")
            for prefix in forbidden_prefixes
        )
    )
    require(not forbidden, f"forbidden SASS instructions present: {forbidden}")
    return {
        "encoded_instructions_including_nop": len(mnemonics),
        "nop": counts["NOP"],
        "operational_instructions": len(mnemonics) - counts["NOP"],
        "ldg_u16": counts["LDG.E.U16.CONSTANT"],
        "stg_u16": counts["STG.E.U16"],
        "shuffle_down": counts["SHFL.DOWN"],
        "fadd": counts["FADD"],
        "ffma_static": counts["FFMA"],
        "bf16_conversion": counts["F2FP.BF16.F32.PACK_AB"],
        "cta_barrier": 0,
        "local_load_store": 0,
        "calls": 0,
        "atomics": 0,
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def audit(args: argparse.Namespace) -> dict[str, object]:
    attestation = json.loads(read_ascii(args.build_attestation))
    require(
        attestation.get("schema")
        == "fr13.fixed32.bf16_gemvx_k64_m1_shuffle_r64_build.v1",
        "build attestation schema drifted",
    )
    require(attestation.get("status") == "BUILT_UNQUALIFIED", "build status drifted")
    require(attestation.get("torch_version") == "2.11.0+cu130", "Torch drifted")
    require(attestation.get("cuda_release") == "13.0", "CUDA release drifted")
    require(attestation.get("cuda_arch") == "12.1a", "CUDA arch drifted")
    for false_claim in (
        "performance_measurement",
        "byte_equality_claim",
        "resource_claim",
        "production_default_enabled",
    ):
        require(attestation.get(false_claim) is False, f"{false_claim} must be false")

    source_sha = sha256_file(args.source)
    binary_sha = sha256_file(args.binary)
    require(attestation["source"]["sha256"] == source_sha, "source hash drifted")
    require(attestation["binary"]["sha256"] == binary_sha, "binary hash drifted")
    require(attestation["binary"]["bytes"] == args.binary.stat().st_size, "binary size drifted")
    expected_contract = {
        "grid": [1024, 1, 1],
        "block": [16, 64, 1],
        "threads_per_cta": 1024,
        "output_rows_per_cta": 64,
        "k_partition_lanes": 16,
        "lane_k_iterations": 320,
        "reduction_strides": [8, 4, 2, 1],
    }
    for key, value in expected_contract.items():
        require(attestation["kernel_contract"].get(key) == value, f"{key} drifted")

    source_contract = audit_source(read_ascii(args.source), read_ascii(args.r32_source))
    resources = audit_resource_usage(read_ascii(args.resource_usage))
    elf = audit_elf(read_ascii(args.elf_list), read_ascii(args.elf_dump))
    sass = audit_sass(read_ascii(args.sass))
    payload: dict[str, object] = {
        "schema": "fr13.fixed32.dfwd_k64_m1_shuffle_r64_static_codegen.v1",
        "status": "STATIC_CODEGEN_PASS_UNQUALIFIED",
        "acceptance_valid": False,
        "byte_qualified": False,
        "timing_eligible": False,
        "performance_claim": False,
        "production_default_enabled": False,
        "gpu_used": False,
        "runtime_wired": False,
        "source": {
            "branch": args.source_branch,
            "commit": args.source_commit,
            "cuda_sha256": source_sha,
            "r32_parent_cuda_sha256": sha256_file(args.r32_source),
            "builder_sha256": sha256_file(args.builder),
            "source_test_sha256": sha256_file(args.source_test),
        },
        "toolchain": {
            "build_image_digest": args.build_image_digest,
            "torch": attestation["torch_version"],
            "cuda_release": attestation["cuda_release"],
            "target_arch": "sm_121a",
            "network_enabled": False,
            "gpu_exposed": False,
            "sass_auditor": read_ascii(args.nvdisasm_version).strip().splitlines()[-1],
        },
        "candidate": {
            "operation": "fr13_bf16_k64_head::gemvx_m1_shuffle_r64_out",
            "binary_sha256": binary_sha,
            "binary_bytes": args.binary.stat().st_size,
            "object_sha256": sha256_file(args.object),
            "object_bytes": args.object.stat().st_size,
            "cubin_sha256": sha256_file(args.cubin),
            "cubin_bytes": args.cubin.stat().st_size,
            **source_contract,
            "dynamic_shared_bytes": 0,
        },
        "resources": resources,
        "elf": elf,
        "sass": sass,
        "checks": {
            "extension_registration": "pass",
            "source_exact_order": "pass",
            "resource_audit": "pass",
            "elf_geometry_audit": "pass",
            "sass_audit": "pass",
        },
        "not_run": [
            "GPU kernel runtime",
            "real SWE-Verified task",
            "byte equality gate",
            "B1 or B4 full-step timing",
            "hardware-floor acceptance",
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--cubin", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--r32-source", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--source-test", type=Path, required=True)
    parser.add_argument("--build-attestation", type=Path, required=True)
    parser.add_argument("--resource-usage", type=Path, required=True)
    parser.add_argument("--elf-list", type=Path, required=True)
    parser.add_argument("--elf-dump", type=Path, required=True)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--nvdisasm-version", type=Path, required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--build-image-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args)
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
