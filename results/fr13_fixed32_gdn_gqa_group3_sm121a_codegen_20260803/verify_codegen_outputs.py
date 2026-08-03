#!/usr/bin/env python3
"""Verify deterministic, spill-free GDN GQA3 SM121a code generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")


SCHEMA = "fr13.fixed32.gdn_gqa_group3.sm121a.codegen.v1"
REVISION = "936dd110c01d34f8c1c5c64676dde5739d0d2fa3"
PROFILE_PAIRS = (
    (
        "incumbent_base_production",
        "candidate_gqa_group3_base_production",
        120,
        None,
    ),
    (
        "incumbent_committer_stack_production",
        "candidate_gqa_group3_committer_stack_production",
        126,
        128,
    ),
)
VARIANTS = tuple(
    variant for incumbent, candidate, _registers, _maxnreg in PROFILE_PAIRS
    for variant in (incumbent, candidate)
)
BATCHES = ("b1", "b4")
RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+"
    r"(?:@[!]?(?:U)?P\d+\s+)?([A-Z][A-Z0-9.]*)\b"
)


class VerificationError(RuntimeError):
    pass


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise VerificationError(f"missing regular JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise VerificationError(f"JSON payload is not an object: {path}")
    return payload


def files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )


def run_bytes(argv: list[str]) -> bytes:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def operation_counts(sass: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            operation = match.group(1).split(".", 1)[0]
            result[operation] = result.get(operation, 0) + 1
    return result


def build(summary: dict[str, object], variant: str, batch: str):
    variants = summary.get("variants")
    if not isinstance(variants, dict):
        raise VerificationError("summary variants are missing")
    variant_row = variants.get(variant)
    if not isinstance(variant_row, dict):
        raise VerificationError(f"summary variant is missing: {variant}")
    builds = variant_row.get("builds")
    if not isinstance(builds, dict) or not isinstance(builds.get(batch), dict):
        raise VerificationError(f"summary build is missing: {variant}/{batch}")
    return builds[batch]


def verify_rebuild(primary: Path, rebuild: Path) -> None:
    primary_files = files(primary)
    rebuild_files = files(rebuild)
    if primary_files != rebuild_files:
        raise VerificationError("fresh-cache output file sets differ")
    for relative in primary_files:
        if (primary / relative).read_bytes() != (rebuild / relative).read_bytes():
            raise VerificationError(
                f"fresh-cache output differs byte-for-byte: {relative}"
            )


def verify_disassembly(
    root: Path, summary: dict[str, object], variant: str, batch: str
) -> None:
    row = build(summary, variant, batch)
    cubin = root / variant / batch / "kernel.cubin"
    sass = run_bytes(["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin)])
    resource = run_bytes(
        [
            "/usr/local/cuda/bin/cuobjdump",
            "--dump-resource-usage",
            str(cubin),
        ]
    )
    match = RESOURCE_RE.search(resource.decode())
    if match is None:
        raise VerificationError(f"cannot parse resources for {variant}/{batch}")
    registers, stack_bytes, shared_bytes, local_bytes = map(int, match.groups())
    operations = operation_counts(sass.decode())
    expected = {
        "cubin_sha256": sha256(cubin.read_bytes()),
        "sass_sha256": sha256(sass),
        "registers_per_thread": registers,
        "stack_bytes_per_thread": stack_bytes,
        "local_bytes_per_thread": local_bytes,
        "elf_shared_bytes_per_cta": shared_bytes,
        "ldl": operations.get("LDL", 0),
        "stl": operations.get("STL", 0),
        "calls": sum(
            count
            for operation, count in operations.items()
            if operation.startswith("CALL")
        ),
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise VerificationError(
                f"recorded {key} differs from cubin for {variant}/{batch}"
            )


def verify_contract(summary: dict[str, object]) -> None:
    if summary.get("schema") != SCHEMA or summary.get("revision") != REVISION:
        raise VerificationError("summary schema or revision drifted")
    contract = summary.get("compile_contract")
    if not isinstance(contract, dict):
        raise VerificationError("compile contract is missing")
    exact = {
        "target": "sm_121a",
        "batches": [1, 4],
        "physical_rows_per_request": 32,
        "key_heads": 16,
        "value_heads": 48,
        "value_heads_per_key_head": 3,
        "dim_k": 128,
        "dim_v": 128,
        "block_v": 8,
        "num_warps": 8,
        "num_stages_request": "unset",
        "resolved_num_stages": 3,
        "base_maxnreg": None,
        "committer_stack_incumbent_maxnreg": 80,
        "committer_stack_candidate_maxnreg": 128,
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
        "committer_stack_profile": {
            "k_norm_export": True,
            "gate_export": True,
            "decay_export": True,
        },
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "jit_specialization": (
            "mock_tensor_exact_shape_stride_alignment_and_pointer_dtype"
        ),
        "gpu_execution": False,
    }
    if contract != exact:
        raise VerificationError("exact production compile contract drifted")
    for batch in BATCHES:
        for incumbent_name, candidate_name, registers, maxnreg in PROFILE_PAIRS:
            incumbent = build(summary, incumbent_name, batch)
            candidate = build(summary, candidate_name, batch)
            for row in (incumbent, candidate):
                if (
                    row.get("stack_bytes_per_thread") != 0
                    or row.get("local_bytes_per_thread") != 0
                    or row.get("ldl") != 0
                    or row.get("stl") != 0
                    or row.get("calls") != 0
                    or row.get("threads_per_cta") != 256
                    or row.get("launch_shared_bytes_per_cta") != 16
                    or row.get("resolved_num_stages") != 3
                    or row.get("gpu_execution") is not False
                ):
                    raise VerificationError(
                        f"spill/launch gate failed for {candidate_name}/{batch}"
                    )
            if incumbent.get("registers_per_thread") != 80:
                raise VerificationError(
                    f"incumbent register count drifted for {incumbent_name}/{batch}"
                )
            if candidate.get("registers_per_thread") != registers:
                raise VerificationError(
                    f"candidate register count drifted for {candidate_name}/{batch}"
                )
            if candidate.get("resolved_maxnreg") != maxnreg:
                raise VerificationError(
                    f"candidate maxnreg drifted for {candidate_name}/{batch}"
                )
            expected_incumbent_maxnreg = 80 if maxnreg == 128 else None
            if incumbent.get("resolved_maxnreg") != expected_incumbent_maxnreg:
                raise VerificationError(
                    f"incumbent maxnreg drifted for {incumbent_name}/{batch}"
                )
            if candidate.get("programs_per_layer_event") * 3 != incumbent.get(
                "programs_per_layer_event"
            ):
                raise VerificationError(f"GQA3 CTA reduction drifted for {batch}")
            if candidate.get("register_bytes_per_cta") != registers * 1024:
                raise VerificationError(
                    f"candidate register footprint drifted for {candidate_name}/{batch}"
                )
            if incumbent.get("register_bytes_per_cta") != 81920:
                raise VerificationError(
                    f"incumbent register footprint drifted for {incumbent_name}/{batch}"
                )
            for metric in ("static_sass_instructions", "ldg", "stg"):
                if candidate.get(metric) * candidate.get(
                    "programs_per_layer_event"
                ) >= incumbent.get(metric) * incumbent.get(
                    "programs_per_layer_event"
                ):
                    raise VerificationError(
                        f"aggregate static {metric} proxy did not decrease for "
                        f"{candidate_name}/{batch}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--rebuild", type=Path, required=True)
    args = parser.parse_args()
    primary = args.primary.resolve()
    rebuild = args.rebuild.resolve()
    verify_rebuild(primary, rebuild)
    summary = load_json(primary / "summary.json")
    verify_contract(summary)
    for variant in VARIANTS:
        for batch in BATCHES:
            verify_disassembly(primary, summary, variant, batch)
    result = {
        "schema": "fr13.fixed32.gdn_gqa_group3.sm121a.verify.v1",
        "status": "PASS",
        "revision": REVISION,
        "builds_verified": 8,
        "fresh_cache_byte_identity": True,
        "candidate_base_registers_per_thread": {"b1": 120, "b4": 120},
        "candidate_committer_stack_registers_per_thread": {
            "b1": 126,
            "b4": 126,
        },
        "incumbent_registers_per_thread": {"b1": 80, "b4": 80},
        "candidate_base_register_bytes_per_cta": 122880,
        "candidate_committer_stack_register_bytes_per_cta": 129024,
        "incumbent_register_bytes_per_cta": 81920,
        "candidate_stack_local_bytes_per_thread": 0,
        "candidate_ldl_stl_calls": 0,
        "candidate_grid_reduction": "3x",
        "resource_gate": "PASS",
        "gpu_execution": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
