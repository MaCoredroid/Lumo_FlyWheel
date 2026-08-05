#!/usr/bin/env python3
"""Verify deterministic B1+B4 GQA3 value-domain SM121a codegen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")


SCHEMA = "fr13.fixed32.gdn_gqa_group3_value_domain.sm121a.codegen.v1"
BASELINE_REVISION = "cbca5f65a5af17364e356045a3e633f885908d11"
CANDIDATE_REVISION = "1d08d3952d806306816de12988e5aa1258620566"
SOURCE = "src/lumo_flywheel_serving/fr13_gdn_gqa_group3.py"
PROFILE_PAIRS = (
    (
        "baseline_static_schedule_base",
        "candidate_value_domain_base",
        False,
        None,
        116,
        108,
        2080,
        2040,
        2012,
        1972,
        54,
    ),
    (
        "baseline_static_schedule_committer_stack",
        "candidate_value_domain_committer_stack",
        True,
        128,
        118,
        118,
        2184,
        2144,
        2119,
        2078,
        82,
    ),
)
BATCHES = (1, 4)
VARIANTS = tuple(
    variant
    for baseline, candidate, *_metrics in PROFILE_PAIRS
    for variant in (baseline, candidate)
)
RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+"
    r"(?:@[!]?(?:U)?P\d+\s+)?([A-Z][A-Z0-9.]*)\b"
)
CONTROL_OR_PADDING = {"NOP", "BAR", "BRA", "EXIT"}


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


def build(
    summary: dict[str, object], variant: str, batch_label: str
) -> dict[str, object]:
    variants = summary.get("variants")
    if not isinstance(variants, dict):
        raise VerificationError("summary variants are missing")
    variant_row = variants.get(variant)
    if not isinstance(variant_row, dict):
        raise VerificationError(f"summary variant is missing: {variant}")
    builds = variant_row.get("builds")
    if not isinstance(builds, dict) or not isinstance(
        builds.get(batch_label), dict
    ):
        raise VerificationError(
            f"summary build is missing: {variant}/{batch_label}"
        )
    return builds[batch_label]


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
    root: Path,
    summary: dict[str, object],
    variant: str,
    batch_label: str,
) -> None:
    row = build(summary, variant, batch_label)
    build_root = root / variant / batch_label
    cubin = build_root / "kernel.cubin"
    ptx = build_root / "kernel.ptx"
    sass_path = build_root / "kernel.sass"
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
        raise VerificationError(
            f"cannot parse resources for {variant}/{batch_label}"
        )
    registers, stack_bytes, shared_bytes, local_bytes = map(int, match.groups())
    operations = operation_counts(sass.decode())
    expected = {
        "cubin_sha256": sha256(cubin.read_bytes()),
        "ptx_sha256": sha256(ptx.read_bytes()),
        "sass_sha256": sha256(sass),
        "registers_per_thread": registers,
        "stack_bytes_per_thread": stack_bytes,
        "local_bytes_per_thread": local_bytes,
        "elf_shared_bytes_per_cta": shared_bytes,
        "encoded_sass_instructions": sum(operations.values()),
        "static_sass_instructions": sum(
            count
            for operation, count in operations.items()
            if operation not in CONTROL_OR_PADDING
        ),
        "ldg": operations.get("LDG", 0),
        "stg": operations.get("STG", 0),
        "ldl": operations.get("LDL", 0),
        "stl": operations.get("STL", 0),
        "calls": sum(
            count
            for operation, count in operations.items()
            if operation.startswith("CALL")
        ),
        "base_operations": dict(sorted(operations.items())),
    }
    if sass_path.read_bytes() != sass:
        raise VerificationError(f"saved SASS differs from nvdisasm for {variant}")
    for key, value in expected.items():
        if row.get(key) != value:
            raise VerificationError(
                f"recorded {key} differs from build for "
                f"{variant}/{batch_label}"
            )


def verify_contract(summary: dict[str, object]) -> None:
    if (
        summary.get("schema") != SCHEMA
        or summary.get("baseline_revision") != BASELINE_REVISION
        or summary.get("candidate_revision") != CANDIDATE_REVISION
    ):
        raise VerificationError("summary schema or revisions drifted")
    variants = summary.get("variants")
    if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
        raise VerificationError("summary variant set drifted")
    contract = summary.get("compile_contract")
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
        "device_descriptor_pointer_args_removed": 5,
        "device_descriptor_loads_removed_per_cta": 59,
        "device_descriptor_loads_removed_per_48_layer_event": {
            "b1": 724992,
            "b4": 2899968,
        },
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
        raise VerificationError("exact B4 compile contract drifted")

    producer = None
    for (
        baseline_name,
        candidate_name,
        committer_stack,
        maxnreg,
        baseline_registers,
        candidate_registers,
        baseline_encoded,
        candidate_encoded,
        baseline_static,
        candidate_static,
        stg,
    ) in PROFILE_PAIRS:
        for batch in BATCHES:
            batch_label = f"b{batch}"
            baseline = build(summary, baseline_name, batch_label)
            candidate = build(summary, candidate_name, batch_label)
            for name, row, revision in (
                (baseline_name, baseline, BASELINE_REVISION),
                (candidate_name, candidate, CANDIDATE_REVISION),
            ):
                row_exact = {
                    "label": name,
                    "revision": revision,
                    "source_path": SOURCE,
                    "batch": batch,
                    "committer_stack": committer_stack,
                    "trust_fixed32_node_domain": True,
                    "static_physical32_schedule": True,
                    "resolved_maxnreg": maxnreg,
                    "resolved_num_stages": 3,
                    "num_warps": 8,
                    "threads_per_cta": 256,
                    "grid": [16, 16, batch],
                    "programs_per_layer_event": 256 * batch,
                    "programs_per_48_layer_event": 12288 * batch,
                    "launch_shared_bytes_per_cta": 16,
                    "elf_shared_bytes_per_cta": 1024,
                    "stack_bytes_per_thread": 0,
                    "local_bytes_per_thread": 0,
                    "ldl": 0,
                    "stl": 0,
                    "calls": 0,
                    "ldg": 74,
                    "stg": stg,
                    "gpu_execution": False,
                }
                for key, value in row_exact.items():
                    if row.get(key) != value:
                        raise VerificationError(
                            f"{name}/{batch_label} exact {key} drifted"
                        )
                if producer is None:
                    producer = row.get("backend_producer")
                elif row.get("backend_producer") != producer:
                    raise VerificationError(
                        "backend producer differs across builds"
                    )
            expected_metrics = (
                (
                    baseline,
                    baseline_registers,
                    baseline_encoded,
                    baseline_static,
                ),
                (
                    candidate,
                    candidate_registers,
                    candidate_encoded,
                    candidate_static,
                ),
            )
            for row, registers, encoded, static in expected_metrics:
                if (
                    row.get("registers_per_thread") != registers
                    or row.get("register_bytes_per_cta") != registers * 1024
                    or row.get("encoded_sass_instructions") != encoded
                    or row.get("static_sass_instructions") != static
                ):
                    raise VerificationError("resource or SASS metric drifted")
            if candidate_registers > baseline_registers:
                raise VerificationError("candidate register regression")
            if candidate.get("cubin_sha256") == baseline.get("cubin_sha256"):
                raise VerificationError(
                    "candidate cubin unexpectedly equals baseline"
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
            verify_disassembly(primary, summary, variant, f"b{batch}")
    result = {
        "schema": (
            "fr13.fixed32.gdn_gqa_group3_value_domain.sm121a.verify.v1"
        ),
        "status": "PASS",
        "baseline_revision": BASELINE_REVISION,
        "candidate_revision": CANDIDATE_REVISION,
        "builds_verified": 8,
        "fresh_cache_byte_identity": True,
        "registers_per_thread": {
            "base": {"baseline": 116, "candidate": 108},
            "committer_stack": {"baseline": 118, "candidate": 118},
        },
        "static_sass_instructions": {
            "base": {"baseline": 2012, "candidate": 1972},
            "committer_stack": {"baseline": 2119, "candidate": 2078},
        },
        "ldg_sites": {"baseline": 74, "candidate": 74},
        "value_domain_masks_removed_per_cta": 291,
        "value_domain_masks_removed_per_48_layer_event": {
            "b1": 3575808,
            "b4": 14303232,
        },
        "invocation_atomics_removed_per_event": {
            "b1": 0,
            "b4": 3,
        },
        "stack_local_bytes_per_thread": 0,
        "ldl_stl_calls": 0,
        "resource_gate": "PASS",
        "performance_promotion": False,
        "gpu_execution": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
