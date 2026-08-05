#!/usr/bin/env python3
"""Fail-closed SM121a audit for the fixed32 B1/B4 K64 Tensor Core head."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


FUNCTION = re.compile(r"^\s*Function : (.+)$", re.MULTILINE)
INSTRUCTION = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+"
    r"(?:@[!]?(?:P\d+|UP\d+|UPT)\s+)?([A-Z][A-Z0-9_.]*)\b"
)
RESOURCE = re.compile(
    r"Function ([^\n]*?(M([14])IdentitySwizzle)[^\n]*):\s*\n"
    r"\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)"
)

EXPECTED_STATIC_INSTRUCTIONS = 760
EXPECTED_COUNTS = {
    "HMMA.16816.F32.BF16": 32,
    "LDG.E.LTC128B.128": 34,
    "LDG.E.64": 1,
    "LDG.E": 1,
    "STG.E.128": 4,
    "BAR.SYNC.DEFER_BLOCKING": 6,
}
EXPECTED_RESOURCES = {
    "registers_per_thread": 168,
    "stack_bytes_per_thread": 0,
    "static_shared_bytes": 1024,
    "local_bytes_per_thread": 0,
    "launch_dynamic_shared_bytes": 69632,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _section(sass: str, needle: str) -> str:
    matches = list(FUNCTION.finditer(sass))
    selected = [index for index, match in enumerate(matches) if needle in match.group(1)]
    require(len(selected) == 1, f"expected exactly one {needle} kernel")
    index = selected[0]
    start = matches[index].start()
    end = matches[index + 1].start() if index + 1 < len(matches) else len(sass)
    return sass[start:end]


def _counts(section: str) -> Counter[str]:
    return Counter(
        match.group(1)
        for line in section.splitlines()
        if (match := INSTRUCTION.match(line)) is not None
    )


def _audit_counts(counts: Counter[str], batch: int) -> None:
    require(
        sum(counts.values()) == EXPECTED_STATIC_INSTRUCTIONS,
        f"B{batch} static instruction count drifted",
    )
    for operation, expected in EXPECTED_COUNTS.items():
        require(counts[operation] == expected, f"B{batch} {operation} count drifted")
    require(
        sum(count for operation, count in counts.items() if operation.startswith("HMMA"))
        == EXPECTED_COUNTS["HMMA.16816.F32.BF16"],
        f"B{batch} unexpected HMMA variant appeared",
    )
    require(
        sum(count for operation, count in counts.items() if operation.startswith("LDG"))
        == sum(
            count
            for operation, count in EXPECTED_COUNTS.items()
            if operation.startswith("LDG")
        ),
        f"B{batch} global-load instruction count drifted",
    )
    require(
        sum(count for operation, count in counts.items() if operation.startswith("STG"))
        == sum(
            count
            for operation, count in EXPECTED_COUNTS.items()
            if operation.startswith("STG")
        ),
        f"B{batch} global-store instruction count drifted",
    )
    for forbidden in ("LDL", "STL", "ATOM", "CALL"):
        require(
            not any(operation.startswith(forbidden) for operation in counts),
            f"B{batch} contains forbidden {forbidden} instruction",
        )


def _resources(resource: str) -> dict[str, dict[str, int]]:
    require("arch = sm_121a" in resource, "resource image is not SM121a")
    records: dict[str, dict[str, int]] = {}
    for _, marker, batch, registers, stack, shared, local in RESOURCE.findall(resource):
        require(marker == f"M{batch}IdentitySwizzle", "kernel marker parse drifted")
        key = f"b{batch}"
        require(key not in records, f"duplicate {key} resource record")
        records[key] = {
            "registers_per_thread": int(registers),
            "stack_bytes_per_thread": int(stack),
            "static_shared_bytes": int(shared),
            "local_bytes_per_thread": int(local),
            "launch_dynamic_shared_bytes": EXPECTED_RESOURCES[
                "launch_dynamic_shared_bytes"
            ],
        }
    require(set(records) == {"b1", "b4"}, "B1/B4 resource records drifted")
    for batch, record in records.items():
        require(record == EXPECTED_RESOURCES, f"{batch.upper()} resources drifted")
    return records


def _traffic_model() -> dict[str, object]:
    weight_bytes = 65536 * 5120 * 2
    pair8_reads = {"b1": 838860800, "b4": 1342177280}
    batches = {"b1": 1, "b4": 4}
    model: dict[str, object] = {
        "scope": "algorithmic global bytes; not measured DRAM traffic",
        "weight_bytes_per_call": weight_bytes,
        "input_replication_factor": 256,
    }
    for key, batch in batches.items():
        input_bytes = 256 * batch * 5120 * 2
        output_bytes = batch * 65536 * 2
        reads = weight_bytes + input_bytes
        model[key] = {
            "duplicated_input_read_bytes": input_bytes,
            "global_read_bytes": reads,
            "output_write_bytes": output_bytes,
            "total_bytes": reads + output_bytes,
            "pair8_global_read_bytes": pair8_reads[key],
            "pair8_global_read_reduction_fraction": 1.0
            - reads / pair8_reads[key],
        }
    return model


def _compute_model() -> dict[str, object]:
    return {
        "tensor_core_executed_flops_per_call": 2 * 16 * 65536 * 5120,
        "useful_flops": {"b1": 2 * 1 * 65536 * 5120, "b4": 2 * 4 * 65536 * 5120},
        "padding_factor": {"b1": 16, "b4": 4},
        "claim": "modeled only; no timing or achieved-throughput measurement",
    }


def audit(sass: str, resource: str) -> dict[str, object]:
    require(".target\tsm_121a" in sass, "SASS image is not SM121a")
    b1_counts = _counts(_section(sass, "M1IdentitySwizzle"))
    b4_counts = _counts(_section(sass, "M4IdentitySwizzle"))
    _audit_counts(b1_counts, 1)
    _audit_counts(b4_counts, 4)
    require(b1_counts == b4_counts, "B1/B4 fixed kernel bodies diverged")
    resources = _resources(resource)
    return {
        "schema": "fr13.fixed32.dfwd_k64_tc16x256x64_s2_codegen_audit.v1",
        "status": "STATIC_CODEGEN_PASS_UNQUALIFIED",
        "acceptance_valid": False,
        "performance_measurement": False,
        "real_task_correctness": False,
        "production_default_enabled": False,
        "runtime_wired": False,
        "gpu_used": False,
        "architecture": "sm_121a",
        "kernel_contract": {
            "problem_mnk": {"b1": [1, 65536, 5120], "b4": [4, 65536, 5120]},
            "threadblock_mnk": [16, 256, 64],
            "warp_mnk": [16, 64, 64],
            "instruction_mnk": [16, 8, 16],
            "stages": 2,
            "threads_per_cta": 128,
            "logical_grid_ctas": 256,
            "epilogue_source_reads_disabled": True,
        },
        "resources": resources,
        "sass": {
            "b1_static_instructions": sum(b1_counts.values()),
            "b4_static_instructions": sum(b4_counts.values()),
            "b1_selected_instruction_counts": {
                operation: b1_counts[operation] for operation in EXPECTED_COUNTS
            },
            "b4_selected_instruction_counts": {
                operation: b4_counts[operation] for operation in EXPECTED_COUNTS
            },
        },
        "only_alpha_epilogue_codegen_delta": {
            "scope": "static compiler output; not a performance measurement",
            "generic_epilogue_static_instructions": 952,
            "only_alpha_static_instructions": EXPECTED_STATIC_INSTRUCTIONS,
            "static_instruction_reduction_fraction": 1.0
            - EXPECTED_STATIC_INSTRUCTIONS / 952,
            "generic_epilogue_global_loads": 42,
            "only_alpha_global_loads": 36,
            "generic_epilogue_global_stores": 8,
            "only_alpha_global_stores": 4,
            "generic_epilogue_barriers": 10,
            "only_alpha_barriers": 6,
        },
        "logical_global_traffic_model": _traffic_model(),
        "compute_model": _compute_model(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(
        args.sass.read_text(encoding="ascii"),
        args.resource.read_text(encoding="ascii"),
    )
    encoded = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="ascii")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
