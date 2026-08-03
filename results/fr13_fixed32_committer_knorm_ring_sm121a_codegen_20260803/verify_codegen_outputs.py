#!/usr/bin/env python3
"""Verify paired fixed32 K-norm-ring SM121a builds and static gates."""

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


RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+"
    r"(?:@[!]?(?:U)?P\d+\s+)?([A-Z][A-Z0-9.]*)\b"
)
LABELS = (
    "producer_parent_incumbent",
    "producer_current_incumbent",
    "producer_candidate_knorm_export",
    "committer_parent_incumbent",
    "committer_current_incumbent",
    "committer_candidate_knorm_reuse",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str]) -> str:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout


def opcounts(sass: str) -> tuple[dict[str, int], dict[str, int]]:
    base: dict[str, int] = {}
    full: dict[str, int] = {}
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            operation = match.group(1)
            full[operation] = full.get(operation, 0) + 1
            root = operation.split(".", 1)[0]
            base[root] = base.get(root, 0) + 1
    return base, full


def verify_build(root: Path, expected: dict[str, object]) -> None:
    cubin = root / "kernel.cubin"
    sass = run(["/usr/local/cuda/bin/nvdisasm", "-c", str(cubin)])
    resource = run(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(cubin)]
    )
    elf = run(["/usr/local/cuda/bin/cuobjdump", "--dump-elf", str(cubin)])
    if ".target\tsm_121a" not in sass:
        raise RuntimeError(f"{root}: cubin target is not sm_121a")
    match = RESOURCE_RE.search(resource)
    if match is None:
        raise RuntimeError(f"{root}: resource record missing")
    resources = tuple(map(int, match.groups()))
    wanted = (
        expected["registers"],
        expected["stack_bytes"],
        expected["elf_shared_bytes"],
        expected["local_bytes"],
    )
    if resources != wanted:
        raise RuntimeError(f"{root}: resource mismatch {resources} != {wanted}")
    base, full = opcounts(sass)
    if base != expected["base_operations"]:
        raise RuntimeError(f"{root}: base operation census mismatch")
    if full != expected["full_operations"]:
        raise RuntimeError(f"{root}: full operation census mismatch")
    if digest(cubin) != expected["cubin_sha256"]:
        raise RuntimeError(f"{root}: cubin digest mismatch")
    if digest(root / "kernel.ptx") != expected["ptx_sha256"]:
        raise RuntimeError(f"{root}: PTX digest mismatch")
    if hashlib.sha256(sass.encode()).hexdigest() != expected["sass_sha256"]:
        raise RuntimeError(f"{root}: SASS digest mismatch")
    if "ptxas-blackwell" not in elf:
        raise RuntimeError(f"{root}: unexpected cubin producer")


def count(build: dict[str, object], operation: str) -> int:
    return int(build["base_operations"].get(operation, 0))


def full_count(build: dict[str, object], operation: str) -> int:
    return int(build["full_operations"].get(operation, 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--rebuild", type=Path, required=True)
    args = parser.parse_args()
    primary = json.loads((args.primary / "summary.json").read_text())
    rebuild = json.loads((args.rebuild / "summary.json").read_text())
    if primary != rebuild:
        raise RuntimeError("fresh-cache summary differs from primary")

    for label in LABELS:
        for batch in ("b1", "b4"):
            expected = primary["variants"][label]["builds"][batch]
            first = args.primary / label / batch
            second = args.rebuild / label / batch
            verify_build(first, expected)
            verify_build(second, expected)
            for name in (
                "kernel.cubin",
                "kernel.ptx",
                "kernel.sass",
                "resource.txt",
                "elf.txt",
            ):
                if digest(first / name) != digest(second / name):
                    raise RuntimeError(f"{label}/{batch}/{name}: rebuild differs")
            if any(int(expected[key]) for key in ("stack_bytes", "local_bytes")):
                raise RuntimeError(f"{label}/{batch}: local storage is nonzero")
            if any(count(expected, op) for op in ("LDL", "STL", "CALL")):
                raise RuntimeError(f"{label}/{batch}: spill or call gate failed")

    for batch in ("b1", "b4"):
        parent_p = primary["variants"]["producer_parent_incumbent"]["builds"][batch]
        current_p = primary["variants"]["producer_current_incumbent"]["builds"][batch]
        candidate_p = primary["variants"]["producer_candidate_knorm_export"]["builds"][batch]
        parent_c = primary["variants"]["committer_parent_incumbent"]["builds"][batch]
        current_c = primary["variants"]["committer_current_incumbent"]["builds"][batch]
        candidate_c = primary["variants"]["committer_candidate_knorm_reuse"]["builds"][batch]

        for left, right, kernel in (
            (parent_p, current_p, "producer"),
            (parent_c, current_c, "committer"),
        ):
            if left["sass_sha256"] != right["sass_sha256"]:
                raise RuntimeError(f"{batch}: default-off {kernel} SASS changed")
            if left["base_operations"] != right["base_operations"]:
                raise RuntimeError(f"{batch}: default-off {kernel} operations changed")
            if left["registers"] != right["registers"]:
                raise RuntimeError(f"{batch}: default-off {kernel} registers changed")

        for operation in ("LDG", "MUFU", "SHFL", "BAR", "LDS", "STS"):
            if count(candidate_p, operation) != count(current_p, operation):
                raise RuntimeError(
                    f"{batch}: producer candidate changed {operation}; "
                    "the exported norm must reuse existing reduction work"
                )
        if full_count(candidate_p, "MUFU.RSQ") != full_count(current_p, "MUFU.RSQ"):
            raise RuntimeError(f"{batch}: producer candidate added an RSQ")
        if count(candidate_p, "STG") <= count(current_p, "STG"):
            raise RuntimeError(f"{batch}: producer candidate has no scalar store")
        if candidate_p["registers"] != current_p["registers"]:
            raise RuntimeError(f"{batch}: producer candidate raises registers")
        producer_work = candidate_p["logical_work"]
        if producer_work["key_norm_reductions_added"] != 0:
            raise RuntimeError(f"{batch}: producer logical work adds reductions")
        expected_values = 48 * int(batch[1:]) * 32 * 16
        if producer_work["key_norm_scalar_values_stored"] != expected_values:
            raise RuntimeError(f"{batch}: producer scalar-ring census drift")

        if full_count(current_c, "MUFU.RSQ") != 1 or full_count(candidate_c, "MUFU.RSQ") != 0:
            raise RuntimeError(f"{batch}: committer RSQ was not removed")
        if count(current_c, "SHFL") - count(candidate_c, "SHFL") != 7:
            raise RuntimeError(f"{batch}: committer reduction shuffle delta drift")
        if count(current_c, "BAR") != 3 or count(candidate_c, "BAR") != 0:
            raise RuntimeError(f"{batch}: committer reduction barriers remain")
        if candidate_c["launch_shared_bytes"] != 0:
            raise RuntimeError(f"{batch}: committer candidate retains reduction scratch")
        if count(candidate_c, "LDS") or count(candidate_c, "STS"):
            raise RuntimeError(f"{batch}: committer candidate retains shared traffic")
        if int(candidate_c["registers"]) >= 200:
            raise RuntimeError(f"{batch}: committer candidate register gate failed")
        for depth in ("accepted_0", "accepted_4", "accepted_11"):
            incumbent_work = current_c["logical_work"]["dynamic_step_census"][depth]
            candidate_work = candidate_c["logical_work"]["dynamic_step_census"][depth]
            if candidate_work["key_norm_reductions"] != 0:
                raise RuntimeError(f"{batch}/{depth}: candidate retains reductions")
            if candidate_work["key_norm_reductions_removed"] != incumbent_work["key_norm_reductions"]:
                raise RuntimeError(f"{batch}/{depth}: removed-reduction census drift")
            if candidate_work["key_norm_scalar_loads"] != incumbent_work["key_norm_reductions"]:
                raise RuntimeError(f"{batch}/{depth}: scalar-load census drift")

    print(
        json.dumps(
            {
                "schema": "fr13.fixed32.committer_knorm_ring.sm121a.verify.v1",
                "status": "PASS",
                "builds_verified": 24,
                "fresh_cache_byte_identity": True,
                "default_off_sass_identity": True,
                "producer_extra_norm_reductions": 0,
                "committer_norm_reduction_removed": True,
                "gpu_execution": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
