#!/usr/bin/env python3
"""Verify isolated fixed32 committer SM121a builds and static gates."""

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


def opcounts(sass: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in sass.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match:
            op = match.group(1).split(".", 1)[0]
            result[op] = result.get(op, 0) + 1
    return result


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
    counts = opcounts(sass)
    for key, op in (
        ("ldg", "LDG"),
        ("stg", "STG"),
        ("lds", "LDS"),
        ("sts", "STS"),
        ("ldl", "LDL"),
        ("stl", "STL"),
        ("bar", "BAR"),
        ("redux", "REDUX"),
    ):
        if counts.get(op, 0) != expected[key]:
            raise RuntimeError(f"{root}: {op} count mismatch")
    global_atomics = sum(
        count for op, count in counts.items() if op.startswith("ATOM")
    )
    if global_atomics != expected["global_atomics"]:
        raise RuntimeError(f"{root}: global atomic count mismatch")
    if expected["calls"] != sum(
        count for op, count in counts.items() if op.startswith("CALL")
    ):
        raise RuntimeError(f"{root}: call count mismatch")
    if digest(cubin) != expected["cubin_sha256"]:
        raise RuntimeError(f"{root}: cubin digest mismatch")
    if digest(root / "kernel.ptx") != expected["ptx_sha256"]:
        raise RuntimeError(f"{root}: PTX digest mismatch")
    if hashlib.sha256(sass.encode()).hexdigest() != expected["sass_sha256"]:
        raise RuntimeError(f"{root}: SASS digest mismatch")
    if "ptxas-blackwell" not in elf:
        raise RuntimeError(f"{root}: unexpected cubin producer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--rebuild", type=Path, required=True)
    args = parser.parse_args()
    primary = json.loads((args.primary / "summary.json").read_text())
    rebuild = json.loads((args.rebuild / "summary.json").read_text())
    if primary != rebuild:
        raise RuntimeError("fresh-cache summary differs from primary")
    labels = ("incumbent_vector_result", "candidate_sticky_scalar")
    for label in labels:
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
            if any(expected[key] for key in ("stack_bytes", "local_bytes")):
                raise RuntimeError(f"{label}/{batch}: local storage is nonzero")
            if any(expected[key] for key in ("ldl", "stl", "calls")):
                raise RuntimeError(f"{label}/{batch}: spill or call gate failed")
    for batch in ("b1", "b4"):
        incumbent = primary["variants"]["incumbent_vector_result"]["builds"][batch]
        candidate = primary["variants"]["candidate_sticky_scalar"]["builds"][batch]
        if candidate["registers"] > incumbent["registers"]:
            raise RuntimeError(f"{batch}: candidate raises registers")
        if candidate["ldg"] != incumbent["ldg"]:
            raise RuntimeError(f"{batch}: validation loads changed")
        if incumbent["stg"] != 1 or candidate["stg"] != 0:
            raise RuntimeError(f"{batch}: valid-path result store was not removed")
        if incumbent["global_atomics"] != 0 or candidate["global_atomics"] != 1:
            raise RuntimeError(f"{batch}: failure-only atomic shape drifted")
        if any(candidate[key] for key in ("lds", "sts", "bar")):
            raise RuntimeError(f"{batch}: candidate is not warp-local")
        if candidate["launch_shared_bytes"] != 0:
            raise RuntimeError(f"{batch}: candidate launch shared is nonzero")
        incumbent_work = incumbent["logical_work"]
        candidate_work = candidate["logical_work"]
        if incumbent_work["scalar_reduction_launches_per_event"] != 1:
            raise RuntimeError(f"{batch}: incumbent reduction not represented")
        if candidate_work["scalar_reduction_launches_per_event"] != 0:
            raise RuntimeError(f"{batch}: candidate retains scalar reduction")
        if candidate_work["source_visible_guard_pipeline_launches_per_event"] != 2:
            raise RuntimeError(f"{batch}: candidate launch model drifted")
        if candidate_work["result_bytes_stored_on_valid_event"] != 0:
            raise RuntimeError(f"{batch}: candidate valid path still stores")
        sass_lines = (
            args.primary
            / "candidate_sticky_scalar"
            / batch
            / "kernel.sass"
        ).read_text().splitlines()
        atomic_indices = [
            index for index, line in enumerate(sass_lines) if "ATOMG" in line
        ]
        if len(atomic_indices) != 1 or not any(
            "@!P" in line and "EXIT" in line
            for line in sass_lines[max(0, atomic_indices[0] - 8) : atomic_indices[0]]
        ):
            raise RuntimeError(f"{batch}: atomic is not behind the failure branch")
    print(
        json.dumps(
            {
                "schema": "fr13.fixed32.committer_sticky_guard.sm121a.verify.v1",
                "status": "PASS",
                "builds_verified": 8,
                "fresh_cache_byte_identity": True,
                "gpu_execution": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
