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
    ):
        if counts.get(op, 0) != expected[key]:
            raise RuntimeError(f"{root}: {op} count mismatch")
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
    for label in ("incumbent", "superseded_v3", "candidate"):
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
        incumbent = primary["variants"]["incumbent"]["builds"][batch]
        superseded = primary["variants"]["superseded_v3"]["builds"][batch]
        candidate = primary["variants"]["candidate"]["builds"][batch]
        if candidate["registers"] > incumbent["registers"]:
            raise RuntimeError(f"{batch}: candidate raises registers")
        if candidate["ldg"] >= incumbent["ldg"]:
            raise RuntimeError(f"{batch}: candidate does not reduce static LDG")
        if candidate["stg"] != incumbent["stg"]:
            raise RuntimeError(f"{batch}: candidate changes guard stores")
        if any(candidate[key] for key in ("lds", "sts", "bar")):
            raise RuntimeError(f"{batch}: candidate is not warp-local")
        if candidate["launch_shared_bytes"] != 0:
            raise RuntimeError(f"{batch}: candidate launch shared is nonzero")
        if superseded["launch_shared_bytes"] == 0:
            raise RuntimeError(f"{batch}: superseded v3 defect not reproduced")
        if not all(superseded[key] > 0 for key in ("lds", "sts", "bar")):
            raise RuntimeError(f"{batch}: superseded v3 barriers are missing")
        if candidate["encoded_sass_instructions"] > superseded[
            "encoded_sass_instructions"
        ]:
            raise RuntimeError(f"{batch}: candidate raises encoded SASS")
        if candidate["logical_work"]["kernel_launches_per_event"] != 1:
            raise RuntimeError(f"{batch}: guard launch count changed")
    print(
        json.dumps(
            {
                "schema": "fr13.fixed32.cfwd_ownerpath_warp32.sm121a.verify.v1",
                "status": "PASS",
                "builds_verified": 12,
                "fresh_cache_byte_identity": True,
                "gpu_execution": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
