#!/usr/bin/env python3
"""Verify two fresh-cache fixed32 GDN SM121a builds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


VARIANTS = (
    "incumbent_index_scaled",
    "candidate_prescaled_path_base",
)
BATCHES = ("b1", "b4")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def verify_build(root: Path, summary: dict, label: str, batch: str) -> None:
    build = summary["variants"][label]["builds"][batch]
    directory = root / label / batch
    cubin_path = directory / "kernel.cubin"
    cubin = cubin_path.read_bytes()
    ptx = (directory / "kernel.ptx").read_bytes()
    sass = subprocess.run(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(cubin_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
    if sha256(cubin) != build["cubin_sha256"]:
        raise RuntimeError(f"{label}/{batch}: cubin hash drift")
    if sha256(ptx) != build["ptx_sha256"]:
        raise RuntimeError(f"{label}/{batch}: PTX hash drift")
    if sha256(sass) != build["sass_sha256"]:
        raise RuntimeError(f"{label}/{batch}: SASS hash drift")
    for field in ("stack_bytes", "local_bytes", "ldl", "stl", "calls"):
        if build[field] != 0:
            raise RuntimeError(f"{label}/{batch}: {field}={build[field]}")
    if build["registers"] > 99:
        raise RuntimeError(f"{label}/{batch}: registers={build['registers']}")
    if build["backend_producer"]["target"] != "sm_121a":
        raise RuntimeError(f"{label}/{batch}: target drift")


def verify_pair(summary: dict, batch: str) -> None:
    incumbent = summary["variants"]["incumbent_index_scaled"]["builds"][batch]
    candidate = summary["variants"]["candidate_prescaled_path_base"][
        "builds"
    ][batch]
    for field in (
        "registers",
        "stack_bytes",
        "local_bytes",
        "launch_shared_bytes",
        "elf_shared_bytes",
        "static_sass_instructions",
        "sass_addressed_lines",
        "ldg",
        "stg",
        "ldl",
        "stl",
        "calls",
    ):
        if candidate[field] > incumbent[field]:
            raise RuntimeError(
                f"{batch}: candidate {field} regressed "
                f"{candidate[field]} > {incumbent[field]}"
            )
    if candidate["sass_addressed_lines"] >= incumbent["sass_addressed_lines"]:
        raise RuntimeError(f"{batch}: candidate did not reduce SASS")
    incumbent_work = incumbent["logical_work"]
    candidate_work = candidate["logical_work"]
    if incumbent_work["path_base_scale_operations_per_program"] != 11:
        raise RuntimeError(f"{batch}: incumbent work accounting drift")
    if candidate_work["path_base_scale_operations_per_program"] != 0:
        raise RuntimeError(f"{batch}: candidate retained path-base scaling")
    for source_hash in (
        "node_source_sha256",
        "recurrence_source_sha256",
    ):
        if incumbent[source_hash] != candidate[source_hash]:
            raise RuntimeError(f"{batch}: {source_hash} differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--rebuild", type=Path, required=True)
    args = parser.parse_args()
    primary = load(args.primary / "summary.json")
    rebuild = load(args.rebuild / "summary.json")
    if primary != rebuild:
        raise RuntimeError("fresh-cache summary differs")
    for root in (args.primary, args.rebuild):
        for label in VARIANTS:
            for batch in BATCHES:
                verify_build(root, primary, label, batch)
    for batch in BATCHES:
        verify_pair(primary, batch)
    print(
        json.dumps(
            {
                "status": "pass",
                "fresh_cache_outputs_identical": True,
                "builds_verified": 8,
                "target": "sm_121a",
                "register_ceiling": 99,
                "resource_regressions": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
