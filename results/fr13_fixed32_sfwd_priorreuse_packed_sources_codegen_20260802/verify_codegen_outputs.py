#!/usr/bin/env python3
"""Verify the reduced packed-source SFWD offline-codegen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


EXPECTED = {
    8: {
        "registers": 40,
        "cubin_bytes": 49432,
        "cubin_sha256": (
            "03741e95a5bd41d8fba62108d5950e2e863b7bca41f316783c919eb4d67c4840"
        ),
        "ptx_sha256": (
            "7bfda07b2ab396f17fa183d6b6590283e0a4fd4bd7c51b4e46380d353c131fb6"
        ),
        "sass_sha256": (
            "72dda02a4bd8b96f38ef6b47d0afdc2c45a5e1e8b268a67cd874200c96719c0f"
        ),
        "encoded_sass_instructions": 688,
        "static_sass_instructions": 677,
        "ldg": 40,
        "stg": 20,
        "nop": 8,
        "threads": 256,
        "compile_hashes": {
            1: "c873dc845862918edef90d22ea37a68a88dfcd23d894f76155482a4a35c3d315",
            4: "e21926afa931ecf1708285b1abfe92a727f9b9417237e3705a70c28c6b07afa1",
        },
    },
    16: {
        "registers": 44,
        "cubin_bytes": 33824,
        "cubin_sha256": (
            "93a0d2b9c33a744c7cb8297bb28b9c2464dbd12d61b8f99641a7c1cde4aab913"
        ),
        "ptx_sha256": (
            "3cc87fec7069e47aacf9574af9b800fa02e52edd0fccb00628b29cf789366ac6"
        ),
        "sass_sha256": (
            "59c815d350af499e20809322ce805a7cdca537c8fa2058e89b30b5382d523ff2"
        ),
        "encoded_sass_instructions": 416,
        "static_sass_instructions": 405,
        "ldg": 24,
        "stg": 12,
        "nop": 8,
        "threads": 512,
        "compile_hashes": {
            1: "ad17559fa37dac1619dc331960ec70c82a2a491f8a867a62b514cba03673bbe6",
            4: "8227e1c01eef3f90e1090b8a459bc6c684a699e6a8ccb221f2c4ad5f54410862",
        },
    },
}
SOURCE_FUNCTION_SHA256 = (
    "600a7558e19fe8908470c08d9ef2369de65fec2a891e3d5492aca318a72adb38"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(argv: list[str]) -> bytes:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def normalized(summary: dict[str, object]) -> dict[str, object]:
    result = dict(summary)
    for key in ("batch", "compile_hash", "ctas_per_launch"):
        result.pop(key)
    return result


def verify_tree(root: Path, warps: int) -> dict[int, dict[str, bytes]]:
    expected = EXPECTED[warps]
    variants: dict[int, dict[str, bytes]] = {}
    summaries: dict[int, dict[str, object]] = {}
    for batch in (1, 4):
        variant = root / f"b{batch}"
        summary = json.loads((variant / "summary.json").read_text())
        summaries[batch] = summary
        require(summary["batch"] == batch, f"w{warps} b{batch} batch")
        require(summary["num_warps"] == warps, f"w{warps} b{batch} warps")
        require(summary["rows_per_program"] == 32, f"w{warps} rows")
        require(summary["block_c"] == 64, f"w{warps} block")
        require(summary["ctas_per_request"] == 160, f"w{warps} CTAs/request")
        require(
            summary["ctas_per_launch"] == 160 * batch,
            f"w{warps} b{batch} CTAs/launch",
        )
        require(
            summary["compile_hash"] == expected["compile_hashes"][batch],
            f"w{warps} b{batch} compile hash",
        )
        for key in (
            "registers",
            "cubin_bytes",
            "cubin_sha256",
            "ptx_sha256",
            "sass_sha256",
            "encoded_sass_instructions",
            "static_sass_instructions",
            "ldg",
            "stg",
            "nop",
        ):
            require(summary[key] == expected[key], f"w{warps} b{batch} {key}")
        for key in (
            "stack_bytes",
            "local_bytes",
            "elf_shared_bytes",
            "launch_shared_bytes",
            "ldl",
            "stl",
            "lds",
            "sts",
            "calls",
            "bar",
        ):
            require(summary[key] == 0, f"w{warps} b{batch} {key}")
        require(
            summary["source_function_sha256"] == SOURCE_FUNCTION_SHA256,
            f"w{warps} b{batch} function source",
        )

        blobs = {
            name: (variant / f"kernel.{name}").read_bytes()
            for name in ("cubin", "ptx", "sass")
        }
        for name, blob in blobs.items():
            require(sha256(blob) == summary[f"{name}_sha256"], f"w{warps} {name}")
        cubin_path = str(variant / "kernel.cubin")
        require(
            run(["/usr/local/cuda/bin/nvdisasm", "-c", cubin_path])
            == blobs["sass"],
            f"w{warps} b{batch} fresh disassembly",
        )
        ptx = blobs["ptx"].decode()
        require(".target sm_121a" in ptx, f"w{warps} b{batch} target")
        require(
            re.search(
                rf"(?m)^\s*\.reqntid\s+{expected['threads']}\s*$",
                ptx,
            )
            is not None,
            f"w{warps} b{batch} threads",
        )
        variants[batch] = blobs

    require(
        normalized(summaries[1]) == normalized(summaries[4]),
        f"w{warps} B1/B4 non-launch identity",
    )
    for name in ("cubin", "ptx", "sass"):
        require(
            variants[1][name] == variants[4][name],
            f"w{warps} B1/B4 {name} identity",
        )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    for warps in (8, 16):
        parser.add_argument(f"--w{warps}-primary", type=Path, required=True)
        parser.add_argument(f"--w{warps}-rebuild", type=Path, required=True)
    args = parser.parse_args()

    for warps in (8, 16):
        primary = verify_tree(getattr(args, f"w{warps}_primary"), warps)
        rebuild = verify_tree(getattr(args, f"w{warps}_rebuild"), warps)
        for batch in (1, 4):
            for name in ("cubin", "ptx", "sass"):
                require(
                    primary[batch][name] == rebuild[batch][name],
                    f"w{warps} b{batch} fresh-cache {name} identity",
                )

    print(
        json.dumps(
            {
                "status": "pass",
                "target": "sm_121a",
                "schedules": ["row32_c64_w8", "row32_c64_w16"],
                "batches": [1, 4],
                "b1_b4_binary_identity": True,
                "fresh_cache_binary_identity": True,
                "zero_spills_stack_local_calls": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
