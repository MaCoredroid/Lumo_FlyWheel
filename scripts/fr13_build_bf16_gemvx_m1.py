#!/usr/bin/env python3
"""Build and attest the default-off FR13 BF16 full-head M=1 CUDA op."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "csrc" / "fr13_bf16_gemvx_m1.cu"
EXPECTED_TORCH = "2.10.0+cu130"
EXPECTED_CUDA = "13.0"
EXPECTED_ARCH = "12.1a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def recorded_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def build(output: Path, build_dir: Path, attestation: Path) -> dict[str, object]:
    import torch
    from torch.utils.cpp_extension import load

    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(
            f"pinned build requires torch {EXPECTED_TORCH}, got {torch.__version__}"
        )
    cuda_version = subprocess.run(
        ["nvcc", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if f"release {EXPECTED_CUDA}" not in cuda_version:
        raise RuntimeError("pinned build requires CUDA 13.0 nvcc")

    output = output.resolve()
    build_dir = build_dir.resolve()
    attestation = attestation.resolve()
    if output == SOURCE or attestation == SOURCE:
        raise ValueError("build outputs must not replace the CUDA source")
    output.parent.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    python_bin_dir = str(Path(sys.executable).parent)
    os.environ["PATH"] = python_bin_dir + os.pathsep + os.environ["PATH"]
    os.environ["TORCH_CUDA_ARCH_LIST"] = EXPECTED_ARCH
    built = Path(
        load(
            name="fr13_bf16_gemvx_m1",
            sources=[str(SOURCE)],
            build_directory=str(build_dir),
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--fmad=true",
                "--expt-relaxed-constexpr",
                "--threads=1",
            ],
            is_python_module=False,
            verbose=True,
        )
    )
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    shutil.copyfile(built, temporary)
    temporary.chmod(0o555)
    temporary.replace(output)

    if not hasattr(torch.ops.fr13_bf16_head, "gemvx_m1_out"):
        raise RuntimeError("built library did not register the FR13 CUDA op")

    payload: dict[str, object] = {
        "schema": "fr13.fixed32.bf16_gemvx_m1_build.v1",
        "status": "BUILT_UNQUALIFIED",
        "performance_measurement": False,
        "byte_equality_claim": False,
        "production_default_enabled": False,
        "torch_version": torch.__version__,
        "cuda_release": EXPECTED_CUDA,
        "cuda_arch": EXPECTED_ARCH,
        "source": {
            "path": str(SOURCE.relative_to(REPO)),
            "sha256": sha256_file(SOURCE),
        },
        "binary": {
            "path": recorded_path(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "mode": "0555",
        },
        "kernel_contract": {
            "grid": [31040, 1, 1],
            "block": [16, 8, 1],
            "dynamic_shared_bytes": 544,
            "shared_row_stride_floats": 17,
            "gemv_mnk": [1, 248320, 5120],
            "k_partition_lanes": 16,
            "lane_k_iterations": 320,
            "reduction_strides": [8, 4, 2, 1],
            "accumulator": "fp32 positive zero",
            "multiply_accumulate": "__fmaf_rn dependent scalar chain",
            "reduction": "__fadd_rn shared-memory tree",
            "epilogue": "__fmaf_rn(1.0f, reduced_sum, 0.0f)",
            "output": "__float2bfloat16_rn",
        },
    }
    atomic_json(attestation, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.output, args.build_dir, args.attestation),
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
