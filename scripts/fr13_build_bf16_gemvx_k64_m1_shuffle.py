#!/usr/bin/env python3
"""Build the default-off FR13 BF16 K64 M1 warp4 global-x op."""

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
SOURCE = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shuffle.cu"
EXPECTED_TORCH = "2.11.0+cu130"
EXPECTED_CUDA = "13.0"
EXPECTED_ARCH = "12.1a"
CUDA_PACKAGE_INCLUDE = Path(
    "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
)


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
    if not (CUDA_PACKAGE_INCLUDE / "cusparse.h").is_file():
        raise RuntimeError("pinned CUDA 13 package headers are unavailable")

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
            name="fr13_bf16_gemvx_k64_m1_warp4_globalx_pair8bits",
            sources=[str(SOURCE)],
            build_directory=str(build_dir),
            extra_cflags=["-O3", f"-I{CUDA_PACKAGE_INCLUDE}"],
            extra_cuda_cflags=[
                "-O3",
                f"-I{CUDA_PACKAGE_INCLUDE}",
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

    if not hasattr(
        torch.ops.fr13_bf16_k64_head,
        "gemvx_m1_warp4_globalx_pair8bits_out",
    ):
        raise RuntimeError("built library did not register the FR13 CUDA op")

    payload: dict[str, object] = {
        "schema": "fr13.fixed32.bf16_gemvx_k64_m1_warp4_globalx_pair8bits_build.v1",
        "status": "BUILT_UNQUALIFIED",
        "performance_measurement": False,
        "byte_equality_claim": False,
        "resource_claim": False,
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
            "grid": [2048, 1, 1],
            "block": [32, 8, 1],
            "dynamic_shared_bytes": 0,
            "static_shared_bytes": 0,
            "gemv_mnk": [1, 65536, 5120],
            "output_rows_per_cta": 32,
            "warps_per_cta": 8,
            "output_rows_per_warp": 4,
            "k_partition_lanes": 32,
            "elements_per_load": 8,
            "input_global_loads_per_cta": 5120,
            "lane_input_global_iterations": 20,
            "lane_weight_load_iterations": 80,
            "lane_fma_iterations": 640,
            "packed_unpack": "BF16 bits shifted/masked into exact FP32 bits",
            "reduction_strides": [16, 8, 4, 2, 1],
            "accumulator": "fp32 positive zero",
            "multiply_accumulate": "eight __fmaf_rn per packed BF16 octet",
            "reduction": "__fadd_rn width-32 shuffle tree",
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
