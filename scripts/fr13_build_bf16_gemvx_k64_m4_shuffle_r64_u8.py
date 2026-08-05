#!/usr/bin/env python3
"""Build the default-off fixed32 B4 K64 M4 R64-U8 CUDA op."""

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
SOURCE = REPO / "csrc" / "fr13_bf16_gemvx_k64_m4_shuffle_r64_u8.cu"
SOURCE_SHA256 = "a52361be1c9052a46509cc230ea320c4beb6d15f261327edc835d8da3ae00d9e"
EXPECTED_TORCH = "2.11.0+cu130"
EXPECTED_CUDA = "13.0"
EXPECTED_ARCH = "12.1a"
EXTENSION_NAME = "fr13_bf16_k64_m4_r64_u8_sm121a"
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


def newly_loaded_library(
    build_dir: Path, loaded_before: set[str], loaded_after: set[str]
) -> Path:
    candidates = sorted(
        candidate
        for raw_path in loaded_after - loaded_before
        if (candidate := Path(raw_path).resolve()).parent == build_dir.resolve()
        and candidate.is_file()
        and candidate.suffix == ".so"
        and (
            candidate.name == f"{EXTENSION_NAME}.so"
            or (
                candidate.stem.startswith(f"{EXTENSION_NAME}_v")
                and candidate.stem.removeprefix(f"{EXTENSION_NAME}_v").isdigit()
            )
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "pinned build did not register exactly one newly loaded FR13 B4 library"
        )
    return candidates[0]


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
    observed_source_sha256 = sha256_file(SOURCE)
    if observed_source_sha256 != SOURCE_SHA256:
        raise RuntimeError(
            "FR13 BF16 K64 M4 R64-U8 source drift: "
            f"{observed_source_sha256} != {SOURCE_SHA256}"
        )

    output = output.resolve()
    build_dir = build_dir.resolve()
    attestation = attestation.resolve()
    if output == SOURCE or attestation == SOURCE:
        raise ValueError("build outputs must not replace the CUDA source")
    if output == attestation:
        raise ValueError("binary output and build attestation must be distinct")
    output.parent.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    python_bin_dir = str(Path(sys.executable).parent)
    os.environ["PATH"] = python_bin_dir + os.pathsep + os.environ["PATH"]
    os.environ["TORCH_CUDA_ARCH_LIST"] = EXPECTED_ARCH
    loaded_before = set(torch.ops.loaded_libraries)
    load(
        name=EXTENSION_NAME,
        sources=[str(SOURCE)],
        build_directory=str(build_dir),
        extra_cflags=["-O3", f"-I{CUDA_PACKAGE_INCLUDE}"],
        extra_cuda_cflags=[
            "-O3",
            f"-I{CUDA_PACKAGE_INCLUDE}",
            "--fmad=true",
            "--frandom-seed=fr13_bf16_k64_m4_r64_u8",
            "--expt-relaxed-constexpr",
            "--threads=1",
        ],
        is_python_module=False,
        verbose=True,
    )
    built = newly_loaded_library(
        build_dir, loaded_before, set(torch.ops.loaded_libraries)
    )
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    shutil.copyfile(built, temporary)
    temporary.chmod(0o555)
    temporary.replace(output)

    if not hasattr(torch.ops.fr13_bf16_k64_head, "gemvx_m4_shuffle_r64_u8_out"):
        raise RuntimeError("built library did not register the FR13 B4 CUDA op")

    payload: dict[str, object] = {
        "schema": "fr13.fixed32.dfwd_k64_m4_r64_u8_sm121a_build.v1",
        "status": "BUILT_UNQUALIFIED",
        "performance_measurement": False,
        "byte_equality_claim": False,
        "real_task_correctness": False,
        "production_default_enabled": False,
        "runtime_wired": False,
        "gpu_runtime_used": False,
        "torch_version": torch.__version__,
        "cuda_release": EXPECTED_CUDA,
        "cuda_arch": EXPECTED_ARCH,
        "source": {
            "path": str(SOURCE.relative_to(REPO)),
            "sha256": observed_source_sha256,
        },
        "binary": {
            "path": recorded_path(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "mode": "0555",
        },
        "kernel_contract": {
            "batch_scope": "B4_exact",
            "grid": [1024, 1, 1],
            "block": [16, 64, 1],
            "threads_per_cta": 1024,
            "rows_per_cta": 64,
            "input": "BF16[4,5120] contiguous",
            "weight": "BF16[65536,5120] contiguous",
            "output": "BF16[4,65536] contiguous",
            "lane_products_per_request": 320,
            "unroll_steps": 8,
            "independent_accumulators": 4,
            "weight_reuse_batch": 4,
            "reduction_strides": [8, 4, 2, 1],
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
