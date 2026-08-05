#!/usr/bin/env python3
"""Build the default-off fixed32 B1/B4 K64 tensor-core draft head."""

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
SOURCE = REPO / "csrc" / "fr13_bf16_gemm_k64_tc16x256x64_s2.cu"
SOURCE_SHA256 = "8c55f0c1b8dc18b37b0cf6f06b5a8c608a62868cb027019b63b28126fa622095"
EXPECTED_TORCH = "2.11.0+cu130"
EXPECTED_CUDA = "13.0"
EXPECTED_ARCH = "12.1a"
EXPECTED_CUTLASS_COMMIT = "da5e086dab31d63815acafdac9a9c5893b1c69e2"
CUDA_PACKAGE_INCLUDE = (
    Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "nvidia"
    / "cu13"
    / "include"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
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


def require_cutlass(cutlass_root: Path) -> tuple[Path, str]:
    if cutlass_root.is_symlink() or not cutlass_root.is_dir():
        raise ValueError("CUTLASS root must be a real directory")
    root = cutlass_root.resolve()
    header = root / "include" / "cutlass" / "cutlass.h"
    if header.is_symlink() or not header.is_file():
        raise ValueError("CUTLASS root does not contain a real cutlass.h")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_CUTLASS_COMMIT:
        raise RuntimeError(
            f"pinned build requires CUTLASS {EXPECTED_CUTLASS_COMMIT}, got {commit}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("pinned build requires a clean CUTLASS source tree")
    return root, sha256_file(header)


def traffic_model() -> dict[str, object]:
    weight_bytes = 65536 * 5120 * 2
    output_bytes = {1: 1 * 65536 * 2, 4: 4 * 65536 * 2}
    duplicated_input_bytes = {
        1: 256 * 1 * 5120 * 2,
        4: 256 * 4 * 5120 * 2,
    }
    return {
        "scope": "algorithmic global bytes; not measured DRAM traffic",
        "weight_bytes_per_call": weight_bytes,
        "b1": {
            "duplicated_input_read_bytes": duplicated_input_bytes[1],
            "global_read_bytes": weight_bytes + duplicated_input_bytes[1],
            "output_write_bytes": output_bytes[1],
            "total_bytes": weight_bytes + duplicated_input_bytes[1] + output_bytes[1],
        },
        "b4": {
            "duplicated_input_read_bytes": duplicated_input_bytes[4],
            "global_read_bytes": weight_bytes + duplicated_input_bytes[4],
            "output_write_bytes": output_bytes[4],
            "total_bytes": weight_bytes + duplicated_input_bytes[4] + output_bytes[4],
        },
    }


def compute_model() -> dict[str, object]:
    return {
        "tensor_core_executed_flops_per_call": 2 * 16 * 65536 * 5120,
        "useful_flops": {
            "b1": 2 * 1 * 65536 * 5120,
            "b4": 2 * 4 * 65536 * 5120,
        },
        "padding_factor": {"b1": 16, "b4": 4},
        "claim": "modeled only; no timing or achieved-throughput measurement",
    }


def build(
    output: Path,
    build_dir: Path,
    attestation: Path,
    cutlass_root: Path,
) -> dict[str, object]:
    import torch
    from torch.utils.cpp_extension import load

    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(
            f"pinned build requires torch {EXPECTED_TORCH}, got {torch.__version__}"
        )
    cuda_version = subprocess.run(
        ["nvcc", "--version"], check=True, capture_output=True, text=True
    ).stdout
    if f"release {EXPECTED_CUDA}" not in cuda_version:
        raise RuntimeError("pinned build requires CUDA 13.0 nvcc")
    if not (CUDA_PACKAGE_INCLUDE / "cusparse.h").is_file():
        raise RuntimeError("pinned CUDA 13 package headers are unavailable")
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("FR13 K64 tensor-head source identity drifted")
    cutlass_root, cutlass_header_sha256 = require_cutlass(cutlass_root)

    output = output.resolve()
    build_dir = build_dir.resolve()
    attestation = attestation.resolve()
    if any(path == SOURCE for path in (output, build_dir, attestation)):
        raise ValueError("build outputs must not replace the CUDA source")
    output.parent.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    os.environ["TORCH_CUDA_ARCH_LIST"] = EXPECTED_ARCH
    built = Path(
        load(
            name="fr13_bf16_gemm_k64_tc16x256x64_s2_sm121a",
            sources=[str(SOURCE)],
            build_directory=str(build_dir),
            extra_include_paths=[
                str(cutlass_root / "include"),
                str(cutlass_root / "tools" / "util" / "include"),
            ],
            extra_cflags=["-O3", f"-I{CUDA_PACKAGE_INCLUDE}"],
            extra_cuda_cflags=[
                "-O3",
                f"-I{CUDA_PACKAGE_INCLUDE}",
                "--fmad=true",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--frandom-seed=fr13_bf16_gemm_k64_tc16x256x64_s2",
                "--ptxas-options=-v",
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

    operations = torch.ops.fr13_bf16_k64_tc_head
    for operation in (
        "gemm_m1_tc16x256x64_s2_out",
        "gemm_m4_tc16x256x64_s2_out",
    ):
        if not hasattr(operations, operation):
            raise RuntimeError(f"built library did not register {operation}")

    payload: dict[str, object] = {
        "schema": "fr13.fixed32.dfwd_k64_tc16x256x64_s2_sm121a_build.v1",
        "status": "BUILT_UNQUALIFIED",
        "acceptance_valid": False,
        "performance_measurement": False,
        "real_task_correctness": False,
        "production_default_enabled": False,
        "runtime_wired": False,
        "gpu_runtime_used": False,
        "torch_version": torch.__version__,
        "cuda_release": EXPECTED_CUDA,
        "cuda_arch": EXPECTED_ARCH,
        "cutlass": {
            "commit": EXPECTED_CUTLASS_COMMIT,
            "cutlass_h_sha256": cutlass_header_sha256,
        },
        "source": {"path": str(SOURCE.relative_to(REPO)), "sha256": SOURCE_SHA256},
        "binary": {
            "path": str(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "mode": "0555",
        },
        "kernel_contract": {
            "batch_scopes": [1, 4],
            "problem_mnk": {"b1": [1, 65536, 5120], "b4": [4, 65536, 5120]},
            "threadblock_mnk": [16, 256, 64],
            "warp_mnk": [16, 64, 64],
            "instruction_mnk": [16, 8, 16],
            "stages": 2,
            "threads_per_cta": 128,
            "logical_grid_ctas": 256,
            "dynamic_shared_storage_bytes": 69632,
            "split_k_slices": 1,
            "workspace_bytes": 0,
            "input": "BF16[B,5120] contiguous, B in {1,4}",
            "weight": "BF16[65536,5120] contiguous",
            "output": "BF16[B,65536] contiguous",
            "proposal_only": True,
            "target_authority_changed": False,
        },
        "logical_global_traffic_model": traffic_model(),
        "compute_model": compute_model(),
    }
    atomic_json(attestation, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.output, args.build_dir, args.attestation, args.cutlass_root),
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
