#!/usr/bin/env python3
"""Build the default-off fixed32 verifier-head N256/K32/stage3 CUDA op."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "csrc" / "fr13_bf16_verifier_head_m32_n256k32s3_sm121a.cu"
EXPECTED_TORCH = "2.11.0+cu130"
EXPECTED_CUDA = "13.0"
EXPECTED_ARCH = "12.1a"
EXPECTED_CUTLASS_COMMIT = "da5e086dab31d63815acafdac9a9c5893b1c69e2"
CUDA_PACKAGE_INCLUDE = Path(sysconfig.get_path("purelib")) / "nvidia/cu13/include"


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
        ["nvcc", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if f"release {EXPECTED_CUDA}" not in cuda_version:
        raise RuntimeError("pinned build requires CUDA 13.0 nvcc")
    if not (CUDA_PACKAGE_INCLUDE / "cusparse.h").is_file():
        raise RuntimeError("pinned CUDA 13 package headers are unavailable")
    cutlass_root, cutlass_header_sha256 = require_cutlass(cutlass_root)

    output = output.resolve()
    build_dir = build_dir.resolve()
    attestation = attestation.resolve()
    for path in (output, build_dir, attestation):
        if path == SOURCE:
            raise ValueError("build outputs must not replace the CUDA source")
    output.parent.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    os.environ["TORCH_CUDA_ARCH_LIST"] = EXPECTED_ARCH
    built = Path(
        load(
            name="fr13_bf16_verifier_head_m32_n256k32s3_sm121a",
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

    if not hasattr(torch.ops.fr13_verifier_head, "bf16_m32_n256k32s3_out"):
        raise RuntimeError("built library did not register the verifier-head op")

    payload: dict[str, object] = {
        "schema": "fr13.fixed32.bf16_verifier_head_m32_n256k32s3_sm121a_build.v1",
        "status": "BUILT_UNQUALIFIED",
        "performance_measurement": False,
        "performance_claim": False,
        "byte_equality_claim": False,
        "verifier_distribution_claim": False,
        "real_task_correctness": False,
        "production_default_enabled": False,
        "torch_version": torch.__version__,
        "cuda_release": EXPECTED_CUDA,
        "cuda_arch": EXPECTED_ARCH,
        "cutlass": {
            "commit": EXPECTED_CUTLASS_COMMIT,
            "cutlass_h_sha256": cutlass_header_sha256,
        },
        "source": {
            "path": str(SOURCE.relative_to(REPO)),
            "sha256": sha256_file(SOURCE),
        },
        "binary": {
            "path": str(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "mode": "0555",
        },
        "kernel_contract": {
            "problem_mnk": [32, 248320, 5120],
            "logical_output": "BF16[32,248320] contiguous",
            "weight": "BF16[248320,5120] contiguous",
            "hidden": "BF16[32,5120] contiguous",
            "threadblock_mnk": [32, 256, 32],
            "warp_mnk": [32, 64, 32],
            "instruction_mnk": [16, 8, 16],
            "stages": 3,
            "split_k_slices": 1,
            "workspace_bytes": 0,
            "dynamic_shared_storage_bytes": 55296,
            "logical_grid_mn": [1, 970],
            "logical_grid_ctas": 970,
            "weight_bytes_per_launch": 2542796800,
            "full_vocabulary_preserved": True,
            "input_or_weight_quantized": False,
        },
        "static_baseline_delta": {
            "baseline_threadblock_mnk": [32, 128, 64],
            "baseline_stages": 3,
            "baseline_logical_grid_ctas": 1940,
            "logical_grid_ctas_delta": -970,
            "baseline_dynamic_shared_storage_bytes": 61440,
            "dynamic_shared_storage_bytes_delta": -6144,
            "baseline_ptxas_registers": 158,
            "ptxas_registers_delta": -30,
        },
        "qualification_required": [
            "one real SWE-Verified B1 shadow task with every BF16 element compared",
            "zero raw BF16 mismatches and unchanged served incumbent logits",
            "frozen-source exact4 B1 stock-versus-candidate full-step timing",
            "only after B1 passes, repeat byte gate and exact4 timing on B4",
        ],
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
