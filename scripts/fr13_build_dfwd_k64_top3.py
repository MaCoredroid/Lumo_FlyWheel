#!/usr/bin/env python3
"""Build the default-off fixed32 B1 K64 mapped-top3 CUDA op."""

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
SOURCE = REPO / "csrc" / "fr13_dfwd_k64_top3.cu"
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
            name="fr13_dfwd_k64_mapped_top3_sm121a",
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

    if not hasattr(torch.ops.fr13_dfwd_top3, "mapped_top3_out"):
        raise RuntimeError("built library did not register the FR13 CUDA op")

    payload: dict[str, object] = {
        "schema": "fr13.fixed32.dfwd_k64_mapped_top3_sm121a_build.v1",
        "status": "BUILT_UNQUALIFIED",
        "performance_measurement": False,
        "byte_equality_claim": False,
        "real_task_correctness": False,
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
            "grid": [1, 1, 1],
            "block": [256, 1, 1],
            "input": "BF16[1,65536] contiguous",
            "id_map": "int64[65536] contiguous",
            "outputs": ["int64[1] spine", "int64[1,3] mapped top3"],
            "input_logit_bytes_per_launch": 131072,
            "launches_per_head": 1,
            "heads_per_fixed32_event": 5,
            "minimum_redundant_argmax_logit_bytes_eliminated_per_event": 655360,
            "stock_reduction_launches_per_head": {
                "argmax": 1,
                "topk_block_digit_counts": 2,
                "topk_digit_cumsum": 2,
                "topk_scan_by_key": 2,
                "topk_within_k_counts": 2,
                "topk_gather": 1,
            },
            "candidate_reduction_launches_per_head": 1,
            "minimum_reduction_launches_eliminated_per_event": 45,
            "selection": "top3 descending, NaN first, lower index tie-break",
            "mapping": "top3 subset indices mapped inside the same kernel",
            "graph_outputs": "direct persistent spine/top3 writes",
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
