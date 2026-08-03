#!/usr/bin/env python3
"""Build the default-off K64 FP8 draft-head plus mapped-top3 CUDA op."""

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
SOURCE = REPO / "csrc" / "fr13_dfwd_k64_fp8_mapped_top3.cu"
EXPECTED_TORCH = "2.11.0+cu130"
EXPECTED_CUDA = "13.0"
EXPECTED_ARCH = "12.1a"
CUDA_PACKAGE_INCLUDE = Path(
    "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
)

VOCAB = 65_536
HIDDEN = 5_120
GROUP = 128
GROUPS = HIDDEN // GROUP
PARTIALS = VOCAB // GROUP
TOPK = 3
CALLS_PER_PHYSICAL32_EVENT = 5
FP8_WEIGHT_BYTES = VOCAB * HIDDEN
FP32_WEIGHT_SCALE_ELEMENTS = (VOCAB // GROUP) * GROUPS
FP32_WEIGHT_SCALE_BYTES = FP32_WEIGHT_SCALE_ELEMENTS * 4


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


def physical32_work_model(batch: int) -> dict[str, int]:
    if batch not in (1, 4):
        raise ValueError("the closed work model serves only B1 and B4")
    macs_per_head = batch * VOCAB * HIDDEN
    full_logit_roundtrip_per_head = batch * VOCAB * 2 * 2
    partial_one_way_per_head = batch * PARTIALS * TOPK * (2 + 4)
    partial_roundtrip_per_head = partial_one_way_per_head * 2
    activation_q_requests_per_head = batch * HIDDEN * PARTIALS
    activation_scale_requests_per_head = batch * GROUPS * 4 * PARTIALS
    id_map_reads_per_head = batch * TOPK * 8
    output_writes_per_head = batch * (8 + TOPK * 8 + TOPK * 2)
    requested_bytes_per_head = (
        FP8_WEIGHT_BYTES
        + FP32_WEIGHT_SCALE_BYTES
        + activation_q_requests_per_head
        + activation_scale_requests_per_head
        + partial_roundtrip_per_head
        + id_map_reads_per_head
        + output_writes_per_head
    )
    return {
        "served_batch": batch,
        "physical_tree_nodes": 32,
        "head_calls_per_event": CALLS_PER_PHYSICAL32_EVENT,
        "macs_per_head": macs_per_head,
        "flops_per_head": macs_per_head * 2,
        "macs_per_event": macs_per_head * CALLS_PER_PHYSICAL32_EVENT,
        "flops_per_event": macs_per_head * 2 * CALLS_PER_PHYSICAL32_EVENT,
        "qweight_bytes_per_head": FP8_WEIGHT_BYTES,
        "qweight_bytes_per_event": (
            FP8_WEIGHT_BYTES * CALLS_PER_PHYSICAL32_EVENT
        ),
        "fp32_weight_scale_bytes_per_head": FP32_WEIGHT_SCALE_BYTES,
        "fp32_weight_scale_elements_per_head": FP32_WEIGHT_SCALE_ELEMENTS,
        "fp32_weight_scale_batch_multiplier": 1,
        "activation_q_requested_bytes_per_head": (
            activation_q_requests_per_head
        ),
        "activation_scale_requested_bytes_per_head": (
            activation_scale_requests_per_head
        ),
        "removed_full_bf16_logit_write_plus_read_per_head": (
            full_logit_roundtrip_per_head
        ),
        "partial_bf16_i32_write_plus_read_per_head": (
            partial_roundtrip_per_head
        ),
        "net_intermediate_bytes_removed_per_head": (
            full_logit_roundtrip_per_head - partial_roundtrip_per_head
        ),
        "net_intermediate_bytes_removed_per_event": (
            (full_logit_roundtrip_per_head - partial_roundtrip_per_head)
            * CALLS_PER_PHYSICAL32_EVENT
        ),
        "id_map_reads_per_head": id_map_reads_per_head,
        "final_output_writes_per_head": output_writes_per_head,
        "candidate_requested_bytes_per_head": requested_bytes_per_head,
        "candidate_requested_bytes_per_event": (
            requested_bytes_per_head * CALLS_PER_PHYSICAL32_EVENT
        ),
        "partial_blocks_per_head": PARTIALS,
        "kernel_launches_per_head": 2,
    }


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
            name="fr13_dfwd_k64_fp8_mapped_top3_sm121a",
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

    if not hasattr(torch.ops.fr13_dfwd_fp8_top3, "mapped_top3_out"):
        raise RuntimeError("built library did not register the FR13 CUDA op")

    payload: dict[str, object] = {
        "schema": "fr13.fixed32.dfwd_k64_fp8_mapped_top3_sm121a_build.v1",
        "status": "BUILT_UNQUALIFIED",
        "performance_measurement": False,
        "numerical_equality_claim": False,
        "real_task_correctness": False,
        "production_default_enabled": False,
        "runtime_integration_present": False,
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
            "head_input": "float8_e4m3fn[B,5120] contiguous, B in {1,4}",
            "qweight": "float8_e4m3fn[65536,5120] contiguous",
            "activation_scale": "FP32[B,40] stride (1,B)",
            "weight_scale": "FP32[512,40] contiguous",
            "id_map": "int64[65536] contiguous",
            "workspace": [
                "BF16[B,512,3] partial scores",
                "int32[B,512,3] partial subset IDs",
            ],
            "outputs": [
                "int64[B] mapped spine",
                "int64[B,3] mapped top3 IDs",
                "BF16[B,3] top3 scores",
            ],
            "stage1_grid": [PARTIALS, 1, 1],
            "stage2_grid": ["B", 1, 1],
            "block": [256, 1, 1],
            "selection": (
                "BF16-rounded score descending, NaN first, lower subset "
                "index tie-break"
            ),
            "mapping": "map only after subset top3 order is final",
            "b4_weight_reuse": (
                "each qweight byte and tile weight scale is loaded once and "
                "reused across four rows"
            ),
            "restrict_overlap_guard": (
                "all ten dense tensor byte ranges must be pairwise disjoint"
            ),
            "full_logits_materialized": False,
        },
        "physical32_work_model": {
            "B1": physical32_work_model(1),
            "B4": physical32_work_model(4),
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
