#!/usr/bin/env python3
"""Offline SM121 build and spill audit for fixed32 GDN coefficient staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lumo_flywheel_serving import fr10_gdn_tree_kernel as kernel


_POINTER_SIGNATURES = {
    "q": "*bf16",
    "k": "*bf16",
    "v": "*bf16",
    "g": "*fp32",
    "beta": "*fp32",
    "raw_a": "*bf16",
    "raw_b": "*bf16",
    "A_log": "*fp32",
    "dt_bias": "*fp32",
    "h0": "*fp32",
    "h0_indices": "*i64",
    "h0_num_accepted_tokens": "*i32",
    "invocation_counter": "*i32",
    "path_nodes": "*i32",
    "path_parent": "*i32",
    "path_parent_slots": "*i32",
    "path_lengths": "*i32",
    "state_export": "*fp32",
    "export_mask": "*i32",
    "out": "*bf16",
    "ring_k": "*bf16",
    "ring_v": "*bf16",
    "ring_a": "*bf16",
    "ring_b": "*bf16",
    "flags_ptr": "*i32",
}

_COMMON_CONSTANTS = {
    "N_ACTUAL": 32,
    "NUM_KH": 16,
    "NUM_VH": 48,
    "DIM_K": 128,
    "DIM_V": 128,
    "BLOCK_V": 8,
    "OUTPUT_SCALE": 128**-0.5,
    "H0_INDEX_ROW": 0,
    "H0_BATCH_INDEX": 0,
    "H0_BANK_STRIDE": 48 * 128 * 128,
    "H0_USE_ACCEPTED_COLUMN": False,
    "RAW_GATING": True,
    "SCAN_ALIGN": False,
    "COEFF_ROW_ELEMS": 48 * 128 * 128,
    "Q_OFFSET": 0,
    "K_OFFSET": 32 * 16 * 128,
    "DECAY_OFFSET": 2 * 32 * 16 * 128,
    "BETA_OFFSET": 2 * 32 * 16 * 128 + 32 * 48,
    "RING_EXPORT": True,
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _command(*arguments: str) -> str:
    completed = subprocess.run(
        list(arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stdout + completed.stderr).strip()


def _signature(jit_function) -> dict[str, str]:
    return {
        parameter.name: _POINTER_SIGNATURES[parameter.name]
        for parameter in jit_function.params
        if not parameter.is_constexpr
    }


def _specializations() -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    for batch, jit_function in (
        (1, kernel._tree_gdn_path_kernel),
        (4, kernel._tree_gdn_path_kernel_fixed32_batch),
    ):
        batch_constants = (
            {
                "H0_IS_BANK": True,
                "H0_BATCH_INDEX": 0,
            }
            if batch == 1
            else {
                "H0_INDEX_BATCH_STRIDE": 1,
                "H0_ACCEPTED_BATCH_STRIDE": 1,
                "NUM_PATHS": 0,
                "BATCH_SIZE": 4,
                "EXPORT_SLOTS": 5,
            }
        )
        for candidate in (False, True):
            for level, max_path_len, num_paths in ((0, 5, 1), (1, 7, 11)):
                constants = {
                    **_COMMON_CONSTANTS,
                    **batch_constants,
                    "USE_QK_L2NORM_IN_KERNEL": not (candidate and level == 1),
                    "COUNT_INVOCATION": level == 0,
                    "MAX_PATH_LEN": max_path_len,
                    "PRECOMPUTE_LEVEL1": candidate and level == 0,
                    "LOAD_PRECOMPUTED": candidate and level == 1,
                    "COEFF_ROW_START": 31 if batch == 1 else 28,
                    "STATE_SOURCE": 1 if level == 0 else 2,
                    "EXPORT_MODE": 1 if level == 0 else 2,
                    "FLAGS_EXPORT": level == 0,
                    "FLAGS_ROWS": batch,
                }
                if batch == 4:
                    constants["NUM_PATHS"] = num_paths
                variants.append(
                    {
                        "name": (
                            f"b{batch}_{'candidate' if candidate else 'stock'}"
                            f"_level{level}"
                        ),
                        "batch": batch,
                        "candidate": candidate,
                        "level": level,
                        "function": jit_function,
                        "constants": constants,
                    }
                )
    return variants


def _compile_one(output: Path, variant: dict[str, object]) -> dict[str, object]:
    name = str(variant["name"])
    jit_function = variant["function"]
    constants = variant["constants"]
    assert isinstance(constants, dict)
    source = ASTSource(
        jit_function,
        signature=_signature(jit_function),
        constexprs=constants,
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("cuda", 121, 32),
        options={"num_warps": 8},
    )

    artifact_hashes: dict[str, str] = {}
    for extension in ("cubin", "ptx", "ttgir"):
        payload = compiled.asm[extension]
        raw = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        path = output / f"{name}.{extension}"
        path.write_bytes(raw)
        artifact_hashes[extension] = _sha256(raw)

    cubin = output / f"{name}.cubin"
    resources = _command("cuobjdump", "--dump-resource-usage", str(cubin))
    sass = _command("nvdisasm", "--print-code", str(cubin))
    (output / f"{name}.resources.txt").write_text(
        resources + "\n", encoding="ascii"
    )
    (output / f"{name}.sass.txt").write_text(sass + "\n", encoding="ascii")

    local_bytes = [int(value) for value in re.findall(r"\bLOCAL:(\d+)", resources)]
    spill_instructions = re.findall(r"^\s*(?:LDL|STL)(?:\.[A-Z0-9]+)*\b", sass, re.MULTILINE)
    if any(local_bytes) or spill_instructions:
        raise RuntimeError(
            f"{name} spills: local_bytes={local_bytes} "
            f"spill_instructions={len(spill_instructions)}"
        )

    metadata = compiled.metadata._asdict()
    target = metadata["target"]
    metadata["target"] = {
        "backend": target.backend,
        "arch": target.arch,
        "warp_size": target.warp_size,
    }
    return {
        "name": name,
        "batch": variant["batch"],
        "candidate": variant["candidate"],
        "level": variant["level"],
        "kernel": compiled.name,
        "compile_hash": compiled.metadata.hash,
        "metadata": metadata,
        "artifact_sha256": artifact_hashes,
        "local_bytes": local_bytes,
        "spill_instruction_count": len(spill_instructions),
        "resources": resources.splitlines(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    source_path = Path(kernel.__file__).resolve()
    records = [_compile_one(output, variant) for variant in _specializations()]
    manifest = {
        "schema": "fr13.fixed32.gdn_level0_coeff.sm121_build.v1",
        "status": "pass_zero_spill",
        "target": "sm_121",
        "triton_version": triton.__version__,
        "ptxas_version": _command("ptxas", "--version").splitlines(),
        "cuobjdump_version": _command("cuobjdump", "--version").splitlines(),
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path.read_bytes()),
        "specializations": records,
    }
    manifest_path = output / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    checksums = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.name == "SHA256SUMS":
            continue
        checksums.append(f"{_sha256(path.read_bytes())}  {path.name}")
    (output / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="ascii"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
