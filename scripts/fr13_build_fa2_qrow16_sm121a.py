#!/usr/bin/env python3
"""Build the private qrow16 SM121a object at the spill-free RU3 boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


REGISTER_USAGE_LEVEL = 3
EXPECTED_REGISTERS = 216
QROW16_SOURCE = (
    "csrc/flash_attn/src/flash_fwd_fr13_qrow16_hdim256_bf16_sm80.cu"
)


class BuildError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_directory(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise BuildError(f"{label} must be an existing directory")
    return path


def _new_output(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise BuildError(f"{label} must be a new absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


def build(args: argparse.Namespace) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise BuildError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    fa2_src = _existing_directory(args.fa2_src, "FA2 source root")
    cutlass_src = _existing_directory(args.cutlass_src, "CUTLASS source root")
    torch_root = _existing_directory(args.torch_root, "Torch package root")
    python_include = _existing_directory(args.python_include, "Python include root")
    output = _new_output(args.output_object, "output object")
    manifest = _new_output(args.manifest, "manifest")
    source = fa2_src / QROW16_SOURCE
    if not source.is_file():
        raise BuildError(f"qrow16 source is missing: {source}")
    if "__global__ __maxnreg__(216)" not in source.read_text():
        raise BuildError("qrow16 source is missing the RU3-paired 216-register cap")

    command = [
        str(args.nvcc),
        "-forward-unknown-to-host-compiler",
        "-DFLASHATTENTION_DISABLE_BACKWARD",
        "-DFLASHATTENTION_DISABLE_DROPOUT",
        "-DFLASHATTENTION_DISABLE_PYBIND",
        "-DPy_LIMITED_API=3",
        "-DTORCH_EXTENSION_NAME=_vllm_fa2_C",
        "-D_vllm_fa2_C_EXPORTS",
        f"-I{fa2_src / 'csrc'}",
        f"-I{fa2_src / 'csrc/flash_attn'}",
        f"-I{fa2_src / 'csrc/flash_attn/src'}",
        f"-I{fa2_src / 'csrc/common'}",
        f"-I{cutlass_src / 'include'}",
        "-isystem",
        str(python_include),
        "-isystem",
        str(torch_root / "include"),
        "-isystem",
        str(torch_root / "include/torch/csrc/api/include"),
        "-isystem",
        str(args.cuda_root / "include"),
        "-DONNX_NAMESPACE=onnx_c2",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-Xcompiler=-fPIC",
        "-DENABLE_FP8",
        "--use_fast_math",
        "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1",
        f"-Xptxas=-v,--register-usage-level={REGISTER_USAGE_LEVEL}",
        "-gencode",
        "arch=compute_121a,code=sm_121a",
        "-x",
        "cu",
        "-c",
        str(source),
        "-o",
        str(output),
    ]
    compiled = _run(command)
    compiler_output = compiled.stdout + compiled.stderr
    required = (
        f"Used {EXPECTED_REGISTERS} registers",
        "0 bytes stack frame",
        "0 bytes spill stores",
        "0 bytes spill loads",
    )
    if any(value not in compiler_output for value in required):
        raise BuildError("qrow16 RU3 ptxas resource contract drifted")

    resources = _run(
        [str(args.cuobjdump), "--dump-resource-usage", str(output)]
    ).stdout
    resource_match = re.search(
        r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", resources
    )
    if resource_match is None:
        raise BuildError("cuobjdump did not report the qrow16 resource tuple")
    registers, stack, shared, local = map(int, resource_match.groups())
    if (registers, stack, local) != (EXPECTED_REGISTERS, 0, 0):
        raise BuildError("qrow16 RU3 object resource contract drifted")

    payload: dict[str, object] = {
        "schema": "fr13.fixed32.fa2_qrow16_sm121a_ru3_build.v1",
        "status": "PASS_HOST_ONLY_BUILD",
        "source_sha256": _sha256(source),
        "output_object_sha256": _sha256(output),
        "output_object_size": output.stat().st_size,
        "target": "sm_121a",
        "register_usage_level": REGISTER_USAGE_LEVEL,
        "registers": registers,
        "stack_bytes": stack,
        "local_bytes": local,
        "static_shared_bytes": shared,
        "spill_store_bytes": 0,
        "spill_load_bytes": 0,
        "cuda_visible_devices": "",
        "gpu_used": False,
        "performance_measurement": False,
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fa2-src", type=Path, required=True)
    parser.add_argument("--cutlass-src", type=Path, required=True)
    parser.add_argument("--torch-root", type=Path, required=True)
    parser.add_argument("--python-include", type=Path, required=True)
    parser.add_argument("--output-object", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    parser.add_argument(
        "--cuobjdump",
        type=Path,
        default=Path("/usr/local/cuda/bin/cuobjdump"),
    )
    args = parser.parse_args()
    print(json.dumps(build(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
