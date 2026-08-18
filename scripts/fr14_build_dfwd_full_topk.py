#!/usr/bin/env python3
"""Build the default-off FR14 fused full-vocabulary draft top-k CUDA op.

Mirrors `scripts/fr13_build_dfwd_k64_top3.py` (the K64 precedent) with the two
corrections the campaign banked afterwards:

  * the reproducibility credential is the SASS digest, not the `.so` sha256 --
    nvcc stamps its build-container PID into ~87 kB of host-side name-table
    bytes, so a byte-identical source can produce a different `.so` sha with
    identical device code (fr14_treeattn_v2_build_env_proof.json, pass 37).
    Both are recorded; only the SASS digest attests the kernel.
  * the attestation states, in the artifact itself, that nothing was measured
    and nothing is qualified by building.

Usage (build container, GPU deliberately NOT required):

  python3 scripts/fr14_build_dfwd_full_topk.py \
      --output  output/<runroot>/fr14_dfwd_full_topk_sm121a.abi3.so \
      --build-dir output/<runroot>/build \
      --attestation output/<runroot>/build_attestation.json
"""

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
SOURCE = REPO / "csrc" / "fr14_dfwd_full_topk.cu"
EXPECTED_TORCH = "2.11.0+cu130"
EXPECTED_CUDA = "13.0"
EXPECTED_ARCH = "12.1a"
MODULE_NAME = "fr14_dfwd_full_topk_sm121a"
CUDA_PACKAGE_INCLUDE = Path(
    "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
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


def sass_digest(binary: Path) -> tuple[str, dict]:
    """SASS text digest + resource usage: the reproducibility credential.

    `cuobjdump --dump-sass` refuses the linked host `.so` on this toolchain but
    reads the relocatable `.cuda.o` fine, so the object is preferred and the
    shared object is the fallback.
    """
    sass = None
    for candidate in (binary,):
        try:
            sass = subprocess.run(
                ["cuobjdump", "--dump-sass", str(candidate)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            break
        except (OSError, subprocess.CalledProcessError) as exc:
            last = exc
    if sass is None:
        return f"UNAVAILABLE:{last}", {}
    # Strip the address column so the digest attests instructions, not layout.
    lines = []
    for line in sass.splitlines():
        stripped = line.strip()
        if stripped.startswith("/*") and "*/" in stripped:
            stripped = stripped.split("*/", 1)[1].strip()
        if stripped:
            lines.append(stripped)
    digest = hashlib.sha256("\n".join(lines).encode("ascii", "replace")).hexdigest()
    body = "\n".join(lines)

    usage: dict = {
        "sass_lines": len(lines),
        "ldl_count": body.count("LDL"),
        "stl_count": body.count("STL"),
        "call_count": body.count("CALL"),
    }
    try:
        text = subprocess.run(
            ["cuobjdump", "--dump-resource-usage", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for token in ("REG", "STACK", "SHARED", "LOCAL", "CONSTANT"):
            for line in text.splitlines():
                if f"{token}:" in line:
                    usage.setdefault(token, line.strip())
                    break
        usage["raw"] = text.strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        pass
    return digest, usage


def build(output: Path, build_dir: Path, attestation: Path, strict: bool) -> dict:
    import torch
    from torch.utils.cpp_extension import load

    if strict and torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(
            f"pinned build requires torch {EXPECTED_TORCH}, got {torch.__version__}"
        )
    cuda_version = subprocess.run(
        ["nvcc", "--version"], check=True, capture_output=True, text=True
    ).stdout
    if strict and f"release {EXPECTED_CUDA}" not in cuda_version:
        raise RuntimeError("pinned build requires CUDA 13.0 nvcc")

    output = output.resolve()
    build_dir = build_dir.resolve()
    attestation = attestation.resolve()
    if output == SOURCE or attestation == SOURCE:
        raise ValueError("build outputs must not replace the CUDA source")
    output.parent.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    os.environ["TORCH_CUDA_ARCH_LIST"] = EXPECTED_ARCH
    extra_include = []
    if (CUDA_PACKAGE_INCLUDE / "cusparse.h").is_file():
        extra_include = [f"-I{CUDA_PACKAGE_INCLUDE}"]

    built = Path(
        load(
            name=MODULE_NAME,
            sources=[str(SOURCE)],
            build_directory=str(build_dir),
            extra_cflags=["-O3", *extra_include],
            extra_cuda_cflags=[
                "-O3",
                *extra_include,
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

    namespace = getattr(torch.ops, "fr14_fused_draft_topk", None)
    if namespace is None or not hasattr(namespace, "select_out"):
        raise RuntimeError("built library did not register the FR14 CUDA op")

    sass_source = build_dir / f"{SOURCE.stem}.cuda.o"
    if not sass_source.is_file():
        sass_source = output
    digest, usage = sass_digest(sass_source)
    usage["source"] = recorded_path(sass_source)
    payload = {
        "schema": "fr14.fused_draft_topk.build.v1",
        "status": "BUILT_UNQUALIFIED",
        "performance_measurement": False,
        "byte_equality_claim": False,
        "real_task_correctness": False,
        "production_default_enabled": False,
        "torch_version": torch.__version__,
        "cuda_release": EXPECTED_CUDA,
        "cuda_arch": EXPECTED_ARCH,
        "strict_env_pins_enforced": strict,
        "source": {
            "path": str(SOURCE.relative_to(REPO)),
            "sha256": sha256_file(SOURCE),
        },
        "binary": {
            "path": recorded_path(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "sha256_is_reproducibility_credential": False,
        },
        "sass": {
            "digest_sha256": digest,
            "is_reproducibility_credential": True,
            "resource_usage": usage,
        },
        "next_gate": (
            "results/fr14_nvfp4_port_20260816/fr14_fused_draft_topk_probe.py "
            "--so <this binary>  (0 raw-byte mismatches required)"
        ),
    }
    atomic_json(attestation, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--attestation", required=True)
    parser.add_argument(
        "--no-strict-env",
        action="store_true",
        help="record the toolchain instead of refusing on a drift (dev only)",
    )
    args = parser.parse_args()
    payload = build(
        Path(args.output),
        Path(args.build_dir),
        Path(args.attestation),
        strict=not args.no_strict_env,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
