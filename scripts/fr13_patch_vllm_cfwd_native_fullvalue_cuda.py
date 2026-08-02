#!/usr/bin/env python3
"""Install the default-off key-group precompute CFWD op in pinned vLLM `_C`."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


VLLM_COMMIT = "fe9c3d6c5f66c873d196800384ed6880687b9e52"
MARKER = "FR13_FIXED32_CFWD_NATIVE_KEYGROUP_CONTIGUOUS_K_CUDA_V8"
CUDA_DESTINATION = Path("csrc/fr13_fixed32_cfwd_native_fullvalue.cu")

PINNED_SHA256 = {
    Path("CMakeLists.txt"): (
        "b12cd47f5761442551d6e1966e8a37ad94175382c1b014d2b65f67b74fbb6e3b"
    ),
    Path("csrc/ops.h"): (
        "50138aa29f09388e6a22416349cf45e18abf0f069dc1a7401f85ce12feb89a0a"
    ),
    Path("csrc/torch_bindings.cpp"): (
        "2a05256c48c6bad44ac7e13b0f83ae1be440fc95b9338eca72f0be76f6600a7c"
    ),
}

CMAKE_ANCHOR = (
    '  list(APPEND VLLM_EXT_SRC "csrc/minimax_reduce_rms_kernel.cu")\n'
)
CMAKE_REPLACEMENT = (
    '  list(APPEND VLLM_EXT_SRC "csrc/minimax_reduce_rms_kernel.cu")\n'
    f"  # {MARKER}\n"
    f'  list(APPEND VLLM_EXT_SRC "{CUDA_DESTINATION.as_posix()}")\n'
)

OPS_ANCHOR = "void paged_attention_v1(\n"
OPS_REPLACEMENT = f"""// {MARKER}
void fr13_fixed32_cfwd_native_fullvalue(
    torch::Tensor& bank_anchor, const torch::Tensor& bank_off16,
    const torch::Tensor& accepted_paths, const torch::Tensor& accepted_lens,
    const torch::Tensor& spec_state_indices, const torch::Tensor& k_rings,
    const torch::Tensor& v_rings, const torch::Tensor& a_rings,
    const torch::Tensor& b_rings, const torch::Tensor& gate_coeffs,
    int64_t batch_size, bool bank_offset_table_prevalidated,
    bool accepted_values_device_guarded);

{OPS_ANCHOR}"""

BINDINGS_ANCHOR = """  // vLLM custom ops
  //

"""
BINDINGS_REPLACEMENT = BINDINGS_ANCHOR + f"""  // {MARKER}
  ops.def(
      "fr13_fixed32_cfwd_native_fullvalue("
      "Tensor! bank_anchor, Tensor bank_off16, Tensor accepted_paths, "
      "Tensor accepted_lens, Tensor spec_state_indices, Tensor k_rings, "
      "Tensor v_rings, Tensor a_rings, Tensor b_rings, Tensor gate_coeffs, "
      "int batch_size, bool bank_offset_table_prevalidated, "
      "bool accepted_values_device_guarded) -> ()");
  ops.impl("fr13_fixed32_cfwd_native_fullvalue", torch::kCUDA,
           &fr13_fixed32_cfwd_native_fullvalue);

"""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_pinned(source_root: Path) -> dict[Path, str]:
    sources: dict[Path, str] = {}
    for relative, expected in PINNED_SHA256.items():
        path = source_root / relative
        payload = path.read_bytes()
        actual = _sha256(payload)
        if actual != expected:
            raise RuntimeError(
                f"pinned vLLM source drift for {relative}: {actual} != {expected}"
            )
        sources[relative] = payload.decode("utf-8")
    return sources


def _replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"{label} anchor count must be one, got {count}")
    return source.replace(anchor, replacement, 1)


def _already_patched(source_root: Path, cuda_payload: bytes) -> bool:
    marker_files = (
        source_root / "CMakeLists.txt",
        source_root / "csrc/ops.h",
        source_root / "csrc/torch_bindings.cpp",
    )
    hits = [MARKER in path.read_text(encoding="utf-8") for path in marker_files]
    if not any(hits):
        return False
    if not all(hits):
        raise RuntimeError(
            "partial FR13 native key-group precompute CFWD patch detected"
        )
    destination = source_root / CUDA_DESTINATION
    if not destination.is_file() or destination.read_bytes() != cuda_payload:
        raise RuntimeError(
            "patched native key-group precompute CFWD CUDA bytes drifted"
        )
    return True


def patch_source_root(source_root: Path, cuda_source: Path) -> bool:
    source_root = source_root.resolve()
    cuda_payload = cuda_source.resolve().read_bytes()
    if _already_patched(source_root, cuda_payload):
        return False

    sources = _read_pinned(source_root)
    patched = {
        Path("CMakeLists.txt"): _replace_once(
            sources[Path("CMakeLists.txt")],
            CMAKE_ANCHOR,
            CMAKE_REPLACEMENT,
            "CMake",
        ),
        Path("csrc/ops.h"): _replace_once(
            sources[Path("csrc/ops.h")],
            OPS_ANCHOR,
            OPS_REPLACEMENT,
            "ops declaration",
        ),
        Path("csrc/torch_bindings.cpp"): _replace_once(
            sources[Path("csrc/torch_bindings.cpp")],
            BINDINGS_ANCHOR,
            BINDINGS_REPLACEMENT,
            "torch binding",
        ),
    }

    destination = source_root / CUDA_DESTINATION
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite unexpected {CUDA_DESTINATION}")
    destination.write_bytes(cuda_payload)
    for relative, text in patched.items():
        (source_root / relative).write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument(
        "--cuda-source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "native/fr13_fixed32_cfwd_native_fullvalue.cu",
    )
    args = parser.parse_args()
    changed = patch_source_root(args.source_root, args.cuda_source)
    print("patched" if changed else "already-patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
