#!/usr/bin/env python3
"""Add a default-off fixed32 PDL launch candidate to pinned vLLM CUTLASS.

The live vLLM block-FP8 path swaps A/B for small M, so the fixed32 row count
can appear in either of the first two CUTLASS problem dimensions. This patch
changes only launch admission; GEMM arguments and arithmetic are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


TARGET_RELATIVE_PATH = Path(
    "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/"
    "cutlass_gemm_caller.cuh"
)
EXPECTED_UNPATCHED_SHA256 = (
    "c3c606d787502fc7cebadd288f386e3913f5ed5539df12236e9bf0bd9d49fb8d"
)
MARKER = "// FR13_FIXED32_CUTLASS_PDL:"

INCLUDE_ANCHOR = "#include <torch/csrc/stable/tensor.h>\n"
INCLUDE_REPLACEMENT = """#include <cstdlib>
#include <cstring>

#include <torch/csrc/stable/tensor.h>
"""

NAMESPACE_ANCHOR = "namespace vllm::c3x {\n\n"
PDL_HELPER = r"""namespace vllm::c3x {

// FR13_FIXED32_CUTLASS_PDL: opt in at process start; unset is stock CUTLASS.
static inline bool fixed32_cutlass_pdl_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("FR13_FIXED32_CUTLASS_PDL");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

static inline bool fixed32_cutlass_pdl_shape(
    cute::Shape<int, int, int, int> prob_shape) {
  const int32_t problem_m = cute::get<0>(prob_shape);
  const int32_t problem_n = cute::get<1>(prob_shape);
  const int32_t problem_k = cute::get<2>(prob_shape);
  // The block-FP8 small-M path swaps A/B, moving decode rows to problem N.
  const int32_t rows = problem_m < problem_n ? problem_m : problem_n;
  const bool fixed32_cobatch =
      rows == 32 || rows == 64 || rows == 96 || rows == 128;
  return fixed32_cobatch && problem_k >= 5120;
}

"""

RUN_ANCHOR = (
    "  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);\n"
)
RUN_REPLACEMENT = r"""  // PDL changes dependency admission only; the selected GEMM is unchanged.
  const bool launch_with_pdl = fixed32_cutlass_pdl_enabled() &&
                               fixed32_cutlass_pdl_shape(prob_shape);
  cutlass::Status status = gemm_op.run(
      args, workspace.data_ptr(), stream, nullptr, launch_with_pdl);
"""


def patch_text(source: str) -> tuple[str, bool]:
    """Return the patched caller and whether it changed."""
    if MARKER in source:
        required = (INCLUDE_REPLACEMENT, PDL_HELPER, RUN_REPLACEMENT)
        if not all(fragment in source for fragment in required):
            raise RuntimeError("partial FR13 fixed32 PDL patch found")
        return source, False

    anchors = {
        "include": INCLUDE_ANCHOR,
        "namespace": NAMESPACE_ANCHOR,
        "CUTLASS run": RUN_ANCHOR,
    }
    for label, anchor in anchors.items():
        count = source.count(anchor)
        if count != 1:
            raise RuntimeError(f"expected exactly one {label} anchor, found {count}")

    patched = source.replace(INCLUDE_ANCHOR, INCLUDE_REPLACEMENT, 1)
    patched = patched.replace(NAMESPACE_ANCHOR, PDL_HELPER, 1)
    patched = patched.replace(RUN_ANCHOR, RUN_REPLACEMENT, 1)
    return patched, True


def patch_source_root(source_root: Path) -> bool:
    target = source_root / TARGET_RELATIVE_PATH
    source = target.read_text(encoding="utf-8")
    if MARKER not in source:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest != EXPECTED_UNPATCHED_SHA256:
            raise RuntimeError(
                "pinned CUTLASS caller SHA256 mismatch: "
                f"{digest} != {EXPECTED_UNPATCHED_SHA256}"
            )
    patched, changed = patch_text(source)
    if changed:
        target.write_text(patched, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_root",
        type=Path,
        help="root of vLLM fe9c3d6c5f66c873d196800384ed6880687b9e52",
    )
    args = parser.parse_args()

    changed = patch_source_root(args.source_root)
    state = "patched" if changed else "already patched"
    print(f"[FR13] {state}: {args.source_root / TARGET_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
