#!/usr/bin/env python3
"""Add a default-off fixed32 PDL launch candidate to pinned vLLM CUTLASS.

This patch changes only the launch mode passed to CUTLASS. It does not alter
GEMM arguments, tensor layouts, scaling, scheduling, or epilogue arithmetic.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TARGET_RELATIVE_PATH = Path(
    "csrc/quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh"
)
MARKER = "// FR13_FIXED32_CUTLASS_PDL:"

INCLUDE_ANCHOR = "#include <torch/all.h>\n"
INCLUDE_REPLACEMENT = """#include <cstdlib>
#include <cstring>

#include <torch/all.h>
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

"""

RUN_ANCHOR = (
    "  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);\n"
)
RUN_REPLACEMENT = r"""  // PDL changes only dependency admission; GEMM work and math stay identical.
  const bool launch_with_pdl = fixed32_cutlass_pdl_enabled() &&
                               cute::get<0>(prob_shape) == 32;
  cutlass::Status status = gemm_op.run(
      args, workspace.data_ptr(), stream, nullptr, launch_with_pdl);
"""


def patch_text(source: str) -> tuple[str, bool]:
    """Return the patched caller and whether it changed."""
    if MARKER in source:
        required = (
            INCLUDE_REPLACEMENT,
            PDL_HELPER,
            RUN_REPLACEMENT,
        )
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
    patched, changed = patch_text(source)
    if changed:
        target.write_text(patched, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_root",
        type=Path,
        help="root of the exact pinned vLLM source tree to patch",
    )
    args = parser.parse_args()

    changed = patch_source_root(args.source_root)
    state = "patched" if changed else "already patched"
    print(f"[FR13] {state}: {args.source_root / TARGET_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
