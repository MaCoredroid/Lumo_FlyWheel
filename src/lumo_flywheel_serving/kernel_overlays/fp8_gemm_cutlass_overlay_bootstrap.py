"""CUTLASS FP8 GEMM L0c bootstrap overlay.

This file is intentionally repo-owned metadata, not a vLLM vendor CUTLASS
implementation. L0c may patch it to record or stage narrow CUTLASS-source
overlay hypotheses, but the controller must not treat a patch here as a live
vendor binary mutation until a runtime wiring layer explicitly consumes it.
"""

CUTLASS_FP8_GEMM_OVERLAY_BOOTSTRAP = {
    "schema": "l0c.fp8_gemm.cutlass_overlay_bootstrap.v1",
    "runtime_wired": False,
    "backend": "cutlass",
    "allowed_scope": "metadata_only_until_runtime_overlay_wiring_exists",
    "blocked_runtime_reason": "l0c_fp8_gemm_cutlass_overlay_not_runtime_wired",
}
