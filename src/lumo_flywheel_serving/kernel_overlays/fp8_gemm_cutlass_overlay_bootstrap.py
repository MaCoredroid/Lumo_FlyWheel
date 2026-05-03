"""CUTLASS FP8 GEMM L0c bootstrap overlay.

This repo-owned file is the only supported CUTLASS fp8_gemm L0c source surface.
L0c candidates may patch ``runtime_overlay.source_replacements`` to express
small, exact source overlays against vLLM's Python CUTLASS scaled-MM dispatch
module. The controller materializes those replacements into a runtime-consumed
import hook and refuses candidates whose effective runtime overlay is unchanged.
"""

CUTLASS_FP8_GEMM_OVERLAY_BOOTSTRAP = {
    "schema": "l0c.fp8_gemm.cutlass_overlay_bootstrap.v1",
    "runtime_wired": True,
    "backend": "cutlass",
    "allowed_scope": "exact_vllm_cutlass_scaled_mm_source_replacements",
    "runtime_overlay": {
        "schema": "l0c.fp8_gemm.cutlass_runtime_overlay.v1",
        "target_modules": [
            "vllm.model_executor.kernels.linear.scaled_mm.cutlass",
            "vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass",
        ],
        "source_replacements": [],
    },
}
