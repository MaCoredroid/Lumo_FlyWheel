# Phase A FP8 GEMM Action-Space Audit

Date: 2026-05-02

Source audited: `src/lumo_flywheel_serving/kernel_activation.py`

## Summary

The current repo-owned runtime activation hook supports exactly two executable
FP8 GEMM backend names for `tune-kernel-select`: `cublas` and `cutlass`.
Other backend names mentioned in the Phase A pivot report are not safe action
space values until explicit runtime hooks are added and verified.

## Backend Status

| action-space name | runtime support | dispatch/resolved name | activation mechanism | isolated invocation status |
|---|---|---|---|---|
| `cublas` | supported | `torch_scaled_mm` | sets `VLLM_DISABLED_KERNELS` to disable Marlin, FlashInfer, and CUTLASS FP8 scaled-MM kernels | unknown / not yet implemented |
| `cutlass` | supported | `CutlassFP8ScaledMMLinearKernel` | sets `VLLM_DISABLED_KERNELS` to disable Marlin, FlashInfer, per-tensor Torch, channel-wise Torch, and row-wise Torch FP8 scaled-MM kernels | unknown / not yet implemented |
| `cutlass_blackwell_scaled_mm` | unsupported | not resolved | no repo-owned exact dense FP8 GEMM launch hook | unknown / not yet implemented |
| `triton_fp8_scaled_mm` | unsupported | not resolved | no repo-owned exact dense FP8 GEMM launch hook | unknown / not yet implemented |
| `marlin` | unsupported | not resolved | no repo-owned exact dense FP8 GEMM launch hook | unknown / not yet implemented |
| `machete` | unsupported | not resolved | no repo-owned exact dense FP8 GEMM launch hook | unknown / not yet implemented |
| `tensorrt_llm_fp8` | unsupported | not resolved | no repo-owned exact dense FP8 GEMM launch hook | unknown / not yet implemented |

## Notes

`resolve_kernel_runtime_activation()` maps `fp8_gemm_kernel=cublas` and
`fp8_gemm_kernel=cutlass` through environment-based exclusion of competing
vLLM FP8 scaled-MM kernels. For any other `fp8_gemm_kernel` value, the hook
returns `l0a_kernel_selection_runtime_unsupported_knobs` with the reason
`repo has no safe exact dense FP8 GEMM launch hook for this value`.

The current Phase A executable action space is therefore:

```yaml
fp8_gemm_kernel: [cublas, cutlass]
```

Dispatch identity and standalone synthetic-tensor invocation are not yet
implemented as separate audit artifacts. They remain required follow-up work
before adding backend names beyond the two currently wired values.
