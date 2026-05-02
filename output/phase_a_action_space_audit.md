# Phase A FP8 GEMM Action-Space Audit

Date: 2026-05-02

Source audited: `src/lumo_flywheel_serving/kernel_activation.py`

## Summary

The current repo-owned runtime activation hook supports exactly two executable
FP8 GEMM backend names for `tune-kernel-select`: `cublas` and `cutlass`.
Other backend names mentioned in the Phase A pivot report are not safe action
space values until explicit runtime hooks are added and verified.

## Backend Status

| action-space name | runtime support | dispatch/resolved name | dispatch identity | activation mechanism | isolated invocation status |
|---|---|---|---|---|---|
| `cublas` | supported | `torch_scaled_mm` | `src/lumo_flywheel_serving/kernel_activation.py::lumo_flywheel_serving.kernel_activation._apply_fp8_gemm_kernel` sha256 `393754c04b9b38c56cb8bd5b1addbd3629f161a0fe6e79d517287d9371ed896b` | sets `VLLM_DISABLED_KERNELS` to disable Marlin, FlashInfer, and CUTLASS FP8 scaled-MM kernels | unknown / not yet implemented |
| `cutlass` | supported | `CutlassFP8ScaledMMLinearKernel` | `src/lumo_flywheel_serving/kernel_activation.py::lumo_flywheel_serving.kernel_activation._apply_fp8_gemm_kernel` sha256 `393754c04b9b38c56cb8bd5b1addbd3629f161a0fe6e79d517287d9371ed896b` | sets `VLLM_DISABLED_KERNELS` to disable Marlin, FlashInfer, per-tensor Torch, channel-wise Torch, and row-wise Torch FP8 scaled-MM kernels | unknown / not yet implemented |
| `cutlass_blackwell_scaled_mm` | unsupported | not resolved | unsupported by repo-owned dispatch hook | no repo-owned exact dense FP8 GEMM launch hook | unknown / not yet implemented |
| `triton_fp8_scaled_mm` | unsupported | not resolved | unsupported by repo-owned dispatch hook | no repo-owned exact dense FP8 GEMM launch hook | unknown / not yet implemented |
| `marlin` | unsupported | not resolved | unsupported by repo-owned dispatch hook | no repo-owned exact dense FP8 GEMM launch hook | unknown / not yet implemented |
| `machete` | unsupported | not resolved | unsupported by repo-owned dispatch hook | no repo-owned exact dense FP8 GEMM launch hook | unknown / not yet implemented |
| `tensorrt_llm_fp8` | unsupported | not resolved | unsupported by repo-owned dispatch hook | no repo-owned exact dense FP8 GEMM launch hook | unknown / not yet implemented |

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

Dispatch identity is recorded in Phase A round artifacts as
`phase_a_backend_identities.json` and repeated in `round_spec.yaml` /
bundle provenance. The identity binds each scheduled action-space value to the
repo-owned dispatch hook symbol, source path, support status, resolved runtime
name, and source hash above. This is not an upstream vLLM symbol/source hash.
If a later run compares against a stale manifest and the dispatch hook changed, it raises
`HALT_REASON: phase_a_backend_dispatch_drift`.

Standalone synthetic-tensor invocation remains unimplemented; do not add
backend names beyond the two currently wired values until their exact dispatch
identity and isolated invocation status are both verified.
