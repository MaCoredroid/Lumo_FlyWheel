# L0c FP8 CUTLASS Tier-4 Parity and Next Surface

Date: 2026-05-03

Scope: follow-up research after run
`qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260503T060649Z`.

## Question

Three candidates failed `parity_fp8_tier4_downstream_logit_diverged`.
Should tier-4 downstream-logit parity be loosened, and how should the next
CUTLASS-only round move toward real dispatch/shape/scale/schedule behavior?

## Finding

Keep strict tier-4 parity as the final acceptance gate, but do not treat the
current `1e-3/1e-3` downstream-logit check as the only useful exploration
gate for real CUTLASS schedule work.

The run4 failures were not borderline:

- candidate 001 overshoot: `0.340469`
- candidate 002 overshoot: `0.369375`
- candidate 003 overshoot: `0.392879`

Those are absolute logit excesses beyond the tolerance, not tiny relative
rounding misses. Relaxing `1e-3` to a normal numeric-tolerance value would not
save these candidates. They were semantic failures or no-value wrapper edits.

However, a future schedule/tile/dispatch mutation can legitimately change
floating-point reduction order. For that class of patch, tier-4 should be
interpreted as the final correctness rail, while earlier candidate admission
should rely on cheaper direct checks:

- tier-3 GEMM output compare
- shape/dtype/finite checks
- sampled top-k or target-token stability
- calibrated baseline jitter from repeated unmutated tier-4 probes

## Online Research

vLLM documents FP8 inference as a quantized path with explicit accuracy
evaluation, not bit-identical execution. Dynamic FP8 in vLLM computes activation
scales during each forward pass for high accuracy, and vLLM recommends
evaluating accuracy after quantization. Source:
https://docs.vllm.ai/en/v0.7.0/features/quantization/fp8.html

NVIDIA's FP8 scaling material emphasizes that FP8 requires explicit per-tensor,
current/delayed, or block scaling to maintain numerical stability and accuracy.
That supports treating scale semantics as correctness-critical. Source:
https://developer.nvidia.com/blog/?p=102820

Transformer Engine delayed scaling is described as a memory-efficiency tradeoff:
current scaling reads once for amax and once for cast, while delayed scaling
uses historical amax to avoid the extra read. This is a real FP8 speed surface,
but it changes scale timing and must be guarded by parity. Source:
https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html

CUTLASS 3.x GEMM is organized around a mainloop and epilogue, with kernel
schedule selected through dispatch policy. The collective builder exposes
tile shape, cluster shape, stage count, and kernel schedule as concrete tuning
parameters. Source:
https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/gemm_api_3x.html

CUTLASS grouped schedulers assign tiles to persistent threadblocks; scheduler
mode and load balance are real performance levers. Source:
https://docs.nvidia.com/cutlass/media/docs/cpp/grouped_scheduler.html

vLLM's CUTLASS scaled-mm path has correctness-sensitive handling around
`process_weights_after_loading`, including transposed weights, scale conversion,
static input-scale handling, and AZP adjustment for int8. Even when mutating
FP8, the lesson is that loading/scale semantics must not be replaced wholesale.
Source:
https://docs.vllm.ai/en/v0.13.0/api/vllm/model_executor/layers/quantization/kernels/scaled_mm/cutlass/

## Local Code Evidence

The fixture uses:

```yaml
tier_4_tolerances:
  rtol_downstream_logit: 0.001
  atol_downstream_logit: 0.001
```

The parity probe computes absolute excess:

```python
allowed = atol + rtol * abs(reference)
excess = abs(candidate - reference) - allowed
overshoot = max(0.0, max(excess))
```

So an overshoot around `0.34` means the candidate exceeded the allowed logit
envelope by roughly one third of a logit at some vocab position.

## Next Round Surface

Do not spend another loop on wrapper cosmetics. The next CUTLASS-only loop
should require each candidate to touch one of these mechanisms:

- Dispatch: predicates deciding whether CUTLASS handles a shape/config and
  whether an alternate scaled-mm path is selected.
- Shape: guarded changes to the 2D GEMM problem shape, padding, grouping, or
  output shape only where the mathematical result is provably unchanged.
- Scale: activation scale shape, static vs dynamic scale handling, scale
  placement, and per-token/per-channel/per-tensor compatibility.
- Schedule: only if the source/compile surface is expanded beyond the installed
  Python package, because the current wheel exposes compiled `_C*.so` custom ops
  rather than editable CUTLASS C++ sources.

## Recommendation

For the immediate next loop:

1. Keep final tier-4 at `1e-3/1e-3`.
2. Add a baseline-jitter calibration before deciding whether to relax any
   exploratory tier-4 admission threshold.
3. Keep agents on the local CUTLASS source workspace, but require a declared
   dispatch/shape/scale mechanism before patching.
4. Treat C++ schedule mutation as a separate enablement task: stage vLLM source
   and a rebuild path for `_C.abi3.so`, or it is not a real runtime mutation.

Do not start a larger-budget loop until either:

- the brief blocks cosmetic wrapper edits strongly enough, or
- the controller supports a compiled CUTLASS C++ workspace.
