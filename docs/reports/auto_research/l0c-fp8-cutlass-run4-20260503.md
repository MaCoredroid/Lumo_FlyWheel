# L0c FP8 CUTLASS Auto-Research Run 4

Date: 2026-05-03

Round: `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260503T060649Z`

Round dir: `/home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260503T060649Z`

## Summary

The CUTLASS-only auto-research loop ran to terminal completion after the controller-side workspace/preflight changes.

Outcome: `ROUND_BLOCKED`

Terminal condition: `proposer_stuck`

Accepted candidates: 0

Rejected candidates: 3

Baseline objective mean: `0.0555612`

Winner objective mean: none

## Harness Behavior Check

The loop now matches the intended controller/agent split:

- Agents received a local `cutlass_source_workspace` copied from the live vLLM CUTLASS scaled-mm source.
- Agents generated patches against `cutlass_source_workspace`, not the old bootstrap overlay.
- Agents had cheap checks in the brief: `patch --dry-run`, local `py_compile`, and `lumoserve auto-research preflight-patch`.
- Agents did not run `apply-and-test`.
- The controller owned patch application, live container restart, parity, and measurement.
- Expensive restarts only happened after candidates passed patch/preflight checks.

This fixed the previous blocker where preflight rejected file edits instead of checking correctness.

## Baseline

Remeasured baseline objective values:

- `0.055893`
- `0.057052`
- `0.056852`
- `0.051353`
- `0.056656`

Mean: `0.0555612`

## Candidate Analysis

### Candidate 001

Patch:

```diff
--- cutlass_source_workspace/scaled_mm/cutlass.py
+++ cutlass_source_workspace/scaled_mm/cutlass.py
@@ -170,4 +170,6 @@
         output = ops.cutlass_scaled_mm(
             A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
         )
+        if len(output_shape) == 2:
+            return output
         return output.view(*output_shape)
```

Controller result: rejected.

Reason: `parity_fp8_tier4_downstream_logit_diverged`

First diverging probe: `0`

Tolerance overshoot: `0.340469`

Analysis: this is mostly a reshape fast path. It passed cheap checks but changed downstream logits beyond tolerance. It is not a credible performance candidate for the target workload unless paired with stronger shape-specific proof, because it avoids only a cheap `view` and has immediate model-output risk.

### Candidate 002

Patch:

```diff
--- cutlass_source_workspace/scaled_mm/cutlass.py
+++ cutlass_source_workspace/scaled_mm/cutlass.py
@@ -155,6 +155,27 @@
     def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
         return True, None
 
+    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
+        w_name, w_s_name, _, _ = self.layer_param_names
+
+        weight = getattr(layer, w_name)
+        if not weight.is_contiguous():
+            replace_parameter(
+                layer,
+                w_name,
+                torch.nn.Parameter(weight.contiguous().data, requires_grad=False),
+            )
+
+        weight_scale = getattr(layer, w_s_name)
+        if not weight_scale.is_contiguous():
+            replace_parameter(
+                layer,
+                w_s_name,
+                torch.nn.Parameter(
+                    weight_scale.contiguous().data, requires_grad=False
+                ),
+            )
+
     def apply_scaled_mm(
         self,
         *,
```

Controller result: rejected.

Reason: `parity_fp8_tier4_downstream_logit_diverged`

First diverging probe: `0`

Tolerance overshoot: `0.369375`

Analysis: this attempted a weight/scale contiguity adjustment, but it overrode CUTLASS weight-loading behavior. Upstream vLLM CUTLASS scaled-mm loading transposes weights, converts fused per-tensor scales to channelwise form where needed, handles static input scale, and computes AZP adjustment. Replacing that path with a contiguity-only implementation is correctness-risky and explains the parity divergence.

### Candidate 003

Patch:

```diff
--- cutlass_source_workspace/scaled_mm/ScaledMMLinearKernel.py
+++ cutlass_source_workspace/scaled_mm/ScaledMMLinearKernel.py
@@ -132,14 +132,16 @@
         #   If dynamic, layer.input_scale is None and x_s computed from x.
         #   If static, layer.input_scale is scalar and x_s is input_scale.
         # View input as 2D matrix for fp8 methods
-        x_2d = x.view(-1, x.shape[-1])
-        output_shape = [*x.shape[:-1], w.shape[1]]
+        x_shape = x.shape
+        x_2d = x.view(-1, x_shape[-1])
+        output_shape = [*x_shape[:-1], w.shape[1]]
         out_dtype = x.dtype if maybe_out_dtype is None else maybe_out_dtype
 
         # If input not quantized
         # TODO(luka) remove this path if not used anymore
-        x_2d_q = x_2d
-        if x.dtype != fp8_dtype:
+        if x.dtype == fp8_dtype:
+            x_2d_q = x_2d
+        else:
             x_2d_q, x_s = self.quant_fp8(
                 x_2d,
                 x_s,
```

Controller result: rejected.

Reason: `parity_fp8_tier4_downstream_logit_diverged`

First diverging probe: `0`

Tolerance overshoot: `0.392879`

Analysis: this is a control-flow rewrite intended to avoid redundant assignment when input is already FP8. It is too small and too indirect to plausibly move the latency metric. The parity failure means even apparently equivalent wrapper edits must be treated as unsafe until tier-4 parity passes.

## Online Research Notes

- NVIDIA CUTLASS documents grouped kernels as persistent kernels where a scheduler assigns tiles to threadblocks. The performance-sensitive levers include scheduler mode and load balance, not just Python wrapper reshapes. Source: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html
- CUTLASS 4.5.0 describes support for FP8 types, block-scaled types, and Python-native CuTe DSLs. This supports using a local source workspace for kernel/source-level exploration, but credible speedups likely need schedule/tile/dispatch changes rather than no-op wrapper edits. Source: https://github.com/NVIDIA/cutlass
- vLLM's CUTLASS scaled-mm wrapper documents `process_weights_after_loading`; the source path includes weight transposition, scale conversion, static input-scale handling, and AZP adjustment. Agents should not replace this method wholesale unless they preserve those semantics. Source: https://docs.vllm.ai/en/v0.14.0/api/vllm/model_executor/layers/quantization/kernels/scaled_mm/cutlass/

## Recommendation

Run another CUTLASS-only auto-research loop only after tightening the agent prompt/guidance. This round proved the infrastructure now works, but the proposer got stuck generating low-value or correctness-breaking wrapper edits.

Better next-loop guidance:

- Require every candidate to name the expected performance mechanism before patching.
- Prefer changes that preserve vLLM loading semantics and only add guarded fast paths.
- Forbid wholesale replacement of `process_weights_after_loading` unless the patch keeps transposition, fused scale conversion, static input-scale handling, and AZP adjustment.
- Ask agents to inspect rejected patches first and avoid reshape/no-op/control-flow-only mutations.
- Bias toward dispatch/shape gating, scale handling, and CUTLASS selection surfaces that can change actual kernel behavior.

Artifacts:

- `round_spec.yaml`
- `iteration_brief.md`
- `strategy_brief.md`
- `measurements.tsv`
- `mutations_rejected.tsv`
- `run_log.json`
- `measurement_trace_combined.json`
- candidate patches and parity files under `candidates/001`, `candidates/002`, and `candidates/003`
