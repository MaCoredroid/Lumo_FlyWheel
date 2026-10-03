# M1 initialization result review

**Disposition: the recorded untimed initialization/collection stage completed for all four methods, with no semantic coverage or provenance blocker found in this review.** This establishes executable collection and publication paths for the frozen component case, not numerical qualification or a performance result.

Run: `experiments/review-response-20260927/runs/m1-init-stage/m1-init-20260928T053031Z`.

| Binding | SHA256 |
| --- | --- |
| `RUN-RECEIPT.json` | `7df69127a6b7688643c7b3fcd175e253f07089c665f1c679687beb830c472df3` |
| Final `FREEZE-M1-ADAPTERS-v3.2.json` | `cf78f1985807b3b9dfcd1ba5dbf9e6c382283a4b0a838eaf107a6e5228639f4c` |
| `out/m1/stage_manifest.json` | `d06574b5d244b923b275692175d2da0ddd4b6718988abb637c796ec8ba4b250b` |
| `out/m1/tensors/index.json` | `977617743eeb6fc919d37dc2d6c462fccc259f36fbe08a32b06b48ecb0065fdf` |

The one owned container records stage exit 0, cleanup exit 0, `stopped_and_removed`, and empty stderr. Gate, launch binding, design, final freeze, image, container ID/name and nonce agree. The preflight reports all 52 expected source hashes matching; its canonical digest matches the stage and method receipts. Runtime records GB10, CUDA 13.0, Torch 2.11.0+cu130 and Triton 3.6.0 in pinned image `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`. No stage deviation, refusal, primary failure, secondary failure, failed call or finalization error is recorded.

Independently verified all 266 receipt-listed member hashes/sizes and all 22 stage-manifest file hashes. Despite its name, `files_recursive_excluding_raw_objects` includes the 205 `.bin` objects. The parent's separate raw audit also checked binary shapes and finiteness; this review focuses on recorded diagnostics and call semantics and did not independently recompute C2 from raw inputs.

## Actual coverage and publication semantics

The fixed scope is one process, L48/B1, seed 20260928, ordinary-random operands, and the serial methods Lumo native, Weaver author-default, Weaver aligned-local, TreeWY author-default. Each method runs one three-step forced-path cycle: root-only, n14, n15. These are not independent workload trials. `dot_bf16=True` is recorded for TreeWY.

- Every receipt contains exactly the same 48-layer input hash dictionary as `inputs.json`, combined digest `b613936ca771eed9f9aecfdffecf68ed69d4ca2664b5c4cb8290df42a74a884c`.
- All four initial state digests equal `a3cad07442232f929278a8f2dca3762450ee308b99bbd317d00fcdc9687c44cd`; each initial capture records all 48 exact-reset flags true.
- Each method contains 48 × 28 = **1,344** first-verification cells, in the exact active-node order, with 48 finite max-absolute-error and RMS entries per cell, valid metrics, zero nonfinite counts and zero unavailable cells. Each has 48 bf16 raw-output references of shape `[32,48,128]`, including physical padding; diagnostics cover the 28 active nodes.
- Each has exactly four FP32 state references of shape `[48,48,128,128]`: initial reset and three publication endpoints. Reference metadata matches the content-addressed index. Capture digests match receipt start/end and per-step endpoints, with each next input digest equal to the previous endpoint. All 48-layer endpoint diagnostic arrays report valid finite metrics and zero nonfinite counts.
- Lumo logs three actual `Backend.scan` and three actual `Backend.publish` calls, each covering 48 layers. Forced accepted paths match `[0]`, `[0,1,4,9,14]`, `[0,1,4,9,15]`.
- Each Weaver variant logs 144 actual verifier calls covering every `(step,layer)` pair and three all-layer `advance_ssm_states_along_accept_paths` calls with leaves `[0]`, `[14]`, `[15]`. The aligned variant explicitly records local FP32 gating/normalization helpers; it is not mislabeled as unchanged author-default arithmetic.
- TreeWY logs **192 actual fused wrappers**: 48 layers for each of three verification steps plus **48 final deferred-flush calls**. The 144 commit observations cover every `(published step,layer)` pair. Wrapper leaf values are `[0]`, `[0]`, `[4]`, `[14]` across four call rounds; the latter IDs map to physical leaves 14 and 15. The final capture follows the flush publishing n15. First-round sentinel observations, gather/remap calls and final flush are present; commit observations are not extra wrapper executions.

Counts above exclude disposable warmup calls and distinguish log observations from actual calls. They do not imply latency or GPU kernel-launch speed.

## Diagnostic differences, without a pass threshold

These values were reduced directly from saved per-head JSON diagnostics. Output Linf is the maximum absolute error across all first-verification layers, active nodes and heads; RMS is the maximum per-head RMS, not a pooled estimator. State Linf is the maximum over all layers, heads and state elements versus the declared sequential float64 C2 continuation at that publication endpoint.

| Method | First-output Linf | First-output max head RMS | State Linf: root | State Linf: n14 | State Linf: n15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lumo native | 4.818775e-4 | 7.150392e-5 | 8.045952e-7 | 2.615014e-6 | 4.501290e-6 |
| Weaver author-default | 5.279203e-4 | 1.084888e-4 | 5.146231e-3 | 4.776353e-3 | 4.121821e-3 |
| Weaver aligned-local | 5.181787e-4 | 8.701718e-5 | 1.952249e-7 | 5.745790e-7 | 9.573177e-7 |
| TreeWY author-default | 6.312657e-4 | 1.326614e-4 | 4.452449e-3 | 5.218635e-3 | 4.430514e-3 |

All reported cells are finite, but the states are not numerically identical. Author-default normalization/storage/dot policies differ from the aligned/native operands as recorded in the receipts. The observed gaps therefore do not by themselves isolate an algorithmic defect. In this case the aligned Weaver endpoint discrepancy is smaller than Lumo's; these diagnostics must not be compressed into a claim that Lumo is universally most accurate. No tolerance was frozen for this stage, so none of these values is labeled a numerical pass/fail.

## Bookkeeping limitation and claim boundary

`RUN-RECEIPT.json.raw_objects` incorrectly reports count/bytes zero because the launcher used an `objects` path while the collector stores `tensors`. This does **not** imply missing raw storage: all 205 raw objects are individually listed and hash/size bound in the same receipt. Their indexed total is 2,038,431,744 bytes (192 bf16 output objects plus 13 unique FP32 state objects; four methods share the identical initial object). Preserve the original receipt and explain the counter defect in derived summaries; do not silently replace it.

The in-job resource check recorded 12.048 GiB free against the declared 12 GiB minimum and allowed this stage. This is an observed resource-preflight outcome, not a new memory capacity or stability result.

The receipts consistently exclude timing and numerical qualification. The wall interval includes initialization, warmup, diagnostic references and raw serialization and is not a method speed comparison. This run also does not qualify full-model continuation, attention/convolution numerics, task quality, or long-running agent behavior. Bole and an aligned TreeWY variant were not executed here. No new experiment, GPU operation, remote operation, code edit or gate change was performed during this review.
