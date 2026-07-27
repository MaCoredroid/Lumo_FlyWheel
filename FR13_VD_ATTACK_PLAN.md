# FR13 V+D ATTACK PLAN — verify-forward + drafter, accept-agnostic (2026-07-27)

User directive: attack the two accept-agnostic denominators first — the verify
forward and the drafter (= MTP head + suffix path combined). Scope guard: using
the suffix to SKIP MTP iterations is the deleted adaptive-skip path (collapsed
accept 3.6→2.0) — excluded by the accept-agnostic criterion itself. The
suffix side already costs ~0.3ms (host walk); the whole drafter cost is the
5-level MTP head loop.

## Bytes ledger (model config, /models/qwen3.6-27b-fp8)
64 layers = 48 GDN linear-attention + 16 full-attention (interval 4);
hidden 5120, intermediate 17408, vocab 248320, attn 24 heads x 256 (4 KV),
GDN 48 v-heads/16 k-heads x128. MTP head = 1 full-attn-style layer
(~0.39GB fp8: qkvo ~73M + mlp ~267M + fc ~52M params) sharing embeddings.
GB10: 273 GB/s unified; model weight read 27GB fp8 = the 98.6ms verify floor.
- Drafter per-iteration floor: MTP layer 0.39GB + DVK-64k lm_head slice
  (0.33GB fp8, pre-materialized at boot — verified in the shim: index_select
  once, cached) + KV ≈ ~0.8-1.1GB → ~3-4ms/iter → 4-iter loop floor ~12-16ms.
  MEASURED ~54ms/step → ~3.5-4.5x over floor INSIDE the R4 captured graph
  (launch overhead already gone) = real kernel inefficiency.
- Prime drafter suspect (inference, needs the probe): the fp8 GEMM on the 64k
  slice at M=1..4 — DVK bake moved 94.9→56.3 (−38.6ms ≈ −7.7ms/call x5),
  implying ~7.4ms/call remaining for a ~1.2-1.5ms read → ~5x off bandwidth.
- Verify fixed excess (155−98.6 ≈ 56ms): full-attn KV reads at agentic context
  (16 layers x ~30-75k ctx x bf16 = multi-GB/step, scales with CONTEXT not
  eps) + norm-soup remnants + per-layer fixed costs. Marginal 31/event =
  22-row tree work (scan/attn/norms).

## What is already refuted / shipped (do NOT redo)
- Shipped since the 2026-07-22 nsys profile: RING_EXPORT (B1), CONV_WB_FUSED
  (B2a), KV_REMAP_SYNCFREE, batched committer, burn-off, SLOT_REORDER,
  PARENT_GATHER (micro, ~3ms), subtree-parallel, DVK-64k, R4+L5 drafter graph.
- Refuted: GEMM M-tile scaling (rows free), lm_head as tree lever, verify
  layer-fusion (online residual dep), committer-fuse-into-verify (accepted
  path unknown), replay multistream (SM-saturated: 91.6>76.6ms), BV widen
  re-tile (register wall), .contiguous() kills (proven no-ops), HC (retired),
  drafter meta-reuse (dead heat), adaptive-skip (deleted).

## Ranked levers (accept-agnostic; sized honestly)
- D1 (largest, pending probe): drafter lm_head fp8 GEMM efficiency at tiny M —
  if the probe confirms ~7ms/call for a 0.33GB read, candidate fixes: better
  gemv kernel/config for the sliced head (fp8 block-quant path at M=1), or a
  resident bf16 slice (0.67GB, 2.4ms floor) via simple gemv. Target −15-25ms/step.
- V1 (cheap, built): FR13_CONV_WB_BATCHED (B2c) — gate + bake, ~2-5ms/step.
- V2 (probe-ranked): tree-attention 22-row decode config (the +14/draft bucket;
  splitkv/num_splits investigation) — up to ~10-20ms/step at operating eps.
- V3 (medium-high): GDN scan h_cache→shared-memory + O(1) parent load (§10 #2;
  the parent-only contract is now PROVEN by baked PARENT_GATHER). ~6-9ms/draft
  era-sizing; re-rank from the probe.
- V4 (structural, committer-adjacent, after V/D): committer-under-drafter
  overlap — the drafter needs only the committed TOKEN, not the advanced GDN
  state; the state replay + DtoH could hide under the next propose. Bounded by
  the multistream SM-saturation lesson (overlap idle, not bandwidth). Sized by
  the cfwd share that is genuinely idle — needs the probe's committer window.
- Parked by user (state-bytes, behavior-gated, NOT in this pass): KV fp8,
  mamba dtype.

## Instrument (committed this pass)
FR13_TORCH_PROF="<skip>:<active>" — one torch.profiler window driven from the
sfwd timer's per-step tick; the three span timers enter record_function windows
(FR13_W_VERIFY / FR13_W_DRAFTER / FR13_W_COMMITTER) so kernels attribute per
component; dump = per-window cuda ms/step + top-40 kernels (.json/.txt).
Fail-safe: any error → one-shot needle + disabled, never crashes the worker.

## Probe (queued behind tempfix1): output/fr13_msr/run_vdprof1.sh
EAGER-LABELED diagnostic boot (ENFORCE_EAGER=1, FR13_DRAFTER_GRAPH=0, canonical
tail6 shape, temp 0.6): profiler window 150:24, ~40 completion probes. Read
per-kernel GPU times + component ratios ONLY (walls inflated by eager+CUPTI).
Deliverable: named kernels for D1 and re-ranked V2/V3/V4 → then author the
top fix behind a flag with a same-boot byte/band gate.

## PROBE RESULTS (run_20260727T195333Z, 24 active steps, EAGER basis)
DRAFTER window 79.2 ms/step (72 window-calls):
- gemvx **BF16** 27.9 ms/step, n=217 (~9/step, ~3.1ms/call ≈ 216GB/s — the
  gemv itself is near-bandwidth-EFFICIENT; the DVK slice is resident in BF16
  = 0.67GB/read; the lever is TRAFFIC, not kernel choice)
- cutlass fp8 blockwise (MTP layer projections) 8.6 | everything else < 1
=> D1' (fp8 DVK slice, ~−14ms) — **PARKED by user 2026-07-27**: draft-logit
   quantization is distribution-touching; joins KV-fp8 + mamba-dtype in the
   parked state-bytes/quantization family. The campaign stays EXACT-MATH.
   Remaining drafter lever: the call-multiplicity audit (~9 gemv/step vs ~5
   expected — if redundant, exact-math savings ~6-9ms; one code-read decides).
VERIFY window 306.1 ms/step eager (kernel sum ~148; eager launch-gap
inflation ~150 — graphs already reclaim this in prod):
- fp8 GEMMs 115.6 (n~256/step) = only ~17% over the 98.6 weight floor →
  GEMMs are near-efficient, NOT a lever (re-confirms the M-tile refutation)
- tree-specific kernels are SMALL: _tree_gdn_path 12.9 + index_copy 4.1 +
  scatter_gather 3.7 + conv_wb_fused 2.8 + soup ~5 ≈ ~28 ms/step
- attention ~0.9 at the probe's SHORT context — the agentic verify fixed
  excess must be KV reads at long context (not probed here; state-bytes
  levers are the matching attack, currently user-PARKED)
=> V-front honest re-rank: kernel-side verify headroom ≈ 20-28ms (scan+soup),
   NOT ~90; the remainder of graphed-verify-over-floor at agentic contexts =
   KV traffic + in-graph bubbles. B2c/V3 still valid but small.
COMMITTER window 43.3 ms/step eager (kernel sum ~14; dominated by the 248k
softmax 4.1 + hundreds of micro-kernels/step) → launch-count-bound, already
graph-captured in prod; matches the ~40-45ms full-coverage staged spans.
NEXT: regress1 arm (running) delivers the per-step F/m regression; then
author D1' (fp8 DVK slice) behind a flag + accept-gate.
