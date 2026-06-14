# FR13 — fundamental speed: native runs at 45% of peak; ~1.45-1.55x s/fwd reachable on GB10, lossless

Date 2026-06-14. Workflow `wmxi1gypm` (`wf_52c62123-a3b`), Verify **holds=FALSE** (bucket-split optimistic;
core findings sound). Raw: `research/fr13_workflows/fundamental_speed_floor_wmxi1gypm.raw.json`. The BUILD is
queued AFTER the lossless 22->3 fix + OPT-1.

## The fundamental gap: native = 45% of peak bandwidth
Weight-bandwidth floor = 26.9 GB / 273 GB/s = **98.6 ms/fwd** (M-independent, dense non-MoE, hard wall —
can't cut bytes without lossy quantization). Native MTP-5 = 218.2 ms = **2.21x floor = only ~45% of peak**
(123 GB/s effective; matches Hazy Research ~50% for single-seq decode). So 119.6 ms/fwd is slack: ~no
cross-layer weight prefetch (dominant ~55-75 ms), per-layer launch/dispatch (~20-40 ms, ~525 host
launches/fwd vs ~12 graph), sync gaps (~15-30 ms). CAVEAT: per-kernel times NOT measured (nsys export empty);
the split is literature-anchored, not traced.

## KEY SURPRISE: there is NO GB10/sm_121 fp8 config -> generic default
`get_w8a8_block_fp8_configs` (fp8_utils.py:803) has no Spark JSON -> falls to DEFAULT (BLOCK_SIZE_M=64,
GROUP_SIZE_M=32, num_warps=4, **num_stages=2**). At decode M=6-10 with BLOCK_SIZE_M=64, only ~9-16% of the
M-tile rows are real; num_stages=2 = shallow double-buffering that barely hides LPDDR5X weight-DMA latency.

## Ranked fundamental optimizations (GB10-feasible, lossless)
- **OPT-A (FIRST, highest-ROI): GB10/sm_121-tuned skinny-GEMV config** — BLOCK_SIZE_M=tree-rows, num_stages=3-4,
  tuned for M=6-10 + LPDDR5X latency-hiding. **FEASIBLE NOW (config JSON drop / default-bucket override), pure
  Triton, no Hopper primitives. LOSSLESS BY CONSTRUCTION**: BLOCK_SIZE_K=128 is pinned by the block-scale
  contract, so changing BLOCK_SIZE_M/GROUP_SIZE_M/num_warps/num_stages does NOT reorder the fp32 K-accumulation
  (verified fp8_utils.py:774/784). Touches NATIVE's own un-tuned GEMM path -> beats native fundamentally.
- OPT-C: full CUDA-graph capture of the tree-verify forward (preallocated + on-device accept) — moderate, on
  top of OPT-1; targets the per-layer launch/sync buckets.
- OPT-D: cross-layer async weight prefetch (cp.async multi-stage) — the DOMINANT lever (~55-75 ms) BUT
  RISK-FLAGGED on GB10 (D2 = CUDA C++, bit-exact reduction-order risk; sm_121 lacks TMA-multicast/WGMMA/
  warp-specialization so a true Blackwell megakernel is BLOCKED — only async-prefetch persistent grid is reachable).
- OPT-B: WY one-pass no-copy GDN tree-scan (removes our +2.42 GB tree-state HBM amplification) — hard (FR13 #1).

## Honest beat-native arithmetic
Realistic GB10 target ~**140-150 ms/fwd (60-70% of peak)**, optimistic ~126 ms (78%, Hazy ceiling) — NOT the
98.6 ms floor (GB10 lacks the Hopper/Blackwell primitives for full megakernel saturation). vs native 218 ms =
**~1.45-1.55x faster on s/fwd alone**, and the accept edge (~3.18 vs ~3.07 tok/fwd) compounds it into a bigger
TPS win. The hard wall is the 98.6 ms weight-bandwidth floor; the practical asymptote is ~1.3-1.5x floor.

## Build order (after lossless 22->3 + OPT-1 committer): OPT-A -> OPT-C -> OPT-D(risky) -> OPT-B
Each gated bit-exact (byte A/B + per-token argmax probe). Pairs with
[[reference_fr10_speed_measurement_pitfalls]], [[reference_gb10_gdn_backend_fla]],
[[feedback_build_deliverable_form_once_contract_proven]], [[feedback_flag_gate_metrics_reuse_infra]].
