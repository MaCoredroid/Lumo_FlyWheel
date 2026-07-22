# FR13 verify-profile findings — where the tree tax actually lives (2026-07-22)

**Goal:** beat native MTP-5 (42.74 tok/s measured_tps_fullstep_wall, 16-task live gate)
via verifier+committer work on the 21-node tail6 tree, accept pinned at 4.286 (golden
gate: same drafter/tree). Decide-by-profile, not by guess: two nsys capture campaigns
under `output/fr13_verify_profile/`.

## Captures

1. **Static-probe arm** (`nsys_tail6_21node/`): B=1, 4 static SWE-shaped ~1k-token
   prompts, greedy, 64 tok. Diagnostic-only (per feedback_live_swe_verified_only).
   186 drafts in 69s of probe walls.
2. **Real-task arms** (`kvremap_tail6_nsysreal/`, `native5_nsysreal/`): the DECISIVE
   pair — identical conditions (task astropy-14598 via the full campaign driver:
   offloaded qwen-code, temp 0.6 forced, nudge-free, cache-ON, CONC=1), 250s
   metrics-bracketed windows, 733 (tail6) vs 982 (native5) drafts.
   Capture infra notes: nsys sw-trace (`cuda,cuda-sw,nvtx`) mandatory on GB10;
   nsys roughly doubles boot (DELAY=840/1200); the variant's teardown must be
   SIGSTOP-frozen during the window or a task finishing in-window `docker rm -f`s
   the container mid-report-write (lost attempt 1 exactly this way).
   Reducer: `output/fr13_verify_profile/reduce_tail6_nsys.py` (category buckets,
   per-draft, in-graph/eager split; needs the 2026.2.1 nsys binary for export).

## The differential (per-draft ms, in-graph = inside the captured decode graph)

| category | tail6 | native5 | Δ tree-specific |
|---|---|---|---|
| norm/elementwise (gather/scatter/copy soup) | 21.15 | 1.04 | **+20.1 — #1 lever** |
| gdn_tree_scan (vs native fused_sigmoid in other) | 18.43 | 0 | +18.4 gross / ~+15 net |
| attention (22 query rows vs 6; KV shared) | 34.30 | 20.35 | **+14.0** |
| gemm_mlp_proj | 125.22 | 125.79 | **−0.6 = ZERO** |
| lm_head (eager both: 5× drafter gemvx @15.1ms + verify logits ~12ms) | 77.36e | 77.85e | **ZERO** |
| eager committer-side (other + sampler) | 18.5e | 3.9e | +14.6 |
| **total tree tax** | | | **≈ +67ms/draft** |

Per-committed-token GPU: tail6 61.7ms vs native5 58.5ms — only 5% apart; accept
(4.29 vs 3.42) nearly pays the tree tax already.

## Hypotheses killed by this data

- **GEMM M-tile/row-scaling: DEAD.** M=22 and M=6 decode GEMMs cost identically —
  weight-read-bound, rows free. (Earlier row-slope economics attributed the verify
  gap here; wrong bucket.)
- **GDN scan as the main verify cost: DEAD** (already suspected from the 07-21
  attack-scoping "-1.7% PARENT_GATHER" refutation; now measured at 18.4ms/draft,
  ~6% of the step).
- **lm_head as a tree lever: DEAD** (identical both arms; the 5×15.1ms drafter gemvx
  inefficiency is real but SHARED → doesn't count toward the beat).

## Measured norm-soup fingerprint (per GDN layer per draft, tail6-only, in-graph)

- 2× `index_elementwise_kernel` @84.6µs → 8.2 ms/draft (single biggest item)
- 4× `elementwise_kernel<128,4>` @19.7µs + 4× `<128,2>` @4.5µs → 4.7 ms/draft
  (prime suspect: the 4 replay-ring `.copy_()` calls, patch:5127-5138)
- 2× `vectorized_gather_kernel` @16.4µs → 1.6 ms/draft
- 1-2×/layer each: unrolled_elementwise, indexSelectSmallIndex, CatArrayBatchedCopy,
  vectorized_elementwise, elementwise_kernel_with_index → ~1.5 ms/draft combined

## Build plan (attack order by measured size; each behind a same-boot byte gate)

1. **B1 ring-export-in-scan**: `_tree_gdn_kernel` already receives k/v/raw_a/raw_b —
   `tl.store` them to the replay ring inside the scan (RING_EXPORT constexpr; v
   partitioned by BLOCK_V grid, k/a/b single-program-gated), patcher skips the 4
   per-layer copies. ~4-5 ms/draft.
2. **B2 index_elementwise pair** (call-site map in flight, workflow wf_704284f5-9f9):
   fold into the fused-conv/prep kernels or a single fused staging kernel. Up to
   ~8 ms/draft.
3. **Committer overlap-under-drafter** (side stream + event fence; requires the
   sync-free launch_attn_kv_linear_remap rewrite — identity-safe fixed-shape
   scatter, no bool()/numel() syncs). Hides most of the +14.6 eager committer tax.
4. Scan E2/E3 (exact loop span; h_cache internal-nodes-only) — now worth only
   ~6-9 ms/draft of the scan's 18.4; do after 1-3.
5. Attention splitkv config for 22-row decode — investigate after 1-4.

Levers NOT counted toward the beat (shared with native5): drafter gemvx efficiency,
any KV-cache compression, drafter graph capture. Any of these applied ⇒ re-baseline
native5 with the same change (honest-comparison rule).

## Call-site map of the norm-soup (workflow wf_704284f5-9f9, 2026-07-22)

Measured-fingerprint → call-site attribution (all in the captured decode graph):
- **84.6µs `index_elementwise` pair** = (1) `conv_state.index_copy_` patch:3696 —
  per-node conv-state write-back SCATTER, 22 page-strided bf16 rows ~17MB traffic
  (page-strided dst = low effective BW); (2) same-kernel `index_copy_` in
  `replay_conv_state_linear_remap_prepared` (fr13_tree_conv_fused.py:394 via
  patch:2799), ~6 rows.
- **Under-counted 2nd lever**: `_scatter_gather_elementwise` @73.2µs 1×/layer
  (~3.5ms/draft) = `fused_tree_conv_state_rows` gather (528 rows → 8.6MB
  materialization, fr13_tree_conv_fused.py:270) + transpose-`.contiguous()` copy
  (:273) — feeds the 3696 scatter.
- **`<128,4>` @19.7µs quartet** = ring k/v copies ×num_spec_decodes (B1 target ✓);
  **`<128,2>` @4.5µs quartet** = conv tap fp32 accumulation (bias + 3 ordered adds,
  fr13_tree_conv_fused.py:246-250), NOT ring a/b (those are 2 of the small copies).
- 2× `vectorized_gather` = conv window gather (patch:3198) + state-rows gather.
- All 7 `.contiguous()` on scan inputs = NO-OPS (verified provenance). The 5419
  self-copy is DEAD under EAGER_PACK. Launch site itself adds zero aten kernels.

## B-series build plan (revised by the map)

- **B1 (DONE, gated)**: FR13_RING_EXPORT — in-kernel ring staging. Offline byte
  gate PASS (tail6-21n-BV8 + cat9-9n-BV16; out + all 4 rings byte-identical).
- **B2a (biggest, ~8.3ms/draft)**: fuse state-rows gather + transpose-contiguous +
  conv_state.index_copy_ (3696) into ONE Triton kernel writing gathered source
  rows directly to the page-strided conv_state destinations (skip the 8.6MB
  round-trip and the aten scatter).
- **B2b**: replay-remap index_copy (394) — fuse similarly or absorb into the
  committer-overlap side stream.
- **B2c (~1.5-2ms/draft)**: fold tap multiply + fp32 cast + 4 accumulation adds
  into an extended fused_tree_conv_taps_acc.
- Then: committer overlap-under-drafter; live same-seed gate; 16-task deploy gate.
