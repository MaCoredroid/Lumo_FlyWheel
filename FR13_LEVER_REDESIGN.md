# FR13 lever redesign — hardware-measurement-driven (2026-07-25)

## The hardware frame that ranks everything

Decode-only, cache-ON, B-aware floor (now recorded per arm by fr13_measure):
`floor(B) = max(weight-read 98.6ms, 0.54ms/row x rows_per_step)`.

| regime | rows/step | floor | measured (tree) | gap |
|---|---|---|---|---|
| B=1 (eps 1) | 22 | 98.6ms | ~330-400ms | ~3.5x |
| B=4 deploy (eps 2.2-3.3) | 48-70 | 98.6ms | 473-663ms | 4.8-6.7x |
| B=8 (eps ~6-7, sweep pending) | ~140-155 | ~98.6ms (crossover ~183 rows) | TBD | TBD |

Marginal cost per extra co-resident event: hardware ~12ms (tree, 22 rows x
0.54) vs measured ~140ms. **The 10x marginal slope is the campaign.** Native's
marginal: hardware ~3ms, measured ~49ms — native wastes 16x on the margin but
has 3.7x fewer rows to waste on.

## Composition decision (g3 + isolation-gate data)

g3 (sealed 6-lever stack): 24.64 tps @ eps 2.16, step 473ms — bar17r2-class,
NOT g1-class (36.36 @ 2.67, step 397ms). Isolation gates already ranked the
levers: PG 35.22, HC 36.69, CPG 34.16, flags 33.35 (winners) vs nodebank
28.05, cap 29.62, wb 31.14 (**all below the 32.14 no-lever baseline**).

**Lean stack** (tree_lean_b4, running): PG + CPG + FLAGS + SUBTREE.
**Unbake-and-DELETE list** (executes when the lean arm confirms, per the
delete-dead-flags discipline; git preserves):
- FR13_CONV_NODEBANK family: conv_nodebank_get/preseed/dst_rows (kernel
  module), patcher dual-arm writeback wrap, bank fetch, committer leaf bank
  read, builder preseeds. ~150 kernel + ~700 patcher lines.
- FR13_SPEC_BLOCKS_CAP: structurally tied to nodebank (capped pages need
  bank storage for replay reads) — deletes with it. The cache-hit-rate
  concern it addressed moves to the mamba_block_size 1024->8192 route
  (project_fr13_apc_blocksize_fix, queued big-N).
- FR13_CONV_WB_BATCHED: batched-writeback kernel + capacity-keyed staging.
  ~200 lines. (The single-arm-writeback redesign is MOOT if nodebank goes.)

Keep baked: sync-kill, SLOT_REORDER, ATTN_KV_REMAP (correctness), stateless
core, PG, CPG, FLAGS_INKERNEL, SUBTREE_PARALLEL.

## Redesign queue (ranked by hardware-measured size at deployment eps)

1. **R4: full drafter-loop CUDA-graph capture (~90ms/step, the largest).**
   dfwd ~93-95ms is host python BETWEEN piecewise-captured pieces (2g:
   propose python = 20% of window; meta-reuse A/B PROVED the metadata builds
   are ~0 of it — the cost is piecewise dispatch + set_forward_context +
   sampling glue x4 iterations). The dmr selfcheck result is the key
   enabler: iterations 2..N are FIELD-IDENTICAL in metadata construction,
   i.e. the loop body is graph-invariant — the whole 4-iteration loop can
   capture as ONE graph per batch size. Design: dedicated graph runner
   around the spine loop (inputs: hidden, tokens, positions; outputs: 4
   draft tokens + hidden), device-side metadata increments (already device
   tensors + the fused slot-mapping kernel), _greedy_sample stays in-graph
   (argmax). Merged-drafter host branches (fired/skip) become captured
   variants keyed (batch, flavor) or device selects.
2. **R3/soup: gather-chain fusion (~15ms/step at B=1, ~3x at deploy eps).**
   The pre/post-scan index_select/gather soup in the patched forward —
   fuse into single kernels; candidates from the verify-kernel-first
   differential (norms/gather-soup +20 > attn-rows +14).
3. **Norms (~8.5ms at B=1, row-scaled).** Same 48-layer norm kernels over
   22 rows vs native 6 — data-movement bound; wins come from row-count
   reduction (NOT tree reshape — anti-solution) or fusing norm into
   neighbor ops where vLLM hasn't already.
4. **Committer -> floor (~4ms/event target).**

## Test protocol per lever (established today)

Offloaded real-SWE arms only (agent never local), cache-ON, decode-only
metric + B-aware floor; 4-task at B=4 for iteration, B-sweep table
(tree/native x B1/B4/B8) as the frame; every record read eps-matched;
byte/equivalence selfcheck gate BEFORE any speed read (dmr pattern).
