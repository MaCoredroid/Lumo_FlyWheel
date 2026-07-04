# FR13 recompute — cost model + the node-parallel optimization (recover the −7.6% bit-exactly)

Context: `FR13_SCAN_ALIGN=1 FR13_SCAN_ALIGN_MODE=recompute` replaces the co-resident tree-scan
(`_tree_gdn_kernel` + `h_cache` bank) with `_tree_gdn_recompute_kernel`
(`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:765-916`). It is bit-exact to the per-path
serial recurrence (= the native packed-decode reference) and fixes the tree+cache give-up, at
−7.6% decode / 0 HBM tax. This note records WHY the −7.6% exists and the cheapest bit-exact way
to claw it back.

## 1. Compute-cost ladder (cat8 tree, depths {1,1,2,2,3,3,4,5})

- **native linear MTP**: a straight depth-5 chain ≈ **5** sequential rank-1 steps.
- **cat8 tree, minimal**: **8** unique node-states (each node computed once). Costs more than the
  chain because it verifies MORE candidates (8 vs 5) → higher accept rate → more tokens/forward.
  That extra is the POINT of tree spec, not waste.
- **cat8 tree, plain recompute**: **21** = 1+1+2+2+3+3+4+5. No spine sharing — every node on the
  `0`-branch re-applies `(0,)`, `(0,0)`, … from the root independently (21/8 ≈ 2.6× redundant).

Ladder: `5 (linear) → 8 (tree minimal) → 21 (recompute, no sharing)`.

## 2. Why "spill-free co-resident @ native geometry does not exist"

The tempting shortcut — make the FAST co-resident kernel spill-free at native geometry, precision
without recompute — is structurally impossible:

1. **Co-residency ⇒ big register tile.** All nodes' states live in one bank
   `h_cache = (N_SPAN, BLOCK_V, DIM_K)` fp32 (`:636`), ~2048 regs/lane.
2. **Big tile ⇒ spill ⇒ multi-warp geometry.** Deployed co-resident launch is `BV=16, num_warps=8`
   (`:18-19`); native decode is `BV=32, num_warps=1, num_stages=3` (`:63-67`).
3. **The divergence is the K-reduction, and its rounding is warp-partition-bound.** The GDN step
   reduces over the 128-wide key dim (`tl.sum(..., axis=1)`). FP add is non-associative, so the
   summation ORDER changes the result. With 8 warps the 128 lanes are split 8 ways (per-warp
   partials + cross-warp combine) — a different tree than 1 warp reducing all 128. w8 basis ≠ w1
   basis ⇒ the measured 0.0289 state gap. The basis is fixed by `num_warps`.
4. **You can't dial the co-resident kernel back to w1.** One warp makes the bank spill HARDER, not
   less; and `BLOCK_V` is the VALUE tile, not the `DIM_K` reduction axis, so shrinking it neither
   shrinks the bank nor changes the 128-wide reduce.

|  | shares spine (fast) | native w1 rounding (precise) |
|---|:---:|:---:|
| **co-resident** (one bank) | ✅ | ❌ bank spills → w8 → divergence |
| **recompute** (one tile/node) | ❌ must replay (2.6×) | ✅ spill-free → w1 → bit-exact |

No cell is both. Co-residency IS the spill IS w8 IS the divergence. The only route to native w1 is
one node-state at a time (a single `[BLOCK_V, DIM_K]` tile = recompute), which loses the shared
bank and forces ancestry replay.

## 3. The −7.6% is occupancy + per-program serial length, NOT HBM

Recompute has 0 HBM tax (it re-reads ancestor k/v/g/β that are tiny + L2-cached). The −7.6% is:
- **low occupancy**: native `num_warps=1` = one warp/program → weak latency-hiding on a
  memory-bound decode.
- **long serial chain per program**: the current grid is
  `rgrid = (num_vh, cdiv(dim_v, BV))` (`:1924`) — the NODE dimension is SERIAL inside each program
  (`for i in tl.static_range(0, N_PAD)`, `:835`), so one program grinds all 21 steps in sequence.

## 4. The fix — node-parallel recompute (bit-exact, attacks the −7.6% directly)

The N_PAD node replays are **embarrassingly parallel**: each reads shared read-only data
(`strict_mask`, `h0`, ancestors' k/v/g/β) and writes only its own output row — zero cross-node
dependency. So add the node axis to the grid:

```
rgrid = (num_vh, cdiv(dim_v, BV), N_PAD)     # one program per (head, v-tile, node)
```

Each program then replays ONE node's path (1–5 steps); the 8 nodes run in parallel. Wall-clock ≈
the longest single path (5), not the sum (21). ~8× more programs → more warps in flight → better
occupancy (the thing w1 costs) AND each program's chain shrinks 21→≤5. Both effects push toward
recovering the −7.6%.

**Why this beats depth-incremental (the 21→8 dedup):** it stays BIT-EXACT. Each node is still an
isolated in-register replay — no materialized parent, no handoff, so none of the `state+0.0` /
gather rounding risk (`:794-798`). Purely a grid + program-id change; the math is untouched, so it
can only move speed (up or down), never correctness.

**Options ranked by risk:**
1. **Node-parallel recompute** — bit-exact, hits the −7.6% via occupancy. *Best first try.*
2. **Ship plain recompute** — −7.6%, proven to engage.
3. **Depth-incremental (21→8)** — only helps if compute (not occupancy) were the limiter, and it
   reintroduces handoff-rounding risk. Weakest, given decode is memory-bound.

(#1 and #3 compose, but #1 alone is the clean, safe move.)

## 5. Caveat (settle empirically)

At batch=1 on GB10, whether there's room for 8× more programs — or whether 48 heads × dim-tiles
already saturates the SMs — determines if node-parallel actually recovers the −7.6%. The w1 −7.6%
suggests occupancy IS the limiter (so it should help), but it's a cheap, correctness-safe test: a
grid + `pid` change, bit-exact by construction.
