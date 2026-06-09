# FR13 h_cache scaling — FUTURE WORK (workflow wozd2k89a, source-verified CPU; ptxas/TPS = LIVE)

Status: NOT a blocker for the current MTP-5 e2e gate (the kernel is bit-exact at BV=16 today). This is a deferred TPS/scaling item — recorded so it isn't re-discovered.

## The finding (corrects an earlier hand-wave: "the tree-scan has no HBM tax")
The register-resident state cache `h_cache = tl.zeros((N_PAD, BLOCK_V, DIM_K), fp32)` (fr10_gdn_tree_kernel.py:277) is sized by NODE COUNT, not depth. `h_cache_KB = N_PAD * BV * 128 * 4 / 1024 = N_PAD*BV/2`.
- MTP-5 (~10 nodes) -> `padded_nodes(10)` = N_PAD=16 (L74-76). At BV=16: **128 KB/program = 256 fp32 regs/lane** at the default 4 warps (no num_warps= at L548) -> **> GB10's 255-reg/thread cap -> SPILLS to local mem = LPDDR5X 273 GB/s** (the same pool decode already saturates). 128 KB also > 99 KB shared cap (can't park in SRAM). So the spill is REAL at the DEPLOYED tree size, not just hypothetical large trees. Crossover is exactly N_PAD=16.
- N_PAD=32 (any drafter wider than ~14 nodes) = 256 KB = infeasible in BOTH the register file (512 regs/lane even @8 warps) and SRAM. Hard wall.
- B does NOT enter h_cache size (the tile is per-(v-head, V-tile), launch L547) -> this is a TREE-SHAPE problem, not a batch problem.

## The fixes (ranked)
1. **NOW / interim (1 line):** `num_warps=8` at the launch (L548). Halves per-lane pressure (256->128 regs/lane at N_PAD=16) -> kills the hard spill WITHOUT shrinking the tile (keeps BV=16's proven bit-exact 0.0). Re-confirm raw out_vs_native_max_abs==0.0 at N_PAD=1 AND 16 (thread-mapping change could perturb op-order). Alternative: BV=8 (largest BV register-resident across all families, predicted bit-exact). This is enough for the current MTP-5 deliverable.
2. **FUTURE (kernel rewrite) — RECOMPUTE-from-spine (STree "activation replay"):** drop the full `[N_PAD,BV,DIM_K]` state cache; cache only per-node activations (q/k/v/g/β ~1.5 KB/head, ~42x smaller) and replay the rank-1 ancestry from h0 per node. Properties (verified vs source/math): (a) **bit-exact** by construction — identical native rank-1 op order (L338-341), only the SOURCE of h_j changes (replay instead of `h_cache` read at L283); (b) **Triton-expressible** — reuses the SAME `static_range`+`tl.where(strict_mask)` ancestry machinery already at L280-286; (c) **spill-free at ANY tree size** — state mem is O(1) (one `[BV,DIM_K]` working tile), activation cache is O(N*K-vec); (d) **cheap on a bandwidth-bound decode** — worst-case full re-walk (N=14, depth ~6, 48 heads, ~165 MFLOP) ~5-33 µs = 0.005-0.03% of one ~99 ms forward; avoiding ONE spill round-trip of the full state dwarfs it (decode arithmetic intensity <1 FLOP/byte -> trade ~free FLOPs to make the spill structurally impossible). This is the route to support drafters/trees wider than ~14 nodes.

## What was REJECTED
- **DFS-ancestry-stack (depth-bounded cache):** NOT Triton-idiomatic — no recursion, no data-dependent loop trip counts; a true push/pop stack with runtime depth would unroll to a fixed DEPTH bound + `tl.where` = you reinvent the exact mask-select already at L280-286, sized to depth. No production tree-decode kernel (SpecInfer/Medusa/EAGLE/DEFT/STree) does in-kernel DFS traversal; all pack N nodes + an N×N topology mask in one fused pass. Depth-only relief, strictly weaker than recompute. Drop it.
- **BLOCK_V=1 (8 KB tile):** NOT FEASIBLE bit-exact — the reduce tile IS the state tile; a 1-V-row reduce collapses the degenerate [1,128] butterfly -> 1.19e-7 drift (FR13_BV_SPILL_VERDICT.md §1).
- **chunked/streaming state:** reintroduces the +35.8% HBM state-traffic tax FR13 fled (same 273 GB/s pool).

## LIVE gate when this is picked up (none settled on CPU)
(a) raw out_vs_native_max_abs==0.0 at N_PAD=1 AND 16 for the chosen config (read the RAW float, NOT scripts/fr10_scan_output_replay_gate.py's atol=1e-3 exit code); (b) ptxas -v / TRITON_KERNEL_DUMP spill-bytes==0 at N_PAD=16; (c) decode-TPS (metrics OFF, B=4) vs the BV=16 baseline — justified only if (c) shows measurable TPS.
