# FR13 — DECISION: match native GDN-scan launch geometry (BV=32, num_warps=4)

Date 2026-06-14. Follows `FR13_BF16_FP32_SEAM_SCAN_BIND.md` (the seam scan that found the GDN-scan
**launch geometry** — not the conv tap — is the open diffuse-carrier seam). **User decision: match
native.** The op body is already byte-identical to native; only the launch geometry differs.

## The seam (recap)
Our GDN tree-verify scan runs **BV=16 / num_warps=8** (`fr10_gdn_tree_kernel.py:18`, num_warps=8
at the scan launches); native `fused_sigmoid_gating_delta_rule_update` runs **BV=32 / num_warps=4**.
Same fp32 math, op-for-op identical body — but the warp/lane map reshapes the `tl.sum(axis=1)`
reduction tree + FMA scheduling → ~1 bf16-ULP/node on every GDN node on every ~48 GDN layers =
the "diffuse L0-L58 GDN accumulation" carrier, amplified ~32× by the gate 1/rms. Never gated below
atol=1e-3 vs native. **DECISION: align our launch geometry to native (BV=32, num_warps=4)**, gated
on RAW max_abs == 0.0 (not atol=1e-3). NB: the old "BV=16 is bit-exact / NO BV change"
(`FR13_BV_SPILL_VERDICT`) compared BV=16 vs BV=8 (both ours), NOT vs native's BV=32/warps=4 — a
different question; vs native our geometry is the seam.

## What BV and N_PAD are
- **BV = BLOCK_V** (`:18`) = Triton tile size along the value dim of the GDN scan. A *launch*
  constexpr (not a model param). Sets the `tl.sum(axis=1)` reduction-tree shape, the `h_cache`
  register-tile size, and ptxas FMA scheduling.
- **N_PAD** = padded tree-node count = `1 << (n-1).bit_length()`, next pow2 ≥ tree size, **capped
  at 16** (`:75-77`). cat9 = 9 draft nodes (10 w/ root) → **N_PAD=16**. The tree verify kernel
  carries the recurrent state for ALL N_PAD nodes at once (`h_cache = tl.zeros((N_PAD, BLOCK_V,
  DIM_K))`, `:458`) — the tree co-residency.

## THE SPILL BLOCKER (the math, real numbers: DIM_K=128, cap 255 regs/thread)
Per-thread h_cache registers = `N_PAD·BV·DIM_K / (num_warps·32)`:
| geometry | per-thread h_cache regs | fits (≤255)? |
|---|---:|---|
| ours BV=16 / warps=8 / N_PAD=16 | 16·16·128/256 = **128** | yes |
| **native BV=32 / warps=4 / N_PAD=16** | 16·32·128/128 = **512** | **NO — 2× over → launch fail / heavy spill** |
| BV=32 / warps=16 / N_PAD=16 | 16·32·128/512 = 128 | fits, but warps=16 ≠ native 4 → WRONG reduction tree |

- **Max tree size at native's BV=32/warps=4 ≈ N_PAD=4** (`N_PAD·32 ≤ ~128`). cat9 (N_PAD=16) is 4× over.
- The bind: native's exact (BV=32, warps=4) does NOT fit our 16-node register cache; the only warp
  count that fits N_PAD=16 at BV=32 (warps=16) gives a different reduction tree than native's
  warps=4 → would not match. So matching native's geometry is **blocked by the h_cache register
  footprint** at the deployed N_PAD=16.
- (Spill = register overflow → compiler pushes the tile to off-chip "local" DRAM → kernel runs
  much slower, or the launch fails outright with "too many resources requested".)

## Plan
1. **MEASURE FIRST (non-destructive, ships nothing):** a 1-boot in-process A/B — run the scan at
   BV=16/warps=8 vs BV=32/warps=4 on bit-identical captured layer-0 inputs, compare RAW max_abs vs
   the native `fused_sigmoid_gating` reference (same boot). Confirms (a) BV=32→native RAW 0.0
   (geometry is the seam + the fix), and (b) whether BV=32 compiles/launches at N_PAD=16 or spills.
2. If BV=32 fits + → native 0.0 → adopt it (re-gate the per-token argmax probe; weigh any spill
   speed cost). If it spills/fails → the FUTURE-WORK cache fix is required first.

## FUTURE WORK — remove the spill limit entirely (decouple register footprint from tree size)
The root cause = the verify kernel holds ALL N_PAD nodes' state in registers. Options (the
replay kernel `:588` already has NO h_cache = existence proof this is doable):
1. **Recompute-state-from-spine** (`FR13_CACHE_SCALING_FUTURE`) — recompute the path's recurrent
   state on demand instead of caching all N_PAD nodes → h_cache shrinks to one path → BV=32/warps=4
   fits at ANY tree size. Preferred.
2. **Node-tiling** — process N_PAD nodes in sub-tiles so the live tile stays small.
3. ~~**Shared-memory h_cache**~~ — **DEAD (CORRECTION, ranking wf wwrov62yp):** GB10/DGX Spark is
   compute capability 12.1 with **~99 KB shared-mem per block** (NOT 228 KB — that's the datacenter
   B200/sm_100 number I wrongly cited). The h_cache tile is **256 KB at BV=32** (and **128 KB even at
   shipped BV=16**) — both over the 99 KB cap, so it does NOT fit in SMEM either. Plus Triton has no
   primitive to place a non-`tl.dot` accumulator in SMEM. Struck on both the cap and expressibility.
This also lifts the **N_PAD ≤ 16 cap** (`:76-77`) → larger trees become possible (relevant to any
future suffix-fusion / deeper trees). Pairs with [[reference_diffuse_gdn_accumulation_explained]],
[[feedback_math_correct_vs_bitexact]], [[feedback_no_reroute_reward_hacking]],
[[project_fr13_pipeline_lock]].

---

## MEASURED spill numbers (GPU compile-test wp5hsu63v, 2026-06-14) — REAL ptxas, holds=True

Compiled `_tree_gdn_kernel` (committed, unchanged) at each geometry on GB10 (triton 3.6.0,
cu130-nightly, NO model load); read `compiled.n_regs`/`.n_spills` off the Triton handle (numbers
DIVERGE from the math prediction in every config ⇒ real, not echoed).

| cfg | geometry | tree | pred regs | **measured n_regs** | **spill B/thread** | verdict |
|---|---|---|---:|---:|---:|---|
| C0 | BV=16, warps=8, N_PAD=16 | cat9 (current) | 128 | **254** | **0** | FITS (right at the 255 cap) |
| C1 | BV=32, warps=4, N_PAD=2 | cat2 | 64 | 140 | 0 | FITS |
| C2 | BV=32, warps=4, N_PAD=4 | 3-4 node | 128 | 235 | 0 | FITS |
| C3 | BV=32, warps=4, N_PAD=16 | **cat9 native-geom** | 512 | 255 (clamped) | **636** | **SPILLS hard (runs, no 701)** |
| C4 | BV=32, warps=8, N_PAD=16 | cat9, more warps | 256 | 255 (clamped) | **96** | spills 6.6× less, still spills |

**KEY REFRAME — the spill is a SPEED cost, not a correctness wall.** Matching native's full
geometry (BV=32/warps=4) at cat9 (N_PAD=16) does NOT fail to launch (no CUDA 701) — ptxas clamps
to 255 regs and **spills 636 B/thread to local memory**: the kernel runs, just slow. So we can get
the (presumably) lossless BV=32/warps=4 config NOW at a speed penalty, then optimize. Findings:
- Small trees (cat2 N_PAD=2/4) match native's geometry cleanly — 0 spill. The spill is purely a
  large-tree (N_PAD=16) problem.
- `num_warps` is the dominant lever: warps=8 cuts the spill 6.6× (636→96 B) but doesn't clear it
  (256 regs > 255), and warps=8 ≠ native's warps=4 reduction anyway.
- Only the current BV=16/warps=8 is 0-spill at cat9 — but it's the geometry that DOESN'T match
  native (the open seam). So: BV=16 = fast but seam; BV=32/warps=4 = matches native but spills.

**Two-step path this implies:** (1) **confirm lossless** — the A/B verify (does BV=32/warps=4
reproduce native to RAW max_abs==0.0?) can run despite the spill (it launches), answering the
*correctness* question independent of speed. (2) if lossless, the **cache workaround** (spill-rank
wf) removes the 636 B spill → lossless AND fast. The spill is no longer a wall — it's a speed
optimization that follows the lossless confirmation.

---

## SPILL-WORKAROUND RANKING (wf wwrov62yp, CPU; red-team holds=False = over-stated, order holds)
Rank the 3 spill-fixes by lossless + speed risk vs our impl. **RECOMPUTE ≫ NODE-TILING ≫ SMEM(dead).**

1. **RECOMPUTE-state-from-spine — THE ROUTE (the only register-feasible one at native BV=32/warps=4).**
   - LOSSLESS by construction: it relocates only the SOURCE of a node's seed state (replay ancestors
     via the shared `_gdn_node_step` from b_h0, using the EXISTING ancestry machinery :459-467) — it
     never touches the [BV=32,DIM_K=128] tile, the two `tl.sum(axis=1)` reductions (:379,:382), or the
     op-order. The reduction tree (set by BV+warps on that tile) is preserved → lossless PROVIDED
     recompute runs at BV=32/warps=4.
   - FITS: holds ONE [BV,DIM_K] tile = ~64-90 regs/thread at BV=32/warps=4 (vs h_cache's 512) →
     comfortably under 255, O(1) in tree size → fits native geometry at N_PAD=16 AND lifts the
     N_PAD≤16 cap.
   - SPEED ≈ neutral-to-positive: the scan is a tiny sliver of the ~99 ms GEMV-dominated forward; the
     extra ~2.57× rank-1 steps ≈ 5-33 µs (and it AVOIDS the 256 KB spill round-trip). (Red-team
     corrected the *mechanism*: not "bandwidth hides it" — the scan is latency-bound — but "the scan
     is negligible vs the weight-GEMV.")
   - EXISTENCE PROOF: `_tree_gdn_replay_kernel` (:588) already runs no-h_cache spill-free on the
     identical body. **CAVEAT (why red-team holds=False): that replay kernel runs at warps=8 (not
     native 4) AND is BROKEN LIVE (gate-4, accept/event 2.02→1.58 wiring seam). So recompute's
     losslessness is a GPU OBLIGATION, not a CPU-proven fact — "offline-bit-identical ≠ live-correct."**
2. **NODE-TILING — inferior.** Our tree is wide/shallow + multi-parent (node 0 parents 1,12,13), so
   tiles either carry a large cross-tile live set (the all-resident problem re-expressed) or spill
   cross-tile parents to HBM (reintroduces the +35.8% state-traffic tax FR13 fled). No production
   tree kernel tiles over nodes — the multi-parent boundary handoff is why.
3. **SMEM — DEAD** (256 KB ≫ 99 KB GB10 cap; 128 KB even at BV=16; no Triton SMEM primitive). See the
   corrected future-work note above.

**GATING NEXT (GPU, the decisive test for the whole BV path):** (1) the cheap A/B first — run the
EXISTING cache kernel at BV=32/warps=4 (it RUNS despite the 636 B spill) vs native → **RAW max_abs==0.0?**
That answers "is the launch geometry actually THE seam" independent of recompute. (2) If yes →
build the recompute-verify variant at BV=32/warps=4 and confirm in one boot: 0 spill (ptxas -v) +
RAW 0.0 vs native at N_PAD=1 AND N_PAD=16. If the A/B is NOT 0.0, the geometry isn't the (sole) seam
and S1 may be the irreducible floor — recompute won't rescue it. Lossless is a GPU obligation here,
not a CPU conclusion.
