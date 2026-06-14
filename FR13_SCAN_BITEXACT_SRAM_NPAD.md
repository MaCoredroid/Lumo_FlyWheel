# FR13 SCAN bit-exactness × SRAM × N_PAD=16 — resolution (design, CPU/read-only)

Workflow: scan-bitexact-sram-npad. CPU-ONLY, READ-ONLY. 2026-06-14. Pairs with
`FR13_BV_SPILL_VERDICT.md` (w921xvgzx), `FR13_CACHE_SCALING_FUTURE.md` (wozd2k89a),
`FR13_BV_NATIVE_MATCH_BIND.md` (wp5hsu63v ptxas, wwrov62yp ranking, **2026-06-14**),
`FR13_BF16_FP32_SEAM_SCAN_BIND.md` (wf_55f0d466). Bug-class rows quoted: **#10 codegen-identity**
(shared-source ≠ shared-SASS; gate by int-view/byte-A-B, NEVER atol; re-arm per toolchain) and
**#12 measurement traps** (raw counters only; label every estimate; no hand-rolled TPS÷accept).

---

## TL;DR — the task's premise is HALF-SUPERSEDED; read this first

The task is framed on the OLD `FR13_BV_SPILL_VERDICT` numbers: "deployed cat9 = BV=16/warps=8 is an
interim SPILL fix; N_PAD=16/BV=16 spills." **Two facts measured AFTER that doc change the answer:**

1. **The deployed BV=16/num_warps=8 does NOT spill at N_PAD=16.** REAL ptxas (wp5hsu63v, GB10
   triton 3.6.0 cu130-nightly, today): config C0 = **254 n_regs, 0 spill B/thread, FITS** (right at
   the 255 cap). The "N_PAD=16/BV=16 → 256 regs/lane → spill" arithmetic in `FR13_BV_SPILL_VERDICT`
   §2 / `FR13_CACHE_SCALING_FUTURE` is the **4-warp** case; deployed is **8 warps** (`:843,:1284,
   :1536`), which halves it to ~128 regs and clears the spill. **So the "shrink-to-fit / spill-tax"
   problem the task asks me to resolve does NOT exist for the deployed config.** The SCAN as
   deployed is spill-free at the deepest deployed tree (N_PAD=16).

2. **The deployed BV=16/warps=8 IS output-bit-exact to a per-path serial native reference** — banked
   `output/fr13_accept_only_20260610T002243Z/gdn_scan_warp_gate.json` (`fr13_gdn_scan_warp_gate.py`,
   `torch.equal` + RAW max_abs==0.0): at N_PAD=1 AND N_PAD=16, every node `out_vs_native_max_abs=0.0`,
   `out_bit_exact=true`. (State differs ~1.5e-8 = post-step `state_i`, an off-output fp32-order
   wobble, but the served OUTPUT is byte-identical to that reference.)

**The ACTUAL open tension** (reframed by `FR13_BV_NATIVE_MATCH_BIND` / `_SEAM_SCAN_`, today) is NOT
"BV=16 spills." It is: **is BV=16/warps=8 bit-exact to NATIVE'S OWN FUSED KERNEL geometry, which is
BV=32 / num_warps=4 / num_stages=3?** The warp-gate's reference is a *serial per-path* reproduction,
not native's `fused_sigmoid_gating_delta_rule_update` at its real `[32,128]`/4-warp tiling. Per
bug-class #10, "bit-exact to a serial reference" ≠ "bit-exact to the incumbent SASS." That gap is
the suspected carrier of the diffuse 22-flip (`FR13_DIFFUSE_GDN_EXPLAINED`: native drifts 7× less on
the same model+fp8 = a bit-aligned version provably exists).

**Consequence for the task's PROMISING LEAD (BV=4/warps=4):** it is **off-target**. Native's tile is
**BV=32**. Matching native's K-reduction tree requires matching native's *layout*, and BV=4 produces a
**different** layout than BV=32 (shown below) — so BV=4 is not a native-bit-exact candidate, and the
deployed BV=16/warps=8 is *also* not guaranteed native-bit-exact. The lead as stated ("BV=4 ⇒ native
K-order + spill-free + scales") does not hold. The real resolution is in §3/§4. (I verify the BV=4
layout claim rigorously below rather than just asserting it — and it fails the native-match test, so
I fall to the next option exactly as instructed.)

---

## §1 — The tension with EXACT arithmetic

### 1.0 Ground facts (read from source/ttgir, not assumed)
- SCAN kernel `_tree_gdn_kernel` (`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`): shared body
  `_gdn_node_step` (`:337-383`); the K=128 contraction is **two** reductions:
  `b_v -= tl.sum(state_i * b_k[None,:], axis=1)` (`:379`) and `out_i = tl.sum(state_i * b_q[None,:],
  axis=1)` (`:382`). `state_i` is `[BLOCK_V, DIM_K] = [BV, 128]`; **axis=1 = the K=128 axis**.
- `h_cache = tl.zeros((N_PAD, BLOCK_V, DIM_K), fp32)` (`:458`) — register-resident node-state cache,
  **N_PAD × BV × 128 × 4 bytes**. Bytes = `N_PAD·BV·128·4`; KB = `N_PAD·BV/2`.
- Deployed: `BV=16` (`:18`), `num_warps=8` at all three scan/replay launches (`:843,:1284,:1536`).
- N_PAD families: `NODE_FAMILIES=(2,3,6,8,14)` (`:11`) → `padded_nodes = 1<<(n-1).bit_length()`
  (`:74`) → **N_PAD ∈ {2,4,8,8,16}**. cat9 = 9 nodes → N_PAD=16. Deepest deployed = 14 nodes →
  **N_PAD=16**. (`n_pad>16` raises, `:76`.)
- GB10 sm_121: **255 fp32 regs/thread cap**, **~99 KB shared/CTA** (NOT 228 KB — that is sm_100/B200;
  corrected in `FR13_BV_NATIVE_MATCH_BIND:58-62`), 128 lanes/CTA at 4 warps (256 at 8 warps),
  273 GB/s LPDDR5X (B=1 decode bandwidth-bound → any spill = a real TPS tax on the saturated bus).
- NATIVE geometry (read from the live ttgir
  `/tmp/lumo-l0c-.../fused_sigmoid_gating_delta_rule_update_kernel.ttgir`): the 2-D state tile is
  `tensor<32x128xf32, #blocked>` with
  **`#blocked = sizePerThread=[1,4] threadsPerWarp=[1,32] warpsPerCTA=[4,1] order=[1,0]`**. Both
  K-reductions are `tt.reduce axis=1` over that layout (native ttgir `b_v_125` and `b_o_136`). So
  native's K-reduction tree is: **each lane sums its 4 contiguous K-elems in-thread (sizePerThread
  axis-1 = 4), then a 32-lane intra-warp shfl.xor butterfly; warps live on the 32-ROW axis
  (warpsPerCTA=[4,1]) and never touch K** → there is NO cross-warp reduction on K. *That* is the
  op-order any "native-bit-exact" config must reproduce.

### 1.1 The full table — h_cache bytes, regs/lane, spill, SRAM-park, layout, candidacy
`regs/lane = N_PAD·BV·128 / (num_warps·32)` (fp32, h_cache ONLY; the working tile + q/k/v/g add a
near-constant ~30-60 more, which is why measured n_regs > this prediction). "SRAM-parkable" is N/A
in practice: Triton has no primitive to park a non-`tl.dot` accumulator in SMEM (`FR13_BV_NATIVE_
MATCH_BIND:61`), so SMEM is dead regardless of the byte fit — column kept only to show the cap.

| N_PAD (family) | BV | warps | h_cache KB | regs/lane (h_cache) | >255 spill? | <99KB SMEM? | native K-layout match? |
|---|---:|---:|---:|---:|:--:|:--:|:--:|
| 2 (2-node)   | 32 | 4 | 16  | 64  | no  | yes | **CANDIDATE** (matches native) |
| 4 (3-node)   | 32 | 4 | 32  | 128 | no  | yes | **CANDIDATE** |
| 8 (6/8-node) | 32 | 4 | 64  | 256 | **YES (borderline)** | yes | candidate-if-fits |
| **16 (14-node, cat9)** | 32 | 4 | **128** | **512** | **YES, hard (636 B meas.)** | no | native geom — but **SPILLS** |
| 16 (cat9)    | 32 | 8 | 128 | 256 | YES (96 B meas.) | no | **NO** (8≠4 warps → different tree) |
| 16 (cat9)    | 32 | 16| 128 | 128 | no (fits) | no | **NO** (16≠4 warps → different tree) |
| **16 (cat9, DEPLOYED)** | 16 | 8 | 128 | 128 | **no (254 regs meas., 0 spill)** | no | **UNKNOWN — must gate** (BV16≠BV32) |
| 16 (cat9)    | 16 | 4 | 128 | 256 | **YES** (this is the OLD verdict's "spill" row) | no | NO (BV16≠BV32) |
| 16 (cat9)    | 8  | 8 | 64  | 64  | no | no | NO (BV8≠BV32) |
| 16 (cat9)    | 8  | 4 | 64  | 128 | no | no | NO (BV8≠BV32) |
| 16 (cat9)    | **4** | **4** | **32** | **64** | **no** | yes(32KB) | **NO — re-warps K (see §2)** |
| 16 (cat9)    | 4  | 8 | 32  | 32  | no | yes | NO (BV4≠BV32 + over-warped) |

MEASURED ptxas anchors (wp5hsu63v, real `compiled.n_regs`/`.n_spills`, today — numbers diverge from
the math ⇒ real not echoed): **C0 BV16/w8/N16 = 254 regs / 0 spill (FITS)**; C1 BV32/w4/N2 = 140/0;
C2 BV32/w4/N4 = 235/0; **C3 BV32/w4/N16 (native geom) = 255-clamped / 636 B spill (HARD SPILL, still
LAUNCHES no CUDA-701)**; C4 BV32/w8/N16 = 255-clamped / 96 B (spills 6.6× less, still spills, and
w8≠w4 so wrong tree anyway). The math under-predicts by the working-tile overhead but the FIT/SPILL
verdicts and the lever (warps dominate, BV second) are confirmed.

### 1.2 Which (BV,warps) are even bit-exact CANDIDATES to native
A config can be native-bit-exact ONLY if it reproduces native's `[1,4]/[1,32]/[4,1]` K-reduction
tree. Necessary conditions (from Triton blocked-layout semantics, §2): **BV=32** (so the row axis has
exactly 32 = native's leading extent, letting `warpsPerCTA` sit on rows and keep K un-warped) **AND
num_warps=4** (so warps fill the 32 rows 1:8-per-warp and never spill onto K). That is EXACTLY
native. **Every config with BV≠32 or warps≠4 is, a priori, NOT a guaranteed native-match** — it may
*happen* to compile to the same tree (codegen-dependent, bug-class #10: must be gated, never
assumed), but it is not a candidate by construction. → The only by-construction native-bit-exact
geometry is **BV=32/warps=4**, which **SPILLS at N_PAD=16** (C3). That is the whole tension in one
line: *native-bit-exact ⟺ BV=32/warps=4 ⟺ spills at the deepest deployed tree.*

---

## §2 — Rigorous evaluation of the task's lead: BV=4 / num_warps=4

### (a) Does BV=4 preserve native's K=128 reduction order? — NO. It does not, by two independent arguments.

**Triton layout mechanics.** For a reduce over the contiguous axis (K=128, `order=[1,0]`), the
default blocked encoding is built fast-axis-first: `threadsPerWarp` is laid on the K axis until 32
lanes are placed (32 lanes × `sizePerThread` K-elems = the 128 K-extent → `sizePerThread`[K]=4),
giving `[1,4]/[1,32]` on K — this part is **the same for any BV** (K=128 is fixed). The DIFFERENCE is
`warpsPerCTA`. With `num_warps=W` warps and leading (row) extent M=BV:
- Native BV=32, W=4: M=32 ≥ W=4 ⇒ all 4 warps fit on rows: `warpsPerCTA=[4,1]`, **0 warps on K** →
  the single intra-warp 32-lane butterfly is the entire K-reduction. ✔ (matches the ttgir exactly).
- BV=4, W=4: M=4 ≥ W=4 ⇒ warps STILL fit on rows: `warpsPerCTA=[4,1]` — *naively the same K-layout*.
  **So the task's intuition is not crazy.** BUT two things break the match:
  1. **It's not native's BV.** Native is `[32,128]`. A `[4,128]` reduce is a DIFFERENT op instance
     (different tensor shape) → different ptxas instruction selection/scheduling for the in-thread
     4-elem `sizePerThread` partial sums and the FMA contraction feeding the reduce
     (`state_i*b_k`). Bug-class #10: same source body, different constexpr/shape ⇒ different SASS ⇒
     **not bit-exact by re-execution; must be byte-gated, and it is a DIFFERENT compilation than the
     one whose `[32,128]` tree we're trying to clone.** Cloning native's tree means compiling
     native's shape, i.e. BV=32. BV=4 clones nothing native ran.
  2. **BV=4 is BELOW the native row extent that the deployed kernel's `h_cache`/`offs_v` indexing was
     proven on, and `cdiv(dim_v,BV)=cdiv(128,4)=32` programs** vs native's `cdiv(128,32)=4`. The
     reduction *axis* (K) layout can coincide, but the per-program partial state and the boundary
     `v_mask` change; and crucially the warp-gate's *banked* 0.0 evidence is at **BV=16**, not BV=4 —
     BV=4 has NO banked bit-exact evidence at all.
- BV=4, W=8 (the other listed BV=4 row): M=4 < W=8 ⇒ 4 warps on rows, **4 warps forced onto K**
  → `warpsPerCTA=[4,2]` → the 128-K is now split across 2 warp-groups → a **cross-warp** reduction
  tree (warp-shuffle THEN a 2-warp SMEM/shfl combine) → **definitively a different op-order** → not
  native-bit-exact. (This is the general failure the verdict warned about for small leading extent.)

**Verdict on (a): BV=4/warps=4 may coincidentally share the *K-lane* mapping, but it is a different
tensor shape than native's `[32,128]`, so it does NOT "reproduce native's reduction" in the bug-class
#10 sense — it reproduces a NEW kernel that has never been gated against native. The lead's claim
"per-row K-reduction order IDENTICAL → plausibly bit-exact to native BV=32" conflates "same lane→K
map" with "same compiled reduction," which #10 explicitly separates. SKEPTIC RULING: the lead is
OVERSTATED — fall to §3.**

### (b) No-spill arithmetic at N_PAD=16 — TRUE (this part of the lead holds).
BV=4/warps=4/N_PAD=16: h_cache = 16·4·128·4 = **32 KB** → 32768/(4·32·4 bytes) = **64 fp32 regs/lane**
< 255 ✔, and 32 KB < 99 KB ✔. So BV=4 IS spill-free and SMEM-byte-fits. But SMEM is moot (no Triton
primitive) and the no-spill win is irrelevant because (a) already disqualifies it on bit-exactness.
**(And note: the DEPLOYED BV=16/warps=8 is ALSO spill-free — 254 regs, measured — so BV=4 buys no
spill advantage over what ships.)**

### (c) PERF cost of BV=4 at B=1 bandwidth-bound decode — a real, un-hidden tax.
BV=4 ⇒ `cdiv(128,4)=32` value-tile programs per (layer,vhead) vs BV=16's 8 vs native BV=32's 4 → **8×
the program count of native, 4× the deployed**. The GDN scan is latency-bound (a tiny sliver of the
~99 ms GEMV-dominated forward, `FR13_BV_NATIVE_MATCH_BIND:116`), NOT hidden behind weight DMA in the
way a GEMM tail is — it is a serial dependency on the critical path of each decode step. More, smaller
programs = more launch/scheduling overhead and worse SM occupancy per tile, paid on EVERY decode step
× 48 GDN layers. At B=1 there is no batch to amortize it. **So BV=4 is the worst perf of the listed
configs AND not bit-exact → reject on both axes (skeptic: do not adopt a slow non-bit-exact config).**

---

## §3 — Alternatives, ranked by (bit-exact × spill-free × scales-to-N_PAD16 × speed)

Bar reminder (user, `project_fr13_active_worker`/`feedback_math_correct_vs_bitexact`): the FINAL bar
is NOT literal abs-0.0 everywhere — it is **per-depth argmax + within-E5-self-noise-floor at the e2e
gate**. But for THIS scan seam the discriminating instrument is RAW max_abs==0.0 vs native (#10:
NEVER atol), because a per-node bf16-ULP is exactly what compounds to the 22-flip. So §3 ranks by
"reproduces native's compiled K-reduction tree (gate RAW 0.0)" first.

**(i) RE-CONFIRM num_warps=8 (the DEPLOYED config) is bit-exact to native — CHEAPEST, could dissolve
everything.** Status: PARTIALLY banked, INCOMPLETELY. The warp-gate `gdn_scan_warp_gate.json` already
shows **out_bit_exact=true / RAW 0.0 at N_PAD=1 AND 16** — but vs a *serial per-path* reference
(`native_update_serial_per_path`), NOT vs native's real `[32,128]`/4-warp fused kernel. So the
*output* is byte-identical to a correct serial recurrence, yet the seam binds claim the open question
is the *fused-geometry* match. **This is genuinely ambiguous and is the cheapest thing to settle:** if
the deployed BV=16/warps=8 SCAN output is ALSO RAW-0.0 vs the *actual native fused_sigmoid kernel*
(same boot, same captured inputs), then **the tension DISSOLVES — ship as-is, no spill, native-exact,
scales to N_PAD=16.** Design of the re-measure in §"num_warps=8 re-check" below. Spill-free ✔
(measured), scales ✔ (254 regs at N_PAD=16), speed ✔ (it's what ships). **Rank #1 to TEST.** Caveat:
the warp gate's serial reference passing while the seam binds say "open" suggests the serial reference
is NOT the same op-order as the fused kernel — so do not assume; the int-view A/B vs the *fused*
kernel is the decider.

**(ii) RECOMPUTE-from-spine at BV=32/warps=4 (`FR13_CACHE_SCALING_FUTURE` route 2; the spill-rank
WINNER).** Drop the `[N_PAD,BV,DIM_K]` h_cache; hold ONE `[BV,DIM_K]` working tile and replay each
node's ancestry from `b_h0` via the EXISTING `tl.where(strict_mask)` ancestry machinery (`:459-467`),
reusing the shared `_gdn_node_step`. Properties: **bit-exact to native BY CONSTRUCTION at BV=32/
warps=4** — it relocates only the SOURCE of a node's seed state, never touches the `[32,128]` tile,
the two `tl.sum(axis=1)` reductions, or the op-order → it compiles native's EXACT reduction tree.
**Spill-free**: one tile = ~64-90 regs/lane at BV=32/w4, O(1) in tree size → fits N_PAD=16 AND lifts
the N_PAD≤16 cap (supports >14-node / deeper suffix-fusion trees — the HARD CONSTRAINT). **Speed
≈ neutral-to-positive**: the extra ~2.57× rank-1 replay steps ≈ 5-33 µs vs a 99 ms forward, and it
AVOIDS the 636 B/thread (256 KB-class) spill round-trip on the 273 GB/s bus (`:116-118`). **THE
strategic route.** Caveat (red-team holds=False on the ranking wf = overstated-but-order-holds):
losslessness is a **GPU obligation, not a CPU fact** — the existence-proof replay kernel
(`_tree_gdn_replay_kernel:546`, no h_cache, spill-free) runs at **warps=8, not native 4**, and is
known BROKEN LIVE (gate-4 accept 2.02→1.58 wiring seam). So recompute must be (1) built at BV=32/w4
and (2) gated RAW-0.0 vs native AND live-correct (bug-class #8 offline≠live). Bit-exact ✔(by
construction, GPU-gated), spill-free ✔, scales ✔(>16 too), speed ✔. **Rank #1 to BUILD if (i) fails.**

**(iii) N_PAD streaming / node-tiling — INFERIOR, reject.** Process N_PAD nodes in SRAM-fitting
sub-tiles. Our tree is wide/shallow + MULTI-PARENT (node 0 parents 1,12,13; cat9 parents
`[-1,0,1,1,2,2,4,4,6,6]`), so any node-tiling either carries a large cross-tile live set (= the
all-resident problem re-expressed, no relief) OR spills cross-tile parent states to HBM
(reintroduces the +35.8% state-traffic tax FR13 fled, on the same saturated bus). No production tree
kernel (SpecInfer/Medusa/EAGLE/DEFT/STree) tiles over nodes — the multi-parent boundary handoff is
why. Op-order of each node's reduce CAN be kept native (the tile is still `[BV,128]`), so
bit-exact-CAPABLE, but the HBM tax kills speed. Bit-exact ~✔, spill-free ✗(HBM tax), scales ~✔,
speed ✗. **Reject** (strictly dominated by recompute).

**(iv) Two-pass (spill-friendly state pass + bit-exact reduction pass) — reject as ill-posed.** The
reduce tile IS the state tile (`FR13_BV_SPILL_VERDICT:6`): `state_i` is `[BV,128]`, written by the
rank-1 and consumed by the two `tl.sum(axis=1)`. You cannot separate "a spill-friendly state pass"
from "a bit-exact reduction pass" because the reduction operand is the live state — splitting them
either re-materializes the full `[N_PAD,BV,128]` (the same spill) or changes the op-order (the same
bit-exact loss). It collapses into either (ii) recompute or (iii) tiling. **Reject (not a distinct
option).**

**(v) Accept the spill at BV=32/warps=4 + QUANTIFY the TPS tax — viable BRIDGE, not the destination.**
C3 measured 636 B/thread spill at N_PAD=16; it LAUNCHES (no CUDA-701), it is native-bit-exact by
construction (BV=32/w4). The spill is to LPDDR5X 273 GB/s — the bus B=1 decode already saturates.
**TPS-tax estimate (LABELED ESTIMATE per bug-class #12, NOT a measured fact):** 636 B/thread ×
256 threads/CTA = ~163 KB spill traffic per scan-CTA per node; over N_PAD=16 nodes × ~48 GDN layers ×
(load+store) the spill round-trips are O(10s of MB) per decode step on a 273 GB/s bus = O(tens of µs)
added to a ~99 ms forward IF fully serialized on the bus — i.e. **plausibly <0.1% TPS** if the scan's
spill overlaps other layers' compute, but **must be MEASURED** (metrics OFF, B=1 and B=4, vs the
BV=16/w8 baseline) before any claim. The honest read: spill at BV=32/w4 may be CHEAP enough to ship
as the bit-exact bridge while recompute (ii) is built. Bit-exact ✔(construction), spill-free ✗(but
maybe-cheap), scales ✗(N_PAD=32 = 256 regs even ignoring spill, worse), speed ?(MEASURE). **Rank: use
as the lossless BRIDGE to confirm "geometry is THE seam" cheaply, then move to (ii).**

---

## §4 — RECOMMENDATION (single deployable path + dependency/fallback order)

**Primary deployable path: confirm-then-recompute.**

1. **GATE FIRST (1 boot, ships nothing): int-view A/B of the SCAN output, three arms vs the REAL
   native fused kernel, same boot, bit-identical captured layer inputs:**
   - arm A = deployed **BV=16/warps=8** (what ships),
   - arm B = native-geom **BV=32/warps=4** (spills but launches, C3),
   - ref  = native `fused_sigmoid_gating_delta_rule_update` captured the SAME boot.
   Compare **int-view equality** (`.view(torch.int32)` / `torch.equal`, NEVER atol — bug-class #10)
   of the served OUTPUT (`out_i`) at **N_PAD=1 AND N_PAD=16**.
   - **If arm A == ref (RAW 0.0):** the deployed BV=16/warps=8 IS native-bit-exact → **the tension
     DISSOLVES. Ship as-is. No spill (measured 254 regs), scales to N_PAD=16, fastest.** DONE.
   - **If arm A ≠ ref but arm B == ref:** launch geometry is THE seam; BV=32/warps=4 is the fix but
     spills → go to step 2.
   - **If neither == ref:** geometry is not the sole carrier → S1 floor; escalate (do NOT grind
     blindly), recompute won't rescue it alone.
2. **BUILD recompute-from-spine at BV=32/warps=4** (alternative (ii)). One boot to confirm: ptxas
   `n_spills==0` at N_PAD=16 AND int-view RAW 0.0 vs native at N_PAD=1 AND 16, on BOTH spine and a
   BRANCH winner (`[0,2]`,`[0,1,4]`; branch oracle = native-on-path-to-root, not native-MTP which has
   no branch counterpart). Then the live gate: B=1 same-seed byte-identical streams + accept/event
   unchanged + per-token argmax-vs-clean-teacher probe (`fr13_gold_margin_probe.py`) — bug-class #8
   (offline≠live; the existence-proof replay kernel is broken live).
3. **Fallback order:** (i) deployed-as-is [if step-1 arm A passes] → (v) ship BV=32/w4 WITH the
   measured spill as a bit-exact bridge [if its TPS tax is <~1-2%] → (ii) recompute [the durable
   destination, lifts the N_PAD≤16 cap for >14-node trees] → (iii) node-tiling ONLY if recompute hits
   a multi-parent wall (it should not). BV=4 and two-pass are OFF the list (§2, §3-iv).

**One-line answer to "does BV=4/warps=4 resolve the whole tension?": NO.** It is not native's BV, so it
does not clone native's reduction tree (bug-class #10), it has zero banked bit-exact evidence, and it
is the SLOWEST config (32 programs). The thing that resolves the tension is **recompute-from-spine at
BV=32/warps=4** (bit-exact by construction, spill-free, scales past N_PAD=16) — *after* the cheap
step-1 gate first checks whether the already-deployed, already-spill-free BV=16/warps=8 is secretly
native-exact (in which case nothing needs to change).

---

## §"num_warps=8 re-check": the empirical int-view re-measure design (alternative i)

GOAL: settle whether the DEPLOYED BV=16/num_warps=8 SCAN output is bit-exact to NATIVE's REAL
`fused_sigmoid_gating` fused kernel (not the serial per-path reference the existing warp gate used).
Design (extends `scripts/fr13_gdn_scan_warp_gate.py`, which already does N_PAD=1 & 16 with
`torch.equal`):
- **Reference arm = native fused kernel**, not `native_update_serial_per_path`. Capture native's
  `fused_sigmoid_gating_delta_rule_update` output on the SAME captured payload in the SAME boot (reuse
  the FR12 paired-capture harness `FR10_TREE_GDN_CAPTURE_PAYLOAD`). This is the load-bearing change —
  the existing 0.0 is vs the serial ref, which may not share the fused kernel's `[32,128]`/4-warp
  op-order.
- **Comparison = int-view, NEVER atol** (#10): `a.view(torch.int32).eq(b.view(torch.int32)).all()`
  AND report the RAW max_abs float + first-mismatch index (the warp gate already prints these). A
  single non-equal int = NOT bit-exact. Pin `negative_control_powered` (a deliberately-wrong arm must
  show >0) so a vacuous all-zero isn't read as a pass (bug-class #9).
- **Arms in ONE boot:** {BV16/w8 (deployed), BV32/w4 (C3, native-geom), BV8/w8, BV8/w4} all vs the
  native-fused ref, at N_PAD=1 AND N_PAD=16. (The launch currently hardcodes BV via the module global
  `:18` and num_warps inline `:1536`; the GPU worker must add a BV/num_warps override param to
  `launch_tree_gdn_prepared` for the sweep — a test-only parameterization, not a behavior change to
  the default path; flag-gated/diagnostic.)
- **Headers** (#9/#12): record flag-state, seed, BV, num_warps, N_PAD, n_regs, n_spills, and the
  native ref's own ttgir hash in every artifact; raw counters only.

If BV16/w8 == native-fused at int-view → tension dissolves, ship as-is. This is the single cheapest
experiment and MUST run before any kernel rewrite.

---

## SCAN vs REPLAY (confirm the spill is the SCAN h_cache, not the replay) — CONFIRMED

- The spill is the per-forward **SCAN** kernel `_tree_gdn_kernel` (`:387`): `h_cache =
  tl.zeros((N_PAD, BLOCK_V, DIM_K))` (`:458`) caches ALL N_PAD node states at once → the
  `N_PAD·BV·128·4` register footprint that hits 512 regs at BV=32/N_PAD=16. THIS is the SRAM/N_PAD
  tension; resolved per §3-§4 (recompute).
- The **REPLAY** kernel `_tree_gdn_replay_kernel` (`:546`) is SEQUENTIAL over the LINEAR accepted
  chain: it holds **ONE `(BLOCK_V, DIM_K)` register tile** (`:609`, source comment `:588` "No
  h_cache: one (BLOCK_V, DIM_K) register tile per program, so the replay is spill-free at any tree
  size"). It has **NO N_PAD cache** → the alignment-plan STEP-0 (replay → BV=4/32, durable-state
  regen) does **NOT** spill, at any BV up to 32 (one `[32,128]` tile = ~64-90 regs/lane). **Confirmed
  by source read.** So the spill problem is strictly the SCAN's all-node co-residency, exactly as the
  task states. (This is also why recompute works: it makes the SCAN behave like the replay — one tile,
  ancestry replayed — which is the existence proof that no-h_cache is feasible on this body.)

---

## Bug-class playbook rows in force (quoted)
- **#10 Shared-source ≠ shared-SASS (codegen identity):** "two kernels inline the same body but
  compile differently (constexpr/pressure)" → discriminator "byte A/B on captured payloads, int-view
  equality (NEVER atol), SASS hash pin" → fix "one shared body + identical constexprs/num_warps + the
  A/B gate re-armed per toolchain." **This is THE governing row** — BV=4 and BV=16/w8 are different
  COMPILATIONS than native's BV=32/w4 and cannot be assumed bit-exact; they must be int-view gated.
- **#12 Measurement traps:** "raw counters only; … label every estimate." The §3-v TPS-tax number is
  a LABELED ESTIMATE; the spill/reg numbers are MEASURED (ptxas wp5hsu63v); the warp-gate 0.0 is vs a
  SERIAL reference (labeled) not the fused kernel.
- (live) **#8 offline≠live** and **#9 vacuous instrument** govern the recompute build's GPU gate.

## Open obligations (GPU, none settled on CPU)
(1) int-view A/B of BV16/w8 SCAN output vs native FUSED kernel @ N_PAD=1 AND 16 (the dissolve test);
(2) recompute-from-spine built at BV=32/w4: ptxas n_spills==0 @ N_PAD=16 + int-view RAW 0.0 vs native
on spine AND a branch winner; (3) live B=1 same-seed byte-identical + accept/event unchanged + gold
margin per-token argmax probe. Reuse `fr13_gdn_scan_warp_gate.py` (extend ref to the fused kernel),
the FR12 paired-capture harness, and `fr13_gold_margin_probe.py`. Native ttgir reference:
`/tmp/lumo-l0c-fp8-cutlass-run30-triton/.../fused_sigmoid_gating_delta_rule_update_kernel.ttgir`
(`#blocked = sizePerThread=[1,4] threadsPerWarp=[1,32] warpsPerCTA=[4,1]`).
