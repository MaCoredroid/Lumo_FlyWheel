# FR13 — Scan bit-exact × SRAM × N_PAD tension: RESOLVED (analysis, verify HOLDS)

Date 2026-06-14. Workflow wg5ilg5ua (FR13_SCAN_BITEXACT_SRAM_NPAD.md, design 34916c2a). Answers the user's
concrete follow-up to FR13_BV_SPILL_VERDICT (which deferred this). ptxas-MEASURED on GB10 (sub-agent, triton3.6
cu130). Verify HOLDS=True with one material sharpening (the dissolve-test reference).

> ## ⚠ FACTUAL CORRECTION 2026-06-14 (scan-alignment-math w0n91rty5 34b92317, VERIFIED in live source)
> **Native decode geometry is `BV32 / num_warps=1 / num_stages=3`, NOT `BV32/w4`.** Confirmed at
> `/tmp/vllm_live_019/.../fla/ops/fused_recurrent.py:438` (`num_warps=1`, `num_stages=3`) — the packed-decode
> launcher. The `BV32/w4` in this bind (and the lineage) was WRONG. Consequences: (1) EXIT-2 recompute-from-spine
> must re-pin to **BV32/w1/s3** (not w4). (2) The spill arithmetic must be RE-DERIVED at w1: h_cache at
> N_PAD=16/BV=32/**w1** = 16·32·128·4 = 256 KB over 1 warp (32 lanes) = ~2048 fp32 regs/lane = CATASTROPHIC
> spill (far worse than w4's 636 B) — so the native-geom full-tree SCAN is unusable at N_PAD=16; only
> **recompute-from-spine** (one tile, no h_cache, no spill) is the deployable bit-exact route. The FREE geom
> pre-test (BV32/w1 at N_PAD=1, one node) has no h_cache so no spill. (3) BIGGER FINDING (FR13_SCAN_ALIGNMENT_MATH.md):
> native decode is RECURRENT rank-1 = the SAME algorithm as our scan (byte-for-byte same 5 ops) — so the
> carrier is CODEGEN (geometry + l2norm rsqrt-vs-1/sqrt + beta bf16-cast, all alignable), NOT a chunk-vs-recurrent
> algorithmic gap. The "diffuse within-floor IRREDUCIBLE" pessimism rested on the WRONG (chunked-prefill)
> reference and is REFUTED.

## BV=4 lead = DEAD (overturned — the monitor-flagged load-bearing claim)
BV=4/warps=4 does NOT preserve native's reduction: it is a DIFFERENT TENSOR SHAPE ([4,128] vs native [32,128])
⇒ different compilation/ptxas instruction-selection of the partial-sum + FMA feeding the K-reduce (bug-class
#10), and has ZERO banked bit-exact evidence (the banked 0.0 is at BV=16). Also worst perf: BV=4 = cdiv(128,4)
= 32 programs = 8× native's 4, on the latency-bound scan ×48 GDN layers at B=1. REJECTED on bit-exact AND speed.
(The no-spill arithmetic — 32KB/64 regs at N_PAD=16 — is true but irrelevant since deployed BV16/w8 is also
spill-free.) Lesson: a smaller V-tile is NOT a free bit-exact dial; cloning native's tree means compiling
native's SHAPE (BV=32). feedback_math_correct_vs_bitexact / bug-class #10.

## The tension (ptxas-measured)
regs/lane = N_PAD·BV·128/(warps·32) fp32. Key measured cells (FR13_BV_NATIVE_MATCH_BIND anchors):
- BV32/w4/N_PAD16 (native geom, the ONLY by-construction native-bit-exact) = 512 pred → **636 B HARD SPILL**
  (255-clamped, still launches). ⇒ native-bit-exact ⟺ BV32/w4 ⟺ SPILLS at the deepest deployed tree. THAT line
  IS the tension.
- BV16/w8/N_PAD16 (DEPLOYED) = 254 regs / **0 spill** / FITS / scales — but native-match UNKNOWN (BV16≠BV32).
- BV16/w4/N_PAD16 = 256 regs → SPILL (the OLD verdict's "spill" row — the 4-warp case, NOT deployed).
- SMEM "park" is N/A (no Triton primitive to put a non-tl.dot accumulator in SMEM).

## Resolution: CONFIRM-THEN-RECOMPUTE (verify HOLDS, reward-hack PASS, >4-node honored)
- **EXIT 1 (cheapest, test FIRST — could dissolve everything):** the deployed BV16/w8 scan's bit-exactness was
  only ever checked vs a PER-PATH SERIAL ref (`native_update_serial_per_path`, banked 0.0 in
  gdn_scan_warp_gate.json) — NOT vs native's REAL kernel. **Sharpened reference (verify correction): native's
  PACKED decode kernel `fused_recurrent_gated_delta_rule_packed_decode` — the kernel live tree-verify actually
  dispatches** (the doc conflated packed-vs-per-path with serial-vs-fused). Re-test int-view (NEVER atol) at
  N_PAD=1 AND 16 on spine + a branch winner. **If BV16/w8 == native-packed → tension DISSOLVES, ship deployed
  as-is** (254 regs, 0 spill, scales, fastest). bug-class #10 (bit-exact-to-serial-ref ≠ bit-exact-to-incumbent-SASS).
- **EXIT 2 (if Exit 1 fails): recompute-from-spine at native BV=32/w4** — drop h_cache, hold ONE [BV,128]
  register tile, replay ancestry via the existing tl.where(strict_mask) on the shared _gdn_node_step. Bit-exact
  BY CONSTRUCTION (compiles native's exact [32,128] tree + the two tl.sum(axis=1) in native's order, never
  touches the cached N_PAD tile), spill-free (~64-90 regs, O(1) in tree size — proven by the existence-proof
  replay kernel _tree_gdn_replay_kernel:588 "No h_cache: one tile, spill-free at any tree size"), and **LIFTS
  the N_PAD≤16 cap** for >14-node/suffix-fusion trees. Reward-hack PASS: builds OUR kernel, no splice/reroute,
  no tree-shrink. CAVEAT: losslessness is a GPU obligation (the existence-proof replay runs w8 not w4 + has a
  broken-live gate-4 history) — must GPU-gate.
- REJECTED: node-tiling (HBM +35.8% state-traffic tax), two-pass (ill-posed — reduce tile IS the state tile),
  accept-the-spill (only a measured bit-exact BRIDGE if TPS tax <~1-2%).

## scanVsReplay (confirmed by source)
The spill is STRICTLY the per-forward SCAN kernel `_tree_gdn_kernel` (h_cache=tl.zeros((N_PAD,BV,DIM_K)) :458
caches ALL N_PAD node states). The REPLAY kernel `_tree_gdn_replay_kernel` (:546) is SEQUENTIAL over the linear
accepted chain = ONE (BV,DIM_K) tile, spill-free at any tree size. So this tension is the scan deployment
problem only; replay / alignment-plan STEP 0 unaffected.

## Carrier-relevant angle
"scan is bit-exact, ruled out" (the decomposition) rests on the serial/per-path-ref 0.0, NOT vs native-packed.
EXIT 1's dissolve test is ALSO a re-validation of that ruling. If BV16/w8 ≠ native-packed at int-view, the scan
per-forward divergence becomes a NEW carrier candidate (and EXIT 2 becomes mandatory).

## Next GPU action (queued, secondary to the conv carrier front)
EXIT 1 = one boot: extend scripts/fr13_gdn_scan_warp_gate.py so the reference is native-PACKED captured
same-boot; int-view-equal the deployed BV16/w8 scan out_i (+ arms BV32/w4, BV8/w8, BV8/w4) at N_PAD=1 AND 16,
spine + branch winner; record int32-equality + raw max_abs + first-mismatch + n_regs/n_spills + a powered
negative control (bug-class #9). Conv carrier front (ww22n39bi) stays primary; this runs when the GPU sequencing
allows (or combined with a conv A/B boot on the same cat9 build).
