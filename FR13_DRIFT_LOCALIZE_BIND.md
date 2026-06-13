# FR13 Drift-Localize — where the tree spine diverges from native E5, in OUR kernels

Workflow `wf_7ce1baf2-286` (CPU source-read, 5 agents, no GPU). Raw:
`research/fr13_workflows/drift_localize_wf_7ce1baf2.raw.json`. Adversarial verify
`holds=FALSE` → acted only on the independently-reverified parts. HEAD 813cb9fd.

## Question (user 2026-06-13)
"We control all the kernel — chase down and FIX the drift but NEVER make it slower.
Where does the drift happen and how to MATHEMATICALLY clean it?" Native E5 (FLASH, 5-chain)
runs the SAME weights + SAME fp8 and flips the greedy argmax 7x LESS than the tree (3 vs 22
clear-margin flips vs the same non-MTP oracle) at the deepest spine node (node 7, depth 4-5,
~1.9 nats; commit 0b5de164) → a clean fast version provably exists → make the tree SPINE
bit-exact to native's CHAIN, not slower.

## DURABLE RESULTS (live evidence — config / ttgir / source on current HEAD)

### 1. fp8 GEMM batch-shape scaling — RULED OUT (the user's + my PRIME suspect, killed)
The fp8 activation quant is **per-token-group dynamic** (group_shape (1,128)), one Triton
program per (row,group), amax over ONLY that row's 128 cols, **no cross-row reduction**
(`fp8_utils.py:635-664`, launch grid `(M,)` M=numel/group). The GEMM
(`cutlass_scaled_mm` `_custom_ops.py:845`) has M as a pure parallel axis, K-only reduction —
a spine row gets **bit-identical fp8 bytes + output** whether it rides native's 6-row chain
or the tree's 9-10-row batch. Every M-dependent knob is **dead on GB10**: the autotune
config table has no GB10/family-120 entry → default `BLOCK_SIZE_K=128` fixed; DeepGEMM
(TMA-pads M) needs cap-90/family-100 → off; `VLLM_BATCH_INVARIANT` doesn't change the
block-fp8 math. Adding branch rows CANNOT change a spine row's scale. **No fp8 fix is
warranted** (would be a no-op and could only slow things).

### 2. conv-tap bf16 rounding — RULED OUT (and the proposed "fix" would BREAK bit-exactness)
One reader's headline ("tree rounds each conv tap to bf16 before fp32 accum; native keeps
fp32") is **WRONG vs live ttgir**: native `_causal_conv1d_update_kernel.ttgir` does
`arith.mulf : bf16` (tap product ROUNDED to bf16) THEN `extf bf16->f32` THEN `addf f32` —
**zero fp32-tap mulf variants** (grep count 0). The fused tree path
(`fr13_tree_conv_fused.py:234-235`) already produces exactly that bf16-rounded product →
**already bit-exact**. **DO NOT** promote the tap product to fp32 — it would diverge FROM
native. (FIX-3 was gated bit-exact-by-construction; this confirms it.)

### 3. GDN scan launch-geometry codegen seam — REAL, un-closed, likely marginal
Tree scan runs `num_warps=8` / module-const `BV=16` (`fr10_gdn_tree_kernel.py:18,1536`);
native `fused_sigmoid_gating` runs `num_warps=4` / `BV=32` (`:223,227`). Same fp32 op
sequence (`_gdn_node_step` `:364-383` is op-for-op identical to native `:144-167`), but a
different warp/lane map → different ptxas FMA scheduling. **Only ever gated at atol=1e-3,
never raw==0.0 vs native** at the deployed N_PAD. Magnitude ~1-bf16-ULP, no depth-growth
signature → unlikely to BE the 7x carrier alone, but it's the one our-kernel seam never
driven to literal 0.0. Speed-preserving fix candidate (if the ladder flags it):
`BV=8` per FR13_BV_SPILL_VERDICT option-2 (register-resident at N_PAD=16, kills the h_cache
spill, leading extent 8 ≥ 4 warps preserves native's K-reduction layout, lets num_warps drop
toward native's 4). Gate on RAW `root_out_vs_native_max_abs==0.0`, NOT the atol=1e-3 replay.

## REFUTED on STALE evidence (do NOT act; needs a fresh current-HEAD ladder)
The adversarial verify re-asserted the **conv prior-window READ** as the carrier
(tree bank row 6 cols [0,1,2] vs native row 1 cols [5,6,7] rolled-tail at num_accepted>1,
conv1d_out=18.375) and REFUTED synthesis #1 (bf16 in_proj M-bucket GEMM swap) using
`pre_conv=0.0`. **BOTH rest on `output/fr13_gdn_substate_prompt0_20260609T061732Z` (dated
2026-06-09).** That capture PRE-DATES the conv prior-window fixes: `c0b53f5d` (06-10
committed-path conv window, branch-valid), `02b1627a` (06-11 page-safe conv remap,
boundary-trace root cause), and FIX-3 `ef4d7514` (06-12). Current `:1523-1538`
(`FR13_CONV_COMMITTED_PATH` default-ON) reads the window from the accepted path's leaf-node
column, "spine winners byte-identical to legacy post-remap read." So the 18.375 is **stale**
and that specific conv bug is fixed — yet the gold gate (06-13, current HEAD) STILL shows
22-vs-3 flips. **⇒ the current-HEAD first-diverging sub-op is UNKNOWN and must be re-measured.**

## DECISIVE NEXT (GPU, queued behind the cat10 gate — GPU serialized, 1 container at a time)
A FRESH per-sub-op ladder on HEAD 813cb9fd at node 7 (deepest flipping spine row), tree-spine
vs native-chain on ONE paired run (pin the prompt — reference_capture_once_native_pin_prompt):
capture pre_conv → conv1d_out → scan_out → gate_out → o_proj_out on BOTH arms; **first nonzero
sub-op = the current carrier.** Candidates after the rule-outs: a RESIDUAL conv-window seam
(fix incomplete), the GDN-scan num_warps/BV codegen seam (#3, confirm with RAW==0.0 not
atol=1e-3), or — only if pre_conv is nonzero on current code — the bf16 in_proj/MLP GEMM
M-bucket reorder (synthesis #1, currently only refuted by stale pre_conv=0.0). Then fix the
first nonzero sub-op (wiring → fix wiring; codegen → align warp/BV; all speed-preserving,
fp8 untouched), confirm via FR13_HIDDEN_SUBSTITUTE (splice native@op → flip reverts), re-gate
with the per-token argmax probe.
