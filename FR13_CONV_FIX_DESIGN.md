# FR13 — L0-GDN conv1d_out M-dependence: trace + bit-exact M-invariant fix design

Date 2026-06-14. READ-ONLY design workflow (no kernel/patcher edits — a separate
GPU workflow is editing `scripts/fr10_phase4_patch_vllm_tree_gdn.py` concurrently).
Scope: trace the deep-spine row's `conv1d_out` end-to-end in the tree-verify
forward, pin the EXACT M-dependence, confirm/refute conv1d is the M-dependent
L0-GDN sub-op, draft the bit-exact fix.

HEAD `8b7684dd` (main). Pipeline-locked SWE-gold-gate build; FR13_TREE_CONV_FUSED,
FR13_REPLAY_ROUTE, FR13_CONV_COMMITTED_PATH, FR13_EAGER_PACK all baked ON.

---

## 0. TL;DR verdict (report-by-exception, two-part)

**The deep-spine `conv1d_out` IS M-dependent, but NOT because of a wrong bank-row /
wrong-column read.** The bank-row + column selection for the deep-spine row is
**correct by construction** (the live substate ladder confirms `h0_state_in=0.0`,
same bank row, at the same deep-accept event where `conv1d_out` first diverges).
The M-dependence that survives is the **bf16-tap finite-precision realization of the
conv multiply-accumulate** versus native's single fused `causal_conv1d_update`
kernel — a ~1 bf16-ULP per-layer seam (`conv1d_out` ≈ 9.77e-4 born from a byte-exact
`pre_conv`), the same realization-diff class that
[[reference_diffuse_gdn_accumulation_explained]] names as the diffuse carrier.

So on the live evidence:
- `isItReallyConv` = **TRUE for "first-nonzero L0 sub-op"** (pre_conv 0.0 →
  conv1d_out is the FIRST diverging sub-op at the deep-spine carrier event;
  `FR13_GATEA_DEEP_DIVERGENCE.md:65`), but
- the divergence is the bf16-tap/silu realization, **not** the prior-window
  bank/column wiring (which the 2026-06-09 `18.375` finding flagged and which is
  already FIXED on HEAD — that capture is STALE, `project_fr13_conv_priorwindow_root`).

There is a **real, bit-exact, policy-compliant fix** available for the surviving
seam (Section 5): route the deep-spine (path0) rows through native
`causal_conv1d_update` is BANNED (reward-hack reroute); instead replicate native's
EXACT conv op-sequence (fp32 accumulate order + `ex2.approx` silu + bf16 store) in
our manual conv on the spine rows so the spine `conv1d_out` == the M=1 decode
`conv1d_out` bit-for-bit. This was the user-chosen route
(`FR13_GATEA_DEEP_DIVERGENCE.md:99`) and is the one drafted here.

---

## 1. How the deep-spine row's conv1d_out is computed (mechanism, end-to-end)

The tree-conv emulation lives in the `conv_replacement` block of
`scripts/fr10_phase4_patch_vllm_tree_gdn.py` (the `if spec_sequence_masks is not
None:` branch, replacing native `causal_conv1d_update`). The path that runs on HEAD
is `_FR13_TREE_CONV_FUSED=True` (baked ON). For each batch sequence `b` and each
tree node `i` (flat row `b*tree_n + i`):

1. **Prior-window read (the only co-residency-sensitive input).**
   `_fr13_committed_prior_bank = gather_committed_path_conv_prior_prepared(conv_state,
   bank_rows)` (`:1641-1646`), where `bank_rows` = `prepare_committed_path_conv_rows(...)`
   (`:1596-1610`; library `gather_committed_path_conv_prior`,
   `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:281-334`). This selects, **per
   batch b**, the conv-state bank row =
   `spec_state_indices[b, accepted_paths[b, num_accepted-1]]` — the accepted leaf
   node's bank row (node-indexed, pre-remap). The prior window is then
   `prior_bank[b].index_select(1, prior_cols)` (`:2197-2199`) with
   `prior_cols = arange(width-1)` (cols [0..width-2]).

2. **Per-node window construction.**
   `source = cat(prior_window.T, x_b)` (FR13_TREE_CONV_FUSED appends a zero row for
   the write-back gather, `fused_tree_conv_source`, `:2209-2213`), then
   `window = source.index_select(0, source_flat).view(tree_n, width, dim)`
   (`:2219-2221`). `source_flat` (= `fr10_tree_conv_source_indices[width]`, built
   in the metadata builder) makes node `i`'s window the **last `width` taps of
   `(prior ++ committed-path tokens of node i)`** — pure data movement, M-invariant
   gather table (value-static per tree topology).

3. **Tap multiply-accumulate (the seam).**
   `acc = fused_tree_conv_taps_acc(window, conv_weights, bias)`
   (`src/lumo_flywheel_serving/fr13_tree_conv_fused.py:200-249`): per column,
   `tap_c = (window_c.to(bf16) * w_c.to(bf16)).to(bf16).to(f32)`; `acc =
   bias.to(f32) + tap_0; for c in 1..width-1: acc = acc + tap_c` (explicit ordered
   fp32 adds, bf16-rounded taps — the FR11-overturning "native bf16 taps" arm).
4. **Activation.** `out = triton_ex2_silu_bf16(acc)` (`:2510-2513`) →
   `_fr10_tree_conv_out[start:end] = out`.
5. **State write-back.** `new_state = fused_tree_conv_state_rows(...)` +
   `conv_state.index_copy_(...)` (`:2536-2541`, `:2702-2707`) — does not feed THIS
   forward's conv1d_out; it feeds the NEXT step's prior window.

The deep-spine row is node `i` on path0 (e.g. node 5 / node 7 in cat9, depth 4-5).
Its `conv1d_out` = silu(bias + Σ_c bf16tap(window[i,c], w_c)) where `window[i]` is
built from the per-b `prior_window` (step 1) plus the spine tokens of `x_b`
(step 2).

### What "M" means here (two distinct senses — both addressed)
- **Row-occupancy M** = number of co-resident tree rows in the forward (tree_n=10
  for cat9 vs 5 for chain5 vs 1 for decode). The conv emulation is a per-`b` Python
  loop over `tree_n` rows; there is **no cross-row reduction** in the tap-acc or the
  window gather, so the deep-spine row's arithmetic does NOT depend on how many
  OTHER rows are present (unlike the forked-FA2 query-tile, where kBlockM=64 MMA
  fragment occupancy made it M_DEPENDENT — `FR13_FA2_MDEPENDENT_BIND.md`, since
  OVERTURNED as a downstream correlate). So conv is **row-occupancy M-INVARIANT by
  construction**.
- **num_accepted M** = the prior-window column/bank selection driven by
  `num_accepted_tokens` (`_fr10_accepted_lens_tensor`, = committer `best_lcp`). This
  is the genuine M-dependence in the conv read (Section 2), and it is per-`b`, not
  per-row-occupancy.

---

## 2. The EXACT source of M-dependence (cite file:line)

### 2a. The prior-window column/bank selection — num_accepted-driven, CORRECT on HEAD
`prepare_committed_path_conv_rows` / `gather_committed_path_conv_prior`
(`fr13_tree_conv_fused.py:283-316`, `fr10_gdn_tree_kernel.py:281-334`):
```
lens      = num_accepted_tokens[:b]                 # = best_lcp per b
path_cols = clamp(lens - 1, 0, accepted_paths.size(-1)-1)
read_cols = accepted_paths[:b].gather(1, path_cols) # accepted leaf NODE column
bank_rows = spec_state_indices[:b].gather(1, read_cols)
```
This is M-dependent through `num_accepted_tokens` (`lens-1` picks the column). It is
the descendant of the 2026-06-09 `conv1d_out=18.375` root (wrong bank-row/cols at
`num_accepted>1`). On HEAD it is **FIXED** (`c0b53f5d`/`02b1627a`/`ef4d7514`): the
read is the node-indexed accepted-leaf, snapshotted PRE-remap, which is the
committed token path's window by construction — branch-winner-valid and spine-winner
byte-identical to the legacy post-remap linear read.

**Decisive live confirmation it is correct for the deep-spine carrier:** at the
deep-spine deep-accept flip event the L0 substate ladder shows `h0_state_in = 0.0`
(SSM recurrent state byte-exact, **same bank row**) while `conv1d_out` is the first
nonzero (`FR13_GATEA_DEEP_DIVERGENCE.md:65`; same bank-row/column convention across
all four touchpoints — forward-SSM-read, forward-conv-read `:1907-1916`,
replay-publish, replay-h0 — `FR13_NODE5_LADDER_DIFFUSE_BIND.md:44-46`). A wrong-row
read would have moved `h0_state_in` too. So the bank/column wiring is **not** the
surviving carrier.

### 2b. The surviving seam — bf16-tap MAC realization vs native fused kernel
`fused_tree_conv_taps_acc` (`fr13_tree_conv_fused.py:200-249`) and the
`_fr11_conv_tap_product` bf16-taps arm (`fr10_phase4...:2117-2129`):
the per-column tap is `(x.to(bf16) * w.to(bf16)).to(bf16).to(f32)`, accumulated in
fp32 with the legacy `(((bias + p0) + p1) + p2) + p3` order, then silu via
`triton_ex2_silu_bf16`. Native `causal_conv1d_update`
(`/tmp/vllm_live_019/.../ops/causal_conv1d.py:749-`, the `_causal_conv1d_update_kernel`
spec-decoding branch) loads the prior window at
`conv_state_token_offset = num_accepted_tokens[idx_seq] - 1` (L859-861, L873) and
does its OWN fp32 MAC + `ex2.approx` silu + bf16 store in a single fused Triton
program. The two are ℝ-equal but NOT bit-equal: ~1 bf16 ULP per layer
(`conv1d_out ≈ 9.77e-4` at the carrier; `FR13_GATEA_DEEP_DIVERGENCE.md:79`).

This realization seam is the same M-class as
[[reference_diffuse_gdn_accumulation_explained]]: born at L0 (2 bf16-ULP =
7.8125e-3 at the hidden level, `output/fr13_node7_ladder/ladder_summary.json` and
`output/fr13_node5_ladder/per_layer_maxabs.json`), monotone-smooth growth to L63,
amplified by gate 1/rms + the deep full-attn stack until the deep-spine argmax flips.

**M-dependence direction:** the seam is present at every co-resident depth but only
the DEEP-accept rows (num_accepted≥3) carry enough accumulated divergence to flip —
i.e. the visible M-dependence is `num_accepted` (deep-accept), realized through the
bf16-tap MAC, not row-occupancy. Refuting the FA2-tile carrier
(`FR13_FA2_CARRIER_OVERTURNED_BIND.md`): the first full-attn layer is L3, DOWNSTREAM
of the L0 GDN birth, so an FA2 fix cannot remove a divergence already 2-ULP at L0.

---

## 3. conv-state cache layout (banks / rows / cols) + committed-spine read

**Layout.** `conv_state` is the GDN conv cache `[num_banks, dim, state_len]`
(`conv_state.size(2) == state_len`, `width-1 == 3` for width-4; the live zero-pad
branch has `state_len=12 > width-1=3`). It is an `as_strided` VIEW over the SAME
mamba page as the ssm cache (kv[0]=conv, kv[1]=ssm) with `stride(0) ==
num_element_per_page` (`fr13_replay_conv_remap.py:7-18`,
`fr10_phase4...:2096-2099`) — which is why the page-safe torch remap exists (a
whole-page Triton copy would drag co-resident ssm bytes; `02b1627a`).

- **bank (dim 0)** = physical cache line, addressed by `spec_state_indices[b, col]`
  (one bank row per tree NODE column; `spec_state_indices` is kv-cache-group-local,
  hence the FR13_TREE_CONV_FUSED per-GROUP prep, `:1571-1585`).
- **rows (dim 1)** = `dim` feature channels.
- **cols (dim 2)** = `state_len` conv taps; cols [0..width-2] hold the rolling
  prior taps, cols [width-1..state_len) are zero-pad.

**How a committed-spine token reads its prior window.** Pre-remap snapshot:
`bank_row = spec_state_indices[b, accepted_paths[b, num_accepted-1]]` (the accepted
LEAF NODE's bank), `cols = [0..width-2]` (`prior_cols`). The per-node window gather
then composes `(prior ++ committed-path tokens)` last-`width` taps. For a pure spine
winner this is byte-identical to native's rolling-buffer read at column
`num_accepted-1` (native: `[history2..historyM, draft1..draftN]`, prior_tokens at
offset `num_accepted-1`; `causal_conv1d.py:846-873`). The convention matches; only
the MAC realization differs.

---

## 4. num_accepted_tokens role in the prior-window selection

`num_accepted_tokens` enters TWICE and is the `best_lcp` committed length:
1. **Conv prior read** (Section 2a): `read_col = accepted_paths[b, num_accepted-1]`,
   `bank_row = spec_state_indices[b, read_col]` — selects WHICH node's window is the
   prior. M-dependent, correct on HEAD.
2. **Native kernel reference** (M=1 decode / native MTP-5):
   `conv_state_token_offset = num_accepted_tokens[idx_seq] - 1`
   (`causal_conv1d.py:859-861`) — selects the prior-window column in the rolling
   buffer. Same `num_accepted-1` convention.

The original `project_fr13_conv_priorwindow_root` bug (conv1d_out=18.375) was a
`num_accepted>1` WRONG-ROW read (bank 6 cols[0,1,2] vs native bank 1 cols[5,6,7]
rolled tail). That is the 2026-06-09 finding and it is **FIXED + STALE on HEAD** —
do NOT re-chase it. The current deep-accept M-signature is the bf16-tap realization
(Section 2b), confirmed by `h0_state_in=0.0` co-located with `conv1d_out≠0`.

---

## 5. priorWindowGather / replay remap M-invariance

- `gather_committed_path_conv_prior_prepared` (`fr13_tree_conv_fused.py:319-327`):
  per-layer `index_select(conv_state, 0, bank_rows)` — pure gather, **M-invariant
  given bank_rows**; bank_rows itself is num_accepted-dependent (Section 2a) but
  correct.
- `replay_conv_state_linear_remap_prepared` (`fr13_tree_conv_fused.py:369-386`) /
  `replay_conv_state_linear_remap` (`fr13_replay_conv_remap.py:41-121`):
  gather-then-scatter (materialize all source rows via `index_select` BEFORE any
  `index_copy_` write) — race-free in-place permutation (playbook class 3), conv-view
  logical elements only (page-safe), M-invariant permutation. The prior-window
  SNAPSHOT is taken BEFORE this remap (`:1527-1646`), so the deep-spine prior read is
  unaffected by the remap regardless of M.

So the gather + remap machinery is M-invariant; the only M-coupled input to
conv1d_out is the (correct) num_accepted-driven bank/column selection, and the only
M-coupled OUTPUT divergence is the bf16-tap MAC realization.

---

## 6. Is it really conv (vs gate / o_proj / in_proj)?

**Confirmed: conv1d_out is the FIRST-NONZERO L0-GDN sub-op at the deep-spine carrier
event** (`FR13_GATEA_DEEP_DIVERGENCE.md:65`, dual-verified substate ladder):
```
input_hidden 0.0 → pre_conv 0.0 → conv1d_out 9.77e-4 (FIRST NONZERO)
  → h0_state_in 0.0 (same bank row) → gdn_scan_out 1e-6 → gate_out 4.88e-4
  → o_proj_out 1.95e-3
```
- **gate / o_proj** are DOWNSTREAM of conv (and of the scan) — they amplify, they do
  not originate (o_proj_out is a symptom: fix its input → it cascades to 0.0;
  [[feedback_fr12_subkernel_zero_gate]]).
- **in_proj** is UPSTREAM and produces `pre_conv`, which is 0.0 (byte-exact) — so
  in_proj is NOT the source.
- **GDN scan** is M-invariant as an OP (BV A/B D16=D32=0.0, reduction over DIM_K is
  geometry-invariant; `FR13_FA2_CARRIER_OVERTURNED_BIND.md`,
  `reference_diffuse_gdn_accumulation_explained`). Its `gdn_scan_out 1e-6` here is
  downstream of the conv 9.77e-4 (the scan consumes conv output as v/k/q), not an
  independent source.

**Caveat (the one place "scan-side" is the better frame):** the dual-verified
`FR13_NODE5_LADDER_DIFFUSE_BIND` frames the carrier as the GDN recurrent
**state-feed** (live rank-1 tree-scan over the accepted chain vs the clean
chunked-prefill realization) rather than conv. That is consistent with this doc:
the conv1d_out 9.77e-4 is the FIRST nonzero sub-op (so conv is the entry point), but
the L0→L63 GROWTH is dominated by the recurrent state-feed realization compounding
through the scan + gate + deep full-attn. So: **conv is the localizable first
divergence to align; the residual after a perfect conv alignment is the diffuse
state-feed accumulation** (the within-floor regime). Both fixes are needed for full
parity; conv is the one with a concrete bit-exact patch (this doc). If a perfect
conv alignment does NOT collapse the e2e flips, the remaining carrier is the
state-feed — route to tree-reshape per [[project_fr13_tree_reshape_unifying_lever]],
NOT a per-op patch.

---

## 7. The bit-exact M-invariant fix (concrete, file:line)

**Goal:** make the deep-spine (path0) row's `conv1d_out` == the M=1 native decode
`conv1d_out` bit-for-bit, by replicating native's EXACT conv op-sequence on our
manual conv. NO reroute through `causal_conv1d_update` (banned reward-hack,
FR-12 / `feedback_no_reroute_reward_hacking`); our kernel still computes.

### 7a. The change (in `src/lumo_flywheel_serving/fr13_tree_conv_fused.py`)
Replace the silu in `fused_tree_conv_taps_acc`'s consumer chain — specifically the
activation applied to `acc` — with a bit-exact `ex2.approx` replica of native's
silu, AND verify the tap MAC order matches native's. Native's relevant ops
(`/tmp/vllm_live_019/vllm/model_executor/layers/mamba/ops/causal_conv1d.py`,
`_causal_conv1d_update_kernel` L749+, width-3/4 branches L874-): per-column
`col_k = tl.load(prior + k*stride)`, accumulate in fp32, then
`out = x / (1 + exp(-x))` realized via the hardware `ex2.approx` (the silu seam the
user chose to grind, `FR13_GATEA_DEEP_DIVERGENCE.md:99`).

Concrete edits:
1. `src/lumo_flywheel_serving/fr13_ex2_silu.py` (`triton_ex2_silu_bf16`, consumed at
   `fr10_phase4...:2511`): make the silu an EXACT `ex2.approx` replica of native's
   (exponent/fraction split + the same polynomial + bf16 store rounding), so
   `silu(acc)` matches native's silu bit-for-bit on identical `acc`. (Already the
   intended replica; the gate is conv1d_out → 0.0 on clean-input layers L0,L4,L8,…)
2. `fused_tree_conv_taps_acc` (`fr13_tree_conv_fused.py:234-248`): confirm the bf16
   tap product + fp32 accumulate order (`bias + p0`, then `+p_c` ascending) matches
   native's column accumulation order. If native accumulates in width-descending
   order (col0 = oldest), reverse the `start..width` loop accordingly — this is the
   load-bearing op-order match (`acc = acc + prod_f32[:, col, :]`). Today the order
   is ascending col 0..width-1; native loads col0..col(width-1) as
   oldest..newest with `tl.fma`/add in that order — VERIFY by reading the native
   width-4 branch and align the loop direction exactly.

### 7b. Scope: spine rows only is unnecessary — the fix is uniform
Because the tap MAC + silu are per-element (no cross-row reduction), making them
bit-exact to native makes ALL rows (spine + branches) bit-exact wherever the input
window is the committed-path window. So the fix is uniform (replace the silu
replica + confirm MAC order), not a path0-special-case. This avoids the
`project_fr13_conv_priorwindow_root` CAUTION (the naive layer≥4 band-aid made
gateA WORSE — fix the realization, not a column band-aid).

### 7c. Gate (offline-first, then ONE live)
Per `FR13_GATEA_DEEP_DIVERGENCE.md:164`: ONE spine-only capture of conv sub-op
inputs (`pre_conv` + `conv_state` + `conv_weights` + `bias`) for a SPREAD of
clean-input layers (L0,L4,L8,L12,L24,L36,L44 — all `pre_conv==0.0`); iterate the
silu/MAC replica OFFLINE (boot-free) vs native `causal_conv1d_update` on the
IDENTICAL captured inputs until `conv1d_out == 0.0` (RAW, not atol) for EVERY clean
layer. Then ONE live full-ladder (all spine rows + branches + logits + the e2e
per-token argmax-vs-clean flip count: does cat9 22 → ~native 3?). The decisive gate
is the e2e flip count, NOT per-layer 0.0 (per
[[reference_scalar_metric_per_token_blindspot]],
[[project_fr13_active_worker_codex_fr15]]).

---

## 8. Lossless / speed risks + default-OFF safety

- **Lossless:** the fix makes the spine conv1d_out CLOSER to native (the lossless
  reference is native MTP-5 / E5), so it cannot make losslessness worse for the
  spine. Risk: a WRONG silu replica regresses L0 (the fp32-taps variant did:
  conv1d_out 0.0625 + input_hidden 0.3125 globally, `FR13_GATEA_DEEP_DIVERGENCE.md:80`)
  — mitigated by the offline multi-layer clean-input gate (catches L0 regression
  before any boot). The bf16-tap arm must stay (FR11-overturning fix); do NOT promote
  taps to fp32.
- **Speed:** silu replica is the SAME `triton_ex2_silu_bf16` launch object (shared
  lines, `:2491`) — no extra device nodes; the MAC order change is free
  (reorder a Python loop over `width≤4`). GB10 decode is weight-bandwidth-bound
  (~27 GB/forward); the conv MAC is <1% of the forward, so a bit-exact realization
  costs nothing on the bandwidth floor.
- **Default-OFF-safe / replay-compatible:** gate the silu-replica behind a flag
  (e.g. `FR13_CONV_EX2_REPLICA`, default OFF) until the offline+live gate passes;
  the OFF arm executes the current `triton_ex2_silu_bf16` verbatim (the A/B
  instrument). FR13_TREE_CONV_FUSED / FR13_REPLAY_ROUTE / FR13_CONV_COMMITTED_PATH
  are untouched (the fix is inside the tap-acc/silu, downstream of the
  remap/snapshot), so replay compatibility is preserved by construction. The byte
  A/B (`tests/test_fr13_tree_conv_fused_byte_ab.py`) must be re-armed for the new
  replica (class 10: shared-source ≠ shared-SASS — pin SASS hash / int-view 0.0).

---

## 9. Relevant FR13_BUG_CLASS_PLAYBOOK rows (quoted)

- **Class 4 — Spine-only-valid column arithmetic on branch winners** (`c0b53f5d`
  conv fix): "EPISODIC whole-forward corruption (10-25× baseline) AFTER branch
  commits; transient, recovers … derive from the committed PATH's tokens; snapshot
  BEFORE in-place mutations." → THIS is the class the conv prior-window read belongs
  to; on HEAD the committed-path snapshot-before-remap fix is in place (Section 2a),
  which is why the bank/column read is correct and the surviving seam is the MAC.
- **Class 10 — Shared-source ≠ shared-SASS (codegen identity)** (matrix R4): "two
  kernels inline the same body but compile differently … byte A/B on captured
  payloads, int-view equality (NEVER atol), SASS hash pin." → the silu-replica gate
  MUST be RAW==0.0 / int-view, the atol=1e-3 scan-replay gate does NOT certify
  (`FR13_BV_SPILL_VERDICT.md:22`).
- **Class 11 — Batch-composition / BI-flag sensitivity:** "native itself only 0.714
  draft-identical across BI flag; near-ties flip on sub-ULP shifts … pin BI on BOTH
  arms." → relevant because cat9+BI=34 (WORSE) shows BI is NOT the fix; the conv MAC
  realization is the real seam, not a BI-coverable GEMM.
- **Class 12 — Measurement traps:** "non-like-for-like trajectories after fixes …
  single-draw floors." → the e2e flip gate (Section 7c) forks the stream
  (class-12 confounded); read the per-token argmax-vs-clean probe with a multi-sample
  native floor, not a single cross-boot count.

---

## 10. One-line answer to the task question

The deep-spine `conv1d_out` is **conv-localized and num_accepted-(deep-accept)-
M-dependent**, but the live carrier is the **bf16-tap + silu MAC realization** (1
bf16-ULP/layer, born at L0 from byte-exact pre_conv), NOT a wrong bank-row/column
read (that 18.375 root is FIXED + STALE on HEAD, and `h0_state_in=0.0` proves the
row is correct). The bit-exact M-invariant fix = replicate native
`causal_conv1d_update`'s exact MAC order + `ex2.approx` silu in our manual conv
(`fr13_tree_conv_fused.py:200-249` + `fr13_ex2_silu.py`), flag-gated default-OFF,
gated by an offline multi-clean-layer conv1d_out==0.0 (RAW) then ONE live e2e flip
count. The residual after a perfect conv alignment is the diffuse recurrent
state-feed accumulation → tree-reshape, not a per-op patch.

Pairs with [[project_fr13_conv_priorwindow_root]],
[[reference_diffuse_gdn_accumulation_explained]],
[[reference_gdn_verify_sequential_dispatch]],
[[project_fr13_tree_reshape_unifying_lever]],
[[feedback_no_reroute_reward_hacking]], [[feedback_fr12_subkernel_zero_gate]],
[[reference_scalar_metric_per_token_blindspot]].
