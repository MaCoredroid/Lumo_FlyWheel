# FR13_NODE7_LADDER_BIND — per-layer ladder at the deep verify row for the p2/p3 argmax flips (DRAFT, not committed)

Date 2026-06-13 UTC. Tree boot `fr13-forked-fa2-tree` cat9 / TREE_ATTN / num_spec=9,
**ENFORCE_EAGER=1**, all FIX default ON (C0), FORKED FA2 .so. Probes =
`output/fr13_acceptance_ladder/prompts_swe4.json` (4 pinned), greedy seed 1313 top_p 1.0.
Captures armed: `FR10_LAYER_HIDDEN_CAPTURE` + `_ROWS=0..9,849,1686` (per-layer hidden at
the 10 tree-forward rows + the clean prefill last rows) gated `NUM_TOKENS=10,850,1687`;
`FR13_FINAL_LOGIT_CAPTURE` same rows; `FR13_COMMIT_ARGMAX_GATE=1`; `FR13_TREE_ATTN_OP_CAPTURE`
/ `FR13_FLASH_ATTN_OP_CAPTURE` on `language_model.model.layers.3.self_attn`. Reduce =
`output/fr13_node7_ladder/reduce.py` (per-layer node-row hidden vs clean last-prefill row).
Artifacts `output/fr13_node7_ladder/{ladder_result.json,ladder_summary.json,cap/}`.

## WIRING CORRECTION (engagement, class 9) — re-grepped against HEAD
- The tree VERIFY forward has **10 rows, not 9**: root = forward-row 0, the 9 tree nodes =
  forward-rows 1..9 (confirmed by `tree_attn_bias` shape (10,10) + ancestry structure).
  `FR10_LAYER_HIDDEN_CAPTURE_NUM_TOKENS=9` captured NOTHING; the correct gate is `=10`.
- The committer's `committed_row` is in NODE space (0..8) and indexes the post-remap
  `target_logits`/`tree_self_logits` (`logits[metadata.tree_self_logits_indices]`), which is
  NOT a trivial forward-row+1 map. **node-N's self/target logit lives at a DIFFERENT physical
  forward row** — localized EMPIRICALLY by argmax: the served-flip row is the forward row
  whose lm-head argmax == the served (flipped) token with the gate's exact top-2 margin.
  For this boot the flip landed on **forward-row 4** for BOTH p2 and p3.
- The clean reference's `compute_logits` is NOT captured at num_tokens=ctx_len (spec-decode
  prefill computes logits only on the sampled row), but the per-layer HIDDEN **is** captured
  at num_tokens=850/1687 (the `Qwen3NextModel.forward` hook), which is all the ladder needs;
  the clean argmax (3425 / 71093) comes from the served response (validated == native E5 / ch2).

## WITHIN-BOOT DETERMINISM (class 8): PASS
- det_gate `within_boot_det_rep1_eq_rep2 = [True,True,True,True]` on all 4 probes.
- p2 within-boot same-seed repeat rep1==rep2 byte-identical, pos21=1970 both reps.
- This boot reproduces the two banked flips at the SAME served positions (p2 pos21 = 1970
  ` code`; p3 pos73 = 9764 `Let`) and only diverges from the banked 4-prompt stream AFTER
  the flip (p2 idx 25, p3 idx 79 — single-prompt vs batched co-residency, class 11).

## THE p2 LADDER — decisive, input-aligned (spine node, forward-row 4, margin 0.5)
p2 verify forward (call 6, row 4): argmax **1970 (` code`) @24.5**, runnerup **3425 (` files`)
@24.0**, **margin 0.5** — exactly the gate/ch2-banked flip. Clean teacher-force (prompt +
served[:21]) argmax = **3425 (` files`)**.
**`input_hidden` max_abs = 0.0 (EXACT)** — verify-row-4 input embedding == clean-row-849
input embedding (both = served[20]=9468); the comparison is apples-to-apples.

| layer | type | max_abs | cos | note |
|---|---|---:|---:|---|
| input | embed | **0.0** | 1.0000 | exact (same input token) |
| **L00** | **GDN** | **0.0078** | 0.99996 | **FIRST nonzero** |
| L02 | GDN | 0.0312 | 0.99992 | |
| L03 | full_attn | 0.0041 | 0.99878 | full-attn does NOT introduce it (drift drops) |
| … (slow accumulation, cos > 0.995 throughout L0–L58) … | | | | |
| L58 | GDN | 0.5625 | 0.99581 | residual already ~2.0 |
| **L59** | **full_attn** | 0.7734 | **0.9711** | first SHARP directional drop (cos 0.996→0.971) |
| L60 | GDN | 0.875 | 0.98669 | |
| L62 | GDN | 1.6875 | 0.96499 | |
| **L63** | **full_attn** | **30.0** | 0.9542 | **catastrophic amplification (Δ +28.3)** |
| final_norm | RMSNorm | 2.5 | 0.98714 | argmax flips 3425→1970 |

## FIRST-DIVERGENT-LAYER + ROOT OP
- **FIRST nonzero divergence: LAYER 0 = linear_attention (GDN).** max_abs 0.0078, the input
  is byte-exact (0.0), so the very first GDN block already differs from the clean forward.
- It is **DIFFUSE, not a single seam**: every layer adds a little (cos stays > 0.995 for 59
  layers), i.e. the tree verify forward is ℝ-correct-but-not-bit-exact to the clean forward at
  essentially every op — the classic fp-nonassociativity / reduction-order / cast-boundary
  drift ([[feedback_math_correct_vs_bitexact]]). The L0 GDN nonzero is the leading edge.
- The **argmax-FLIPPING amplification is the DEEP FULL-ATTENTION layers**: cos is flat
  (>0.995) until **L59 full_attention** (cos 0.996→0.971) and the explosion is at **L63
  full_attention** (max_abs 1.69→30.0, the last full-attn layer). The deepest tree row
  (node-row-4, the most-accumulated) routed through TREE_ATTN (ancestry-mask + online-softmax
  over the 839+-token KV) vs the clean prefill (FA2-fork native, FLASH-path) is where the
  accumulated drift turns the 0.5-nat margin. This matches the named suspect (a) TREE_ATTN
  deep-row attention and the FR11 "full-attn not tree-exact in fp8" flag.
- **rootOp = TREE_ATTN deep-row full-attention (ancestry-mask online-softmax) AS AMPLIFIER, on
  top of a diffuse GDN-from-L0 accumulation.** It is NOT a single localizable wiring seam
  (conv prior-window / fp8 bucket / one GEMM): the divergence is present from L0 and grows
  monotonically; no layer "first introduces" a clean→broken step.

## PROPAGATION CONFIRMATION
- Magnitude argument (no substitution re-forward available in-instrument): the per-layer cos
  is ≥0.995 through L58 (final-norm-direction essentially preserved → a 0.5-nat argmax would
  NOT flip from L0-L58 drift alone), and only crosses argmax-threatening territory at
  **L59/L63 full_attention** (cos 0.971 / max_abs 30, final_norm cos 0.987). The flip is
  carried by the deep full-attn amplification of the accumulated drift; zeroing the L0 GDN
  seed (0.0078) would not by itself stop a diffuse re-accumulation, but the dominant
  contributor to the final 2.5/0.987 final-norm divergence is the L59→L63 full-attn block.
- The verify logit flip is reproduced exactly (1970 @24.5 vs 3425 @24.0, margin 0.5) and the
  clean argmax (3425) is the validated reference (== ch2 == native E5).

## p3 (reject_correction; SECONDARY — reduce auto-picked the input-aligned instance)
p3 has TWO 9764-emitting verify forwards in-stream (margins 0.125 and 7.25). The reject-
correction served row's input embedding does NOT match the clean last-input on the 7.25 call
(input max_abs 0.328 — the parent-edge correction row carries a different input token than
served[72]), so that instance is not a clean apples-to-apples ladder. The input-aligned p3
instance (call 13, input max_abs 0.0) shows the SAME shape but **much faster** divergence
(cos collapses to ~0.45 by L31, max_abs explodes through the deep full-attn to 21–24 by
L58/L63) — same class (diffuse-from-L0 + deep-full-attn amplification), worse because the
reject-correction row is even more accumulated. p3 is consistent with p2; p2 is the decisive
clean case.

## RECOMMENDED FIX (the named, bit-exact approach + expected cost)
Because the gap is (i) diffuse-from-L0 and (ii) dominated by TREE_ATTN-vs-FLASH deep full-
attention, the within-floor max_abs gate already drove the per-layer drift to ~0.00195 but a
clear-margin argmax flip survives at the deepest row. Two routes, in priority order:
1. **Route the verify full-attention through the FA2-fork additive `-inf` tree-bias path**
   (the [[project_fr13_fa2_fork_nocopy_floor]] fork) for ALL 16 full-attn layers, so the tree
   verify full-attn is byte-identical to the native FLASH path on the ancestry mask (the fork
   gave 14/16 calls whole-tree 0.0, 2 single-ULP). This removes the L59/L63 amplifier — the
   dominant argmax-flip carrier — and is already the accepted no-copy floor. Expected speed
   cost: ~0 (additive-bias fork is the same FA2 kernel; it is the deploy path).
2. **Batch-invariant / non-fp8 the deep full-attn GEMMs + GDN scan reductions** to kill the
   diffuse L0-onward drift seed (make the per-op reductions N-independent, #42960 class).
   This is the within-floor-to-bit-exact grind; expensive (BI GEMMs are slow — OFF for speed
   per [[reference_fr10_speed_measurement_pitfalls]]) and only needed if route 1's residual
   single-ULP still flips a margin. Recommend route 1 FIRST, re-ladder, and only BI the
   residual if a clear-margin flip survives.

Do NOT splice/copy/reroute through native as a parity hack (banned, [[feedback_no_reroute_reward_hacking]]):
the deliverable is OUR tree verify full-attn made bit-exact via the FA2-fork additive-bias path.

## Artifacts (`output/fr13_node7_ladder/`)
- `ladder_result.json` / `ladder_summary.json` — the per-layer ladders (p2 decisive, p3 secondary).
- `cap/node7_hidden.call*.pt`, `cap/node7_logits.call*.pt` — per-layer hidden + final logits at
  the 10 tree rows (+ clean prefill rows); `cap/tree_attn_op.call*.pt` (layer-3 full-attn).
- `cap/fr13_commit_argmax_gate.jsonl` — in-process committer-row gate (flip serve-events).
- `run_ladder.py` (driver), `reduce.py` (ladder reducer), `boot.sh` (capture-armed boot).
