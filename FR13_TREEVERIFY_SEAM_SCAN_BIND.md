# FR13 tree-verify seam scan — handoff+conv CLOSED, full-attn REFUTED, carrier = serving-path bug + diffuse-drift

Workflow `wf_448b7735-45d` (wj3u6ag0v, 5 agents, CPU code scan vs native source). Raw:
`research/fr13_workflows/treeverify_seam_scan_wj3u6ag0v.raw.json`. Red-team **holds=FALSE**
(it refuted the catalog's full-attn top-suspect on a factual error). 2026-06-14. Generalizes the
bf16/fp32 scan to the tree-verify pipeline now that per-forward GDN is proven bit-exact
(FR13_BV_GEOMETRY_NOT_THE_SEAM_BIND, D16=D32=0.0).

## CLOSED (ruled out as carriers — fixed on HEAD)
- **Cross-event h0 handoff** — CLOSED. The gate-4 collapse (accept/event 2.02→1.58) was a
  conv-remap **page-stomp** (whole-page as_strided copy dragging never-written node-column ssm
  bytes over published linear-column states), fixed WIRING-ONLY in `02b1627a`
  (`fr13_replay_conv_remap.py`, view-logical extent), merged `776368a9`, re-gated all-pass. Replay
  now EXCEEDS native (cat9 greedy accept/event 3.1789 > E5 3.1613). Our fp32 bank is intentionally
  MORE precise than native's bf16 roll — do NOT align to native's lossy bf16.
- **Conv prior-window** — CLOSED. The 18.375 divergence at num_accepted>1 was a read-COLUMN wiring
  bug (read accepted_len instead of accepted_len-1), fixed `3a9039cc`+`c0b53f5d`, byte A/B 283/283.

## REFUTED — full-attn (catalog ranked it #1; red-team holds=False)
The catalog claimed native E5 uses **FA3** on GB10 → "FA2-fork vs FA3 = different kernel" carrier.
FALSE: GB10 = sm_121 (major==12) → `get_flash_attn_version` returns **FA2** (FA3 is major==9 Hopper
only; `fa_utils.py:82-83`). **Both arms are FA2** (the cited `flash_attn.py:110` ">=3" is a
capability predicate, False on GB10, not the version). And the FA2-fork is the user-ACCEPTED
within-floor (14/16 whole-tree 0.0, 2 single-ULP, max 0.0039, NO depth growth) — it cannot be the
clear-margin 22-flip carrier; an amplifier at most.

## THE RECONCILIATION (resolves the session's confusion)
node7 ladder "L0 GDN = 0.0078 diffuse drift" was measured vs the no-spec **DECODE** oracle (a
different kernel path: sequential rank-1, not the tree-scan). The BV A/B (vs native **tree-verify**
`fused_sigmoid_gating`) = **0.0**. So the 0.0078/layer is the **tree-verify-path vs decode-path
realization difference**, NOT a kernel bug — both match their native kernels. native E5 (linear
MTP-5) has the same kind of difference but less (3 flips); cat9 (tree) accumulates more (22).

## THE CARRIER = two distinct things
1. **Diffuse near-tie drift** (~0.5 nat flips) = the tree-verify-vs-decode realization difference,
   accumulating over layers, amplified by the deep full-attn. **UNLOCALIZED to a single fixable
   seam** (geometry ruled out; conv/scan match native; gate/o_proj native). This is the
   within-floor question (the user-accepted bar = within-floor argmax-lossless), not a kernel bug.
2. **A concrete clear-margin SERVING-PATH bug** — the gold gate (FR13_B1_SWE_GOLD_BIND) found the
   tree committer serves a NON-argmax at **168× margin** (served 'Let' 0.59% vs argmax '```'
   98.88%; native TF==served 4/4 but tree TF!=served 3/4 = localized to the tree serving path,
   within-boot deterministic). That is a SELECTION/LOGIC bug, FIXABLE. **Named suspects:** FIX-A
   bonus/self at accept-run ends, eager-pack replay row, **conv-fusion committed-path at
   num_accepted>1** (the running conv-ab GPU wf tests this).

## NEXT (the right place)
Code-scan the **SERVING PATH** (the committer's token selection: FIX-A bonus/self, eager-pack
replay row, conv-fusion committed-path at num_accepted>1) vs the gold-gate evidence — the 168×
clear-margin serve is concrete and fixable, unlike the diffuse drift. Re-gate on the per-token
argmax probe (fr13_gold_margin_probe). The diffuse near-tie drift is the separate within-floor
question to evaluate vs native's 3 (is ours within an acceptable floor or genuinely worse).
Pairs with [[reference_gdn_verify_sequential_dispatch]], [[reference_scalar_metric_per_token_blindspot]],
[[project_fr13_fa2_fork_nocopy_floor]], [[feedback_math_correct_vs_bitexact]].
