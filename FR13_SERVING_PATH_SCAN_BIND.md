# FR13 serving-path scan — the 168x serve is CHANNEL-2 (verify-forward row wrong), committer EXONERATED

Workflow `wf_42183c71-21b` (wphhdokw1, 5 agents, CPU). Raw:
`research/fr13_workflows/serving_path_scan_wphhdokw1.raw.json`. Red-team **holds=True**. 2026-06-14.

## The finding: NOT the committer, NOT the 3 named suspects — it's CHANNEL-2
The gold-gate 168x serve (p3: served 'Let' 0.59% vs clean argmax '```' 98.88%) is **CHANNEL-2**:
the tree-verify FORWARD's logits row is itself wrong; the committer faithfully argmax'd it.
- **Committer (channel-1) is STRUCTURALLY TAUTOLOGICAL** — it serves `argmax(tree_token_ids[k][node])`
  and the gate recomputes argmax of the SAME tensor → ch1_clear_margin_violations = **0/944** (the 10
  ch1 "violations" are exact ties). So the served token IS the argmax of the row it used.
- The p3 event: `reject_correction` at node 7, best_path=[0,1,3,5,7], **num_accepted=4** (deep accept);
  served = argmax(target_logits[node 7]) which the metadata builder (`:7851`) maps to **NODE 5's
  verify-forward row** (the deepest accepted spine node). That row's LIVE tree argmax = 'Let' at a
  **7.25-logit** margin, but a CLEAN single-forward of the byte-identical prefix [0,1,3,5] gives '```'
  at 98.88% (rank-2 'Let' at lp -2.033). **~7 logits of distortion = clear-margin, NOT a 1-ULP drift.**

## The 3 named suspects EXCLUDED (red-team verified on HEAD)
- **FIX-A / FR13_TREE_SAMPLE_ROW** (`:9514-9568`): rewrites only the NEXT-step DRAFTER row
  (accept-gated) — cannot change the current served reject_correction. p3 is reject_correction.
- **eager-pack**: byte-identical transport (channel-1 clean over 944 records would catch corruption).
- **conv-committed-path**: the row-SELECTION math byte-A/B (`test_t4_prepared_rows_match_frozen_library`)
  PASSES 8/8 on HEAD; the conv prior-window read-column bug is CLOSED. Row math is correct. (Caveat:
  this proves the row SELECTION is right, NOT that the conv VALUES feeding the deep spine are right.)

## The carrier = deep-spine verify-forward row distortion at num_accepted>1 (state-feed, NOT yet op-pinned)
The carrier is CHANNEL-2: node-5's verify-forward logits are distorted ~7 logits at a deep accept run
(num_accepted=4), "**growing with accept-run depth**" (matches FR13_GATE_BLINDSPOT "~4.3% argmax flip
at deep committed spine rows"). Candidate state-feeds (NOT yet pinned to one): (a) the GDN
recurrent-state handoff carried across the committed spine inside the batched tree forward, (b) the
conv prior-window prepended per-request (`:1894-1897`/`:2193-2217`).

**Open reconciliation:** the BV A/B showed our tree-scan == native fused_sigmoid_gating on the spine
(D16=0) — but that was a per-LAYER scan output at one forward; this 7-logit divergence is the
ALL-64-layer node-5-row logits at DEEP accept (num_accepted=4), a regime the BV A/B did not target.
Every per-forward GDN sub-op matches native, yet the deep-spine row diverges at clear margin → the
divergence is in how the deep-spine STATE is fed across the accept run, not a single GDN op.

## NEXT (cheapest GPU confirm — the corrected per-layer ladder, RIGHT reference)
One in-process same-boot run at the p3 step-103 event (best_path=[0,1,3,5,7], num_accepted=4): add a
per-LAYER dump of node-5's verify-forward row AND, in the same boot, teacher-force a clean
max_tokens=1 forward of the accepted prefix [0,1,3,5] capturing its per-layer last-row logits.
Compare row-by-layer; **the first layer where node-5's row diverges at clear margin pins the seam**
(GDN recurrent handoff vs conv prior-window vs full-attn). Reuses fr13_gold_margin_probe.py +
ch2_tf_probe.py, no new kernel, same-boot (no cross-boot confound). FIX = align the deep-spine
state-feed to the clean recurrent computation (NOT native's lossy bf16, NOT a reroute). The running
conv-ab GPU wf tests candidate (b). Pairs with [[reference_scalar_metric_per_token_blindspot]],
[[feedback_top_down_per_layer_lossless_gate]], [[reference_gdn_verify_sequential_dispatch]],
[[feedback_no_reroute_reward_hacking]].
