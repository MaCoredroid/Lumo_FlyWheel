# FR13 node7-ladder (corrected localization) — flips real + per-node; carrier sub-op BLOCKED by instrument

Workflow `wf_5bb14658-632` (2 boots + CPU localize). Raw:
`research/fr13_workflows/node7_ladder_wf_5bb14658.raw.json`. Adversarial verify **holds=TRUE**.
HEAD f0bf9e0e (CPU re-analysis; no commit-relevant boot).

## SOLID
- **22 clear-margin channel-2 flips REAL on the baked build** — recomputed on THIS boot's OWN stream
  (no banked reuse; cross-boot lesson applied); within-boot rep1==rep2 PASS. The tree-forward
  distribution itself prefers the divergent token (pos61 ` grep` -0.54 vs ` head` -0.92) = real
  loss, not a near-tie. Deviations 1.12-9.75 nat. Committer EXONERATED (0/944 at nonzero margin; 10
  exact-tie records). So it is purely **channel-2 verify-forward logits diverging from the clean
  no-spec oracle argmax**.
- **Deep-spine vs deep-branch = BOTH (per-node carrier).** Only prompt-3's 6 flips are reliably
  node-mappable (the other 16 have an off-by-one at served-pos 0). Of the 6: **SPINE=4, BRANCH=2**,
  spread across root (d0 node0 x2), deep spine (d3 node5, d4-tip node7 via reject-correction), a
  d2 leaf (node4, the decisive dev9.75 ` grep`->` head` flip), and an off-spine reject-correction
  (node2). NOT one depth, NOT deep-branch-specific -> a **per-node GDN verify-forward** carrier.

## BLOCKED: the carrier SUB-OP is INDETERMINATE (3 instrument mismatches)
The per-sub-op ladder (pre_conv->conv1d_out->scan->gate->o_proj) could NOT be run as a
context-matched tree-vs-native diff:
1. **Wrong topology:** the op-capture .pt files are a depth-5 BINARY tree
   (tree_parent=[-1,0,1,1,2,2,4,4,6,6], path0=[0,1,2,4,6,8]) — NOT cat9 (spine 0-1-3-5-7). The
   FR12_SUBKERNEL_CAPTURE is not capturing the locked cat9 verify path.
2. **Wrong event:** all 64 captures are the FIRST tree event (accepted_lens=[0], num_accepted=1),
   not the prompt3/pos61 (or pos73) flip events.
3. **Native ref is a PREFILL not a tree-DECODE:** the native-on-path reference is a chunked-WY
   prefill (initial_state norm=0, num_spec_decodes=0) — a different codepath than the sequential
   rank-1 fused_sigmoid_gating tree-decode; non-comparable. The no-spec DECODE hook is gated on
   num_spec_decodes>0, so it fell back to prefill.
So conv-tap/conv-window/fp8/BV remain ruled out, but scan-vs-gate-vs-o_proj-vs-in_proj is OPEN.

## DECISIVE NEXT (instrument fix, then re-capture)
(A) Fix FR12_SUBKERNEL_CAPTURE to capture the **cat9 topology at the ACTUAL flip events**
   (prompt3/pos61 node4-leaf-d2 the decisive dev9.75, and pos73 node7-spine-tip-d4) — not the
   first/binary event. (B) Add a **no-spec single-path TREE-CONTEXT DECODE** native-on-path oracle
   (sequential rank-1, fused_sigmoid_gating dispatch) on each flip's path-to-root — NOT a prefill.
   THEN run input_hidden->pre_conv->conv1d_out->gdn_scan_out->gate_out->o_proj_out per node-row;
   first super-floor nonzero sub-op = carrier.

## STRATEGIC NOTE (where to look)
Prior FR12 work drove the GDN sub-ops (scan/gate/o_proj/conv/in_proj) to **0.0 vs native**
spine+branches across all GDN layers (committed 62516997), after which the propagation
**first-diverged at LAYER 3 = full_attention** (the TREE_ATTN-vs-FLASH front). So on the current
baked build the carrier may be the **full-attention path (TREE_ATTN vs FLASH) in the deep layers**
(the "deep full-attn = amplifier" of the diffuse picture), NOT the GDN sub-ops — the re-capture
should cover BOTH GDN sub-ops AND the full-attn layers (3,7,...,63). The flips being sparse (~4.6%)
+ clear-margin suggests ONE seam crossing the margin at specific positions, not uniform diffuse
drift -> likely ONE localizable seam once the matched capture exists.

## CAVEATS (verify, none overturn the core)
HEAD is f0bf9e0e not 80ebee8c (CPU re-analysis on pre-existing artifacts). "committer 0/944" =
0 at nonzero margin (10 exact ties). p0/p1/p2 unmappable = off-by-one at pos0 (not "11-12 tok").
"pre_conv=0.0 upstream-clean" is NOT re-verified on this baked build (pre_conv nonzero both sides).
