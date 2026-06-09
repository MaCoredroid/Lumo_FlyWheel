# FR13 L4 conv — the chase was CONFOUNDED (workflow wchoqqijj, VERIFIED at captures, 2026-06-09)

## VERDICT: the L4 conv subop comparison used a POISONED native reference. Row/read/write/spine-row fixes were all chasing a capture artifact. The row-index hypothesis is FALSIFIED.

## PROVEN at output/fr13_seq_l4_subop_bv16_20260609T003555Z/l4_conv_detail_summary.json:
- `prior_bank_row_native = 1` == `prior_bank_rows_tree = [[1]]` -> tree reads the CORRECT row (row selection is right).
- `prior_cols_native = [0,1,2]` == `prior_cols_tree = [0,1,2]` -> same columns.
- `source_indices_equal_by_depth = [True,True,True,False,False,False]` -> depth-0 (the SEED) matches; divergence only at depth>=3 (separate later concern).
- conv INPUT bit-exact (pre_conv 0.0).
- **`prior_window_source_native = "post_update_fallback"`** = THE CONFOUND: native's window was read AFTER native's causal_conv1d_update shift-rolled it (one conv-step advanced), vs the tree's PRE-update window. So native[:,0]≈tree[:,1], native[:,1]≈tree[:,2] (a spurious 1-COLUMN SHIFT). The 0.0556 conv1d_out / 6.05 window "divergence" is STRUCTURALLY GUARANTEED regardless of correctness -> it CANNOT localize the L4 root.

## What's real vs poisoned
- REAL: the authoritative gateA ladder L4 hidden **0.0126** (first_nonzero layer 4), input 0.0, -> logits 1.004. L4 IS divergent.
- POISONED: the conv1d_out 0.0556 subop localization (post-update native reference). It does NOT reliably say the conv is the L4 root.

## DO NOT (banned, metric-hack): re-point the tree read/row/cols to make the post_update_fallback native window match. That fits a poisoned reference (same class as the rejected "read the tail" which moved gateA the WRONG way 0.0126->0.0253). 

## NEXT (the only valid step): clean PRE-UPDATE re-capture
Re-boot the native arm with **FR12_TREE_CONV_STATE_FULL_CAPTURE=1** (code :693,:1669-1690,:1758 -> records `prior_full_row_pre_update`, prior_window_source='pre_update'). THEN the tree-vs-native conv prior-window comparison is VALID. Re-localize the REAL L4 first_nonzero with the clean reference:
- If conv1d_out is now ~0.0 with the clean reference -> the conv was NEVER the L4 root; re-localize (a later sub-op / the gdn_scan_out / state). The whole conv chase was the confound.
- If conv1d_out genuinely diverges with the clean reference -> the real conv content fix (vs a clean reference, on the gateA ladder).
Geometry (12 vs 8 cols / 1173 vs 1196 rows) = inherent slack, NOT the bug; do not chase. Measure via the AUTHORITATIVE gateA L4. NO self-declare.
