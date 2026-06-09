# FR13 L4 conv — no-op diagnosis (codex_fr19, 2026-06-09, NO GPU boots): the verdict's read-fix was a MISDIAGNOSIS. Root = the STORED conv-state CONTENT (write-back), not the read column.

## What the read/write fixes did (authoritative gateA ladder, L4 first_nonzero):
- baseline 0.0125732421875
- band-aid read-fix (tail) **0.0252532** (WORSE) — reads EMPTY tail cols
- uniform write-back **0.0125732** (byte-IDENTICAL to baseline) — did nothing

## Why (from existing captures, no boot):
1. **Read-fix engages** (prior_read_mode=rolled_tail_remapped, prior_cols=[9,10,11]) — engagement is NOT the issue.
2. **Tree conv state has data only in cols [0,1,2] (HEAD).** So reading the TAIL [9,10,11] reads EMPTY columns -> the band-aid is WORSE. The verdict's "read the tail" is WRONG; the original head-read already reads the populated columns.
3. **The STORED content is wrong** (the real root): tree state shape [171,10240,12] vs native [1196,10240,8]; populated cols [0,1,2] do NOT match native's per-token windows (best diffs 1.27/0.24/2.75, no column matches). Geometry (12 vs 8) is inherent (tree token count vs MTP-5 chain), but the per-node window CONTENT the tree stores does not match native's per-token windows.

## ROOT: the tree-verify conv-state WRITE-BACK (the handoff, scripts/fr10_phase4_patch_vllm_tree_gdn.py:1254-1312) stores the WRONG content into the conv-state bank. Fixing the READ column cannot help (data is in the head; content is wrong). 

## NEXT: reconcile the write-back CONTENT, not the read.
Read native causal_conv1d_update (what it stores per-token: the rolled window of the last width-1 token values) vs the tree write-back (:1254-1312). The tree must store, per accepted-path node, the SAME per-token conv window native stores. Measure via the AUTHORITATIVE gateA ladder (L4 0.0126), NOT the geometry-confounded subop 0.0556. Artifact: output/fr13_l4_conv_redteam_no_more_boots.json. tests/test_fr10_phase4_sampled_committer_wiring.py = 13 passed.
