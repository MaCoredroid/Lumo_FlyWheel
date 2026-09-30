# Independent history-recovery review — 30 September 2026

Reviewer: Codex subagent `history_qualification_audit`, read-only. Review covers the new recovery reducer, audit, historical appendix and generated table. No experiment or file mutation was performed by the reviewer.

## Result: PASS after two corrections

The reviewer independently read all 123 bound Git blobs and reproduced blob IDs, SHA-256 values and lengths. All 48 July evaluator/runner pairs agree, with 10/16 tree, 10/16 MTP-5, and 8/16 MTP-11 resolves. Summed agent seconds are 36,672.257 / 22,423.922 / 33,427.373; generated table minutes are 611.20 / 373.73 / 557.12. All adverse attempts remain included, with zero recorded runner timeouts. Paired task outcomes agree with the audit.

The four synthetic records independently reduce to 56 completed requests and their recorded output rates. Their exclusion from agent-performance and current-tree comparisons is appropriate. Qualification dispositions preserve shadow/reference serving, the rejected B4 lifecycle, the August 15 invalidation, missing June replay raw data, and the limited current Cqc10 flush scope. The reviewer found no inappropriate transfer of historical proof to NVFP4 Hydra27.

Corrections applied by the parent:

1. Replaced branch-relative `git log -1` calibration selection with literal full commit hashes. Reproduction can no longer silently select a later calibration revision.
2. Removed the DFlash smoke/target-tap clause from the new recovery paragraph because this audit does not bind DFlash source receipts. Existing separately audited comparator discussion remains unchanged.
3. Clarified that tree and MTP-5 resolve nine tasks **in common**, each with ten total successes.

After inspecting the correction and added July31/Cqc10 paragraphs, the reviewer returned: “PASS for correction and prose closure.” Previously verified counts and the 123-blob audit remain accepted. No substantive issue remains. Final build and visual/source-bundle checks are parent-owned and separately recorded.
