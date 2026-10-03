# Completed-result manuscript update — 2 October 2026

## Source and arithmetic review

The snapshot in `results/claude-results-20261002/SYNC-MANIFEST.json` contains 271 remote files whose local SHA256 hashes match. `audit_results.py` independently reconstructs rates from individual replay records, task rates from server-counter brackets, all task outcomes from the previous hash-bound evaluator reports, and per-cycle native-normalized state ratios from the complete verdict JSON. No new inference or unit tests were run. Earlier raw records and the pre-edit paper are preserved.

## Scientific corrections to the executor summaries

- The complete new study is 30 attempts on the same ten Astropy tasks, with 26 actual test evaluations and four empty-patch failures. It resolves 5/10, 6/10 and 5/10; the older 6/10 Lumo-only deployment is a different cohort/run and is removed from the active paper summary. The historical four-attempt scikit-learn campaign is not combined with this study.
- The confirmation corpus has no message or conversation overlap with the earlier replay. Its Lumo/MTP prompt counts match, but SGLang's chat prompt counts are 33, 146 or 156 tokens greater. This is not the identical-token-ID comparison used in the earlier three-run SGLang replay.
- The pooled LumoTree/SG ratio is 1.014, while the median request-level ratio is 0.939 and LumoTree is faster on 18/43 requests. Neither statistical parity nor SG superiority is established from one run per arm.
- The proposed new tree topologies were not executed. The experiment actually varies MTP drafting passes; the suffix tail does not incur one neural pass per depth. Repaired three-pass drafting records a small +0.9% pooled difference, not a convincing winner. Earlier faulty-gate runs remain in the supplement rather than disappearing. Baseline rates and timers pool matched totals rather than average rates.
- The full-model rule detects 1/10 faults and remains INCONCLUSIVE. The previous invalid candidate run remains excluded. The post-hoc state denominator is the maximum of three native controls at the same cell, floored at0.02; it is not merely one native-kernel pair. Maximum clean state ratio is1.290348 and minimum ratio of the eight state faults is2.832897. The two stale-input faults concern KL and are described separately, rather than claiming all ten faults have that state ratio.
- Same greedy-flip count does not mean identical flips or equivalent next-token distributions. The independently reconstructed interpolated p90 KL is0.112841 for LumoTree and0.093420 for the packed native control; rounded values0.113/0.093 replace the summary's unsupported0.127/0.095. Median/max descriptions remain bounded.
- The ten-boot sampled-output sensitivity rule fails. Its8,000 samples establish no distribution-preservation verdict.

## Manuscript scope

Abstract, setup, workload tables, replay discussion, telemetry, numerical diagnostics, limitations and conclusion use the new evidence. The superseded84-path reference appendix is removed from the active narrative and retained in original source/history. Commit identities, queue names, launch chronology and failure-recovery details remain outside the academic prose. Five MTP passes are distinguished from maximum tree depth11.

## Delivery checks

Canonical compilation has no critical LaTeX warnings; bibliography entries retain their existing verified venue forms. No new external citations were introduced. The legacy writing ledger's section-path lookup is stale; a new scoped results-update ledger replaces it for this revision's QA without changing historical ledgers. The forced bibliography page break was removed to avoid an unnecessary page. Final source/PDF/render and supplement hashes are bound by this delivery's receipt. No git push, release-tag change, arXiv upload or submission is performed.

Final verification: the canonical and extracted-source PDFs both have 14 pages and identical extracted text. Both builds have zero critical LaTeX warnings. All 14 final pages were rendered and visually inspected; new result tables and numerical disclosures were also reviewed individually. The final source manifest matches all 19 canonical/package/clean-build dependencies.
