# FR13 cost-gate research — RAW workflow results (2026-06-09)

Raw, unedited outputs of the four parallel research workflows launched to price the FR13 B=4
cost-gate (lossless feasibility × speed × drift-tracking × prior-art). The distilled,
human-readable verdicts are in the top-level `FR13_*_VERDICT.md` / `FR13_DRIFT_TRACKER_DESIGN.md`;
these are the **source** results (synthesis + adversarial-verify, verbatim) so claims are auditable.

| file | workflow | verify | verdict |
|---|---|---|---|
| `why_slower_wacoxe6i2.raw.json` | wacoxe6i2 — why slower (MEASURED) | holds=True | 2.336× = 1.432× fwd × 1.632× per-fwd; WY one-pass is the per-fwd lever; native 76% saturated + 0 branch accepts → conditional/capped → `FR13_WHY_SLOWER_VERDICT.md` |
| `nondet_source_ws5783inp.raw.json` | ws5783inp — B=4 non-det source | holds=True | diffuse (state/bank-row feedback loop, not the det-by-construction kernels); 53% floor is a diff-seed artifact; 2 cheap residual probes before STOP → `FR13_NONDET_SOURCE_VERDICT.md` |
| `drift_tracking_wet1mdi93.salvaged.json` | wet1mdi93 — robust drift tracking | flow failed (1 agent socket error); **3 readers salvaged from transcripts** | multi-sample p95 floor + forced-decode logit-KL/TV + K-seed tracker → `FR13_DRIFT_TRACKER_DESIGN.md` |
| `git_archaeology_w6m7imv2j.raw.json` | w6m7imv2j — prior-fix archaeology | *(pending — appended on completion)* | — |

Note: `output/` is gitignored, so these were copied here (tracked) to survive context resets.
