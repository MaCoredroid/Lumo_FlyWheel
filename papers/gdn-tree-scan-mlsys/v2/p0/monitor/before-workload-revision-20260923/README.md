# GDN Tree-Scan v2: verifier design and optimization

This directory contains the revision requested on 2026-09-21 and the author-approved system-design reframing on 2026-09-22. The preceding manuscript remains in the parent directory; dated review packages preserve earlier v2 revisions.

The paper centers on the integrated verifier architecture, GPU scan/replay execution, and optimization mechanisms of the qualified Cat10/stock TREE_ATTN flat-slot route. Current descriptions are bound to the exact loaded source, including recorded overlays; repository HEAD alone is not an execution receipt. Superseded implementation narratives and pilot-only quantitative summaries have been removed. Historical quantitative results remain excluded; original evidence and dated snapshots remain available for audit.

All 18 original E1 timing cells remain unchanged and independently verified. The reported values are three-boot means for fixed qualified eager configurations, not a maximum-performance tuning search. B4 rates aggregate four active requests. The seed deviation, differing output streams and failed compact-candidate promotion criteria remain explicit.

E8 is complete: two qualification boots and all six fixed B1 timing boots independently PASS. ON/OFF mean rates are 12.69119/9.69541 tokens/s; the frozen mean of three paired relative changes is +30.89888%, with coarse 95% paired-block bootstrap interval [29.80504%, 32.90320%]. All original cells remain included. The last ON boot differs on all eight continuations, including p021 EOS at 101 tokens. This is an instrumented rate comparison, not equal-output acceleration or quality evidence. E1 already enabled reuse; E8 does not increase its Cat10 row again.

Latest source audit: `notes/latest-design-supersession-audit-2026-09-22.md`. Actual E8 review: `p0/monitor/e8-timing-final-redteam.md`. Final manuscript/build/artifact identities are recorded in `artifacts/FINAL-DELIVERY.json`. This is a private review draft, not an arXiv submission.

- `main.pdf`: compiled review draft.
- `main.tex`, `abstract.tex`, `ref.bib`, `figures/`: editable manuscript sources.
- `review-experiments.md`: final recommended bounded scope: three core experiments plus provenance audit, with one conditional optimization study. Completed evidence, original failures and deferred extensions are tracked in `experiments/STATUS.md` and `p0/monitor/closure-ledger.md`.
- `notes/evidence-sources.json`: paths and hashes of selected archived evidence.
- `results/historical-metrics.csv`: archived three-arm accounting proxies; `committed` is accepted + 1, `tps` is a mixed-population rate proxy, and `residual_ms` is not measured host time.
- `results/historical-supports.json`: exact global, pure-forward, pure-wall, and component-span supports.
- `p0/P0-MEASUREMENT-ERRATUM.md`: historical measurement-label correction; original P0 report and raw measurements are preserved.
- `notes/citation-verification.json`: primary arXiv metadata checked during drafting.
- `notes/closest-work-reading.md`: reading record; the substantive comparison is in the manuscript.
- `notes/numerical-comparison-decision.md`: source-linked WY chronology, Bole comparison rationale, and selected experiment decisions.
- `issues/2026-09-21-v2.csv`: drafting and validation status.
- `p0/P0-REPORT.md`: completed no-inference audit, source reconstructions, binary/model hashes, and readiness decision.
- `p0/HANDOFF-CLAUDE.md`, `p0/handoff-status.json`: authorized remote experiment scope and verified session handoff state.

`verified` in the historical evidence/design files means verified against the archive, not rerun on current HEAD. P0 reconstructed duplicate target sampling constraints for the three July patchers and retained missing historical loaded-binary/model identities as unknown. Fresh model measurements remain separate from that audit; completed execution chronology is in `experiments/STATUS.md`. Final claim scope is in `notes/claim-evidence-ledger.md`; review reports and PDF verification are in `p0/monitor/`; private evidence packages and the delivery receipt are under `artifacts/`.

From this directory, using the Python runtime in `paper.config.yaml`:

```sh
python3 scripts/audit_evidence.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The audit requires the accompanying repository's stored campaign JSON. It performs no inference, networking, GPU allocation, or model experiment. The LaTeX build uses the checked-in derived tables. The review draft is not a submission bundle.
