# GDN Tree-Scan v2: completed bounded review draft

This directory contains the revision requested on 2026-09-21, completed through independent final review on 2026-09-22. The preceding manuscript remains in the parent directory.

All 18 original timing cells are independently verified. In the as-executed comparison, the tree rate is 7.14% below native chain-5 at B1 and 14.61% below it at B4. The paper preserves the engine-seed deviation and differing output streams, so these are configuration measurements, not a quality-preserving causal speedup claim. The final 11-page PDF is built and visually checked. No unresolved material finding or necessary additional experiment remains for the stated bounded claims; broader equivalence/quality/composed-stack studies remain deferred. This is a private review draft, not an arXiv submission.

Historical quantitative results were removed from the manuscript at the author's request. Quantitative results now use fresh v2 captures or explicitly labeled fresh synthetic controls. Raw historical records and earlier review snapshots remain audit-only. The final removal review and build proof are `p0/monitor/historical-numbers-removal-redteam.md` and `p0/monitor/2026-09-22-historical-removal-build.json`.

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
