# GDN Tree-Scan v2: verifier design and optimization

This directory contains the revision requested on 2026-09-21 and the author-approved system-design reframing on 2026-09-22. The preceding manuscript remains in the parent directory; dated review packages preserve earlier v2 revisions.

The 14-page paper centers on the integrated verifier architecture, GPU scan/replay execution, and implemented optimization mechanisms. The contribution list and main sections explain the shared descriptor, state lifetime, rounding boundaries, memory/recomputation tradeoff, and backend constraints. The accounting audit supports the methods and appendix. Independent manuscript review passes with no unresolved material finding for these scoped claims.

All 18 original timing cells remain independently verified. In the as-executed comparison, the tree rate is 7.14% below native chain-5 at B1 and 14.61% below it at B4. The paper preserves the engine-seed deviation and differing output streams, so these are configuration measurements, not a quality-preserving causal speedup claim. The PDF is compiled and all pages are visually checked. This is a private review draft, not an arXiv submission.

The separate E8 single-logits study has passed both qualification boots and independent raw review: each arm completed515paired head checks, and all eight32-token API streams match across arms. This is bounded same-input head/candidate evidence, not timing or universal model-state equivalence. The independently reviewed six-cell B1 timing campaign has been handed to the existing DGX worker; no E8 timing result is available yet. Existing E1/E2/E7 evidence is unchanged. See `p0/monitor/E8-ACTIVE.json`, `p0/monitor/e8-qualification-results-redteam.md` and `experiments/e8-single-logits-timing-v1/`.

Historical quantitative results remain excluded at the author's request. Quantitative results use fresh v2 captures or explicitly labeled fresh synthetic controls. Raw historical records and earlier review snapshots remain audit-only. The current design review and build proof are `p0/monitor/design-reframe-redteam.md` and `p0/monitor/2026-09-22-design-reframe-build.json`.

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
