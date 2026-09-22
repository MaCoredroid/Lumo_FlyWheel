# GDN Tree-Scan v2 review draft

This directory contains the revision requested on 2026-09-21. The preceding manuscript remains in the parent directory.

- `main.pdf`: compiled review draft.
- `main.tex`, `abstract.tex`, `ref.bib`, `figures/`: editable manuscript sources.
- `review-experiments.md`: final recommended bounded scope: three core experiments plus provenance audit, with one conditional optimization study. No new model experiment was run.
- `notes/evidence-sources.json`: paths and hashes of selected archived evidence.
- `results/historical-metrics.csv`: values derived from the archived three-arm campaign.
- `notes/citation-verification.json`: primary arXiv metadata checked during drafting.
- `notes/closest-work-reading.md`: reading record; the substantive comparison is in the manuscript.
- `notes/numerical-comparison-decision.md`: source-linked WY chronology, Bole comparison rationale, and selected experiment decisions.
- `issues/2026-09-21-v2.csv`: drafting and validation status.
- `p0/P0-REPORT.md`: completed no-inference audit, source reconstructions, binary/model hashes, and readiness decision.
- `p0/HANDOFF-CLAUDE.md`, `p0/handoff-status.json`: authorized remote experiment scope and verified session handoff state.

`verified` in the historical evidence/design files means verified against the archive, not rerun on current HEAD. P0 reconstructed duplicate target sampling constraints for the three July patchers and retained missing historical loaded-binary/model identities as unknown. New model measurements are separate from this completed audit; remote execution status belongs in `experiments/STATUS.md` when created.

From this directory, using the Python runtime in `paper.config.yaml`:

```sh
python3 scripts/audit_evidence.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The audit requires the accompanying repository's stored campaign JSON. It performs no inference, networking, GPU allocation, or model experiment. The LaTeX build uses the checked-in derived tables. The review draft is not a submission bundle.
