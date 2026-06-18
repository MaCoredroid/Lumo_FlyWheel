# Paper Plan: GDN Tree-Scan MLSys LaTeX Conversion

## Goal
- Convert the live GitHub Pages paper `gdn-tree-scan.html` into a compact MLSys-oriented LaTeX systems paper.
- Preserve the source page's scoped claim: served-realization tree verification for a recurrent hybrid model, lossless within the native recurrent-oracle floor.
- Keep related-work positioning precise: SGLang is adjacent production served comparison, not an apple-to-apple fused-kernel baseline.

## Scope
- In: LaTeX source, BibTeX, compiled PDF, core figures/tables, code and source-trail citations, MLSys-style systems framing.
- Out: New experiments, new performance claims, submission packaging, non-public data fabrication.

## Kickoff Gate
- [x] User confirmed scope + outline in chat by asking to use the latest GitHub Pages HTML as the blueprint and produce the LaTeX paper.
- Venue/template: MLSys-oriented paper in the fixed IEEEtran/arXiv workflow template.
- Target length: compact conference-style main paper; references excluded from main-text estimate.
- Latest definition: live GitHub Pages fetched on 2026-06-18 and byte-matched to local `Lumo_FlyWheel-gh-pages/gdn-tree-scan.html`.
- Scope boundaries: preserve the HTML's B=1/temp-0.6 clean decode result and caveats; do not promote B=4 or end-to-end task wall claims.

## Confirmed Outline
1. Introduction: problem, contribution, and scoped headline result.
2. Background and Related Work: tree speculative decoding, stateful/hybrid tree work, SGLang production comparison.
3. Verifier Contract: accepted-state theorem, attention invariant, GDN invariant, committer, publication.
4. System Design: candidate tree descriptor, FA2 tree bias, GDN scan/replay, vLLM state publication.
5. Kernel Realization: GDN recurrence, `h_cache`, scan/replay, BV/native-seam constraints.
6. Experimental Setup and Results: B=1 deployment setup, cat9/cat6root/native E5 table, measurement basis.
7. Lossless Closure and Measurement Discipline: p-rescore, oracle floor, probe separation, request anatomy.
8. Failure Modes and Limitations: failed routes, contaminated readings, B=4 and Stage D open gates.

## Visualization Plan
- System verifier pipeline flowchart.
- Verifier contract table.
- Prior-work comparison table.
- Tree-shape diagram for E5/cat9/cat6root.
- GDN scan/replay dataflow diagram.
- Results table.
- Lossless gate table.
- Canonical config table.

## Issue CSV
- Path: `papers/gdn-tree-scan-mlsys/issues/2026-06-18_10-35-gdn-tree-scan-mlsys.csv`
- This CSV is the execution contract for this conversion pass.
