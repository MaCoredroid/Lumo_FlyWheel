# Lumo FlyWheel — gh-pages

This branch is published to GitHub Pages at
**<https://macoredroid.github.io/Lumo_FlyWheel/>**.

It contains self-contained HTML mini-papers:

- [`index.html`](index.html): Track-B Round-4b speculative-decoding ablation.
- [`round-f.html`](round-f.html): Volume II, the Round-F / Gated-DeltaNet reversal.
- [`gdn-tree-scan.html`](gdn-tree-scan.html): Volume III preview for the lossless GDN tree-scan paper.
- [`gdn-prefix-cache.html`](gdn-prefix-cache.html): Volume IV, bit-exact prefix caching (APC / EXACT_SEED) under speculative decoding.
- [`stateless-tree.html`](stateless-tree.html): Volume V, stateless tree speculative decoding — a branched verifier that caches and commits like native, plus the committed-token attention-KV garble fix.

The full source, run drivers, graders, and formal report live on
[`main`](https://github.com/MaCoredroid/Lumo_FlyWheel/tree/main).
The formal report this page condenses from is
[`docs/reports/auto_research/track-b-round4b-ablation-formal-report-20260516.md`](https://github.com/MaCoredroid/Lumo_FlyWheel/blob/main/docs/reports/auto_research/track-b-round4b-ablation-formal-report-20260516.md).

## Publish notes
- `.nojekyll` is present so Pages serves the files as-is.
- Pages are standalone HTML files with inline CSS and local narrative/data.
- Update by editing the target HTML file and pushing to this branch.
