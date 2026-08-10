# Lumo FlyWheel — gh-pages

This branch is published to GitHub Pages at
**<https://macoredroid.github.io/Lumo_FlyWheel/>**.

It contains self-contained HTML mini-papers:

- [`index.html`](index.html): Track-B Round-4b speculative-decoding ablation.
- [`round-f.html`](round-f.html): Volume II, the Round-F / Gated-DeltaNet reversal.
- [`gdn-tree-scan.html`](gdn-tree-scan.html): Volume III preview for the lossless GDN tree-scan paper.
- [`gdn-prefix-cache.html`](gdn-prefix-cache.html): Volume IV, bit-exact prefix caching (APC / EXACT_SEED) under speculative decoding.
- [`stateless-tree.html`](stateless-tree.html): Volume V, stateless tree speculative decoding — a branched verifier that caches and commits like native, plus the committed-token attention-KV garble fix.
- [`spine-reorder.html`](spine-reorder.html): Volume VI, one bit in the last place — why a wider speculative tree accepted fewer tokens (a superset violation), traced to online-softmax reduction reassociation (1 bf16 ULP) in the forked attention kernel, and the contiguous-spine reorder that restores the superset. Includes the MTP + Arctic suffix-decoding merge as future work.
- [`keep-or-replay.html`](keep-or-replay.html): Volume VII, keep or replay — why the branched tree out-accepts native linear MTP and still loses on throughput. The committer asymmetry (native keeps its recurrent states; the tree replays the accepted path through the 48-layer GDN scan on the critical path), the per-component decomposition (committer ~78% of the gap, drafter equal), the committer optimization ladder (burn redundant, replay isolated to ~16ms), the break-even model, and the honest native-wins verdict on the depth-matched 3-arm run.
- [`where-the-milliseconds-went.html`](where-the-milliseconds-went.html): Volume IX, where every millisecond went — a full accounting of the 232.780 ms/step batch-one decode step against the 119.658 ms mandatory-weight floor. What the campaign bought (step ~342 → 232.8 ms/step, drafter −61%, quality back to 9–11/16 resolved with zero give-ups, deployment batch 1.3 → 3.20), then the Nsight accounting: the target GEMM is 48.4% of the envelope with **no legal scheduler lever** (0.009 ms/step total, no wave-quantization penalty, 85.8% of the 273 GB/s roofline); the recoverable time is a 10.08 ms host-idle tail, 3.45 ms of scan imbalance, and 17.03 ms of attention headroom behind the exact-math wall. Full legal ladder = 19.4 ms/step against a 95.2 ms gap — it does not close, and that is the deliverable. Plus the composed six-kernel stack measuring null (1.0016×), the batch-four promotion criterion (a candidate posting +17.2% aggregate while every request got 2.96% slower), and Tier B: a second kernel-admission tier for candidates byte-impossible by construction, with a hard AST-pinned sampler invariant.
- [`every-lever.html`](every-lever.html): Volume VIII, every lever we pulled — a full speed-campaign inventory: 16 attempts to make the branched tree faster, reported with the observation that motivated each, its mechanism, its measured effect, and the gate that decided it. 8 paid (truncated draft head 94.9->56.3ms/step, drafter graph capture, batched committer 47->36ms, subtree-parallel scan -22.3ms/event, ring staging 32.85->38.01 tok/s); 8 did not (whole-region graph capture: 35 boots, below the detection floor; the overlap class: dead at deployment, GPU 97% busy). Together they killed the ~140ms/event row tax, leaving a purely fixed ~85-90ms/step deficit.

The full source, run drivers, graders, and formal report live on
[`main`](https://github.com/MaCoredroid/Lumo_FlyWheel/tree/main).
The formal report this page condenses from is
[`docs/reports/auto_research/track-b-round4b-ablation-formal-report-20260516.md`](https://github.com/MaCoredroid/Lumo_FlyWheel/blob/main/docs/reports/auto_research/track-b-round4b-ablation-formal-report-20260516.md).

## Publish notes
- `.nojekyll` is present so Pages serves the files as-is.
- Pages are standalone HTML files with inline CSS and local narrative/data.
- Update by editing the target HTML file and pushing to this branch.
