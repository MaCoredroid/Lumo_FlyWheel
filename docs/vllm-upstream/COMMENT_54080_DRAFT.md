# Comment for RFC #54080 (TreeWY) — v4 FINAL (red-team merged replacement). Mark posts.

> All numbers below are the PUBLISHED site figures (keep-or-replay,
> tree-vs-native): committer 100→17 ms, four refuted routes, 4.29 vs 3.42
> accept, 32.9 vs 42.7 tok/s. Opens on her named blocker; WY paragraph cut;
> one cross-ref (#47572); one self-link; one question; disclosure line.
> ~330 words.

---

Congrats on the paper and the branch — the `N+1`-snapshot framing and the
memory table at 397B are the part I hadn't seen quantified anywhere.

One data point on the capture blocker, since that's the thing you name as
blocking the tree. We run a fixed-shape tree verifier for GDN hybrids
out-of-tree (27B Qwen hybrid, our own vLLM fork, single GB10): a constant
node count per step, padded when the proposer emits fewer, so the verify
step has a static shape. The ancestor mask enters as an additive bias on the
score tile inside an FA2 varlen fork rather than as a separate backend. Full
CUDA-graph capture held — no PIECEWISE fallback. So width > 1 doesn't have
to cost you capture.

It didn't make the tree win at first, and our early sign matches yours: on
our FP8 generation the tree accepted 4.29 tokens/event against our native
MTP-5 chain's 3.42 and was still ~23% slower at B=1 (32.9 vs 42.7 tok/s),
worse at B=4. The wall was the recurrent commit — replaying the accepted
path is a train of small latency-bound per-layer kernels; we cut the
committer ~100 → ~17 ms and native still won. Four escape routes measured,
all lost; written up here if it saves you the builds:
https://macoredroid.github.io/Lumo_FlyWheel/keep-or-replay.html. Then a
further ladder of levers on our NVFP4 generation — pair-merged GQA loads, a
fused draft top-k, a split-K attention kernel — took the tree to 28.8 tok/s
at under 200 ms/step, B=1, serving a 27B on one GB10
(https://macoredroid.github.io/Lumo_FlyWheel/only-quantization.html). The
same-stack native control for that generation is a measurement we still owe
ourselves, so our honest summary is: gap mostly closed, not won. Different
silicon and benchmark throughout — a second data point, not a rebuttal of
your capture diagnosis.

Separately, and independent of whose verify mechanism wins: byte-exact state
does not imply byte-exact output. A reduction-order change in our attention
kernel cost 0.087 tok/event of acceptance at greedy, and sibling-state
selection errors are invisible to numerical closeness checks — a wrong
ancestor-mask entry blends a sibling into the solve and every state-level
closeness test still passes. We have CPU regression fixtures for that class
(reduction-order reassociation, sibling-state selection, tie-break
determinism); they'd cover #47572 equally and I'd like to contribute them
upstream either way.

One question. We have a small substrate change ready that TreeWY would need
too — per-node parent indexing, a declared carry budget, and a replay hook
on `MambaSpecDecodeGPUContext`, each a no-op for chains. Would you rather
that land under this RFC as its substrate piece, or as a separate interface
RFC yours depends on? Happy either way — and if your branch already covers
it, say so and I'll just send the tests.

_Disclosure: AI-assisted analysis; I ran the benchmarks and reviewed the
traces myself._
