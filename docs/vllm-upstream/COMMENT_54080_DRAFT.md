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

Where we ended up, on published numbers: on our FP8 3.6 model at B=4, a
matched three-arm control ran the tree at 32.85 tok/s vs native MTP-5's
42.74 (accept 4.286 vs 3.422, tied 10/16 on tasks resolved) — same sign as
your sweep. The cost sat in the recurrent commit; we cut the committer
~100 → ~17 ms
(https://macoredroid.github.io/Lumo_FlyWheel/keep-or-replay.html). After
the full lever stack, the best tree arm measured 43.57 tok/s at accept
4.749, where our native fit predicts ~43.7 — parity read literally, though
that comparator is a projection, so we don't claim the win
(https://macoredroid.github.io/Lumo_FlyWheel/every-lever.html). We never
ran a B=1 tree-vs-native race on that generation. On our NVFP4 3.8 model,
after pair-merged GQA loads, a fused draft top-k, and a split-K attention
kernel, the tree serves 28.8 tok/s at 196 ms/step at B=1 — a 27B on one
GB10 (https://macoredroid.github.io/Lumo_FlyWheel/only-quantization.html);
the native control for that generation is still owed. A second data point,
not a rebuttal of your capture diagnosis.

On the correctness axis we hold one enforced result: greedy output is
byte-exact against native decode at fan-out 1, checked as a boot-time and
per-generation contract in our stack (the 10/16 task tie above is the
task-level check). And independent of whose verify mechanism wins:
byte-exact state does not imply byte-exact output. A reduction-order change in our attention
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
