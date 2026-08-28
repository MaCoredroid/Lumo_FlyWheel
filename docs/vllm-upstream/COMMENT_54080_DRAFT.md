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

Our sign matches yours: in our matched B=4 control the tree accepted 4.286
vs native MTP-5's 3.422 yet ran 32.85 vs 42.74 tok/s (tied 10/16 on tasks
resolved), and a later lever stack brought the best tree arm to ~parity
with our native fit. Our NVFP4 arm now serves 28.8 tok/s at 196 ms/step at
B=1 — a 27B on one GB10. The numbers, the committer story, and what didn't
work are in our volume series:
https://macoredroid.github.io/Lumo_FlyWheel/. A second data point, not a
rebuttal of your capture diagnosis.

On the correctness axis we hold one enforced result: greedy output is
byte-exact against native decode at fan-out 1, checked as a boot-time and
per-generation contract in our stack (the 10/16 task tie above is the
task-level check). And independent of whose verify mechanism wins:
byte-exact state does not imply byte-exact output — a reduction-order
change alone cost us 0.087 tok/event of acceptance at greedy, which is why
our contracts bind at the output level. We have CPU regression fixtures for
this (reduction-order reassociation, tie-break determinism); they'd cover
#47572 equally and I'd like to contribute them upstream either way.

One question. We have a small substrate change ready that TreeWY would need
too — per-node parent indexing, a declared carry budget, and a replay hook
on `MambaSpecDecodeGPUContext`, each a no-op for chains. Would you rather
that land under this RFC as its substrate piece, or as a separate interface
RFC yours depends on? Happy either way — and if your branch already covers
it, say so and I'll just send the tests.

_Disclosure: AI-assisted analysis; I ran the benchmarks and reviewed the
traces myself._
