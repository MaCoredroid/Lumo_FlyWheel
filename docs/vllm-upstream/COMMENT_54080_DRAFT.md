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

It didn't make the tree win, and our sign matches yours. The cost moved to
the recurrent commit — replaying the accepted path is a train of small
latency-bound per-layer kernels. We cut the committer ~100 → ~17 ms and
native still won, because the cost moved into the verify forward: our tree
accepts 4.29 tokens/event against our own native MTP-5 chain's 3.42 and is
still ~23% slower on decode throughput (32.9 vs 42.7 tok/s). Four escape
routes measured, all lost; written up here if it saves you the builds:
https://macoredroid.github.io/Lumo_FlyWheel/keep-or-replay.html. Different
silicon and a different benchmark — a second data point, not a rebuttal of
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
