# Comment for RFC #54080 (TreeWY) — v2 draft, plain voice. Mark reviews, Mark posts.

---

Nice work, and thanks for publishing the negative result along with the
method. We've spent the past several months on the same problem and can
confirm your main finding independently: we run tree speculative decoding
for GDN hybrids in agent serving (our own vLLM fork, 27B Qwen hybrid, the
whole campaign is written up in
[twelve public volumes](https://macoredroid.github.io/Lumo_FlyWheel/)), and
our chain baseline also beats our tree end-to-end — different hardware,
different benchmark, same sign as your sweep.

Two things we learned that might save you time:

**1. Graph capture is solvable, but it isn't the last wall.** We serve a
fixed-shape tree (32 slots per step, padded when the proposer emits fewer),
so the verify step has a static shape and captures in a full CUDA graph. The
ancestor mask sits inside the attention kernel as an additive bias, not a
separate backend; our implementation does this in an FA2 varlen fork. Once
capture held, the bottleneck moved to the commit: replaying the accepted
path through the recurrent layers is a train of small latency-bound kernels,
~77 ms/step in our setup. We measured four ways around it — commit inside
the native kernel, multi-stream, batched replay, copy-all-states — and all
four lost ([writeup](https://macoredroid.github.io/Lumo_FlyWheel/keep-or-replay.html)).
The one route we haven't exhausted is fusing the replay into the next step's
forward pass.

**2. We evaluated a WY-form verify early on and set it aside for serving** —
not on the math, which is the same one you use, but because chunked WY is a
different summation order than native decode, so bit-exact equality with the
served model can't hold. We kept WY as an fp32 oracle
([writeup](https://macoredroid.github.io/Lumo_FlyWheel/gdn-tree-scan.html),
predates this RFC). Related facts from getting the tree lossless: byte-exact state does not
imply byte-exact output; a pure reduction-order change cost us 0.087
tok/event of acceptance; and sibling-state selection errors are invisible to
numerical closeness checks — they surface only at the output level. In your
design the equivalent surface is the ancestor mask: a wrong entry blends a
sibling's contribution into the solve just as silently. We have regression
tests for these three failure classes (reduction-order reassociation,
sibling-state selection, tie-break determinism) and would like to contribute
them upstream; they don't depend on the verify mechanism, so they apply to
TreeWY as-is.

We were about to file an RFC for the state-layer interfaces this work needs:
per-node parent indexing, a declared carry budget, and a replay hook on
`MambaSpecDecodeGPUContext`, each a no-op for chains. Given your RFC, we'd
rather do that under your thread than next to it. Interested in
collaborating? We have a draft interface PR ready to open.
