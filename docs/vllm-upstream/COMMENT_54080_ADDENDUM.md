# Design addendum for #54080 (TreeWY RFC) — v2 (Codex round-1 applied)
> **POSTED 2026-09-10 23:06 UTC (Mark greenlight, Codex final check #2 GO):** https://github.com/vllm-project/vllm/issues/54080#issuecomment-5626620604

> Plan v3 item P2. One substantive comment on the existing tree-for-GDN RFC;
> our public design home. v2 = Codex's 529-word replacement (v1 was 785
> words and carried unsupported claims: depth as parent identity, native-state
> equality beyond the site's scoped evidence, "both are workable", fixed
> shape ⇒ capture, "I asked" in #55688 before it was posted, "tree-hostile"
> contradicting the outside-mapping alternative). Links pinned; no numerical
> headline; ends with one question to benchislett. Not a proposal to merge
> our stack; commits to no port. Mark posts after his review; post AFTER the
> #55688 comment if that one goes first (then "the open question there" is
> literally on record). Body below the separator.

---

**Design addendum: tree verification and accepted-state publication for GDN hybrids**

@sneha5gsm — we run tree speculative decoding for Qwen GDN hybrids out of
tree on GB10. Here is the existing implementation and the boundary I think
is useful to discuss alongside TreeWY.

**1. Existing implementation.** Our verifier packs a fixed tree into one
forward. Attention rows see the committed prefix and their ancestors through
an additive bias in a forked FA2 varlen operation; the documented prefill
path stays native FA2. This uses the existing attention integration, with a
separate operation and bindings.
[Pinned patcher](https://github.com/MaCoredroid/Lumo_FlyWheel/blob/984f613d7885d9bf7898866827552b68a3199d0c/scripts/fr13_patch_fa2_tree_bias.py);
[system wiring](https://macoredroid.github.io/Lumo_FlyWheel/gdn-tree-scan.html#wiring).

**2. Verification information and state mechanisms.** Tree verification
needs parent/ancestor relationships, positions, accepted-row identity, and
consistent publication into recurrent state, convolution history, and KV
bookkeeping. Depth alone cannot identify a parent among siblings. TreeWY
reconstructs the accepted GDN state from its WY/UT representation. Our tree
scan advances each node from its parent's state and replays the accepted
path for publication. These are distinct implementations; the public
correctness evidence is scoped and does not establish interchangeable
representations or bit-identical arithmetic.
[Tree scan](https://macoredroid.github.io/Lumo_FlyWheel/gdn-tree-scan.html#kernel);
[commit experiments](https://macoredroid.github.io/Lumo_FlyWheel/keep-or-replay.html).

**3. Relationship to #55688.** That PR's ReplaySSM lifecycle takes a
per-request accepted count and maintains contiguous replay history. When a
prefix snapshot is needed, it is materialized in the same post-sampling
path. For verification records in tree-node order, the open question is
whether accepted records should be mapped outside that lifecycle or whether
it should receive accepted-node metadata. Both alternatives need matching
storage and kernel work; neither is established by our current upstream
diff.

**4. Execution and maintenance.** The out-of-tree campaign made
whole-region capture possible with fixed buffers and prerequisite host work
removed; it found no detectable additional capture-mode speedup. Fixed
shapes alone do not establish capture support in an upstream serving
configuration.
[Capture results](https://macoredroid.github.io/Lumo_FlyWheel/every-lever.html).

#42121 removed unsupported tree-attention machinery to reduce maintenance
and simplify refactoring. Avoiding a separate backend removes one
integration surface, but our forked operation, bindings, metadata, and
state publication still need maintenance. In the now-closed #46105 tracker,
FlashInfer XQA custom masks were identified as an integration opportunity
on Hopper. That is a candidate route to assess, with hardware, numerical,
and capture behavior still to establish. The open #42449 discussion also
matters; its proposed interface is not an assumed final contract. Any
future extraction needs an actual consumer and a clear maintenance
boundary. This record commits us to no port.

**5. Evidence and limits.** The public results do not establish a general
tree throughput win. Read the historical measurements with the later
sampling correction and retractions, rather than treating increased
acceptance as a serving win.
[Historical results](https://macoredroid.github.io/Lumo_FlyWheel/gdn-tree-scan.html#results);
[sampling correction](https://macoredroid.github.io/Lumo_FlyWheel/every-lever.html#bugs);
[retractions](https://macoredroid.github.io/Lumo_FlyWheel/numbers-that-didnt-survive.html).
The first-divergence observations in #54928 are others' evidence about
specific stacks, not validation of our verifier or proof of a common cause.

**6. Design recommendation.** Describe accepted verification-row identity
separately from physical replay storage. A contiguous-history consumer can
still use a count after mapping. This gives the implementations a boundary
to discuss without declaring their commit kernels interchangeable or adding
an unused interface.

cc @LucasWilkinson @Johnny-Liou

@benchislett, which concrete benchmark or maintenance constraint would be
most useful to settle before revisiting tree verification? We are recording
the existing design here and focusing current contributions on GDN
correctness.
