# Comment for RFC #54080 (TreeWY) — DRAFT for Mark's review; Mark posts

> Post as a single GitHub comment on
> https://github.com/vllm-project/vllm/issues/54080. Nothing below the line
> is sent until you approve. Tone: supportive co-traveler with evidence,
> not competitor.

---

Congratulations on filing this — the WY/UT reconstruction is elegant, and
benchmarking against ReplaySSM before proposing is exactly the right way to
position it. We've been working the same problem from the serving side
(tree speculative decoding for GDN hybrids in low-batch agentic serving,
out-of-tree on a 27B Qwen hybrid at NVFP4, single GB10), and two of our
results bear directly on the blockers you name.

**1. Your capture wall is solvable — and we can tell you what the next wall
behind it is, because we hit it.** Our implementation serves a fixed
32-slot tree — uniform node count per step, padded when the proposer fills
fewer — so the branching verify step has a static shape and captures in a
full graph; the non-causal ancestor mask lives *inside the native attention
kernel* as a bias rather than a separate backend (FA4's `mask_mod` can
express it on capability-9/10/11 GPUs in a few hundred lines, no
flash-attention fork — happy to share the design). But we should be honest
about what that bought us: **our same-stack result has the same sign as
yours.** With capture held, our chain baseline still beats our tree
(+48% full-step in our measurements) — the bottleneck just moves from graph
capture to *committer-replay occupancy* (for us, a ~77 ms floor of
latency-bound accepted-path replay kernels). We measured and refuted four
escape routes (native-kernel commit, multi-stream, batched replay, copy-all)
before finding one still open (fusing the replay into the next step's
forward). Two teams, different silicon and benchmarks, same conclusion:
acceptance rises with width, throughput doesn't follow — yet. Where trees
did pay for us is the latency/acceptance regime under real agent workloads
(5.66 committed tokens/step vs 3.36 for a tuned chain-EAGLE recipe, batch 1,
SWE-bench tasks; cross-stack baseline, stated as such). The wall map — which
routes are dead, with numbers — is probably worth more to this RFC than any
single win.

**2. Reconstruction-on-commit needs an *output-level* losslessness
contract — state-level checks pass while outputs diverge.** Three measured
results from our campaign that apply to any tree implementation, TreeWY
included:
- byte-exact intermediate state does **not** imply byte-exact output;
- a one-bf16-ULP reduction-order difference (masked columns contribute
  exactly zero, yet physical column count changes the softmax reduction
  order) cost us 0.087 tokens/event of acceptance until verification ran
  through the same kernel realization as native decode;
- the worst failure mode of branch-local parent-state selection corrupts
  *near-neighbor* content (~40% identifier corruption in generated code)
  while passing every numerical closeness gate — only output-level and
  task-level contracts catch it.

We have adversarial regression fixtures for all three classes (reduction
reassociation, sibling-state corruption, tie-break determinism) that we'd
like to contribute upstream — they're implementation-agnostic.

Worth disclosing because it's directly on point: **we evaluated the WY-form
tree algebra ourselves and moved off it for the serve path** — not because
the math is wrong (it isn't; it's elegant) but because for our target the
verify is a sequential rank-1 recurrence, and the chunked-WY formulation is
a different *summation tree* than native decode — a different reduction
order, which is precisely the one-ULP class above. It made our bit-level
losslessness bar unprovable against the serving kernel, so WY stayed in our
stack as an **fp32 oracle** while the serve path verifies each branch with
the native scan and commits via accepted-path replay. That division of
labor — WY as the memory-optimal/oracle form, native-realization replay as
the lossless serve form — might be the right shape here too, and it's the
conversation we'd most like to have with you. (Documented before this RFC in
[the tree-scan volume](https://macoredroid.github.io/Lumo_FlyWheel/gdn-tree-scan.html).)

The campaign is written up publicly in our engineering volumes
([index](https://macoredroid.github.io/Lumo_FlyWheel/)); most relevant here:
[the branch-local GDN tree scan](https://macoredroid.github.io/Lumo_FlyWheel/gdn-tree-scan.html)
(parent selection and the near-neighbor corruption class),
[keep-or-replay](https://macoredroid.github.io/Lumo_FlyWheel/keep-or-replay.html)
(the commit-cost accounting your reconstruction approach also confronts),
[the stateless tree lifecycle](https://macoredroid.github.io/Lumo_FlyWheel/stateless-tree.html),
and [numbers that didn't survive](https://macoredroid.github.io/Lumo_FlyWheel/numbers-that-didnt-survive.html)
(our negative results, including the reduction-order one). Code lives in the
same repo ([MaCoredroid/Lumo_FlyWheel](https://github.com/MaCoredroid/Lumo_FlyWheel)).

We were preparing an RFC for the shared substrate this work needs —
per-node parent indexing, a declared carry budget, and a replay hook on
`MambaSpecDecodeGPUContext`, each a no-op for chains, complementary to
ReplaySSM's mechanism and to yours. Given your filing, we'd rather build it
under this thread than parallel to it: would you be open to collaborating?
We have a draft phase-0 interface PR ready to open for concreteness.
