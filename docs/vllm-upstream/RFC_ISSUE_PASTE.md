# Paste-ready RFC issue for vllm-project/vllm

> **How to use:** New issue → "RFC" template. Title and body below. After filing,
> post the Slack message (bottom of this file) in #contributors.
> The full design annex stays in our repo; the issue carries the load-bearing
> content inline so reviewers never need an external link.

---

## Title

[RFC]: Lossless token-tree speculative decoding for hybrid (attention + linear-recurrent) models

## Motivation

Agentic serving is dominated by low-batch, latency-bound decode: few concurrent
requests, long generations, tool-call round-trips. In this regime the
verification-cost argument that removed tree attention (#42121) inverts: verify
slots are cheap (the step is weight-bound; padded verify rows ride tiles that
are already paid for) and acceptance dominates. Measured on identical silicon
(single GB10, same checkpoint family, same agent harness): a 31-node tree over
an MTP head + suffix cache commits **5.66 tokens/step** where a chain EAGLE
3/1/4 commits **3.36** — +68% acceptance a chain structurally cannot reach.
Tree-served and chain-served runs resolve the same SWE-bench tasks (verdict
parity): the tree changes speed, not answers.

Hybrid models (interleaved attention + linear-recurrent layers, e.g. the
Qwen3-Next/GDN family) are where trees will matter next and where every
implementation will get stuck, because the hard problems are structural. We
hit and solved them over a months-long ledgered campaign on a 27B hybrid:

- **Branch-local recurrent scans with per-node parent-state selection** — a
  recurrent layer's state at tree node *n* must descend from *n*'s parent, not
  the physically previous row. Getting it wrong does not crash; it corrupts
  *near-neighbor* content (our worst bug produced ~40% identifier corruption
  while passing every numerical closeness gate, because a sibling draft's state
  is a plausible neighbor of the correct one).
- **Re-linearization of every state a positional read touches** on commit.
  Attention KV does not need this (positions are explicit); recurrent state
  does (position is implicit in scan order).
- **Keep-vs-replay** — a branched accept cannot pre-export every node's carries
  (~13.7 GB/step of state traffic at 27B) and the accepted leaf is unknown
  until after the rejection walk; the commit is a replay through the recurrent
  layers, a cost to engineer around, not wish away.
- **One bf16 ULP breaks superset guarantees** — masked branch keys contribute
  exactly zero to the softmax and still changed accepted tokens (0.087
  tok/event) because the reduction order differed with physical column count.
  Same-kernel-realization verification is a correctness requirement, not a
  style preference. Byte-exact intermediate state does not imply byte-exact
  output; contracts must bind at the output level.

Three things already in `main` answer the objections that sank trees before:

1. The adaptive-verification budgeter's own docstring states the
   monotone-survival rule ("Survival only decreases along a request, so a
   global top-k always admits continuously along steps") — path survival is
   monotone along a root→node path, so the identical rule over
   `[request, node]` admits a **connected subtree**. Runtime tree-shape
   selection is a ~10-line generalization of shipped, profiled code.
2. `DFlash2Speculator` already walks a transition-score lattice and
   re-linearizes into the **unmodified** rejection sampler — the
   propose-tree/verify-losslessly pattern is merged; we generalize it.
3. Device-decided, non-uniform verification width is the shipped contract on
   two backend families (#52157, #52795), and hybrid MTP spec decode was
   un-skipped on Model Runner V2 in #51410.

## Proposed Change

Four phases, each independently useful, each a separate reviewable PR series.
Nothing resurrects the deleted tree-attention backend: verification runs
through the model's own attention kernel realization with the tree expressed
as a visibility mask, and the recurrent-state discipline is the contribution
the removed code never had.

**Phase 0 — tree-capable recurrent spec-state interfaces (no behavior
change).** The recurrent spec-state commit (`MambaSpecDecodeGPUContext`,
`vllm/v1/worker/mamba_utils.py`) is substrate-shared — imported by both the V1
model runner and Model Runner V2 — so these land in shared code: per-node
parent indexing for branch-local scans; a declared carry-slot budget;
an accepted-path replay hook. All default to today's chain semantics (a chain
is a tree with fan-out 1). Interface + tests only.

**Phase 1 — tree visibility inside the native attention kernel.** A tree mask
via FA4 `mask_mod` (`causal=False`, self-contained), with a FLEX_ATTENTION twin
for non-FA4 platforms, and an equivalence gate asserting tree-with-fanout-1 ==
causal, bit-level. Explicitly not a new backend.

**Phase 2 — tree proposer composition.** A model drafter (MTP/EAGLE-family)
and a cache drafter (ngram/suffix) fill different tree regions; first landing
re-linearizes into the unmodified rejection sampler (the merged DFlash2
pattern), touching zero attention masks and zero kernels.

**Phase 3 — tree rejection walk + lossless commit.** The rejection kernel
gains a `TREE_VERIFICATION` mode as a peer of the existing
`USE_BLOCK_VERIFICATION` `tl.constexpr` axis (whose cross-position
residual-mass bookkeeping is the closest existing relative of a lossless tree
walk); the re-linearizing commit; output-level lossless contract tests
(greedy: byte-exact vs native; sampled: distributional gates).

Substrate: phases 0–1 are substrate-neutral by construction (shared files).
Phases 2–3 target Model Runner V2.

## Feasibility

Working out-of-tree implementation: 27B hybrid (Qwen3.8) at NVFP4 on a single
GB10, ~196 ms steps at 5.66 committed tokens/step under real agent load, with
the equivalence contract enforced at boot and per generation. We will
contribute CPU-runnable tests for every contract above, plus the adversarial
regression fixtures (reduction reassociation, sibling-state corruption,
tie-break determinism). We expect to coordinate with the authors of the
current Qwen-GDN MTP kernel and recurrent spec-state work, and would welcome
them as co-reviewers.

## Open questions for this thread

1. **RNG node identity.** The Gumbel stream keys on absolute `pos`; tree
   siblings at one depth would share draws. Re-key on `(request, node_index)`,
   a path hash, or `(pos, sibling_ordinal)`? Must match bit-for-bit between
   drafter and verifier and preserve today's chain behavior exactly.
2. **Should tree width ride `AdaptiveVerificationManager`** (widen its method
   gate; get the shipped budgeter for ~10 lines) **or be a peer mechanism?**
3. **Carry budget: struct or scalar?** `SpecCarryBudget(temporal_slots,
   conv_tokens, max_branch_depth)` models the two carry geometries honestly;
   a single integer over-allocates conv columns. Which shape do maintainers
   want on `MambaSpec`?
4. **Non-FA4 portability.** Is a FLEX_ATTENTION twin acceptable as the
   non-FA4 path, or should the tree mask go into the FA2 kernel (a
   vllm-flash-attn-repo change on its own cadence)?
5. **Re-entry sequencing.** Three independently useful PRs (a small
   fused-recurrent bugfix, the state interfaces, the mask primitive) that each
   stand alone — or the whole path behind one flag with an end-to-end
   acceptance benchmark first? We prefer the former and will follow the
   maintainers' call.

## Feedback Period

Two weeks for direction; phase-0 draft PR available on request sooner.

## CC

Area owners for spec decode and hybrid/mamba models; authors of the recent
Qwen-GDN MTP kernel and recurrent spec-state work; adaptive-verification
authors. (Fill handles at filing time from the PRs cited above.)

---

## Slack post for #contributors (after filing)

> Just filed [RFC #____]: lossless token-tree speculative decoding for hybrid
> (attention + linear-recurrent) models. TL;DR: trees double chain acceptance
> in low-batch agent serving, and the hard part on hybrids is recurrent-state
> discipline (branch-local scans, re-linearization on commit) — which we've
> built and measured out-of-tree on a 27B GDN hybrid, and want to contribute
> in phases (state interfaces first, all substrate-shared, no behavior
> change). Five concrete open questions in the issue — feedback very welcome,
> especially from the GDN/MTP kernel and adaptive-verification folks.
