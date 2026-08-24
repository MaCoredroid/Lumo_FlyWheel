# [RFC]: Lossless token-tree speculative decoding for hybrid (attention + linear-recurrent) models

> **INTERNAL DRAFT v1 — for Mark's review. Not filed.**
> When approved, this text becomes a GitHub issue on `vllm-project/vllm` using the RFC
> template, then gets posted to `#contributors` in vLLM Slack with the area owners CC'd.
> Nothing in this file ships as code.

## Summary

We propose adding token-**tree** speculative decoding for hybrid models (interleaved
attention + linear-recurrent layers, e.g. the Qwen3-Next/GDN family), built on two design
commitments that we believe answer why the previous tree support was removed:

1. **No separate tree-attention backend.** Tree verification runs through the *same*
   attention kernel realization the model uses for native decode, with the tree expressed
   as a bias/visibility mask *inside* that kernel. Equivalence with native decode is then a
   property of the kernel, not a hope of the integration.
2. **Recurrent-state discipline as a first-class interface.** For linear-recurrent layers,
   tree verification requires branch-local scans with per-node parent-state selection, and
   a commit protocol that re-linearizes every state a positional read can touch. We propose
   extending the recurrent spec-decode state context (the `MambaSpecDecodeGPUContext`
   line of work) with tree-capable interfaces that are no-ops for today's chain drafters.

We have a working out-of-tree implementation serving a 27B hybrid model in production-like
agent workloads, and we are proposing a phased contribution, each phase independently
useful and reviewable.

## Motivation

**The workload.** Agentic serving is dominated by low-batch, latency-bound decode: one or
few concurrent requests, long generations, tool-call round-trips. In this regime the
verification-cost argument that removed tree attention (#42121) inverts: verify slots are
cheap (the step is weight-bound; padded verify rows ride tiles that are already paid for),
and acceptance is the whole ballgame.

**Measured on identical silicon** (single GB10, same machine, same checkpoint family, same
agent harness and 24k output ceiling):

| stack | committed tokens/step | step wall | decode TPS |
|---|---|---|---|
| chain EAGLE 3/1/4 (sglang, their blessed recipe) | 3.36 | 115.1 ms | 29.2 |
| our 31-node tree (MTP + suffix-cache drafter) | 5.66 | 196.4 ms | 28.8 |

The tree already doubles chain acceptance (+68%) at equal end throughput despite a
per-step cost we have since reduced further; the acceptance headroom is what a chain
structurally cannot reach. On SWE-bench-Verified agent tasks, tree-served and chain-served
runs resolve the same tasks (verdict parity in our A/B set) — the tree changes *speed*,
not answers. That is the point of the losslessness contract below.

**Why hybrid models specifically.** Hybrid architectures are becoming a first-class vLLM
citizen (GDN kernels, FLA vendoring, recurrent spec-decode state work, hybrid MRV2
coverage — all active this quarter). Chain MTP for these models is landing now. Trees are
the natural next step, and **trees over recurrent state are where every implementation
will get stuck**, because the hard problems are structural, not incidental. We hit and
solved them over a months-long, ledgered campaign:

- **Branch-local recurrent scan with parent-state selection.** A recurrent layer's state
  at tree node *n* must descend from *n*'s parent, not from the physically previous row.
  Getting this wrong does not crash; it corrupts *near-neighbor* content.
- **The near-neighbor trap (why numerical gates don't save you).** A sibling branch's K/V
  or recurrent state is a *plausible* neighbor of the correct one. Our worst bug class
  produced ~40% identifier corruption in generated code while passing every numerical
  closeness gate we had — because the corrupted content was drawn from a sibling draft of
  the same context. Fixed by state re-linearization (below); detectable only by task-level
  and byte-level contracts, which we ship as tests.
- **Re-linearization of every state a positional read touches.** After accepting a path
  through the tree, any state that positional reads can later touch must be rewritten as
  if the accepted path had been decoded natively. Attention KV does not need this
  (positions are explicit); recurrent state does (position is implicit in scan order).
  This asymmetry is, in our experience, the single least-obvious requirement of trees on
  hybrid models.
- **Keep-vs-replay: the structural commit cost.** A linear chain commits with an index
  select. A branched accept cannot pre-export every node's carries (for our 27B config
  that is ~13.7 GB/step of state traffic — infeasible), and the accepted leaf is unknown
  until after the rejection walk. The commit is therefore a replay through the recurrent
  layers on the critical path — a cost that must be engineered around (fused replay,
  carry-slot budgeting), not wished away. We propose interfaces that make this cost
  explicit and schedulable.
- **One bf16 ULP is enough to break superset guarantees.** In tree verification, masked
  branch keys contribute exactly zero to the softmax — and still changed accepted tokens
  in our system, because the *reduction order* differed with physical column count
  (measured: 0.087 tok/event acceptance loss from reassociation alone). This is why
  "same kernel realization" is a correctness requirement, not a style preference. Also
  measured: byte-exact intermediate state does **not** imply byte-exact output; contracts
  must bind at the output level.

Each of these comes with reproduction evidence, regression tests, and in most cases a
counterintuitive negative result that we expect reviewers to probe. We would rather
contribute the scar tissue than watch each implementation re-earn it.

**Why this does not relitigate #42121.** The removed backend was a *separate* tree
attention path with no lossless contract and no recurrent-model story, evaluated in a
throughput regime where trees pay worst. This proposal is (a) opt-in and targeted at the
low-batch regime, (b) integrated into the native kernel realization rather than a parallel
backend, (c) primarily a *state-management* contribution — the part that did not exist in
the removed code at all. The recent adaptive-verification work establishes precedent that
non-uniform, dynamically shaped verification is acceptable; a token tree is one more
shape, with the state discipline to make it correct.

## Proposed change (phased)

**Phase 0 — tree-capable recurrent state interfaces (no behavior change).**
Extend the recurrent spec-decode state context with: per-node parent indexing for
branch-local scans; a declared carry-slot budget; an accepted-path replay hook. All
default to today's chain semantics (a chain is a tree with fan-out 1). Pure interface +
tests. Target: the V2 model-runner state context, in coordination with the authors of the
current GDN/MTP kernel work.

**Phase 1 — tree visibility inside the native attention kernel.**
Express the tree mask as a bias in the model's own attention kernel path (FA4 `mask_mod`
where available; a bias-capable variant of the FA2 varlen path where not), with an
equivalence gate asserting tree-verify == native for a chain-shaped tree, bit-level.
Explicitly *not* a new backend.

**Phase 2 — tree proposer composition.**
Allow a model drafter (MTP/EAGLE-family) and a cache drafter (ngram/suffix) to fill
different tree regions. Builds on the plural `proposal_methods` schema that speculators
configs already carry.

**Phase 3 — rejection walk + lossless commit.**
The tree rejection sampler (chain sampler generalized to a walk), the re-linearizing
commit, and the output-level lossless contract tests (greedy: byte-exact vs native;
sampled: distributional gates).

Each phase is a separate PR series with its own tests; phases 0–1 are useful without 2–3
(they make *any* future tree work correct-by-construction).

## Feasibility

Working implementation exists out-of-tree (27B hybrid model, NVFP4, single GB10; ~200ms
steps at 5.66 committed tokens/step under real agent load, with the equivalence contract
enforced at boot and per-generation). We are prepared to contribute CI-runnable CPU tests
for every contract listed above, plus the adversarial cases (reassociation, sibling-state,
tie-break determinism) as regression fixtures. We expect to coordinate with — and would
welcome as co-reviewers — the authors of the current Qwen-GDN MTP kernel and recurrent
spec-state work.

## Alternatives considered

- **Separate tree-attention backend** (the removed design): rejected — equivalence with
  the served model becomes unprovable, and maintenance history speaks for itself.
- **Chain-only forever:** leaves ~1.7x acceptance on the table for agent workloads, and
  the gap grows with cache-style drafters that propose deep suffixes.
- **Block-diffusion drafting (DFlash) instead of trees:** complementary, not competing —
  trees verify *any* proposer's output, including block drafts; phase 2's composition is
  where they meet.

## Risks / open questions

- **High-batch regression risk:** mitigated by opt-in gating (engage ≤ small batch), and
  by the adaptive-verification precedent for shape-diverse batches.
- **Complexity budget:** the state interfaces are the *small* part (phase 0 is
  interface-only); the walk/commit is where LOC lives, and it arrives last, behind
  demonstrated value.
- **V1 vs Model Runner V2:** we target V2 for state work (that is where recurrent
  spec-state lives) and ask for guidance on the drafter-composition surface, which
  currently spans both.

## Evidence & provenance

Ten published engineering volumes (methodology + negative results included), a
continuously-audited measurement ledger, and same-silicon calibration against sglang's
EAGLE recipe. All numbers above carry runnable provenance; the campaign's fail-closed
methodology (every expectation derived from a single authority; four classes of
stale-expectation defects and the tripwires that catch them) ships as tests, not prose.

---
*Draft prepared 2026-08-24 from the FR13/FR14 campaign evidence. AI-assisted drafting
disclosed per vLLM contribution policy; the human submitter has reviewed and owns every
claim.*
