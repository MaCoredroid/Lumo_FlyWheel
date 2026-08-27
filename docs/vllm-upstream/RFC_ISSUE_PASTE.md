# Paste-ready RFC issue for vllm-project/vllm — v3 (Phase-0-only, post-audit, post-#54080)

> **v3 restructure:** cut to the Phase-0 substrate ask; positioned as shared
> infrastructure under #54080 (TreeWY) and #47572 (ReplaySSM) rather than a
> competing program. Full evidence table incl. the throughput number. Sequence:
> Mark posts the #54080 comment FIRST (COMMENT_54080_DRAFT.md), ideally gets a
> reply, then files this. CC list is live handles. ~900 words.

---

## Title

[RFC]: Tree-capable recurrent spec-state interfaces for hybrid models (shared substrate for TreeWY / ReplaySSM)

## Motivation.

Tree speculative decoding for hybrid (attention + linear-recurrent) models is
now an active area: #54080 (TreeWY) proposes tree decoding for GDN models via
WY/UT reconstruction-on-commit; #47572/#47576 (ReplaySSM) establishes
replay-instead-of-snapshot as the state primitive for chain spec decode;
#52959 proposes internal state checkpoints on `MambaSpec` for align mode;
#52817 covers last-block replay with spec decode and APC on hybrid SSMs. All
four need the same three things from the recurrent spec-state layer, and none
of them currently has a home for them.

This RFC proposes those three interfaces — nothing else. It contributes the
*recurrent-state correctness discipline* the tree threads will need, not
another tree kernel.

We have been running tree speculative decoding for GDN hybrids out-of-tree
(27B Qwen hybrid, NVFP4, single GB10, agentic workloads — publicly documented
in [our engineering volumes](https://macoredroid.github.io/Lumo_FlyWheel/))
and hit the failure modes these interfaces prevent:

- **Branch-local parent selection.** A recurrent layer's state at tree node
  *n* must descend from *n*'s parent, not the physically previous row. The
  failure is not a crash: it substitutes a *sibling draft's* state — a
  plausible near-neighbor — and in our worst case corrupted ~40% of
  identifiers in generated code while passing every numerical closeness gate.
  Only output-level and task-level contracts caught it.
- **Carry accounting.** A branched accept cannot pre-export every node's
  state (for our 27B config that is ~13.7 GB/step of traffic), and the
  accepted leaf is unknown until after the rejection walk. #51855 already
  made per-algorithm slot demand a declared quantity for RecoverSSM; trees
  need the same declaration generalized (a node-count and a branch-depth are
  different carry geometries).
- **Replay on commit.** After accepting a path, every state a later read can
  touch must be rewritten as if the path had been decoded natively. Attention
  KV does not need this (positions are explicit); recurrent state does
  (position is implicit in scan order). ReplaySSM and TreeWY both already
  implement variants of this — the hook standardizes where it runs.

Measured context (all first-party, batch 1, real agent tasks — SWE-bench
subsets; baseline caveat stated honestly): our fixed-shape 32-node tree
commits **5.66 tokens/step at 196 ms** vs a tuned chain-EAGLE baseline at
**3.36 tokens/step at 115 ms** — decode throughput 28.8 vs 29.2 tok/s.
The baseline is sglang's EAGLE recipe on the same GPU and checkpoint family;
a same-stack vLLM control is owed and planned. The relevant point for this
RFC is not the throughput (parity, not a win, at width 31 — consistent with
#54080's finding that capture, not acceptance, is the bottleneck) but that
fixed shape held CUDA-graph capture through a branching verify step, and
that the correctness failures above are what the interfaces exist to prevent.

## Proposed Change.

Three interfaces on the recurrent spec-state layer
(`MambaSpecDecodeGPUContext`, `vllm/v1/worker/mamba_utils.py` — imported by
both the V1 model runner and Model Runner V2, so this lands once in shared
code), each a provable no-op for today's chain path (a chain is a tree with
fan-out 1; parent index = predecessor):

1. **Per-node parent indexing** for branch-local scans: the scan consumes an
   explicit parent table instead of assuming physical adjacency. Chain
   default: `parent[i] = i - 1`.
2. **Declared carry budget** on `MambaSpec`, generalizing the RecoverSSM
   precedent (#51855): an algorithm declares its temporal-slot and
   branch-depth demand instead of implying it. Chain default: today's
   `num_speculative_tokens` accounting, unchanged.
3. **Accepted-path replay hook**: one named point where
   reconstruct-or-replay-on-commit runs. ReplaySSM's and TreeWY's mechanisms
   both slot in; chains keep the current commit unchanged.

Interface + tests only; no tree kernels, no attention masks, no proposer or
scheduler changes. Draft PR: **[link at filing — the branch is ready]**.
We also offer, as contributed CPU tests, the adversarial regression fixtures
from our campaign (reduction-order reassociation, sibling-state corruption,
tie-break determinism) — implementation-agnostic, they apply to TreeWY and
ReplaySSM alike. Losslessness contracts are output-level; byte-exactness
claims are scoped to fan-out 1, where they are provable against native
decode.

Maintenance surface: three interfaces with chain-default tests in one shared
file, no new subsystem, no new backend — the named consumers (#54080,
#47572, #52959, #52817) are the parties who would exercise and co-own them.

**Non-goals:** no tree attention backend (the #42121 removal stands; any
future tree mask belongs inside native kernel realizations, out of scope
here); no changes to chain spec-decode behavior; no new drafting algorithms.

## Open questions.

1. **Carry-budget shape:** a struct (`temporal_slots`, `conv_tokens`,
   `max_branch_depth` — the honest model of two carry geometries) or a
   single integer (one-line change, over-allocates conv columns for trees)?
2. **Process:** would the spec-decode and hybrid-model owners prefer this to
   proceed as a standalone interface PR series, or folded under #54080's
   thread as its substrate piece? We are equally happy either way.

## Feedback Period.

Two weeks for direction. The draft PR is open now for concreteness.

## CC List.

@sneha5gsm (#54080 TreeWY) @Johnny-Liou (#47572 ReplaySSM, GDN kernels)
@ZJY0516 (#52959) @roikoren755 (#52817) @benchislett (speculators)
@LucasWilkinson (spec-decode attention, #42121/#52795)

## Any Other Things.

Risks: the main risk of *not* standardizing is each tree/replay effort
hand-rolling parent selection and commit semantics — the corruption class
above is silent and survives numerical gates. Risk of this proposal:
interface churn if TreeWY's design moves; mitigated by the no-op-for-chains
contract and by co-owning the shape with the CC'd authors.

AI assistance was used in preparing this RFC and the draft PR; the submitter
has reviewed every line. All measurements are from our own serving campaign,
documented publicly in our engineering volumes
([index](https://macoredroid.github.io/Lumo_FlyWheel/) — see in particular
[gdn-tree-scan](https://macoredroid.github.io/Lumo_FlyWheel/gdn-tree-scan.html),
[keep-or-replay](https://macoredroid.github.io/Lumo_FlyWheel/keep-or-replay.html),
[stateless-tree](https://macoredroid.github.io/Lumo_FlyWheel/stateless-tree.html),
and the negative-results volume
[numbers-that-didnt-survive](https://macoredroid.github.io/Lumo_FlyWheel/numbers-that-didnt-survive.html));
code at [MaCoredroid/Lumo_FlyWheel](https://github.com/MaCoredroid/Lumo_FlyWheel).

---

## Slack post for #contributors (after filing)

> Filed [RFC #____]: tree-capable recurrent spec-state interfaces for hybrid
> models — a small, no-behavior-change substrate piece under the active tree
> threads (#54080 TreeWY, #47572 ReplaySSM): per-node parent indexing,
> declared carry budgets, and a replay-on-commit hook, each a no-op for
> chains, with a draft PR open and adversarial correctness fixtures offered
> as contributed tests. Two bounded open questions in the issue. Feedback
> very welcome.
