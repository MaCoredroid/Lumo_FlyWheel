# Paste-ready RFC issue for vllm-project/vllm — v4 (red-team verified, Phase-0-honest)

> **v4:** every claim now matches the branch at `e4638b334` (Phase-0 scoping
> said out loud in each bullet); published-site numbers only; #48018 (merged)
> and #54103 cited; template mechanics fixed (title is a form field; open
> questions live under Any Other Things; body headers are plain text).
> **Sequence (enforced):** Mark posts the #54080 comment first → HOLD until
> reply or ~3 days → open the Phase-0 PR upstream → file this with the live
> PR link → backfill + Slack. Note: the piggyback direction is already
> public on the site (Family C), so the only genuinely held asset is the
> unpublished derivation detail and fixture implementations.

---

## Title (form field — do not paste into body)

[RFC]: Tree-capable recurrent spec-state interfaces for hybrid models (shared substrate for TreeWY / ReplaySSM)

## Motivation.

Tree speculative decoding for hybrid (attention + linear-recurrent) models
is an active area this month: #54080 (TreeWY, with paper arXiv:2608.20961)
proposes tree decoding for GDN models via WY/UT reconstruction-on-commit;
#47572/#47576 (ReplaySSM) proposes cache-inputs-and-replay for spec decode,
and #48018 (merged) already lands replay-on-commit for standard decode
in-tree behind `--use-replayssm`; #52959 proposes internal state checkpoints
on `MambaSpec` (and #51855, also ZJY0516's, made per-algorithm slot demand a
declared quantity); #52817 covers last-block replay with spec decode and
APC; #54103, opened three hours after #54080, adds a third
`replayssm_commit()` entry point with an explicit CUDA/ROCm fork over which
committer runs. The fragmentation this RFC addresses is happening this week:
three replay-on-commit entry points are in flight with no shared interface.

This RFC proposes three state-layer interfaces and nothing else. No tree
kernel, no mask, no proposer.

We run tree speculative decoding for GDN hybrids out-of-tree (27B Qwen
hybrid on a single GB10; NVFP4 serving, tree measurements below from our FP8
configuration — publicly documented in
[our engineering volumes](https://macoredroid.github.io/Lumo_FlyWheel/)).
The interfaces correspond to what that took in practice:

- **Branch-local parent selection.** A recurrent layer's state at tree node
  *n* must descend from *n*'s parent, not the physically previous row. A
  wrong selection substitutes a sibling branch's state; the error is
  invisible to numerical closeness checks and surfaces only under
  output-level and task-level contracts.
- **Carry accounting.** A branched accept cannot pre-export every node's
  state (for our 27B config at 21 nodes, ~13.7 GB/step of traffic), and the
  accepted leaf is unknown until after the rejection walk. A node count and
  a branch depth are different carry geometries; the allocator should hear
  the difference declared, not implied.
- **Replay on commit.** After accepting a path, every state a later read can
  touch must be rewritten as if the path had been decoded natively.
  Attention KV does not need this (positions are explicit); recurrent state
  does (position is implicit in scan order). Three implementations of this
  are now in flight (#47576, #54080's branch, #54103); the hook names where
  it runs.

For context on where our numbers stand: our tree accepts 4.29 tokens/event
against our native MTP-5 chain's 3.42 and is still ~23% slower on decode
throughput — the same sign #54080 reports. The point of this RFC is not a
throughput claim. It is that fixed shape held CUDA-graph capture through a
branching verify step, and that the failure modes above are real, silent,
and preventable at the interface layer.

## Proposed Change.

Three Phase-0 interfaces (~130 net lines across four existing files, plus
tests), each a provable no-op for today's chain path:

1. **Per-node parent indexing.** `MambaSpecDecodeGPUContext` (in
   `vllm/v1/worker/mamba_utils.py`, imported by both the V1 model runner and
   Model Runner V2) gains an optional `node_parent_slot` table. Phase 0
   threads it through state-copy source selection only (no kernel changes)
   and consumes the root column; the per-node columns are the seam a tree
   scan would read. Chain default: `None`, which reproduces today's
   accepted-depth selection exactly.
2. **Declared carry budget.** An optional `SpecCarryBudget` on `MambaSpec`,
   generalizing the declared-slot-demand precedent (#51855). Chain default:
   `None`; today's `num_speculative_blocks` accounting remains authoritative
   and is asserted consistent.
3. **Accepted-path replay hook.** An `AcceptedPath` type threaded through
   the RecoverSSM committers, which rejects non-linear paths loudly rather
   than mis-committing them. Adoption by the ReplaySSM and TreeWY commit
   mechanisms is the follow-on this RFC exists to coordinate.

Interface + tests only. Draft PR: <insert live link — the PR is opened
immediately before this files>. We also offer, as contributed CPU tests, the
adversarial regression fixtures from our campaign (reduction-order
reassociation, sibling-state corruption, tie-break determinism) —
implementation-agnostic; they apply to TreeWY and ReplaySSM alike.
Losslessness contracts are output-level; byte-exactness claims are scoped to
fan-out 1, where they are provable against native decode.

Maintenance surface: three optional fields with chain-default tests, no new
subsystem, no new backend — the named consumers (#54080, #47576, #54103,
#52959, #52817) are the parties who would exercise and co-own them.

**Non-goals:** no tree attention backend (the #42121 removal stands); no
changes to chain spec-decode behavior; no new drafting algorithms.

## Feedback Period.

Two weeks for direction. The draft PR is open for concreteness.

## CC List.

@sneha5gsm (#54080 TreeWY) @Johnny-Liou (#47572 ReplaySSM)
@ZJY0516 (#52959, #51855) @roikoren755 (#52817) @benchislett (speculators)
@LucasWilkinson (spec-decode attention, #42121/#52795)

## Any Other Things.

Open questions:

1. **Carry-budget shape:** a struct (`temporal_slots`, `conv_tokens`,
   `max_branch_depth` — the honest model of two carry geometries) or a
   single integer (one-line change, over-allocates conv columns for trees)?
2. **RNG node identity** (a hazard worth deciding early, relevant to any
   tree method including TreeWY): sibling nodes at one tree depth share
   `pos`, so a position-keyed Gumbel stream gives siblings perfectly
   correlated draws. Re-key on `(request, node_index)`, a path hash, or
   `(pos, sibling_ordinal)`?
3. **Process:** should this proceed as a standalone interface PR series, or
   folded under #54080's thread as its substrate piece? We are equally happy
   either way.

Risks: the main risk of *not* standardizing is each replay effort
hand-rolling parent selection and commit semantics, whose failure modes are
silent to numerical checks (above); three implementations are already in
flight. Risk of this proposal: interface churn if TreeWY's design moves;
mitigated by the no-op-for-chains contract and by co-owning the shape with
the CC'd authors.

AI assistance was used in preparing this RFC and the draft PR; the submitter
has reviewed every line. All measurements are from our own serving campaign,
documented publicly in
[our engineering volumes](https://macoredroid.github.io/Lumo_FlyWheel/).

<!-- DO NOT PASTE BELOW THIS LINE -->

---

## Slack post for #contributors (after filing)

> Filed [RFC #____]: tree-capable recurrent spec-state interfaces for hybrid
> models — a small, no-behavior-change substrate piece under the active tree
> threads (#54080 TreeWY, #47572 ReplaySSM, #54103): per-node parent
> indexing, declared carry budgets, and a replay-on-commit hook, each a
> no-op for chains, with a draft PR open and adversarial correctness
> fixtures offered as contributed tests. Feedback very welcome.
