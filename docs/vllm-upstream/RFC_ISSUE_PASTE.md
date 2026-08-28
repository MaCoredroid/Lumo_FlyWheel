# Paste-ready RFC issue for vllm-project/vllm — v4 (red-team verified, Phase-0-honest)

> **v4:** every claim now matches the branch at `e4638b334` (Phase-0 scoping
> said out loud in each bullet); published-site numbers only; #48018 (merged)
> and #54103 cited; template mechanics fixed (title is a form field; open
> questions live under Any Other Things; body headers are plain text).
> **Sequence (enforced, peer posture — v7.1):** (optional) stride_indices_tok
> bugfix PR first → open the Phase-0 PR upstream → file this RFC with the
> live PR link → Mark posts the #54080 comment citing the RFC number →
> Slack #contributors. No hold: the comment states, it does not ask.
> Held assets: the piggyback derivation detail and fixture implementations.

---

## Title (form field — do not paste into body)

[RFC]: Tree-capable recurrent spec-state interfaces for hybrid models (shared substrate for TreeWY / ReplaySSM)

## Motivation.

Tree speculative decoding for hybrid (attention + linear-recurrent) models
is an active area this month: #54080 (TreeWY, with paper arXiv:2608.20961)
proposes tree decoding for GDN models via WY/UT reconstruction-on-commit;
#47572/#47576 (ReplaySSM) proposes cache-inputs-and-replay for spec decode,
and #48018 (merged) lands the cache-inputs ReplaySSM for Mamba2 standard
decode (no spec path yet); #52959 proposes internal state checkpoints
on `MambaSpec` (and #51855, also ZJY0516's, made per-algorithm slot demand a
declared quantity); #52817 covers last-block replay with spec decode and
APC; #54103, opened three hours after #54080, adds a third
`replayssm_commit()` entry point with an explicit CUDA/ROCm fork over which
committer runs. The fragmentation this RFC addresses is happening now: four commit-time
state-rewrite implementations exist — one merged (#51855, Kimi-K3-scoped) —
with no shared interface, and #54103 explicitly forks around #51855's.

This RFC proposes three state-layer interfaces and nothing else. No tree
kernel, no mask, no proposer. The scope line, stated plainly: everything a
tree needs below the proposer, in files both runners share, that can be a
provable no-op for chains — and nothing #54080's branch already owns above
that line.

We run tree speculative decoding for GDN hybrids out-of-tree (27B Qwen
hybrid on a single GB10; NVFP4 serving, tree measurements below from our FP8
configuration — publicly documented in
[our engineering volumes](https://macoredroid.github.io/Lumo_FlyWheel/)).
Nothing below is new design: each interface is the upstream adaptation of a
mechanism our serving stack runs in production — slot-table parent
selection, per-profile declared slot ledgers, accepted-path replay —
reshaped to be a no-op for chains:

- **Branch-local parent selection.** A recurrent layer's state at tree node
  *n* must descend from *n*'s parent, not the physically previous row. A
  wrong selection substitutes a sibling branch's state; the error is
  invisible to numerical closeness checks and surfaces only under
  output-level and task-level contracts.
- **Carry accounting.** A branched accept cannot pre-export every node's
  state (for our 27B config at 21 nodes, ~13.7 GB/step of traffic), and the
  accepted leaf is unknown until after the rejection walk. A node count and
  a branch depth are different carry geometries; the allocator should hear
  the difference declared, not implied. This is already contested ground:
  #51855 (merged) and #54080's branch write different booleans onto the same
  `abstract.py` expression, and in #52959 review the preference on this
  dataclass was a declared integer over a capability bool.
- **Replay on commit.** After accepting a path, every state a later read can
  touch must be rewritten as if the path had been decoded natively.
  Attention KV does not need this (positions are explicit); recurrent state
  does (position is implicit in scan order). Four implementations exist
  (#51855 merged, #47576, #54080's branch, #54103); the hook names where it
  runs.

For context on where our numbers stand: in our matched B=4 control the tree
accepted 4.286 tokens/event against native MTP-5's 3.422 yet ran 32.85 vs
42.74 tok/s — the same sign #54080 reports — and our best lever stack later
measured 43.57 tok/s against a ~43.7 native fit (a projection, so we do not
claim the win). We have no FP8 B=1 tree-vs-native comparison; our NVFP4 3.8
arm serves 28.8 tok/s at 196 ms/step at B=1 on one GB10, with its
native-chain control still owed. On correctness: greedy output is byte-exact
against native decode at fan-out 1, enforced as a boot-time and
per-generation contract, and tree and native resolved the same 10/16 tasks
in that control. The point of this RFC is not a throughput claim. It is that a branching verify step can be fully CUDA-graph captured
when the ancestor mask is a kernel-argument bias on a persistent buffer
rather than a host-planned mask (#54080's PIECEWISE fallback is a property
of FlashInfer's `plan(custom_mask=)` channel, not of branching verify), and
that the failure modes above are real, silent, and preventable at the
interface layer.

## Proposed Change.

Three Phase-0 interfaces (≈183 net production lines across five existing
files, plus ~291 lines of tests), each a provable no-op for today's chain
path. No tree-shape producer exists at HEAD — that channel is the tree
effort's to add, and #54080's branch has already started
(`spec_decode/metadata.py`):

1. **Per-node parent indexing** (#54080's branch reaches this through
   `SpecDecodeMetadata.draft_parents`; the proposal is that that table
   become the shared one rather than per-method). `MambaSpecDecodeGPUContext` (in
   `vllm/v1/worker/mamba_utils.py`, imported by both the V1 model runner and
   Model Runner V2) gains an optional `node_parent_slot` table. Phase 0
   threads it through state-copy source selection only (no kernel changes)
   and consumes the root column; the per-node columns are the seam a tree
   scan would read. Chain default: `None`, which reproduces today's
   accepted-depth selection exactly.
2. **Declared carry budget.** Our stack declares slot demand per profile as
   a ledger the audit derives from; upstream, #51855 (merged) established
   the same declared-demand direction. The adaptation: an optional
   `SpecCarryBudget` on `MambaSpec`. Chain default: `None`; today's
   `num_speculative_blocks` accounting remains authoritative and is asserted
   consistent. The declaration's shape — scalar or struct — is open question
   1; we are equally ready to land the scalar.
3. **Accepted-path replay hook.** Widen #51855's `RecoverSSMMetadata` ABC
   beyond its Kimi-K3 scoping with an `AcceptedPath` type that rejects
   non-linear paths loudly rather than mis-committing them. Stated plainly:
   neither TreeWY nor #54103 implements that ABC today — this hook is the
   invitation this RFC exists to extend, not an existing adoption.

Interface + tests only. Draft PR: <insert live link — the PR is opened
immediately before this files>. We also offer, as contributed CPU tests, the
adversarial regression fixtures from our campaign (reduction-order
reassociation, tie-break determinism) plus an acceptance-length parity
harness — the check that catches what numerical closeness misses, on the
axis #54080's paper itself claims ("identical acceptance length"). Implementation-agnostic; they apply to TreeWY and
ReplaySSM alike.
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
2. (Pointer only) RNG node-identity keying for vectorized tree sampling is
   a sampler-layer question raised where its consumer lives, in #54080's
   thread.
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
