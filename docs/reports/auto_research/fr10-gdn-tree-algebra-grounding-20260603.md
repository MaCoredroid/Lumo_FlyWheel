# FR10 GDN Tree-Algebra Grounding (researcher → worker note)

Author: Claude (online researcher + red-team). Audience: codex worker on P1.
Date: 2026-06-03. This is GROUNDING, not the proof. Your proof deliverable is
`fr10-gdn-tree-algebra-proof-20260603.md`; use this to get the algebra right the
first time and to know exactly which negative control must fail.

## The real recurrence (from vLLM 0.22 CPU reference)

`/tmp/vllm-0.22-src/vllm-0.22.0/vllm/model_executor/layers/mamba/ops/cpu/recurrent_gated_delta_rule.py`

Per head, state `S` is a `[v_head_dim, k_head_dim]` matrix. For token t with
gate scalar `g_t` (the code uses `g_t = exp(g)`), key `k_t`, value `v_t`,
`beta_t`, query `q_t`:

```
S_t      = g_t * S_{t-1}                      # scalar decay of whole state
kv_mem   = S_t · k_t                          # read memory at this key
delta    = (v_t - kv_mem) * beta_t            # delta-rule correction
S_t      = S_t + delta ⊗ k_t                  # rank-1 write
out_t    = S_t · q_t
```

i.e. `S_t = g_t·S_{t-1} + beta_t·(v_t − g_t·S_{t-1}·k_t)·k_tᵀ`. This is a **linear
recurrence with a scalar gate + rank-1 update**. `S_t` depends ONLY on `S_{t-1}`
and token t. That single fact is the whole losslessness story: a path's state is
a pure function of `(S_prefix, tokens-along-the-path)`, so two sibling branches
that each start from a **private copy** of the parent state evolve independently —
appending a sibling can never mutate the trunk, by construction of a causal
recurrence. The serial per-path evaluator is literally
`recurrent_gated_delta_rule()` replayed on `prefix + ancestors(node)`. Use it as
oracle #1.

## The chunked form is where the trap lives — and where the kernel comes from

`chunk_gated_delta_rule()` (same file) is the efficient form. Within a chunk it
builds:

```
cum_g      = cumsum(g)                              # along time
decay[i,j] = exp(cum_g[i] − cum_g[j])
interaction[i,j] = beta_j * (k_i · k_j)
system     = I + tril(interaction * decay, diagonal=-1)   # LOWER-TRIANGULAR
solved_*   = solve_triangular(system, …, upper=False)
intra      = tril((q @ kᵀ) * decay)                # strictly causal
```

`system` is lower-triangular with unit diagonal ⇒ row i of the solve depends only
on tokens `j ≤ i`. THAT triangularity is the mechanical reason "appending a leaf
does not mutate the trunk." Oracle #2: a single-path packed run of this chunked
form must match oracle #1.

### The tree kernel = this solve with linear masks → tree-ancestry masks

To verify a token tree in one pass (STree, arXiv:2505.14969 — "accumulating state
transition matrices according to the tree structure"), replace the *linear*
causal masks with *tree-ancestry* masks:

1. Order nodes topologically (BFS/DFS) so every ancestor precedes its descendants.
2. `cum_g[node]` = sum of `g` along the node's **root path** (ancestor-path
   accumulation), not a linear cumsum over packed position. This is STree's
   accumulated state transition.
3. `decay[i,j] = exp(cum_g[i] − cum_g[j])` is meaningful **only when j is a strict
   ancestor of i**; otherwise force the entry to 0.
4. `A[i,j] = 1 iff j is a strict ancestor of i`. Use `A` in place of `tril(...,-1)`
   for `interaction`, and `A_with_diag` in place of `tril(...)` for `intra`.

**Why it stays correct and solvable (the proof core):** under topological order,
the ancestry mask `A` is a SUBSET of the strictly-lower-triangular mask. So
`system = I + A∘interaction∘decay` is still lower-triangular with unit diagonal ⇒
still solvable by `solve_triangular` ⇒ row i (node i) depends only on its
ancestors ⇒ packed-tree node state/logit == serial-per-path, and a sibling leaf
(not an ancestor of any trunk node) contributes nothing to any trunk row. Trunk
state is computed once; each branch reads (never writes) its parent state. ∎

## RED-TEAM: the negative controls that MUST fail loudly

1. **Linear-mask leak.** If you keep the plain linear `tril` mask instead of the
   tree-ancestry mask `A`, two siblings packed adjacently land in the same
   triangular solve and one attends to the other. The packed-vs-serial parity test
   MUST FAIL in that configuration. If it "passes," your serial oracle is wrong or
   your tree is degenerate — investigate, do not celebrate.
2. **Shared mutable parent state.** If branches extend from a *shared* (not copied)
   parent `S`, the second branch sees the first branch's rank-1 writes. Parity MUST
   FAIL. Copy-on-branch (or read-only parent) is mandatory.
3. **Greedy drift.** Per-node state parity within a loose tolerance can still flip
   a greedy argmax at a low logit margin. Gate B requires the public path0 tokens
   be **byte/token-exact** vs native spine-1 decode, not merely close. Track the
   minimum logit margin and assert the tolerance is well inside it.
4. **Longest-accepted hidden winner.** Sampled selection must preserve the target
   distribution; a max/order-statistic "pick the branch that accepted most tokens"
   selector MUST FAIL the distribution gate. Shape: `tests/test_lossless_selector_gate_c_stub_design.py`.

## Cost framing (GB10)

Decode on GB10 (DGX Spark, unified memory) is **memory-bandwidth bound**, so the
kernel's real cost is state reads/writes, not FLOPs. STree's own overhead shrinks
from 2.04x→1.26x as model size grows; our win has to come from (a) computing trunk
GDN work once, (b) extending each leaf as an incremental rank-1 from the parent
state rather than replaying the trunk, (c) attention sharing prefix KV. Report the
marginal cost of one extra branch row at depth 0/1/2 (spec §7 Phase 2 table) — that
is the number the whole speed case rests on.

## Sources
- STree: Speculative Tree Decoding for Hybrid State-Space Models — arXiv:2505.14969
  (https://arxiv.org/abs/2505.14969, OpenReview a95Vd41o1u).
- vLLM 0.22 CPU GDN reference: `mamba/ops/cpu/recurrent_gated_delta_rule.py`.
- Qwen3.6 GDN layer: `mamba/gdn/qwen_gdn_linear_attn.py` (`QwenGatedDeltaNetAttention`).
