# Round F Unified Spec: Cheap Proposal + Cost-Gated State-Tree Verification

**Generated:** 2026-05-29
**Scope:** `Qwen/Qwen3.6-27B-FP8`, dense 27B FP8, vLLM Track-B, native MTP
speculative decoding, batch-1 agent decode first.
**Status:** Canonical next-step spec. Not shipped behavior.
**Supersedes:** the split F_b row1 proposal spec and F_a low-cost tree verifier
spec. The split documents were useful while isolating ideas, but the next
implementation should treat proposal, trim, verification, and commit as one
pipeline.

---

## 1. Thesis

The project should be framed by the real step cost:

```text
total_spec_step_cost =
  propose_cost
  + trim_cost
  + verify_cost
  + commit/control_cost
```

The split mental model was:

```text
F_b = cheap proposal from MTP alternatives
F_a = cheap tree verification
```

The unified model is:

```text
CandidateProposer -> TreeTrimmer -> StateTreeVerifier -> Committer
```

Proposal and verification are separate budgets. Cheap proposal is already
plausible from cached MTP alternatives, and later from suffix/ngram trees with
no LLM forward. The hard work is verifying the selected tree cheaply on
`Qwen/Qwen3.6-27B-FP8`, then trimming so the verifier never sees nodes whose
expected benefit cannot pay for their marginal cost.

The first implementation goal is not "wide K." It is:

```text
prove the verifier is as close to the physical minimum as possible
```

Only after that should we spend nodes on a smarter tree.

---

## 2. Exact Model Basis

Primary model source:

- <https://huggingface.co/Qwen/Qwen3.6-27B-FP8>

Relevant model-card facts:

- model id: `Qwen/Qwen3.6-27B-FP8`;
- dense 27B language model;
- FP8 checkpoint with fine-grained FP8 quantization, block size 128;
- 64 layers;
- hidden layout:
  `16 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))`;
- Gated DeltaNet: 48 linear-attention heads for V, 16 for QK, head dim 128;
- Gated Attention: 24 Q heads, 4 KV heads, head dim 256, RoPE dim 64;
- MTP trained with multi-steps;
- vLLM recommended MTP launch uses `qwen3_next_mtp`.

The verifier implication is direct:

```text
16 Gated Attention layers can use tree/sparse attention semantics.
48 Gated DeltaNet layers need parent-indexed recurrent state semantics.
```

Tree attention alone does not define how Gated DeltaNet branch state is read and
written. That is the core F_a blocker for this model.

---

## 3. Research Basis

Public systems agree that tree verification is useful because it packs shared
prefixes into unique nodes instead of verifying root-to-leaf paths repeatedly.
They also agree that it is not free.

References:

- SpecInfer verifies token trees in parallel with a tree-based verifier:
  <https://arxiv.org/abs/2305.09781>
- Medusa uses tree-based attention for multiple candidate continuations:
  <https://arxiv.org/abs/2401.10774>
- TensorRT-LLM's Medusa notes describe consolidating shared-prefix paths with a
  sparse mask, and warn that branch/path count must be pruned:
  <https://nvidia.github.io/TensorRT-LLM/legacy/advanced/speculative-decoding.html>
- EAGLE-2 shows tree shape should be context-aware, not static:
  <https://arxiv.org/abs/2406.16858>
- Sequoia uses hardware-aware tree sizing:
  <https://arxiv.org/abs/2402.12374>
- SMART expands only while marginal benefit exceeds marginal cost:
  <https://arxiv.org/abs/2604.09731>
- STree targets tree decoding for SSM and hybrid architectures, reinforcing that
  non-Transformer state needs special handling:
  <https://arxiv.org/abs/2505.14969>

Local conclusion:

```text
cheap proposal is not enough;
cheap tree verification is not enough;
we need cheap proposal + smart trim + physically cheap state-tree verify.
```

---

## 4. Architecture

### 4.1 CandidateProposer

The proposer emits a candidate pool, not a final verified shape.

Near-term sources:

```text
top1 MTP spine
cached MTP top-p/top2 alternatives observed while drafting the spine
```

Later source:

```text
suffix/ngram trie candidates
```

Rules:

```text
row0/top1 spine must match the K1 MTP path
MTP alternatives must come from already-computed logits where possible
no extra _extend_one for a candidate that may be trimmed away
suffix/ngram candidates must not require GPU model work
proposal may over-generate a pool, but only the trimmer selects verifier nodes
```

### 4.2 TreeTrimmer

The trimmer turns the candidate pool into a selected compact trie.

Contract:

```text
candidate_pool_nodes >= selected_nodes
selected_nodes == verified_nodes
trimmed_nodes are never verified
```

The trimmer owns benefit/cost scoring. The verifier should not contain policy
logic beyond honoring the selected trie.

### 4.3 StateTreeVerifier

The verifier receives only selected unique nodes.

For each selected node:

```text
node_id
parent_id
token_id
depth
source
score
```

Full-attention layers:

```text
node attends to prompt prefix + ancestor nodes
node does not attend to siblings, descendants, or unrelated branches
position(node) = base_position + depth(node)
```

Gated DeltaNet layers:

```text
parent_state = gather(state[parent_id] or prefix_state)
parent_conv = gather(conv[parent_id] or prefix_conv)
node_state = gdn_step(node_hidden, parent_state, parent_conv)
scatter node_state by node_id
```

Sampler:

```text
greedy: descend through child matching target argmax
sampling: tree-aware rejection with residual renormalization / without-replacement semantics
bonus token: from final accepted node logits
```

### 4.4 Committer

Commit only the accepted path:

```text
append accepted token ids
copy/adopt accepted suffix KV nodes
copy/adopt final accepted GDN/conv state
discard losing branch scratch
```

First implementation can copy the accepted suffix; pointer adoption is only worth
doing after telemetry shows copy bytes matter.

---

## 5. Physical-Minimum Verify Contract

Cheap tree verification has to be measured by invariants, not asserted.

Required invariants:

```text
verified_nodes == selected_nodes
selected_nodes == unique_tree_nodes
path_rows == 0
scheduler_visible_clone_requests == 0
prefix_kv_copy_bytes == 0
recomputed_shared_prefix_nodes == 0
extra_proposer_for_trimmed_nodes == 0
accepted_path_commit_only == true
```

Physically necessary work for a selected tree:

```text
target model work for selected unique node hidden states
full-attention reads from prompt prefix + ancestor node KV
GDN parent-state reads for selected nodes
GDN state writes for selected nodes
logits required for child verification and bonus token
accepted-path-only commit
```

Suspect work:

```text
path-row duplication
recomputing shared ancestor nodes
materializing internal scheduler requests
copying prompt prefix KV into node-private buffers
verifying nodes after they have been trimmed
copying losing-branch state back to parent
```

Hard first gate:

```text
spine_only_state_tree_event_ms <= 1.05 * E3_event_ms
spine_only_top1_accept == E3_top1_accept on paired greedy prompts
physical_minimum_invariant_failures == 0
```

If spine-only state-tree is slower than this, do not evaluate branching policy.

---

## 6. Cost Model

Fit a local verifier cost model before trusting any tree policy:

```text
C_tree(N, D, E) =
  C_base
  + N * C_node
  + D * C_depth_sync
  + E * C_gdn_parent_gather
  + C_commit(accepted_depth)
```

Where:

```text
N = selected unique node count
D = max selected depth
E = parent edges / GDN parent gathers
C_node = target per-node hidden/logit/KV/GDN work
C_depth_sync = depth-group scheduling / fixed-shape launch overhead
C_gdn_parent_gather = parent-state gather/scatter overhead
C_commit = accepted-path-only commit cost
```

Fit using synthetic trees:

```text
N = 5, 8, 12, 16
same prompts
same depth budget where possible
dummy side nodes to isolate cost from candidate quality
```

The tree policy uses this fitted model to decide whether another selected node
is worth verifying.

---

## 7. Trim Algorithm

### 7.1 Candidate pool

Start with:

```text
top1 spine from MTP
cached MTP alternatives at each spine position
```

Later add:

```text
suffix/ngram trie nodes
```

Each candidate node carries:

```text
parent_id
token_id
depth
source
p_draft_token
p_alt_ratio
p_global_accept_estimate
delta_accept_estimate
delta_cost_estimate
score = delta_accept_estimate / delta_cost_estimate
```

### 7.2 Mandatory selection

Always select the top1 spine first:

```text
selected = [top1_1, top1_2, ..., top1_D]
```

Side nodes are eligible only when:

```text
parent is selected
p_global_accept_estimate >= min_global_accept
score >= min_benefit_per_cost
estimated_event_ms <= event_budget_ms
selected_nodes + subtree_nodes <= node_budget
```

### 7.3 Greedy trim

Initial policy:

```text
frontier = cached MTP alternatives whose parent is selected

while frontier is not empty:
  compute marginal_cost from C_tree
  compute expected_accept_gain
  score = expected_accept_gain / marginal_cost
  select best node if score and event budget pass
  add selected node's cheap children to frontier
  stop when no candidate pays for its marginal cost
```

Start with single-flip nodes only:

```text
top1: A -> B -> C -> D -> E
side: B -> X
side: C -> Y
side: D -> Z
```

Do not expand a side branch into a deep subtree until telemetry proves
single-flip nodes win often enough.

### 7.4 Adaptive budget

Initial runtime budgets:

```text
event_budget_ms = E3_event_ms * 1.05 for spine-only proof
event_budget_ms = E3_event_ms * 1.20 for N<=8 branch proof
max_nodes = min(config_max_nodes, nodes_allowed_by_event_budget)
```

Shrink or disable F_a when:

```text
recent_branch_win_rate is low
recent_event_ms exceeds budget
recent_accepted_per_node falls below threshold
recent_tree_score_ema < K1_score_ema
```

---

## 8. Telemetry

Every event should log one record:

```json
{
  "event": "round_f_unified_step",
  "candidate_pool_nodes": 13,
  "selected_nodes": 8,
  "verified_nodes": 8,
  "trimmed_nodes": 5,
  "max_depth": 5,
  "sources": {"mtp_top1": 5, "mtp_alt": 3, "suffix": 0},
  "path_rows": 0,
  "scheduler_visible_clone_requests": 0,
  "prefix_kv_copy_bytes": 0,
  "recomputed_shared_prefix_nodes": 0,
  "extra_proposer_for_trimmed_nodes": 0,
  "tree_attention": true,
  "gdn_parent_gather": true,
  "depth_positions": true,
  "tree_sampler": true,
  "top1_spine_accept_depth": 4,
  "accepted_depth": 4,
  "accepted_node_path": [0, 1, 2, 3],
  "estimated_event_ms": 189.5,
  "event_budget_ms": 194.2,
  "tree_score": 0.021,
  "proposer_us": 12000,
  "trim_us": 250,
  "verify_us": 180000,
  "tree_attention_us": 8000,
  "gdn_parent_gather_us": 2600,
  "depth_sync_us": 1400,
  "commit_us": 900,
  "gdn_state_bytes_copied": 123456,
  "kv_suffix_bytes_copied": 23456,
  "physical_minimum_invariant_failures": []
}
```

Aggregate:

```text
candidate_pool_nodes/event
selected_nodes/event
verified_nodes/event
trimmed_nodes/event
selected_nodes == verified_nodes rate
accepted tokens/event
accepted tokens/verified node
top1 spine accept versus E3
branch win rate by source
event_ms and tps
C_base, C_node, C_depth_sync, C_gdn_parent_gather
```

---

## 9. Sequence Of Work

### Stage 0: canonical doc and cleanup

Done by this spec:

```text
merge proposal and verifier specs into one canonical design
delete split docs to avoid competing lanes
```

### Stage 1: freeze baselines and telemetry schema

Goal: make cost comparisons trustworthy.

Run or re-use same-runtime baselines:

```text
E3 linear MTP baseline
F_b/K1 exact top1 spine baseline
K2 duplicate/path-row overhead baseline if available
```

Add telemetry fields before algorithm work:

```text
candidate_pool_nodes
selected_nodes
verified_nodes
trimmed_nodes
proposer_us
trim_us
verify_us
commit_us
copy bytes
invariant failures
```

Stop if we cannot measure event-level cost cleanly.

### Stage 2: cheap proposal shadow

Goal: prove proposal can provide candidates without extra neural work.

Implement MTP cached-alt candidate pool:

```text
row0/top1 spine exact K1
capture top-p/top2 alternatives during row0 drafting
no extra _extend_one for alternatives
no active commit change
```

Shadow output:

```text
candidate_pool_nodes
candidate source
flip position
alt probability/gap
would-have-won signal if target trace can score it
extra proposer calls == 0
```

Stop if MTP alternatives require extra proposer work or show no value on
low-accept events.

### Stage 3: spine-only state-tree verifier

Goal: prove the verifier can reproduce E3 at near physical-minimum cost.

Tree:

```text
top1 spine only, depth D
selected_nodes = verified_nodes = D
```

Pass:

```text
byte-exact greedy versus OFF where applicable
top1 accept == E3 on paired prompts
event_ms <= 1.05 * E3_event_ms
physical_minimum_invariant_failures == 0
```

Stop if this fails. Branches cannot fix a verifier that makes the spine slower
or less correct than E3.

### Stage 4: dummy-node slope

Goal: fit verifier marginal cost before using real branch acceptance.

Trees:

```text
N = 5, 8, 12, 16
top1 spine + dummy side nodes
```

Pass:

```text
event_ms monotonic with selected_nodes
fitted C_node and C_gdn_parent_gather stable enough for trim decisions
state tree cheaper than path-row equivalent for same candidate set
```

Stop if node slope is too high for N<=8 to stay within the event budget.

### Stage 5: single-flip trim

Goal: spend the smallest number of real side nodes.

Tree:

```text
top1 spine
one MTP cached alternative at low-confidence positions
max_nodes initially 8
```

Pass:

```text
acc/event >= E3
tps >= E3 and >= F_b/K1 control
accepted/verified_node improves over spine-only
branch win rate justifies node cost
```

Stop if side nodes rarely win or the trim policy overfires.

### Stage 6: adaptive trim

Goal: use online cost and branch-value estimates.

Policy:

```text
score = expected_accept_gain / marginal_verify_cost
select until score or event budget fails
disable or shrink tree when EMA score falls below K1
```

Pass:

```text
same-runtime tps >= K1
accepted/event >= K1
selected_nodes bounded by fitted budget
```

### Stage 7: suffix/ngram join

Goal: add a no-LLM proposal source only after verifier cost is proven.

Add suffix candidates to `candidate_pool`, not directly to verifier:

```text
suffix/ngram source produces trie nodes
TreeTrimmer scores them against MTP alternatives
StateTreeVerifier remains unchanged
```

Pass:

```text
suffix source improves branch win rate or accepted/event
no proposer GPU cost increase
tree budget remains respected
```

### Stage 8: productionization

Only after previous stages pass:

```text
move out of prelaunch heredoc into proper vLLM patch/fork
CUDA-graph fixed node budgets
fused GDN gather/step/scatter if needed
accepted-path pointer adoption if copy bytes matter
SWE-Bench only after microbench gates pass
```

---

## 10. Test Matrix

| Test | Purpose | Pass condition |
|---|---|---|
| E3 baseline | reference cost and acceptance | stable event_ms, acc/event |
| F_b/K1 baseline | exact top1 spine control | row0 exactness, tps reference |
| MTP cached-alt shadow | proposal value without active tree | extra proposer calls == 0 |
| Spine-only state tree | verifier correctness and cost lower bound | event_ms <= 1.05 * E3, top1 == E3 |
| Dummy-node slope | marginal node cost | stable cost fit, monotonic event_ms |
| Pruned-tree no-op | trim avoids work | selected_nodes == verified_nodes == spine nodes |
| Path-row equivalent | prove unique-node win | state tree cheaper for same candidate set |
| Single-flip trim | first real branch value | tps and acc/event >= K1/E3 |
| Adaptive trim | runtime policy | score/cost gate improves or disables safely |
| Suffix join | no-LLM proposal source | accepted/event improves without GPU proposer cost |

---

## 11. Decision Rules

Do not ship or scale F_a if any of these fail:

```text
spine-only verifier fails top1 exactness
spine-only verifier > 1.05 * E3 event_ms
selected_nodes != verified_nodes
prefix_kv_copy_bytes > 0
scheduler_visible_clone_requests > 0
node slope makes N<=8 exceed budget
single-flip branch win rate cannot pay for marginal cost
```

Proceed only when:

```text
cheap proposal is proven with extra proposer calls == 0
state-tree verifier is physically cheap on spine-only
node slope supports N<=8 within budget
trimmed side nodes improve accepted/event and tps
```

This keeps the work honest: cheap propose first, cheap verify second, smart trim
third, suffix join only after the verifier is worth feeding.
