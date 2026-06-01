# Round-F Enhanced Tree: Shallow MTP + Harness-Aware Suffix Strategy - 2026-05-30

## Purpose

This is a future-direction spec for beating E3 speed without repeating the
14-node F-w2 branching failure.

The core proposal is:

```text
Use MTP only as a short, expensive anchor.
Use cheap suffix decoding to extend that anchor.
Submit a small, mostly chain-shaped enhanced tree to the verifier.
Use MTP confidence and harness-aware suffix statistics to trim the tree.
```

Target:

```text
decode_tps > E3 on the same B=4 SWE/Codex workload
lossless output relative to base decoding
verified nodes/event stays in the affordable 3-8 range unless measured otherwise
```

## Current Local Evidence

The latest Round-F results split the problem cleanly:

| Arm | Nodes | accepted/draft | Decode TPS | Read |
| --- | ---: | ---: | ---: | --- |
| F-spine-d3 tree-delta | 3 | 0.788 | 34.21 | affordable, lossless, only modestly slower than E3 in clean B=4 |
| F-w2-d3 full branching | 14 | 0.142 | 14.85 | too many low-value branch nodes |

The closeout also recorded:

- F-spine state-copy reached `mean_acc_per_event = 1.995` on the uncapped
  real SWE-Verified workload.
- Branched K=2 improved after state-copy fixes, but remained below the spine.
- The root issue is not only proposal quality. The branch verifier/state-copy
  path pays extra GDN/tree overhead, and vLLM 0.19.0 does not keep
  `GDNAttentionBackend` / `TreeAttentionBackend` fully CUDA-graph captured.

Therefore, the next strategy should not be "make a wider MTP tree." It should
be "spend a small verifier budget on the highest-confidence chain."

## External Research Basis

### Snowflake / ArcticInference SuffixDecoding

Snowflake's SuffixDecoding is directly aligned with agent workloads:

- It is model-free and uses CPU suffix-tree lookups rather than an additional
  GPU draft model.
- It maintains both a global suffix tree from completed requests and a
  per-request suffix tree from the active prompt/generated tokens.
- It is designed for agentic/repetitive workloads such as coding agents,
  self-reflection loops, and multi-agent pipelines.
- ArcticInference exposes suffix decoding through vLLM-style speculative config
  and supports combining suffix decoding with a model-based speculator through
  `enable_suffix_decoding`.

Relevant sources:

- https://arcticinference.readthedocs.io/en/latest/suffix-decoding.html
- https://www.snowflake.com/en/blog/engineering/suffixdecoding-arctic-inference-vllm/
- https://github.com/snowflakedb/ArcticInference
- https://arxiv.org/abs/2411.04975
- https://suffix-decoding.github.io/

### vLLM suffix configuration

Recent vLLM speculative config exposes suffix-specific controls:

- `suffix_decoding_max_tree_depth`
- `suffix_decoding_max_spec_factor`
- `suffix_decoding_min_token_prob`
- `suffix_decoding_max_cached_requests`

This matters because the enhanced-tree policy should be adaptive: longer suffix
chains only when the match is strong, shorter or disabled suffix otherwise.

Relevant source:

- https://docs.vllm.ai/en/latest/api/vllm/config/speculative/

### Tree speculation literature

SpecInfer, Medusa, Sequoia, and related work support tree-shaped verification,
but none of them imply "verify every branch." Sequoia's central lesson is that
tree topology is a hardware-aware optimization problem. SuffixDecoding similarly
builds practical-size subtrees from frequency statistics rather than expanding
every continuation.

Our local F-w2 result is a concrete example of why: a 14-node tree can be much
slower than a 3-node spine if branch win rate is low.

## Design Thesis

Native MTP has a local sweet spot because:

```text
MTP draft cost rises with depth.
MTP acceptance quality decays with depth.
Verifier cost rises with submitted tokens.
```

For Qwen3.6 on our workload, the useful MTP depth is plausibly `n=2` or `n=3`,
not an unbounded chain and not a wide tree.

Suffix decoding changes the economics:

```text
suffix proposal cost is CPU/index lookup, not another MTP forward
agent workloads contain long repeated tool/code/schema/log patterns
suffix can propose longer chains when the match is strong
```

The verifier is still expensive, so suffix must be used as a chain extender, not
as a wide branch generator.

## Proposed Enhanced Tree Shape

### Shape A: MTP-2 + suffix chain

Primary candidate for first implementation:

```text
MTP anchor:      m1 -> m2
suffix extend:        s3 -> s4 -> s5 -> ... -> sK
submitted path: m1 -> m2 -> s3 -> s4 -> s5 ...
```

Properties:

- one chain, not a bushy tree;
- no side subtree state fork;
- verifier nodes are `2 + suffix_len`;
- if MTP prefix is wrong, suffix tail likely rejects too, but losslessness is
  preserved because the target verifier decides acceptance.

Gate:

```text
submit suffix tail only if:
  MTP prefix confidence / recent acceptance EMA is high
  suffix match length is high
  suffix token frequency score clears threshold
  total verifier nodes <= node_budget
```

### Shape B: MTP-1 + suffix chain

Safer when MTP-2 anchor confidence is poor:

```text
m1 -> s2 -> s3 -> s4 ...
```

This gives suffix a short model-based anchor without paying for deeper MTP.
Good for repeated code blocks and tool-call continuations where the suffix tree
already has a strong continuation after one predicted token.

### Shape C: spine + tiny rescue branch

Use only when the suffix/MTP scorer says the top path is fragile:

```text
main:  m1 -> m2 -> s3 -> s4
side:       alt2
```

or:

```text
main:  m1 -> s2 -> s3 -> s4
side:  alt1
```

This is not F-w2-d3. It is a top-path chain plus one or two high-value rescue
nodes. The hard budget should start at `N <= 6`.

### Shape D: suffix-only cold-start chain

When MTP is off or disabled for a test:

```text
s1 -> s2 -> s3 -> ...
```

This uses harness-aware suffix trees seeded from traces and prompt patterns.
It is the lowest GPU-overhead path and should be measured as a baseline before
adding MTP.

## Why This Can Beat E3

E3 wins when three MTP tokens have good acceptance and the MTP draft cost is
worth paying. It starts losing when deeper MTP adds cost without enough extra
accepted tokens.

Enhanced MTP+suffix can beat E3 if:

```text
MTP n=1/2 creates a high-quality anchor
suffix proposes 2-8 additional high-confidence tokens cheaply
verifier remains chain-shaped and full/piecewise-capture efficient
accepted/event increases faster than event_ms
```

Example:

```text
E3:             verify 3 nodes, accept ~2.3
MTP2+suffix4:   verify 6 nodes, accept 3.5-5.0 on repeated regions
```

This is plausible for agent harness text: file paths, shell output framing,
JSON/tool envelopes, repeated diagnostic prose, patch hunk scaffolds, and
test-run loops.

It is not plausible for novel reasoning spans. The policy must shrink to MTP-2
or suffix-off there.

## Scoring And Trim Policy

Each candidate extension should carry:

```text
source: mtp | suffix_global | suffix_request | suffix_harness_seed
depth
token_id
parent_id
prefix_match_len
suffix_frequency
mtp_confidence
recent_source_accept_ema
estimated_extra_accepted_tokens
estimated_marginal_verify_cost
score = expected_extra_accepted_tokens / estimated_marginal_verify_cost
```

Initial selection algorithm:

```text
selected = []

if mtp_enabled:
  add MTP top1 token
  if mtp2_confidence >= tau_mtp2:
    add MTP token 2

suffix_anchor = current_context + selected_tokens
suffix_chain = suffix_lookup(suffix_anchor)

for node in suffix_chain:
  if selected_nodes >= node_budget:
    break
  if suffix_score(node) < tau_suffix:
    break
  if estimated_event_ms(selected + node) > event_budget_ms:
    break
  add node

optional_rescue_nodes = top MTP/suffix side nodes by score
add only if score clears branch_threshold and selected_nodes <= rescue_budget
```

Default budgets:

```text
node_budget_initial = 6
node_budget_max = 8
branch_rescue_budget = 1 or 2
event_budget_ms = E3_event_ms * 1.05 for first pass
```

## Harness-Aware Cold-Start Suffix Trees

Snowflake SuffixDecoding uses global and per-request suffix trees. We should add
a Codex/SWE harness seed layer so the suffix tree is useful before the first few
requests have warmed it.

Seed sources:

- local `output/**/vllm_per_turn.json`
- local `output/**/per_req_spec_trace*.jsonl`
- successful `predictions.jsonl`
- patch diffs in completed SWE tasks
- common Codex tool-call wrappers
- common shell output and test-summary patterns
- repository-specific file paths from the active task workspace
- static prompt sections and AGENTS.md / task instructions

Suggested seed classes:

```text
tool_json_envelope
apply_patch_hunk
pytest_failure_loop
git_diff_context
file_path_completion
python_traceback
markdown_plan_section
codex_status_phrase
```

Data-mining pass:

1. tokenize historical generated outputs;
2. mine n-grams / suffix continuations by task family and harness phase;
3. store frequency, source class, and acceptance proxy if available;
4. load a small phase-specific suffix seed at task start;
5. update per-request tree online as Snowflake does.

This cold-start seed is explicitly not a quality shortcut. It only proposes
tokens; the target verifier still enforces losslessness.

## Integration With Snowflake ArcticInference

Preferred first implementation is to reuse ArcticInference/vLLM suffix behavior
as much as possible:

```text
ARCTIC_INFERENCE_ENABLED=1
speculative_config:
  method: suffix                  # suffix-only baseline
  num_speculative_tokens: 16       # max, not fixed length
  suffix_decoding_max_tree_depth: 24
  suffix_decoding_max_spec_factor: sweep
  suffix_decoding_min_token_prob: sweep
```

Then add Qwen MTP as a shallow anchor. There are two possible integration paths:

### Path 1: suffix primary, MTP fallback

Matches the older tau-hybrid design:

```text
if suffix score high:
  use suffix
else:
  use MTP n=1/2
```

This is easiest, but it does not express "MTP anchor then suffix extension."

### Path 2: MTP anchor, suffix extension

The desired enhanced-tree path:

```text
mtp_prefix = MTP(n=1 or 2)
suffix_anchor = context + mtp_prefix
suffix_tail = suffix_lookup(suffix_anchor)
proposal = mtp_prefix + suffix_tail
```

This likely requires a custom proposer that can call both the Qwen MTP proposer
and the suffix proposer before returning one chain to vLLM.

Path 2 should be the target if Path 1 only matches suffix-alone.

## Experiment Matrix

Use the same frozen B=4 workload and runtime as the latest E3/F-spine clean
sweep.

| Arm | MTP depth | Suffix | Shape | Max verify nodes | Purpose |
| --- | ---: | --- | --- | ---: | --- |
| E2 | 2 | off | chain | 2 | shallow MTP baseline |
| E3 | 3 | off | chain | 3 | current speed target |
| S-only | 0 | on | suffix chain | 4/8/16 | suffix cold-start and warm behavior |
| M1+S | 1 | on | chain | 4/6/8 | cheap MTP anchor |
| M2+S | 2 | on | chain | 4/6/8 | primary enhanced-tree candidate |
| M2+S+R1 | 2 | on | chain + 1 rescue | 6/8 | test tiny branch rescue |
| M2+S+R2 | 2 | on | chain + 2 rescue | 8 | upper bound before redesign |

Do not run full F-w2-d3 14-node branching as a candidate; keep it only as a
negative control.

## Metrics

Required per event:

```text
source mix: mtp_nodes, suffix_nodes, rescue_nodes
suffix_match_len
suffix_frequency_score
suffix_seed_class
selected_nodes
verified_nodes
accepted_tokens
accepted_tokens / verified_node
suffix_accepted_tokens
mtp_anchor_accept_rate
rescue_branch_win_rate
event_ms
decode_tps
verify_us
gdn_parent_gather_us
commit_us
cudagraph_runtime_mode
```

Decision metrics:

```text
decode_tps > E3
accepted/event >= E3
accepted/verified_node >= E3 or event_ms decreases enough to compensate
suffix_chain_acceptance improves with harness seed
node budget does not creep toward 14
```

## Stop Conditions

Stop or redesign if:

```text
M2+S with node_budget <= 8 cannot beat E3 decode_tps
suffix-only cold-start does not beat ngram/prompt-lookup baseline
suffix proposals mostly reject before the MTP anchor length
rescue branches have low win rate after 2-3 tasks
verify_us slope dominates suffix accepted-token gain
GDN parent gather/commit overhead rises with suffix chain length
```

## Implementation Sequence

1. Reproduce suffix-only ArcticInference/vLLM on a small local target model or
   the DGX stack if compatible.
2. Add a trace miner that builds a small Codex/SWE seed suffix corpus from local
   `output/**` artifacts.
3. Run suffix-only cold-start vs no seed vs online-only seed.
4. Run E2/E3 paired baseline on the same workload.
5. Implement M1/M2 + suffix-chain custom proposer.
6. Sweep:

```text
mtp_depth in {1, 2}
node_budget in {4, 6, 8}
suffix_min_token_prob in {0.1, 0.2, 0.4, 0.6}
suffix_max_spec_factor in {0.5, 1.0, 1.5}
```

7. Add at most one rescue branch after M2+S chain proves useful.
8. Only after chain-shaped enhanced tree beats E3 should we revisit wider
   branch trees.

## Recommendation

Proceed with enhanced MTP+suffix, but keep the tree narrow.

The most promising next point is:

```text
MTP n=2
suffix extension, chain-shaped
node_budget <= 6 initially
harness-aware suffix seed enabled
no full binary branching
```

This uses MTP where it is valuable, uses suffix where it is cheap, and respects
the current verifier bottleneck. The success criterion is not "propose more
tokens"; it is "accepted tokens per verifier millisecond beats E3."
