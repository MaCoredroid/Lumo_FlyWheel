# FR13 eager regime/decompose results

Date: 2026-06-07

Config held fixed unless noted:
- SWE-4 subset: `docs/reports/auto_research/swe-bench-agentic-b4-four-verified-20260530.json`
- Probe: `scripts/fr12_deliverable_swe4_probe.py`
- Sampling: `temperature=0.6`, `top_p=0.95`, `seed=1313`, `max_tokens=64`
- Eager B=1: `--enforce-eager`, `MAX_NUM_SEQS=1`, probe `--batch-size 1`

## Branches on, tree eager B=1

Artifact:
`output/fr13_regime_decompose_20260607T021451Z/tree_branches_on_eager_b1_swe4_spp4_mt64.json`

Result:
- samples per prompt: `4`
- records: `16`
- accepted/event: `0.736936936936937`
- accepted/draft-token: `0.08188188188188188`
- warm decode TPS: `3.396447755143011`
- spec accepted tokens: `409`
- spec draft events: `555`
- spec draft tokens: `4995`

Interpretation:
Eager B=1 branches-on reproduces the low-accept regime. It is not an exclusively
B=4/cuda-graph-captured failure. The next eager-only decomposition is
spine-only branches-off vs native E5.

## Spine only, tree eager B=1

Artifact:
`output/fr13_regime_decompose_20260607T021451Z/tree_spine_only_eager_b1_swe4_spp4_mt64_fixed2.json`

Result:
- samples per prompt: `4`
- records: `16`
- accepted/event: `0.8275862068965517`
- accepted/draft-token: `0.16551724137931034`
- warm decode TPS: `4.121595655909422`
- spec accepted tokens: `408`
- spec draft events: `493`
- spec draft tokens: `2465`

Interpretation:
The low-accept eager regime reproduces with branches disabled. Branch topology is
not required. The decisive remaining eager check is whether the branchless tree
committer incorrectly dispatches through the greedy tree-LCP path at sampled
temperature, or whether the canonical stochastic path fires but miscounts or
mis-hands off accepted length.

## Tree committer policy check

Artifact:
`output/fr13_regime_decompose_20260607T021451Z/tree_spine_only_policy_check/logs/tree_sampler_debug.jsonl`

Artifact:
`output/fr13_regime_decompose_20260607T021451Z/tree_spine_only_policy_check/logs/tree_path_lcp_spine_only.jsonl`

Result:
- one-shot request: `temperature=0.6`, `top_p=0.95`, `seed=1313`, `max_tokens=32`
- `sampler_metadata`: `all_greedy=false`, `has_tree_parent_indices=true`, `has_tree_self_logits=true`
- dispatch row: `sampler_branch_enter`, `max_spec_len=5`
- accept rows: `event=tree_sample_accept`, `policy=canonical_multidraft`, `node_count=5`
- example canonical accept probabilities: `0.28299781680107117`, `0.6970598101615906`, `0.9655547142028809`

Branches-on existing artifact:
`output/fr13_regime_decompose_20260607T021451Z/tree_branches_on/logs/tree_path_lcp_branches_on.jsonl`

Branches-on result:
- rows: `982`
- events: all `tree_sample_accept`
- node count: `9`
- mean logged accepted length: `0.6934826883910387`

Interpretation:
The sampled-temperature tree path is not accidentally taking the greedy
tree-LCP committer. Spine-only explicitly reports `all_greedy=false` and enters
the canonical stochastic branch. Branches-on also uses the canonical tree sample
event path. The remaining eager failure is therefore inside canonical
tree-commit accounting/output handoff, or in an unmeasured alignment difference
between tree and native E5 around the committed/recovered token stream.

## Native E5, eager B=1

Artifact:
`output/fr13_regime_decompose_20260607T021451Z/native_e5_eager_b1_swe4_spp4_mt64.json`

Result:
- samples per prompt: `4`
- records: `16`
- accepted/event: `2.7454545454545456`
- accepted/draft-token: `0.5490909090909091`
- warm decode TPS: `15.051406888534899`
- spec accepted tokens: `755`
- spec draft events: `275`
- spec draft tokens: `1375`

Interpretation:
The same eager B=1 prompts/seed are high-accept under native E5, while tree
spine-only is low-accept. The eager failure is tree-commit/path-specific, not a
prompt/sample-size artifact.

## Matched per-node accept probability diff

Tree artifact:
`output/fr13_regime_decompose_20260607T021451Z/tree_spine_only_policy_check/logs/tree_sampler_debug.jsonl`

Native artifact:
`output/fr13_regime_decompose_20260607T021451Z/per_node_diff/native_matched_one_shot_debug.appended.jsonl`

Diff artifact:
`output/fr13_regime_decompose_20260607T021451Z/per_node_diff/tree_vs_native_matched_one_shot_full5.json`

Matched one-shot:
- prompt: `Write a concise Python function that returns the square of an integer.`
- sampling: `temperature=0.6`, `top_p=0.95`, `seed=1313`, `max_tokens=32`

Spine event 0:
- draft tokens match native at all 5 positions: `[16, 13, 2972, 2425, 64700]`
- target argmax matches native at all 5 positions
- position 0 token `16`: tree constrained `target_prob_draft=0.28299781680107117`, native `0.39712056517601013`
- position 2 token `2972`: tree `0.7825524806976318`, native `0.8094282150268555`

Spine event 1:
- draft tokens match native at all 5 positions: `[12305, 198, 727, 9057, 3456]`
- target argmax matches native at all 5 positions
- position 1 token `198`: tree `0.9241418838500977`, native `1.0`

Row-convention check:
- spine tree rows use `target_logits_index` `0..4`, matching native
  `target_logits_index` `0..4`
- spine tree rows use `self_logits_index` `1..5`
- canonical accept rows consume the same constrained probability shown by
  `tree_logit_gather.target_prob_draft`
- branches-on canonical traces use `target_prob_row=children[0]` for multi-child
  branch sets under the same parent, for example parent `0`, children `[1, 2]`,
  target row `1`

Interpretation:
Canonical is not miscounting accepted length for these matched rows. It is
accepting/rejecting against the tree constrained target probabilities it is
given. Those probabilities are lower than native for the same draft tokens,
same target row positions, and same argmax. The spine row convention is not
off-by-one. The remaining bug is upstream of canonical sampling: the tree verify
target probability tensor differs from the native E5 tensor after sampling
constraints, even when the emitted draft-token sequence and target argmax match.

## Raw vs constrained probability diff

Instrumentation commit:
`44adae75`

Tree artifact:
`output/fr13_regime_decompose_20260607T021451Z/rawprob_compare/tree_tree_sampler_debug.jsonl`

Native artifact:
`output/fr13_regime_decompose_20260607T021451Z/rawprob_compare/native_tree_sampler_debug.jsonl`

Diff artifact:
`output/fr13_regime_decompose_20260607T021451Z/rawprob_compare/tree_vs_native_raw_temp_post_matched_by_tokens.json`

Matched event 0:
- draft tokens: `[16, 13, 2972, 2425, 64700]`
- target argmax matches native at all 5 positions
- target row indices match native at all 5 positions
- position 0 token `16`: tree raw `0.29153531789779663`, native raw `0.3405103385448456`
- position 0 token `16`: tree temp `0.2808581292629242`, native temp `0.3937857747077942`
- position 0 token `16`: tree post top-p `0.19958575069904327`, native post top-p `0.4140564203262329`
- position 2 token `2972`: tree raw `0.5454047322273254`, native raw `0.5635294318199158`
- position 2 token `2972`: tree temp `0.7552117109298706`, native temp `0.7788468599319458`
- position 2 token `2972`: tree post top-p `0.957912266254425`, native post top-p `0.957912266254425`

Matched event 1:
- draft tokens: `[12305, 198, 727, 9057, 3456]`
- target argmax matches native at all 5 positions
- target row indices match native at all 5 positions
- position 1 token `198`: tree raw `0.6841292977333069`, native raw `0.9997411370277405`
- position 1 token `198`: tree temp `0.9084097146987915`, native temp `0.9999995231628418`
- position 1 token `198`: tree post top-p `1.0`, native post top-p `1.0`

Interpretation:
The probability deficit is already present in raw post-logits-processor logits,
before temperature and top-p. Top-p can amplify the deficit on boundary tokens
as in matched event 0 position 0, but it is not the root cause. The spine row
convention remains aligned (`target_logits_index` matches native). The bug is
upstream of canonical sampling and upstream of sampling constraints: tree verify
is producing a different target logit distribution from native E5 on the same
spine draft tokens and row positions.

## Scheduled verifier rows

Instrumentation commits:
`f77ea71f`, `f2cc03a6`

Tree artifact:
`output/fr13_regime_decompose_20260607T021451Z/scheduled_rows/tree_flash_sched2_debug.jsonl`

Native artifact:
`output/fr13_regime_decompose_20260607T021451Z/scheduled_rows/native_sched_debug.jsonl`

Diff artifact:
`output/fr13_regime_decompose_20260607T021451Z/scheduled_rows/tree_vs_native_scheduled_rows_by_tokens_seqpaired.json`

Matched event 0:
- draft tokens: `[16, 13, 2972, 2425, 64700]`
- tree sampled token rows: `[271, 16, 13, 2972, 2425, 64700]`
- native sampled token rows: `[271, 16, 13, 2972, 2425, 64700]`
- `logits_indices`: `[0, 1, 2, 3, 4, 5]` for both
- `target_logits_indices`: `[0, 1, 2, 3, 4]` for both
- `bonus_logits_indices`: `[5]` for both

Matched event 1:
- draft tokens: `[12305, 198, 727, 9057, 3456]`
- tree sampled token rows: `[71093, 12305, 198, 727, 9057, 3456]`
- native sampled token rows: `[71093, 12305, 198, 727, 9057, 3456]`
- `logits_indices`: `[0, 1, 2, 3, 4, 5]` for both
- `target_logits_indices`: `[0, 1, 2, 3, 4]` for both
- `bonus_logits_indices`: `[5]` for both

Interpretation:
The verifier scheduled token rows and target/bonus row indices match for the
matched spine events. The pre-constraint logit drift is not explained by
draft-token row gathering, target row indexing, or bonus row indexing. Proceed
to a full per-layer hidden-state capture and find the first diverging layer.

## Per-layer spine hidden first divergence

Capture artifact root:
`output/fr13_layer_hidden_compare_20260607T040643Z`

Tree files:
- `tree/logs/tree_layer_hidden.call2.pt`
- `tree/logs/tree_spine_logits.call2.pt`

Native files:
- `native/logs/native_layer_hidden.call2.pt`
- `native/logs/native_spine_logits.call2.pt`

Diff artifact:
`output/fr13_layer_hidden_compare_20260607T040643Z/layer_hidden_spine_compare_call2.json`

Matched spine event:
- draft tokens: `[16, 13, 2972, 2425, 64700]`
- scheduled sampled rows: `[271, 16, 13, 2972, 2425, 64700]` on both tree and native
- target row indices: `[0, 1, 2, 3, 4]` on both tree and native
- captured mRoPE positions: `[[19, 20, 21, 22, 23, 24], [19, 20, 21, 22, 23, 24], [19, 20, 21, 22, 23, 24]]` on both tree and native

First divergence:
- `first_divergence.where`: `input`
- `first_divergence.max_abs`: `0.3870849609375`
- input max-abs by depth: `[0.0, 0.326171875, 0.3870849609375, 0.29888916015625, 0.107177734375]`
- layer 0 type: `linear_attention` on both sides
- layer 0 max-abs after propagation: `0.4296875`
- final norm max-abs by depth: `[12.3125, 23.75, 25.25, 23.3125, 21.1875]`

Probability consequence on the same matched event:
- depth 0 token `16`: tree `0.19958573579788208`, native `0.4140564203262329`
- depth 1 token `13`: tree `1.0`, native `1.0`
- depth 2 token `2972`: tree `0.957912266254425`, native `0.957912266254425`
- depth 3 token `2425`: tree `0.0`, native `0.0`
- depth 4 token `64700`: tree `0.0`, native `0.0`

Interpretation:
The scheduled token rows and captured positions match exactly, so this is not a
scheduled-position or RoPE-position mismatch. The first divergence is already in
the verifier model input hidden tensor before layer 0. Therefore the root is
pre-layer verifier input/hidden-state wiring for tree verification, not GDN,
RoPE position construction, full-attention, row indexing, or the canonical
committer.
