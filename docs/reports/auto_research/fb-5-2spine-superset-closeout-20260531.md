# Fb-5 Two-Spine Superset Closeout - 2026-05-31

## Operator Decision

Stop the sibling-request / inject-collapse agentic gate path on vLLM 0.19. It is now closed as an implementation dead end, not as a mechanism failure.

The next route is a fork or upgrade of vLLM for a single-request token-tree two-spine verifier with full-capturable GDN plus TreeAttention. The target is to keep the controlled-probe superset property while removing the sibling-row collapse class and avoiding the vLLM 0.19 piecewise CUDA graph ceiling.

## Controlled Superset Proof

Commit `5bfdcc96` (`WIP prove fb 2-spine superset on controlled B4`) is the durable mechanism proof.

Controlled uniform B=4 decode probe:

| Metric | Path0 / E spine | Winner |
|---|---:|---:|
| Events | 1,390 | 1,390 |
| Accept/event | 0.872662 | 1.061151 |
| Gain vs path0 | n/a | +0.188489 / +21.6% |
| acc=0 rate | 54.2446% | 43.3094% |

Validation counters:

- Superset violations: `0`
- Winner less than best: `0`
- Winner equals best: yes
- Internal winner events: `165`
- Stale-free-block detected: `0`
- Stale-free-block repair: `0`

Interpretation: the two-spine verifier is a clean event-local superset of path0 in a controlled uniform B=4 decode setting. It selects the longest valid segment and never accepts fewer tokens than path0.

## vLLM 0.19 Dead End

The agentic sibling-request route tried to keep path0 on the native parent request and verify path1 in a separate clone row, then collapse the loser. This fixed the earlier path0 corruption: short agentic gates reported parent-native path0 in the E5 control band (`3.0`, `3.25`, `2.25` accepted/event samples vs E5 `3.150`) with `superset_violations=0`.

The path still failed because clone collapse interacts badly with vLLM 0.19 batch-row and draft-feedback invariants:

1. Removing or collapsing the `::lumo_fb::` clone triggers `InputBatch.condense()`.
2. `condense()` reindexes batch rows and moves per-index state, including block-table rows.
3. GDN/Mamba recurrent state is effectively keyed through the first block-table slot.
4. After reindex, parent `req_id_to_index` and draft feedback can diverge.
5. Parent `spec_token_ids` stops being refilled.
6. The scheduler computes no new parent tokens, the parent remains `running`, and the run plateaus after the draft pipeline drains.

Observed symptom: the real agentic B=4 gates repeatedly reached only about four accept events, then hit zero-token scheduler steps, stale clone metadata, or KeyErrors around clone request ids in scheduler/model-runner state. Later patches handled individual symptoms but kept running into adjacent vLLM invariants.

Representative failed gate stats before engine death:

| Gate | Events | Path0 | Path1 | Winner | Violations |
|---|---:|---:|---:|---:|---:|
| gate12 | 4 | 3.00 | 0.00 | 3.00 | 0 |
| gate13 | 4 | 3.25 | 0.00 | 3.25 | 0 |
| gate14 | 4 | 2.25 | 0.00 | 2.25 | 0 |
| gate15 | 4 | 2.25 | 0.00 | 2.25 | 0 |
| gate16 | 4 | 3.25 | 0.00 | 3.25 | 0 |

These are not valid gate runs. They only show that parent-native path0 can reproduce the E5 band before the collapse/starvation failure.

## Why Not Continue Option C

The sibling-row design requires preserving all of these properties across clone collapse:

- parent batch index remains stable;
- parent GDN recurrent-state slot remains stable;
- parent draft feedback is reinserted every step;
- removed clone ids disappear consistently from scheduler output, input batch, request state, and model-runner metadata;
- no zero-token model-runner step reaches `_prepare_inputs`.

On vLLM 0.19, those constraints span scheduler, `InputBatch`, block manager, model runner, and speculative-draft feedback. The operator decision is to stop spending engineering on this collapse path.

## Upgrade Scope Snapshot

Primary upstream signals checked on 2026-05-31:

- vLLM release notes: https://github.com/vllm-project/vllm/releases
- vLLM speculative config docs/source: https://docs.vllm.ai/en/stable/api/vllm/config/speculative/
- Qwen3.5/Qwen3.6 recipe: https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md
- MRV2 Qwen3.5/Mamba hybrid PR: https://github.com/vllm-project/vllm/pull/35520
- MTP CUDA graph capture issue class: https://github.com/vllm-project/vllm/issues/28207

Current read:

- `v0.19.0` added zero-bubble async scheduling for speculative decoding and MRV2 maturation, but the local evidence shows the two-spine token-tree route was still constrained by piecewise CUDA graph / TreeAttention capture behavior on this stack.
- `v0.20.0` brought PyTorch 2.11, CUDA 13 default wheels/images, FA4 and many Qwen3.5/GDN fixes, but it does not by itself document full GDN plus TreeAttention capture for token-tree speculative decoding.
- `v0.21.0` is the first clear post-0.19 upgrade candidate for this work because it merged Model Runner V2 support for Qwen3.5/Mamba hybrid models and documents speculative decoding fixes for reasoning/thinking budgets. The MRV2 PR was tested with Qwen3.5 FP8 plus MTP, but it still does not explicitly prove the exact requirement: full-captured single-request token-tree verification over GDN plus TreeAttention.
- The current Qwen recipe documents Qwen3.6 serving and MTP speculative decoding, including `{"method": "mtp", "num_speculative_tokens": 2}`, but the recipe is a main-branch usage guide rather than a pinned release guarantee.

Provisional upgrade base: start from current `v0.21.0` or newer main/nightly, not `v0.20.x`, then run a local capture audit before porting the two-spine token tree.

Required acceptance checks for the upgrade/fork:

1. Qwen3.6 FP8/hybrid boots with the production serving posture and MTP enabled.
2. Token-tree speculative config accepts a two-branch shape for the parent request.
3. GDN recurrent state is keyed to the parent request/token tree, not a mutable sibling batch row.
4. CUDA graph logs show the target GDN plus TreeAttention path is full-captured for uniform decode, not falling back to the vLLM 0.19 piecewise path.
5. Controlled uniform B=4 proof reproduces `winner > path0`, `violations=0`.
6. Real agentic B=4 gate reproduces path0 near E5 (`3.150` accepted/event, `13.3%` acc=0) before trusting winner.
7. Winner accepted/event exceeds path0 and then exceeds the E5 absolute gate.
8. Decode TPS is reported against E5 `26.86 tps`.

If upstream main still lacks full capture, the minimal fork patch should be scoped around the single-request path only:

- keep the normal parent request, no clone rows;
- express spine A and spine B as a breadth-first token tree in `speculative_token_tree`;
- add or adapt the proposer so branch 0 is the native E5 chain and branch 1 is the alternate root plus greedy continuation;
- make the GDN recurrent-state update and rollback tree-aware inside the parent request;
- keep TreeAttention masks graph-capturable for the fixed two-spine/depth-5 shape;
- add diagnostics for path0, path1, winner, best, violations, graph mode, and accept/event distributions.

The implementation goal remains unchanged: prove `accept/event(2-spine) > E5` on the real agentic B=4 workload without new per-spine state-copy and without sacrificing speed.
