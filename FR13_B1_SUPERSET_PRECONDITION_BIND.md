# FR13 B=1 Superset Precondition Bind

Date: 2026-06-11 UTC

## Verdict

The current B=1 cat9 caterpillar fails the strong native-spine/superset
precondition needed to derive `tree accept/event >= native MTP-5 accept/event`.

The native MTP-5 spine is structurally present in the cat9 tree, and the first
prompt's first event has byte-identical spine draft tokens versus chain5. That
is not enough. On the same served prefix, before the cat9 and chain5 output
streams fork, cat9 stops preserving the chain/native spine proposal opportunity:
a chain5 five-token spine accept becomes multiple cat9 events, including a root
reject, so the event denominator grows and accepted tokens are lost before any
served-output divergence can explain it.

Bound precondition:

- Structural inclusion: **true** for cat9, spine node ids `[0,1,3,5,7]`.
- Current S1 internal tree superset checks: **clean**, but only relative to
  cat9's own degraded spine/path diagnostics.
- Strong native-spine inclusion/state preservation: **failed or still
  unproven**, and empirically false at the observable proposal-token level after
  a partial cat9 accept.
- Therefore the `2.1515` cat9 accept/event versus `3.1613` native MTP-5 result
  is a theorem-precondition failure, not an accept/event tuning target.

No B=4 command was run or modified for this bind.

## Surfaces Kept Separate

| surface | role in this bind |
|---|---|
| Ground-truth target decode / no MTP | Conceptual lossless target. Not rerun here and not used as the accept/event comparator. |
| Native MTP-5 single spine | Deployed B=1 reference: `3.1613` accept/event from `FR13_B1_CURRENT_GATE_BIND.md`. |
| Tree chain5 | Tree backend with only the native-length spine: `3.2562` accept/event from `FR13_B1_CHAIN_SPEED_DISCRIMINATOR.md`. This proves the tree backend can exceed native accept/event when the geometry is a linear chain, but it is not a full lossless proof. |
| Tree cat9 caterpillar | Current 9-node tree under investigation: `2.1515` accept/event, S1-clean internally, below native and chain5. |

## Existing Gate Numbers Reconfirmed

These numbers were taken from the already-bound B=1 docs and artifacts rather
than rerunning the full gates:

- `FR13_B1_CURRENT_GATE_BIND.md`
  - cat9: `2.151515` accept/event, `165` events, `1485` draft tokens.
  - native MTP-5: `3.161290` accept/event, `124` events, `620` draft tokens.
  - S1 clean: `bonus_violations_count=0`, `superset_violations_count=0`.
- `FR13_B1_CHAIN_SPEED_DISCRIMINATOR.md`
  - chain5: `3.256198` accept/event, `121` events, `605` draft tokens.
  - chain5 is above native accept/event while cat9 is below, isolating the
    failure to branch/tree-geometry proposal or commit/state semantics rather
    than to native-width tree serving in general.

## Trace Artifact

Local artifact:
`output/fr13_b1_superset_precondition_bind/cat9_chain5_trace_bind.json`

Inputs recorded inside the artifact:

- cat trace:
  `output/fr13_b1_current_gate/tree/logs/tree_path_lcp.jsonl`
- chain trace:
  `output/fr13_b1_chain_speed_discriminator/chain/logs/tree_path_lcp.jsonl`
- cat probes:
  `tree_warmup_probe.json`, `tree_greedy_probe.json`,
  `tree_greedy_rep2_probe.json`
- chain probes:
  `chain_warmup_probe.json`, `chain_greedy_probe.json`,
  `chain_greedy_rep2_probe.json`

Inspection commands used from repo root:

```bash
jq '.first_event_spine_identity_prompt0' \
  output/fr13_b1_superset_precondition_bind/cat9_chain5_trace_bind.json

jq '.main_prompt_pairs[] | {
  prompt_id,
  first_output_diff: .cat_vs_chain_first_output_diff_completion_pos,
  equal_event_stream_prefix_tokens,
  pre_fork_cat_event_count,
  pre_fork_chain_event_count,
  pre_fork_cat_accepted_total,
  pre_fork_chain_accepted_total
}' output/fr13_b1_superset_precondition_bind/cat9_chain5_trace_bind.json

jq '.main_prompt_pairs[0].first_split_examples[0]' \
  output/fr13_b1_superset_precondition_bind/cat9_chain5_trace_bind.json
```

First event identity on prompt 0:

- cat global event `7`, chain global event `12`.
- cat spine `[0,1,3,5,7]`, chain spine `[0,1,2,3,4]`.
- spine draft tokens are byte-identical:
  `[550,37028,271,16,13]`.

Pre-fork counters across the four measured main prompts:

| prompt | first cat/chain output diff | equal event-stream prefix | cat events before diff | chain events before diff | cat accepted before diff | chain accepted before diff |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 54 | 53 | 18 | 13 | 37 | 42 |
| 1 | 27 | 26 | 10 | 7 | 16 | 19 |
| 2 | 21 | 20 | 7 | 6 | 16 | 16 |
| 3 | 61 | 60 | 20 | 14 | 41 | 49 |

This binds the denominator problem to real behavior before stream divergence:
cat9 spends more events on the same served prefix and usually accepts fewer
tokens over that shared prefix.

## First Concrete Lost Opportunity

Prompt 0, served stream range `[5,11]`, before the first output fork:

Chain5 has a single full-spine accept:

- chain global event `14`
- parent targets: `[3172,1151,539,23218,279]`
- spine draft: `[3172,1151,539,23218,279]`
- `chain_spine_lcp=5`
- emitted segment: `[3172,1151,539,23218,279,26622]`
- `chain_accepted_len=5`

Cat9 splits the same served segment into two events:

- cat global event `9`
  - previous cat event accepted `[0,1,3]` with `accepted_len=3`.
  - parent targets on spine: `[3172,1151,279,3274,3425]`
  - spine draft: `[668,1301,279,2450,3425]`
  - `cat_spine_lcp=0`
  - winner path `[0,2]`
  - emitted segment `[3172]`
- cat global event `10`
  - parent targets on spine: `[1151,539,23218,279,26622]`
  - spine draft: `[1151,539,23218,279,9468]`
  - `cat_spine_lcp=4`
  - winner path `[0,1,3,5,7]`
  - emitted segment `[1151,539,23218,279,26622]`

The decisive failure is not that cat9 lacks a five-node spine. It is that after
a prior partial cat9 accept, the next event's spine proposal is no longer the
chain/native proposal at the same served prefix. The existing S1
`superset_violations=0` result only says cat9's chosen path is at least as good
as cat9's own true-spine diagnostic for that event; it does not prove cat9's
true spine remained native-equivalent.

## Forced-Spine Diagnostic

I ran one short B=1 diagnostic with `FR13_FORCE_SPINE_COMMIT=1` on cat9. This
is explicitly diagnostic-only and was not treated as a pass condition.

Launch command:

```bash
CONTAINER=fr13-b1-forced-spine \
PORT=9960 \
GPU_UTIL=0.82 \
MAX_NUM_SEQS=1 \
BATCH_INVARIANT=0 \
FR13_BI_TREE_ATTN=0 \
FR10_METRICS=0 \
FR13_REPLAY_ROUTE=1 \
FR13_FORCE_SPINE_COMMIT=1 \
FR13_RUN_DIR=/home/mark/shared/lumoFlyWheel/output/fr13_b1_superset_precondition_bind/forced_spine \
LOG_DIR=/home/mark/shared/lumoFlyWheel/output/fr13_b1_superset_precondition_bind/forced_spine/logs \
scripts/fr13_launch_forked_fa2_tree_server.sh
```

Probe command:

```bash
python3 scripts/fr10_quick_decode_tps_probe.py \
  --endpoint http://127.0.0.1:9960 \
  --prompts-file output/fr13_acceptance_ladder/prompts_swe4.json \
  --prompt-limit 1 \
  --out output/fr13_b1_superset_precondition_bind/forced_spine/forced_spine_probe.json \
  --modes tree_mtp \
  --samples-per-prompt 1 \
  --batch-size 1 \
  --max-tokens 64 \
  --temperature 0.0 \
  --top-p 1.0 \
  --seed 1313 \
  --wait-health 900 \
  --warmup-samples 1 \
  --request-metrics-out output/fr13_b1_superset_precondition_bind/forced_spine/forced_spine_request_metrics.jsonl \
  --require-tree-engagement \
  --tree-sampler-debug-log output/fr13_b1_superset_precondition_bind/forced_spine/logs/tree_sampler_debug.jsonl \
  --tree-accept-log output/fr13_b1_superset_precondition_bind/forced_spine/logs/tree_path_lcp.jsonl \
  --expected-draft-count 9
```

Teardown:

```bash
docker rm -f fr13-b1-forced-spine
docker ps --format '{{.Names}}'
```

Artifacts:

- `output/fr13_b1_superset_precondition_bind/forced_spine/forced_spine_probe.json`
- `output/fr13_b1_superset_precondition_bind/forced_spine/forced_spine_request_metrics.jsonl`
- `output/fr13_b1_superset_precondition_bind/forced_spine/logs/tree_path_lcp.jsonl`
- `output/fr13_b1_superset_precondition_bind/forced_spine/logs/fr10_mtp_draft_trace.jsonl`
- `output/fr13_b1_superset_precondition_bind/forced_spine/logs/tree_sampler_debug.jsonl`
- `output/fr13_b1_superset_precondition_bind/forced_spine/container_env.txt`
- `output/fr13_b1_superset_precondition_bind/forced_spine/docker_full.log`

Result:

- `accepted_per_draft_event=2.3157894736842106`
- `spec_accepted_tokens=44`
- `spec_drafts=19`
- `spec_draft_tokens=171`
- `returned_tokens=64`
- `warm_decode_tps=10.743829399441998`

Trace head, using known cat9 spine nodes `[0,1,3,5,7]`:

```bash
jq -s '[.[] | select(.event == "tree_path_lcp_max")][0:4] | map({
  accepted_len,
  path0_lcp,
  winner_path,
  accepted_node_ids,
  emitted_tokens,
  forced_spine_commit,
  spine_draft: [.draft_token_ids[0], .draft_token_ids[1],
                .draft_token_ids[3], .draft_token_ids[5],
                .draft_token_ids[7]],
  parent_targets_on_spine: [.parent_target_ids[0], .parent_target_ids[1],
                            .parent_target_ids[3], .parent_target_ids[5],
                            .parent_target_ids[7]]
})' output/fr13_b1_superset_precondition_bind/forced_spine/logs/tree_path_lcp.jsonl
```

The forced-spine run still hits the same root class:

- event 0 forced winner path `[0,1,3,5,7]`, spine draft
  `[550,37028,271,16,13]`.
- event 1 forced winner path `[0,1,3,5,7]`, `accepted_len=3`, emitted
  `[271,248069,271,40]`.
- event 2 forced winner path `[0,1,3,5,7]`, but spine draft
  `[668,1151,539,5122,279]` against parent targets
  `[3172,1151,539,23218,279]`, so `accepted_len=0` and emitted `[3172]`.

Forcing the committer to choose the true spine does not restore native/chain
spine proposals. It also diverges from the clean chain/native prompt-0 token
stream earlier, at completion position `35` (`5454` versus chain/native
`44675`). Branch winner selection alone is therefore not the fix.

## Code Surfaces Inspected

Search commands:

```bash
rg -n "FR13_FORCE_SPINE_COMMIT|spine_path_idx|accepted_paths|accepted_lens" \
  scripts/fr10_phase4_patch_vllm_tree_gdn.py

rg -n "gather_committed_path_conv_prior|_tree_gdn_replay_kernel|prev_lens|accepted_paths" \
  src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py

rg -n "FORCE_SPINE|committed_path|accepted_paths|accepted_lens" \
  tests scripts src
```

Relevant behavior:

- `scripts/fr10_phase4_patch_vllm_tree_gdn.py` computes the true
  `spine_path_idx` and has a diagnostic-only `FR13_FORCE_SPINE_COMMIT` override
  for the greedy committer. The forced run proves that selecting the true spine
  path is not sufficient.
- The replay route publishes `accepted_paths` and `accepted_lens` as GDN node
  ids/lens for replay.
- `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` has committed-path conv
  prior logic that reads `accepted_paths[b, len - 1]`, which is the right shape
  for a non-linear cat9 spine.
- The replay kernel still has a state-handoff surface worth checking: its
  initial h0 reads from a linearized prior column derived from `prev_lens`.
  That may be correct if replay has already linear-published the prior accepted
  path state, but the observed failure happens exactly after a cat9 partial
  spine accept plus bonus row. This is a candidate discriminator, not yet a
  proven bug.

## Next Concrete Discriminator

Add a B=1-only spine-state parity reducer for the cat9-vs-chain prompt-0 point
above. It should compare identical served prefixes immediately before the lost
opportunity:

- chain event `14`, stream start `5`, accepted path `[0,1,2,3,4]`.
- cat event `8`, then cat event `9`, stream start `5`, prior accepted path
  `[0,1,3]`, next winner `[0,2]`.

Capture and compare:

- accepted GDN path ids and `accepted_lens` published by the committer.
- replay `prev_lens` and the replay destination columns.
- conv prior read row/column and SSM h0 row/column for the next event.
- next-event drafter root/spine proposal tokens and, if cheap, logits for the
  root token `3172` versus the cat9 proposed root `668`.

Decision rule:

- If the cat9 post-event-8 state equals the chain/native state for the same
  served prefix but the next proposal still differs, localize to MTP drafter
  topology/position mapping.
- If the state differs, fix the residual replay/linear-publication/bonus-state
  handoff before looking at accept/event again.

Do not optimize accept/event directly. Treat accept/event only as the theorem
consequence check once the strong native-spine inclusion and state-preservation
preconditions are proven.
