# FR9 Spine-2 Lossless Winner Closeout

**Date:** 2026-06-02
**Branch:** `fr9-spine2-lossless-winner`
**Verdict:** implementation complete for selector-off fail-closed public commit;
full losslessness proof is not complete. Gate B still fails exact-token equality
for the controlled `spines=1` versus `spines=2` probe, even though both arms are
batch-shape invariant under the required controls and the recent spine-2 trace
shows no hidden-spine public commits.

## What changed

- Deleted the launchable longest-accepted hidden-winner public commit path in
  `scripts/swe_x86_helpers/relaunch_qwen36_round.py`.
- Added explicit public commit policy validation:
  `LUMO_IR_PUBLIC_COMMIT_POLICY=lossless` is the default and only accepted
  public policy.
- Fail-closed before model launch for:
  `best_of_spines`, `unsafe_best_of_spines`, `deterministic_best`, unknown
  policy, hidden-publication request before selector implementation, selector
  enabled before implementation, and disabled independent-row winner commit.
- In the injected winner patch, public commit row is selected by
  `spine_id == 0`, not tensor row order.
- Hidden rows remain diagnostic only. The trace records `candidate_winner_*`,
  but `winner_spine` remains 0 and recurrent state copies from spine 0 to
  hidden sibling rows.
- Kept the Qwen parser/protocol guards from `argrepair7`.
- Added explicit policy fields to winner trace, independent winner summary, and
  agentic summary annotation.
- Added `scripts/fr9_gate_b_greedy_probe.py` to collect exact temp=0 token IDs
  from the live vLLM endpoint for fixed b1 and b4 prompt shapes and compare
  artifacts without hand-editing trace files.

## Evidence Collected

Focused tests:

```text
python3 -m pytest -q \
  tests/test_run_codex_experiment_spines.py \
  tests/test_independent_winner_commit_patch.py \
  tests/test_independent_winner_trace.py \
  tests/test_lossless_selector_gate_c_stub_design.py

36 passed in 0.40s
```

Compile/static checks:

```text
python3 -m py_compile \
  scripts/swe_x86_helpers/relaunch_qwen36_round.py \
  scripts/verify_independent_winner_trace.py \
  scripts/run_codex_experiment.py \
  scripts/fr9_gate_b_greedy_probe.py
git diff --check
```

Both passed.

Gate A fail-closed probes:

- `LUMO_IR_PUBLIC_COMMIT_POLICY=best_of_spines` failed before model launch.
- `LUMO_IR_PUBLIC_COMMIT_POLICY=longest_prefix` failed before model launch.
- `LUMO_IR_ALLOW_STOCHASTIC_HIDDEN_WINNER=1` failed before model launch.
- `LUMO_IR_LOSSLESS_SELECTOR_ENABLED=1` failed before model launch because the
  selector is not implemented.

Live relaunch:

```text
LUMO_GPU_MEMORY_UTILIZATION=0.88 \
python3 scripts/swe_x86_helpers/relaunch_qwen36_round.py \
  --config Fb --mtp 5 --row-mode independent --spines 2

READY config=Fb mtp=5 row_mode=independent spines=2 policy=lossless
```

The first 0.90-memory relaunch failed honestly before serving:

```text
Free memory on device cuda:0 (105.35/117.51 GiB) on startup is less than
desired GPU memory utilization (0.9, 105.76 GiB).
```

Live temp=0.6 probe appended 11 new winner events:

```json
{
  "rows": 11,
  "policies": {"lossless": 11},
  "lossless_public_stream_events": 11,
  "selector_enabled_events": 0,
  "winner_spines": {"0": 11},
  "winner_nonzero_spine_events": 0,
  "copy_missing_sum": 0,
  "superset_violations": 0
}
```

Live temp=0 probe appended 10 new winner events:

```json
{
  "rows": 10,
  "policies": {"lossless": 10},
  "lossless_public_stream_events": 10,
  "selector_enabled_events": 0,
  "winner_spines": {"0": 10},
  "winner_nonzero_spine_events": 0,
  "copy_missing_sum": 0,
  "superset_violations": 0
}
```

Gate C design stub, not a selector proof:

- The old hollow placeholder was renamed to
  `tests/test_lossless_selector_gate_c_stub_design.py`.
- It now characterizes the analytic max-order-statistic negative control from
  `/tmp/fr9_lossless_research.md`: the best-of-spines order statistic must fail
  versus target `p` and match the biased analytic distribution
  `p'(z)=p(z)*(2*CDF(z)-p(z))`.
- It still does not exercise a production multi-draft selector because no
  selector-on path is implemented on this branch.

Controlled Gate B probe after the batch-invariance directive:

```text
LUMO_BATCH_INVARIANT_VLLM=1 LUMO_GPU_MEMORY_UTILIZATION=0.88 \
python3 scripts/swe_x86_helpers/relaunch_qwen36_round.py \
  --config Fb --mtp 5 --row-mode independent --spines 1

READY config=Fb mtp=5 row_mode=independent spines=1 policy=lossless
```

The relaunch log for this arm reported `--attention-backend FLASH_ATTN` and
`max_num_seqs=4`. Collection:

```text
python3 scripts/fr9_gate_b_greedy_probe.py collect \
  --arm s1_batchinv_flashattn \
  --out /tmp/fr9_gate_b_s1_batchinv_flashattn.json \
  --wait-health 30

{"arm": "s1_batchinv_flashattn", "out": "/tmp/fr9_gate_b_s1_batchinv_flashattn.json", "records": 8, "reset_prefix_cache_error": null}

python3 scripts/fr9_gate_b_greedy_probe.py compare-batches \
  /tmp/fr9_gate_b_s1_batchinv_flashattn.json

"exact_match": true, "matched_records": 4
```

Then:

```text
LUMO_BATCH_INVARIANT_VLLM=1 LUMO_GPU_MEMORY_UTILIZATION=0.88 \
python3 scripts/swe_x86_helpers/relaunch_qwen36_round.py \
  --config Fb --mtp 5 --row-mode independent --spines 2

READY config=Fb mtp=5 row_mode=independent spines=2 policy=lossless
```

The relaunch log for this arm reported `--attention-backend FLASH_ATTN`,
`max_num_seqs=8`, and `LUMO_IR_PUBLIC_COMMIT_POLICY=lossless`. Collection:

```text
python3 scripts/fr9_gate_b_greedy_probe.py collect \
  --arm s2_batchinv_flashattn \
  --out /tmp/fr9_gate_b_s2_batchinv_flashattn.json \
  --wait-health 30

{"arm": "s2_batchinv_flashattn", "out": "/tmp/fr9_gate_b_s2_batchinv_flashattn.json", "records": 8, "reset_prefix_cache_error": null}

python3 scripts/fr9_gate_b_greedy_probe.py compare-batches \
  /tmp/fr9_gate_b_s2_batchinv_flashattn.json

"exact_match": true, "matched_records": 4
```

Recent spine-2 winner trace for that collection window:

```json
{
  "rows": 40,
  "policies": {"lossless": 40},
  "winner_spines": {"0": 40, "1": 0},
  "selector_enabled": 0,
  "non_lossless": 0,
  "suppressed": 2,
  "copy_missing": 0
}
```

Cross-arm exact comparison still failed:

```text
python3 scripts/fr9_gate_b_greedy_probe.py compare \
  /tmp/fr9_gate_b_s1_batchinv_flashattn.json \
  /tmp/fr9_gate_b_s2_batchinv_flashattn.json

"exact_match": false, "matched_records": 8
```

Mismatches were present at matched batch shapes (`b1` and `b4`) for two prompts:

```text
prompt 0: "Q: Count from one to five. A:"
  first_diff_index=18
  s1 token_ids=[271, 248068, 271, 248069, 271, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 13, 248044]
  s2 token_ids=[271, 248068, 271, 248069, 271, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 248044]

prompt 2: "Q: Write three lowercase letters in alphabetical order. A:"
  first_diff_index=2
  s1 token_ids=[271, 248068, 198, 8160, 579, 264, 7047, 1817, 25, 271, 16, 13, 220, 2972, 2014, 53983, 2570, 5396, 64700, 198, 256, 471, 2972, 14162]
  s2 token_ids=[271, 248068, 271, 248069, 271, 13290, 248044]
```

This result must not be described as the earlier b4-only batch-shape mismatch:
the same-arm b1-vs-b4 controls passed for both arms. It also must not be
described as hidden-winner publication: recent trace shows selector-off
spine-2 committed spine 0 for all 40 recent winner events. The honest current
status is that Gate B exact equality is still not passed under the controlled
spines1-versus-spines2 probe, and the remaining mismatch needs separate
investigation.

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| A policy fail-closed | Passed | Unit tests plus launch-time fail-closed probes |
| B greedy equality | Failed / incomplete | Same-arm b1-vs-b4 passed under `LUMO_BATCH_INVARIANT_VLLM=1` + FLASH_ATTN, and s2 trace committed spine 0 only; controlled s1-vs-s2 exact token comparison still failed on two prompts; target-only arm was not run |
| C synthetic selector convergence | Not passed | Only a clearly marked Gate C stub/design-power check exists; no selector-on implementation is exercised |
| D target-model sampling equivalence | Not run | No target-only versus selector-off distribution run collected |
| E SWE admission | Not run | No 16-task SWE run collected |

## Honesty Notes

- I do not claim full losslessness proof because Gate B failed the controlled
  s1-vs-s2 exact-token probe and Gates D/E were not run.
- I do not classify the earlier b4-only divergence as a spine-2 distribution
  bug; after batch-invariance controls, both arms were internally b1-vs-b4
  stable. The remaining mismatch is at matched batch shapes and with
  spine0-only public commits in the recent s2 trace.
- I do not claim a speed win and did not run kernel profiling.
- Selector-on is intentionally fail-closed on this branch; hidden recovery is
  trace-only until a real distribution-preserving multi-draft selector exists.
- Any future non-spine0 public commit must also implement GDN/linear-attention
  public recurrent-state recompute from the prior committed spine0 state; copying
  a sibling spine state across a divergent prefix remains forbidden.
