# FR9 Spine-2 Lossless Winner Closeout

**Date:** 2026-06-02
**Branch:** `fr9-spine2-lossless-winner`
**Verdict:** implementation complete for selector-off lossless public commit;
full losslessness proof is incomplete because target-only Gate B/D comparisons
and SWE Gate E were not run.

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

## Evidence Collected

Focused tests:

```text
python3 -m pytest -q \
  tests/test_run_codex_experiment_spines.py \
  tests/test_independent_winner_commit_patch.py \
  tests/test_independent_winner_trace.py \
  tests/test_lossless_selector_gate_c.py

36 passed in 0.35s
```

Compile/static checks:

```text
python3 -m py_compile \
  scripts/swe_x86_helpers/relaunch_qwen36_round.py \
  scripts/verify_independent_winner_trace.py \
  scripts/run_codex_experiment.py
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

Synthetic Gate C negative control:

- Target sampler converged on a known two-token distribution.
- Deliberately naive longest-accepted hidden winner failed the distribution
  check by forcing the hidden token with frequency >0.98 against target
  probability 0.30.

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| A policy fail-closed | Passed | Unit tests plus launch-time fail-closed probes |
| B greedy equality | Partial only | Live temp=0 spine-2 selector-off commits spine 0; exact target-only token equality was not run |
| C synthetic selector convergence | Partial | Negative control catches naive longest winner; no selector-on implementation exists |
| D target-model sampling equivalence | Not run | No target-only versus selector-off distribution run collected |
| E SWE admission | Not run | No 16-task SWE run collected |

## Honesty Notes

- I do not claim full losslessness proof because Gates B, D, and E are not fully
  passed.
- I do not claim a speed win and did not run kernel profiling.
- Selector-on is intentionally fail-closed on this branch; hidden recovery is
  trace-only until a real distribution-preserving multi-draft selector exists.
