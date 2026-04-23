# Benchmark Run: `parallel-explorer-root-cause-map`

## Model Under Test

```
codex exec --model gpt-5.4 --reasoning-effort high
```

## Target Difficulty Shape

| Variant | Added difficulty axis | Probe target |
|---|---|---:|
| `v1-clean-baseline` | clean normalization-versus-renderer read | ~28 |
| `v2-noisy-distractor` | stale renderer-centric artifacts | ~24 |
| `v3-dirty-state` | sunk-cost renderer patch trap | ~20 |
| `v4-multi-corpus-objective` | hotfix-scope constraint against aggregation rewrite | ~16 |
| `v5-recovery-in-thread` | prior renderer rollback must be acknowledged | ~10 |

Family target:

- family mean in `[15, 25]`
- max variant mean `<= 40`
- at least one variant mean `<= 10`
- monotonic non-increasing V1 → V5 within `±3`

## attempt_00 — baseline family design

Design goals:

- make the true cause live in `normalize_fs_owner_alias()` feeding `merge_blocked_owner_rows()`
- keep `render_blocked_owner_section()` visibly noisy but causally downstream
- force evidence-backed disproof of stale docs and operator notes
- use a structured CLI brief so the scorer can grade exact symbols and exact artifact references deterministically
- keep remediation read-only and bounded: identify the smallest safe patch, not implement it

Seeded false leads:

- duplicated markdown headings
- stale `docs/triage-policy.md` guidance
- V3 abandoned renderer hotfix patch
- V4 temptation to dedupe inside aggregation
- V5 temptation to repeat a previously rolled-back renderer fix

## attempt_01 — family generation + deterministic sanity

Completed in this turn:

- authored full workspace bundles for V1–V5
- authored family CLI `bin/cnb55-brief`
- authored deterministic scorer with dual-band output and integrity checks
- generated oracle briefs, workspace manifests, gold data, milestone scripts, and manifest lock
- generated verification matrices for V1 and V5
- verified the seeded visible failure in V1 with `pytest tests/test_root_cause_map.py -q`

Observed deterministic baselines:

- oracle: `100 / 100` on all five variants
- empty: `0 / 100` on all five variants
- renderer-shortcut: `20 / 100` on all five variants

Verification matrix snapshots:

- V1 (`verification_matrix.md`):
  - Oracle `100`
  - Empty `0`
  - RAWR grounding stripped `30` via `grounding_stripped`
  - Pick renderer `20`
  - Top1 wrong `20`
  - Delete-tests adversarial `0` with integrity flag
- V5 (`verification_matrix_v5-recovery-in-thread.md`):
  - Oracle `100`
  - Empty `0`
  - RAWR grounding stripped `30` via `grounding_stripped`
  - Pick renderer `20`
  - Top1 wrong `20`
  - Delete-tests adversarial `0` with integrity flag

Visible workspace failure:

- `pytest tests/test_root_cause_map.py -q` fails with:
  - `assert summary["blocked_owner_total"] == 2, "scheduler aliases should collapse to the same blocked owner"`
  - observed `4 == 2`

Live probe status at this stage:

- one live V1 smoke run completed via `verifiers/parallel-explorer-root-cause-map/run_live_probe.py`
- observed V1 smoke result after rescoring with cache-safe integrity rules:
  - `codex_exit=0`
  - `codex_seconds=152`
  - `raw_score_pre_ceiling=89`
  - final `P_benchmark=35`
  - ceiling: `missing_contradictory_disproof`
  - misses: contradictory artifact disproof and explicit non-goals ruling out renderer + aggregation churn
- interpretation: the workspace and CLI are live-runnable, and the family is already producing a meaningful first-live miss; the **full live probe loop is still pending**
