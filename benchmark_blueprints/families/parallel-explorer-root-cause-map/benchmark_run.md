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

## attempt_02 — real whole-family live probe (`probe_run_id=20260423T051922Z`)

Exact commands run:

```bash
python3 verifiers/parallel-explorer-root-cause-map/run_live_probe.py --probe-run-id 20260423T051922Z --runs 3 --timeout 900
python3 scripts/probe_report.py \
  benchmark_blueprints/families/parallel-explorer-root-cause-map/report/probe_runs.jsonl \
  --probe-run-id 20260423T051922Z \
  --emit-json \
  > benchmark_blueprints/families/parallel-explorer-root-cause-map/report/attempt_02_probe_report.txt
```

Family-local outputs:

- probe rows: `report/attempt_02_probe_runs.jsonl`
- aggregated report: `report/attempt_02_probe_report.txt`
- live logs: `report/live_probe_logs/20260423T051922Z-*.log`

Per-variant results:

| Variant | n | mean | stdev | min | max | scores | ceilings |
|---|---:|---:|---:|---:|---:|---|---|
| `v1-clean-baseline` | 3 | 35.00 | 0.00 | 35 | 35 | `[35,35,35]` | `missing_contradictory_disproof x3` |
| `v2-noisy-distractor` | 3 | 95.00 | 0.00 | 95 | 95 | `[95,95,95]` | `-` |
| `v3-dirty-state` | 3 | 35.00 | 0.00 | 35 | 35 | `[35,35,35]` | `missing_contradictory_disproof x3` |
| `v4-multi-corpus-objective` | 3 | 35.00 | 0.00 | 35 | 35 | `[35,35,35]` | `missing_contradictory_disproof x3` |
| `v5-recovery-in-thread` | 3 | 35.00 | 0.00 | 35 | 35 | `[35,35,35]` | `missing_contradictory_disproof x3` |

Layer A gate values:

- family mean: `47.00` -> FAIL (target `[15,25]`)
- max variant mean: `95.00` -> FAIL (target `<= 40`)
- min variant mean: `35.00` -> FAIL (target `<= 10`)
- monotonic V1>=V2>=V3>=V4>=V5 within ±3: FAIL
  - break: `v1-clean-baseline (35.0) < v2-noisy-distractor (95.0)`
- oracle baseline: `100 / 100 / 100 / 100 / 100` -> PASS
- empty baseline: `0 / 0 / 0 / 0 / 0` -> PASS
- shortcut baseline: `20 / 20 / 20 / 20 / 20` -> PASS

Spot-check diagnosis from actual live briefs:

- **V1**. The live brief in `/var/folders/xc/sy7ktq0n42d1n78zg10b8p_r0000gn/T/probe_v1-clean-baseline_7l9w3fx7/workspace/brief/manager_brief.md` correctly identifies `normalize_fs_owner_alias` as the primary cause, correctly rules the renderer downstream-only, quotes the exact failing assertion, and explicitly explains away `artifacts/logs/operator_note_2026_04_14.md` as weaker contradictory evidence. The score still caps at `35` because the scorer currently expects the contradictory artifact path to be exactly `docs/triage-policy.md` in V1. That is too narrow for the evidence the agent is actually using.
- **V2**. The live brief in `/var/folders/xc/sy7ktq0n42d1n78zg10b8p_r0000gn/T/probe_v2-noisy-distractor_6cpy_tji/workspace/brief/manager_brief.md` also identifies the correct root cause, but V2's contradictory artifact path in `gold_ranking.json` is already the operator note (`artifacts/logs/operator_note_2026_04_15.md`). That means V2 escapes the `missing_contradictory_disproof` ceiling entirely and lands at `95`, producing the V1<V2 inversion.

Interpretation:

- This attempt does **not** show the model missing the core root-cause task. It shows a **variant-design / scorer-alignment problem**:
  - V1 / V3 / V4 / V5 all collapse onto the same `35` cap because the contradictory-artifact acceptance logic is too path-specific.
  - V2 is much easier because its contradictory artifact is the exact operator note the model naturally chooses to disprove, so it clears the ceiling and jumps to `95`.
- That is a calibration defect, not an honest frontier-difficulty curve. The next hardening pass should fix the contradictory-artifact contract so each variant's disproof target is either unambiguous from the evidence or the scorer accepts the legitimate alternative artifact paths the task naturally supports.

Next-attempt hypothesis:

- Replace the exact-path contradictory-artifact check with a variant-local accepted set or a stronger symbol+resolution check, then re-probe.
- Keep the change family-local and singular; do not edit shared probe infra.
