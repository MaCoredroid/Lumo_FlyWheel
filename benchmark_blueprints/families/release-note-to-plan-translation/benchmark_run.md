# Benchmark Run — release-note-to-plan-translation

Family run protocol for CNB-55 Track 10, family `release-note-to-plan-translation`.

## Model under test

```bash
codex exec --model gpt-5.4 -c 'model_reasoning_effort="high"'
```

Whole-family live probes use the family-local runner at `benchmark_blueprints/families/release-note-to-plan-translation/tools/probe_family.py`, the scorer at `verifiers/release-note-to-plan-translation/score_ranking.py`, and `CNB55_SEED=42`.

## Variants and Layer A target

| Variant | Difficulty axis added | Live-probe target |
|---|---|---|
| v1-clean-baseline | clean release-note to execution-plan translation | ~25 |
| v2-noisy-distractor | stale release-note fragments and distractor notes | ~20 |
| v3-dirty-state | prior-session detritus and sunk-cost pressure | ~18 |
| v4-multi-corpus-objective | competing corpora and objective re-weighting | ~12 |
| v5-recovery-in-thread | incident-aware recovery and rollback-sensitive replanning | <=10 |

Layer A freeze gate follows the benchmark-family-authoring skill: family mean in `[15, 25]`, max variant mean `<= 40`, at least one variant mean `<= 10`, monotonic `V1 >= V2 >= V3 >= V4 >= V5` within `+/-3`, oracle `>= 90`, empty `= 0`, shortcut `<= 30`.

## Baseline sanity before live probe

- Oracle checks after family build: `100 / 97 / 100 / 100 / 100` on V1-V5.
- Empty brief checks: `0 / 0 / 0 / 0 / 0`.
- Shortcut checks: `35 / 30 / 30 / 30 / 30`.
- Verification matrix reruns completed for V1 and V5:
  - `verification_matrix.md`
  - `verification_matrix_v5-recovery-in-thread.md`
- Matrix observations:
  - Oracle rows stay in-range with `M_training ~= 1.0`.
  - Empty rows score `0`.
  - RAWR `grounding_stripped` stays capped at `25` on V1 and V5.
  - Pick-ceiling rows land at `35` on V1 and `30` on V5.
  - Delete-tests adversarial rows force integrity failure with `P_benchmark = 100`, `M_training = 0`, `pass = false`.

## Attempt history

- `attempt_00` — baseline family authoring and Layer B surface implementation.
  - Authored the five-variant workspace bundle, deterministic dual-band scorer, milestone scripts, family-local CLI, verifier-data oracles, manifests, and hidden tests.
  - Regenerated the family with `python3 benchmark_blueprints/families/release-note-to-plan-translation/tools/build_family.py`.
  - Confirmed visible oracle tests on all five variants with `pytest -q tests/test_plan_brief.py` in temp workspace copies: `6 passed` for each variant.
  - Result: Layer B artifacts implemented, but no whole-family live probe yet.

- `attempt_01_live_probe` — whole-family live verification with family-local probe/report flow.
  - Command:

    ```bash
    python3 benchmark_blueprints/families/release-note-to-plan-translation/tools/probe_family.py \
      --attempt attempt_01_live_probe \
      --n 3 \
      --timeout-seconds 240
    ```

  - Report regeneration:

    ```bash
    python3 benchmark_blueprints/families/release-note-to-plan-translation/tools/probe_report.py \
      --attempt-dir benchmark_blueprints/families/release-note-to-plan-translation/report/attempt_01_live_probe
    ```

  - Probe artifacts recorded under:
    - `benchmark_blueprints/families/release-note-to-plan-translation/report/attempt_01_live_probe/probe_runs.jsonl`
    - `benchmark_blueprints/families/release-note-to-plan-translation/report/attempt_01_live_probe/summary.json`
    - `benchmark_blueprints/families/release-note-to-plan-translation/report/attempt_01_live_probe/attempt_01_live_probe_probe_report.txt`
    - `benchmark_blueprints/families/release-note-to-plan-translation/report/attempt_01_live_probe/logs/`
    - `benchmark_blueprints/families/release-note-to-plan-translation/report/attempt_01_live_probe/artifacts/`

  | Variant | n | mean P | stdev P | mean M | stdev M | scores | ceilings hit |
  |---|---:|---:|---:|---:|---:|---|---|
  | v1-clean-baseline | 3 | 92.67 | 3.77 | 0.9252 | 0.0385 | [98, 90, 90] | none |
  | v2-noisy-distractor | 3 | 84.33 | 14.06 | 0.8402 | 0.1434 | [65, 98, 90] | none |
  | v3-dirty-state | 3 | 92.67 | 3.77 | 0.9252 | 0.0385 | [90, 90, 98] | none |
  | v4-multi-corpus-objective | 3 | 65.67 | 8.01 | 0.6496 | 0.0818 | [77, 60, 60] | none |
  | v5-recovery-in-thread | 3 | 94.33 | 3.77 | 0.9422 | 0.0385 | [97, 89, 97] | none |

  - Family-level probe summary:
    - `family_mean = 85.93`
    - `family_mean_M_training = 0.8565`
    - `current_observed_stdev_M_training = 0.1353`
    - `max_variant_mean = 94.33`
    - `min_variant_mean = 65.67`

  - Layer A verdict:
    - `[FAIL]` family mean in `[15, 25]`
    - `[FAIL]` max variant mean `<= 40`
    - `[FAIL]` at least one variant mean `<= 10`
    - `[FAIL]` monotonic `V1 >= V2 >= V3 >= V4 >= V5 +/-3`

  - Interpretation:
    - The family is not near the Layer A freeze window. Even the hardest current variant averages `65.67`.
    - The intended progression is not monotonic. `V5` is currently easier for the model than `V4`.
    - No live runs fired any variant-specific ceiling. The current scorer rewards high-quality grounded briefs, but the workspace variants are not yet forcing the documented failure modes under real `codex exec` behavior.
    - `V2` shows the largest variance (`M_training` stdev `0.1434`), so the family now legitimately crosses the variance-escalation threshold for follow-up hardening work.

## Explicit judgment

Layer A is **red** as of `attempt_01_live_probe`. The family has a usable whole-family probe/report loop and a green Layer B packaging surface, but it does not satisfy the skill's freeze-gate acceptance criteria yet. The honest next move is hardening the variants and/or ceilings so that at least one stress variant reliably collapses without introducing fake ambiguity.
