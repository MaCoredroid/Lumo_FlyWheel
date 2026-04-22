# Benchmark Run

## attempt_00 - baseline design

### Hypotheses

- V1 should be the floor check: a capable model should choose `schema-snapshot-alignment` and define a replay-divergence metric cleanly.
- V2 should only get harder if stale pager evidence can still pull the model off the true blocker.
- V3 should only get harder if the abandoned partial patch creates a real sunk-cost temptation.
- V4 should flip the preferred focus to `streaming-watermark-alignment` because the release gate changes the objective.
- V5 should step back from the rolled-back watermark plan and return to `schema-snapshot-alignment`, while explicitly fencing off retry work.

### Initial acceptance expectation

- Target live family mean near `20/100`.
- At least one hard variant should honestly fail.
- Oracle / empty / shortcut baselines should be healthy before any live probe.

## attempt_03 - whole-family live probe, pre-fix

### Commands

```bash
python3 benchmark_blueprints/families/multi-round-software-evolution/scripts/run_probe.py \
  --runs 1 \
  --json-out benchmark_blueprints/families/multi-round-software-evolution/report/attempt_03_probe_results.json \
  --report-out benchmark_blueprints/families/multi-round-software-evolution/report/attempt_03_probe_report.txt
```

### What happened

- The full five-variant live probe completed and wrote `report/attempt_03_probe_report.txt`.
- V2 and V3 scored `0` even though the agent chose the intended focus and produced structured plans.
- Focused diagnosis on V2 showed `integrity_flag=1` with `immutable_slice_mutated`, caused by cache files created during local validation inside readonly trees.

### Learning

- The failure was grader noise, not task difficulty.
- The trusted-final-state hash needed to ignore ephemeral validation artifacts (`.pytest_cache/`, `__pycache__/`, `.pyc`) inside readonly trees.

## attempt_04 - corrected whole-family live probe

### Commands

```bash
python3 benchmark_blueprints/families/multi-round-software-evolution/scripts/refresh_manifest_lock.py

python3 benchmark_blueprints/families/multi-round-software-evolution/scripts/run_probe.py \
  --runs 1 \
  --json-out benchmark_blueprints/families/multi-round-software-evolution/report/attempt_04_probe_results.json \
  --report-out benchmark_blueprints/families/multi-round-software-evolution/report/attempt_04_probe_report.txt
```

### Variants covered

- `v1-clean-baseline`
- `v2-noisy-distractor`
- `v3-dirty-state`
- `v4-multi-corpus-objective`
- `v5-recovery-in-thread`

### Per-variant results

| Variant | Selected focus | P_benchmark | M_training | Pass |
| --- | --- | ---: | ---: | --- |
| `v1-clean-baseline` | `schema-snapshot-alignment` | 86 | 0.9348 | `true` |
| `v2-noisy-distractor` | `schema-snapshot-alignment` | 86 | 0.9348 | `true` |
| `v3-dirty-state` | `schema-snapshot-alignment` | 86 | 0.9348 | `true` |
| `v4-multi-corpus-objective` | `streaming-watermark-alignment` | 86 | 0.9348 | `true` |
| `v5-recovery-in-thread` | `schema-snapshot-alignment` | 86 | 0.9348 | `true` |

### Layer A gate judgment

- Family mean `86.0`: FAIL for required `[15, 25]`
- Max variant mean `86.0`: FAIL for required `<= 40`
- Min variant mean `86.0`: FAIL for required `<= 10`
- Monotonic within `+/-3`: PASS, but non-discriminating because every variant landed at the same score
- Oracle / empty / shortcut reference remains healthy from regen: oracle `92`, empty `0`, shortcut `25`

### Verdict

`multi-round-software-evolution` is **not Layer A accepted**. The current family is dramatically too easy for the live model and fails the freeze gate honestly. The variants are semantically different, but the scorer does not yet create real separation between V1/V2/V3/V4/V5.

### Spot-check explanation

- The live agent chose the intended focus on every variant, including the V4 objective flip and the V5 rollback-aware recovery choice.
- That means the current dirty-state, stale-evidence, and recovery pressures are not strong enough to lower scores under the present rubric.
- The hardest current miss is not "model confusion"; it is the opposite: the model is clearing every variant cleanly.

### Next-attempt hardening hypothesis

- V2 needs a stronger stale-evidence ceiling that only clears when the plan explicitly names the stale screenshot as stale, not just when it picks the right focus.
- V3 needs a stronger sunk-cost check keyed to the abandoned partial patch, so the family can distinguish "mentions the patch" from "correctly rejects it as non-progress."
- V4 needs a second release-context requirement that forces the plan to operationalize the objective shift, not merely name the new focus.
- V5 needs a stricter recovery metric requirement tying the progress measure to rollback replay, so the family can penalize generic replay metrics that ignore the incident-specific retry gate.
