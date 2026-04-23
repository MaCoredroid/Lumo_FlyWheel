# attempt_08_counted_live_probe live probe

- command: `python3 verifiers/release-manifest-v2-modernization/run_live_probe.py --attempt attempt_08_counted_live_probe --timeout-seconds 600 --runs 3 --model gpt-5.4 --reasoning-effort high --variants v1-clean-baseline v2-noisy-distractor v3-dirty-state v4-multi-corpus-objective v5-recovery-in-thread`
- family mean: `100`
- max variant mean: `100`
- min variant mean: `100`
- monotonicity within +/-3: `True`
- Layer A judgment: `not yet passed`

| variant | run | codex_exit | seconds | score | M_training | pass | integrity | ceilings | errors |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| v1-clean-baseline | 1 | 0 | 147.53 | 100 | 1.00 | True | 0 | — | — |
| v1-clean-baseline | 2 | 0 | 96.93 | 100 | 1.00 | True | 0 | — | — |
| v1-clean-baseline | 3 | 0 | 105.27 | 100 | 1.00 | True | 0 | — | — |
| v2-noisy-distractor | 1 | 0 | 97.94 | 100 | 1.00 | True | 0 | — | — |
| v2-noisy-distractor | 2 | 0 | 112.71 | 100 | 1.00 | True | 0 | — | — |
| v2-noisy-distractor | 3 | 0 | 124.5 | 100 | 1.00 | True | 0 | — | — |
| v3-dirty-state | 1 | 0 | 95.76 | 100 | 1.00 | True | 0 | — | — |
| v3-dirty-state | 2 | 0 | 107.95 | 100 | 1.00 | True | 0 | — | — |
| v3-dirty-state | 3 | 0 | 108.54 | 100 | 1.00 | True | 0 | — | — |
| v4-multi-corpus-objective | 1 | 0 | 107.77 | 100 | 1.00 | True | 0 | — | — |
| v4-multi-corpus-objective | 2 | 0 | 135.15 | 100 | 1.00 | True | 0 | — | — |
| v4-multi-corpus-objective | 3 | 0 | 122.73 | 100 | 1.00 | True | 0 | — | — |
| v5-recovery-in-thread | 1 | 0 | 148.46 | 100 | 1.00 | True | 0 | — | — |
| v5-recovery-in-thread | 2 | 0 | 137.07 | 100 | 1.00 | True | 0 | — | — |
| v5-recovery-in-thread | 3 | 0 | 139.09 | 100 | 1.00 | True | 0 | — | — |

## Per-Variant Means

| variant | mean | stdev | min | max | scores |
|---|---:|---:|---:|---:|---|
| v1-clean-baseline | 100.00 | 0.00 | 100 | 100 | [100, 100, 100] |
| v2-noisy-distractor | 100.00 | 0.00 | 100 | 100 | [100, 100, 100] |
| v3-dirty-state | 100.00 | 0.00 | 100 | 100 | [100, 100, 100] |
| v4-multi-corpus-objective | 100.00 | 0.00 | 100 | 100 | [100, 100, 100] |
| v5-recovery-in-thread | 100.00 | 0.00 | 100 | 100 | [100, 100, 100] |
