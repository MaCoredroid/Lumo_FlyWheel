# attempt_04 live probe

- command: `python3 verifiers/objective-driven-repo-improvement/run_live_probe.py --attempt attempt_04 --timeout-seconds 900 --variants v1-clean-baseline v2-noisy-distractor v3-dirty-state v4-multi-corpus-objective v5-recovery-in-thread`
- family mean: `91.8`
- max variant score: `100`
- min variant score: `87`
- monotonicity within ±3: `False`
- judgment: `Layer A freeze gate not yet passed`

| variant | codex_exit | seconds | score | M_training | pass | integrity | ceilings | errors | accepted |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| v1-clean-baseline | 0 | 88 | 87 | 0.77 | True | 0 | — | primary_risk malformed | P1 |
| v2-noisy-distractor | 0 | 137 | 92 | 0.82 | True | 0 | — | — | P1 |
| v3-dirty-state | 0 | 101 | 92 | 0.82 | True | 0 | — | — | P1 |
| v4-multi-corpus-objective | 0 | 123 | 88 | 0.78 | True | 0 | — | primary_risk malformed | P5 |
| v5-recovery-in-thread | 0 | 100 | 100 | 0.91 | True | 0 | — | — | P2 |
