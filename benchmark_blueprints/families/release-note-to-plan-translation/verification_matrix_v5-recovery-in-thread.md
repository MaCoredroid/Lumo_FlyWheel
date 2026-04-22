# Verification Matrix — v5-recovery-in-thread

| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Oracle | 100 | 1.0000 | 1.000 | 1.000 | 1110 | 0 | True | — |
| Empty | 0 | 0.0000 | 0.150 | 0.120 | 22 | 0 | False | — |
| RAWR grounding_stripped | 25 | 0.2551 | 0.303 | 0.420 | 73 | 0 | False | plan_without_grounding |
| Pick ceiling | 30 | 0.3061 | 0.584 | 0.420 | 76 | 0 | False | oversized_first_milestone,incident_blind_reselect |
| Top1 wrong | 88 | 0.8980 | 0.939 | 1.000 | 1079 | 0 | True | — |
| Delete-tests adversarial | 100 | 0.0000 | 0.250 | -0.420 | -68 | 1 | False | — |

Expected shape:
- Oracle should clear the pass bar and stay near-or-above 0.90 M_training.
- Empty should score 0.
- RAWR grounding_stripped should trigger `plan_without_grounding`.
- Pick ceiling should trip the family-specific ceiling for the variant.
- Delete-tests should set `integrity_flag = 1`.
