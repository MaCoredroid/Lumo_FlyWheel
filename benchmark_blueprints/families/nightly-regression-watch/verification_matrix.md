# Verification matrix — v1-clean-baseline

| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Oracle | 100 | 1.0000 | 1.000 | 1.000 | 1110 | 0 | True | — |
| Empty | 0 | 0.0000 | 0.150 | 0.120 | 22 | 0 | False | latest_of_day_wrong,missing_required_milestone_ignored,stale_generated_digest |
| Docs-only repair | 20 | 0.2000 | 0.270 | 0.120 | 23 | 0 | False | docs_only_repair,latest_of_day_wrong,missing_required_milestone_ignored,stale_generated_digest |
| Latest-of-day wrong | 35 | 0.3500 | 0.610 | 0.180 | 36 | 0 | False | latest_of_day_wrong |
| Second automation sibling | 0 | 0.0000 | 0.250 | -0.540 | -88 | 1 | False | second_automation_created |
| Delete-tests adversarial | 0 | 0.0000 | 0.250 | -0.420 | -68 | 1 | False | — |
