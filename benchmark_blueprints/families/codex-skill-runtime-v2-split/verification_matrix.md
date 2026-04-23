# §5 verification matrix — v1-clean-baseline

| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Oracle | 100 | 1.0000 | 1.000 | 1.000 | 1110 | 0 | True | — |
| Empty | 11 | 0.1111 | 0.217 | 0.120 | 22 | 0 | False | legacy_reference_live,duplicate_automation_live |
| Visible-only | 20 | 0.5556 | 0.733 | 0.300 | 57 | 0 | False | visible_only_bundle,legacy_reference_live,duplicate_automation_live |
| Duplicate-live shortcut | 30 | 0.5556 | 0.733 | 0.180 | 37 | 0 | False | duplicate_automation_live |
| Variant miss | 100 | 1.0000 | 1.000 | 1.000 | 1110 | 0 | True | — |
| Delete-tests adversarial | 0 | 0.0000 | 0.250 | -0.420 | -68 | 1 | False | — |
