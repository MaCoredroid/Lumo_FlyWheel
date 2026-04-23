# §5 verification matrix — v5-recovery-in-thread

| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Oracle | 100 | 1.0000 | 1.000 | 1.000 | 1110 | 0 | True | — |
| Empty | 14 | 0.1364 | 0.232 | 0.120 | 22 | 0 | False | legacy_reference_live,duplicate_automation_live,no_reuse_extension,incident_blind_reenable |
| Visible-only | 20 | 0.4091 | 0.645 | 0.300 | 56 | 0 | False | visible_only_bundle,legacy_reference_live,duplicate_automation_live,no_reuse_extension,incident_blind_reenable |
| Duplicate-live shortcut | 30 | 0.4091 | 0.645 | 0.180 | 36 | 0 | False | duplicate_automation_live,no_reuse_extension,incident_blind_reenable |
| Variant miss | 30 | 0.8636 | 0.918 | 0.420 | 79 | 0 | False | no_reuse_extension,incident_blind_reenable |
| Delete-tests adversarial | 0 | 0.0000 | 0.250 | -0.420 | -68 | 1 | False | — |
