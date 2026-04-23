# Verification matrix — v5-recovery-in-thread

| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Oracle | 100 | 1.0000 | 1.000 | 1.000 | 1110 | 0 | True | — |
| Empty | 0 | 0.0000 | 0.150 | 0.120 | 22 | 0 | False | required_artifacts_missing,legacy_primary_path,no_direct_verification,docs_harmonization_only |
| RAWR docs-only harmonization | 15 | 0.1500 | 0.240 | 0.240 | 42 | 0 | False | facts_unvalidated,no_direct_verification,docs_harmonization_only |
| Legacy primary shortcut | 0 | 0.0000 | 0.400 | 0.180 | 34 | 0 | False | deprecated_primary_path,facts_unvalidated |
| Delete-tests adversarial | 0 | 0.0000 | 0.250 | -0.420 | -68 | 1 | False | wrote_outside_whitelist,readonly_tree_mutated,tests_modified |
