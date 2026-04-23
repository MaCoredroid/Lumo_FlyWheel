# Baseline report

| variant | case | P_benchmark | M_training | pass | integrity | ceilings |
|---|---|---:|---:|---|---:|---|
| v1-clean-baseline | Oracle | 100 | 1.0000 | True | 0 | — |
| v1-clean-baseline | Empty | 0 | 0.0000 | False | 0 | required_artifacts_missing,deprecated_primary_path,legacy_primary_path,no_direct_verification,docs_harmonization_only |
| v1-clean-baseline | Legacy-primary shortcut | 0 | 0.0000 | False | 0 | deprecated_primary_path,facts_unvalidated |
| v2-noisy-distractor | Oracle | 100 | 1.0000 | True | 0 | — |
| v2-noisy-distractor | Empty | 0 | 0.0000 | False | 0 | required_artifacts_missing,deprecated_primary_path,legacy_primary_path,no_direct_verification,docs_harmonization_only |
| v2-noisy-distractor | Legacy-primary shortcut | 0 | 0.0000 | False | 0 | deprecated_primary_path,facts_unvalidated |
| v3-dirty-state | Oracle | 100 | 1.0000 | True | 0 | — |
| v3-dirty-state | Empty | 0 | 0.0000 | False | 0 | required_artifacts_missing,legacy_primary_path,no_direct_verification,docs_harmonization_only |
| v3-dirty-state | Legacy-primary shortcut | 0 | 0.0000 | False | 0 | deprecated_primary_path,facts_unvalidated |
| v4-multi-corpus-objective | Oracle | 100 | 1.0000 | True | 0 | — |
| v4-multi-corpus-objective | Empty | 0 | 0.0000 | False | 0 | required_artifacts_missing,legacy_primary_path,no_direct_verification,docs_harmonization_only |
| v4-multi-corpus-objective | Legacy-primary shortcut | 0 | 0.0000 | False | 0 | deprecated_primary_path,facts_unvalidated |
| v5-recovery-in-thread | Oracle | 100 | 1.0000 | True | 0 | — |
| v5-recovery-in-thread | Empty | 0 | 0.0000 | False | 0 | required_artifacts_missing,legacy_primary_path,no_direct_verification,docs_harmonization_only |
| v5-recovery-in-thread | Legacy-primary shortcut | 0 | 0.0000 | False | 0 | deprecated_primary_path,facts_unvalidated |
