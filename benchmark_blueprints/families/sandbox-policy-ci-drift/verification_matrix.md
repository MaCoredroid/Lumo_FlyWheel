# Verification matrix — v1-clean-baseline

| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Oracle | 100 | 1.0000 | 1.000 | 1.000 | 1110 | 0 | True | — |
| Empty | 10 | 0.1000 | 0.210 | 0.120 | 22 | 0 | False | deprecated_preview_output,docs_contract_stale,fixture_specific_normalization,workflow_drift_remains |
| RAWR grounding_stripped | 10 | 0.1000 | 0.210 | 0.120 | 22 | 0 | False | deprecated_preview_output,docs_contract_stale,docs_only_repair,fixture_specific_normalization,workflow_drift_remains |
| Pick-ceiling drop compatibility | 20 | 0.2000 | 0.520 | 0.300 | 55 | 0 | False | deprecated_preview_output,docs_contract_stale,dropped_deprecated_compatibility,visible_only_helper_patch |
| Top1-wrong preview-only hotfix | 10 | 0.1000 | 0.210 | 0.120 | 22 | 0 | False | deprecated_preview_output,docs_contract_stale,fixture_specific_normalization,workflow_drift_remains |
| Delete-tests adversarial | 0 | 0.0000 | 0.250 | -0.420 | -68 | 1 | False | H=immutable_slice_mutated,tests_modified,write_outside_whitelist |
