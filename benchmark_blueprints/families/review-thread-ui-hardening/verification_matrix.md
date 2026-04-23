# Verification matrix - v1-clean-baseline

| trajectory | P_benchmark | M_training | integrity | pass | ceilings |
|---|---:|---:|---:|---|---|
| Oracle | 100 | 1.0000 | 0 | True | — |
| Empty | 0 | 0.0000 | 0 | False | no_brief_file |
| Artifact-only | 10 | 0.1111 | 0 | False | artifact_only,blanket_accessibility_labeling,clipping_fix,missing_runtime_or_config_fix,wrong_viewport_mapping |
| Wrong viewport | 30 | 0.3333 | 0 | False | wrong_viewport_mapping |
| Wrong thread | 25 | 0.2778 | 0 | False | wrong_thread_mapping |
| Delete-tests adversarial | 0 | 0.0000 | 1 | False | readonly_tree:repo/tests,tests_modified |
