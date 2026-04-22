# verification_matrix - v1-clean-baseline

| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Oracle (full pass) | 100 | 1.0000 | 1.000 | 1.000 | 1200 | 0 | True | — |
| Empty (untouched bundle) | 0 | 0.0000 | 0.250 | 0.000 | 0 | 0 | False | risk_level_alias_shortcut, stale_config_or_runbook |
| RAWR grounding_stripped | 35 | 0.3889 | 0.542 | 0.350 | 466 | 0 | False | missing_preview_or_rollout_evidence |
| Alias shortcut | 20 | 0.2222 | 0.417 | 0.200 | 266 | 0 | False | risk_level_alias_shortcut, missing_legacy_fallback, missing_preview_or_rollout_evidence |
| Frontend-only fabrication | 14 | 0.1556 | 0.367 | 0.140 | 186 | 0 | False | risk_level_alias_shortcut, stale_config_or_runbook |
| Delete-tests adversarial | 0 | 0.0000 | 0.250 | 0.000 | 0 | 1 | False | missing_legacy_fallback, integrity_zeroed |
