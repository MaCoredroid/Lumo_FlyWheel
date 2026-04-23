# verification_matrix - v1-clean-baseline

| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Oracle (full pass) | 100 | 1.0000 | 1.000 | 1.000 | 1200 | 0 | True | — |
| Empty (untouched bundle) | 0 | 0.0000 | 0.250 | 0.000 | 0 | 0 | False | no_visible_green, legacy_default_selected, proxy_route_incorrect, continuity_not_exact, config_rewritten, docs_unaligned |
| RAWR grounding_stripped | 25 | 0.2500 | 0.438 | 0.250 | 300 | 0 | False | no_visible_green, docs_unaligned |
| Pick-ceiling | 15 | 0.1500 | 0.362 | 0.150 | 180 | 0 | False | no_visible_green, legacy_default_selected, proxy_route_incorrect, continuity_not_exact, config_rewritten, docs_unaligned |
| Top1-wrong | 30 | 0.3000 | 0.475 | 0.300 | 360 | 0 | False | config_rewritten |
| Delete-tests adversarial | 0 | 0.0000 | 0.250 | 0.000 | 0 | 1 | False | — |
