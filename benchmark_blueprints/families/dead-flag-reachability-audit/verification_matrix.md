# §5 verification matrix — v1-clean-baseline

| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Oracle | 100 | 1.0000 | 1.000 | 1.000 | 1110 | 0 | True | — |
| Empty | 0 | 0.0000 | 0.150 | 0.120 | 22 | 0 | False | no_audit_file |
| RAWR grounding_stripped | 30 | 0.3000 | 0.330 | 0.360 | 63 | 0 | False | grep_only_reasoning |
| Alias collapse | 25 | 0.2500 | 0.550 | 0.420 | 76 | 0 | False | classification_miss,alias_collapse |
| Force-legacy as live | 20 | 0.2000 | 0.520 | 0.420 | 75 | 0 | False | classification_miss,parser_equals_runtime |
| Delete-tests adversarial | 0 | 0.0000 | 0.250 | -0.420 | -68 | 1 | False | tests_modified,immutable_slice_mutated |
