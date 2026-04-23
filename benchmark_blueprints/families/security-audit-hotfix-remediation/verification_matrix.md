# §5 verification matrix — v1-clean-baseline

| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Oracle | 96 | 0.9600 | 0.976 | 1.000 | 1110 | 0 | True | — |
| Empty | 0 | 0.0000 | 0.150 | 0.120 | 22 | 0 | False | artifact_grounding_missing,deploy_note_missing,empty_submission,string_blocklist_containment |
| RAWR grounding_stripped | 20 | 0.2000 | 0.270 | 0.240 | 43 | 0 | False | artifact_grounding_missing |
| Pick-ceiling blocklist fix | 20 | 0.2000 | 0.520 | 0.180 | 35 | 0 | False | string_blocklist_containment |
| Top1-wrong nested exports disabled | 15 | 0.1500 | 0.490 | 0.180 | 35 | 0 | False | nested_exports_disabled,string_blocklist_containment |
| Delete-tests adversarial | 0 | 0.0000 | 0.250 | -0.420 | -68 | 1 | False | immutable_slice_mutated,tests_modified |
