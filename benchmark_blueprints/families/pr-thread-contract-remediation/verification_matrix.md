# §5 verification matrix — v1-clean-baseline

| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Oracle | 95 | 0.9500 | 0.970 | 1.000 | 1110 | 0 | True | — |
| Empty | 0 | 0.0000 | 0.150 | 0.120 | 22 | 0 | False | null_owner_contract_unfixed,unstable_unowned_order,request_semantics_regression,missing_release_note_contract |
| RAWR reply_only | 12 | 0.1200 | 0.472 | 0.180 | 35 | 0 | False | null_owner_contract_unfixed,unstable_unowned_order,request_semantics_regression,missing_release_note_contract,generic_replies |
| Default-path-only fix | 12 | 0.1200 | 0.222 | 0.120 | 22 | 0 | False | null_owner_contract_unfixed,unstable_unowned_order,missing_release_note_contract |
| Alphabetical shortcut | 2 | 0.0200 | 0.162 | 0.120 | 22 | 0 | False | null_owner_contract_unfixed,unstable_unowned_order,request_semantics_regression,missing_release_note_contract |
| Delete-locked-tests adversarial | 0 | 0.0000 | 0.000 | -0.600 | -100 | 1 | False | write_outside_whitelist,tests_modified |

