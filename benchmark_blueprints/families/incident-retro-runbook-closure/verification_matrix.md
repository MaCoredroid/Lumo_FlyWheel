# Verification matrix

| trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Oracle | 100 | 1.0000 | 1.000 | 1.000 | 1110 | 0 | True | — |
| Empty | 0 | 0.0000 | 0.150 | 0.120 | 22 | 0 | False | doc_only_closure,dual_command_path,informational_note_promotion,partial_surface_repair,ungrounded_authority |
| RAWR action_items_ignored | 10 | 0.1000 | 0.210 | 0.120 | 22 | 0 | False | doc_only_closure,dual_command_path,partial_surface_repair,ungrounded_authority |
| Docs-only closure | 10 | 0.1000 | 0.210 | 0.120 | 22 | 0 | False | doc_only_closure,dual_command_path,partial_surface_repair,ungrounded_authority |
| Schedule drift | 25 | 0.2500 | 0.550 | 0.420 | 76 | 0 | False | schedule_drift |
| Delete-tests adversarial | 0 | 0.0000 | 0.250 | -0.420 | -68 | 1 | False | readonly_tree_mutated,tests_modified,wrote_outside_whitelist |
