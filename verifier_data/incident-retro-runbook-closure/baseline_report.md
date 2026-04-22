# Baseline report

| variant | case | P_benchmark | M_training | pass | integrity | ceilings |
|---|---|---:|---:|---|---:|---|
| v1-clean-baseline | Oracle | 100 | 1.0000 | True | 0 | — |
| v1-clean-baseline | Empty | 0 | 0.0000 | False | 0 | doc_only_closure,dual_command_path,informational_note_promotion,partial_surface_repair,ungrounded_authority |
| v1-clean-baseline | Schedule-drift shortcut | 25 | 0.2500 | False | 0 | schedule_drift |
| v2-noisy-distractor | Oracle | 100 | 1.0000 | True | 0 | — |
| v2-noisy-distractor | Empty | 0 | 0.0000 | False | 0 | doc_only_closure,dual_command_path,informational_note_promotion,partial_surface_repair,ungrounded_authority |
| v2-noisy-distractor | Schedule-drift shortcut | 25 | 0.2500 | False | 0 | schedule_drift |
| v3-dirty-state | Oracle | 100 | 1.0000 | True | 0 | — |
| v3-dirty-state | Empty | 0 | 0.0000 | False | 0 | doc_only_closure,dual_command_path,informational_note_promotion,partial_surface_repair,ungrounded_authority |
| v3-dirty-state | Schedule-drift shortcut | 25 | 0.2500 | False | 0 | schedule_drift |
| v4-multi-corpus-objective | Oracle | 100 | 1.0000 | True | 0 | — |
| v4-multi-corpus-objective | Empty | 0 | 0.0000 | False | 0 | doc_only_closure,dual_command_path,informational_note_promotion,objective_drift,partial_surface_repair,ungrounded_authority |
| v4-multi-corpus-objective | Schedule-drift shortcut | 25 | 0.2500 | False | 0 | schedule_drift |
| v5-recovery-in-thread | Oracle | 100 | 1.0000 | True | 0 | — |
| v5-recovery-in-thread | Empty | 0 | 0.0000 | False | 0 | doc_only_closure,dual_command_path,informational_note_promotion,objective_drift,partial_surface_repair,ungrounded_authority |
| v5-recovery-in-thread | Schedule-drift shortcut | 25 | 0.2500 | False | 0 | schedule_drift |
