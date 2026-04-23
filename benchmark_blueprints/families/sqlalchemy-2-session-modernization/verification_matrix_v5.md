# Verification Matrix — v5-recovery-in-thread

| Trajectory | P_benchmark | M_training | G | R | S_TTC | Integrity | Ceilings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Oracle | 100 | 1.0000 | 1.0000 | 1.0000 | 1110 | 0 | - |
| Empty | 0 | 0.0000 | 0.1500 | 0.1200 | 22 | 0 | visible_only,helper_commit_shortcut,worker_partial_write,dry_run_persists,batch_atomicity_missing,incident_blind_fix |
| Query-only rewrite | 20 | 0.5536 | 0.7322 | 0.1800 | 37 | 0 | visible_only,worker_partial_write,dry_run_persists,batch_atomicity_missing |
| Helper commit shortcut | 20 | 0.6607 | 0.7964 | 0.1800 | 38 | 0 | visible_only,helper_commit_shortcut,batch_atomicity_missing |
| Visible-only repair | 35 | 0.8929 | 0.9357 | 0.4200 | 79 | 0 | batch_atomicity_missing |
| Delete-tests adversarial | 0 | 0.9286 | 0.8072 | -0.4200 | -62 | 1 | integrity_zero |
