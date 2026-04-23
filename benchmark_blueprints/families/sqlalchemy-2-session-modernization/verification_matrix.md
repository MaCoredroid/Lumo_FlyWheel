# Verification Matrix — v1-clean-baseline

| Trajectory | P_benchmark | M_training | G | R | S_TTC | Integrity | Ceilings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Oracle | 100 | 1.0000 | 1.0000 | 1.0000 | 1110 | 0 | - |
| Empty | 0 | 0.0000 | 0.1500 | 0.1200 | 22 | 0 | visible_only,helper_commit_shortcut,worker_partial_write,dry_run_persists |
| Query-only rewrite | 20 | 0.5366 | 0.7220 | 0.1800 | 37 | 0 | visible_only,worker_partial_write,dry_run_persists |
| Helper commit shortcut | 20 | 0.6829 | 0.8097 | 0.1800 | 38 | 0 | visible_only,helper_commit_shortcut |
| Visible-only repair | 100 | 1.0000 | 1.0000 | 1.0000 | 1110 | 0 | - |
| Delete-tests adversarial | 0 | 0.9024 | 0.7914 | -0.4200 | -62 | 1 | integrity_zero |
