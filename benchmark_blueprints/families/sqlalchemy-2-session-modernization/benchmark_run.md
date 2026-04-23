# Benchmark Run

## attempt_00 — baseline design

- Original family existed only as a single loose workspace plus shallow docs.
- No five-variant ladder, no family-local scorer, no verifier data, no
  manifests, no Layer-B declaration.
- Hypothesis: visible worker/admin failures alone would not give an honest
  `~20/100` signal because the bundle did not yet encode batch, context, or
  integrity pressure.

## attempt_01 — pre-rebuild hardening snapshot

- Earlier family-local notes already documented a hardened visible/hidden split
  that targeted `20/100` for a shallow GPT-5.4/high solver.
- That work remained partial because the shipped family layout still lacked the
  CNB-55 five-variant bundle and Layer-B artifacts.

## attempt_02 — family rebuild for CNB-55 shape

- Replaced the single runnable bundle with five explicit variants:
  `v1-clean-baseline` through `v5-recovery-in-thread`.
- Added immutable context files for stale-shim noise, dirty local edits,
  release-objective drift, and rollback incident recovery.
- Added family-local scorer, hidden tests, oracle file set, milestone scripts,
  and variant-local verifier-data directories.
- Rewrote `task_spec.md` and `evaluator_contract.md` to match the new family
  shape and narrow write allowlist.

## attempt_03 — regeneration / acceptance status

- Ran `scripts/regen_family.py` end to end.
- Generated per-variant `workspace_manifest.json`, `gold_fix.json`, and
  `manifest.lock.json`.
- Observed baseline scores on all five variants:
  - oracle: `100`
  - untouched / empty: `0`
  - query-only shortcut: `20`
- Generated verification matrices:
  - `verification_matrix.md` for `v1-clean-baseline`
  - `verification_matrix_v5.md` for `v5-recovery-in-thread`
- Stress-matrix spot check:
  - V5 `Visible-only repair` lands at `35` with `batch_atomicity_missing`
    still firing, which is the intended shape for the batch / incident stress
    row.
- Live probe status: pending.
- Next gate: run a real `codex exec` family probe and decide whether Layer A is
  already honest enough to keep or still needs another hardening pass.

## attempt_04 — real whole-family live probe / calibration

- Counted live probe command:

```bash
python3 benchmark_blueprints/families/sqlalchemy-2-session-modernization/scripts/probe_family.py --attempt attempt_04_live_probe --n 3 --timeout-seconds 1200
```

- Underlying per-run `codex exec` command recorded in
  `report/attempt_04_live_probe/metadata.json`:

```bash
timeout 1200 codex exec --cd <workspace> --skip-git-repo-check --sandbox workspace-write --color never --ephemeral --model gpt-5.4 -c 'model_reasoning_effort="high"' -o <last_message_file> 'Read AGENTS.md in this directory and follow it exactly. Inspect the current code and docs before editing. Fix the SQLAlchemy 2 modernization task in this workspace. Keep edits within the allowed files from AGENTS.md. Run `pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py` before you finish. Do not modify tests, seed data, or contextual note files.'
```

- Report-generation command:

```bash
python3 benchmark_blueprints/families/sqlalchemy-2-session-modernization/scripts/probe_report.py --attempt-dir benchmark_blueprints/families/sqlalchemy-2-session-modernization/report/attempt_04_live_probe
```

- Verification matrix commands:

```bash
python3 benchmark_blueprints/families/sqlalchemy-2-session-modernization/scripts/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/sqlalchemy-2-session-modernization/verification_matrix.md
python3 benchmark_blueprints/families/sqlalchemy-2-session-modernization/scripts/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/sqlalchemy-2-session-modernization/verification_matrix_v5.md
```

### Per-variant live results

| Variant | n | mean P | stdev P | mean M | stdev M | min | max | scores | ceilings | integrity hits |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| v1-clean-baseline | 3 | 88.00 | 0.00 | 0.9024 | 0.0000 | 88 | 88 | [88, 88, 88] | - | 0 |
| v2-noisy-distractor | 3 | 92.67 | 5.19 | 0.9407 | 0.0419 | 89 | 100 | [89, 89, 100] | - | 0 |
| v3-dirty-state | 3 | 92.67 | 5.19 | 0.9407 | 0.0419 | 89 | 100 | [89, 89, 100] | - | 0 |
| v4-multi-corpus-objective | 3 | 100.00 | 0.00 | 1.0000 | 0.0000 | 100 | 100 | [100, 100, 100] | - | 0 |
| v5-recovery-in-thread | 3 | 56.67 | 30.64 | 0.8929 | 0.0758 | 35 | 100 | [100, 35, 35] | incident_blind_fix | 0 |

### Layer A gate values

| Gate | Value | Result |
|---|---:|---|
| family_mean in [15, 25] | 86.00 | FAIL |
| max variant mean <= 40 | 100.00 | FAIL |
| at least one variant mean <= 10 | 56.67 | FAIL |
| monotonic V1 >= V2 >= V3 >= V4 >= V5 within +/-3 | n/a | FAIL |
| oracle >= 90 | 100 | PASS |
| empty = 0 | 0 | PASS |
| shortcut <= 30 | 20 | PASS |

Layer B probe metadata from the same counted run:

- family mean `M_training`: `0.9353`
- current observed stdev `M_training`: `0.0572`

### Verification matrix outputs

V1 clean baseline:

| Trajectory | P_benchmark | M_training | G | R | S_TTC | Integrity | Ceilings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Oracle | 100 | 1.0000 | 1.0000 | 1.0000 | 1110 | 0 | - |
| Empty | 0 | 0.0000 | 0.1500 | 0.1200 | 22 | 0 | visible_only,helper_commit_shortcut,worker_partial_write,dry_run_persists |
| Query-only rewrite | 20 | 0.5366 | 0.7220 | 0.1800 | 37 | 0 | visible_only,worker_partial_write,dry_run_persists |
| Helper commit shortcut | 20 | 0.6829 | 0.8097 | 0.1800 | 38 | 0 | visible_only,helper_commit_shortcut |
| Visible-only repair | 100 | 1.0000 | 1.0000 | 1.0000 | 1110 | 0 | - |
| Delete-tests adversarial | 0 | 0.9024 | 0.7914 | -0.4200 | -62 | 1 | integrity_zero |

V5 recovery in thread:

| Trajectory | P_benchmark | M_training | G | R | S_TTC | Integrity | Ceilings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Oracle | 100 | 1.0000 | 1.0000 | 1.0000 | 1110 | 0 | - |
| Empty | 0 | 0.0000 | 0.1500 | 0.1200 | 22 | 0 | visible_only,helper_commit_shortcut,worker_partial_write,dry_run_persists,batch_atomicity_missing,incident_blind_fix |
| Query-only rewrite | 20 | 0.5536 | 0.7322 | 0.1800 | 37 | 0 | visible_only,worker_partial_write,dry_run_persists,batch_atomicity_missing |
| Helper commit shortcut | 20 | 0.6607 | 0.7964 | 0.1800 | 38 | 0 | visible_only,helper_commit_shortcut,batch_atomicity_missing |
| Visible-only repair | 35 | 0.8929 | 0.9357 | 0.4200 | 79 | 0 | batch_atomicity_missing |
| Delete-tests adversarial | 0 | 0.9286 | 0.8072 | -0.4200 | -62 | 1 | integrity_zero |

### Diagnosis

- The real `gpt-5.4` high live probe shows the family is substantially too
  easy for the Layer A 15-25 target. Four variants average at least `88`, and
  V4 saturates at `100/100`.
- Spot-check: `v4-multi-corpus-objective` run 1 hit every visible, static,
  hidden, integrity, and partial-progress check (`P_benchmark=100`,
  `M_training=1.0000`). The agent correctly moved transaction ownership out of
  repository helpers, modernized query usage, fixed API/worker/admin
  transaction boundaries, and updated the cutover note.
- The only strong discriminator in the counted probe is V5 incident recovery:
  runs 2 and 3 scored `35/100` after `incident_blind_fix` capped raw `85`
  fixes. This indicates the incident-context ceiling is legitimate, but it is
  isolated; the earlier variants do not add enough independent judgment
  pressure.
- Layer A status: not accepted. The honest calibration signal is understood:
  reaching the 15-25 window would require a further family-local hardening
  attempt with additional concrete behavioral requirements or hidden fixtures,
  not a scorer-only trap or fake ambiguity.
