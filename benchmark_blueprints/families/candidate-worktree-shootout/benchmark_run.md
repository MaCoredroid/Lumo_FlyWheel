# Benchmark Run — `candidate-worktree-shootout`

Family run log for CNB-55 Track 11, family `candidate-worktree-shootout`.

## Model Under Test

Live probe target:

```bash
codex exec --model gpt-5.4 --reasoning-effort high
```

The family now has deterministic baselines plus a recorded whole-family live
probe on `codex exec`.

## Attempt History

### `attempt_00` — baseline design

Scoped the family around one legitimate orchestration judgment:

- a visible CLI-only bug fix exists,
- a stronger shared service-layer fix exists,
- the solver must evaluate both in isolation,
- the grader should reward concrete worktree evidence instead of prose claims.

Variant ladder chosen:

- `v1-clean-baseline`: direct-caller evidence only
- `v2-noisy-distractor`: stale archived CLI memo
- `v3-dirty-state`: abandoned CLI-local partial patch
- `v4-multi-corpus-objective`: importer release gate shifts the right objective
- `v5-recovery-in-thread`: prior CLI-local hotfix rollback / incident context

### `attempt_01` — family-local implementation and deterministic verification

Implemented inside the family scope only:

- workspace bundle for all five variants under `workspace_bundle/`
- deterministic scorer at `verifiers/candidate-worktree-shootout/score_shootout.py`
- verifier data, oracle overlays, hidden test stubs, workspace manifests, and milestone scripts
- Layer-B declaration at `family.yaml`
- verification matrices for `v1-clean-baseline` and `v5-recovery-in-thread`

Deterministic baseline results across all five variants:

| Variant | Oracle `P_benchmark` | Empty `P_benchmark` | Shortcut `P_benchmark` |
| --- | ---: | ---: | ---: |
| `v1-clean-baseline` | 100 | 0 | 25 |
| `v2-noisy-distractor` | 100 | 0 | 25 |
| `v3-dirty-state` | 100 | 0 | 25 |
| `v4-multi-corpus-objective` | 100 | 0 | 25 |
| `v5-recovery-in-thread` | 100 | 0 | 25 |

Interpretation:

- oracle is cleanly above the ≥90 Layer-A sanity bar
- empty is exactly 0 on every variant
- the shallow CLI-local shortcut is capped at 25 on every variant

Verification matrix snapshots:

- V1 matrix:
  - Oracle `100`
  - Empty `0`
  - Grounding stripped `20`
  - Pick ceiling `25`
  - Blended ownership `40`
  - Delete-tests adversarial `0` with `integrity_flag = 1`
- V5 matrix:
  - Oracle `100`
  - Empty `0`
  - Grounding stripped `20` with `incident_blind_reselect`
  - Pick ceiling `25` with `incident_blind_reselect`
  - Blended ownership `40`
  - Delete-tests adversarial `0` with `integrity_flag = 1`

Files produced by this attempt:

- [`family.yaml`](./family.yaml)
- [`manifest.lock.json`](./manifest.lock.json)
- [`verification_matrix.md`](./verification_matrix.md)
- [`verification_matrix_v5.md`](./verification_matrix_v5.md)

## Layer Status After `attempt_01`

- **Layer A:** `in_progress`
  - deterministic baselines are wired and sane
  - live `codex exec` probe data is still missing
- **Layer B:** `implemented`
  - dual-band scorer, milestones, capability tags, state-delta rules,
    integrity rules, rawr modes, manifests, and verification matrices are all
    present inside the family scope

### `attempt_02` — first live probe surfaced a harness defect

Initial live launch command:

```bash
python3 benchmark_blueprints/families/candidate-worktree-shootout/scripts/run_probe.py --runs-per-variant 3 --attempt-name attempt_02
```

Observed issue:

- the family-local validation command `python -m pytest -q tests/test_cli.py tests/test_service.py`
  failed before the model could act because the workspace bundle lacked a
  family-local import path shim for `src/report_filters`

Family-local repair:

- added `tests/conftest.py` to every workspace variant to inject `src/` onto
  `sys.path`
- regenerated `workspace_bundle/`, `manifest.lock.json`, and
  `verifier_data/.../workspace_manifest.json` hashes via
  `python3 benchmark_blueprints/families/candidate-worktree-shootout/scripts/regen_family.py`

Interpretation:

- this was benchmark harness drift, not intended task difficulty
- `attempt_02` is not a valid Layer-A measurement and should not be used for
  gate math

### `attempt_03` — timeout policy repair

Follow-up launch command:

```bash
python3 benchmark_blueprints/families/candidate-worktree-shootout/scripts/run_probe.py --runs-per-variant 1 --attempt-name attempt_03 --timeout-seconds 120
```

Observed issue:

- the family-local runner advanced after timeout, but descendant `codex exec`
  processes could continue mutating timed-out workspaces after scoring

Family-local repair:

- hardened `scripts/run_probe.py` to launch each probe in a fresh process group
- on timeout, the runner now kills the full process group before scoring the
  workspace snapshot

Interpretation:

- `attempt_03` established the need for process-group cleanup, but its scores
  are not the canonical family measurement because descendant mutation was not
  yet fully contained

### `attempt_04` — whole-family live probe on `codex exec`

Canonical whole-family live probe command:

```bash
python3 benchmark_blueprints/families/candidate-worktree-shootout/scripts/run_probe.py --runs-per-variant 1 --attempt-name attempt_04 --timeout-seconds 180
```

Per-variant live `codex exec` template used by the runner:

```bash
codex exec --full-auto --ephemeral -m gpt-5.4 -c reasoning_effort="high" --cd <workspace> --output-last-message <last_message_file> --json "Read AGENTS.md and complete the benchmark task in this workspace.
Use two isolated directories under artifacts/comparison/worktrees/ for Candidate A and Candidate B.
Produce the required files under artifacts/comparison/.
Land one coherent final patch in the main workspace and do not modify immutable surfaces.
Return when the workspace satisfies the task."
```

Measured results from [`report/attempt_04_probe_report.txt`](./report/attempt_04_probe_report.txt):

| Variant | Live `P_benchmark` | Live `M_training` | Ceilings |
| --- | ---: | ---: | --- |
| `v1-clean-baseline` | 20 | 0.20 | `missing_worktree_proof`, `cli_only_fix`, `docs_drift` |
| `v2-noisy-distractor` | 20 | 0.20 | `missing_worktree_proof`, `cli_only_fix`, `docs_drift`, `anchored_on_stale_cli_memo` |
| `v3-dirty-state` | 20 | 0.20 | `missing_worktree_proof`, `cli_only_fix`, `docs_drift`, `sunk_cost_finish` |
| `v4-multi-corpus-objective` | 20 | 0.20 | `missing_worktree_proof`, `cli_only_fix`, `docs_drift`, `objective_drift` |
| `v5-recovery-in-thread` | 20 | 0.20 | `missing_worktree_proof`, `cli_only_fix`, `docs_drift`, `incident_blind_reselect` |

Layer-A gate values from [`report/attempt_04_probe_results.json`](./report/attempt_04_probe_results.json):

- family mean: `20.00`
- max variant mean: `20.00`
- min variant mean: `20.00`
- monotonicity (`V1>=V2>=V3>=V4>=V5`, tolerance `±3`): `true`
- acceptance judgment: `red`

Acceptance interpretation:

- `[PASS]` family mean in target band `[15, 25]`
- `[PASS]` max variant mean `<= 40`
- `[FAIL]` at least one variant mean `<= 10` (observed min `20.00`)
- `[PASS]` monotonicity within tolerance

What the live probe says:

- the family now has a real whole-family live measurement on `codex exec`
- the current prompt/model combination clusters all five variants at the same
  `20/100` band
- the family misses Layer A only on the "at least one variant <= 10" freeze-gate
  requirement
- dominant failure pattern is consistent and honest: the model usually lands a
  partial repair but does not complete the required comparison evidence, does not
  consistently select the service-layer fix in final artifacts, and leaves docs
  alignment incomplete under the scorer's current checks

## Layer Status After `attempt_04`

- **Layer A:** `red`
  - live family mean is calibrated into band at `20.00`
  - max variant mean is controlled at `20.00`
  - monotonicity holds
  - freeze gate still fails because no variant probes at or below `10`
- **Layer B:** `implemented`
  - unchanged from `attempt_01`

## Next Step

Use the live `attempt_04` artifacts to harden the family locally against the
observed uniform `20/100` plateau, especially the scorer-visible comparison
artifact completeness and final service-layer-selection evidence.
