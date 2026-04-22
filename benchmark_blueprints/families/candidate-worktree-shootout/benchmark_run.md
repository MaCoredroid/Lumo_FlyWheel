# Benchmark Run — `candidate-worktree-shootout`

Family run log for CNB-55 Track 11, family `candidate-worktree-shootout`.

## Model Under Test

Planned live probe target:

```bash
codex exec --model gpt-5.4 --reasoning-effort high
```

That live probe loop is **not** part of this handoff. The work below brings the
family to a runnable Layer-B state and records deterministic baselines only.

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

## Next Step

Run the live probe loop against all five variants and append the measured family
mean / per-variant means to this file. That step was intentionally not launched
from this handoff.
