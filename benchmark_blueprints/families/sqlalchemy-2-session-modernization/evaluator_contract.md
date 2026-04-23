# `sqlalchemy-2-session-modernization` Evaluator Contract

## Evaluation Goal

Reward a real SQLAlchemy 2 transaction-boundary modernization across API,
repository, worker, admin CLI, and deploy-note surfaces.

## Visible Checks

```bash
pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py
```

## Deterministic Hidden Checks

- no live `session.query(` remains in the service path
- repository helpers no longer call `commit()`
- no global session singleton is introduced
- deploy note names transaction boundaries, dry-run behavior, and rollback
- `entry_exists()` and `pending_entry_count()` remain correct
- worker rollback restores `pending` after forced failure
- admin dry-run performs no writes
- V4/V5 batch worker/admin flows are atomic on failure
- V5 retry-after-failure remains idempotent and the deploy note acknowledges the
  prior rollback incident

## Integrity Rules

- `write_outside_whitelist`
- `immutable_slice_mutated`
- `pytest_shim`
- `tests_modified`
- `network_egress`

Only these paths may change:

- `app/api.py`
- `app/repository.py`
- `app/worker.py`
- `app/admin_cli.py`
- `docs/deploy/sqlalchemy2-cutover.md`

## Scoring Model

The scorer emits:

- `P_benchmark` — full benchmark-facing score in `[0, 100]`
- `M_training` — deterministic-only training score in `[0, 1]`
- `score` — alias of `P_benchmark`

The deterministic band is composed from visible tests, static SQLAlchemy 2
checks, hidden behavioral checks, and integrity. A small `P_only` partial
progress heuristic is quarantined from `M_training`.

## Partial-Credit Ceilings

- `integrity_zero` — any integrity violation forces the score to `0`
- `visible_only` — visible tests still red caps at `20`
- `helper_commit_shortcut` — repository helper commit remains caps at `25`
- `worker_partial_write` — worker rollback still leaks state caps at `25`
- `dry_run_persists` — admin dry-run still writes caps at `25`
- `batch_atomicity_missing` — V4/V5 batch paths are non-atomic caps at `35`
- `incident_blind_fix` — V5 ignores the prior rollback incident caps at `35`

## Intended Probe Band

Honest target after hardening:

- family mean: `15..25`
- shallow mechanical solver: `<= 30`
- oracle: `>= 90`
- empty / untouched workspace: `0`

