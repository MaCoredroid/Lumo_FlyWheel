# `sqlalchemy-2-session-modernization` Task Spec

**Track:** 03 — Refactor Modernization  
**Family id:** `sqlalchemy-2-session-modernization`  
**Spec version:** CNB-55 v1.0  
**Variants:** 5 (`v1` through `v5`)

## Canonical Task Prompt

Modernize the `ledger-sync` service from legacy SQLAlchemy `session.query(...)`
usage and implicit transaction behavior to SQLAlchemy 2.0 `select(...)` access
 with explicit transaction boundaries.

The fix must hold across:

- `app/api.py` — API create/read path
- `app/repository.py` — shared row-loading and count/existence helpers
- `app/worker.py` — retrying settlement flow
- `app/admin_cli.py` — dry-run and batch planning paths
- `docs/deploy/sqlalchemy2-cutover.md` — operator cutover / rollback note

The benchmark is about **behavioral modernization**, not a mechanical API swap.
Replacing `query()` with `select()` while preserving helper-owned commits,
partial writes on failure, or a writeful dry-run is a scoreable failure.

## Required Surfaces

- `shell`
- `apply_patch`
- `pytest`
- local file inspection
- docs editing

No network, no MCP, no browser, no subagents.

## Workspace Layout

Each variant ships a self-contained workspace under
`workspace_bundle/<variant_id>/`:

```text
AGENTS.md
Dockerfile
.scenario_variant
app/
docs/deploy/sqlalchemy2-cutover.md
seed/ledger_seed.sql
tests/test_api.py
tests/test_worker.py
tests/test_admin_cli.py
notes/                # V2+ and V3 only
README.md             # V3 only
release_context/      # V4+
incident_context/     # V5
artifacts/README.md
```

Visible validation is always:

```bash
pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py
```

## Variant Progression

### v1 — clean baseline

Single-service cutover with the core bug set only:

- repository helper still uses `session.query(...)`
- helper-owned commits leak transaction control
- worker failure persists `processing`
- admin dry-run persists `queued`
- deploy note still recommends a compatibility shim

**Primary stress:** can the solver land a real SQLAlchemy 2 migration with
explicit transaction ownership rather than a syntax-only rewrite.

### v2 — noisy distractor

V1 plus stale compatibility-shim notes under `notes/`.

**Primary stress:** can the solver ignore stale rollout lore instead of
re-introducing a shim or preserving helper commits because an old note once
claimed they were safe.

### v3 — dirty state

V2 plus a nearby `README.md` edit and an abandoned partial patch note.

**Primary stress:** can the solver fix the transaction model without reverting
or "cleaning up" unrelated local edits, and without finishing the abandoned
per-row-commit direction.

### v4 — multi-corpus objective

V3 plus `release_context/` that makes the release objective more explicit:
operator-safe dry-run behavior and batch rollback are now first-class blockers.

**Primary stress:** can the solver re-weight toward release-safe transaction
behavior across batch helpers rather than only fixing the single-row visible
tests.

### v5 — recovery in thread

V4 plus `incident_context/` describing a rolled-back cutover caused by retry
state leakage and writeful dry-runs.

**Primary stress:** can the solver fix the code and update the deploy note in a
way that acknowledges the incident and proves the retried cutover is now safe.

## Expected Deliverables

- `app/repository.py` uses SQLAlchemy 2 `select(...)` access patterns and no
  helper-owned commits
- `app/worker.py` uses explicit transaction boundaries so forced failures roll
  back to the pre-attempt state
- `app/admin_cli.py` keeps dry-run read-only and batch flows atomic
- `docs/deploy/sqlalchemy2-cutover.md` names the new transaction model and a
  concrete rollback procedure

Allowed write targets are intentionally narrow:

- `app/api.py`
- `app/repository.py`
- `app/worker.py`
- `app/admin_cli.py`
- `docs/deploy/sqlalchemy2-cutover.md`

Everything else is immutable benchmark context.

## Hidden Expectations

The hidden grader checks behavior the visible suite does not fully cover:

- `entry_exists()` and `pending_entry_count()` still behave correctly after the
  migration
- repository helpers do not own commits
- no global session singleton is introduced
- V4/V5 batch worker/admin flows are atomic on failure
- V5 retry after a forced failure is idempotent and the deploy note acknowledges
  the rollback incident

## Saturation And Renewal Plan

This family is considered saturated when the probe model's mean
`P_benchmark > 80` for two consecutive rounds.

Renewal queue:

1. Add a sixth variant with a second ORM helper module so the solver must avoid
   a partial modernization that leaves one transaction path behind.
2. Retire the current V1 and promote V2 as the baseline if compatibility-shim
   noise stops discriminating.

