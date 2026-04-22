# Workspace Approval State Threading

## Task Metadata
- `task_id`: `cnb55-core-workspace-approval-state-threading-admin-ui`
- `family_id`: `workspace-approval-state-threading`
- `track`: `08`
- `scenario_type`: `cross_layer_changes`
- `solver_runtime`: `repo-edit`

## Canonical Task Prompt
The admin workspace page needs to expose a new `approval_state` field end to end. Add it across backend models, service normalization, API serialization, CLI export, default config, and the React admin panel. Preserve the existing `risk_level` behavior and do not break old workspace rows that are missing the new field.

The operator docs and screenshot contract artifact must match the implemented UI. The rollout note must explicitly document the legacy-row fallback behavior, and on the recovery variant it must also acknowledge why the earlier `risk_level` alias hotfix was rolled back.

## Workspace Bundle Layout
Each variant workspace is a small monorepo rooted at `/workspace` with these writable paths:

- `backend/workspaces/models.py`
- `backend/workspaces/service.py`
- `backend/api/serializers.py`
- `cli/export_workspace.py`
- `frontend/src/pages/WorkspaceAdmin.tsx`
- `frontend/src/components/WorkspaceTable.tsx`
- `config/defaults.toml`
- `docs/runbooks/workspace-approvals.md`
- `tests/**`
- `artifacts/preview/workspace_admin_capture.json`
- `artifacts/rollout/approval_state_rollout_note.json`

Read-only evidence and context:

- `seed_data/mixed_workspaces.json`
- `seed_data/**` extra stale artifacts
- `release_context/**` on V4
- `incident_context/**` on V3 and V5
- `AGENTS.md`
- `Dockerfile`
- `bin/run-visible-tests`
- `.scenario_variant`

## Required Surfaces
- `shell`
- `apply_patch`
- backend and frontend test execution
- JSON preview / screenshot-contract verification

## Structured Deliverable Contract
This family uses a repo-edit submission plus one JSON deliverable:

- Deliverable path: `artifacts/rollout/approval_state_rollout_note.json`
- Schema version: `cnb55.rollout_note.v1`
- Required keys:
  - `schema_version`
  - `legacy_row_fallback`
  - `consistency_surfaces`
  - `screenshot_name`
  - `variant_notes`

Expected `bin` subcommands:

- `bin/run-visible-tests`
- `python3 cli/export_workspace.py`

There is no LLM judge in the milestone layer. All scorer decisions must be reproducible from code and hidden checks.

## Seeded Breakage
- Backend model mentions `approval_state` but service normalization drops it for mixed rows.
- API serialization leaks the field only on one path and otherwise falls back to stale alias behavior.
- CLI export still emits legacy shape, so operator JSON differs from the UI contract.
- Frontend table assumes `risk_level` is the only status-like column and leaves the new badge contract unimplemented.
- `config/defaults.toml` still uses `approval_mode`.
- Runbook text and screenshot contract artifact still reflect the previous column set.
- Tests are intentionally stale and must be updated to assert the new cross-surface contract.

## Variant Progression
- `v1-clean-baseline`
  - Minimal bundle; solver must wire `approval_state` across backend, CLI, UI, config, docs, tests, and preview artifact.
- `v2-noisy-distractor`
  - Adds stale ops snapshots and an outdated memo that still use `approval_mode`; solver must not anchor on the stale token.
- `v3-dirty-state`
  - Adds read-only abandoned patch context that proposes a frontend-only alias; solver must not complete the sunk-cost shortcut.
- `v4-multi-corpus-objective`
  - Adds `release_context/` shifting the operator objective toward launch-readiness triage; the preview artifact must show the blocked row called out in the release brief.
- `v5-recovery-in-thread`
  - Adds `incident_context/` documenting a rollback of the earlier `risk_level` alias hotfix; the rollout note must acknowledge the rollback before describing the new fix.

## Expected Deliverables
- Backend, serializer, CLI, and frontend code patch.
- Updated visible tests for backend, frontend, and same-dataset consistency.
- `config/defaults.toml` patch aligning defaults with `approval_state`.
- Updated runbook text.
- Refreshed preview contract artifact at `artifacts/preview/workspace_admin_capture.json`.
- Valid rollout note JSON at `artifacts/rollout/approval_state_rollout_note.json`.

## Visible Gate
- `python3 -m unittest discover -s tests -p 'test_*.py'`
- Mixed-dataset test compares service output, API response, and CLI export for the same rows and requires consistent `approval_state` values plus legacy-row fallback.
- Frontend contract test checks that the table renders an `Approval state` column and a fallback badge label.
- Preview contract test checks screenshot filename, column labels, badge values, and the expected filtered row for the active variant.

## Hidden Checks
- Legacy row without `approval_state` still renders with documented fallback `manual_review` and `legacy_fallback` source.
- API and CLI export agree for the mixed dataset and do not alias `risk_level`.
- Preview contract matches the expected filtered row for the active variant.
- `config/defaults.toml` and runbook use `approval_state`, not `approval_mode`.
- Recovery variant rollout note explicitly acknowledges the prior rollback.
- Immutable context trees (`seed_data`, `release_context`, `incident_context`, `bin`, `.scenario_variant`, `AGENTS.md`, `Dockerfile`) remain unchanged.

## Hard Failure Modes
- Frontend-only fabrication of `approval_state`.
- Serializer or table aliasing `approval_state` directly from `risk_level`.
- Missing legacy-row fallback.
- Stale `approval_mode` left in config or docs.
- Preview contract or rollout note omitted while code otherwise appears correct.

## Saturation And Renewal Plan
Trigger saturation review when mean `P_benchmark > 80` for two consecutive probe rounds.

Renewal queue:
- Add a variant where the admin table supports batched edits and the solver must preserve per-row fallback provenance.
- Retire the current V1 baseline and promote V2 if the stale-token distractor becomes trivial.
