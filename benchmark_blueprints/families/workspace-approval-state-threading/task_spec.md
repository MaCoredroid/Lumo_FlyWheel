# Workspace Approval State Threading

## Task Metadata
- `task_id`: `cnb55-core-workspace-approval-state-threading-admin-ui`
- `family_id`: `workspace-approval-state-threading`
- `scenario_type`: `cross_layer_changes`

## Task Prompt
The admin workspace page needs to expose a new `approval_state` field end to end. Add it across backend models, API serialization, CLI export, default config, and the React admin panel. Preserve the existing `risk_level` behavior and do not break old workspace rows that are missing the new field. The operator docs and screenshot artifact must match the implemented UI.

## Workspace Bundle
- Small monorepo rooted at `/workspace`.
- `backend/workspaces/models.py`, `backend/workspaces/service.py`, and `backend/api/serializers.py`.
- `cli/export_workspace.py` produces JSON snapshots for ops handoff.
- `frontend/src/pages/WorkspaceAdmin.tsx` and `frontend/src/components/WorkspaceTable.tsx`.
- `config/defaults.toml` defines default workspace policy values.
- `docs/runbooks/workspace-approvals.md` documents operator handling.
- `tests/backend/`, `tests/frontend/`, and one screenshot snapshot harness under `tests/e2e/`.

## Seeded Breakage
- Backend model defines `approval_state` in one dataclass but the service drops it when normalizing DB rows.
- API serializer omits the field for non-default rows.
- CLI export still emits `risk_level` only, so operator JSON differs from the UI.
- Frontend mock data still assumes `risk_level` is the only status-like column and renders blank cells.
- `config/defaults.toml` uses the old `approval_mode` token that no longer matches runtime naming.
- Runbook screenshots and copy still describe the previous column set.

## Required Surfaces
- `shell`
- `apply_patch`
- backend and frontend test execution
- browser or screenshot preview

## Expected Deliverables
- Backend and frontend code patch.
- Test updates for serialization, backward compatibility, and table rendering.
- `config/defaults.toml` patch aligning defaults with `approval_state`.
- Updated runbook text and one refreshed screenshot artifact reference.
- Short rollout note describing old-row fallback behavior.

## Grader Contract
- Visible gate:
  - backend serializer tests and frontend table tests pass.
  - screenshot harness produces the expected admin page image name.
  - One mixed-dataset test compares service output, API response, and CLI export for the same rows and requires consistent `approval_state` values plus legacy-row fallback.
- Hidden checks:
  - Legacy row without `approval_state` must still render using a documented fallback.
  - CLI JSON export and API response must agree on `approval_state` for mixed datasets.
  - Screenshot diff checks the column label, badge values, and one filtered row state.
  - Grep checks that the runbook and defaults use `approval_state`, not stale `approval_mode`.
  - Grep or AST guard rejects stale `risk_level` alias logic in serializer or frontend table layers where true schema threading is required.
- Fail conditions:
  - Backend-only patch with stale UI or docs.
  - UI-only patch that fabricates the value client-side.
  - Removal of fallback handling for old rows.

## Red-Team Traps
- Alias `approval_state` to `risk_level` in the serializer without true schema threading.
- Add a frontend-only computed column and leave API or CLI output stale.
- Update runbook prose but not the screenshot or default config.
- Hardcode the expected screenshot name while leaving visual state incorrect.
- Delete the backwards-compatibility test instead of implementing fallback logic.

## Quality Gate
- Probe target:
  - A naive `gpt-5.4/high` solver should land around `20/100`, not `30/100`, unless it coordinates backend, frontend, config, docs, and screenshot surfaces with real evidence.
- Hardening applied after adversarial solver run:
  - The task spans enough layers that single-surface patches are intentionally partial.
  - Hidden checks compare API, CLI, UI, and docs rather than trusting one representation.
  - Backward compatibility prevents a blunt rename-only patch from passing.
  - Visible grading now includes a same-dataset consistency test across service, API, and CLI export.
  - Screenshot validation is expected to prove badge text and filtered row state, not only a filename.
- Actual run result:
  - Child subagent solver attempt using `gpt-5.4/high` scored `20/100` under the family rubric.
  - The run proposed coherent edits but could not supply implementation, screenshot proof, or tested consistency evidence.
  - Final assessment: `target hardness met; naive gpt-5.4/high stays around 20/100`.
