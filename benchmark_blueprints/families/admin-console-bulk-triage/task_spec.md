# Admin Console Bulk Triage

## Task Identity
- `task_id`: `admin-console-bulk-triage/queue-rebalance`
- `family_id`: `admin-console-bulk-triage`
- `scenario_type`: `feature_evolution`

## Task Prompt
You are working in the `ops-console` monorepo. Shift leads need a bulk triage action in the admin dashboard so they can select multiple unassigned incidents, move them into a queue from the live UI, and capture evidence for the handoff. Implement the bulk queue assignment flow end to end, verify it in the running browser app, and update the operator runbook so the new workflow is documented.

## Workspace Bundle
- `frontend/`: React admin console with seeded incident table, selection state, and modal components.
- `api/`: small Node or Python API with queue-assignment endpoint, request validation, and audit log writes.
- `seed/`: deterministic SQLite or Postgres seed with duplicate queue labels and mixed incident states.
- `docs/runbooks/bulk_triage.md`: shift lead workflow that still describes single-item triage only.
- `scripts/dev_up.sh`: launches API, frontend, and seed restore locally.
- `artifacts/screenshots/`: expected location for captured browser evidence.

## Seeded GUI Or Browser-Only Breakage
- Row checkboxes and the bulk action bar render, but the modal submits against a stale queue key that only appears after interacting with the live dashboard.
- The visible table updates optimistically, but the persisted queue assignment and audit-log row are wrong unless the browser flow is fixed end to end.
- Duplicate queue labels in the seed data mean a solver cannot pass by wiring a label-only client shortcut; the live UI must use the canonical queue id.

## Required Surfaces
- `computer_use`
- `browser_preview`
- `shell`
- `apply_patch`
- `screenshots`

## Expected Deliverables
- Code patch across frontend and API as needed.
- Updated `docs/runbooks/bulk_triage.md`.
- Screenshot bundle showing selected incidents, bulk modal, and post-submit state.
- Short validation note summarizing what was tested in the live UI.

## Grader Contract
- Start the seeded stack with `scripts/dev_up.sh`.
- Run hidden unit or integration coverage for queue-id resolution and audit-log persistence.
- Run a hidden browser flow that:
  - selects three specific incidents with mixed visible labels,
  - opens the bulk triage modal,
  - assigns them to the intended queue,
  - reloads the page and confirms the queue badges persist,
  - verifies the audit-log table or API state records the exact canonical queue id for all three seeded incident ids.
- Trusted post-run checks confirm:
  - the runbook mentions the new bulk flow and the screenshot path,
  - at least two screenshots exist in `artifacts/screenshots/`,
  - screenshots are not blank and correspond to the seeded scenario ids,
  - no test files or seed fixtures were deleted.
- Score heavily weights persisted state plus browser evidence. Frontend-only visual success is insufficient.

## Red-Team Traps To Defend Against
- Hardcode the first visible queue label and pass the visible fixture only.
- Stub the bulk action client-side without writing the correct persisted queue id.
- Pre-edit the seed database instead of fixing the flow.
- Capture fake or blank screenshots unrelated to the seeded incidents.
- Remove or weaken the visible tests that check only optimistic UI updates.

## Quality Gate
- Initial weakness to watch: if the grader only checks for a rendered button and one toast, a naive GPT-5.4/high solver could score above 30 by patching the frontend superficially.
- Required hardening:
  - hidden reload check on persisted queue state,
  - audit-log verification against canonical queue ids,
  - exact queue-id assertion across all three seeded incidents, not a majority pass,
  - duplicate-label seed data,
  - screenshot-to-scenario validation.
- Actual GPT-5.4/high scored run: `19/100` from the family bundle alone. The child agent diagnosed the right defect and proposed the right patch surfaces, but earned zero implementation, persistence, and browser-proof points because no runnable repo surfaces were present.
- Naive solver hardness verdict: `under_30 = yes`.
