# Release Cockpit Approval Threading

## Task Identity
- `task_id`: `release-cockpit-approval-threading/pause-resume-handoff`
- `family_id`: `release-cockpit-approval-threading`
- `scenario_type`: `cross_layer_changes`

## Task Prompt
The release cockpit needs a real `paused_for_approval` state that threads through the API, admin UI, and recurring automation so release managers can pause a rollout for human review and later resume it from the same workflow. Implement the state end to end, verify the pause and resume path in the live admin UI, and update the automation and runbook artifacts so the behavior is operationally correct.

## Workspace Bundle
- `services/release_api/`: backend models, serializers, and transition rules.
- `apps/cockpit/`: admin SPA with release detail page, status badges, and action menu.
- `automations/release_followup/automation.toml`: recurring automation that still treats paused releases as active.
- `docs/runbooks/release_cockpit.md`: stale runbook with no paused-state instructions.
- `seed/releases.json`: seeded releases with one item that should enter `paused_for_approval`.
- `artifacts/screenshots/`: required screenshot output path.

## Seeded GUI Or Browser-Only Breakage
- The backend partially knows the new state, but the admin UI badge mapping and action gating are inconsistent, so the live browser workflow misrepresents paused releases.
- The automation still wakes on paused releases and posts the wrong follow-up action, which is only apparent after the state is changed through the UI.
- Generated seed data includes a near-duplicate status label, so frontend-only string mapping is insufficient.

## Required Surfaces
- `computer_use`
- `browser_preview`
- `shell`
- `apply_patch`
- `automation`
- `screenshots`

## Expected Deliverables
- Code patch across backend and UI as needed.
- Updated `automations/release_followup/automation.toml`.
- Updated `docs/runbooks/release_cockpit.md`.
- Screenshot bundle covering pause and resume states in the admin UI.
- Short deploy or rollout note describing the new approval-state semantics.

## Grader Contract
- Start the seeded release cockpit stack.
- Run hidden API or integration checks for state transitions, serializer output, and resume behavior.
- Run a hidden browser flow that:
  - opens the seeded release,
  - pauses it for approval,
  - confirms badge, action menu, and detail timeline reflect `paused_for_approval`,
  - resumes it and confirms the release returns to the exact prior active substate and work item, not a generic active label.
- Run hidden automation checks that the recurring automation suppresses follow-up while paused, stores resumable provenance while paused, and resumes the correct thread or task when unpaused.
- Trusted checks confirm:
  - the runbook documents pause and resume operator steps,
  - screenshots show both paused and resumed states,
  - the automation config references the new state explicitly,
  - no tests or release fixtures were removed.
- Hidden scoring rejects fixes that only relabel the frontend while leaving API or automation semantics unchanged.

## Red-Team Traps To Defend Against
- Add a frontend-only badge alias without persisting the new state.
- Globally suppress the automation instead of gating only the paused state.
- Skip resume semantics and leave the release stuck after approval.
- Edit fixtures or screenshots to make the state appear implemented.
- Patch tests to accept the old active-state behavior.

## Quality Gate
- Initial weakness to watch: if paused-state grading is UI-only, a naive GPT-5.4/high solver could score above 30 with label mapping and screenshot theater.
- Required hardening:
  - hidden API transition checks,
  - resumable provenance check on pause,
  - automation suppression and exact resume-target verification,
  - seeded near-duplicate labels,
  - trusted runbook and screenshot checks.
- Actual GPT-5.4/high scored run: `20/100`. After the provenance and exact-resume hardening, the child agent could still plan the cross-surface patch accurately, but it earned no implementation, provenance, automation, or browser-proof points without the missing runtime bundle.
- Naive solver hardness verdict: `under_30 = yes` after hardening.
