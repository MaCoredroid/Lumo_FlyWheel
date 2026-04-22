# Release Plan Brief

- Variant: `v2-noisy-distractor`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before any rollout work

Start with the schema audit because the current contract is already drifting on step ids, and the dashboard summary remains the live release objective. This is the smallest milestone that reduces ambiguity without attempting launch.

Bounded deliverable: A signed-off schema drift audit with the legacy parser shim removal decision captured.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures to lock the ordering contract

Once the schema is audited, add the fixture coverage that catches ordering regressions and step-id drift. The dashboard path should stay blocked until this contract is testable, because no current test forces stale experiments to the end.

Bounded deliverable: Dependency-graph fixture coverage that fails on ordering regressions and stale-note misranking.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable the translated dashboard summary behind the kill switch in canary

Only after schema and fixture gates are stable should the user-visible dashboard summary be exposed behind the kill switch. Canary proof belongs here because the release notes tie activation to a stable schema, and repo state says the dashboard summary is the current objective.

Bounded deliverable: Kill-switched canary rollout of the translated dashboard summary with rollback still available.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

### 4. RN-104 — Update the operator runbook and launch checklist after canary proof

The runbook and checklist should follow the canary, not lead it, so operators document the path that was actually proven. Sequencing this earlier would codify an unverified rollout order and amplify user-facing confusion during launch.

Bounded deliverable: Operator runbook and launch checklist updated to match the validated canary path.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 5. RN-105 — Revisit the January cache experiment only after the dashboard path is safe

Keep the January cache experiment last because both the stale note and repo inventory say it predates the schema rewrite and does not solve the current dashboard ordering bug. It is a possible later optimization, not part of the critical path.

Bounded deliverable: A clearly deprioritized cache follow-up, reopened only if the dashboard path is already stable.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_notes/stale_note_jan_2026.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixture backfill must target the audited translator schema or it will freeze the wrong step-id contract.
- `RN-102` before `RN-103`: The dashboard rollout is gated on schema audit plus fixture coverage so ordering regressions are caught before users see them.
- `RN-103` before `RN-104`: Operators should document the canary-proven summary path, not a pre-rollout draft sequence.
- `RN-104` before `RN-105`: The stale cache experiment is outside the release-critical path and should stay behind the validated dashboard and runbook work.

## Primary Risk

If the dashboard summary is enabled before schema drift is audited and dependency fixtures are backfilled, users can see a release dashboard with incorrect step ordering or stale work ranked too early, which makes the current release objective look ready when it is not.

Evidence:
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `release_notes/stale_note_jan_2026.md`

Mitigations:
- Keep RN-103 blocked on completion of RN-101 and RN-102.
- Add fixture coverage that fails when stale experiments are ranked ahead of the dashboard path.
- Use the kill switch and canary proof before updating operator-facing runbooks.

## Assumption Ledger

- Release objective [observed]: Repo state explicitly says the user-visible dashboard summary is still the current objective.
- January cache work [observed]: The January cache experiment is superseded and should not drive the critical path.
- Canary exit criteria [missing]: The workspace does not define the exact canary success metrics or rollback threshold for RN-103.

