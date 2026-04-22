# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before touching exposure paths

Start by auditing and freezing the translator step-id schema because the legacy parser shim currently hides drift in local smoke runs. This milestone should expose where translator output differs from the intended contract and define the exit criteria for eventual shim removal, without enabling any new user-visible surface.

Bounded deliverable: A schema drift audit with the target step-id contract, the shim-masked failure cases, and explicit blocker criteria for downstream fixture and rollout work.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures against the audited schema

Once the audited schema is known, update dependency-graph fixtures and contract coverage so new step ids and bad ordering fail deterministically. This is the point where the repo starts catching sequencing regressions that local smoke runs and current canary coverage would otherwise miss.

Bounded deliverable: Fixture updates and plan-contract checks that fail on schema drift and dependency ordering regressions for the audited translator output.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary in canary behind the kill switch

Only after schema audit and fixture backfill should the translated release-plan dashboard summary be exposed in canary. The kill switch should remain the controlling guardrail because the dashboard is user-visible and currently renders whatever order the translator emits.

Bounded deliverable: Canary-only dashboard summary enablement with the kill switch intact and with schema-compatible fixtures protecting the release-plan ordering contract.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 4. RN-104 — Update operator runbook and launch checklist after canary proof

The operator runbook and launch checklist come last, after the canary path proves both rollout behavior and rollback shape. Writing the docs earlier would hard-code assumptions about a summary path whose operational failure modes are not yet validated.

Bounded deliverable: A runbook and launch checklist updated with the proven canary activation steps, rollback path, and operator expectations for the translated summary.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixture backfill must target the audited schema, or the contract will lock in the wrong step ids and preserve drift.
- `RN-102` before `RN-103`: The dashboard summary must not ship until fixtures reflect the new schema and can catch bad dependency ordering.
- `RN-101` before `RN-103`: The legacy shim masks schema drift in smoke runs, so the user-visible dashboard cannot be trusted until the audit makes drift explicit.
- `RN-103` before `RN-104`: The runbook only stabilizes after rollout and rollback behavior are observed through the guarded canary summary path.

## Primary Risk

If the dashboard summary is enabled before schema drift is audited and fixtures are backfilled, operators and end users can see a polished but incorrectly ordered release plan because the dashboard renders translator output directly while current coverage does not reliably catch bad dependency ordering.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Keep the dashboard summary behind the kill switch until the translator schema audit is complete and the target step-id contract is explicit.
- Backfill dependency-graph fixtures and plan-contract checks before canary exposure so schema and ordering regressions fail pre-release.
- Delay runbook and checklist edits until the canary path proves both activation and rollback behavior.

## Assumption Ledger

- Current operating objective [observed]: No release_context or incident_context directory is present, so the frozen notes and repo inventory are the current planning inputs.
- Dashboard guardrail [observed]: The dashboard summary is user-visible and consumes the translator step-id schema, so its sequencing must follow schema and fixture hardening.
- Shim removal timing [missing]: The notes require auditing drift before removing the legacy parser shim, but the evidence does not state whether shim removal is in-scope for this release or deferred.
- Canary exit criteria [to_verify]: The evidence confirms canary-only summary coverage, but it does not define how much successful canary evidence is required before updating the runbook and checklist.

