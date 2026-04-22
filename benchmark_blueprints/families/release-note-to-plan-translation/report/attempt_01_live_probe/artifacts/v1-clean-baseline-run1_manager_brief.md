# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before unmasking smoke runs

Start by auditing the translator step-id schema and deciding the legacy parser shim removal boundary so local smoke runs stop hiding contract drift before any downstream rollout work proceeds.

Bounded deliverable: A confirmed step-id contract and shim-removal decision that exposes real translator drift instead of masked local smoke success.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

### 2. RN-102 — Backfill dependency-graph fixtures for the audited schema

Once the translator contract is explicit, update the dependency-graph fixtures so tests fail on ordering regressions tied to the new step ids rather than after the dashboard path is already exposed.

Bounded deliverable: Fixture coverage that accepts the audited schema and rejects bad ordering for newly introduced step ids.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Canary the dashboard summary behind the kill switch

Only after the schema and fixtures are stable should the translated release-plan dashboard summary be enabled in canary behind the kill switch, because the dashboard directly reflects translator output to users.

Bounded deliverable: A kill-switched canary of the dashboard summary using the audited schema and refreshed fixtures.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

### 4. RN-104 — Update the operator runbook after canary behavior is proven

Finish by updating the operator runbook and launch checklist after canary proves the rollout and rollback shape, since documentation is not stable until the summary path is exercised with real gating behavior.

Bounded deliverable: An operator runbook and launch checklist aligned to the proven canary and rollback flow.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixture backfill only has durable value after the translator step-id contract is audited and the shim no longer hides drift.
- `RN-102` before `RN-103`: The dashboard summary must not ship until dependency-graph fixtures reflect the new schema and catch ordering regressions first.
- `RN-103` before `RN-104`: The runbook stabilizes only after canary reveals the real rollout and rollback shape of the dashboard summary path.

## Primary Risk

If the dashboard summary is exposed before translator drift is audited and fixture coverage is updated, users can see a release plan rendered in the wrong order with no test net for bad dependency sequencing.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Keep the dashboard summary behind the kill switch until RN-101 and RN-102 are complete.
- Use fixture backfill to turn bad step ordering into a contract failure before canary exposure.
- Delay runbook and launch-checklist updates until canary confirms the real rollback path.

## Assumption Ledger

- release context override [observed]: No release_context directory is present, so the frozen April release notes remain the current objective source.
- incident rollback specifics [missing]: No incident_context directory is present, so the exact prior rollback trigger is unknown and the plan stays conservative by keeping dashboard work after schema and fixture hardening.
- dashboard canary gate [to_verify]: The notes mention a kill switch and canary-only coverage, but the exact operational gate and success threshold are not described in the repo evidence.

