# Release Plan Brief

- Variant: `v2-noisy-distractor`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before touching rollout paths

Start by auditing translator schema drift and deciding whether the legacy parser shim can be removed, because the current dashboard objective is blocked on schema stability and the existing contract already shows step-id drift.

Bounded deliverable: A bounded schema-audit decision with the shim outcome recorded so downstream fixture and rollout work targets the stable contract.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 2. RN-102 — Backfill dependency fixtures and plan-contract coverage

Once the schema target is clear, backfill the dependency-graph fixtures so ordering regressions are caught by the plan contract instead of surfacing later in the dashboard summary path.

Bounded deliverable: Dependency-graph fixtures and contract coverage aligned to the audited schema, including protection against step-order regressions.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable the dashboard summary behind the kill switch

Only after schema and fixtures are stable should the translated release-plan dashboard summary be enabled behind its kill switch, because that summary remains the user-visible release objective.

Bounded deliverable: A kill-switch-gated dashboard summary canary running on the stable translator contract.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook after canary proof

Refresh the operator runbook and launch checklist only after the summary path has been proven in canary so operational guidance matches the actual shipped sequence instead of a draft ordering.

Bounded deliverable: Runbook and launch checklist updated against the canary-proven summary flow.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 5. RN-105 — Rank the January cache experiment as optional cleanup

Treat the January cache experiment as a last-pass exploratory item, because both the frozen notes and repo evidence say it predates the translator rewrite and does not address the current dashboard ordering bug.

Bounded deliverable: A consciously deferred cache follow-up, kept out of the main release path unless the current dashboard objective is already stable.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_notes/stale_note_jan_2026.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixture backfill has to target the audited translator schema, or the contract will codify the wrong step ids and ordering.
- `RN-102` before `RN-103`: The dashboard rollout is gated on schema audit plus fixture backfill, so summary enablement must wait until ordering regressions are caught in tests.
- `RN-103` before `RN-104`: Operator guidance should be updated only after the kill-switch summary path is proven in canary, otherwise the checklist will document an unverified sequence.
- `RN-104` before `RN-105`: The cache experiment is stale and non-blocking, so it belongs after the current release objective and its rollout instructions are stable.

## Primary Risk

If the dashboard summary is enabled before schema drift is audited and dependency fixtures are backfilled, users will see a release-plan summary with the wrong ordering, and operators may launch from a checklist that reinforces the same broken sequence.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`
- `release_notes/release_notes_2026_04.md`

Mitigations:
- Keep schema audit as the first bounded milestone so downstream work anchors on the stable translator contract.
- Backfill dependency fixtures before rollout so the plan contract catches ordering regressions prior to dashboard exposure.
- Use the kill switch and canary proof before updating operator-facing launch guidance.

## Assumption Ledger

- Release objective [observed]: Repo state says the user-visible dashboard summary is still the current release objective.
- January cache experiment [observed]: Release notes and repo inventory both mark the cache experiment as superseded and not a fix for the ordering bug.
- Canary exit criteria [missing]: The workspace does not define what evidence is sufficient to declare the kill-switch canary proven for RN-103 to RN-104 handoff.

