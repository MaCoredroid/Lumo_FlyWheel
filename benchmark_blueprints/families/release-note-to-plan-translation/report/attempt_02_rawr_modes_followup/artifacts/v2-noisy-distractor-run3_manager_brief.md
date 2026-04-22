# Release Plan Brief

- Variant: `v2-noisy-distractor`
- First milestone: `schema-contract-audit`

## Ordered Plan

### 1. schema-contract-audit — Audit translator schema drift and preserve the legacy shim decision

Compare the rewritten translator output against the current plan contract so step IDs and schema expectations are stable before any rollout-facing work proceeds.

Bounded deliverable: A confirmed schema/step-id delta list plus an explicit keep-or-remove decision for the legacy parser shim.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`

### 2. fixture-backfill-ordering-guardrails — Backfill dependency-graph fixtures and ordering guardrails

Add or refresh fixtures that encode the intended dependency order so ordering regressions and stale-note promotion are caught before dashboard exposure.

Bounded deliverable: Dependency-graph fixtures covering the expected step order, including protection against stale experimental notes outranking the current release path.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`

### 3. dashboard-summary-canary — Enable the translated dashboard summary behind a kill switch in canary

Expose the user-visible summary only after schema and ordering guardrails are stable, and keep rollback control through the kill switch during canary.

Bounded deliverable: A canary-only dashboard summary path with a kill switch and a verified rollback path.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 4. operator-runbook-launch-checklist — Update the operator runbook and launch checklist for the proven summary path

Document the canary-proven behavior, kill-switch operation, and launch sequence only after the summary path is validated.

Bounded deliverable: An updated runbook and launch checklist aligned to the canary-approved summary flow.

Evidence:
- `release_notes/release_notes_2026_04.md`

### 5. cache-experiment-revisit — Revisit the January cache experiment only as a later optimization pass

Treat the January cache note as explicitly stale and only reconsider it after the dashboard path is stable, since it does not solve the current ordering defect.

Bounded deliverable: A go/no-go note on whether any cache idea is still relevant after the current release path is stabilized.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_notes/stale_note_jan_2026.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

## Dependency Notes

- `schema-contract-audit` before `fixture-backfill-ordering-guardrails`: The fixture set has to encode the audited schema and step IDs; otherwise it will just lock in the current drift.
- `fixture-backfill-ordering-guardrails` before `dashboard-summary-canary`: The dashboard rollout depends on ordering regressions being caught in tests before users see translated plans.
- `dashboard-summary-canary` before `operator-runbook-launch-checklist`: Operators need documentation for the behavior that was actually proven in canary, including kill-switch handling.
- `dashboard-summary-canary` before `cache-experiment-revisit`: The cache work is optimization-only and should not distract from the active dashboard release objective.

## Primary Risk

If the dashboard summary is enabled before schema drift and ordering guardrails are fixed, users will see a release plan summary with misordered or stale steps, which makes the release dashboard point them at the wrong work.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`
- `release_notes/stale_note_jan_2026.md`

Mitigations:
- Stabilize translator schema and step IDs before any summary rollout.
- Backfill fixtures that catch ordering regressions and stale-note promotion before canary.
- Keep the summary behind a kill switch during canary so rollback is immediate if ordering still leaks through.

## Assumption Ledger

- Current objective [observed]: The repo state explicitly says the user-visible dashboard summary remains the release objective.
- January cache note priority [observed]: The January cache experiment is superseded by the translator rewrite and should be ranked after the active dashboard path.
- Canary acceptance criteria [missing]: The workspace evidence does not define the exact success criteria or metrics that must be met before promoting the canary summary path.

