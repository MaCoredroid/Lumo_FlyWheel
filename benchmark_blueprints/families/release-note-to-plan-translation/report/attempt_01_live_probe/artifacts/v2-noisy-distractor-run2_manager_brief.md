# Release Plan Brief

- Variant: `v2-noisy-distractor`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before parser-shim removal

Start by reconciling the rewritten translator schema with the still-present legacy parser shim so downstream fixtures and dashboard work target the same contract.

Bounded deliverable: An audited schema decision with the drift surface and shim dependency explicitly recorded.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`

### 2. RN-102 — Backfill dependency-graph fixtures against the audited contract

Once the schema target is stable, refresh fixtures so the plan contract catches ordering regressions and step-id drift before any user-facing summary is exposed.

Bounded deliverable: Dependency-graph fixtures aligned to the audited schema and covering the known ordering-regression surface.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind a kill switch after stability work

Ship the translated release-plan dashboard summary only after the schema and fixtures are stable, and keep it behind the kill switch while canary evidence is collected.

Bounded deliverable: A kill-switched canary of the translated dashboard summary using the stabilized translator contract.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 4. RN-104 — Update runbook and launch checklist after canary proof

Document operator actions only after the summary path has been proven in canary so the checklist matches the real guarded rollout sequence.

Bounded deliverable: An operator runbook and launch checklist that reflect the validated canary behavior and kill-switch rollback path.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 5. RN-105 — Revisit the January cache experiment only as a later optimization

Treat the cache experiment as stale context and rank it last because it predates the translator rewrite and does not address the current dashboard ordering problem.

Bounded deliverable: A post-stabilization decision on whether any cache findings still apply to the rewritten translator path.

Evidence:
- `release_notes/stale_note_jan_2026.md`
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixtures should be backfilled only after the schema audit defines the canonical translator contract; otherwise the team encodes the same drift that the audit is meant to resolve.
- `RN-102` before `RN-103`: The dashboard summary is user-visible, so ordering-regression fixtures need to exist before the kill-switched canary to catch the known step-id and dependency ordering failures.
- `RN-103` before `RN-104`: Runbook and launch-checklist updates are only trustworthy after the canary proves the guarded summary path and rollback behavior.
- `RN-104` before `RN-105`: The January cache experiment is a follow-on optimization question, not a prerequisite for the current dashboard release objective.

## Primary Risk

If the dashboard summary is enabled before schema drift is audited and fixtures cover ordering regressions, users will see a release-plan summary with the wrong step order or mismatched step IDs, and operators may document and launch the broken sequence as if it were safe.

Evidence:
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/repo_state.md`

Mitigations:
- Complete RN-101 before any rollout work so the translator contract is explicit.
- Use RN-102 to add regression fixtures for dependency ordering and step-id stability before the canary.
- Keep RN-103 behind the kill switch until canary evidence is strong enough to support RN-104 documentation.

## Assumption Ledger

- Current release objective [observed]: Repo evidence says the user-visible dashboard summary is still the active target, so the plan should optimize for that path rather than the cache experiment.
- January cache experiment priority [observed]: Both the stale note and dependency map say the January cache experiment was superseded and does not fix the present dashboard ordering bug.
- Parser shim exit criteria [missing]: The repo evidence says the shim should not be removed before the schema audit, but it does not define the exact audit acceptance criteria or who signs off on shim retirement.

