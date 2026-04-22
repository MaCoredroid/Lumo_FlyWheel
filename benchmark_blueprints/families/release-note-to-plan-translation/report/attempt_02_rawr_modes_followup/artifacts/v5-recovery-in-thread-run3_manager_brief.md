# Release Plan Brief

- Variant: `v5-recovery-in-thread`
- First milestone: `RN-102-fixture-backfill`

## Ordered Plan

### 1. RN-102-fixture-backfill — Backfill dependency-order fixtures before any rollout work

Start with the missing dependency-graph fixture coverage so ordering regressions become detectable before the team revisits any user-visible summary path. This is the smallest bounded recovery milestone because the rollback was caused by enabling the summary before those fixtures existed.

Bounded deliverable: Dependency-fixture coverage is added and the release-plan contract can catch misordered execution plans before launch decisions are revisited.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`
- `incident_context/rollback_incident.md`

### 2. RN-101-schema-audit — Audit translator schema drift against the now-covered ordering contract

Once the fixture net exists, audit schema drift and decide whether the legacy parser shim can be removed without reintroducing ordering mismatch. Doing the audit second keeps the schema decision tied to verified dependency behavior rather than to stale release-note sequencing.

Bounded deliverable: A schema-drift audit records whether the translator contract is aligned with the dependency fixtures and whether the legacy parser shim is safe to retire.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `release_context/current_objective.md`

### 3. RN-104-runbook-recovery-update — Update the operator runbook and rollback checklist before canary decisions

The frozen note placed the runbook update after canary proof, but the rollback incident changes that order. The runbook and launch checklist need the dependency-order guardrails and rollback instructions folded in before anyone considers re-enabling the summary path.

Bounded deliverable: Operators have a structured launch and rollback checklist that reflects the incident learnings and the verified dependency order.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `incident_context/rollback_incident.md`
- `repo_inventory/test_inventory.md`

### 4. RN-105-rollout-reapproval — Re-approve the summary rollout only after the recovery prerequisites are complete

The re-approval review should happen after fixture backfill, schema audit, and runbook recovery work are complete, because the dependency map says those are gating prerequisites. This converts RN-105 from a paperwork step into an explicit checkpoint that the rollback conditions were actually resolved.

Bounded deliverable: A recovery review signs off that fixture coverage, schema alignment, and rollback guidance are all in place before any user-visible summary return.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `incident_context/rollback_incident.md`

### 5. RN-103-summary-canary — Re-enable the translated dashboard summary behind the kill switch in canary

Only after the recovery review passes should the translated summary be reintroduced behind the kill switch and observed in canary. This note is no longer an early rollout step; it is the final validation step after the rollback-driven prerequisites are satisfied.

Bounded deliverable: The dashboard summary returns in a controlled canary behind the kill switch, with rollback guidance already prepared if ordering regressions reappear.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-102-fixture-backfill` before `RN-101-schema-audit`: The incident follow-up explicitly requires fixture backfill first so schema decisions are checked against a contract that can detect ordering regressions.
- `RN-101-schema-audit` before `RN-104-runbook-recovery-update`: The runbook needs the verified dependency-order and schema findings, not the stale rollout-first assumptions from the frozen notes.
- `RN-104-runbook-recovery-update` before `RN-105-rollout-reapproval`: The dependency map says re-approval depends on an updated rollback and runbook path in addition to the technical prerequisites.
- `RN-105-rollout-reapproval` before `RN-103-summary-canary`: The summary should only return behind the kill switch after recovery sign-off confirms the rollback conditions were addressed.

## Primary Risk

If the team restores the dashboard summary before fixture coverage, schema verification, and rollback guidance are in place, users will again see a misordered execution plan in a still-user-visible dashboard and support escalation will repeat.

Evidence:
- `incident_context/rollback_incident.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Make dependency-fixture backfill the first milestone so ordering regressions are detectable before any rollout work resumes.
- Fold the rollback lessons into the operator checklist before re-approval so the recovery path is executable, not just described in prose.
- Keep the summary behind the kill switch and limit it to canary only after the recovery review signs off.

## Assumption Ledger

- Dashboard exposure [observed]: The repo-state evidence says the release dashboard remains user-visible, so any sequencing mistake has immediate external impact.
- Kill-switch coverage [to_verify]: The notes mention a kill switch, but the inventory does not confirm whether every user-visible summary entry point is actually gated by it.
- Legacy parser shim consumers [missing]: The inventory does not identify which downstream paths still depend on the legacy parser shim, so the exact blast radius of removing it is unknown.

