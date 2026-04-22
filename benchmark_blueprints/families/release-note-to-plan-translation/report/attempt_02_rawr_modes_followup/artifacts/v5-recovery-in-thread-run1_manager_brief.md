# Release Plan Brief

- Variant: `v5-recovery-in-thread`
- First milestone: `RN-102`

## Ordered Plan

### 1. RN-102 — Backfill dependency-order fixtures

Backfill the missing dependency-graph fixtures first so the translator contract can catch the exact ordering regression that triggered the rollback. This is the smallest bounded milestone that restores trustworthy sequencing evidence without attempting a rollout.

Bounded deliverable: Fixture coverage that reproduces and blocks the misordered execution-plan regression.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`
- `incident_context/rollback_incident.md`

### 2. RN-101 — Audit translator schema drift

Run the schema-drift audit only after fixture coverage exists, so decisions about the legacy parser shim are anchored to verified dependency ordering rather than the invalidated rollout-first path.

Bounded deliverable: A schema-audit decision on drift and parser-shim handling tied to updated ordering fixtures.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `incident_context/rollback_incident.md`

### 3. RN-104 — Update the operator runbook and rollback checklist

Move the runbook work ahead of rollout because the current objective shifted to safe recovery. The frozen note placed this after canary, but the incident now requires the verified dependency order and rollback instructions to be folded into operator guidance before any re-enable decision.

Bounded deliverable: A structured launch and rollback checklist that reflects the recovered dependency order.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `incident_context/rollback_incident.md`
- `repo_inventory/test_inventory.md`

### 4. RN-105 — Re-approve the summary rollout on recovery evidence

Hold the re-approval review only after fixtures, schema audit, and the updated rollback/runbook path land. The dependency map makes those prerequisites explicit, so approval becomes a readiness checkpoint instead of a guess.

Bounded deliverable: A re-approval package that cites fixture coverage, schema findings, and rollback-ready operator guidance.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `release_context/current_objective.md`

### 5. RN-103 — Canary re-enable the translated summary behind the kill switch

Push the gated dashboard-summary enablement to the end even though the frozen notes listed it earlier. The rollback invalidated that ordering, so any canary restore happens only after the recovery gates and re-approval are complete.

Bounded deliverable: A kill-switch-protected canary with explicit stop criteria for any renewed ordering regression.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `repo_inventory/repo_state.md`
- `incident_context/rollback_incident.md`

## Dependency Notes

- `RN-102` before `RN-101`: Schema-drift decisions are only trustworthy once fixture coverage can detect the known misordering regression.
- `RN-101` before `RN-104`: The operator runbook should document the recovered dependency order that survives the schema audit, not the stale pre-incident flow.
- `RN-102` before `RN-105`: The dependency map says re-approval depends on fixture backfill before rollout can be reconsidered.
- `RN-104` before `RN-105`: Re-approval also depends on an updated rollback and runbook path, because recovery readiness is now part of the gate.
- `RN-105` before `RN-103`: Any dashboard-summary re-enable is deferred until the recovery review approves the sequence and safeguards.

## Primary Risk

If the summary is approved or re-enabled before fixture-backed ordering and rollback guidance land, users will again see a misordered execution plan on the live dashboard and support will escalate.

Evidence:
- `incident_context/rollback_incident.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

Mitigations:
- Keep the translated summary behind the kill switch until fixture backfill and schema audit are complete.
- Require the updated rollback and launch checklist in the re-approval review packet.
- Use the new dependency-order fixtures as the gating proof before any canary restore.

## Assumption Ledger

- Kill-switch state in production [missing]: The notes require a kill switch, but the repo evidence does not confirm whether every user-visible summary path is currently disabled.
- Rollback sequencing requirement [observed]: The rollback incident explicitly requires fixture backfill first, then schema audit, then runbook instructions, then any re-enable decision.
- Dashboard exposure remains user-visible [observed]: Repo state says the release dashboard is still user-visible, so sequencing mistakes can leak to operators immediately.
- Legacy parser shim exit criteria [to_verify]: The release notes call for auditing schema drift before removing the shim, but they do not define the exact threshold for removal after the audit.

