# Release Plan Brief

- Variant: `v3-dirty-state`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before touching the legacy shim

Reconcile the actual translator schema against the frozen release-note contract and keep the legacy parser shim in place until the canonical shape is confirmed.

Bounded deliverable: A schema-drift audit that names the canonical translator fields, identifies mismatches, and explicitly states whether shim removal is still blocked.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/in_progress_patch/README.md`

### 2. RN-102 — Backfill dependency-graph fixtures to lock plan ordering

Add the missing dependency fixtures immediately after schema stabilization so the plan contract starts catching hidden ordering regressions before any user-visible summary path changes.

Bounded deliverable: Fixture coverage that exercises dependency ordering and fails if downstream plan steps are sequenced ahead of unresolved prerequisites.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable the translated dashboard summary behind a kill switch in canary

Only after schema and fixture guardrails are in place, expose the translated release-plan summary behind the kill switch and prove it in canary while the old path remains recoverable.

Bounded deliverable: A canary-ready summary path gated by the kill switch, with explicit rollback readiness and validation against the stabilized translator contract.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 4. RN-104 — Update operator runbook and launch checklist after canary proof

Refresh operator-facing documentation only once the canary proves the gated summary path, so the checklist reflects the actual safe sequence instead of a stale draft path.

Bounded deliverable: A launch checklist and runbook aligned to the canary-proven summary workflow and its rollback guardrails.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 5. RN-105 — Decide whether to retire or finish the abandoned rollout draft

Treat the inherited draft as optional cleanup after the safe release path is clear, because it depends on the same unresolved schema and fixture work and never added rollback coverage.

Bounded deliverable: A keep-or-retire decision on the abandoned draft, with follow-up work only if it still adds value after the stable path is complete.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/in_progress_patch/README.md`

## Dependency Notes

- `RN-101` before `RN-102`: Ordering fixtures should be derived from the stabilized translator contract; otherwise tests will encode the wrong dependency shape.
- `RN-102` before `RN-103`: The dashboard summary is still user-visible, so kill-switch canary work should not proceed until fixture coverage can catch hidden dependency reordering.
- `RN-103` before `RN-104`: Operator documentation must describe the proven canary path and rollback controls, not a pre-validation design.
- `RN-101` before `RN-105`: The inherited rollout draft assumes the schema is already frozen, so it cannot be evaluated honestly until the schema audit is complete.
- `RN-102` before `RN-105`: The draft shares the unresolved fixture dependency and lacks rollback coverage, so it remains optional cleanup until the safe path is protected.

## Primary Risk

If the dashboard summary or operator checklist ships before schema drift is resolved and ordering fixtures exist, users will see and act on an incorrect release-plan sequence in the still-visible dashboard.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/in_progress_patch/README.md`

Mitigations:
- Keep the legacy parser shim until the schema audit confirms the canonical translator contract.
- Backfill dependency-order fixtures before enabling the translated dashboard summary, even behind the kill switch.
- Delay runbook and checklist changes until the canary path is proven against the stabilized summary flow.

## Assumption Ledger

- Canary success criteria [missing]: The repo evidence does not define the exact canary pass or rollback thresholds for RN-103, so the plan assumes those criteria still need to be formalized before launch.
- Dashboard consumer tolerance [to_verify]: Because the dashboard is explicitly user-visible, the plan assumes any ordering error is externally visible enough to justify blocking summary rollout on schema and fixture work.
- Abandoned draft ownership [observed]: The in-progress patch was started by an engineer who transferred teams, so the draft is treated as inherited cleanup rather than an active delivery path.

