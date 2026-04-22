# Objective Delta Brief

- Variant: `v1-clean-baseline`
- Accepted intervention: `P1`
- Expected delta: `7..9` points up (medium confidence)

## Ranking

1. `P1` - P1 compounds with the already-landed normalization checkpoint and is the highest legal delta inside the current governance window.
   - Guardrails: governance, objective, rollout
   - Citations: allowed_interventions/P1.md, repo_snapshot/changes/already_landed_foundation.md, repo_snapshot/governance_window.md, objective_history/objective_delta_index.md
2. `P2` - P2 is safe and reversible but yields a smaller immediate objective move because it waits on a second guardrail lane before paying back latency.
   - Guardrails: dependency, rollout, objective
   - Citations: allowed_interventions/P2.md, repo_snapshot/dependency_map.md, objective_history/objective_delta_index.md
3. `P3` - P3 has raw upside but needs the speculative optimizer branch to be revived and is too wide for the current checkpoint-first objective window.
   - Guardrails: dependency, regression, objective
   - Citations: allowed_interventions/P3.md, repo_snapshot/dependency_map.md, repo_snapshot/risk_register.md
4. `P4` - P4 is blocked outright because the governance freeze forbids turning on the unsafe global fast-path until the post-freeze audit is complete.
   - Guardrails: governance, regression
   - Citations: allowed_interventions/P4.md, repo_snapshot/governance_window.md

## Primary Risk

The checkpoint adapter can hide state-shape regressions if it is rolled out before parity telemetry confirms the landed normalization path is stable.

Mitigations:
- feature flag the adapter behind the existing rollout gate
- shadow compare checkpoint output before increasing traffic
- canary on a single tenant until parity stays green for 24h

## Assumption Ledger

- `observed` Normalization checkpoint coverage: The landed checkpoint covers the top two objective-loss callsites from the last three weeks.
- `to_verify` Tenant skew after adapter enablement: Need one canary window to confirm low-volume tenants do not diverge.
- `missing` Rollback ownership on Friday freeze: The governance memo does not name the exact weekend rollback owner.
