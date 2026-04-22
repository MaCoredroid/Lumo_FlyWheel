# Objective Delta Brief

- Variant: `v3-dirty-state`
- Accepted intervention: `P1`
- Expected delta: `6..8` points up (medium confidence)

## Ranking

1. `P1` - P1 remains the right move because it compounds with the landed checkpoint and avoids treating the abandoned P3 patch as a free head start.
   - Guardrails: governance, objective, rollout
   - Citations: allowed_interventions/P1.md, repo_snapshot/changes/already_landed_foundation.md, repo_snapshot/in_progress_patch/README.md, objective_history/objective_delta_index.md
2. `P2` - P2 is still viable but slower; it is cleaner than finishing the abandoned patch yet does not compound as directly as P1.
   - Guardrails: dependency, objective, rollout
   - Citations: allowed_interventions/P2.md, repo_snapshot/dependency_map.md, repo_snapshot/in_progress_patch/README.md
3. `P5` - P5 still leans on the superseded Q1 shadow graph, so it stays behind the current-window checkpoint path.
   - Guardrails: objective, regression
   - Citations: allowed_interventions/P5.md, objective_history/q1_shadow_metric.md, objective_history/objective_delta_index.md
4. `P3` - P3 is now an explicit sunk-cost trap: the patch is abandoned, unreviewed, and still blocked by missing dependency ownership.
   - Guardrails: dependency, regression
   - Citations: allowed_interventions/P3.md, repo_snapshot/in_progress_patch/README.md, repo_snapshot/dependency_map.md
5. `P4` - P4 remains frozen by governance regardless of the abandoned patch nearby.
   - Guardrails: governance, regression
   - Citations: allowed_interventions/P4.md, repo_snapshot/governance_window.md

## Primary Risk

Leaving the abandoned optimizer patch in the tree can bias future discussions, so the main risk is accidental re-anchoring on the sunk-cost path after P1 starts to land.

Mitigations:
- document that the in-progress P3 patch is explicitly out of scope for this window
- checkpoint P1 behind a feature flag so it does not inherit P3 assumptions
- rollback if parity or dependency ownership diverges during canary

## Assumption Ledger

- `observed` P3 patch validity: The notes under repo_snapshot/in_progress_patch explicitly call the patch abandoned.
- `to_verify` Ownership for follow-up cleanup: Need to confirm who archives the abandoned branch after this cycle.
- `missing` Dependency review slot after freeze: There is no named reviewer for reviving P3 next cycle.
