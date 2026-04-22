# Objective Delta Brief

- Variant: `v2-noisy-distractor`
- Accepted intervention: `P1`
- Expected delta: `7..9` points up (medium confidence)

## Ranking

1. `P1` - P1 remains the best legal step because it compounds with the landed checkpoint and uses current-window metrics rather than the superseded Q1 shadow graph.
   - Guardrails: governance, objective, rollout
   - Citations: allowed_interventions/P1.md, repo_snapshot/changes/already_landed_foundation.md, objective_history/objective_delta_index.md, objective_history/q1_shadow_metric.md
2. `P2` - P2 is still second: safer than P5 but slower than P1 because it waits on the extra guardrail lane before moving the objective.
   - Guardrails: dependency, rollout, objective
   - Citations: allowed_interventions/P2.md, repo_snapshot/dependency_map.md, objective_history/objective_delta_index.md
3. `P5` - P5 looks attractive on the stale Q1 shadow graph, but that graph predates the landed checkpoint and overstates the current delta.
   - Guardrails: objective, regression
   - Citations: allowed_interventions/P5.md, objective_history/q1_shadow_metric.md, objective_history/objective_delta_index.md
4. `P3` - P3 still depends on reviving the speculative branch and remains wider than the current objective window needs.
   - Guardrails: dependency, regression
   - Citations: allowed_interventions/P3.md, repo_snapshot/dependency_map.md
5. `P4` - P4 is still governance-blocked regardless of the headline delta in the fast-path note.
   - Guardrails: governance, regression
   - Citations: allowed_interventions/P4.md, repo_snapshot/governance_window.md

## Primary Risk

P1 still depends on parity telemetry, and anchoring on the stale Q1 shadow graph could cause the team to over-ramp before the current window proves the adapter is stable.

Mitigations:
- shadow compare against the current window not the stale Q1 run
- holdback traffic behind a feature flag until parity settles
- rollback on any checkpoint divergence alert

## Assumption Ledger

- `observed` Freshness of Q1 shadow graph: The graph is explicitly marked superseded by the April objective window.
- `to_verify` Second guardrail lane for P2: Dependency map says one reviewer is shared with the release branch.
- `missing` Exact owner for post-freeze audit on P4: The governance memo lists the audit as required but not who signs it.
