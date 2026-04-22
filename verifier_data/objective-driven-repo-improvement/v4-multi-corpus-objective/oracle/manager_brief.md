# Objective Delta Brief

- Variant: `v4-multi-corpus-objective`
- Accepted intervention: `P5`
- Expected delta: `5..7` points up (medium confidence)

## Ranking

1. `P5` - P5 moves to the top because the release context redefines success around streaming reliability rather than pure latency, and P5 addresses that objective directly.
   - Guardrails: objective, rollout, incident
   - Citations: allowed_interventions/P5.md, release_context/priority_shift_2026_06.md, release_context/open_reliability_incidents.md, objective_history/objective_delta_index.md
2. `P2` - P2 is the runner-up reliability move: safer than P3 and more aligned to the new objective than P1, but smaller immediate delta than P5.
   - Guardrails: dependency, objective, rollout
   - Citations: allowed_interventions/P2.md, release_context/priority_shift_2026_06.md, repo_snapshot/dependency_map.md
3. `P1` - P1 was the old answer, but under the new reliability objective it optimizes the wrong thing even though it still compounds with the landed checkpoint.
   - Guardrails: objective, governance
   - Citations: allowed_interventions/P1.md, repo_snapshot/changes/already_landed_foundation.md, release_context/priority_shift_2026_06.md
4. `P3` - P3 remains a dependency-heavy sunk-cost path and still does not target the new objective cleanly.
   - Guardrails: dependency, regression, objective
   - Citations: allowed_interventions/P3.md, repo_snapshot/in_progress_patch/README.md, release_context/open_reliability_incidents.md
5. `P4` - P4 is still governance-blocked and would worsen the incident posture if rushed during the freeze.
   - Guardrails: governance, incident, regression
   - Citations: allowed_interventions/P4.md, repo_snapshot/governance_window.md, release_context/open_reliability_incidents.md

## Primary Risk

P5 changes the objective hierarchy rather than the old latency-first plan, so the main risk is rolling it out without proving the new reliability holdback does not starve the checkpoint path.

Mitigations:
- staged rollout with a holdback cohort on the old path
- shadow compare reliability error buckets before widening
- kill switch back to the checkpoint-only path on incident regression

## Assumption Ledger

- `observed` Priority shift recency: The release memo is newer than the base objective window and references live customer-impacting reliability incidents.
- `to_verify` Holdback cohort sizing: The release note names the holdback but not the exact percentage.
- `missing` Final rollback owner if reliability slips again: The release context does not name the final escalation owner for a weekend rollback.
