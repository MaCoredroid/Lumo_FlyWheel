# Objective Delta Brief

- Variant: `v5-recovery-in-thread`
- Accepted intervention: `P2`
- Expected delta: `4..6` points up (medium confidence)

## Ranking

1. `P2` - P2 is the correct recovery choice because it advances the reliability objective without reselecting the P5 path that was rolled back in the incident packet.
   - Guardrails: incident, objective, rollout
   - Citations: allowed_interventions/P2.md, incident_context/rollback_2026_07_P5.md, incident_context/prior_decision.md, release_context/priority_shift_2026_06.md
2. `P1` - P1 remains legal and still compounds with the landed checkpoint, but after the rollback it is less aligned to the active reliability recovery objective than P2.
   - Guardrails: objective, governance
   - Citations: allowed_interventions/P1.md, repo_snapshot/changes/already_landed_foundation.md, incident_context/prior_decision.md
3. `P3` - P3 is still dependency-heavy and remains a sunk-cost trap after the rollback.
   - Guardrails: dependency, regression, incident
   - Citations: allowed_interventions/P3.md, repo_snapshot/in_progress_patch/README.md, incident_context/compatibility_gap_notes.md
4. `P5` - P5 must be demoted because the incident packet says it was rolled back for a compatibility bug with the landed checkpoint.
   - Guardrails: incident, regression, objective
   - Citations: allowed_interventions/P5.md, incident_context/rollback_2026_07_P5.md, incident_context/compatibility_gap_notes.md
5. `P4` - P4 remains governance-blocked and would magnify the same recovery risk if rushed during the freeze.
   - Guardrails: governance, incident, regression
   - Citations: allowed_interventions/P4.md, repo_snapshot/governance_window.md, incident_context/rollback_2026_07_P5.md

## Primary Risk

P2 now inherits the reliability objective after the P5 rollback, so the main risk is under-specifying the compatibility gate and repeating the same failure mode through a different path.

Mitigations:
- checkpoint the compatibility gate before any rollout
- canary the successor path with rollback ready
- shadow compare error buckets against the post-incident baseline

## Assumption Ledger

- `observed` P5 rollback scope: The incident write-up explicitly names the compatibility bug between P5 and the landed checkpoint change.
- `to_verify` P2 successor gate coverage: Need to confirm the gate covers the same tenant mix that triggered the rollback.
- `missing` Who signs off if the successor also regresses on Friday: The incident packet stops at technical mitigation and does not name the business approver.
