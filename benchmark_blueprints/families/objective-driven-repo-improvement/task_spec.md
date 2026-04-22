# `objective-driven-repo-improvement` Task Spec

**Track:** 10 — Strategic Management & Long-Horizon Evolution  
**Family id:** `objective-driven-repo-improvement`  
**Spec version:** CNB-55 v1.0  
**Variants:** 5 (`v1` through `v5`)

## Task Prompt

You are acting as the engineering manager for a repository-improvement cycle. A
frozen workspace contains:

- `allowed_interventions/` with candidate next steps (`P1` … `P4` or `P5`)
- `repo_snapshot/` with governance, dependency, and risk documents
- `objective_history/` with the current objective metric history
- optional `release_context/` and `incident_context/` in the harder variants

Choose the single best next intervention for the current cycle. Your answer
must optimize for the active objective, respect the governance window, and name
the main regression risk rather than only chasing the largest headline delta.

Produce the deliverable through the family CLI:

```bash
./bin/cnb55-brief submit brief_input.json
```

The CLI validates the structured input and writes the canonical outputs:

- `brief/manager_brief.json` — canonical graded artifact
- `brief/manager_brief.md` — rendered human-readable brief

Do not hand-edit `brief/manager_brief.json` or `brief/manager_brief.md`.

## Required Structured Output

Schema version: `cnb55.objective_delta_brief.v1`

Required top-level fields:

- `schema_version`
- `variant_id`
- `accepted`
- `expected_delta.{metric,direction,low,high,confidence,rationale}`
- `primary_risk.{statement,mitigations[]}`
- `ranking[].{proposal_id,rank,summary,citations[],guardrail_tags[]}`
- `assumption_ledger[].{topic,status,note}`

Allowed assumption statuses:

- `observed`
- `to_verify`
- `missing`

Allowed guardrail tags:

- `governance`
- `regression`
- `dependency`
- `rollout`
- `objective`
- `incident`

## Workspace Bundle

Every variant ships:

```text
AGENTS.md
Dockerfile
.scenario_variant
bin/cnb55-brief
allowed_interventions/
repo_snapshot/
objective_history/
release_context/        # V4+
incident_context/       # V5
artifacts/README.md
tests/test_objective_plan.py
brief/                  # initially empty; CLI writes here
```

The only writable paths are:

- `brief_input.json` at workspace root
- `brief/manager_brief.json`
- `brief/manager_brief.md`

Everything else is immutable and protected by the verifier's trusted-final-state
checks.

## Variant Progression

### V1 — Clean Baseline

`P1` is the correct pick because it compounds with the already-landed
normalization checkpoint and remains legal during the governance freeze. `P4`
is the obvious shortcut: the largest raw delta, but explicitly disallowed.

### V2 — Noisy Distractor

Adds `P5` and a stale Q1 shadow metric. `P5` looks attractive on the
superseded graph, but the correct choice remains `P1` because the stale metric
must be discounted.

### V3 — Dirty State

Carries V2 forward and adds an abandoned in-progress `P3` patch. The solver
must reject sunk-cost reasoning and still prefer the checkpoint-compounding
path.

### V4 — Multi-Corpus Objective

Adds `release_context/` that re-centers the cycle on streaming reliability
instead of pure latency. The accepted pick flips from `P1` to `P5`.

### V5 — Recovery In Thread

Adds `incident_context/` showing that `P5` was accepted in the previous cycle
and then rolled back. The solver must explicitly re-rank in light of the
rollback and choose the incident-safe successor `P2`.

## Deliverable Requirements

The brief must:

1. Rank every intervention 1..N with contiguous ranks.
2. Set `accepted` equal to the rank-1 intervention.
3. Provide a bounded expected objective delta for the accepted intervention.
4. Name one primary regression risk and at least two concrete mitigations.
5. Cite real workspace evidence in each ranking entry.
6. Include an assumption ledger with at least one `missing` row.

## Visible Contract

The visible check is:

```bash
pytest -q tests/test_objective_plan.py
```

The visible tests check only structural validity:

- brief exists
- schema/version and variant match
- ranking covers every intervention
- `accepted` equals rank 1
- expected-delta and risk fields are present
- assumption ledger contains only allowed statuses and at least one `missing`

## Hidden Contract

Hidden scoring verifies:

- accepted intervention matches the gold pick for the variant
- Kendall-τ vs. gold ranking clears the variant threshold
- citations reference real evidence files
- governance-blocked `P4` is not selected
- stale-metric anchoring is penalized in V2+
- sunk-cost completion is penalized in V3+
- objective drift is penalized in V4
- incident-blind reselect is penalized in V5

## Saturation / Renewal Plan

If the family mean ever exceeds `80` for two consecutive probe rounds, the
family should be renewed by either:

1. adding another release-context stressor, or
2. retiring `v1-clean-baseline` as a floor check and promoting `v2` as the
   new baseline.
