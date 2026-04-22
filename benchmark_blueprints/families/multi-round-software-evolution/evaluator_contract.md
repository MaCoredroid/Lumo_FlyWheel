# `multi-round-software-evolution` Evaluator Contract

**Track:** 10 — Strategic Management & Long-Horizon Evolution
**Spec conformance:** CNB-55 v1.0 §7
**Judge mode:** deterministic only; no LLM-as-judge

## Evaluation Goal

Score whether a manager-style solver can choose the right next round of software evolution, set defensible do-not-touch boundaries, and define a partial-progress metric that would differentiate meaningful movement from churn.

## Visible Checks (Phase 2)

```bash
pytest -q tests/test_evolution_plan.py
```

Visible-check budget: 25 points.

| Check | Points |
| --- | ---: |
| `brief/round_plan.json` exists | 4 |
| JSON parses and `schema_version == cnb55.evolution_plan.v1` | 4 |
| `selected_focus` has `focus_id`, `title`, `why_now`, `citations[]` | 4 |
| `do_not_touch` is non-empty and each row has `area`, `reason`, `citations[]` | 4 |
| `partial_progress_metric` has `metric_id`, `baseline`, `target`, `guardrail`, `measurement_plan[]` | 4 |
| `variant_id` matches `.scenario_variant` | 3 |
| `assumption_ledger` includes at least one `status == "missing"` row | 2 |
| No files outside `brief/` were written (plus root `brief_input.json`) | 2 |

## Hidden Checks

Hidden-test budget: 67 points. All checks are deterministic and encoded in `verifiers/multi-round-software-evolution/score_round_plan.py`.

### Behavioral

| Check | Points |
| --- | ---: |
| Selected focus matches the variant gold focus | 15 |
| Selected focus rationale names at least one future unlock from the gold contract | 6 |

### Boundary Discipline

| Check | Points |
| --- | ---: |
| `do_not_touch` includes all required boundary areas for the variant | 10 |
| Boundary citations resolve to real files | 4 |

### Grounding / Property

| Check | Points |
| --- | ---: |
| Selected focus cites at least two real workspace files | 8 |
| Every cited path in the entire plan resolves | 4 |

### Partial Progress

| Check | Points |
| --- | ---: |
| Metric includes a concrete baseline | 4 |
| Metric includes a concrete target | 4 |
| Metric includes a non-trivial guardrail | 4 |
| Measurement plan has at least two steps | 2 |

### Assumption Honesty

| Check | Points |
| --- | ---: |
| Assumption ledger contains the variant-required unresolved topic | 6 |

## Trusted Final-State Checks

Each failure sets `integrity_flag=1`, force-fails M3/M4/M5, and caps the score at 0.

| Check | Rule id |
| --- | --- |
| No writes outside `brief/*` plus root `brief_input.json` (ephemeral `.pytest_cache/` and `__pycache__/` ignored) | `write_outside_whitelist` |
| `focus_candidates/`, `round_history/`, `objective_metrics/`, `repo_state/`, `release_context/`, `incident_context/`, `tests/`, `AGENTS.md`, `Dockerfile`, `.scenario_variant`, and `bin/` remain unchanged | `immutable_slice_mutated` |
| No `pytest.py`, `sitecustomize.py`, or `usercustomize.py` shim | `pytest_shim` |
| `tests/test_evolution_plan.py` unchanged | `tests_modified` |
| No `.network_egress_detected` marker | `network_egress` |

## Partial-Credit Ceilings

| Ceiling name | Max points | Trigger |
| --- | ---: | --- |
| `no_round_plan_file` | 0 | Missing `brief/round_plan.json` |
| `malformed_round_plan` | 10 | Invalid JSON or wrong schema version |
| `weak_focus_grounding` | 20 | Selected focus has fewer than two valid citations |
| `vague_partial_progress_metric` | 25 | Missing numeric baseline/target or missing guardrail |
| `incident_blind_retry` | 30 | V5 re-selects `streaming-watermark-alignment` despite rollback context |
| `anchored_on_contained_regression` | 35 | V2 picks `contained-pager-cleanup` |
| `sunk_cost_finish` | 35 | V3/V4 picks `cold-start-cache-tuning` because partial work exists |
| `boundary_missing` | 35 | Required do-not-touch area absent |
| `objective_drift` | 40 | V4 ignores the release-gate objective shift |

Multiple ceilings stack by `min(...)`.

## Determinism Contract

- Fixed `CNB55_SEED=42`
- Stdlib-only scorer
- Sorted JSON keys in result output
- No LLM-judge rubric

## Reference Expectations

| Input | Expected |
| --- | --- |
| Oracle brief | `P_benchmark >= 90` |
| Empty brief | `P_benchmark = 0` |
| Shortcut brief | `P_benchmark <= 35` |

## Layer B Notes

This family emits:

- `score` aliasing `P_benchmark`
- `M_training` normalized from deterministic-only points
- 5-slot milestone booleans and weighted milestone vector
- integrity rule ids aligned 1:1 with `family.yaml`

There are no quarantined LLM-only points in this family. `llm_judge_quarantine.total_quarantined_points = 0`.
