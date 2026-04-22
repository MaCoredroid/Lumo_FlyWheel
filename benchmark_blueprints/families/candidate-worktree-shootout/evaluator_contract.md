# `candidate-worktree-shootout` Evaluator Contract

**Track:** 11 — Subagents Orchestration
**Spec conformance:** CNB-55 v1.0 §7
**Scorer:** `verifiers/candidate-worktree-shootout/score_shootout.py`

## Evaluation Goal

Score whether the solver:

1. evaluates both candidate fixes in isolated worktree directories,
2. rejects the shallow CLI-local fix with evidence,
3. lands the shared service-layer repair cleanly,
4. adds a service-level regression test,
5. keeps the final workspace coherent and invariant-respecting.

## Visible Checks — 30 Points Max

Visible checks stay under the CNB-55 30-point budget.

| Check | Points |
| --- | ---: |
| `candidate_a_eval.md` exists with the required sections | 7 |
| `candidate_b_eval.md` exists with the required sections | 7 |
| `comparison_summary.md` and `verification_note.md` both exist | 4 |
| Candidate A and Candidate B use distinct worktree paths | 4 |
| `python -m pytest -q tests/test_cli.py` passes | 8 |

The scorer treats missing worktree paths, missing command lists, or missing
observed results as a structural failure, not just a small formatting miss.

## Hidden Checks — 70 Points

| Check | Points |
| --- | ---: |
| direct service callers are fixed (`service.compile_filters(...)`) | 20 |
| `comparison_summary.md` selects Candidate B and rejects Candidate A | 8 |
| `tests/test_service.py` contains the separator-heavy regression | 10 |
| `docs/filter_contract.md` explicitly restores service-layer ownership | 6 |
| Candidate A evaluation names the direct-caller failure | 6 |
| Candidate B evaluation names the shared-owner repair and touched files | 6 |
| `cli.py` does not duplicate `normalize_label(...)` after the final fix | 4 |
| `verification_note.md` records the final full validation command | 4 |
| variant-specific reasoning handles the active distractor / state / objective / incident correctly | 6 |

## Partial-Credit Ceilings

These are hard caps applied after point aggregation.

| Ceiling | Max Score | Trigger |
| --- | ---: | --- |
| `missing_worktree_proof` | 20 | one or both candidate evaluation files lack worktree path, commands, touched files, or observed results |
| `cli_only_fix` | 25 | visible CLI path is repaired but direct service callers still fail |
| `no_service_regression_test` | 35 | no separator-heavy service regression exists in `tests/test_service.py` |
| `blended_ownership` | 40 | both the service layer and `cli.py` normalize labels, leaving duplicate ownership |
| `docs_drift` | 55 | docs do not explicitly restore service-layer ownership |
| `anchored_on_stale_cli_memo` | 45 | V2 summary ignores the archived memo's staleness |
| `sunk_cost_finish` | 40 | V3 reasoning treats abandoned CLI-local partial work as a head start |
| `objective_drift` | 45 | V4 summary fails to reflect the importer release blocker |
| `incident_blind_reselect` | 30 | V5 reasoning ignores the prior rollback / incident context |

## Trusted Final-State Checks

These checks force `integrity_flag = 1` and `score = 0` when tripped.

| Check | Consequence |
| --- | --- |
| immutable trees drift (`AGENTS.md`, `normalization.py`, evidence trees, visible tests) | `integrity_flag = 1` |
| `tests/test_cli.py` hash changes | `integrity_flag = 1` |
| `sitecustomize.py`, `usercustomize.py`, or `pytest.py` appears | `integrity_flag = 1` |
| files are written outside the allowed mutable surfaces or `artifacts/comparison/*` | `integrity_flag = 1` |
| `.network_egress_detected` exists | `integrity_flag = 1` |

## Layer B Emission

The scorer emits:

- `P_benchmark`
- `M_training`
- `score` aliasing `P_benchmark`
- milestone booleans `M1` through `M5`
- `milestone_vector`
- integrity flags
- ceiling list

This family keeps `llm_judge_quarantine.total_quarantined_points = 0`; all
points are deterministic today.

## Deterministic Baselines

Measured against the current family-local scorer:

| Trajectory | Expected Score |
| --- | ---: |
| Oracle overlay | 100 |
| Empty workspace | 0 |
| CLI-local shortcut | 25 |
| Grounding stripped | 20 |
| Blended ownership | 40 |
| Delete-tests adversarial | 0 with `integrity_flag = 1` |

The stress verification matrix for V5 additionally confirms the incident-aware
ceiling fires on weak trajectories; see
[`verification_matrix_v5.md`](./verification_matrix_v5.md).
