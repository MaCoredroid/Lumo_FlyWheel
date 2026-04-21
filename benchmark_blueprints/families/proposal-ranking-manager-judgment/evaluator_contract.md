# `proposal-ranking-manager-judgment` Evaluator Contract

**Track:** 10 — Strategic Management & Long-Horizon Evolution
**Spec conformance:** CNB-55 v1.0 §7
**Judge model for LLM-as-judge rubrics:** `gpt-5.4`, `temperature=0.0`, `reasoning_effort=high`

## Evaluation Goal

Score whether a manager-framed agent can choose correctly among several plausible implementation proposals and defend the choice with evidence-grounded reasoning, under varying pressures (noise, dirty state, multi-corpus objective, in-flight recovery).

## Visible Checks (Phase 2)

```
pytest -q tests/test_ranking_brief.py
```

Visible-check budget: ≤ 30 points of the 100-point total. Allocated as:

| Check | Points |
| --- | --- |
| `brief/manager_brief.json` exists (canonical CLI output) | 3 |
| `brief/manager_brief.json` parses as JSON with `schema_version == cnb55.manager_brief.v2` | 5 |
| `ranking` length matches variant (4 for V1, 5 for V2+) | 4 |
| `accepted` resolves to a valid proposal id | 3 |
| Each ranking entry has `proposal_id`, `rank`, `summary`, `citations`, `constraint_tags` | 3 |
| `assumption_ledger` present, ≥ 1 row with `status: "missing"` | 3 |
| No files modified outside `brief/` (attempt_02b hardening: trust bonus reduced 4 → 3) | 3 |
| No disallowed files created (`tests/` checksum, no `sitecustomize.py`) (attempt_02b: 5 → 3) | 3 |

## Hidden Checks

Hidden-test budget: ≥ 50 points. Mounted from `verifier_data/proposal-ranking-manager-judgment/<variant_id>/hidden_tests/`.

### Behavioral

| Check | Points |
| --- | --- |
| Accepted proposal matches gold `accepted` | 10 |
| Top-2 set matches gold top-2 set (unordered) | 4 |

### Differential

| Check | Points |
| --- | --- |
| Kendall τ between submitted ranking and gold ranking ≥ variant threshold | 8 |
| Rank of staffing-blocked proposal ≥ gold rank (i.e., not promoted) | 4 |

τ threshold per variant (attempt_02b hardening): V1 ≥ 0.67; V2–V5 ≥ 0.80.
(N=4 in V1 admits discrete τ values {1.0, 0.67, 0.33, …}; N=5 in V2–V5
admits {1.0, 0.80, 0.60, …}. The thresholds translate to at most one
inversion against gold.)

### Property-based

| Check | Points |
| --- | --- |
| Each rejection rationale cites an existing evidence file | 6 |
| Each rejection rationale references ≥ 1 constraint tag from {staffing, rollout, regression, perf, release, incident} | 4 |
| Accepted proposal's `primary_risk.statement` ≥ 40 chars (attempt_02b: raised from 20) | 3 |
| Accepted proposal's ranking entry lists ≥ 3 citations (attempt_02b grounding depth) | 3 |

### Regression / Collateral

| Check | Points |
| --- | --- |
| No brief citation points to a non-existent file or proposal id | 3 |
| Assumption-ledger `missing` rows are real gaps (not padding) — LLM-judge | 3 |

### LLM-judge rubric (partial-progress sub-metric)

Judge prompt lives at `verifier_data/proposal-ranking-manager-judgment/_rubrics/partial_progress.md`. Scores 0–10 against rubric covering:

- Quality of rejection reasoning for the *hardest* close call (2–3 pairs per variant).
- Coherence of risk + mitigation pairing for the accepted proposal.
- Honest handling of contradictory evidence (V2+).
- Honest reframing under objective drift (V4+).
- Incident anchoring (V5).

Judge model: `gpt-5.4` at `temperature=0.0`. Two-seed average.

## Trusted Final-State Checks (pass/fail gates)

Each failure triggers a ceiling and flags `shortcut_detected=true`.

| Check | Consequence on failure |
| --- | --- |
| `tests/test_ranking_brief.py` checksum unchanged | score → 0, shortcut_detected |
| No `sitecustomize.py`, `usercustomize.py`, `pytest.py` shim | score → 0, shortcut_detected |
| `proposals/`, `repo_evidence/`, `release_context/`, `incident_context/` checksums unchanged | score → 0, shortcut_detected |
| No files outside `brief/` were written (allowed: `brief/*` plus `brief_input.json` at workspace root) | score → 0, shortcut_detected |
| No network egress during run | score → 0, shortcut_detected |

## 100-Point Breakdown (Track 10 scorecard)

Official allocation. Score aggregator merges Phase 2 + hidden + trusted subject to ceilings.

| Sub-metric | Max points | Encoded via |
| --- | --- | --- |
| Proposal ranking / decision quality | 20 | Kendall τ + accepted-match behavioral |
| Objective delta | 20 | V4+: objective-match behavioral. V1–V3: reallocated to Ranking +10, Plan +10 (see §7.3 note in SPEC). |
| Regression-free change | 20 | V5: incident-anchoring behavioral + rejection of rolled-back proposal. V1–V4: reallocated to Maintainability +10, Rejection Quality +10. |
| Maintainability / slop control | 15 | V3+: sunk-cost resistance property check. V1–V2: reallocated to Rejection Quality. |
| Plan / dependency correctness | 15 | Risk+mitigation property + LLM-judge coherence |
| Partial progress | 10 | LLM-judge rubric |

Subtotal before Phase 2: 100. Phase 2 pulls 15 points in (from ranking +5, maintainability +5, plan +5) to anchor visible-check weight, with reallocation documented in the scorer.

## Partial-Credit Ceilings (hard caps, applied after aggregation)

Copied from `task_spec.md` §Partial-Credit Ceilings. Encoded in the scorer `verifiers/proposal-ranking-manager-judgment/score_ranking.py`. Ceilings cap the final score non-negotiably; they are not soft weights.

| Ceiling name | Max points | Trigger |
| --- | --- | --- |
| `no_brief_file` | 0 | `brief/manager_brief.json` missing (agent must run `./bin/cnb55-brief submit`) |
| `malformed_brief` | 10 | `brief/manager_brief.json` not valid JSON or wrong `schema_version` |
| `ranking_without_grounding` | 25 | < 2 rejection entries have ≥ 1 `citations` entry that resolves to a real evidence file |
| `ignored_stale_perf` | **25** | V2+: accepted entry cites a `stale_perf_marker` path without the SAME entry's summary OR `primary_risk` flagging staleness (attempt_02b tightened from 35; ledger aside is insufficient) |
| `ignored_staffing_constraint` | 30 | accepted = staffing-blocked (P3 in V1–V3) |
| `incident_blind_reselect` | 30 | V5: accepted = rolled-back proposal without incident-grounded rationale |
| `sunk_cost_finish` | **30** | V3+: accepted = in-progress partial proposal without maintainability/abandoned/sunk-cost/in-progress/"not a head start" language (attempt_02b tightened from 40) |
| `missed_staffing_update` | **40** | **attempt_02b NEW**. Accepted proposal is owned by the engineer whose availability changed in the mid-quarter staffing update (gold key `staffing_update_owner_proposal` = P4/Priya every variant) AND brief neither cites `staffing_update_2026_06_15.md` nor uses an update keyword (`40%`, `parental`, `reduced availability`, `mid-quarter`, `handoff`, `ramp-down`, `priya`) |
| `objective_drift` | 45 | V4+: accepted solves wrong objective |
| `missed_watermark_assumption` | **55** | **attempt_02b NEW, V5 only**. No `assumption_ledger` row has both `status: "missing"` AND `watermark` in topic or note |

Multiple ceilings stack as `min(…)`.

## Determinism Contract

- Scorer deterministic under fixed `CNB55_SEED=42`.
- LLM-judge component seeded via `temperature=0.0` and a deterministic prompt; score reported as the two-seed average to damp remaining noise.
- Hidden-test invocation order fixed.
- JSON output keys sorted.

## Oracle and Empty Reference Scores (used in the CNB-55 §10.1 freeze gate)

| Input | Expected score range |
| --- | --- |
| Oracle brief (canonical `verifier_data/.../oracle/manager_brief.json` produced by `cnb55-brief submit` against `verifier_data/.../oracle/brief_input.json`) | ≥ 90 (observed: V1=90 / V2=90 / V3=90 / V4=96 / V5=99 on 2026-04-20) |
| Empty `brief/` (no file) | 0 |
| Submitted brief whose rejection entries have no real citation paths | ≤ 25 (ceiling) |
| "Pick P3" shortcut brief in V1–V3 (accepts staffing-blocked proposal) | ≤ 30 (ceiling; observed 30/30/30 on 2026-04-20) |
| "Finish partial" brief in V3–V5 | ≤ 40 (ceiling) |
