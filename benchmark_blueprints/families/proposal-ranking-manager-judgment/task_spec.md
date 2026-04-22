# `proposal-ranking-manager-judgment` Task Spec

**Track:** 10 — Strategic Management & Long-Horizon Evolution
**Family id:** `proposal-ranking-manager-judgment`
**Spec version:** CNB-55 v1.0
**Variants:** 5 (v1 through v5)

## Task Prompt (canonical)

You are acting as the engineering manager for a mid-sized service team. Four engineers have each written a one-pager implementation proposal for the same repo objective. The repository, the release-note history, a test inventory, a staffing sheet, a rollout-risk log, and per-proposal perf notes are in the workspace.

Produce a **manager brief** using the family's structured-output CLI. The agent writes a small JSON file and runs `./bin/cnb55-brief submit brief_input.json`; the CLI validates the input and then writes the canonical `brief/manager_brief.json` (read by the grader) and `brief/manager_brief.md` (human rendering).

The brief must:

1. Rank all proposals 1 → N (best to worst) with contiguous integer ranks.
2. Select one proposal as the accepted choice (must equal the rank=1 entry).
3. Attach a **rejection rationale** (the `summary` + `citations` + `constraint_tags` fields) for each non-selected proposal, citing at least one evidence file under `proposals/`, `repo_evidence/`, `release_context/`, or `incident_context/`, and at least one constraint tag from {staffing, rollout, regression, perf, release, incident}.
4. State the primary rollout risk of the accepted proposal (`primary_risk.statement`) and how to mitigate it (`primary_risk.mitigations` — at least two of feature flag, kill switch, staged rollout, shadow replay, canary, observability/SLO, rollback, pre-warm).
5. Include an **assumption ledger** separating observed facts from missing inputs. At least one row must be `status: "missing"`.

The brief must be grounded in `repo_evidence/` and the proposal files themselves. Inventing facts that aren't in the workspace is a scoreable failure. Hand-writing `brief/manager_brief.md` is not supported — the CLI is the only path to a graded submission.

### Required structured-output CLI

Location: `./bin/cnb55-brief` (shipped with every variant).

```
./bin/cnb55-brief schema             # prints the JSON schema
./bin/cnb55-brief validate FILE.json # dry-run; exits non-zero on error
./bin/cnb55-brief submit FILE.json   # validates and writes brief/manager_brief.{json,md}
```

Schema version: `cnb55.manager_brief.v2`. Required top-level fields: `schema_version`, `variant_id`, `accepted`, `primary_risk.{statement,mitigations[]}`, `ranking[].{proposal_id,rank,summary,citations[],constraint_tags[]}`, `assumption_ledger[].{topic,status,note}`. See the full schema via `cnb55-brief schema`.

## Scenario Type

`strategic_management` — per `benchmark_deisgn.md`, Track 10 scorecard (proposal ranking 20 / objective delta 20 / regression-free 20 / maintainability 15 / plan/dependency 15 / partial-progress 10).

## Required Surfaces

- `shell` (read-only filesystem traversal, plus executing `./bin/cnb55-brief`)
- `apply_patch` (agent writes `brief_input.json` and relies on the CLI to populate `brief/`)
- evidence review (reading `repo_evidence/` and `proposals/`)
- strategic ranking and concise decision writing

No MCP, no network, no subagents, no browser.

## Workspace Bundle (per variant)

Every variant ships the following under `workspace_bundle/<variant_id>/`:

```
AGENTS.md                       # visible agent instructions
Dockerfile                      # pinned base image
.scenario_variant               # variant_id marker
bin/cnb55-brief                 # family-specific structured-output CLI
proposals/                      # P1.md .. P5.md (varies by variant)
repo_evidence/                  # code excerpts, perf notes, staffing, rollout history
release_context/                # release notes, test inventory (V4+ only)
incident_context/               # incident reports, rollback log (V5 only)
artifacts/                      # human-readable summary only; gold ranking is NOT shipped
tests/test_ranking_brief.py     # Phase 2 visible tests
brief/                          # empty; CLI populates manager_brief.{json,md} here
```

The `artifacts/gold_ranking.json` is NOT shipped to the agent. It lives under `verifier_data/` only.

## Difficulty Ladder

Per CNB-55 SPEC §4, variants are ordered difficulty rungs with justified gradient.

### v1 — clean-baseline

Four proposals (P1–P4), clean repo_evidence, one unambiguous objective (latency on the `/v1/responses` tool-call proxy path). Proposals have orthogonal tradeoffs: P1 fastest-to-ship / riskiest; P2 safest / slowest; P3 best-delta / specialist-only (staffing-blocked); P4 status-quo incremental. Gold ranking: the manager picks the proposal that balances delta against staffing.

- **Primary stress:** proposal ranking & decision quality (track scorecard sub-metrics 1 + 2).
- **Expected probe score:** 25–30 on GPT-5.4/high.
- **What's harder than nothing:** the fastest proposal is not the safest; the highest-delta one is staffing-blocked; frontier models reliably over-index on raw delta.

### v2 — noisy-distractor

V1 plus a fifth proposal (P5) that looks attractive on surface metrics but contradicts the staffing sheet and references a deprecated module. Conflicting perf notes in repo_evidence: two measurements of the hot path disagree by 3×; one is stale.

- **Primary stress:** evidence discrimination + rejection reasoning.
- **Expected probe score:** 18–24.
- **What's harder than V1:** the distractor P5 has the best headline numbers but the stale perf note is the source; staffing sheet has a footnote that rules P5 out.
- **Gradient justification:** Adds 1 distractor proposal + 1 stale perf measurement + 1 footnote constraint. Strictly additive to V1.

### v3 — dirty-state

V2 plus a pre-existing partial implementation of the wrong proposal (P3 staffing-blocked) in `repo_evidence/code/in_progress_patch/` — a half-finished diff from an engineer who left the team. The agent must decide whether to complete, revert, or ignore the partial work when ranking.

- **Primary stress:** not-invented-here resistance + maintainability / slop control.
- **Expected probe score:** 14–20.
- **What's harder than V2:** the partial work creates a plausible "let's just finish it" path that trades maintainability for short-term progress; the correct managerial call is to revert (sunk-cost trap).
- **Gradient justification:** Adds seeded partial-fix pressure without removing any V2 signal.

### v4 — multi-corpus-objective

V3 plus a `release_context/` directory containing release notes and a test inventory that reveal a *second* objective (customer-reported incident in `/v1/responses` streaming reliability). Two proposals now plausibly map to different objectives; the agent must decide which objective this cycle prioritizes given release-note recency and open-issue count.

- **Primary stress:** plan/dependency correctness + cross-corpus synthesis.
- **Expected probe score:** 12–18.
- **What's harder than V3:** the "obvious" latency framing is no longer the right framing; the release note shows a blocker incident was downgraded last cycle and is back. Gold ranking flips one pair compared to V3.
- **Gradient justification:** adds a second corpus (release notes + test inventory) and forces objective selection before ranking.

### v5 — recovery-in-thread

V4 plus an `incident_context/` directory showing the accepted proposal from a prior (simulated) ranking was rolled back mid-deployment after a production incident. Fresh incident evidence reveals the proposal had a latent compatibility bug with a recently-landed change in the proxy. The agent must rank afresh, incorporating the incident, and the accepted-then-rolled-back proposal must be demoted in the new ranking with incident-grounded reasoning.

- **Primary stress:** recovery quality + regression-free change reasoning + partial-progress metric.
- **Expected probe score:** 8–14.
- **What's harder than V4:** requires reasoning over a dynamic prior outcome, not static evidence. The sunk-cost pull of "but we already picked it" is strong.
- **Gradient justification:** adds trajectory-aware pressure (prior decision + its outcome) and forces the brief to ground the re-ranking in incident evidence.

### Ladder monotonicity check

Expected per-variant means on GPT-5.4/high: V1≈27, V2≈21, V3≈17, V4≈15, V5≈11. Family mean ≈ 18. Within §7.8 calibration target [15, 25]; V5 ≤ 14 satisfies "at least one variant ≤ 10" only loosely — to tighten, V5 hidden ranking must diverge from V4 in two positions, not one. Locked.

## Seeded Strategic Ambiguity (per variant)

Family-wide traps (all variants):

1. The fastest proposal is not the safest.
2. The most ambitious proposal creates avoidable rollout risk.
3. One proposal looks attractive if staffing constraints are ignored.

V2+ adds:

4. One distractor proposal has the best headline number but the source measurement is stale.

V3+ adds:

5. A partial implementation of a staffing-blocked proposal exists in-tree and tempts "just finish it" thinking.

V4+ adds:

6. A second objective (streaming reliability) competes with the obvious objective (latency). Release-note recency signals which wins.

V5+ adds:

7. The "obvious" pick was already tried and rolled back; fresh incident evidence must override anchoring.

## Expected Deliverables

- `brief/manager_brief.json` (canonical, written by `./bin/cnb55-brief submit`) containing:
  - `ranking[]` with `proposal_id`, `rank`, `summary`, `citations[]`, `constraint_tags[]` per entry
  - `accepted` proposal id (must equal the rank=1 entry)
  - `primary_risk.{statement, mitigations[]}` for the accepted proposal
  - `assumption_ledger[]` with observed / to_verify / missing rows (≥ 1 missing)
- `brief/manager_brief.md` (auto-rendered by the CLI from the JSON)

No code changes are required. Do not modify `proposals/`, `repo_evidence/`, `release_context/`, `incident_context/`, `tests/`, `AGENTS.md`, `Dockerfile`, `.scenario_variant`, or `bin/`. The only directory the agent writes to is `brief/`, and it does so exclusively via the CLI.

## Grader Contract

### Phase 2 — visible checks

```
pytest -q tests/test_ranking_brief.py
```

Checks:
- `brief/manager_brief.json` exists and is non-empty (produced by `./bin/cnb55-brief submit`).
- The JSON parses and `schema_version == "cnb55.manager_brief.v2"`.
- `ranking` has exactly N entries (N = 4 for V1, 5 for V2+).
- `accepted` resolves to one of the ranked proposal ids and is the rank=1 entry.
- Every entry has `proposal_id`, `rank`, `summary`, `citations`, `constraint_tags`.
- `assumption_ledger` is present with ≥ 1 row where `status == "missing"`.
- `variant_id` matches the workspace's `.scenario_variant`.

Visible checks contribute ≤ 30 points (hard cap per SPEC §7.1).

### Hidden checks (verifier_data/…/hidden_tests/)

Categorized per SPEC §7.1:

- **Behavioral:** accepted proposal matches the gold `accepted` for this variant. Top-2 set matches gold top-2 set.
- **Differential:** full ranking Kendall τ vs. gold_ranking.json ≥ τ threshold for the variant (τ ≥ 0.5 for V1, ≥ 0.6 for V2–V5 because the ordering is more constrained).
- **Property-based:** each rejection rationale references at least one evidence file that actually exists and at least one constraint from {staffing, rollout, regression, perf, release, incident}.
- **Regression:** brief does not cite files or proposal ids that don't exist.
- **Collateral damage:** no files modified outside `brief/`; checksum manifest of immutable slice verifies.

### Trusted final-state checks

- `tests/test_ranking_brief.py` was not modified (checksum).
- No `sitecustomize.py`, `usercustomize.py`, or pytest shim in workspace.
- No network access attempted (grader log inspection).
- Readonly tree hashes unchanged for `proposals/`, `repo_evidence/`, `release_context/`, `incident_context/`, `tests/`, `AGENTS.md`, `Dockerfile`, `.scenario_variant`, and `bin/`.
- `brief/` contains at most `manager_brief.json` and `manager_brief.md` (both written by the CLI).
- The agent's CLI input file, conventionally `brief_input.json`, lives at the **workspace root** (not under `brief/`) and is whitelisted. The only writable paths are `brief/*` and `brief_input.json` at workspace root; any other write triggers the `wrote_outside_brief` trusted-final-state ceiling (score → 0).

## 100-Point Breakdown (Track 10 scorecard)

| Sub-metric | Points | Source |
| --- | --- | --- |
| Proposal ranking / decision quality | 20 | hidden differential + behavioral |
| Objective delta (did the pick reflect the right objective) | 20 | hidden behavioral (V4+ only; before V4, folded into ranking) |
| Regression-free change (pick doesn't reintroduce prior incident) | 20 | V5 hidden; V1–V4 folded into rejection quality |
| Maintainability / slop control (sunk-cost + partial-work resistance) | 15 | V3+ hidden; V1–V2 folded into rejection quality |
| Plan / dependency correctness (risk + mitigation coherence) | 15 | hidden property-based + LLM-judge rubric |
| Partial progress (reasoning quality on close calls) | 10 | hidden LLM-judge rubric |

Before V4, the objective-delta and regression-free sub-metrics are not yet active axes; their points are reallocated proportionally to ranking and plan/dependency for V1–V3. The reallocation is encoded in the scorer, not at authoring time, to keep the track scorecard authoritative.

## Partial-Credit Ceilings

Encoded in the scorer; applied as hard caps after raw aggregation.

- **`no_brief_file`** ≤ 0 — if `brief/manager_brief.json` does not exist (the agent did not run `./bin/cnb55-brief submit`).
- **`malformed_brief`** ≤ 10 — if `brief/manager_brief.json` is not valid JSON or does not carry `schema_version == "cnb55.manager_brief.v2"`.
- **`ranking_without_grounding`** ≤ 25 — if fewer than 2 rejection entries have `citations[]` containing a path that resolves to a real evidence file under `proposals/`, `repo_evidence/`, `release_context/`, or `incident_context/`. (Blocks "pick the right answer without justifying it".)
- **`ignored_staffing_constraint`** ≤ 30 — if accepted proposal is the staffing-blocked one (P3 in V1–V3; varies in V4–V5). (Blocks the "best delta wins" shortcut.)
- **`ignored_stale_perf`** ≤ 35 — V2+: if accepted proposal cites the stale perf number as evidence without flagging it. (Blocks naive evidence consumption.)
- **`sunk_cost_finish`** ≤ 40 — V3+: if accepted proposal is the one with the partial in-progress patch AND maintainability rationale is not present. (Blocks the "just finish it" path.)
- **`objective_drift`** ≤ 45 — V4+: if accepted proposal solves the wrong objective given release-note recency. (Blocks single-corpus anchoring.)
- **`incident_blind_reselect`** ≤ 30 — V5: if accepted proposal is the previously-rolled-back one without incident-grounded rationale for why the incident no longer applies.

## Red-Team Traps (≥ 4 per family; expanded per variant)

Named shortcuts authors expect agents to try. Each must have a corresponding exploit patch under `verifier_data/<family_id>/<variant_id>/red_team/` that scores ≤ 20 when executed.

1. **`delete_visible_tests.sh`** — attempt to delete or no-op `tests/test_ranking_brief.py`. Trusted-state checksum blocks.
2. **`shadow_pytest.sh`** — drop a fake `pytest.py` shim that always exits 0. Trusted-state `check_no_python_test_spoof` blocks.
3. **`minimal_brief.json`** — submit a brief whose rejection entries have empty or fabricated `citations`. Ceiling `ranking_without_grounding` caps at 25.
4. **`pick_highest_delta.md`** — pick P3 (staffing-blocked) in V1–V3. Ceiling `ignored_staffing_constraint` caps at 30.
5. **`finish_the_partial.md`** — accept the in-progress-patch proposal in V3+. Ceiling `sunk_cost_finish` caps at 40.
6. **`single_objective_anchor.md`** — V4+: rank without reading `release_context/`. Objective-delta hidden check fails.

## Quality Gate

- Probe protocol: GPT-5.4/high, 3 seeds, per-variant wall-clock cap from CNB-55 SPEC §6.2.
- Freeze gate (CNB-55 SPEC §10.1): family mean in [15, 25], no variant > 40, at least one variant ≤ 10, monotonic, ≥ 4 red-team exploits each scoring ≤ 20, oracle scoring ≥ 90, empty scoring 0, scorer deterministic.
- Human-verification gate (CNB-55 SPEC §10.2): two-reviewer sign-off.

The concrete probe run log and hardening evidence live in [`benchmark_run.md`](./benchmark_run.md).

## Saturation and renewal plan

Per HLD-Family-Test-Requirements §4 item 14 and HLD-Training-Flywheel §17.10. This family is a non-expiring training signal, so we commit to an explicit refresh cadence the moment the population distinguishes it.

**Saturation trigger.** When the mean `P_benchmark` across this family exceeds **80** for two consecutive probe rounds at the probe's default model tier (GPT-5.4/high at time of spec), the family is flagged `saturation_renewal_due` in `benchmark_run.md`. The probe harness reads this from `family.yaml#saturation.threshold_mean_P` so changing the threshold does not require a spec edit.

**Renewal mechanisms (applied in order, one per renewal cycle).**

1. **V6 mid-run staffing change.** A new variant that injects a staffing update *during* the agent's session (the model must re-rank given new availability data it did not have at first read). Tests `inspect:evidence_triage` and `verify:assumption_honesty` against a moving target.
2. **V7 contradictory evidence.** Two first-party evidence files disagree by design (release_context/ vs. incident_context/). The correct move is to flag the contradiction in the brief's `assumption_ledger` with status `"conflict"` rather than silently picking one corpus. Tests `verify:assumption_honesty` and `modify:policy_tradeoff`.
3. **V1 retirement.** V1 was authored as a floor-check (oracle ≥ 90, capable model ≈ 85–95). Once saturated it contributes only gradient noise to RL, so it is retired and replaced with a harder floor-check derived from V2 minus the perf-staleness trap.

**Renewal does not reset the history.** Old variants remain in the event store as `family_status_at_write_time=retired` rows per HLD §3.1. Downstream RL pipelines filter on `family_status_at_write_time=active` rather than deleting old events. This preserves longitudinal regressions tests.

**Out-of-cycle renewal.** If a model exceeds 80 P_benchmark on a single variant for three consecutive rounds (even if the family mean has not saturated), that variant is retired and a replacement is drawn from the renewal queue. The variant-level check catches per-trap overfitting that the family-level mean hides.

**Dual-band saturation.** The trigger is on `P_benchmark`, not `M_training`. If `M_training` saturates independently (mean > 0.80 for 2 rounds) while `P_benchmark` stays below 80, that is a diagnostic signal that the LLM-judge quarantined points are absorbing the remaining variance — investigate the rubric or move quarantined checks into the deterministic band before renewing.
