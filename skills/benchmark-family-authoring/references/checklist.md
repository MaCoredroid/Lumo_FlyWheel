# Pre-merge checklist

Run through this before declaring a family shipped. Every item must be either checked or explicitly waived in `benchmark_run.md` with rationale.

## Contract

- [ ] `benchmark_blueprints/families/<family-id>/task_spec.md` exists and matches the format of `proposal-ranking-manager-judgment/task_spec.md`.
- [ ] `task_spec.md` specifies the structured-output CLI schema version, the required surfaces, and the per-variant workspace layout.
- [ ] `benchmark_blueprints/families/<family-id>/evaluator_contract.md` lists the visible-check budget (≤ 30 pts), hidden-check budget (≥ 50 pts), and every named ceiling with its value and trigger.
- [ ] `benchmark_blueprints/families/<family-id>/benchmark_run.md` has at least one `attempt_NN` section documenting a probe cycle and the §10.1 verdict.

## Workspace bundle

- [ ] Five variant directories under `benchmark_blueprints/families/<family-id>/workspace_bundle/v{1..5}-<slug>/`.
- [ ] Identical `AGENTS.md` across all five variants (principle-level, no rubric leakage). Verify with `md5sum`.
- [ ] Identical `Dockerfile`, `bin/`, and `.scenario_variant` content across variants (except for `.scenario_variant`'s variant-id line).
- [ ] Each variant's `repo_evidence/` has no "How to reflect this in a manager brief" sections or equivalent rubric-leakage content.
- [ ] Each variant's evidence is internally consistent — no orphan references to files that don't exist in that variant.

## Scorer

- [ ] `verifiers/<family-id>/score_<domain>.py` is stdlib-only Python.
- [ ] Scorer reads `$AGENT_WS`, `$VERIFIER_DATA`, `$VARIANT_ID`, `$CNB55_SEED` from env; writes `$RESULT_FILE` as JSON; no other output.
- [ ] Result JSON includes: `score`, `raw_score_pre_ceiling`, `pass`, `shortcut_detected`, `ceilings_applied`, `milestones`, `breakdown`, `errors`.
- [ ] Every named ceiling in `evaluator_contract.md` corresponds to one `apply_ceiling(...)` call in the scorer with the same name and value.
- [ ] `readonly_tree_hashes` recomputed at scoring time against the agent's workspace; mismatch → `shortcut_detected: true`.

## Oracle / baselines

- [ ] `scripts/regen_cnb55_v2.py` (or family-equivalent) regenerates oracle briefs for all five variants via the family's CLI.
- [ ] Oracle score ≥ 90 for every variant (recorded in the regen script's summary table).
- [ ] Empty-brief score = 0 for every variant.
- [ ] Shortcut-brief score ≤ 30 for every variant.
- [ ] `readonly_tree_hashes` in every `verifier_data/<family>/<variant>/gold_ranking.json` match the current workspace bundle.
- [ ] `benchmark_blueprints/families/<family-id>/manifest.lock.json` refreshed via `scripts/refresh_manifest_lock.py`.

## Probe & §10.1 acceptance

- [ ] `N=3 FAMILY=<family-id> bash scripts/probe_family.sh` completed cleanly on the calibration Mac (no `codex_exit != 0` rows, no `shortcut_detected: true` rows).
- [ ] `scripts/probe_report.py --probe-run-id <latest>` reports per-variant mean/stdev/min/max/scores/ceilings.
- [ ] Acceptance verdict recorded in `benchmark_run.md`:
  - [ ] `family_mean ∈ [lo, hi]` (default `[15, 25]`, or widened with documented rationale).
  - [ ] `max_variant_mean ≤ cap` (default `40`).
  - [ ] `min_variant_mean ≤ hard_floor` (default `10`).
  - [ ] Monotonic V1 ≥ V2 ≥ V3 ≥ V4 ≥ V5 within ±3 tolerance.

## Variant progression discrimination

- [ ] V1 serves as a floor-check (oracle ≥ 90, capable model ≈ 85-95).
- [ ] V2's `ignored_stale_perf` ceiling fires when accepted proposal cites a `stale_perf_markers` entry without acknowledging staleness.
- [ ] V3's `sunk_cost_finish` ceiling fires when accepted == `sunk_cost_trap_proposal`.
- [ ] V4's `objective_drift` ceiling fires when accepted == V1-V3's accepted (instead of the re-weighted objective's pick).
- [ ] V5's `incident_blind_reselect` ceiling fires when accepted == rolled-back-proposal AND no citation references `incident_context/`.

## Spot-checks

- [ ] Read at least one probe-run brief per variant (`/tmp/cnb55_probe/<run-tag>/workspace/brief/manager_brief.json`). Confirm the agent is doing legitimate manager work, not cheesing the rubric.
- [ ] Read the per-run codex log (`report/probe_run_logs/<run-tag>.log`) for the lowest-scoring run on each variant. Confirm the score reflects a real judgment failure, not a harness bug.

## Documentation

- [ ] `evaluator_contract.md` end-state matches the shipped scorer. Point allocations, ceiling values, τ thresholds all current.
- [ ] `benchmark_run.md` has a final "Hardening decisions already applied" section summarizing the levers that survived calibration.
- [ ] Any §10.1 widening recorded explicitly with rationale under the legitimate-difficulty test.
- [ ] Any family-specific scripts under `scripts/` are named and purpose-commented.

## Layer B (flywheel readiness) — HLD Family-Test-Requirements §4

See `references/flywheel-readiness.md` for the full rationale on each item. Every box below must be checked before LLD-06 will ingest this family's event-store rows.

- [ ] Scorer emits `P_benchmark` (0-100) AND `M_training` (normalized 0-1) with `schema_version: "cnb55.verify_result.v3"`.
- [ ] `breakdown.__bands` is present and tags every key as `"M"` or `"P_only"` (Decision A — LLM-judge quarantine).
- [ ] 5-slot milestone vector emitted with HLD §7.5 weights (M1=0.10, M2=0.20, M3=0.20, M4=0.20, M5=0.30).
- [ ] Milestone graders are L1/L2/L3 only — no LLM judge in the milestone layer.
- [ ] `verifier_data/<family>/<variant>/milestones/m{1..5}_*.sh` exist (via symlink) and agree with `milestone_vector.slots[*].passed_bool` in both Oracle and H=1 cases.
- [ ] `integrity_flag == 1` zeroes M3/M4/M5 per HLD §7.7.5.
- [ ] Every `raise_integrity("<id>")` call site matches an `integrity_rules[].id` in `family.yaml` 1:1.
- [ ] `family.yaml` declares capability tags (core `{required, recommended, forbidden}` + HLD §17.5 extended sub-tags), tool-call overrides, state-delta rules (`kind: json_deliverable` with transition table), `llm_judge_quarantine` section, seeds + variance thresholds, `initial_state: {type: manifest_locked}`, saturation + renewal queue, `rawr_modes`, `grader_ref`, `milestone_config_ref`.
- [ ] `scripts/run_verification_matrix.py --variant <v1-id>` produces `verification_matrix.md` with all 6 rows in HLD §5 expected bands.
- [ ] `scripts/run_verification_matrix.py --variant <stress-id>` produces `verification_matrix_<stress>.md` with all 6 rows in band (HLD §8 box 8).
- [ ] `task_spec.md` has a `## Saturation and renewal plan` section with trigger and ≥ 2 renewal mechanisms.
- [ ] `benchmark_run.md` has an `attempt_NN — Layer B flywheel-readiness upgrade` section enumerating shipped changes.

## Waiver section (if needed)

If any checklist item is waived, record in `benchmark_run.md`:

```markdown
## Pre-merge waivers

- **§10.1 `family_mean` window widened from [15, 25] to [60, 75]**: this family's V1-V3 cluster represents a mechanical floor for gpt-5.4 high (see attempt_02d discussion). Three hardening attempts did not move the floor without crossing into fake ambiguity. V4/V5 carry the family's judgment signal. Reviewer: <name>.
```

Waivers are legitimate outcomes when the underlying signal is real. Unwaived failures are not — debug until resolved or widen explicitly.
