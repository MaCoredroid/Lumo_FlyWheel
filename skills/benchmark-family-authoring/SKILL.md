---
name: benchmark-family-authoring
description: Author and calibrate a CNB-55-style benchmark family (task spec + workspace bundle + deterministic evaluator + variant progression V1-V5) AND make it flywheel-ready for RL training (dual-band emission, 5-slot milestones, capability tags, integrity rules, verification matrix). Drive it through the probe → harden → re-probe loop until §10.1 freeze-gate acceptance (Layer A) AND through the HLD Family-Test-Requirements §4 14-item readiness list (Layer B), or until the family's honest signal is understood. Use when the user asks to "create a new family", "add a track-N family", "build a benchmark", "write an evaluator", "harden a family", "fail the freeze gate", "calibrate the probe", "make the family RL-ready", or "wire the family into the flywheel".
---

# Benchmark-family authoring & calibration

## Two layers of DONE — read this first

Families in this repo must satisfy **both** of the following before they ship. Missing either one silently ships an asymmetric artifact that calibrates the probe but is unusable as a training signal (or vice versa).

- **Layer A — §10.1 freeze gate** (CNB-55 authoring spec). Family_mean ∈ [15, 25], max ≤ 40, ≥ 1 variant ≤ 10, monotonic V1≥V2≥V3≥V4≥V5 ±3, oracle ≥ 90, empty = 0, shortcut ≤ 30. This is the leaderboard/probe acceptance gate.
- **Layer B — 14-item flywheel readiness** (HLD-Family-Test-Requirements §4). Dual-band scorer (P_benchmark + M_training), 5-slot milestones (M1…M5 with HLD §7.5 weights), capability tags, state-delta rules, integrity-flag detector, LLM-judge quarantine, seeds + variance escalation, verification matrix (6 trajectories × 5 metrics), grader_ref/milestone_config_ref registry entries, saturation + renewal plan. This is the LLD-06 ingestion gate.

A reference implementation of both layers lives at `benchmark_blueprints/families/proposal-ranking-manager-judgment/` (see `family.yaml`, `verifiers/.../score_ranking.py`, `verification_matrix.md`, and `benchmark_run.md` attempt_03a).

## When to use this skill

Invoke when the user is doing any of these against the FlyWheel repo:

- Scaffolding a new `benchmark_blueprints/families/<family-id>/` directory.
- Writing or rewriting the five-variant workspace bundle (`workspace_bundle/v{1..5}/`).
- Writing or modifying the deterministic scorer under `verifiers/<family-id>/`.
- Producing or regenerating oracle briefs and `verifier_data/<family-id>/<variant>/` artifacts.
- Running the probe loop (`scripts/probe_family.sh` + `scripts/probe_report.py`) and interpreting the results against §10.1 freeze-gate acceptance (`family_mean ∈ [15, 25]`, `max ≤ 40`, `≥ 1 variant ≤ 10`, monotonic V1≥V2≥V3≥V4≥V5 ±tolerance, oracle ≥ 90, empty = 0, shortcut ≤ 30).
- Diagnosing why a variant won't calibrate — which is usually more useful than actually calibrating it.

Do NOT invoke this skill for: cross-family infrastructure (`scripts/probe_family.sh`, `refresh_manifest_lock.py`, CNB-55-authoring-spec.md edits), model-registry changes, or anything that isn't a single family's task/evaluator/evidence.

## The one hard rule

Every hardening decision must satisfy the **legitimate-difficulty test**: a strong human manager reading the variant would agree that the specific behavior being penalized is a real judgment failure, not a guessing game. If a reasonable reader couldn't predict the ceiling fire from the evidence alone, the hardening is fake ambiguity — remove it. The corollary: you cannot close the gap between a capable model's score and a narrow acceptance window by inventing rubric traps. Options when the gap won't close are in `references/pitfalls.md` under "Mechanical score floor".

## How to use this skill

Work sequentially. Each phase has a definite exit signal; do not move on until it fires.

### Phase 1 — Contract

Before touching any evidence or scorer code, author three files and get the user's sign-off on all three:

1. **`task_spec.md`** — the canonical agent-visible task prompt, the CLI contract (structured-output schema version, submit/validate/schema subcommands), the required surfaces, and the per-variant workspace layout. Reference `benchmark_blueprints/families/proposal-ranking-manager-judgment/task_spec.md` for the format.
2. **`evaluator_contract.md`** — the full rubric: visible-check budget (≤ 30 pts), hidden-check budget (≥ 50 pts), per-check point allocations, partial-credit ceiling names and values, Kendall-τ thresholds if applicable, and the shortcut/empty baselines (shortcut ≤ 30, empty = 0). Reference the existing family's `evaluator_contract.md`.
3. **`benchmark_run.md`** — the running calibration log. Start with a "attempt_00 — baseline design" section listing hypotheses about which variants will discriminate and why. Every probe cycle appends a new `attempt_NN` section with the design change, the probe results, and the acceptance verdict.

The three files together answer: what does the agent see, what does the scorer reward, and what have we learned from each probe. Do not skip the log — diagnosis depends on the per-attempt history.

### Phase 2 — Variant progression

Design five variants that add one discriminating dimension each. See `references/variant-progression.md` for the canonical pattern and the per-variant ceiling mapping. The short version:

- **V1 clean baseline** — minimal evidence, no traps. Floor for "did the agent produce a valid brief". τ ≥ 0.67.
- **V2 noisy distractor** — adds stale perf markers, a departing-engineer row, an outdated memo. Tests "can the agent *not* anchor on noise". τ ≥ 0.80; adds `ignored_stale_perf` ceiling.
- **V3 dirty state** — workspace has an abandoned in-progress patch, a half-filled previous-session brief, or similar mid-flight detritus. Tests "can the agent *not* complete a sunk cost". Adds `sunk_cost_finish` ceiling.
- **V4 multi-corpus / objective drift** — adds `release_context/`, `incident_context/`, or an alternative-objective corpus that shifts the right accepted choice. Tests "can the agent re-weight under a new objective". Adds `objective_drift` ceiling.
- **V5 recovery in thread** — the accepted pick from a prior cycle was rolled back (INC-XXXX); the agent must acknowledge the incident before re-selecting or selecting a successor. Adds `incident_blind_reselect` and often `ignored_stale_perf` ceilings.

Each workspace bundle ships identical `AGENTS.md`, `Dockerfile`, `bin/<cli>`, and `.scenario_variant`. The evidence under `proposals/`, `repo_evidence/`, etc. differs per variant. Use sha256 tree hashes in `gold_ranking.json.readonly_tree_hashes` to detect agent mutation of anything outside `brief/`.

### Phase 3 — Scorer with partial-credit ceilings

Write the scorer at `verifiers/<family-id>/score_<domain>.py` as stdlib-only Python. The core pattern is a `ScoreState` that accumulates raw points, then applies **post-aggregation ceilings** that cap the score when a specific judgment miss is detected. See `references/scoring-ceilings.md` for the full pattern and the ceiling-design rubric (every ceiling must be (a) named, (b) tied to a concrete brief observation, (c) documented in `evaluator_contract.md`, (d) defensible under the legitimate-difficulty test).

A ceiling's value is the **maximum** score a brief can receive when the ceiling fires — pre-ceiling raw points above that value are clipped down. Record every ceiling fire in the scorer's result JSON so the probe report can surface them.

Also write three baseline harnesses inside the scorer's regen script:
- **Oracle**: the hand-written "correct" brief should score ≥ 90.
- **Empty**: an empty brief (or missing `brief/manager_brief.json`) should score 0.
- **Shortcut**: a brief that picks the obviously-wrong staffing-blocked (or sunk-cost, or incident-rolled-back) option should score ≤ 30.

If any baseline violates its budget, fix the scorer — not the baseline.

### Phase 4 — Oracle + manifest

Run `scripts/regen_cnb55_v2.py` (or the family equivalent) to:

1. Author `verifier_data/<family-id>/<variant>/oracle/brief_input.json` for each variant.
2. Execute the family's structured-output CLI against a copy of the workspace bundle to produce the canonical `manager_brief.{json,md}`.
3. Refresh `readonly_tree_hashes` in `gold_ranking.json` and `workspace_manifest.json` for every variant.
4. Verify oracle ≥ 90, empty = 0, shortcut ≤ 30.

Then run `scripts/refresh_manifest_lock.py` to update the family-level `manifest.lock.json`. Any edit to `AGENTS.md`, `bin/`, `proposals/`, `repo_evidence/`, `release_context/`, `incident_context/`, or `tests/` invalidates the manifest and must be followed by a regen + refresh pair, otherwise probes will fire `shortcut_detected`.

### Phase 5 — Probe loop

Run `N=3 scripts/probe_family.sh` (3 runs × 5 variants = 15 runs, ~38 min wall on a Mac with `codex exec --model gpt-5.4 --reasoning-effort high`). Use the `_run_probe_02b.sh` self-daemonizing nohup launcher pattern when the probe must be fire-and-forgotten from AppleScript — AppleScript's reserved-operator behavior on `&` otherwise drops the nohup silently.

Generate `report/attempt_NN_probe_report.txt` via `scripts/probe_report.py`. Check the four §10.1 gates:

- `family_mean ∈ [15, 25]`
- `max_variant_mean ≤ 40`
- `min_variant_mean ≤ 10` (at least one variant must genuinely fail)
- Monotonic V1 ≥ V2 ≥ V3 ≥ V4 ≥ V5 within ±3 tolerance

If any gate fails, proceed to Phase 6. Record the attempt in `benchmark_run.md` with: design change, full per-variant table, which gates passed/failed, a spot-check of at least one agent brief to explain *why* the scores look the way they do, and the hypothesis for the next attempt.

### Phase 6 — Harden or diagnose

Before adding any hardening lever, diagnose. Load the probe log and read the variant where the gap is largest. Ask: is the score high because the rubric is leaking, or because the task is genuinely within the model's competence? The answer decides the lever:

- **Rubric leaking in AGENTS.md** (keyword lists, exact thresholds, named ceilings). Decouple: rewrite AGENTS.md with principle-level guidance only. See `references/pitfalls.md` § "Disclosure paradox".
- **Rubric leaking in evidence files** (e.g., "How to reflect this in a manager brief" sections, explicit citation instructions in a memo). Strip the leakage but keep the underlying facts.
- **Mechanical score floor** — the agent is hitting every concrete check and the residual gap is scattered across low-weight dimensions. Adversarial evidence will not move this. See `references/pitfalls.md` § "Mechanical score floor" for the three honest options (widen window, add new judgment ceiling, accept the signal).
- **Variant too similar to its neighbor** — add one concrete dimension the neighbor doesn't have. Don't make it harder by being vaguer.

Apply one change per attempt. Multiple simultaneous changes make diagnosis impossible across the `benchmark_run.md` history.

### Phase 7 — Layer A acceptance

A family ships **Layer A** when either (a) all four §10.1 gates pass with the current calibration model, or (b) the `benchmark_run.md` log documents why a narrower window isn't achievable without fake ambiguity AND the user has explicitly sized the window for this family's frontier difficulty. The latter is a legitimate outcome — do not treat "gate failed" as always meaning "family broken". See `references/acceptance-gate-calibration.md` for the decision tree.

### Phase 8 — Layer B flywheel readiness

After Layer A is green (or explicitly waived), make the family ingestable by LLD-06's training views. This is not optional — ship Layer A without Layer B and the event-store rows this family produces are unreadable from the SFT/DPO/RL side.

Read `references/flywheel-readiness.md` before starting. The short version — do all 14:

1. **Dual-band scorer.** Emit `P_benchmark` (probe-facing 0-100) and `M_training` (deterministic-only, normalized to [0, 1]) on every run. Tag every breakdown key with `band="M"` or `band="P_only"` (Decision A — LLM-judge quarantine pattern). Keep `score` as alias for `P_benchmark` for backward compat. Bump schema_version to `cnb55.verify_result.v3`.
2. **5-slot milestones.** M1 Localization 0.10, M2 Primary fix 0.20, M3 Invariants 0.20, M4 Functional 0.20, M5 E2E 0.30. Emit both `milestones` (bool dict) and `milestone_vector.slots[].passed_bool` with weights. Implement at L1/L2/L3 (HLD §7.8) — **never** an LLM-judge in the milestone layer.
3. **Integrity-flag detector.** 5 rules minimum (`write_outside_whitelist`, `immutable_slice_mutated`, `pytest_shim`, `tests_modified`, `network_egress`). One `raise_integrity(rule_id)` call per detected rule. H=1 force-fails M3/M4/M5 per HLD §7.7.5. Rule IDs in `family.yaml#integrity_rules` must match scorer call sites 1:1.
4. **Milestone scripts.** `verifier_data/<family>/<variant>/milestones/m{1..5}_*.sh` symlinking to shared scripts. Each reads `$RESULT_FILE`, exits 0/1/2. This is the **declarative** surface LLD-06's milestone grader reads.
5. **Capability tags.** {localize, inspect, modify, verify, respect_invariants} as the shared core + extended sub-tags per HLD §17.5 (inspect:prioritize, inspect:evidence_triage, modify:policy_tradeoff, verify:assumption_honesty). Per-variant tag blocks override the default if the variant tests a specific sub-capability.
6. **Tool-call overrides.** Map family-specific CLI subcommands to the 5-capability taxonomy (e.g. `cnb55-brief schema` → inspect, `validate` → verify, `submit` → modify terminal).
7. **State-delta rules (Decision B).** Declare the brief/deliverable as `kind: json_deliverable` with a 6-row transition table (absent → present_and_invalid → present_and_valid) + `aggregate_clamp: [0,1]`. This is what LLD-06 uses to decide whether a corrective turn is worth the gradient update.
8. **LLM-judge quarantine.** Any check whose correctness depends on a language model (partial_progress rubric, assumption-ledger padding) goes in the `P_only` band. Document total quarantined points in `family.yaml#llm_judge_quarantine`.
9. **Seeds + variance escalation.** Base count 2 seeds. Thresholds: `stdev_threshold_to_4=0.10`, `to_8=0.20`, `flag_high_variance=0.15` at 8. Record `current_observed_stdev_M_training` from the latest probe.
10. **Initial state.** Pin `manifest.lock.json` and declare `initial_state: {type: manifest_locked, ref: manifest.lock.json}` in `family.yaml`.
11. **Verification matrix — V1 AND at least one stress variant.** `scripts/run_verification_matrix.py` runs 6 trajectories (Oracle, Empty, RAWR grounding_stripped, Pick-ceiling, Top1-wrong, Delete-tests adversarial) × 5 metrics (P_benchmark, M_training, G, R, S_TTC). Write `verification_matrix.md` (V1) AND `verification_matrix_<stress>.md` (any variant that introduces variant-gated traps: stale-perf, sunk-cost, objective-drift, incident-reselect). HLD §8 box 8 requires the stress-variant rerun — V1-only acceptance is incomplete. At minimum, Oracle / Empty / Delete-tests rows must match HLD §5 bands; the other rows may drift for synthesizer reasons and can be documented as caveats. **Field-name gotcha:** the synthesizer must mutate the exact keys the scorer reads (e.g. `brief["accepted"]`, not `brief["accepted_proposal_id"]`). Always re-run the Pick-ceiling row and confirm the expected ceiling fires; if P_benchmark > 35 on a Pick-ceiling row, the synthesizer is mutating the wrong field.
12. **Multi-mode RAWR.** The family declares at least `grounding_stripped` as implemented. `citation_fabricated` and `constraint_named_not_respected` may be declared as `declared_not_yet_implemented` pending a later attempt.
13. **Saturation + renewal plan.** Add to `task_spec.md`. Trigger: `mean P_benchmark > saturation.threshold_mean_P` for 2 consecutive probe rounds → `saturation_renewal_due`. Enumerate at least 2 renewal mechanisms (new variant slot + retire-floor-check) in `family.yaml#saturation.renewal_queue`.
14. **Registry entries.** `family.yaml#grader_ref` and `milestone_config_ref` point at the canonical scorer path and milestone-config path. LLD-06 uses these to lazy-load the grader at event-store ingestion time.

Acceptance signal for Layer B: a reviewer can run `python3 scripts/run_verification_matrix.py --variant <v1-id>`, read `verification_matrix.md`, and trace every row's P / M / integrity / ceiling numbers back to a single `family.yaml` declaration. No loose ends.

## Reference material

Additional reference files live under `references/`. Read the relevant one before starting the matching phase:

- `references/variant-progression.md` — the V1–V5 pattern and per-variant discriminating dimensions.
- `references/scoring-ceilings.md` — partial-credit ceiling mechanics, ceiling-design rubric, and the scorer skeleton.
- `references/acceptance-gate-calibration.md` — §10.1 gate math, probe loop, and the accept-or-widen decision tree.
- `references/pitfalls.md` — disclosure paradox, rubric leakage, adversarial-evidence limits, mechanical score floor, AppleScript/osascript probe-launch gotchas, and the `readonly_tree_hashes` → `shortcut_detected` debug loop.
- `references/checklist.md` — pre-merge checklist for a new family. Run through it before declaring a family shipped.
- `references/flywheel-readiness.md` — Layer B (HLD Family-Test-Requirements §4) detail: dual-band emission, 5-slot milestones, capability tags, state-delta rules (Decision B), integrity flags, LLM-judge quarantine (Decision A), verification matrix, saturation plan. Read this before Phase 8.

## Keywords

benchmark family, new family, track N family, evaluator, scorer, freeze gate, §10.1, probe, calibration, variant progression, V1 V2 V3 V4 V5, partial-credit ceiling, rubric leakage, disclosure paradox, adversarial evidence, CNB-55, flywheel readiness, Layer A, Layer B, dual-band, P_benchmark, M_training, 5-slot milestone, capability tags, integrity flag, H=1, LLM-judge quarantine, Decision A, Decision B, state delta, RAWR, verification matrix, saturation, grader_ref, LLD-06, RL training, RL-ready.
