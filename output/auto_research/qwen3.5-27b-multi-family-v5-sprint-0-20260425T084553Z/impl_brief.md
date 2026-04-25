# IMPL Brief — Auto-Research Substrate (LLD-SB-06)

You are the implementation agent. Your job is to deliver the substrate
the v0.1 auto-research round will run on top of. This is a one-shot
implementation task, not a research loop.

## Context docs (read all three first)

- Parent HLD:  docs/HLD-Serving-Backend-AutoResearch-v0_1.md
- Sub-spec:    docs/HLD-Serving-Backend-AutoResearch-v0_1-SubSpec-AutoResearchAgent.md
- Parent §5.6 (measurement harness), §5.9 (bundle schema), §9.3 (verification)

## Deliverables (all must land on main)

1. src/lumo_flywheel_serving/measurement_harness.py
   - class RealMeasurementHarness per sub-spec §9.1
   - measure() method per §9.1 signature
   - emits MeasuredTrace per §9.2 schema with generator =
     "RealMeasurementHarness v0.1.0"
   - implements the parent §5.6 loop: /admin/load_tuned_config,
     /health wait, seed-trace replay, per-request latency capture,
     /metrics scrape at window boundaries, PromQL-derived p95
     cross-check, purity sample, determinism probe, KV-poisoning probe

2. scripts/capture_seed_workload.py
   - runs family eval set through default-config serving stack once
   - persists per-request jsonl: prompt_tokens, output_tokens,
     thinking_tokens, turn_index
   - emits workload_distribution_id = sha256 of the persisted file

3. CLI subcommands under `lumoserve auto-research …` — all 8 required
   for Phase A completion, plus the backward-compat `run`:
   - bootstrap-round   (sub-spec §8.1)
   - measure           (sub-spec §8.2 — --harness real|synthetic)
   - commit-candidate  (sub-spec §8.3 — --harness real|synthetic)
   - rescreen          (sub-spec §8.4 — required by finalize-round)
   - validate-holdout  (sub-spec §8.5 — required by finalize-round)
   - finalize-round    (sub-spec §8.6 — refuses without rescreen + holdout
                         unless --dry-run is passed, §8.6a)
   - status            (sub-spec §8.7 — read-only round state for Python)
   - run-round         (sub-spec §8.8 — Python outer loop command)
   Existing `run` subcommand stays but is env-guarded per §8.9.

4. skills/auto-research-round-manager/SKILL.md — full rewrite
   - Python outer loop per sub-spec §11
   - spawns `codex exec` per iteration (sub-spec §2.3)
   - owns stop criteria (sub-spec §11.3)
   - calls bootstrap-round, loop-of-codex-exec, finalize-round,
     live family gate in that order

5. tests/fixtures/synthetic_measurement.py
   - move SyntheticMeasurementHarness here, rename to
     SyntheticMeasurementFixture, emit generator =
     "SyntheticMeasurementFixture v<n>"
   - commit-candidate must REFUSE this generator in real mode and accept
     it only in synthetic fixture mode per sub-spec §8.3

6. Unit + integration tests:
   - unit: each CLI subcommand
   - unit: skill watchdog paths (silence, out-of-scope write,
           unsigned commit)
   - integration: dry-run round against SyntheticMeasurementFixture
                  (allowed only under LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT=1)
   - integration: precondition refuses when harness module absent

7. Pre-flight checks for the skill (sub-spec §11.1):
   - RealMeasurementHarness imports cleanly
   - codex --version returns expected version
   - git status clean
   - workload yaml has seed_trace_ref pointing at existing jsonl
   - LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT unset

8. Codex-facing brief templates (strings in the skill):
   - impl_brief.md   (this file — you may update if you discover
                       the spec is wrong; note the update in §14)
   - iteration_brief.md (sub-spec §5.2 template — ship verbatim)

9. src/lumo_flywheel_serving/round_driver.py
   - RoundContext, RoundResult, run_round(ctx), and restore_worktree_head()
   - exposed through `lumoserve auto-research run-round`

## Done when

- All 9 items above land on main
- All unit + integration tests pass
- A dry-run round against SyntheticMeasurementFixture completes
  successfully end-to-end (demonstrates the wiring is correct,
  does not prove the real harness works)
- `python -c "from lumo_flywheel_serving.measurement_harness    import RealMeasurementHarness"` succeeds
- sub-spec §9.3.AR.7 and §9.3.AR.12 verification items pass

## You may

- install packages and add dependencies to pyproject.toml
  (Phase A is the only phase where this is allowed)
- modify any file in the repo
- create new files under src/, scripts/, skills/, tests/
- refactor existing code that conflicts with the new surface

## You may not

- ship a Phase A deliverable that calls SyntheticMeasurementFixture
  from production code paths
- modify docs/ without updating the corresponding sub-spec section
- leave any test failing
- declare done without running the dry-run round end-to-end

## Exit protocol

Open one PR with all 8 deliverables. Title:
  "Phase A: auto-research substrate (LLD-SB-06)"
Body: checklist from "Done when" above, all items checked.
