# Benchmark Run: Review Thread UI Hardening

## attempt_00 - baseline design

Hypothesis:
- The family should reward the solver only when it correctly maps review-state, viewport, and target control together.
- A package without a real repo or artifact bundle was not a legitimate benchmark, only a prompt stub.
- The honest hardening levers here are resolved-thread noise, stale viewport evidence, abandoned stale-path work, release-scope drift, and rollback-aware recovery.

Baseline verdict:
- The prior package-only draft was discarded as incomplete for Layer A and unusable for Layer B.

## attempt_01 - executable family package

Design change:
- Added a real family-local workspace bundle for V1-V5 with repo files, review exports, screenshot notes, config, visible tests, structured-output CLI, deterministic scorer, verifier data, manifest, and Layer B declarations.
- Kept `proposal-ranking-manager-judgment` as the structural reference only.
- Moved the deliverable to `submission_input.json` plus `./bin/review-thread-task submit`, while requiring actual code/config/reply/evidence edits in the workspace.

Verification run:
- Oracle/empty/shortcut baselines are checked by the family-local regen script.
- Verification matrix is generated for V1 and V5.
- Shared repo-level probe script was not used because it hard-codes the proposal-ranking prompt shape; this family now ships a family-local equivalent for future live probing.

Acceptance status after packaging:
- Layer A: not yet proven live, pending real probe runs.
- Layer B: green. Dual-band scorer, manifest lock, milestone scripts, and verification matrices for V1 and V5 are all generated family-locally.

Deterministic verification:
- Oracle / empty / shortcut baselines: all variants scored `100 / 0 / 10`.
- `verification_matrix.md` (V1) and `verification_matrix_v5.md` (stress variant) both show the expected floor rows:
  - Oracle passes at `100`
  - Empty is `0`
  - Wrong-thread and wrong-viewport synthetic rows hit deterministic ceilings
  - Delete-tests trips integrity and scores `0`

Live probe state:
- Pending. I did not run `codex exec` probe rounds in this turn because the family had to be finished from a doc-only stub into a runnable package first.

Next live-probe hypothesis:
- V1 should score high for a competent solver because the route, control, and viewport all align.
- V2 should drop on stale viewport anchoring.
- V3 should punish stale-path completion.
- V4 should drop sharply if the solver follows the pre-release-scope reopen instead of the release-scoped one.
- V5 should further punish rollback-blind replies even when the runtime fix is mechanically correct.

## attempt_02 - post-hardening smoke and last pre-counted scorer change

Design change:
- Removed direct rubric leakage from `AGENTS.md`, visible review exports, screenshot notes, and `repo/tests/test_review_thread_ui.py`.
- Added a second plausible icon control (`pin-thread-menu`) so the solver has to localize the live unresolved control instead of matching a single named menu.
- Added the V5-only `rollback_recovery_scope_miss` ceiling at `10` when the solver fixes the wrong reopened viewport after the rollback incident.

Exact commands run:

```bash
python3 benchmark_blueprints/families/review-thread-ui-hardening/scripts/regen_family.py
python3 benchmark_blueprints/families/review-thread-ui-hardening/scripts/refresh_manifest_lock.py
python3 benchmark_blueprints/families/review-thread-ui-hardening/scripts/run_verification_matrix.py --variant v1-clean-baseline
python3 benchmark_blueprints/families/review-thread-ui-hardening/scripts/run_verification_matrix.py --variant v5-recovery-in-thread
python3 benchmark_blueprints/families/review-thread-ui-hardening/scripts/probe_family.py --attempt attempt_02b_post_hardening_smoke --n 1
python3 benchmark_blueprints/families/review-thread-ui-hardening/scripts/probe_report.py --attempt-dir benchmark_blueprints/families/review-thread-ui-hardening/report/attempt_02b_post_hardening_smoke
python3 benchmark_blueprints/families/review-thread-ui-hardening/scripts/refresh_manifest_lock.py
python3 benchmark_blueprints/families/review-thread-ui-hardening/scripts/run_verification_matrix.py --variant v5-recovery-in-thread
```

Verification matrix outputs after the V5 scorer change:
- `verification_matrix.md` (V1): Oracle `100`, Empty `0`, Wrong-thread `25`, Wrong-viewport `30`, Artifact-only `10`, Delete-tests `0`.
- `verification_matrix_v5.md` (stress variant): Oracle `100`, Empty `0`, Wrong-thread `25`, Wrong-viewport `10`, Artifact-only `10`, Delete-tests `0`.

Smoke probe result:

| Variant | n | scores | mean P | ceilings |
|---|---:|---|---:|---|
| v1-clean-baseline | 1 | [20] | 20.00 | missing_runtime_or_config_fix |
| v2-noisy-distractor | 1 | [20] | 20.00 | clipping_fix |
| v3-dirty-state | 1 | [20] | 20.00 | clipping_fix, missing_runtime_or_config_fix |
| v4-multi-corpus-objective | 1 | [20] | 20.00 | missing_runtime_or_config_fix, wrong_viewport_mapping |
| v5-recovery-in-thread | 1 | [20] | 20.00 | missing_runtime_or_config_fix, wrong_viewport_mapping |

Smoke Layer A checks:
- `[PASS]` family_mean in `[15, 25]`: `20.00`
- `[PASS]` max variant mean `<= 40`: `20.00`
- `[FAIL]` at least one variant mean `<= 10`: `20.00`
- `[PASS]` monotonic `V1>=V2>=V3>=V4>=V5 +/-3`

Diagnosis:
- The smoke run proved the family-local runner worked, but it did not yet produce a genuine fail-floor. Because the V5 scorer changed after this run, the smoke result is not the counted probe.

## attempt_03 - counted whole-family live probe

Design change relative to the last counted artifact:
- No family content changed after the final V5 scorer adjustment. This is the first whole-family `codex exec` run that counts after the last family-local code change.

Exact commands run:

```bash
python3 benchmark_blueprints/families/review-thread-ui-hardening/scripts/probe_family.py --attempt attempt_03_live_probe --n 3
python3 benchmark_blueprints/families/review-thread-ui-hardening/scripts/probe_report.py --attempt-dir benchmark_blueprints/families/review-thread-ui-hardening/report/attempt_03_live_probe
```

Probe metadata:
- Model: `codex exec --model gpt-5.4 -c 'model_reasoning_effort="high"'`
- Workspace mode: `--ephemeral --sandbox workspace-write --skip-git-repo-check`
- Variants: `v1-clean-baseline`, `v2-noisy-distractor`, `v3-dirty-state`, `v4-multi-corpus-objective`, `v5-recovery-in-thread`
- Saved outputs: `benchmark_blueprints/families/review-thread-ui-hardening/report/attempt_03_live_probe/`

Per-variant counted results:

| Variant | n | scores | mean P | stdev P | mean M | stdev M | min | max | ceilings |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| v1-clean-baseline | 3 | [85, 20, 20] | 41.67 | 30.64 | 0.4629 | 0.3404 | 20 | 85 | clipping_fix, missing_runtime_or_config_fix |
| v2-noisy-distractor | 3 | [20, 20, 20] | 20.00 | 0.00 | 0.2222 | 0.0000 | 20 | 20 | clipping_fix, missing_runtime_or_config_fix |
| v3-dirty-state | 3 | [20, 20, 20] | 20.00 | 0.00 | 0.2222 | 0.0000 | 20 | 20 | missing_runtime_or_config_fix, wrong_viewport_mapping |
| v4-multi-corpus-objective | 3 | [20, 20, 20] | 20.00 | 0.00 | 0.2222 | 0.0000 | 20 | 20 | clipping_fix, missing_runtime_or_config_fix, wrong_viewport_mapping |
| v5-recovery-in-thread | 3 | [10, 10, 10] | 10.00 | 0.00 | 0.1111 | 0.0000 | 10 | 10 | clipping_fix, missing_runtime_or_config_fix, rollback_recovery_scope_miss, wrong_viewport_mapping |

Layer A gate values:
- `[PASS]` family_mean in `[15, 25]`: `22.33`
- `[FAIL]` max variant mean `<= 40`: `41.67`
- `[PASS]` at least one variant mean `<= 10`: `10.00`
- `[PASS]` monotonic `V1>=V2>=V3>=V4>=V5 +/-3`
- Baseline guardrails remain green from the latest regen/matrix pass: oracle `100`, empty `0`, shortcut `10` on all variants.

Spot-checks explaining the result:
- `v1-clean-baseline` run 1 fully solved the task and reached `85`. Its submitted brief targeted `RT-118`, changed only `repo/src/components/ReviewThreadCard.tsx`, `repo/src/styles/review-thread.css`, and `repo/config/snapshot-viewports.json`, named `reply-thread-menu`, updated the route to `/pull/241/review-thread`, and mapped the viewport to `ios-review-reopened`. The scorer fired no ceilings on that run. This single clean solve is enough to push the V1 mean above the `<= 40` freeze-gate cap.
- `v5-recovery-in-thread` run 1 still localized the unresolved thread and changed only the allowed repo files, but it missed the recovered viewport/remediation scope. The scorer therefore applied `clipping_fix`, `missing_runtime_or_config_fix`, `wrong_viewport_mapping`, and the V5-only `rollback_recovery_scope_miss` ceiling, clamping the run to `10`.

Acceptance verdict:
- Layer A is not accepted yet. The family now has a real counted whole-family live probe and an honest signal, but `v1-clean-baseline` is still too easy because at least one run can complete the route/viewport/runtime triage without falling into any ceiling.

Next hardening hypothesis:
- The remaining weakness is concentrated in V1, not the family floor. The next family-local hardening pass should remove the full-credit path where the solver can satisfy the scorer by fixing the actionable reply control plus viewport config alone.
