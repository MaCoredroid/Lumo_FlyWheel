
# Benchmark Run

## attempt_00 — baseline design

Hypotheses:

- `v1-clean-baseline` should discriminate honest ESM repairs from docs-only churn because the visible tests still require default + named plugin loading.
- `v2-noisy-distractor` should punish stale CommonJS anchoring.
- `v3-dirty-state` should punish namespace coercion and scratch-patch completion.
- `v4-multi-corpus-objective` should force docs + generated evidence alignment, not just code repair.
- `v5-recovery-in-thread` should punish helper-path or malformed-module regressions that echo the seeded incident.

## attempt_01 — legacy single-workspace probe

- model: `gpt-5.4`
- reasoning: `high`
- agent: `019da338-699a-7a32-aff2-1dd39f3266aa`
- result: over target under the original evaluator because build/test/evidence success was over-credited without a five-variant family shape or Layer B readiness.

## attempt_02 — hardened single-workspace rerun

- model: `gpt-5.4`
- reasoning: `high`
- agent: `019da33f-cf6e-7fd0-9e34-6a8932532223`
- result: `20/100` under the hardened single-workspace evaluator
- judgment: the task signal was promising, but the family still lacked the standard five-variant bundle, verifier-data layout, and Layer B emission contract.

## attempt_03a — Layer B flywheel-readiness upgrade

Shipped changes:

- rebuilt the family into the standard five-variant workspace bundle
- added `family.yaml`, `manifest.lock.json`, milestone scripts, and dual-band `cnb55.verify_result.v3` scoring
- added family-local verifier data with immutable-tree hashes, oracle overlays, and generated capture expectations
- added `verification_matrix.md` for `v1-clean-baseline`
- added `verification_matrix_v5-recovery-in-thread.md` for the stress variant

Local acceptance evidence:

- oracle / empty / shortcut baselines are now encoded in the shipped verification matrices
- local oracle sweep after the Layer B rebuild: all five variants scored `100`, and all five empty baselines scored `0`
- Layer B is implemented locally and traceable through `family.yaml`
- a fresh family-wide live `codex exec` probe has not yet been rerun after this rebuild

Layer A status:

- historical single-workspace hardening reached the intended `~20/100` band
- post-rebuild five-variant live probe: pending

Hardening decisions already applied:

- made built-dist execution self-contained by scoring after removing the source tree
- made malformed-plugin rejection and helper.cjs compatibility first-class hidden checks
- made docs and generated CLI capture part of the end-state contract
- kept immutable evidence and tests outside the allowed write surface

## attempt_04 — whole-family live codex exec probe

Date: 2026-04-23

Scope:

- family: `esm-plugin-loader-modernization`
- model: `gpt-5.4`
- reasoning: `high`
- variants: all five shipped variants
- repeats: `N=3`
- counted probe id: `20260423T194727Z`
- status: completed; live probe is no longer pending

Setup note:

- A pre-count smoke run exposed a scorer integrity false positive where generated `dist/` build outputs were treated as writes outside the whitelist even though `npm run build` is required by the task.
- The scorer was corrected before the counted run to ignore generated `dist/` outputs in the write-surface check.
- Only the post-change whole-family probe `20260423T194727Z` is counted below.

Exact commands:

```bash
N=1 VARIANTS='v1-clean-baseline' benchmark_blueprints/families/esm-plugin-loader-modernization/scripts/probe_family.sh
N=1 VARIANTS='v1-clean-baseline' benchmark_blueprints/families/esm-plugin-loader-modernization/scripts/probe_family.sh
N=3 benchmark_blueprints/families/esm-plugin-loader-modernization/scripts/probe_family.sh
python3 benchmark_blueprints/families/esm-plugin-loader-modernization/scripts/probe_report.py benchmark_blueprints/families/esm-plugin-loader-modernization/report/probe_runs.jsonl --probe-run-id 20260423T194727Z --emit-json
python3 verifiers/esm-plugin-loader-modernization/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/esm-plugin-loader-modernization/verification_matrix.md
python3 verifiers/esm-plugin-loader-modernization/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/esm-plugin-loader-modernization/verification_matrix_v5-recovery-in-thread.md
```

Per-variant numeric results:

| variant | n | scores | mean | stdev | min | max | M_training mean | raw mean | repeated ceilings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v1-clean-baseline` | 3 | `[20,20,20]` | 20.00 | 0.00 | 20 | 20 | 0.2222 | 80.00 | `docs_or_evidence_drift` x3, `malformed_plugin_accepted` x3, `unsafe_namespace_coercion` x3 |
| `v2-noisy-distractor` | 3 | `[20,20,20]` | 20.00 | 0.00 | 20 | 20 | 0.2222 | 80.00 | `docs_or_evidence_drift` x3, `malformed_plugin_accepted` x3, `unsafe_namespace_coercion` x3 |
| `v3-dirty-state` | 3 | `[20,20,20]` | 20.00 | 0.00 | 20 | 20 | 0.2222 | 80.00 | `docs_or_evidence_drift` x3, `malformed_plugin_accepted` x3, `unsafe_namespace_coercion` x3 |
| `v4-multi-corpus-objective` | 3 | `[20,20,20]` | 20.00 | 0.00 | 20 | 20 | 0.2222 | 80.00 | `docs_or_evidence_drift` x3, `malformed_plugin_accepted` x3, `unsafe_namespace_coercion` x3 |
| `v5-recovery-in-thread` | 3 | `[25,20,25]` | 23.33 | 2.89 | 20 | 25 | 0.2593 | 80.00 | `docs_or_evidence_drift` x3, `incident_blind_helper_regression` x3, `malformed_plugin_accepted` x3, `unsafe_namespace_coercion` x1 |

Layer A gate values:

| gate | value | threshold | result |
|---|---:|---:|---|
| family mean | 20.67 | 15.0-25.0 | PASS |
| max variant mean | 23.33 | <= 40.0 | PASS |
| min variant mean | 20.00 | at least one <= 10.0 | FAIL |
| monotonic variant progression | `v4` 20.00 < `v5` 23.33 beyond +/-3.0 | non-increasing within tolerance | FAIL |

Layer A judgment:

- Overall: `HARDEN NEEDED`
- The family is in the target aggregate band, but it does not yet satisfy the hard-variant floor or monotonic variant progression gate.
- The live probe is completed, not pending.

Verification matrix outputs:

`verification_matrix.md` (`v1-clean-baseline`):

| trajectory | P_benchmark | M_training | integrity | pass | ceilings |
|---|---:|---:|---:|---|---|
| Oracle | 100 | 1.0000 | 0 | True | — |
| Empty | 0 | 0.0000 | 0 | False | `docs_or_evidence_drift`, `helper_path_regression`, `malformed_plugin_accepted`, `no_submission`, `unsafe_namespace_coercion` |
| RAWR grounding_stripped | 50 | 0.5556 | 0 | False | `docs_or_evidence_drift` |
| Pick ceiling: source-only dist shortcut | 25 | 0.2778 | 0 | False | `helper_path_regression`, `malformed_plugin_accepted`, `source_path_only` |
| Top1 wrong: unsafe namespace coercion | 20 | 0.2222 | 0 | False | `malformed_plugin_accepted`, `no_visible_green`, `unsafe_namespace_coercion` |
| Delete-tests adversarial | 0 | 0.0000 | 1 | False | `H=immutable_slice_mutated`, `tests_modified`, `write_outside_whitelist` |

`verification_matrix_v5-recovery-in-thread.md`:

| trajectory | P_benchmark | M_training | integrity | pass | ceilings |
|---|---:|---:|---:|---|---|
| Oracle | 100 | 1.0000 | 0 | True | — |
| Empty | 0 | 0.0000 | 0 | False | `docs_or_evidence_drift`, `helper_path_regression`, `incident_blind_helper_regression`, `malformed_plugin_accepted`, `no_submission`, `unsafe_namespace_coercion` |
| RAWR grounding_stripped | 50 | 0.5556 | 0 | False | `docs_or_evidence_drift` |
| Pick ceiling: source-only dist shortcut | 25 | 0.2778 | 0 | False | `helper_path_regression`, `incident_blind_helper_regression`, `malformed_plugin_accepted`, `source_path_only` |
| Top1 wrong: unsafe namespace coercion | 20 | 0.2222 | 0 | False | `incident_blind_helper_regression`, `malformed_plugin_accepted`, `no_visible_green`, `unsafe_namespace_coercion` |
| Delete-tests adversarial | 0 | 0.0000 | 1 | False | `H=immutable_slice_mutated`, `tests_modified`, `write_outside_whitelist` |

Artifacts:

- `benchmark_blueprints/families/esm-plugin-loader-modernization/report/probe_runs.jsonl`
- `benchmark_blueprints/families/esm-plugin-loader-modernization/report/attempt_04_live_probe_runs.jsonl`
- `benchmark_blueprints/families/esm-plugin-loader-modernization/report/attempt_04_live_probe_report.txt`
- `benchmark_blueprints/families/esm-plugin-loader-modernization/report/probe_run_logs/20260423T194727Z-*.last_message.txt`
