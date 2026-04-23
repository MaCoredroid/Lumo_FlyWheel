# Benchmark Run — `dead-flag-reachability-audit`

## Model under test

```bash
codex exec --model gpt-5.4 --reasoning-effort high
```

## Layer A target

- family mean in `[15, 25]`
- max variant mean `<= 40`
- at least one variant mean `<= 10`
- monotonic `V1 >= V2 >= V3 >= V4 >= V5` within `+/-3`
- oracle `>= 90`
- empty `= 0`
- shortcut `<= 30`

## attempt_00 — runtime family build and static verification

Scope completed inside this family only:

- authored `workspace_bundle/v1..v5`
- added family-local CLI `bin/cnb55-flag-audit`
- added deterministic scorer `verifiers/dead-flag-reachability-audit/score_reachability.py`
- added family-local regen, live-probe, and verification-matrix runners
- generated `verifier_data/dead-flag-reachability-audit/v1..v5`
- added `family.yaml`, `manifest.lock.json`, and verification-matrix artifacts

Commands run:

```bash
python3 verifiers/dead-flag-reachability-audit/regen_family.py
python3 verifiers/dead-flag-reachability-audit/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/dead-flag-reachability-audit/verification_matrix.md
python3 verifiers/dead-flag-reachability-audit/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/dead-flag-reachability-audit/verification_matrix_v5.md
```

Observed static sanity:

| variant | oracle | empty | shortcut |
| --- | ---: | ---: | ---: |
| v1-clean-baseline | 100 | 0 | 20 |
| v2-noisy-distractor | 100 | 0 | 20 |
| v3-dirty-state | 100 | 0 | 20 |
| v4-multi-corpus-objective | 100 | 0 | 20 |
| v5-recovery-in-thread | 100 | 0 | 20 |

Verification matrix headline results:

V1 matrix:

| trajectory | P_benchmark | M_training | integrity | ceilings |
| --- | ---: | ---: | ---: | --- |
| Oracle | 100 | 1.0000 | 0 | none |
| Empty | 0 | 0.0000 | 0 | `no_audit_file` |
| RAWR grounding_stripped | 30 | 0.3000 | 0 | `grep_only_reasoning` |
| Alias collapse | 25 | 0.2500 | 0 | `classification_miss`, `alias_collapse` |
| Force-legacy as live | 20 | 0.2000 | 0 | `classification_miss`, `parser_equals_runtime` |
| Delete-tests adversarial | 0 | 0.0000 | 1 | `tests_modified`, `immutable_slice_mutated` |

V5 matrix:

| trajectory | P_benchmark | M_training | integrity | ceilings |
| --- | ---: | ---: | ---: | --- |
| Oracle | 100 | 1.0000 | 0 | none |
| Empty | 0 | 0.0000 | 0 | `no_audit_file` |
| RAWR grounding_stripped | 30 | 0.3000 | 0 | `grep_only_reasoning` |
| Alias collapse | 25 | 0.2500 | 0 | `classification_miss`, `alias_collapse` |
| Force-legacy as live | 20 | 0.2000 | 0 | `classification_miss`, `parser_equals_runtime`, `sunk_cost_finish` |
| Delete-tests adversarial | 0 | 0.0000 | 1 | `tests_modified`, `immutable_slice_mutated` |

Interpretation:

- Layer B scaffolding exists and is executable.
- Oracle, empty, shortcut, and adversarial integrity behaviors are all in the
  expected bands.
- Static verification alone does not say whether Layer A is calibrated; the
  family still needs a real live probe against the authored bundle.

## attempt_01 — live probe after family-local runner hardening

Design change before the counted run:

- hardened `verifiers/dead-flag-reachability-audit/run_live_probe.py` so the
  family-local launcher invokes `codex -a never exec`
- hardened the same runner to record a timed-out `codex exec` as a scored row
  instead of crashing the whole family pass
- discarded the earlier partial/aborted launches; only the post-fix rerun
  below counts

Counted probe run:

- `probe_run_id`: `20260423T061908Z`
- model: `gpt-5.4`
- reasoning: `high`
- repeats: `3`

Commands run:

```bash
python3 verifiers/dead-flag-reachability-audit/run_live_probe.py --repeats 3 --timeout 900
cp benchmark_blueprints/families/dead-flag-reachability-audit/report/probe_runs.jsonl benchmark_blueprints/families/dead-flag-reachability-audit/report/attempt_01_probe_runs.jsonl
cp benchmark_blueprints/families/dead-flag-reachability-audit/report/probe_summary_latest.json benchmark_blueprints/families/dead-flag-reachability-audit/report/attempt_01_probe_summary.json
python3 verifiers/dead-flag-reachability-audit/probe_report.py benchmark_blueprints/families/dead-flag-reachability-audit/report/probe_summary_latest.json --out benchmark_blueprints/families/dead-flag-reachability-audit/report/attempt_01_probe_report.txt
```

Exact `codex exec` command template emitted by the runner:

```bash
codex -a never exec --skip-git-repo-check --json -m "gpt-5.4" -c 'reasoning_effort="high"' -c 'model_reasoning_effort="high"' -s workspace-write -C <workspace> "Read AGENTS.md, inspect the workspace evidence, author brief_input.json, run ./bin/cnb55-flag-audit validate brief_input.json, then run ./bin/cnb55-flag-audit submit brief_input.json. Do not modify files outside brief/, artifacts/, or brief_input.json."
```

Per-variant results:

| variant | scores | raw scores | mean P | mean M | notable ceilings |
| --- | --- | --- | ---: | ---: | --- |
| v1-clean-baseline | `[20, 20, 20]` | `[77, 96, 99]` | 20.00 | 0.2000 | `classification_miss` x3, `parser_equals_runtime` x3, `grep_only_reasoning` x3 |
| v2-noisy-distractor | `[20, 20, 20]` | `[99, 91, 95]` | 20.00 | 0.2000 | `classification_miss` x3, `parser_equals_runtime` x3, `grep_only_reasoning` x3, `stale_docs_truth` x1 |
| v3-dirty-state | `[20, 20, 20]` | `[94, 80, 99]` | 20.00 | 0.2000 | `classification_miss` x3, `parser_equals_runtime` x3, `grep_only_reasoning` x3, `sunk_cost_finish` x3 |
| v4-multi-corpus-objective | `[20, 20, 30]` | `[102, 99, 117]` | 23.33 | 0.2333 | `grep_only_reasoning` x3, `objective_drift` x2 |
| v5-recovery-in-thread | `[20, 20, 20]` | `[99, 99, 99]` | 20.00 | 0.2000 | `classification_miss` x3, `parser_equals_runtime` x3, `grep_only_reasoning` x3, `incident_blind_cleanup` x2, `sunk_cost_finish` x3 |

Layer A gate values:

- family mean: `20.67` -> PASS
- max variant mean: `23.33` -> PASS
- min variant mean: `20.00` -> FAIL (`>= 1` variant must be `<= 10`)
- monotonic: FAIL because `v3 = 20.00` and `v4 = 23.33`, which exceeds the
  allowed `+/-3` tolerance
- max observed `M_training` stdev: `0.0577`
- overall verdict: `LAYER_A_FAIL_HARDEN_NEEDED`

Spot-check from the counted run:

- `v4` run 3 is the strongest brief at `30/117`, but it still fires
  `objective_drift`. That run correctly marks `PREVIEW_FORCE_LEGACY` as `dead`
  while keeping `ENABLE_PREVIEW_V2` as a `partial` alias, yet it still misses
  the objective shift in the release corpus, so the family's hardest
  multi-corpus judgment is real rather than synthetic.
- `v5` stays pinned at `20` on all three runs because the model keeps
  classifying `PREVIEW_FORCE_LEGACY` as `partial` and then proposes generic
  cleanup/doc-telemetry actions instead of an incident-aware recovery posture.
  That reliably fires `incident_blind_cleanup` and `sunk_cost_finish`.

Interpretation:

- the probe is real and complete across all 5 variants; no timeout rows or
  integrity failures were recorded in the counted run
- the family already sits in the target mean band, but it is too flat: no
  variant falls to `<= 10`, and `v4` rebounds above `v3`
- the dominant miss is legitimate: the model often treats parser/reporting
  residue as enough to keep `PREVIEW_FORCE_LEGACY` in `partial`, and it still
  under-weights the recovery/objective evidence in `v4` and `v5`

Hypothesis for the next hardening pass:

- add one stronger but still defensible variant-local cap in `v5` for
  incident-blind recovery recommendations when the brief ignores rollback
  evidence and continues to treat `PREVIEW_FORCE_LEGACY` as an actionable
  cleanup target
- revisit `v4` so the objective-shift corpus more clearly differentiates
  release-freeze compatibility from independent runtime liveness without
  introducing rubric leakage
