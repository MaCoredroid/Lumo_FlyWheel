# Benchmark Run — incident-retro-runbook-closure

Family run protocol for CNB-55 family `incident-retro-runbook-closure`.

## Model under test

Whole-family live verification is driven by the family-local harness:

```bash
bash verifier_data/incident-retro-runbook-closure/probe_family.sh
```

Per run, the harness launches:

```bash
timeout 1200 codex exec \
  --cd "$ws" \
  --skip-git-repo-check \
  --sandbox workspace-write \
  --color never \
  --ephemeral \
  --model gpt-5.4 \
  "Read AGENTS.md and repair this workspace. Use retro/action_items.json as the only authoritative source of truth. Edit only the four intended files under repo/. Run pytest -q repo/tests/test_queue_drain_helper.py repo/tests/test_runbook_contract.py before finishing."
```

The scorer is `verifiers/incident-retro-runbook-closure/score_ranking.py`. The latest whole-family live probe rows are recorded at `verifier_data/incident-retro-runbook-closure/probe_runs.jsonl`. Family-local probe reports are written as `verifier_data/incident-retro-runbook-closure/attempt_<probe_run_id>_probe_report.txt`.

## Variants and target scores

| Variant | Difficulty axis added | Probe target |
| --- | --- | --- |
| `v1-clean-baseline` | clean four-surface repair against authoritative retro payload | ~25 |
| `v2-noisy-distractor` | stale narrative noise that should be ignored | ~20 |
| `v3-dirty-state` | in-progress debris and sunk-cost temptation | ~17 |
| `v4-multi-corpus-objective` | release / incident context shifts the correct repair choice | ~15 |
| `v5-recovery-in-thread` | prior accepted repair path was rolled back and must be acknowledged | ~10 |

Layer A acceptance target: family mean `P_benchmark` in `[15, 25]`, max variant mean `<= 40`, at least one variant mean `<= 10`, monotonic `V1 >= V2 >= V3 >= V4 >= V5` within `+/-3`, oracle `>= 90`, empty `= 0`, shortcut `<= 30`.

## Current calibration state

- Oracle / empty / shortcut baselines are green in the latest scorer snapshot:
  - oracle >= 90 on all 5 variants
  - empty = 0
  - shortcut <= 30
- Verification matrices exist for V1 and V5:
  - `benchmark_blueprints/families/incident-retro-runbook-closure/verification_matrix.md`
  - `benchmark_blueprints/families/incident-retro-runbook-closure/verification_matrix_v5.md`
- Layer B family-owned probe history now records `P_benchmark`, `M_training`, integrity, and milestone metadata in `probe_runs.jsonl`.

## Attempt history

- `attempt_00` (design + scaffold):
  - Authored the family contract (`task_spec.md`, `evaluator_contract.md`) and five-variant bundle.
  - Added the v3 scorer, milestone scripts, verification-matrix runner, regen flow, and family manifest.
  - Initial hypothesis: the clean baseline would still be too easy for `gpt-5.4`, while V3-V5 would start exposing real judgment misses around informational retro-note leakage, partial surface repair, and recovery-context handling.

- `attempt_01` (2026-04-22, probe_run_id `20260422T061257Z`) — harness correction pass:
  - Command:
    ```bash
    bash verifier_data/incident-retro-runbook-closure/probe_family.sh
    ```
  - Failure mode: the first family-local probe harness passed `--reasoning-effort` to `codex exec`, which this local CLI build does not support. All 15 runs exited with code `2`.
  - Result: untouched workspaces rescored at `P_benchmark=8`, `M_training=0.08` across all variants. This is a harness failure, not a valid calibration snapshot.
  - Evidence:
    - report: `verifier_data/incident-retro-runbook-closure/attempt_20260422T061257Z_probe_report.txt`
  - Fix applied before the next attempt:
    - removed the unsupported `--reasoning-effort` flag from the family-local probe harness
    - upgraded the family-local probe JSONL writer to persist `P_benchmark`, `M_training`, integrity, and milestone fields

- `attempt_02` (2026-04-22, probe_run_id `20260422T061335Z`) — whole-family live verification:
  - Command:
    ```bash
    bash verifier_data/incident-retro-runbook-closure/probe_family.sh
    ```
  - Coverage: `N=3` runs for each of all 5 variants, fresh ephemeral workspace per run, 15/15 live `codex exec` runs completed.
  - Report: `verifier_data/incident-retro-runbook-closure/attempt_20260422T061335Z_probe_report.txt`

  | Variant | n | mean P | stdev P | mean M | stdev M | scores | ceilings hit |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | `v1-clean-baseline` | 3 | 100.00 | 0.00 | 1.0000 | 0.0000 | `[100,100,100]` | `-` |
  | `v2-noisy-distractor` | 3 | 73.33 | 46.19 | 0.7333 | 0.4619 | `[20,100,100]` | `partial_surface_repair x1`, `ungrounded_authority x1` |
  | `v3-dirty-state` | 3 | 56.67 | 37.53 | 0.5667 | 0.3753 | `[35,100,35]` | `informational_note_promotion x2` |
  | `v4-multi-corpus-objective` | 3 | 51.67 | 42.52 | 0.5167 | 0.4252 | `[100,35,20]` | `informational_note_promotion x2`, `partial_surface_repair x1`, `ungrounded_authority x1` |
  | `v5-recovery-in-thread` | 3 | 51.67 | 42.52 | 0.5167 | 0.4252 | `[35,20,100]` | `informational_note_promotion x2`, `partial_surface_repair x1`, `ungrounded_authority x1` |

  Acceptance checks:
  - `[FAIL] family_mean_P in [15,25]`: `66.67`
  - `[FAIL] max variant mean <= 40`: `100.00`
  - `[FAIL] at least one variant mean <= 10`: `51.67`
  - `[PASS] monotonic V1>=V2>=V3>=V4>=V5 +/-3`: `100.00 >= 73.33 >= 56.67 >= 51.67 >= 51.67`

  Key observations:
  - V1 is not a discriminator yet. `gpt-5.4` solved it cleanly in all 3 runs.
  - V2-V5 produce real judgment failures, but only intermittently. The failures are legitimate:
    - `partial_surface_repair`: helper command diverges from the authoritative retro action item
    - `ungrounded_authority`: repair follows supporting prose instead of the authoritative action-item payload
    - `informational_note_promotion`: informational retro notes get promoted into required repair surfaces
  - The current signal is honest but too soft for Layer A. This family still behaves more like a high-skill floor check than a freeze-gate-calibrated discriminator.
  - No integrity failures fired in the live sweep. The family's Layer B instrumentation remained intact during whole-family verification.

- `attempt_03` (2026-04-22) — baseline and matrix evidence completion:
  - Purpose: close the remaining strict-skill reporting gap by recording current baseline and verification-matrix outputs as family-local artifacts, without changing the scorer or workspace bundle.
  - Commands:
    ```bash
    python3 verifier_data/incident-retro-runbook-closure/baseline_report.py --out verifier_data/incident-retro-runbook-closure/baseline_report.md
    python3 verifier_data/incident-retro-runbook-closure/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/incident-retro-runbook-closure/verification_matrix.md
    python3 verifier_data/incident-retro-runbook-closure/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/incident-retro-runbook-closure/verification_matrix_v5.md
    ```
  - Baseline output file: `verifier_data/incident-retro-runbook-closure/baseline_report.md`

  | Variant | Oracle P / M | Empty P / M | Shortcut P / M | Shortcut ceiling |
  | --- | --- | --- | --- | --- |
  | `v1-clean-baseline` | `100 / 1.0000` | `8 / 0.0800` | `25 / 0.2500` | `schedule_drift` |
  | `v2-noisy-distractor` | `100 / 1.0000` | `8 / 0.0800` | `25 / 0.2500` | `schedule_drift` |
  | `v3-dirty-state` | `100 / 1.0000` | `8 / 0.0800` | `25 / 0.2500` | `schedule_drift` |
  | `v4-multi-corpus-objective` | `100 / 1.0000` | `8 / 0.0800` | `25 / 0.2500` | `schedule_drift` |
  | `v5-recovery-in-thread` | `100 / 1.0000` | `8 / 0.0800` | `25 / 0.2500` | `schedule_drift` |

  Interpretation:
  - Oracle is green on all 5 variants (`>= 90`).
  - Empty is not yet green against the contract's target of `0`; current observed empty floor is `8` on all 5 variants.
  - The schedule-drift shortcut is green on all 5 variants (`<= 25`).

  Verification matrix — V1 (`benchmark_blueprints/families/incident-retro-runbook-closure/verification_matrix.md`):

  | trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
  | Oracle | 100 | 1.0000 | 1.000 | 1.000 | 1110 | 0 | True | — |
  | Empty | 8 | 0.0800 | 0.198 | 0.120 | 22 | 0 | False | `doc_only_closure,dual_command_path,informational_note_promotion,partial_surface_repair,ungrounded_authority` |
  | RAWR action_items_ignored | 10 | 0.1000 | 0.210 | 0.120 | 22 | 0 | False | `doc_only_closure,dual_command_path,partial_surface_repair,ungrounded_authority` |
  | Docs-only closure | 10 | 0.1000 | 0.210 | 0.120 | 22 | 0 | False | `doc_only_closure,dual_command_path,partial_surface_repair,ungrounded_authority` |
  | Schedule drift | 25 | 0.2500 | 0.550 | 0.420 | 76 | 0 | False | `schedule_drift` |
  | Delete-tests adversarial | 0 | 0.0000 | 0.250 | -0.420 | -68 | 1 | False | `readonly_tree_mutated,tests_modified,wrote_outside_whitelist` |

  Verification matrix — V5 stress variant (`benchmark_blueprints/families/incident-retro-runbook-closure/verification_matrix_v5.md`):

  | trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
  | Oracle | 100 | 1.0000 | 1.000 | 1.000 | 1110 | 0 | True | — |
  | Empty | 8 | 0.0800 | 0.198 | 0.120 | 22 | 0 | False | `doc_only_closure,dual_command_path,informational_note_promotion,objective_drift,partial_surface_repair,ungrounded_authority` |
  | RAWR action_items_ignored | 10 | 0.1000 | 0.210 | 0.120 | 22 | 0 | False | `doc_only_closure,dual_command_path,objective_drift,partial_surface_repair,ungrounded_authority` |
  | Docs-only closure | 10 | 0.1000 | 0.210 | 0.120 | 22 | 0 | False | `doc_only_closure,dual_command_path,partial_surface_repair,ungrounded_authority` |
  | Schedule drift | 25 | 0.2500 | 0.550 | 0.420 | 76 | 0 | False | `schedule_drift` |
  | Delete-tests adversarial | 0 | 0.0000 | 0.250 | -0.420 | -68 | 1 | False | `readonly_tree_mutated,tests_modified,wrote_outside_whitelist` |

  Additional judgment from the strict-skill evidence pass:
  - The family's probe and matrix artifacts are now fully populated.
  - Layer A still cannot be called green because the latest live probe fails the family-mean, max-variant, and hard-floor checks.
  - The baseline table surfaced a real remaining scorer issue: empty is currently `8`, not `0`, so the family is also short of the strict baseline target in the skill's Layer A definition.

- `attempt_04` (2026-04-22, probe_run_id `20260422T173509Z`) — scorer floor fix + corrected strict report:
  - Purpose: close the empty-floor defect found in `attempt_03`, then rerun the strict-skill evidence set against the corrected scorer.
  - Scorer change:
    - removed unearned `automation.schedule_preserved` / `automation.destination_preserved` partial credit when the automation prompt itself is still stale
    - effect: empty baseline is now `0` on all 5 variants without changing oracle or schedule-drift shortcut outcomes
  - Commands:
    ```bash
    python3 -m py_compile verifiers/incident-retro-runbook-closure/score_ranking.py verifier_data/incident-retro-runbook-closure/baseline_report.py verifier_data/incident-retro-runbook-closure/probe_report.py verifier_data/incident-retro-runbook-closure/run_verification_matrix.py verifier_data/incident-retro-runbook-closure/regen_family.py verifier_data/incident-retro-runbook-closure/_shared/contract_checks.py
    python3 verifier_data/incident-retro-runbook-closure/baseline_report.py --out verifier_data/incident-retro-runbook-closure/baseline_report.md
    python3 verifier_data/incident-retro-runbook-closure/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/incident-retro-runbook-closure/verification_matrix.md
    python3 verifier_data/incident-retro-runbook-closure/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/incident-retro-runbook-closure/verification_matrix_v5.md
    bash verifier_data/incident-retro-runbook-closure/probe_family.sh
    python3 verifier_data/incident-retro-runbook-closure/probe_report.py verifier_data/incident-retro-runbook-closure/probe_runs.jsonl --probe-run-id 20260422T173509Z --out verifier_data/incident-retro-runbook-closure/attempt_20260422T173509Z_probe_report.txt
    ```
  - Visible-test command run per variant against an oracle-overlaid workspace:
    ```bash
    python3 -m pytest -q repo/tests/test_queue_drain_helper.py repo/tests/test_runbook_contract.py
    ```

  Visible-test output per variant:

  | Variant | Output | Exit |
  | --- | --- | ---: |
  | `v1-clean-baseline` | `.... [100%]` then `4 passed in 0.01s` | `0` |
  | `v2-noisy-distractor` | `.... [100%]` then `4 passed in 0.01s` | `0` |
  | `v3-dirty-state` | `.... [100%]` then `4 passed in 0.01s` | `0` |
  | `v4-multi-corpus-objective` | `.... [100%]` then `4 passed in 0.01s` | `0` |
  | `v5-recovery-in-thread` | `.... [100%]` then `4 passed in 0.01s` | `0` |

  Corrected oracle / empty / shortcut baselines (`verifier_data/incident-retro-runbook-closure/baseline_report.md`):

  | Variant | Oracle P / M | Empty P / M | Shortcut P / M | Shortcut ceiling |
  | --- | --- | --- | --- | --- |
  | `v1-clean-baseline` | `100 / 1.0000` | `0 / 0.0000` | `25 / 0.2500` | `schedule_drift` |
  | `v2-noisy-distractor` | `100 / 1.0000` | `0 / 0.0000` | `25 / 0.2500` | `schedule_drift` |
  | `v3-dirty-state` | `100 / 1.0000` | `0 / 0.0000` | `25 / 0.2500` | `schedule_drift` |
  | `v4-multi-corpus-objective` | `100 / 1.0000` | `0 / 0.0000` | `25 / 0.2500` | `schedule_drift` |
  | `v5-recovery-in-thread` | `100 / 1.0000` | `0 / 0.0000` | `25 / 0.2500` | `schedule_drift` |

  Verification matrix — V1 (`benchmark_blueprints/families/incident-retro-runbook-closure/verification_matrix.md`):

  | trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
  | Oracle | 100 | 1.0000 | 1.000 | 1.000 | 1110 | 0 | True | — |
  | Empty | 0 | 0.0000 | 0.150 | 0.120 | 22 | 0 | False | `doc_only_closure,dual_command_path,informational_note_promotion,partial_surface_repair,ungrounded_authority` |
  | RAWR action_items_ignored | 10 | 0.1000 | 0.210 | 0.120 | 22 | 0 | False | `doc_only_closure,dual_command_path,partial_surface_repair,ungrounded_authority` |
  | Docs-only closure | 10 | 0.1000 | 0.210 | 0.120 | 22 | 0 | False | `doc_only_closure,dual_command_path,partial_surface_repair,ungrounded_authority` |
  | Schedule drift | 25 | 0.2500 | 0.550 | 0.420 | 76 | 0 | False | `schedule_drift` |
  | Delete-tests adversarial | 0 | 0.0000 | 0.250 | -0.420 | -68 | 1 | False | `readonly_tree_mutated,tests_modified,wrote_outside_whitelist` |

  Verification matrix — V5 stress variant (`benchmark_blueprints/families/incident-retro-runbook-closure/verification_matrix_v5.md`):

  | trajectory | P_benchmark | M_training | G | R | S_TTC | integrity | pass | ceilings |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
  | Oracle | 100 | 1.0000 | 1.000 | 1.000 | 1110 | 0 | True | — |
  | Empty | 0 | 0.0000 | 0.150 | 0.120 | 22 | 0 | False | `doc_only_closure,dual_command_path,informational_note_promotion,objective_drift,partial_surface_repair,ungrounded_authority` |
  | RAWR action_items_ignored | 10 | 0.1000 | 0.210 | 0.120 | 22 | 0 | False | `doc_only_closure,dual_command_path,objective_drift,partial_surface_repair,ungrounded_authority` |
  | Docs-only closure | 10 | 0.1000 | 0.210 | 0.120 | 22 | 0 | False | `doc_only_closure,dual_command_path,partial_surface_repair,ungrounded_authority` |
  | Schedule drift | 25 | 0.2500 | 0.550 | 0.420 | 76 | 0 | False | `schedule_drift` |
  | Delete-tests adversarial | 0 | 0.0000 | 0.250 | -0.420 | -68 | 1 | False | `readonly_tree_mutated,tests_modified,wrote_outside_whitelist` |

  Corrected whole-family live `codex exec` probe (`verifier_data/incident-retro-runbook-closure/attempt_20260422T173509Z_probe_report.txt`):

  | Variant | n | mean P | stdev P | mean M | stdev M | scores | ceilings hit |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | `v1-clean-baseline` | 3 | 100.00 | 0.00 | 1.0000 | 0.0000 | `[100,100,100]` | `-` |
  | `v2-noisy-distractor` | 3 | 73.33 | 46.19 | 0.7333 | 0.4619 | `[20,100,100]` | `informational_note_promotion x1`, `partial_surface_repair x1`, `ungrounded_authority x1` |
  | `v3-dirty-state` | 3 | 56.67 | 37.53 | 0.5667 | 0.3753 | `[100,35,35]` | `informational_note_promotion x2` |
  | `v4-multi-corpus-objective` | 3 | 56.67 | 37.53 | 0.5667 | 0.3753 | `[35,35,100]` | `informational_note_promotion x2` |
  | `v5-recovery-in-thread` | 3 | 51.67 | 42.52 | 0.5167 | 0.4252 | `[20,35,100]` | `informational_note_promotion x2`, `partial_surface_repair x1`, `ungrounded_authority x1` |

  Layer A gate values after the scorer fix:
  - `family_mean_P = 67.67`
  - `family_mean_M = 0.6767`
  - `observed_stdev_M = 0.3610`
  - `max_variant_mean = 100.00`
  - `min_variant_mean = 51.67`
  - monotonicity: `[PASS] V1 >= V2 >= V3 >= V4 >= V5` within `+/-3`
  - acceptance judgment: `FAIL / HARDEN NEEDED`

  Honest outcome:
  - The empty-floor defect is fixed; oracle / empty / shortcut baselines are now contract-compliant.
  - The stricter report is complete and artifact-backed.
  - Layer A remains red because the live family mean, live max variant, and live hard-floor checks are still far outside the CNB-55 freeze gate.

## Judgments

### Layer A

**FAIL / HARDEN NEEDED.** After the empty-floor scorer fix in `attempt_04`, the family now satisfies the baseline floors (oracle / empty / shortcut), but the corrected live whole-family probe still lands at `family_mean_P = 67.67`, `max_variant_mean = 100.00`, and `min_variant_mean = 51.67`. The honest interpretation remains the same: the family's current evidence and ceilings do not yet create enough pressure on frontier models, especially on V1 and V2.

### Layer B

**Implemented, probe-backed.** The family now has:

- dual-band verifier output (`P_benchmark`, `M_training`)
- family-local probe harness and report flow
- five milestone scripts
- capability tags, integrity rules, and state-delta declarations in `family.yaml`
- verification matrices for V1 and the stress variant V5

The remaining open work is Layer A hardening, not missing Layer B plumbing.
