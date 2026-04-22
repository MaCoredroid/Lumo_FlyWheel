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

- Oracle / empty / shortcut baselines were already green before the live sweep:
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

## Judgments

### Layer A

**FAIL / HARDEN NEEDED.** The family remains well above the CNB-55 freeze window. The honest interpretation of `attempt_02` is not that the family is broken; it is that the present evidence and ceilings do not yet create enough pressure on frontier models, especially on V1 and V2.

### Layer B

**Implemented, probe-backed.** The family now has:

- dual-band verifier output (`P_benchmark`, `M_training`)
- family-local probe harness and report flow
- five milestone scripts
- capability tags, integrity rules, and state-delta declarations in `family.yaml`
- verification matrices for V1 and the stress variant V5

The remaining open work is Layer A hardening, not missing Layer B plumbing.
