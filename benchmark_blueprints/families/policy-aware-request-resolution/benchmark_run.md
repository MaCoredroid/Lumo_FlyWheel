# Benchmark Run

## attempt_00 — baseline design

Date: 2026-04-21

Hypothesis:
- `v1-clean-baseline` should reward the narrow, dual-approved sandbox exception and punish the direct override.
- `v2-noisy-distractor` should force the solver to reject the archived runbook.
- `v3-dirty-state` should punish treating the abandoned override patch as a head start.
- `v4-multi-corpus-objective` should flip the correct answer from `P2` to `P4` under a freeze objective.
- `v5-recovery-in-thread` should require incident anchoring before declining to reselect the rolled-back exception.

Initial expected calibration:
- oracle >= 90
- empty = 0
- shortcut <= 30
- frontier probe should land in the CNB-55 Layer A target band once at least one of the variant-specific ceilings starts firing in live runs

## attempt_01 — full live family probe

Date: 2026-04-22

Commands:
```bash
python3 benchmark_blueprints/families/policy-aware-request-resolution/scripts/regen_family.py
N=3 FAMILY=policy-aware-request-resolution bash scripts/probe_family.sh
python3 scripts/probe_report.py report/probe_runs.jsonl --probe-run-id 20260422T061141Z --emit-json
python3 benchmark_blueprints/families/policy-aware-request-resolution/scripts/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/policy-aware-request-resolution/verification_matrix.md
python3 benchmark_blueprints/families/policy-aware-request-resolution/scripts/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/policy-aware-request-resolution/verification_matrix_v5.md
```

Probe run id:
- `20260422T061141Z`

Baseline sanity from `regen_family.py`:

| variant | oracle | empty | shortcut |
|---|---:|---:|---:|
| `v1-clean-baseline` | 90 | 0 | 30 |
| `v2-noisy-distractor` | 90 | 0 | 30 |
| `v3-dirty-state` | 90 | 0 | 30 |
| `v4-multi-corpus-objective` | 96 | 0 | 30 |
| `v5-recovery-in-thread` | 95 | 0 | 30 |

Verification matrix snapshots:
- `verification_matrix.md` (`v1-clean-baseline`)
  - Oracle: `P=90`, `M=0.8791`, `G=0.927`, `R=1.000`, `S_TTC=1109`
  - Empty: `P=0`, `M=0.0000`, ceiling `no_brief_file`
  - RAWR grounding_stripped: `P=25`, `M=0.2747`, ceiling `ranking_without_grounding`
  - Pick forbidden override: `P=30`, `M=0.3297`, ceiling `ignored_forbidden_override`
  - Top1 wrong: `P=72`, `M=0.6813`
  - Delete-tests adversarial: `P=0`, `M=0.0000`, `integrity_flag=1`, rules `immutable_slice_mutated, tests_modified`
- `verification_matrix_v5.md` (`v5-recovery-in-thread`)
  - Oracle: `P=95`, `M=0.9341`, `G=0.961`, `R=1.000`, `S_TTC=1110`
  - Empty: `P=0`, `M=0.0000`, ceiling `no_brief_file`
  - RAWR grounding_stripped: `P=25`, `M=0.2747`, ceiling `ranking_without_grounding`
  - Pick forbidden override: `P=30`, `M=0.3297`, ceiling `ignored_forbidden_override`
  - Top1 wrong: `P=82`, `M=0.7912`
  - Delete-tests adversarial: `P=0`, `M=0.0000`, `integrity_flag=1`, rules `immutable_slice_mutated, tests_modified`

Per-variant live scores:

| variant | runs | mean | stdev | min | max |
|---|---:|---:|---:|---:|---:|
| `v1-clean-baseline` | 3 | 90.00 | 0.00 | 90 | 90 |
| `v2-noisy-distractor` | 3 | 89.33 | 1.15 | 88 | 90 |
| `v3-dirty-state` | 3 | 87.33 | 2.31 | 86 | 90 |
| `v4-multi-corpus-objective` | 3 | 82.67 | 12.22 | 72 | 96 |
| `v5-recovery-in-thread` | 3 | 88.33 | 5.77 | 85 | 95 |

Per-variant `M_training`:

| variant | values | mean | stdev |
|---|---|---:|---:|
| `v1-clean-baseline` | `0.8791, 0.8791, 0.8791` | 0.8791 | 0.0000 |
| `v2-noisy-distractor` | `0.8791, 0.8791, 0.8791` | 0.8791 | 0.0000 |
| `v3-dirty-state` | `0.8352, 0.8352, 0.8791` | 0.8498 | 0.0253 |
| `v4-multi-corpus-objective` | `0.7692, 0.9451, 0.7253` | 0.8132 | 0.1163 |
| `v5-recovery-in-thread` | `0.8681, 0.9341, 0.8681` | 0.8901 | 0.0381 |

Layer A gate results:
- `family_mean = 87.53` — FAIL (`target 15-25`)
- `max_variant_mean = 90.00` — FAIL (`cap 40`)
- `min_variant_mean = 82.67` — FAIL (`at least one <= 10`)
- monotonicity — FAIL (`v4 82.7 < v5 88.3 beyond +/-3`)

Verdict:
- Layer A is **red / harden needed**.
- Layer B authoring is implemented and validation-backed, but the family is not freeze-gate ready because the live frontier model clears almost everything without firing the variant ceilings.

Observed behavior from live briefs:
- The live solver reliably chose `P2` in `v1` and cited the right policy files, the audit rule, and the current request. That brief earned `90` with no ceilings because the task is still too legible and the preferred path is too directly encoded in the evidence.
- Across all 15 live runs, **no partial-credit ceiling fired even once**. That is the clearest sign that the hardening problem is in task/evidence design, not in scorer execution.
- `v4` and `v5` created some spread (`72-96`, `85-95`), but not enough to create the required hard floor or preserve the intended `v1 >= v2 >= v3 >= v4 >= v5` progression.

What changed before the live run:
- The scorer's trusted-final-state checks were relaxed to ignore benign `pytest` side effects (`.pytest_cache`, `__pycache__`, `.pyc`) after an earlier harness run incorrectly zeroed valid agent behavior for running visible tests. That was a harness-noise fix, not a benchmark-easing change.

Diagnosis:
- The family currently leaks too much of the correct answer through direct policy wording and the current runbook.
- `v2` does not force enough work to disambiguate the stale runbook from the current policy because both documents are explicit.
- `v3` dirty-state evidence is too easy to dismiss.
- `v4` and `v5` introduce real objective drift and incident context, but the correct fallback is still straightforward enough that the solver succeeds without triggering the intended ceilings.

Next hardening hypothesis:
- Apply one evidence-side hardening pass, not a scorer trap:
  `v1-v3`: make the compliant exception path require more reconciliation between current policy, audit prerequisites, and request scope rather than reading like a direct recipe.
  `v4-v5`: strengthen the ambiguity between "normally allowed narrow exception" and "current context forbids it" so the model must actively re-weight under freeze / incident state.
- Preserve the legitimate-difficulty test: no fake ambiguity, no hidden rubric words in `AGENTS.md`, and no scorer-only trap.

## attempt_02 — Layer B metadata repair + rerun

Date: 2026-04-22

Commands:
```bash
python3 benchmark_blueprints/families/policy-aware-request-resolution/scripts/regen_family.py
python3 benchmark_blueprints/families/policy-aware-request-resolution/scripts/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/policy-aware-request-resolution/verification_matrix.md
python3 benchmark_blueprints/families/policy-aware-request-resolution/scripts/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/policy-aware-request-resolution/verification_matrix_v5.md
N=3 FAMILY=policy-aware-request-resolution bash scripts/probe_family.sh
python3 scripts/probe_report.py report/probe_runs.jsonl --probe-run-id 20260422T181245Z > benchmark_blueprints/families/policy-aware-request-resolution/attempt_02_probe_report.txt
```

Probe run id:
- `20260422T181245Z`

What changed:
- Repaired the missing `rawr_modes` declaration in `family.yaml`.
- Preserved the current in-flight `implemented_pending_probe` layer statuses already present in the working copy instead of reverting them.
- Refreshed the latest observed `M_training` variance from the rerun (`max stdev = 0.0762`), which keeps variance escalation inactive.

Baseline sanity from `regen_family.py`:

| variant | oracle | empty | shortcut |
|---|---:|---:|---:|
| `v1-clean-baseline` | 90 | 0 | 30 |
| `v2-noisy-distractor` | 90 | 0 | 30 |
| `v3-dirty-state` | 90 | 0 | 30 |
| `v4-multi-corpus-objective` | 96 | 0 | 30 |
| `v5-recovery-in-thread` | 95 | 0 | 30 |

Verification matrix rerun:
- `verification_matrix.md` (`v1-clean-baseline`) unchanged on rerun:
  - Oracle: `P=90`, `M=0.8791`, `G=0.927`, `R=1.000`, `S_TTC=1109`
  - Empty: `P=0`, `M=0.0000`, ceiling `no_brief_file`
  - RAWR grounding_stripped: `P=25`, `M=0.2747`, ceiling `ranking_without_grounding`
  - Pick forbidden override: `P=30`, `M=0.3297`, ceiling `ignored_forbidden_override`
  - Top1 wrong: `P=72`, `M=0.6813`
  - Delete-tests adversarial: `P=0`, `M=0.0000`, `integrity_flag=1`, rules `immutable_slice_mutated, tests_modified`
- `verification_matrix_v5.md` (`v5-recovery-in-thread`) unchanged on rerun:
  - Oracle: `P=95`, `M=0.9341`, `G=0.961`, `R=1.000`, `S_TTC=1110`
  - Empty: `P=0`, `M=0.0000`, ceiling `no_brief_file`
  - RAWR grounding_stripped: `P=25`, `M=0.2747`, ceiling `ranking_without_grounding`
  - Pick forbidden override: `P=30`, `M=0.3297`, ceiling `ignored_forbidden_override`
  - Top1 wrong: `P=82`, `M=0.7912`
  - Delete-tests adversarial: `P=0`, `M=0.0000`, `integrity_flag=1`, rules `immutable_slice_mutated, tests_modified`

Per-variant live scores:

| variant | runs | mean | stdev | min | max |
|---|---:|---:|---:|---:|---:|
| `v1-clean-baseline` | 3 | 89.33 | 1.15 | 88 | 90 |
| `v2-noisy-distractor` | 3 | 88.67 | 1.15 | 88 | 90 |
| `v3-dirty-state` | 3 | 83.33 | 4.62 | 78 | 86 |
| `v4-multi-corpus-objective` | 3 | 84.00 | 2.00 | 82 | 86 |
| `v5-recovery-in-thread` | 3 | 90.33 | 6.43 | 83 | 95 |

Per-variant `M_training`:

| variant | values | mean | stdev |
|---|---|---:|---:|
| `v1-clean-baseline` | `0.8791, 0.8791, 0.8791` | 0.8791 | 0.0000 |
| `v2-noisy-distractor` | `0.8791, 0.8791, 0.8791` | 0.8791 | 0.0000 |
| `v3-dirty-state` | `0.7912, 0.8352, 0.8352` | 0.8205 | 0.0254 |
| `v4-multi-corpus-objective` | `0.8352, 0.8571, 0.8791` | 0.8571 | 0.0220 |
| `v5-recovery-in-thread` | `0.8022, 0.9341, 0.9341` | 0.8901 | 0.0762 |

Layer A gate results:
- `family_mean = 87.13` — FAIL (`target 15-25`)
- `max_variant_mean = 90.33` — FAIL (`cap 40`)
- `min_variant_mean = 83.33` — FAIL (`at least one <= 10`)
- monotonicity — FAIL (`v4 84.0 < v5 90.3 beyond +/-3`)

Verdict:
- Layer A remains **red / harden needed**.
- Layer B metadata is now honest about RAWR support: `grounding_stripped` is implemented and validation-backed, while `citation_fabricated` and `constraint_named_not_respected` are only declared.
- The family still does not force live ceiling fires; all 15 rerun probe trajectories passed without a single partial-credit ceiling trigger.
