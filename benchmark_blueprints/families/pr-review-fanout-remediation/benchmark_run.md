# Benchmark Run: `pr-review-fanout-remediation`

## Run Metadata

- Run type: child-subagent family implementation pass
- Model: `gpt-5.4`
- Reasoning effort: `high`
- Scope restriction:
  - `benchmark_blueprints/families/pr-review-fanout-remediation/**`
  - `verifiers/pr-review-fanout-remediation/**`
  - `verifier_data/pr-review-fanout-remediation/**`

## attempt_00 — prose-only family stub

Initial state before this pass:

- `task_spec.md`, `evaluator_contract.md`, and `benchmark_run.md` existed
- no workspace bundle
- no family-local scorer / verifier wrapper
- no verifier data
- no `family.yaml`
- no `manifest.lock.json`

Recorded baseline from the earlier prose-only child attempt:

- final score: `22 / 100`
- strongest signal: logically plausible triage narrative, but no literal thread
  ids, no real code / test patch, and no family-local artifact bundle

Conclusion:

- Layer A: not calibratable yet because there was no runnable family asset pack
- Layer B: missing almost every required artifact

## attempt_01 — family asset pack + Layer B scaffolding

Implemented during this pass:

- 5-variant workspace bundle under `workspace_bundle/`
- deterministic dual-band scorer:
  - `verifiers/pr-review-fanout-remediation/score_review_fanout.py`
- verifier wrapper:
  - `verifiers/pr-review-fanout-remediation/verify.sh`
- family-local regen + matrix runners:
  - `verifiers/pr-review-fanout-remediation/regen_family.py`
  - `verifiers/pr-review-fanout-remediation/run_verification_matrix.py`
  - `verifiers/pr-review-fanout-remediation/run_live_probe.py`
- family-local verifier data:
  - gold state, manifests, oracle overlays, hidden tests, milestone scripts
- Layer B contract files:
  - `family.yaml`
  - `manifest.lock.json`
  - `verification_matrix.md`
  - `verification_matrix_v5-recovery-in-thread.md`

### Baseline Numeric Checks

Per variant (`v1`..`v5`):

| variant | oracle | empty | code-only shortcut |
|---|---:|---:|---:|
| v1-clean-baseline | 100 | 0 | 20 |
| v2-noisy-distractor | 100 | 0 | 20 |
| v3-dirty-state | 100 | 0 | 20 |
| v4-multi-corpus-objective | 100 | 0 | 20 |
| v5-recovery-in-thread | 100 | 0 | 20 |

### Verification Matrix Spot Checks

V1:

- Oracle: `P=100`, `M=1.0000`, `pass=true`
- Empty: `P=0`, `M=0.0000`, `pass=false`
- Code-only shortcut: `20`
- Wrong duplicate-thread triage: `20`
- Generic replies without thread mapping: `35`
- Delete-tests adversarial: `P=0`, `M=0.0000`, `integrity_flag=1`, `pass=false`

V5 stress rerun:

- Oracle remained `100`
- Empty row correctly picked up `incident_blind_recovery`
- Code-only shortcut stayed at `20`
- Delete-tests adversarial again forced `integrity_flag=1`, `M=0.0000`,
  `pass=false`

### What This Pass Did Not Do

- No live `codex exec` family probe was launched
- No family-mean / monotonicity / max-variant freeze-gate numbers were claimed
- No cross-family infrastructure was edited

### Current Judgment

- Layer B: materially implemented and locally verified
- Layer A: still pending honest probe calibration on the real evaluation loop

Next honest step, owned by the parent loop rather than this subagent:

- run `python3 verifiers/pr-review-fanout-remediation/run_live_probe.py --n 3`
  and record family-mean / monotonicity outcomes

## attempt_02 — initial live probe launch, invalidated

Exact command:

```bash
python3 verifiers/pr-review-fanout-remediation/run_live_probe.py --n 3 --timeout 1800
```

Observed probe run id:

- `20260423T230413Z`

What happened:

- this first live launch exposed a scorer-shape mismatch rather than honest
  model difficulty
- the live solver wrote `artifacts/review/thread_triage.json` as an object with
  a top-level `threads` list and `classification` keys, while the scorer still
  expected a root list with `disposition`
- the live solver also wrote semantically correct obsolete / skipped-language
  phrases that the scorer rejected because it was matching only a narrow
  wording shape

Concrete evidence recorded in the logs:

- `report/live_probe_logs/20260423T230413Z-v3-dirty-state-run1.log` showed the
  solver classifying `T-214-02` as a duplicate and `T-214-03` as obsolete, but
  the pre-fix scorer failed to recognize that artifact shape

Disposition:

- this run was killed after the diagnostic confirmed the scorer brittleness
- it is preserved under `report/live_probe_logs/`
- it does **not** count toward calibration because the scorer changed afterward

## attempt_03 — first counted live probe before late-variant hardening

Commands:

```bash
python3 verifiers/pr-review-fanout-remediation/run_live_probe.py --n 3 --timeout 1800
python3 scripts/probe_report.py benchmark_blueprints/families/pr-review-fanout-remediation/report/probe_runs.jsonl --probe-run-id 20260423T232806Z > benchmark_blueprints/families/pr-review-fanout-remediation/report/attempt_02_probe_report.txt
```

Counted probe run id:

- `20260423T232806Z`

Per-variant scores:

| variant | scores | mean |
|---|---|---:|
| v1-clean-baseline | `[25, 25, 25]` | 25.00 |
| v2-noisy-distractor | `[25, 25, 25]` | 25.00 |
| v3-dirty-state | `[25, 25, 0]` | 16.67 |
| v4-multi-corpus-objective | `[0, 25, 35]` | 20.00 |
| v5-recovery-in-thread | `[25, 25, 25]` | 25.00 |

Layer A gate values from `report/attempt_02_probe_report.txt`:

- `family_mean = 22.33`
- `max_variant_mean = 25.00`
- `min_variant_mean = 16.67`
- gate checks:
  - family_mean window: `PASS`
  - max variant cap: `PASS`
  - at least one variant `<= 10`: `FAIL`
  - monotonic V1>=V2>=V3>=V4>=V5 `+/-3`: `FAIL`

Judgment from this counted run:

- the family was now honestly exercising the late variants, but the V4/V5
  ceilings were still too lenient for two real misses:
  - V4 agents often repaired the code but failed to acknowledge the downstream
    consumer contract around `approval_state`
  - V5 agents often repaired the code but ignored `INC-214` / no-alias
    recovery

Family-local hardening applied after this counted run:

```bash
python3 verifiers/pr-review-fanout-remediation/regen_family.py
python3 verifiers/pr-review-fanout-remediation/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/pr-review-fanout-remediation/verification_matrix.md
python3 verifiers/pr-review-fanout-remediation/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/pr-review-fanout-remediation/verification_matrix_v5-recovery-in-thread.md
```

- scorer change: `objective_contract_missed` cap `45 -> 15`
- scorer change: `incident_blind_recovery` cap `30 -> 10`
- because the scorer changed, only the post-change rerun below counts for the
  current family state

## attempt_04 — counted post-change live probe follow-up

Commands:

```bash
python3 verifiers/pr-review-fanout-remediation/run_live_probe.py --n 3 --timeout 1800
python3 scripts/probe_report.py benchmark_blueprints/families/pr-review-fanout-remediation/report/probe_runs.jsonl --probe-run-id 20260424T000621Z > benchmark_blueprints/families/pr-review-fanout-remediation/report/attempt_03_probe_report.txt
```

Counted probe run id:

- `20260424T000621Z`

Per-variant scores:

| variant | scores | mean |
|---|---|---:|
| v1-clean-baseline | `[25, 25, 25]` | 25.00 |
| v2-noisy-distractor | `[100, 25, 25]` | 50.00 |
| v3-dirty-state | `[100, 0, 25]` | 41.67 |
| v4-multi-corpus-objective | `[0, 15, 15]` | 10.00 |
| v5-recovery-in-thread | `[0, 10, 10]` | 6.67 |

Layer A gate values from `report/attempt_03_probe_report.txt`:

- `family_mean = 26.67`
- `max_variant_mean = 50.00`
- `min_variant_mean = 6.67`
- gate checks:
  - family_mean window: `FAIL`
  - max variant cap: `FAIL`
  - at least one variant `<= 10`: `PASS`
  - monotonic V1>=V2>=V3>=V4>=V5 `+/-3`: `FAIL`

Verification matrix outputs after the scorer change:

- `verification_matrix.md` (`v1-clean-baseline`)
  - Oracle: `P=90`, `M=0.8947`, `integrity=0`
  - Empty: `P=0`, `M=0.0000`
  - Code-only shortcut: `P=20`
  - Wrong duplicate-thread triage: `P=20`
  - Generic replies without thread mapping: `P=35`
  - Delete-tests adversarial: `P=0`, `integrity=1`
- `verification_matrix_v5-recovery-in-thread.md`
  - Oracle: `P=95`, `M=0.8947`, `integrity=0`
  - Empty: `P=0`, `M=0.0000`, with `incident_blind_recovery`
  - Code-only shortcut: `P=10`
  - Wrong duplicate-thread triage: `P=20`
  - Generic replies without thread mapping: `P=35`
  - Delete-tests adversarial: `P=0`, `integrity=1`

Live-probe spot checks from this counted run:

- `report/live_probe_logs/20260424T000621Z-v2-noisy-distractor-run1.log`
  shows a clean full solve with exact thread classification, normalized fallback
  payload, updated docs, and passing `unittest` verification; that run scored
  `100`
- `report/live_probe_logs/20260424T000621Z-v5-recovery-in-thread-run1.log`
  shows the solver making the code/test/docs repair but still omitting the
  required `INC-214` recovery acknowledgement, which now caps the run at `0`
  because an integrity failure also fired

Current judgment:

- live probe is now fully completed and recorded in family-local artifacts
- the post-change counted probe is the governing result for the current family
  state
- Layer A remains `HARDEN NEEDED`
- the remaining failure mode is no longer a late-variant blind spot; it is an
  early-variant variance/easiness issue, with `v2` and `v3` sometimes clearing
  the full rubric
- I did **not** apply another post-probe scorer change after `20260424T000621Z`,
  because lowering those full-solve rows further would require redesigning
  early-variant task evidence or penalizing correct behavior, not another
  honest ceiling tweak
