# Benchmark Run

## `attempt_01` — single-bundle draft

- Model target: `gpt-5.4`
- Reasoning: `high`
- Result: over target. The family only had one partial workspace and the visible skill/config/automation slice was over-credited.

## `attempt_02` — hardened single-bundle evaluator

- Model target: `gpt-5.4`
- Reasoning: `high`
- Result: `20/100` against the old single-bundle scorer after pushing most weight behind hidden reuse, stale-reference retirement, and duplicate-automation semantics.
- Limitation: still not a real CNB-55 family. There was no V1-V5 bundle, no family-local verifier tree, no `family.yaml`, no manifest lock, and no Layer B verification matrix.

## `attempt_03` — family-local V1–V5 packaging and deterministic verifier scaffold

This pass rebuilt `codex-skill-runtime-v2-split` into a proper family-local package:

- added `workspace_bundle/v1-clean-baseline` through `v5-recovery-in-thread`
- added family-local scorer and runners under `verifiers/codex-skill-runtime-v2-split/`
- added family-local verifier data under `verifier_data/codex-skill-runtime-v2-split/`
  - gold metadata
  - oracle overlays
  - milestone scripts
  - manifest-backed hidden checks
- added `family.yaml`, `manifest.lock.json`, `verification_matrix.md`, and `verification_matrix_v5.md`
- rewrote `task_spec.md` and `evaluator_contract.md` to match the new five-variant family shape

### Commands run

```bash
python3 verifiers/codex-skill-runtime-v2-split/build_family_assets.py
python3 verifiers/codex-skill-runtime-v2-split/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/codex-skill-runtime-v2-split/verification_matrix.md
python3 verifiers/codex-skill-runtime-v2-split/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/codex-skill-runtime-v2-split/verification_matrix_v5.md
```

### Local deterministic baselines

From `verification_matrix.md` (`v1-clean-baseline`):

| Trajectory | Score | Notes |
| --- | ---: | --- |
| Oracle | `100` | full canonical bundle |
| Empty | `11` | broken bundle with legacy refs still live |
| Visible-only | `20` | visible checks green, hidden semantic retirement still missing |
| Duplicate-live shortcut | `30` | stale duplicate heartbeat still live |
| Delete-tests adversarial | `0` | `integrity_flag = 1` |

From `verification_matrix_v5.md` (`v5-recovery-in-thread`):

| Trajectory | Score | Notes |
| --- | ---: | --- |
| Oracle | `100` | full canonical bundle plus reuse + incident note |
| Visible-only | `20` | still shallow and capped |
| Duplicate-live shortcut | `30` | duplicate remains live |
| Variant miss | `30` | reuse/incident semantics intentionally removed |
| Delete-tests adversarial | `0` | `integrity_flag = 1` |

### Layer status after this pass

- Layer A at this point: pending live calibration. The family had a real five-rung bundle and deterministic local baselines, but no `codex exec` probe data had been collected yet for the new package.
- Layer B: implemented locally. The scorer emits `P_benchmark`, `M_training`, milestone booleans, milestone vector, integrity flags, state-delta-ready metadata, capability tags, and verification matrices.

### Next honest hardening step

Run the family-local live probe via `verifiers/codex-skill-runtime-v2-split/run_live_probe.py`, then decide whether V1 remains too easy or whether V4/V5 need tighter reusable-bundle and incident-recovery pressure.

## `attempt_09_full_live_20260423T224900Z` — counted whole-family live probe

This is the counted post-change live-probe attempt. Earlier live attempts in this follow-up were discarded because family-local probe fixes and calibration changes were still being made; only this attempt was run after the final `config/runtime.toml` writable-surface fix and V1-V5 shallow-cap ladder were in place.

### Commands run

```bash
python3 verifiers/codex-skill-runtime-v2-split/build_family_assets.py
python3 verifiers/codex-skill-runtime-v2-split/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/codex-skill-runtime-v2-split/verification_matrix.md
python3 verifiers/codex-skill-runtime-v2-split/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/codex-skill-runtime-v2-split/verification_matrix_v5.md
python3 verifiers/codex-skill-runtime-v2-split/run_live_probe.py --run-id attempt_09_full_live_20260423T224900Z --repeats 3 --timeout-sec 900 --model gpt-5.4 --reasoning high
python3 scripts/probe_report.py benchmark_blueprints/families/codex-skill-runtime-v2-split/probe_runs/attempt_09_full_live_20260423T224900Z/probe_runs.jsonl --probe-run-id attempt_09_full_live_20260423T224900Z
python3 scripts/probe_report.py benchmark_blueprints/families/codex-skill-runtime-v2-split/probe_runs/attempt_09_full_live_20260423T224900Z/probe_runs.jsonl --probe-run-id attempt_09_full_live_20260423T224900Z --emit-json
```

### Verification matrix outputs

From `verification_matrix.md` (`v1-clean-baseline`):

| Trajectory | P_benchmark | M_training | Integrity | Notes |
| --- | ---: | ---: | ---: | --- |
| Oracle | `100` | `1.0000` | `0` | full canonical bundle |
| Empty | `0` | `0.1111` | `0` | no primary skill; capped by `missing_primary_skill` |
| Visible-only | `25` | `0.5556` | `0` | visible green but hidden semantic retirement missing |
| Duplicate-live shortcut | `30` | `0.5556` | `0` | duplicate remains live |
| Variant miss | `100` | `1.0000` | `0` | V1 has no variant-specific extra miss |
| Delete-tests adversarial | `0` | `0.0000` | `1` | integrity adversarial detected |

From `verification_matrix_v5.md` (`v5-recovery-in-thread`):

| Trajectory | P_benchmark | M_training | Integrity | Notes |
| --- | ---: | ---: | ---: | --- |
| Oracle | `100` | `1.0000` | `0` | canonical bundle plus reuse and incident note |
| Empty | `0` | `0.1364` | `0` | no primary skill; capped by `missing_primary_skill` |
| Visible-only | `10` | `0.4091` | `0` | visible green but reuse and incident semantics missing |
| Duplicate-live shortcut | `30` | `0.4091` | `0` | duplicate remains live |
| Variant miss | `30` | `0.8636` | `0` | reuse/incident semantics intentionally removed |
| Delete-tests adversarial | `0` | `0.0000` | `1` | integrity adversarial detected |

### Live probe results

Probe artifact: `probe_runs/attempt_09_full_live_20260423T224900Z/probe_runs.jsonl`

Report artifacts:

- `report/attempt_09_probe_report.txt`
- `report/attempt_09_probe_report.json.txt`

| Variant | Scores | Mean | Stdev | Main ceilings |
| --- | ---: | ---: | ---: | --- |
| `v1-clean-baseline` | `[28, 28, 28]` | `28.00` | `0.00` | `visible_only_bundle`, `duplicate_automation_live` |
| `v2-noisy-distractor` | `[19, 19, 19]` | `19.00` | `0.00` | `visible_only_bundle`, `duplicate_automation_live` |
| `v3-dirty-state` | `[16, 16, 16]` | `16.00` | `0.00` | `visible_only_bundle`, `duplicate_automation_live` |
| `v4-multi-corpus-objective` | `[13, 13, 13]` | `13.00` | `0.00` | `visible_only_bundle`, `duplicate_automation_live`, `no_reuse_extension` |
| `v5-recovery-in-thread` | `[10, 10, 10]` | `10.00` | `0.00` | `visible_only_bundle`, `duplicate_automation_live`, `no_reuse_extension`, `incident_blind_reenable` |

### Layer A gate values

| Gate | Value | Requirement | Result |
| --- | ---: | --- | --- |
| Family mean | `17.20` | `15.0 <= mean <= 25.0` | PASS |
| Max variant mean | `28.00` | `<= 40.0` | PASS |
| Min variant mean | `10.00` | `<= 10.0` | PASS |
| Monotonicity | `28 >= 19 >= 16 >= 13 >= 10` | V1 >= V2 >= V3 >= V4 >= V5 within +/-3 | PASS |

### Diagnosis

The live solver consistently gets the visible pytest and smoke checks green and localizes the primary runtime path, but it does not complete the hidden semantic contract: duplicate retirement remains incomplete, the shared contract is absent, runbook alignment is shallow, V4 misses reusable escalation skill structure, and V5 misses the incident-safe recovery note. This is legitimate difficulty rather than prompt/path breakage.

### Layer status

- `live_probe`: `completed`
- Layer A: `accepted_live_probe`
- Layer B: `implemented_local`
