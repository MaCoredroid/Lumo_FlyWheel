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

- Layer A: still pending live calibration. The family now has a real five-rung bundle and deterministic local baselines, but no `codex exec` probe data has been collected yet for the new package.
- Layer B: implemented locally. The scorer emits `P_benchmark`, `M_training`, milestone booleans, milestone vector, integrity flags, state-delta-ready metadata, capability tags, and verification matrices.

### Next honest hardening step

Run the family-local live probe via `verifiers/codex-skill-runtime-v2-split/run_live_probe.py`, then decide whether V1 remains too easy or whether V4/V5 need tighter reusable-bundle and incident-recovery pressure.

## Live probe status

- `live_probe`: `pending`
- Reason: this turn rebuilt and locally verified the family package, but did not run `codex exec` calibration across the new V1-V5 bundle yet.
