# Benchmark Run

## `attempt_01` — legacy single-bundle draft

- Model target: `gpt-5.4`
- Reasoning: `high`
- Result: over target. The family only had one partial workspace, no Layer B package, and visible/config/docs credit dominated the score.

## `attempt_02` — hardened single-bundle rerun

- Model target: `gpt-5.4`
- Reasoning: `high`
- Result: `20/100` after adding hidden replay/interleaving/future-event weight and a visible-only cap.
- Finding: hardness was directionally correct, but the family still was not Layer A or Layer B complete because it had no true V1–V5 bundle, no family-local verifier tree, no manifests, and no verification matrix.

## `attempt_03` — family-local V1–V5 packaging and Layer B scaffolding

This pass converted the family into a proper CNB-55 family-local package:

- Added `workspace_bundle/v1-clean-baseline` through `v5-recovery-in-thread`.
- Added family-local scorer and verifier wrapper under `verifiers/responses-sdk-adapter-cutover/`.
- Added family-local verifier data under `verifier_data/responses-sdk-adapter-cutover/`:
  - gold metadata,
  - hidden tests,
  - oracle solved files,
  - milestone scripts,
  - verification-matrix runner.
- Added `family.yaml`, `manifest.lock.json`, and verification matrices.

### Baseline measurements from this pass

- Broken workspace visible slice (`v1` template): `0/100`
  - command: `pytest -q tests/test_adapter.py tests/test_replay.py tests/test_render.py`
  - result: `3 failed`
- Oracle solved workspace:
  - observed synthetic scorer result: `90/100` on `v1` and `v5`
- Visible-only trajectory:
  - observed synthetic scorer result: `20/100`
- Delete-tests adversarial:
  - observed synthetic scorer result: `0/100`, `integrity_flag = 1`

### Verification matrix snapshots

- `verification_matrix.md` (`v1-clean-baseline`)
  - Oracle: `90`
  - Empty: `0`
  - Visible-only: `20`
  - Legacy-shim shortcut: `20`
  - Delete-tests adversarial: `0`, `integrity_flag = 1`
- `verification_matrix_v5.md` (`v5-recovery-in-thread`)
  - Oracle: `90`
  - Empty: `0`
  - Visible-only: `20`
  - Future-event miss: `30`
  - Delete-tests adversarial: `0`, `integrity_flag = 1`

### Layer status after this pass

- Layer A: materially closer. The family now has a real five-rung bundle and deterministic baselines, but probe-family mean still needs live model runs to declare full freeze-gate acceptance.
- Layer B: implemented locally. The scorer now emits `P_benchmark`, `M_training`, milestone vector, integrity flags, capability tags, state-delta rules, and verification-matrix artifacts.
