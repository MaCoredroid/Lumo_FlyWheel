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

## `attempt_04` — first live-probe pass exposed an impossible hidden config surface

### Commands run

```bash
PROBE_RUN_ID='attempt_04c_smoke_v1_20260422T2210Z' \
N=1 \
VARIANTS='v1-clean-baseline' \
CODEX_TIMEOUT=900 \
benchmark_blueprints/families/responses-sdk-adapter-cutover/tools/run_live_probe.sh
```

### What happened

- The live `codex exec` worker could patch source files and docs, but every direct write to `.codex/config.toml` failed with `PermissionError [Errno 1] Operation not permitted`.
- This was not a model-choice issue. The probe log captured the failure even for direct Python writes inside the copied workspace.
- Resulting smoke artifact: `benchmark_blueprints/families/responses-sdk-adapter-cutover/probe_runs/attempt_04c_smoke_v1_20260422T2210Z/`

### Decision

- Treat the hidden `.codex/config.toml` requirement as a family-local calibration bug.
- Move the mutable runtime config requirement to a normal workspace file, `config/runtime.toml`, and update scorer/profile data accordingly.

## `attempt_05` — writable-runtime-config recalibration plus full family live probe

### Family-local recalibration before rerun

- Replaced the required mutable config surface with `config/runtime.toml` across the task spec, evaluator contract, AGENTS files, scorer, profiles, and workspace bundles.
- Rebuilt family assets with:

```bash
python3 verifier_data/responses-sdk-adapter-cutover/tools/build_family_assets.py
```

- Confirmed the new config surface is writable in-repo and runnable under `codex exec`.

### Commands run

Smoke confirmation:

```bash
PROBE_RUN_ID='attempt_05a_smoke_v1_runtime_config_20260422T2205Z' \
N=1 \
VARIANTS='v1-clean-baseline' \
CODEX_TIMEOUT=900 \
benchmark_blueprints/families/responses-sdk-adapter-cutover/tools/run_live_probe.sh
```

Whole-family live probe:

```bash
PROBE_RUN_ID='attempt_05b_full_live_runtime_config_20260422T2207Z' \
N=3 \
CODEX_TIMEOUT=900 \
benchmark_blueprints/families/responses-sdk-adapter-cutover/tools/run_live_probe.sh
```

Probe report generation:

```bash
python3 scripts/probe_report.py \
  benchmark_blueprints/families/responses-sdk-adapter-cutover/probe_runs/attempt_05b_full_live_runtime_config_20260422T2207Z/probe_runs.jsonl \
  --probe-run-id attempt_05b_full_live_runtime_config_20260422T2207Z \
  > benchmark_blueprints/families/responses-sdk-adapter-cutover/probe_runs/attempt_05b_full_live_runtime_config_20260422T2207Z/probe_report.txt

python3 scripts/probe_report.py \
  benchmark_blueprints/families/responses-sdk-adapter-cutover/probe_runs/attempt_05b_full_live_runtime_config_20260422T2207Z/probe_runs.jsonl \
  --probe-run-id attempt_05b_full_live_runtime_config_20260422T2207Z \
  --emit-json \
  > benchmark_blueprints/families/responses-sdk-adapter-cutover/probe_runs/attempt_05b_full_live_runtime_config_20260422T2207Z/probe_report.json
```

Verification matrices after the recalibration:

```bash
python3 verifier_data/responses-sdk-adapter-cutover/tools/run_verification_matrix.py \
  --variant v1-clean-baseline \
  --out benchmark_blueprints/families/responses-sdk-adapter-cutover/verification_matrix.md

python3 verifier_data/responses-sdk-adapter-cutover/tools/run_verification_matrix.py \
  --variant v5-recovery-in-thread \
  --out benchmark_blueprints/families/responses-sdk-adapter-cutover/verification_matrix_v5.md
```

### Live probe results

Artifacts:

- JSONL: `benchmark_blueprints/families/responses-sdk-adapter-cutover/probe_runs/attempt_05b_full_live_runtime_config_20260422T2207Z/probe_runs.jsonl`
- Text report: `benchmark_blueprints/families/responses-sdk-adapter-cutover/probe_runs/attempt_05b_full_live_runtime_config_20260422T2207Z/probe_report.txt`
- JSON report: `benchmark_blueprints/families/responses-sdk-adapter-cutover/probe_runs/attempt_05b_full_live_runtime_config_20260422T2207Z/probe_report.json`

Per-variant table:

| Variant | Scores | Mean | Stdev | Dominant ceilings |
| --- | --- | ---: | ---: | --- |
| `v1-clean-baseline` | `[85, 100, 100]` | `95.00` | `8.66` | none |
| `v2-noisy-distractor` | `[25, 25, 25]` | `25.00` | `0.00` | `flattened_multi_event_turn` x3 |
| `v3-dirty-state` | `[25, 25, 25]` | `25.00` | `0.00` | `flattened_multi_event_turn` x3, `reordered_chunk_instability` x3 |
| `v4-multi-corpus-objective` | `[25, 25, 25]` | `25.00` | `0.00` | `flattened_multi_event_turn` x3, `reordered_chunk_instability` x3 |
| `v5-recovery-in-thread` | `[25, 30, 25]` | `26.67` | `2.89` | `future_event_corruption` x3, `flattened_multi_event_turn` x2, `reordered_chunk_instability` x2 |

### Layer A gate values

- `family_mean = 39.33` — `FAIL` (target window `[15, 25]`)
- `max_variant_mean = 95.00` — `FAIL` (cap `<= 40`)
- `min_variant_mean = 25.00` — `FAIL` (need at least one variant `<= 10`)
- monotonic `V1 >= V2 >= V3 >= V4 >= V5` within tolerance — `PASS`

Overall verdict from the generated report: `HARDEN NEEDED`

### Post-recalibration verification-matrix snapshots

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

### Spot-check diagnosis

- The recalibration succeeded at the environment boundary: live workers now edit `config/runtime.toml` normally, and the scorer no longer forces a fake `20` ceiling from an unwritable hidden file.
- The family now discriminates on the intended hidden behavior packs for `v2` through `v5`:
  - `v2` repeatedly fails `hidden.multi_block_message`, causing `flattened_multi_event_turn`.
  - `v3` and `v4` repeatedly fail reordered replay stability, adding `reordered_chunk_instability`.
  - `v5` repeatedly fails future-event handling, adding `future_event_corruption`.
- The remaining Layer A problem is concentrated in `v1`, which is too easy once the runtime-config path is writable. The next honest hardening move is to add a real baseline hidden discriminator for `v1`, not to reintroduce a mechanically blocked file path.

### Layer status after this pass

- Layer A: still failing. The family is now legitimately runnable and gives real live ceilings on `v2` through `v5`, but `v1` remains far too easy and no variant is yet below `10`.
- Layer B: still intact after the recalibration. The scorer, matrices, milestone emission, and verifier artifacts remain deterministic and family-local.
