# Benchmark Run

- `family_id`: `request-path-evidence-brief`
- `task_id`: `t2_request_path_owner_source_brief`
- `calibration_model`: `gpt-5.4`
- `reasoning_effort`: `high`

## attempt_00 — skeleton-only attack evidence (`2026-04-18`)

### Context

- Family docs existed, but no family-local workspace bundle or scorer existed yet.
- A child attempt reached into `scenario_families/owner-field-cross-layer/...` because the authored task named repo surfaces that were not actually shipped inside the family bundle.

### Result

- External-evidence cap fired.
- Final score: `20/100`.

### Lesson

- This family needed a real family-local repo bundle, not just a spec.
- It also needed a deterministic artifact contract so a polished cross-bundle substitution would not score well.

## attempt_01 — family-local implementation and deterministic smoke (`2026-04-22`)

### Design change

- Built a real five-variant workspace bundle under `workspace_bundle/v{1..5}`.
- Added deterministic scorer `verifiers/request-path-evidence-brief/score_trace.py`.
- Added family metadata in `family.yaml`, per-variant `gold_path.json`, milestone scripts, `workspace_manifest.json`, and generated `manifest.lock.json`.
- Added verification matrices for `v1-clean-baseline` and `v5-recovery-in-thread`.
- Tightened markdown grounding so a correct-looking `path_map.json` without evidence-backed prose is capped.

### Deterministic verification

Manifest baselines after regen:

| Variant | Oracle | Empty | Grounding stripped | Shortcut |
| --- | ---: | ---: | ---: | ---: |
| V1 | 99 | 0 | 35 | 25 |
| V2 | 99 | 0 | 35 | 25 |
| V3 | 99 | 0 | 35 | 25 |
| V4 | 99 | 0 | 35 | 25 |
| V5 | 100 | 0 | 30 | 25 |

Verification matrix spot checks:

- `verification_matrix.md` (V1): Oracle `99`, Empty `0`, store-decoy `25`, delete-tests adversarial `0` with integrity failure.
- `verification_matrix_v5.md` (V5): Oracle `100`, grounding-stripped `30`, delete-tests adversarial `0` with integrity failure.

Visible-test smoke:

- Oracle-populated copy of `v1-clean-baseline` passed `pytest -q tests/test_sync.py tests/test_docs.py tests/test_trace_outputs.py` with `5 passed`.

### Gate status

- Layer B scaffolding: implemented.
- Oracle / empty / shortcut sanity checks: pass.
- Live probe against `codex exec`: pending.
- Layer A probe gate (`family_mean`, monotonicity, min variant <= 10, etc.): pending until live probe completes.

### Current judgment

- Family is now coherent and family-local.
- The main remaining unknown is true live-model calibration, not bundle completeness or scorer integrity.
