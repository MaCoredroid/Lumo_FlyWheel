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

## attempt_02 — live-probe bring-up and scorer-alignment fixes (`2026-04-22`)

### Commands run

- `python3 benchmark_blueprints/families/request-path-evidence-brief/scripts/probe_family.py --attempt smoke_probe_04 --n 1 --variants v1-clean-baseline --timeout-seconds 600`
- `python3 benchmark_blueprints/families/request-path-evidence-brief/scripts/probe_family.py --attempt attempt_02_live_probe --n 3 --timeout-seconds 900`

### Result

- `smoke_probe_04` proved the family could execute end to end under real `codex exec`; the saved V1 run re-scored to `98/100` after the later scorer fixes.
- The first full-family live attempt under `report/attempt_02_live_probe/` completed, but it was run while the scorer and family prompt were still being corrected.

### Issues found while bringing the live probe online

- The scorer under-credited legitimate live-path encodings because the family's oracle used a chained path representation while strong model outputs often used an orchestrator-style `sync_item` representation.
- `test_observations` matching incorrectly missed family-local oracle entries that used separate `file` and `symbol` fields instead of a single combined `file::symbol` string.
- Negative derivation notes such as `not_derived_in: sync_app/store.py::make_record` still tripped `store_claimed_as_decision_layer` / `pre_owner_routing_claim`, which was a scorer bug rather than a judgment failure.

### Counted status

- This bring-up attempt is **not** the counted Layer A probe because family-local scorer behavior changed after the run.
- The post-fix whole-family probe below is the one that counts.

## attempt_03 — counted whole-family live `codex exec` probe (`2026-04-22`)

### Commands run

- `python3 benchmark_blueprints/families/request-path-evidence-brief/scripts/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/request-path-evidence-brief/verification_matrix.md`
- `python3 benchmark_blueprints/families/request-path-evidence-brief/scripts/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/request-path-evidence-brief/verification_matrix_v5.md`
- `python3 benchmark_blueprints/families/request-path-evidence-brief/scripts/probe_family.py --attempt attempt_04_live_probe_counted --n 3 --timeout-seconds 900`
- `python3 benchmark_blueprints/families/request-path-evidence-brief/scripts/probe_report.py --attempt-dir benchmark_blueprints/families/request-path-evidence-brief/report/attempt_04_live_probe_counted`

### Post-change deterministic verification

Verification matrix outputs after the scorer fixes:

| Matrix | Oracle | Empty | RAWR grounding stripped | Pick store decoy | Wrong live order | Delete-tests adversarial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `verification_matrix.md` (`v1-clean-baseline`) | 99 | 0 | 35 | 25 | 25 | 0 |
| `verification_matrix_v5.md` (`v5-recovery-in-thread`) | 100 | 0 | 30 | 25 | 25 | 0 |

Integrity result:

- Both matrices still force `immutable_slice_mutated` to `0/100` on the delete-tests adversarial row.

### Counted live-probe result

Artifacts:

- Attempt dir: `benchmark_blueprints/families/request-path-evidence-brief/report/attempt_04_live_probe_counted`
- Report: `benchmark_blueprints/families/request-path-evidence-brief/report/attempt_04_live_probe_counted/attempt_04_live_probe_counted_probe_report.txt`
- Runs JSONL: `benchmark_blueprints/families/request-path-evidence-brief/report/attempt_04_live_probe_counted/probe_runs.jsonl`
- Summary JSON: `benchmark_blueprints/families/request-path-evidence-brief/report/attempt_04_live_probe_counted/summary.json`

Per-variant numeric results:

| Variant | Scores | Mean P | Stdev P | Mean M | Stdev M | Dominant ceilings |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `v1-clean-baseline` | `[93, 25, 25]` | 47.67 | 32.06 | 0.4767 | 0.3206 | `missing_symbol_adjacency` |
| `v2-noisy-distractor` | `[98, 98, 98]` | 98.00 | 0.00 | 0.9800 | 0.0000 | `—` |
| `v3-dirty-state` | `[25, 93, 93]` | 70.33 | 32.06 | 0.7033 | 0.3206 | `missing_symbol_adjacency`, `pre_owner_routing_claim`, `store_claimed_as_decision_layer` |
| `v4-multi-corpus-objective` | `[25, 25, 93]` | 47.67 | 32.06 | 0.4767 | 0.3206 | `missing_symbol_adjacency`, `store_claimed_as_decision_layer` |
| `v5-recovery-in-thread` | `[25, 30, 25]` | 26.67 | 2.36 | 0.2667 | 0.0236 | `incident_blind_reselect`, `missing_symbol_adjacency` |

### Layer A gate values

- `family_mean`: `58.07` -> `FAIL` (target `[15, 25]`)
- `max_variant_mean`: `98.00` -> `FAIL` (target `<= 40`)
- `min_variant_mean`: `26.67` -> `FAIL` (target `<= 10`)
- `monotonic_with_tolerance_3`: `FAIL`
- `family_mean_M_training`: `0.5807`
- `current_observed_stdev_M_training`: `0.3474`

### Current judgment

- Counted live probe is complete.
- Layer A fails decisively for honest reasons: the family is too easy live, especially `v2-noisy-distractor`, and often `v3-dirty-state` / `v4-multi-corpus-objective`, where `codex exec` frequently reaches near-oracle outputs instead of staying inside the target band.
- The remaining variance is mostly between near-oracle solves and adjacency/variant-ceiling collapses, not around the intended `15-25` frontier window.
