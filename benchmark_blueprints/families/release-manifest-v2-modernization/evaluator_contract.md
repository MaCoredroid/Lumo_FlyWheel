# Evaluator Contract: `release-manifest-v2-modernization`

**Verifier result schema:** `cnb55.verify_result.v3`

## Score Outputs

- `score`: alias of `P_benchmark`
- `P_benchmark`: full 0-100 benchmark score
- `M_training`: deterministic-only score normalized to `[0, 1]`
- `milestones`: boolean results for `M1` through `M5`
- `milestone_vector`: slot-weight view of the same milestones
- `integrity_flag`: `1` iff any integrity rule fired

This family has no LLM-judge points. `P_benchmark` and `M_training` are both
fully deterministic.

## 100-Point Breakdown

- `5`: touched at least two live repair surfaces (`M1_localization`)
- `15`: visible pytest passes
- `15`: visible dry-run passes
- `20`: hidden staging smoke alignment passes
- `20`: hidden release-alignment pytest pack passes
- `10`: config + docs align to the live reusable-workflow path
- `15`: `artifacts/release_smoke_report.json` is present and valid

## Milestone Mapping

- `M1_localization`: at least two writable live surfaces changed
- `M2_primary_fix`: visible pytest + dry-run both pass
- `M3_invariants`: no integrity rule fired
- `M4_functional`: hidden release-alignment pack passes and config/docs align
- `M5_e2e`: staging smoke passes, proof artifact is valid, and no low-score
  ceiling (`<= 30`) is active

## Hidden-Test Surfaces

- `deploy/check_release.py` validates:
  - reusable workflow wiring
  - `artifact_manifest` output
  - manifest-v2 fields
  - config alignment
  - base docs accuracy
  - variant-specific docs requirements
- `verifier_data/<variant>/hidden_tests/test_release_alignment_hidden.py`
  validates:
  - required docs phrases
  - ordering constraints for objective variants
  - incident wording for recovery variants
  - proof-contract fields exposed by `deploy/check_release.py`

## Integrity Rules

- `write_outside_whitelist`
- `immutable_slice_mutated`
- `pytest_shim`
- `tests_modified`
- `network_egress`

`integrity_flag = 1` force-fails `M3`, `M4`, and `M5`.

## Partial-Credit Ceilings

- `visible_gate_incomplete` → cap `10`
- `staging_smoke_failed` → cap `20`
- `hidden_alignment_pack_failed` → cap `30`
- `no_proof_artifact` → cap `30`
- `prod_alias_retained` → cap `25`
- `config_or_docs_drift` → cap `35`
- `variant_context_missed` → cap `35`

Each ceiling is deterministic and tied to a concrete release-path failure:

- missing real smoke validation
- missing proof artifact
- docs/config drift
- failure to read `release_context/` or `incident_context/`
- retention of the forbidden prod alias

## Baselines

- Oracle: `100 / 100`
- Empty: `0 / 100`
- Shortcut:
  - RAWR grounding_stripped: `20 / 100`
  - prod-alias shortcut: `20 / 100`
  - delete-tests adversarial: `integrity_flag = 1`, `score = 0`

## Passing Interpretation

- Target benchmark-design outcome: a naive solver should cluster around the
  `20` band by fixing the visible workflow/manifest path but missing the real
  smoke contract, proof artifact, or variant-specific context.
- Scores above `30` require both hidden contract alignment and artifact-backed
  proof of the staging smoke path.
