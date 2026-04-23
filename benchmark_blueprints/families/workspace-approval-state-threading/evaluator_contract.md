# Evaluator Contract

## Scope
- Family: `workspace-approval-state-threading`
- Task: `cnb55-core-workspace-approval-state-threading-admin-ui`
- Runtime class: repo-edit monorepo with backend, CLI, frontend, config, docs, preview contract, and rollout-note JSON

## Evaluation Intent
Measure whether the solver can thread `approval_state` across real surfaces without relying on `risk_level` aliases, frontend-only fabrication, or stale `approval_mode` artifacts. The family should reward cross-surface consistency plus legacy-row fallback, and it should punish cosmetic-only fixes.

## Submission Artifact Requirements
- Updated backend normalization and serializer
- Updated CLI export
- Updated frontend admin table/page
- Updated tests
- Updated config and runbook
- Updated preview contract JSON
- Valid rollout note JSON

## Scoring Budget
Total `100` points.

Deterministic M-band (`90` total):
- `10`: backend normalization threads `approval_state`
- `10`: API serializer threads `approval_state`
- `10`: CLI export threads `approval_state`
- `12`: same mixed dataset is consistent across service, API, and CLI
- `8`: legacy-row fallback is implemented and surfaced
- `8`: frontend table adds `Approval state` column
- `6`: frontend badge rendering uses `approval_state` plus fallback, not `risk_level`
- `6`: visible tests updated to assert the new contract
- `6`: preview contract artifact matches the variant expectation
- `4`: config uses `approval_state`
- `4`: runbook updated to the new terminology
- `3`: runbook documents legacy-row fallback
- `3`: rollout note JSON is valid and variant-aware

Probe-only P-band (`10` total, quarantined from training):
- `10`: partial-progress heuristic for submissions that update multiple surfaces coherently but fail a hidden contract check

## Ceiling Rules
- `risk_level_alias_shortcut`: cap `20`
  - Fires when serializer or frontend derives `approval_state` from `risk_level`.
- `frontend_only_fabrication`: cap `20`
  - Fires when frontend and preview update but backend plus CLI remain stale.
- `missing_legacy_fallback`: cap `25`
  - Fires when `approval_state` is threaded for explicit rows but legacy rows still break.
- `v1_baseline_contract_unmet`: cap `10`
  - V1 only; fires when the clean baseline still misses legacy-row fallback and also fails visible verification or preview/rollout evidence. This keeps the minimal variant in the honest floor band instead of awarding a mid-teen score to an incomplete baseline fix.
- `stale_config_or_runbook`: cap `30`
  - Fires when `approval_mode` remains in config or docs.
- `missing_preview_or_rollout_evidence`: cap `35`
  - Fires when code looks substantially correct but preview contract or rollout note is absent or invalid.
- `release_objective_missed`: cap `35`
  - V4 only; fires when launch-readiness preview requirements are ignored.
- `rollback_ack_missing`: cap `35`
  - V5 only; fires when the rollout note does not acknowledge the earlier alias rollback.

## Visible Checks
- `python3 -m unittest discover -s tests -p 'test_*.py'` succeeds.
- Mixed-dataset test asserts consistent `approval_state` values plus fallback.
- Frontend test asserts `Approval state` column and fallback badge text.
- Preview test asserts screenshot filename and filtered-row badge contract.

## Hidden Checks
- Service output contains `approval_state` and `approval_state_source`.
- Serializer output includes `approval_state` for all rows and does not alias `risk_level`.
- CLI output equals the API-shaped rows for the mixed dataset.
- Preview contract matches the active variant gold contract.
- Rollout note JSON carries variant-required keywords.
- Immutable context trees are unchanged.

## Integrity Rules
- `write_outside_whitelist`
- `immutable_context_mutated`
- `bin_wrapper_mutated`
- `pytest_shim`
- `network_egress`

Integrity flag `H=1` force-fails M3/M4/M5 and clamps `M_training` to `0.0`.

## Baseline Targets
- Oracle: `>= 90`
- Empty / untouched bundle: `0`
- Shortcut / alias solution: `<= 30`
- Naive single-surface solver target: around `20`

## Current Calibration Intent
The family is trying to sit in the honest-difficulty band where a strong but non-persistent model proposes the right cross-layer fix but fails to deliver the full repo patch, preview artifact, and rollout-note evidence in one pass.

The current hardening direction is to make `v1-clean-baseline` a stricter floor check. If the solver still misses legacy fallback on the clean baseline and also cannot back the patch with the visible/evidence contract, that run should stay at or below the `<=10` floor target rather than clustering in the mid teens.
