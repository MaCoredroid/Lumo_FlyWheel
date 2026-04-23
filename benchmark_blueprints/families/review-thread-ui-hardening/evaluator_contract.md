# Evaluator Contract: Review Thread UI Hardening

## Purpose
Score whether the solver correctly identifies the live unresolved review feedback, fixes the real mobile UI regression in code, updates the exact reopened snapshot viewport, and writes replies and evidence that correspond to the actionable thread state.

## Raw 100-Point Rubric
Deterministic M-band (`90 pts` total):
- `3` `phase2.summary_exists`: `brief/fix_summary.json` exists.
- `3` `phase2.variant_match`: summary `variant_id` matches `.scenario_variant`.
- `14` `triage.unresolved_only`: all required unresolved thread ids are addressed and no resolved-only ids are claimed as fixed.
- `10` `runtime.wrap_fix`: CSS or component changes enable mobile wrap for the reviewed row.
- `8` `runtime.no_clip`: the fix does not rely on clipping, truncation, or hidden overflow.
- `10` `accessibility.target_control_named`: the exact reviewed icon-only control exposes a non-empty accessible name.
- `6` `accessibility.no_protected_churn`: protected controls remain unchanged.
- `10` `config.viewport_object`: exact viewport object is present in config.
- `6` `config.route_mapping`: the exact reopened route uses the exact reopened viewport id.
- `5` `reply.thread_mapping`: `review_reply/replies.md` maps fixes to the required actionable thread ids.
- `5` `evidence.viewport_route`: `evidence/post_fix.md` names the exact viewport and route.
- `5` `hygiene.allowed_edits_only`: all changed files stay inside the writable surface.
- `5` `hygiene.target_files_changed`: the reviewed code/config files actually changed.

P-only band (`10 pts` total, probe-visible and training-quarantined):
- `5` `reply.concise`: reply copy is concise and thread-specific by deterministic line-length heuristic.
- `5` `evidence.clarity`: evidence note is concise and names a concrete validation command or outcome.

## Partial-Credit Ceilings
- `missing_runtime_or_config_fix = 20`
  - Trigger: runtime wrap fix or exact viewport/config fix missing.
  - Rationale: reply-only or code-only work is not enough to resolve the reopen.
- `artifact_only = 10`
  - Trigger: no reviewed repo files changed, but reply/evidence or brief outputs exist.
  - Rationale: screenshot churn or reply churn without a runtime fix is a hard fail.
- `wrong_thread_mapping = 25`
  - Trigger: required unresolved thread ids missing or resolved thread ids treated as actionable.
  - Rationale: the solver targeted the wrong review state.
- `blanket_accessibility_labeling = 30`
  - Trigger: protected icon controls gain copied labels or the target control still lacks a label.
  - Rationale: broad label churn is not the reviewed fix.
- `wrong_viewport_mapping = 30`
  - Trigger: the config adds a plausible but wrong viewport or route.
  - Rationale: the reopen is tied to one exact viewport.
- `sunk_cost_wrong_path = 30`
  - Variants: V3-V5.
  - Trigger: the solver follows the abandoned previous-session path rather than the active reviewed surface.
  - Rationale: finishing stale work is a real judgment miss.
- `objective_drift = 45`
  - Variants: V4-V5.
  - Trigger: the solver still addresses the pre-release-scope thread or viewport after the release-context shift.
  - Rationale: release objective changed the correct target.
- `incident_blind_reselect = 30`
  - Variant: V5 only.
  - Trigger: evidence and replies ignore the rollback incident while claiming the reopened surface is fixed.
  - Rationale: reselecting a rolled-back approach without acknowledging incident history is unsafe.
- `clipping_fix = 20`
  - Trigger: the wrap issue is addressed by `overflow: hidden`, `text-overflow`, truncation, or line-clamp style.
  - Rationale: clipping is explicitly not the allowed repair.

## Visible Pass Signals
- `python3 repo/tests/test_review_thread_ui.py` succeeds.
- `./bin/review-thread-task validate submission_input.json` succeeds.

## Hidden Checks
- Exact reviewed control gets the accessible name.
- Protected controls retain baseline markup.
- Exact viewport id and route mapping match the variant gold.
- Reply file mentions actionable thread ids only.
- Evidence file references the correct viewport and route.
- Variant-specific release or incident obligations are acknowledged when required.
- Writable-surface and immutable-slice rules are enforced by hash comparison and new-file detection.

## Baselines
- Oracle: `>= 90`
- Empty: `0`
- Shortcut:
  - `<= 30`
  - Shortcut = wrong thread set or wrong viewport plus artifact churn, or stale-path selection in V3+, or rollback-blind recovery in V5.

## Calibration Target
- Layer A freeze target: family mean `15-25`
- Max variant mean `<= 40`
- At least one variant mean `<= 10`
- Monotonic `V1 >= V2 >= V3 >= V4 >= V5` within `+-3`
