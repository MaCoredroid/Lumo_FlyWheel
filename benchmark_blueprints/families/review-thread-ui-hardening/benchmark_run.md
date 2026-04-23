# Benchmark Run: Review Thread UI Hardening

## attempt_00 - baseline design

Hypothesis:
- The family should reward the solver only when it correctly maps review-state, viewport, and target control together.
- A package without a real repo or artifact bundle was not a legitimate benchmark, only a prompt stub.
- The honest hardening levers here are resolved-thread noise, stale viewport evidence, abandoned stale-path work, release-scope drift, and rollback-aware recovery.

Baseline verdict:
- The prior package-only draft was discarded as incomplete for Layer A and unusable for Layer B.

## attempt_01 - executable family package

Design change:
- Added a real family-local workspace bundle for V1-V5 with repo files, review exports, screenshot notes, config, visible tests, structured-output CLI, deterministic scorer, verifier data, manifest, and Layer B declarations.
- Kept `proposal-ranking-manager-judgment` as the structural reference only.
- Moved the deliverable to `submission_input.json` plus `./bin/review-thread-task submit`, while requiring actual code/config/reply/evidence edits in the workspace.

Verification run:
- Oracle/empty/shortcut baselines are checked by the family-local regen script.
- Verification matrix is generated for V1 and V5.
- Shared repo-level probe script was not used because it hard-codes the proposal-ranking prompt shape; this family now ships a family-local equivalent for future live probing.

Acceptance status after packaging:
- Layer A: not yet proven live, pending real probe runs.
- Layer B: green. Dual-band scorer, manifest lock, milestone scripts, and verification matrices for V1 and V5 are all generated family-locally.

Deterministic verification:
- Oracle / empty / shortcut baselines: all variants scored `100 / 0 / 10`.
- `verification_matrix.md` (V1) and `verification_matrix_v5.md` (stress variant) both show the expected floor rows:
  - Oracle passes at `100`
  - Empty is `0`
  - Wrong-thread and wrong-viewport synthetic rows hit deterministic ceilings
  - Delete-tests trips integrity and scores `0`

Live probe state:
- Pending. I did not run `codex exec` probe rounds in this turn because the family had to be finished from a doc-only stub into a runnable package first.

Next live-probe hypothesis:
- V1 should score high for a competent solver because the route, control, and viewport all align.
- V2 should drop on stale viewport anchoring.
- V3 should punish stale-path completion.
- V4 should drop sharply if the solver follows the pre-release-scope reopen instead of the release-scoped one.
- V5 should further punish rollback-blind replies even when the runtime fix is mechanically correct.
