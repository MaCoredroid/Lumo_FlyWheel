# Review Thread UI Hardening

## Canonical Prompt
You are on a front-end feature branch with exported PR review threads. The bundle contains multiple icon-like controls, mixed resolved and unresolved review comments, and several mobile screenshot notes. Only one thread set is still actionable for the current reopen. Investigate the live review state from the artifacts, fix the real UI regression in the repo, update the exact mobile coverage configuration that was reopened, and draft concise replies for the actionable thread ids only. Do not infer the target viewport, route, or control from filenames alone. Do not churn resolved-thread artifacts or broad-apply accessibility labels to unrelated controls.

## Workspace Layout
- `repo/`
  - `src/components/`: front-end component sources with at least one visually similar but unaffected icon button.
  - `src/styles/`: CSS containing the mobile wrapping regression.
  - `config/snapshot-viewports.json`: viewports and route-to-viewport snapshot scenarios.
  - `tests/test_review_thread_ui.py`: visible branch-level check. It is intentionally incomplete.
- `artifacts/review/`
  - `threads.json`: exported review state, thread ids, state metadata, route/viewport metadata, and screenshot refs.
  - `export.md`: reviewer prose copied from the PR export.
- `artifacts/screenshots/`
  - Reviewer screenshot notes for the active mobile break plus visually similar stale or resolved references. Action clusters may contain more than one icon-like control.
- `artifacts/logs/`
  - CI snapshot output and one irrelevant lint warning.
- `review_reply/`
  - Solver-authored `replies.md`.
- `evidence/`
  - Solver-authored `post_fix.md`.

## Structured-Output CLI
Every variant ships the same CLI at `./bin/review-thread-task`.

Subcommands:
- `schema`
- `validate submission_input.json`
- `submit submission_input.json`

The solver must:
1. Edit the repo and output files.
2. Author `submission_input.json` at the workspace root.
3. Run `./bin/review-thread-task submit submission_input.json`.

`submit` writes:
- `brief/fix_summary.json`
- `brief/fix_summary.md`

The scorer reads `brief/fix_summary.json` plus the actual workspace edits.

## Submission Schema
`submission_input.json` must contain:
- `schema_version = "cnb55.review_fix.v1"`
- `variant_id`
- `thread_ids_addressed`
- `route`
- `viewport`
- `target_control`
- `edited_files`
- `reply_file`
- `evidence_file`
- `tests_run`

The CLI validates:
- file existence for every listed path
- variant match with `.scenario_variant`
- non-empty thread ids and edited-file list
- reply/evidence files located under `review_reply/` and `evidence/`
- `edited_files` restricted to repo sources and config files

## Required Surfaces
- Review-thread triage across unresolved and resolved exports.
- Screenshot-note interpretation.
- Runtime repo patching for mobile wrapping and icon-button accessibility without broad label churn across adjacent controls.
- Exact viewport/config update.
- Concise review-reply drafting tied to actionable thread ids.
- Fresh evidence tied to the reopened viewport.

## Visible Checks
- `python3 repo/tests/test_review_thread_ui.py`
- CLI validation passes.

## Hidden Checks
- Exact unresolved thread ids are handled; resolved ids are not churned.
- The target icon-only control gets an accessible name.
- Protected controls do not receive copy-pasted label churn.
- The solver must distinguish the actionable control from other icon-like controls in the same action cluster.
- Mobile layout wraps instead of clipping, truncating, or hiding overflow.
- The exact reopened viewport and route are added to snapshot config.
- Evidence and replies match the live reopen, not a stale screenshot or prior thread.
- Variant-specific traps fire deterministic score ceilings when the solver follows stale state, objective drift, or rollback-blind recovery.

## Variant Progression
- `v1-clean-baseline`: one clear unresolved mobile reopen plus one clearly resolved reference.
- `v2-noisy-distractor`: stale screenshot note and resolved thread metadata are mixed into the export.
- `v3-dirty-state`: an abandoned previous-session fix path points at the wrong control or file.
- `v4-multi-corpus-objective`: release-scope notes change which reopen and viewport matter.
- `v5-recovery-in-thread`: rollback history means the solver must acknowledge the incident before claiming the fix is safe.

## Writable Surface
Agents may write only:
- `repo/src/components/*.tsx`
- `repo/src/styles/*.css`
- `repo/config/snapshot-viewports.json`
- `review_reply/replies.md`
- `evidence/post_fix.md`
- `submission_input.json`
- `brief/*`

Hidden integrity checks fail on writes to `artifacts/`, `repo/tests/`, `release_context/`, `incident_context/`, or any other immutable slice.

## Saturation And Renewal
Saturation trigger:
- mean `P_benchmark > 80` for two consecutive probe rounds

Renewal queue:
- Add a new variant where the reopened mobile route changes mid-thread after the first fix lands.
- Retire the current V1 if it stops discriminating and promote V2 as the new baseline.
