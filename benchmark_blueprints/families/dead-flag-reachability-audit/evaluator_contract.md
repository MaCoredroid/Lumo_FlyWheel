# Evaluator Contract

- `family_id`: `dead-flag-reachability-audit`
- `task_id`: `t2_dead_flag_preview_reachability_audit`
- `total_points`: `100`
- `target_attack_band_for_naive_gpt_5_4_high`: `15-25`

## Scoring Breakdown

- `35` points: Correct live or partial or dead classification for the named controls.
- `20` points: Correct distinction between parser presence, alias handling, and runtime behavioral reachability.
- `20` points: `reachability_matrix.json` includes valid `parser_symbol`, `runtime_branch_symbol`, `alias_of`, and one disproved false-positive path for each control.
- `15` points: Audit narrative uses at least four distinct evidence surfaces and resolves contradictory evidence cleanly.
- `10` points: Cleanup plan stays within proven evidence and does not overreach.

## Caps

- Cap at `20/100` if evidence comes from outside the provided benchmark bundle or from bundle metadata alone without concrete runtime-surface support.
- Cap at `20/100` if the submission never identifies runtime branch evidence or absence.
- Cap at `25/100` if the matrix is present but does not distinguish aliasing from standalone controls.
- Cap at `30/100` if classifications are plausible but justified mostly by grep presence or docs.
- Cap at `15/100` if cleanup recommendations outrun the proven reachability judgment.
- Cap at `10/100` if the audit collapses partial and dead into the same operational state.

## Automatic Failure Conditions

- `0` if all flags are treated as equally live with no branch reasoning.
- `0-5` if required artifacts are missing entirely.
- `0-5` if the solver invents runtime branches or parser symbols not supported by the bundle.

## Evidence Rules

- Only files and commands inside the provided benchmark bundle count as evidence unless the task explicitly names an external surface.
- Runtime claims need either a concrete branch symbol or a concrete absence claim tied to the live path.
- Parser evidence cannot substitute for runtime evidence.
- Tests can support, but not replace, branch reasoning.

## Judging Procedure

1. Verify all required deliverables exist.
2. Score the classification outcome first.
3. Validate matrix structure and branch evidence.
4. Apply caps before cleanup-plan points.

## Meaningfulness Check

- A family instance is valid only if the bundle contains enough contradictory evidence that parser-only reasoning can be separated from runtime reasoning.
- If all three controls are trivially classified from one file, the task instance is too shallow and should not be authored.
