# Evaluator Contract

- `family_id`: `runbook-code-reconciliation`
- `task_id`: `t2_runbook_release_preview_reconciliation`
- `total_points`: `100`
- `target_attack_band_for_naive_gpt_5_4_high`: `15-25`

## Scoring Breakdown

- `35` points: Runbook patch correctly reflects the live command entrypoint, current primary flags, and current environment variables.
- `20` points: `reconciliation_facts.json` accurately separates current versus deprecated names and records evidence commands.
- `20` points: Verification notes include at least one directly checked command, what it proved, and one bounded caveat.
- `15` points: Deploy note accurately explains operator impact without inventing behavior changes.
- `10` points: Overall discipline in privileging code and live CLI help over stale prose.

## Caps

- Cap at `20/100` if evidence comes from outside the provided benchmark bundle or if the solver substitutes workspace-wide absence checks for live bundle verification.
- Cap at `20/100` if the solver does not directly verify any command or help output.
- Cap at `25/100` if `reconciliation_facts.json` is internally consistent but not validated against the live path.
- Cap at `30/100` if the runbook still recommends a legacy alias as the primary flow.
- Cap at `15/100` if the runbook looks like harmonized docs rather than a code-grounded reconciliation.
- Cap at `10/100` if the deploy note claims behavior changes beyond docs and operator instructions.

## Automatic Failure Conditions

- `0` if the runbook still uses a deprecated primary command path.
- `0-5` if required artifacts are missing.
- `0-5` if the solver invents flags, env vars, or helper behaviors not present in the bundle.

## Evidence Rules

- Only files and commands inside the provided benchmark bundle count as evidence unless the task explicitly names an external surface.
- The current primary instruction must be supported by code or live help, not just by a README.
- Deprecated compatibility paths may appear only as secondary notes.
- `evidence_commands` should be checkable commands, not generic statements.

## Judging Procedure

1. Verify the patch and facts artifact exist.
2. Score primary-path correctness first.
3. Apply caps before deploy-note polish.
4. Check whether the submission would remain correct if README fragments were removed.

## Meaningfulness Check

- A valid task instance must contain at least one contradictory prose artifact and one code-grounded source of truth.
- If the bundle allows a solver to succeed by copying the closest README verbatim, the task instance is invalid.
