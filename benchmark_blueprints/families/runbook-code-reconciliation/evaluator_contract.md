# Evaluator Contract

- `family_id`: `runbook-code-reconciliation`
- `task_id`: `t2_runbook_release_preview_reconciliation`
- `total_points`: `100`
- `target_attack_band_for_naive_gpt_5_4_high`: `15-25`

## Scoring Breakdown

- `35` points: `docs/runbooks/release_preview.md` uses the current primary entrypoint, current flag, and current env var, and does not treat the compatibility alias as the default.
- `20` points: `artifacts/reconciliation_facts.json` exactly matches the code-backed current/deprecated split and records evidence commands.
- `20` points: `artifacts/verification_notes.md` contains direct command checks, code-backed inferences, and a bounded caveat.
- `15` points: `artifacts/deploy_note.md` accurately explains the operator-facing runbook change without inventing a code behavior change.
- `10` points: overall evidence discipline favors CLI help and code over README fragments.

## Caps

- Cap at `20/100` if no direct help or pytest command is recorded.
- Cap at `25/100` if the facts artifact is plausible but not validated against direct checks.
- Cap at `30/100` if the runbook still recommends the legacy alias or deprecated config surface as the primary path.
- Cap at `15/100` if the update reads like a docs merge rather than a code-grounded reconciliation.
- Cap at `10/100` if the deploy note claims the alias or deprecated env var was removed.

## Automatic Failure Conditions

- `0` if required artifacts are missing.
- `0` if the runbook still uses the deprecated primary command path.
- `0-5` if the submission invents a flag, env var, alias, or helper behavior not present in the bundle.

## Evidence Rules

- Only files and commands inside the provided benchmark bundle count as evidence.
- The current primary instruction must be supported by code or live help, not by a README fragment alone.
- Deprecated compatibility paths may appear only as secondary caveats.
- `evidence_commands` must be executable commands, not prose summaries.

## Judging Procedure

1. Verify the runbook patch and all three artifacts exist.
2. Score primary-path correctness first.
3. Apply caps before polish.
4. Check whether the answer would remain correct if the README fragments were removed.

## Meaningfulness Check

- A valid task instance must contain at least one contradictory prose artifact and one code-grounded source of truth.
- If a solver can pass by copying the closest README fragment verbatim, the task instance is invalid.
