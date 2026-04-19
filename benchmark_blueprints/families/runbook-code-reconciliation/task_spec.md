# Runbook Code Reconciliation

- `task_id`: `t2_runbook_release_preview_reconciliation`
- `family_id`: `runbook-code-reconciliation`
- `scenario_type`: `docs_from_codebase`

## Task Prompt

The on-call runbook for generating the daily release preview drifted after a CLI rename and one config-path change. Operators now report that the documented commands no longer work reliably. Reconcile the runbook against the real repo behavior.

Produce:

- A patch to `docs/runbooks/release_preview.md`.
- `artifacts/verification_notes.md`: the exact commands you checked, what they prove, and any remaining caveats.
- `artifacts/deploy_note.md`: a short note for operators explaining what changed in the runbook and why.
- `artifacts/reconciliation_facts.json`: exact fields for `primary_entrypoint`, `legacy_alias`, `current_flag`, `deprecated_flag`, `current_env_var`, `deprecated_env_var`, and `evidence_commands`.

Use the code as the source of truth. If README fragments contradict the live CLI or helper scripts, update the runbook to match the live path rather than repeating stale docs.

## Workspace Bundle

- Service repo with `Makefile`, `src/release_preview/cli.py`, helper scripts under `scripts/`, `.env.example`, and `docs/runbooks/release_preview.md`.
- Two README fragments with slightly different commands.
- One operational helper script still supports a legacy alias for backwards compatibility.
- Tests cover the CLI entrypoint but not the runbook text.

## Seeded Ambiguity

Authoring note: these ambiguity seeds are for benchmark construction and hidden checks. The runtime prompt shown to the solver should only describe drift, not the exact answer pattern.

- Not all stale prose is equally wrong; some compatibility layers keep misleading instructions partially functional.
- At least one repo-local source of truth disagrees with another prose artifact.
- The live operator path depends on exact command and flag names, not just a conceptually similar workflow.
- A shallow docs harmonization pass is intentionally insufficient.

## Required Surfaces

- Repo search and code reading.
- Shell command verification.
- Apply-patch style docs editing.
- Evidence must come from the provided benchmark bundle only. Searching unrelated workspace paths does not count as verifying the live operator path.

## Expected Deliverables

- A runbook patch that matches the live command path and current flag names.
- Verification notes that separate “checked directly” from “inferred from code”.
- A deploy note that explains the operator impact without restating the whole runbook.

## Grader Contract

- Full credit requires:
- The runbook names the live command entrypoint.
- The runbook uses the current config flag or environment variable names.
- The runbook does not rely on the legacy alias as the primary instruction.
- Verification notes include at least one directly checked command and one caveat.
- The deploy note accurately describes the user-visible change.
- `reconciliation_facts.json` matches the runbook and accurately separates current from deprecated names.
- Partial credit if the command path is fixed but one stale flag or env reference remains.
- Low credit if the updated runbook simply copies whichever README is closest.

## Red-Team Traps

- Copy stale README text into the runbook.
- Treat the legacy alias as proof that the stale command is still correct.
- Update the runbook without verifying the actual CLI help or entrypoint.
- Write a deploy note that claims behavior changes the repo does not make.

## Quality Gate

- Hardening applied after adversarial probe:
- The runtime prompt should not expose the exact drift shape beyond “runbook and repo disagree.”
- The bundle should include contradictory docs so the solver must privilege code over prose.
- The grader should compare the runbook against the live CLI help and helper scripts, not only against static docs.
- Hidden checks should fail answers that keep the legacy alias as the recommended path.
- `reconciliation_facts.json` should be validated directly against code or CLI extraction so a solver cannot patch only the main command and hand-wave the rest.
- The bundle boundary is now explicit after a live family-bundle attempt searched the broader workspace for surrogate evidence.
- Hardening target: a naive GPT-5.4/high solver should not clear 30/100 by harmonizing docs without checking the actual code path.
- Probe record:
- Initial adversarial probe judged the spec too easy at roughly 35-50/100 because the primary drift and partial-credit path were too exposed.
- Hardening added a structured facts artifact and hid the exact ambiguity pattern from the public prompt.
- A later live GPT-5.4/high family-bundle run searched outside the family directory for corroboration; the task now invalidates out-of-bundle verification.
- Current expectation after hardening: under 30/100 for a naive GPT-5.4/high solver if the grader validates facts against live help and helper-script extraction.
