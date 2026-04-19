# Runbook Reconcile

Use this skill when an operational runbook drifted from the code and the task is to reconcile prose with the real command path.

## Objective

Make the runbook reflect the live operator path, not the most familiar or most copied command.

## Procedure

1. Identify the live entrypoint from code or CLI help.
2. Separate current names from deprecated compatibility names.
3. Update the runbook so the primary procedure uses only current names.
4. Record facts in a structured artifact.
5. Keep the deploy note limited to operator-facing instruction changes.

## Guardrails

- README fragments are evidence, not truth.
- Compatibility aliases belong in caveats, not in the primary steps.
- Verification notes must distinguish what was checked from what was inferred.

