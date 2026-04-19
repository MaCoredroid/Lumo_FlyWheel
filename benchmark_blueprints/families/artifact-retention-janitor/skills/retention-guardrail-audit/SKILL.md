# Retention Guardrail Audit

Use this skill when a cleanup or retention task must preserve current, pinned, or explicitly retained artifacts while still reclaiming eligible old state.

## Inputs
- Retention selector code
- Policy config
- Filesystem fixtures and manifest metadata
- Dry-run or deletion reports

## Workflow
1. Identify all protected-state sources: current markers, manifest pins, retain-on-failure metadata, and symlink targets.
2. Confirm at least one hidden-eligible artifact still deletes; over-preservation is not a pass.
3. Compare actual candidate selection with the dry-run report.
4. Update docs only after selector semantics are correct.

## Avoid
- Protecting all failed or corrupt artifacts.
- Trusting SQLite only when filesystem markers disagree.
- Fixing only rendered reasons while leaving the candidate set unsafe.

## Expected output
- Corrected selector logic
- Config aligned to live protection keys
- Dry-run artifact that explains protection and eligibility decisions
