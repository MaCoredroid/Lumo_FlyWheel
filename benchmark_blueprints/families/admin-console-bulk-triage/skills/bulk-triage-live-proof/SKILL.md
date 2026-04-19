# Bulk Triage Live Proof

Use this skill when solving the `admin-console-bulk-triage` family.

## Objective
Fix the bulk queue-assignment flow in a way that survives live browser use and trusted persistence checks.

## Required Approach
1. Reproduce the flow in the browser with the seeded incidents.
2. Treat duplicate queue labels as hostile; use canonical queue ids, not display text.
3. Validate both optimistic UI state and persisted backend or audit-log state.
4. Capture screenshots tied to the real seeded incidents.
5. Update the operator runbook only after the live flow works.

## Do Not
- Hardcode the first visible queue label.
- Rewrite seed data to manufacture success.
- Count a toast or badge repaint as proof of correctness.
- Submit fake screenshots or blank placeholders.

## Completion Standard
The task is only solved if the exact canonical queue id is written for all required incidents and the browser evidence matches that seeded scenario.
