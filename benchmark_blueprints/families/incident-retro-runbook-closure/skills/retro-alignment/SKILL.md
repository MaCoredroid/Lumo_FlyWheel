# Retro Alignment

Use this skill when solving the `incident-retro-runbook-closure` family.

## Workflow
1. Read `retro/action_items.json` first and treat it as authoritative.
2. Extract the required verification command, sequence, and escalation target.
3. Align exactly three surfaces: runbook, helper, and automation prompt.
4. Keep the helper patch narrow and behavior-preserving.
5. Record only the actionable retro items in the operator-facing follow-up note.

## Avoid
- schedule-only automation edits
- leaving both old and new commands in the runbook
- using prose retro notes to overrule `action_items.json`
- doc-only closure
