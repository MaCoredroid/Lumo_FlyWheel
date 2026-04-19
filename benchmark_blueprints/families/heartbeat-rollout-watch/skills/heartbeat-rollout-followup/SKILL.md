# Heartbeat Rollout Follow-Up

Use this skill when a rollout task requires both a checker repair and a recurring thread heartbeat that returns a bounded follow-up artifact.

## Inputs
- Health-check code and threshold definitions
- Heartbeat automation config or prompt
- Deploy note
- Example follow-up artifact

## Workflow
1. Repair the rollout-state logic first, but do not stop there.
2. Confirm the automation is a thread heartbeat, not a separate scheduled job.
3. Enforce a bounded follow-up schema with `status`, `blocker`, and `next-check`.
4. Keep the deploy note and example artifact consistent with the real automation semantics.

## Avoid
- Fixing the checker only.
- Swapping in cron-style automation.
- Shortening the prompt without bounding the output schema.

## Expected output
- Correct health-check logic
- Exact heartbeat semantics
- One bounded follow-up artifact example aligned to the automation
