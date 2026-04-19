# Heartbeat Rollout Watch

Scenario type: `automation_orchestrated_followup`

## Task prompt
Repair a rollout health checker and then create or update a thread heartbeat automation that revisits the same rollout later with a bounded follow-up artifact. The current checker compares the wrong threshold key and reports healthy when the rollout is actually stalled. The seeded heartbeat automation also targets the wrong destination and produces an unbounded “status dump” instead of a concise follow-up note. Fix the checker, align the automation so it wakes this thread on the intended cadence, and update the deploy note that explains what the follow-up run should report.

## Workspace bundle
- `rollout_watch/src/health_check.py` and `thresholds.py` with a seeded threshold-key drift bug.
- `rollout_watch/automation/heartbeat_rollout_watch.toml` with wrong destination semantics and overly broad prompt wording.
- `rollout_watch/fixtures/rollout/` containing healthy, stalled, and recovering rollout snapshots.
- `rollout_watch/docs/deploy_note.md` with stale follow-up instructions.
- `rollout_watch/reports/example_followup.md` showing the bounded artifact shape expected after the heartbeat wakes.
- `rollout_watch/tests/` for health-check logic, automation config validation, and follow-up note rendering; hidden checks verify thread destination semantics, bounded artifact wording, and that the heartbeat is not created as a separate cron-style job.

## Seeded automation or maintenance drift
- The checker reads `error_budget_remaining` instead of the current rollout-specific threshold key, so stalled rollouts can be marked healthy.
- The automation prompt asks for “everything relevant about rollout state,” which encourages long dumps rather than a bounded follow-up artifact.
- The seeded automation targets a separate job-style destination rather than the current thread.
- The deploy note still describes the old cadence and wrong follow-up scope.

## Required surfaces
- `automation`
- `shell`
- `apply_patch`
- Health-check testing against seeded rollout fixtures
- Automation or schedule reasoning about thread versus separate-job semantics

## Deliverables
- A code patch for the rollout health checker.
- An updated heartbeat automation config or prompt artifact targeted at the correct destination.
- A deploy note describing the repaired cadence and the bounded follow-up artifact.
- A refreshed example follow-up note matching the repaired checker semantics.

## Grader contract
- Visible tests must pass for threshold evaluation, follow-up rendering, exact automation semantics, and a bounded follow-up validator that accepts only `status`, `blocker`, and `next-check` fields or sections.
- Trusted checks verify that the automation targets the thread-style destination and that its prompt asks for a bounded follow-up note instead of an open-ended dump.
- Hidden checks require:
  - A stalled rollout fixture that only fails if the correct threshold key is used.
  - Rejection of a separate cron-style automation even if the cadence string is otherwise valid.
  - A follow-up artifact that stays within a bounded schema with status, blocker, and next-check fields.
- Cross-file consistency checks require the deploy note and example follow-up to describe the same cadence, destination semantics, and bounded artifact shape encoded in the automation config.
- Scoring cap: if the automation is not explicitly `kind = "heartbeat"` with `destination = "thread"`, the submission cannot score above `20/100` even if the checker fix is correct.

## Red-team traps
- Fix the checker but leave the automation as a separate job.
- Preserve the wrong broad prompt and rely on a renderer to trim output locally.
- Hardcode the visible stalled rollout fixture.
- Update the example follow-up note without aligning the actual automation config.
- Switch the cadence but not the destination semantics.

## Quality gate
- Actual GPT-5.4/high family-bundle run score: `20/100` under the current evaluator.
- Naive GPT-5.4/high under-30 verdict after hardening: `yes`.
- Run method note: actual child-agent solver attempt against the family bundle only; concrete `rollout_watch/` workspace files were not present in this authoring repo, so the submission could identify the heartbeat semantics and checker intent but not verify them against a live bundle.
- Hardening applied after probe:
  - The task is not complete until both the checker and the future wakeup behavior are correct.
  - Hidden validation distinguishes thread heartbeat semantics from generic scheduled jobs.
  - The bounded follow-up schema blocks trivial “dump all status” prompt fixes.
  - Exact heartbeat semantics and bounded-artifact validation now moved into visible grading, not only hidden grading.
  - Wrong destination semantics or cron substitution now hard-cap the score at 20.
  - Cross-file consistency checks now punish static doc/example rewrites that do not match the actual automation.
- Residual risk accepted: a solver may still fix the checker first, but without the true heartbeat semantics it should remain near 20 rather than drifting toward 30.
