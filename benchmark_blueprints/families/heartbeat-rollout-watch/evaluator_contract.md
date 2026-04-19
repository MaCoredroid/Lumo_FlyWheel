# Evaluator Contract: Heartbeat Rollout Watch

## Purpose
Evaluate whether the solver repaired both the rollout checker and the thread-heartbeat follow-up behavior, not just the visible checker bug.

## Required submission surfaces
- Rollout health-check code
- Heartbeat automation config or prompt
- Deploy note
- Example follow-up artifact

## Visible gates
1. Threshold-evaluation tests pass on healthy and stalled fixtures.
2. Automation validation tests require exact `kind = "heartbeat"` and `destination = "thread"` semantics.
3. Follow-up validation accepts only the bounded `status`, `blocker`, and `next-check` schema.

## Trusted checks
- Wrong destination semantics or cron substitution are immediate low-score outcomes.
- Prompt must request a bounded follow-up note rather than an open-ended dump.
- Deploy note and example follow-up must describe the same cadence and destination semantics as the automation config.

## Hidden checks
- Stalled rollout fixture fails only under the correct threshold-key lookup.
- Cron-style automation is rejected even if cadence text looks valid.
- Bounded-artifact validation rejects prompt-only shortening that still allows status dumps.
- Cross-file consistency checks reject static doc/example rewrites disconnected from the automation.

## 100-point scoring breakdown
- `15` points: correct diagnosis of the threshold-key drift and wrong heartbeat semantics.
- `20` points: repaired rollout checker behavior in the concrete source. Proposed-only edits score at most `4/20`.
- `20` points: exact `kind = "heartbeat"` and `destination = "thread"` semantics in the concrete automation file. Proposed-only edits score at most `4/20`.
- `15` points: bounded follow-up contract with `status`, `blocker`, and `next-check`. Proposed-only edits score at most `3/15`.
- `10` points: deploy note aligned to the real automation semantics. Proposed-only edits score at most `2/10`.
- `10` points: example follow-up artifact aligned to the repaired checker and heartbeat. Proposed-only artifacts score at most `2/10`.
- `10` points: hidden robustness against cron substitution, broad prompts, and static artifact rewrites. No executed evidence yields `0/10`.

## Hard caps
- Cap at `20` if `kind != "heartbeat"` or `destination != "thread"`.
- Cap at `20` if the follow-up artifact is not schema-bounded.
- Cap at `30` if deploy note or example output diverges from the actual automation semantics.
- Cap at `25` for diagnosis-only or plan-only submissions that do not modify the concrete checker and automation artifacts.

## Shortcut defenses
- Make exact heartbeat semantics part of visible grading.
- Reject cron substitution even when cadence strings look correct.
- Validate bounded output independently of prose wording.

## Final hardness judgment
- Current naive GPT-5.4/high outlook: under `30/100` after hardening.
- Confidence: medium.
- Main reason: checker-only fixes and wrong automation semantics now hard-cap below 30.
