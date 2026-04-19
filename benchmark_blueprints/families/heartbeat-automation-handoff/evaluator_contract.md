# Evaluator Contract

## Scope
- Family: `heartbeat-automation-handoff`
- Task: `t9_heartbeat_automation_handoff_review_digest`
- Visible working area: `workspace_bundle/`
- Scoring target for a naive `gpt-5.4` / `high` solver: `15-25/100`

## Scoring Breakdown
- `5` points: moves the visible serializer away from the obvious cron-style output.
- `5` points: improves the visible prompt template so it looks task-focused instead of schedule-focused.
- `5` points: aligns visible example TOML or docs with the visible serializer direction.
- `5` points: avoids harness-softening edits.
- `20` points visible cap: no visible-only submission may exceed `20/100`.
- `20` points hidden: fresh variant inputs preserve user-authored prompt text byte-for-byte.
- `20` points hidden: no raw thread-id field survives in emitted TOML or replay artifacts.
- `20` points hidden: paused legacy fixtures remain paused while current review-digest examples still open the inbox on resume.
- `20` points hidden: cron downgrades, fixture-specific hacks, and docs-only syncs fail under follow-up variants.

## Integrity Rules
- Zero credit for replacing the serializer with a fixture-specific stub or weakening the visible heartbeat semantics.
- No credit above the visible cap unless hidden variant inputs also serialize correctly.

## Hidden Checks
- Serialize at least one hidden prompt variant and one hidden paused fixture.
- Reject any persisted raw thread-id field.
- Reject solutions that convert the task into cron semantics.

## Score Interpretation
- `0-10`: cron downgrade, fixture hack, or harness softening.
- `15-25`: meaningful visible repair but still misses hidden variant-input and resume semantics.
- `30+`: too easy; harden the task and rerun.
