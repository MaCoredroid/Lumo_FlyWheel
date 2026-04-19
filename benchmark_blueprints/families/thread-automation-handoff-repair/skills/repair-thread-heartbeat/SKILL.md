# Repair Thread Heartbeat

Use this skill when a Codex automation should continue work in the current thread, but the stored automation bundle still reflects older detached-job semantics.

## Do

- Verify the automation is repaired in place rather than replaced
- Align `kind`, `destination`, cadence semantics, prompt wording, and skill docs
- Preserve unrelated operator notes and memory artifacts
- Keep the intended deliverable bounded to an in-thread handoff artifact

## Do Not

- Do not create a second automation file
- Do not leave the prompt asking for a file write or detached cron-style output
- Do not change unrelated notes to make diffs look cleaner

## Working Pattern

1. Treat the existing automation bundle as the source of truth to repair.
2. Make prompt semantics match the intended in-thread output mode.
3. Confirm docs and skill text reinforce the same duplicate-prevention rule.
4. Preserve all unrelated notes exactly.

## Success Signal

The same automation now wakes the thread correctly, asks for the right artifact, and does not spawn duplicate schedules.
