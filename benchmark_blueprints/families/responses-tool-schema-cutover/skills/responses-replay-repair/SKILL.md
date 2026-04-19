# Responses Replay Repair

Use this skill when the solver is repairing a Responses-native adapter or replay reducer.

## Focus
- Treat `call_id` as the stable join key for tool calls and tool results.
- Preserve transcript chronology from event data; do not sort output into a prettier order.
- Verify both adapter and reducer layers before assuming the bug is isolated.

## Required Solver Behavior
- Inspect at least one replay fixture before patching.
- Keep repeated same-name tool calls distinct all the way through render output.
- Update the Codex config and migration note if the runtime contract changed.

## Anti-Shortcut Rules
- Do not dedupe repeated tool calls by tool name.
- Do not synthesize `call_id` from visible fixture position.
- Do not "fix" rendering by filtering duplicate lines after the reducer.

## Minimum Verification
- Run the adapter and replay tests together.
- Confirm the repaired transcript preserves both same-name tool invocations.

