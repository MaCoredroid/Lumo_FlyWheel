# Policy Contract Repair

Use this skill when a Codex policy rename drifted between parser logic, local config, CI workflow, and workflow-preview output.

## Focus
- Keep one deprecated spelling accepted on input if the task calls for compatibility.
- Emit only canonical names in preview or exported artifacts.
- Verify local CI and preview generation agree on the same contract.

## Required Solver Behavior
- Inspect parser and preview logic before patching.
- Check whether release docs promise breaking or compatible behavior.
- Keep workflow jobs consistent with emitted preview output.

## Anti-Shortcut Rules
- Do not rename everything and drop compatibility.
- Do not normalize only the visible fixture path.
- Do not patch workflow YAML alone while leaving preview output stale.

## Minimum Verification
- Run parser and workflow-preview tests.
- Confirm canonical output is emitted for both canonical and deprecated inputs.

