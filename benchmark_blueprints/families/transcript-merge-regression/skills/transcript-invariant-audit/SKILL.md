# Transcript Invariant Audit

Use this skill when session replay corruption appears to come from duplicate or missing tool-output blocks.

## Focus
- Identify the stable event identity used for merge, not just visible render output.
- Separate legitimate deferred output from debug-only post-completion noise.
- Keep incident-summary semantics tied to the trusted merged representation.

## Required Solver Behavior
- Inspect at least one noisy and one clean transcript artifact.
- Verify whether the defect is in merge, render, or summary code before patching.
- Update the incident note if the old remediation guidance is now wrong.

## Anti-Shortcut Rules
- Do not filter duplicates only after rendering.
- Do not drop all post-completion fragments.
- Do not derive event identity from visible fixture order alone.

## Minimum Verification
- Run merge, render, and summary tests together.
- Confirm a same-name tool-output pair survives merge distinctly.

