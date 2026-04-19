# Handoff Summarize Rollout

Use this skill when a repo needs a real `handoff summarize` command that turns runtime bundles into Markdown for the next Codex operator.

## Focus
- Extend the CLI and renderer without regressing JSON export.
- Keep the skill doc, automation template, and sample output aligned to shipped behavior.
- Preserve optional sections such as screenshot evidence and open questions when present.

## Required Solver Behavior
- Inspect one runtime bundle before patching.
- Verify the Markdown output is generated from runtime inputs, not checked-in text.
- Update the skill and automation artifacts to call the canonical command path.

## Anti-Shortcut Rules
- Do not print a static Markdown template.
- Do not patch docs or `SKILL.md` without implementing the command.
- Do not leave automation invoking a removed helper path.

## Minimum Verification
- Run CLI and render tests.
- Confirm JSON export still matches preexisting behavior.

