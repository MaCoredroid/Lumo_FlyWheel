# Constraint Recovery Audit

Use this skill when a task requires recovering a local workflow under several simultaneous constraints.

## Workflow
1. Enumerate the non-negotiable constraints before patching code.
2. Identify any irreversible prior step in the partial state.
3. Repair the workflow so all constraints hold together.
4. Validate the recovered path against both visible and hidden-like partial states.

## Anti-Patterns
- Fixing one constraint while violating another.
- Replaying irreversible steps.
- Documenting recovery without implementing it.

## Done Signal
- The recovered workflow satisfies all declared constraints and documents the boundary correctly.
