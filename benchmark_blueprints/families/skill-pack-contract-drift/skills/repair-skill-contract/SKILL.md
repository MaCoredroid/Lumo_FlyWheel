# Repair Skill Contract

Use this skill when a repo-local Codex skill is broken by drift between `SKILL.md`, example prompts, config wiring, and the live validator command.

## Do

- Trace the documented command path from `SKILL.md` into config and the validator entrypoint
- Compare example output claims against what the validator is supposed to emit
- Preserve unrelated edits and avoid broad rewrites
- Prefer repairing the live contract rather than adding aliases

## Do Not

- Do not keep a legacy helper name alive just to satisfy docs
- Do not rewrite fixtures to fake validator output
- Do not weaken tests or assertions to make the contract “look” aligned

## Working Pattern

1. Identify the single authoritative validator entrypoint.
2. Align `SKILL.md`, examples, and config with that entrypoint.
3. Verify the real emitted report still matches the documented shape.
4. Leave unrelated skill-pack state untouched.

## Success Signal

The documented operator workflow is executable end to end, and no legacy helper or compatibility shim remains.
