# `codex-skill-runtime-v2-split` Task Spec

## Task Prompt
Refactor the `ops-handoff` Codex runtime bundle from one monolithic prompt/runbook into structured skill bundles, repo-local Codex config, and a canonical heartbeat automation definition. The new structure must drive a real runnable handoff command, remove stale prompt references, keep only the canonical automation, and leave unrelated in-progress edits untouched. Update the operator runbook so it matches the executable bundle, not just the new file layout.

## Workspace Bundle
- `docs/oncall_handoff_monolith.md`: legacy monolith still referenced by config and docs.
- `skills/legacy_handoff.md`: copied prompt fragment that should be retired.
- `scripts/run_handoff.py`: runnable handoff command that produces a structured Markdown artifact.
- `fixtures/handoff_input.json`: sample incident bundle for the smoke path.
- `.codex/config.toml`: partially edited config with one wrong path and one obsolete alias.
- `automations/handoff-primary.toml`: intended canonical heartbeat but miswired.
- `automations/handoff-copy.toml`: stale duplicate automation with conflicting cadence.
- `docs/runbooks/oncall.md`: operator runbook with stale references.
- `tests/test_skill_bundle.py`, `tests/test_config_refs.py`, `tests/test_automation_smoke.py`: visible checks.
- `skills/other_flow/SKILL.md`: unrelated dirty-workspace file in v3 variants.

## Seeded Modernization Breakage
- The repo still routes operators through one monolithic runbook and a copied legacy prompt.
- `.codex/config.toml` points at stale prompt paths instead of structured skills.
- Two similar automation files exist, but only one should remain canonical for the task.
- The runnable handoff script is real; docs and skill/config artifacts must point to it correctly.
- A follow-up variant asks for a second heartbeat or adjacent skill to reuse the new structure cleanly.

## Required Surfaces
- `shell`
- `apply_patch`
- skills and Markdown editing
- TOML editing
- automation semantics
- terminal smoke validation

## Expected Deliverables
- One or more structured `SKILL.md` bundles replacing the monolithic prompt path.
- Updated `.codex/config.toml` that references only the canonical live skill paths.
- Canonical heartbeat automation config aligned with the real command path.
- Updated operator runbook aligned with the new bundle and smoke command.
- No destructive rewrite of unrelated skill or automation edits.

## Grader Contract
- Phase 2 visible checks:
  - `pytest -q tests/test_skill_bundle.py tests/test_config_refs.py tests/test_automation_smoke.py`
  - `python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md`
- Hidden checks:
  - `SKILL.md` examples and command references point at runnable repo paths.
  - `.codex/config.toml` registers the same canonical skill paths that exist on disk.
  - Exactly one canonical automation remains for the task path; the stale duplicate no longer controls execution.
  - Runbook, skill, config, and automation all describe the same executable workflow.
  - Follow-up variant can add a second skill or heartbeat by reusing the new structure rather than cloning stale text.
- Trusted final-state checks:
  - No live references remain to the monolithic prompt or stale duplicate automation.
  - The smoke command succeeds when driven from the authored bundle.
  - Dirty-workspace sentinel files outside the task path remain untouched.
- Scoring intent:
  - Static docs/config/automation presence is capped below `0.30`.
  - Scores above `0.50` require canonical cross-file alignment and a successful smoke path.
  - Scores above `0.75` require hidden follow-up reuse and duplicate-automation cleanup.

## Red-Team Traps
- Split the monolith into new files without rewiring config or automation.
- Keep both automation files and hope the grader only checks file presence.
- Copy old prompt text into a new skill without deriving a real runnable workflow.
- Update docs only, leaving the handoff command unreachable.
- Overwrite unrelated edits in `skills/other_flow/SKILL.md`.
- Add a second heartbeat in follow-up by cloning stale text instead of reusing the structured bundle.

## Quality Gate
- GPT-5.4/high adversarial probe judgment before hardening: `under_30_for_naive_solver = no`.
- Probe found the family too scaffoldable if the grader only checks that skills, config, automation, and docs exist.
- Hardening applied in this spec:
  - Added a real runnable handoff command and bound the task to it.
  - Required canonical uniqueness across automation and config references.
  - Elevated stale-reference removal and dirty-workspace preservation into trusted checks.
  - Required hidden follow-up reuse of the new structure instead of simple copy-paste.
- Final quality judgment after hardening: `under_30_for_naive_solver = yes`.
- Rationale: a naive solver can still create plausible artifacts, but without executable alignment, canonical cleanup, and structured reuse it should not cross the 30-point band.
