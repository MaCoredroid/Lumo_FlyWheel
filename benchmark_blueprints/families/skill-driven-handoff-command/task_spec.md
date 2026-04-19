# Skill-Driven Handoff Command

## Task Metadata
- `task_id`: `cnb55-core-skill-driven-handoff-command-report-bundle`
- `family_id`: `skill-driven-handoff-command`
- `scenario_type`: `feature_evolution`

## Task Prompt
Add a repo-local `handoff summarize` command that converts runtime artifact bundles into a structured Markdown handoff for the next Codex operator. The command must be accurately exposed through the skill documentation and an automation template, and the sample output should match the implemented command semantics. Keep the existing JSON export path intact. Do not solve this by checking in a canned Markdown file that ignores the runtime inputs.

## Workspace Bundle
- Developer-productivity repo rooted at `/workspace`.
- `handoff/cli.py` and `handoff/render.py`.
- `samples/runtime_bundle/` with JSON summaries, transcript snippets, and a screenshot manifest.
- `.codex/skills/handoff/SKILL.md`.
- `automations/handoff_digest.toml`.
- `docs/handoff_workflow.md`.
- `tests/test_cli.py`, `tests/test_render.py`, `tests/test_skill_contract.py`.

## Seeded Breakage
- CLI parser exposes `handoff export --format json` only.
- Renderer has a dormant Markdown path that skips screenshot evidence and open questions.
- `SKILL.md` still tells Codex to open JSON directly rather than use the command.
- Automation template invokes a removed script entrypoint.
- Sample output in docs was hand-written and no longer matches runtime field ordering or optional sections.

## Required Surfaces
- `shell`
- `apply_patch`
- skills
- automation config
- terminal tests

## Expected Deliverables
- CLI and renderer patch for `handoff summarize`.
- Tests covering Markdown output, preserved JSON path, and required sections.
- `SKILL.md` update aligned to the real command.
- Automation template update that invokes the supported entrypoint.
- Docs or sample output refresh generated from the implemented command.

## Grader Contract
- Visible gate:
  - CLI and render tests pass.
  - skill-contract test verifies `SKILL.md` references the shipped command.
- Hidden checks:
  - Markdown output must include screenshot evidence and open questions when present.
  - JSON export behavior must remain unchanged.
  - Automation template must call the canonical command path, not a removed helper.
  - Sample output diff must match the renderer on a withheld runtime bundle.
  - End-to-end automation check runs the template against real runtime inputs and asserts it invokes the shipped CLI entrypoint.
  - Contract test generates Markdown from two withheld bundle shapes and diffs required section ordering and content against the documented sample contract.
- Fail conditions:
  - Canned Markdown sample with no true command implementation.
  - Docs and skill updates that drift from actual CLI behavior.
  - Breaking the existing JSON export path.

## Red-Team Traps
- Add a fake `summarize` alias that prints a static template.
- Update `SKILL.md` and docs only.
- Render Markdown from one visible sample bundle shape while failing optional sections on hidden inputs.
- Patch the automation template to echo canned output instead of invoking the CLI.
- Remove JSON-specific tests to make room for the new mode.

## Quality Gate
- Probe target:
  - A naive `gpt-5.4/high` solver should land around `20/100`, not `30/100`, unless it updates code, skills, automation, docs, and the sample-output contract coherently with proof.
- Hardening applied after adversarial solver run:
  - Hidden checks compare generated output from withheld bundles against the checked-in sample contract.
  - JSON preservation blocks a replace-instead-of-extend implementation.
  - Skill and automation surfaces are first-class graded artifacts, not optional docs.
  - Hidden coverage now explicitly exercises automation execution and two withheld bundle shapes, which blocks a visible-sample-only Markdown path.
- Actual run result:
  - Child subagent solver attempt using `gpt-5.4/high` scored `20/100` under the family rubric.
  - The run outlined the right multi-surface edits but had no real CLI implementation, no automation execution, and no output diff evidence.
  - Final assessment: `target hardness met; naive gpt-5.4/high stays around 20/100`.
