# Agent Instructions — `codex-surface-workflow-mapping`

## Task

Map the live triage workflow in this repo into Codex-native artifacts. Do not invent a new workflow. Do not anchor on stale drafts or deprecated helpers.

## Inputs

- `Makefile`
- `scripts/`
- `docs/ops/`
- `ops/automation/`
- `fixtures/`
- `skills/legacy/`
- `.scenario_variant`
- `tests/test_workflow_map.py`


## Output — use the CLI

The only supported submission path is `./bin/cnb55-workflow-map`.

1. Write `workflow_input.json` at workspace root.
2. Validate it:

   ```
   ./bin/cnb55-workflow-map validate workflow_input.json
   ```

3. Submit it:

   ```
   ./bin/cnb55-workflow-map submit workflow_input.json
   ```

The CLI writes:

- `artifacts/workflow_map.json` (canonical, scored by the grader)
- `artifacts/SKILL.md`
- `artifacts/codex_triage.toml`
- `artifacts/automation_proposal.md`
- `artifacts/mapping_note.md`

## Rules

- Use only evidence from this workspace bundle.
- Do not write outside `artifacts/` except for `workflow_input.json`.
- Do not modify `Makefile`, `scripts/`, `docs/`, `ops/automation/`, `fixtures/`, `skills/`, `drafts/`, `release_context/`, `incident_context/`, `tests/`, `.scenario_variant`, `AGENTS.md`, `Dockerfile`, or `bin/`.
- Do not fetch network resources.

## What strong work looks like

- Skill, TOML, and automation all point at the same live entrypoint: `make codex-daily-triage`.
- Exact command literals are quoted from real files, not guessed.
- The automation proposal keeps task semantics separate from schedule semantics.
- Stale candidates are explicitly rejected with evidence-backed reasons.
- Variant-specific context (dirty state, release context, incident rollback) changes the mapping when it should.
