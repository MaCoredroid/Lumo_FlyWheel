# Codex Surface Workflow Mapping

- `task_id`: `t2_codex_surface_daily_triage_mapping`
- `family_id`: `codex-surface-workflow-mapping`
- `scenario_type`: `codex_native_workflow_design`

## Task Prompt

This repo has an existing script-driven daily incident triage workflow, but no durable Codex-native surface for it. Read the repo and convert the real workflow into Codex artifacts without inventing a new process.

Produce:

- `artifacts/SKILL.md`: a repo-local skill for the triage workflow.
- `artifacts/codex_triage.toml`: Codex config for the same workflow entrypoint.
- `artifacts/automation_proposal.md`: a thread-heartbeat or cron-style automation proposal grounded in the repo’s real cadence and commands.
- `artifacts/mapping_note.md`: a short note tying each artifact back to the exact scripts, Make targets, docs, or fixtures it came from, including source paths and the exact command literals that justified each mapping decision.

Do not scaffold generic agent boilerplate. The artifacts should reflect the workflow the repo already encodes.

## Workspace Bundle

- Monorepo or service repo with `scripts/triage.py`, one or more Make targets, `docs/ops/daily_triage.md`, sample artifacts under `fixtures/`, and a stale abandoned skill draft.
- One duplicate helper script that looks similar but is no longer the main path.
- One calendar or schedule mention in docs that conflicts with the actual automation cadence used by the scripts.

## Seeded Ambiguity

Authoring note: these seeds guide benchmark construction and hidden checks. The solver-visible task prompt should not enumerate them directly.

- More than one artifact looks reusable, but at least one attractive option is stale.
- Entry-point, cadence, and repo evidence do not line up cleanly unless the solver checks multiple surfaces.
- Generic Codex scaffolding can look polished while still being disconnected from the actual workflow.
- Correct automation design depends on separating task semantics from schedule semantics.

## Required Surfaces

- Repo search and code or docs inspection.
- Skill authoring.
- Codex config authoring.
- Automation design grounded in repo evidence.
- Evidence must come from the provided benchmark bundle only. Cross-repo or authoring-note scavenging does not count as workflow proof.

## Expected Deliverables

- A skill that points to the real workflow entrypoint and scope.
- A TOML config artifact consistent with the same workflow.
- An automation proposal whose prompt describes the task itself and whose cadence matches evidence.
- A mapping note that cites the source artifacts used.

## Grader Contract

- Full credit requires:
- The skill uses the live script or Make target rather than the stale draft path.
- The TOML artifact and skill agree on entrypoint and scope.
- The automation proposal uses a cadence justified by repo evidence.
- The mapping note cites concrete repo sources for every major artifact decision and includes exact command literals for each mapped entrypoint.
- The artifacts avoid generic placeholders that could fit any repo.
- Partial credit if the workflow mapping is mostly right but one surface is stale or generic.
- Low credit if the output is polished but disconnected from repo reality.

## Red-Team Traps

- Reuse the stale skill draft with light edits.
- Invent a daily automation because it sounds plausible.
- Point different artifacts at different workflow entrypoints.
- Write a generic skill that never mentions the repo’s actual commands, fixtures, or docs.

## Quality Gate

- Hardening applied after adversarial probe:
- The runtime prompt should not expose which candidate script or cadence is stale.
- The grader should require cross-artifact consistency between skill, TOML, and automation proposal.
- Hidden checks should compare the chosen entrypoint against tests or current docs so stale scripts do not score.
- The automation proposal should lose credit if it bakes schedule details into the task prompt or ignores the repo’s real cadence.
- `mapping_note.md` should be validated for exact source paths and command literals so one lucky entrypoint guess is not enough.
- The bundle boundary is now explicit so family-bundle runs cannot upgrade guesses with unrelated workspace evidence.
- Hardening target: a naive GPT-5.4/high solver should not clear 30/100 by producing high-quality but repo-agnostic Codex scaffolding.
- Probe record:
- Initial adversarial probe judged the spec too easy at roughly 35-55/100 because a shallow solver could keep all artifacts internally consistent around one lucky entrypoint.
- Hardening added exact source-path and command-literal requirements and hid the stale-entrypoint clues from the public prompt.
- Current expectation after hardening: under 30/100 for a naive GPT-5.4/high solver if the grader enforces source-backed mapping consistency.
