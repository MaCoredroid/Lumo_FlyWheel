# Workflow Mapping

Use this skill when the task is to turn an existing repo workflow into Codex-native artifacts such as a skill, config, and automation plan.

## Objective

Map the real workflow, not the most convenient-looking draft or script.

## Procedure

1. Identify all plausible workflow entrypoints first.
2. Resolve which one is live using tests, current docs, or repeated script references.
3. Use the same entrypoint across all artifacts.
4. Quote exact commands and source paths in the mapping note.
5. Separate task semantics from schedule semantics in the automation proposal.

## Guardrails

- Internal consistency is necessary but not sufficient.
- A stale skill draft can be useful evidence but not the final source of truth.
- Generic Codex scaffolding should lose to repo-grounded specificity every time.

