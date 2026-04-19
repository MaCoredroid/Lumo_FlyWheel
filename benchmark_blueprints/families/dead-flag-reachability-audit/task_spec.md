# Dead Flag Reachability Audit

- `task_id`: `t2_dead_flag_preview_reachability_audit`
- `family_id`: `dead-flag-reachability-audit`
- `scenario_type`: `reachability_audit`

## Task Prompt

Audit the preview rollout flags in this repo and determine which are live, partially wired, or dead. The operator request names three controls:

- `ENABLE_SHADOW_PREVIEW`
- `ENABLE_PREVIEW_V2`
- `PREVIEW_FORCE_LEGACY`

You must trace defaults, environment parsing, live call sites, tests, and rollout docs before you classify anything. Produce:

- `artifacts/flag_audit.md`: one section per flag with status, evidence, and operational risk.
- `artifacts/reachability_matrix.json`: for each flag, record whether it appears in defaults, env parsing, runtime branching, tests, and docs, along with `parser_symbol`, `runtime_branch_symbol`, `alias_of`, and one disproved false-positive path.
- `artifacts/cleanup.patchplan.md`: a narrow cleanup recommendation for any flag you prove dead or docs-only.

Do not delete flags or patch runtime logic. This task is about correct reachability judgment, not speculative cleanup.

## Workspace Bundle

- Small service repo with `config/defaults.toml`, `src/preview_cli.py`, `src/preview/service.py`, `src/preview/runtime.py`, `.env.example`, deployment docs, and tests.
- One rollout memo that uses the old flag vocabulary.
- One env parser that still accepts a legacy alias.
- One hidden branch where a flag appears in docs and tests but no longer changes runtime behavior.

## Seeded Ambiguity

Authoring note: these ambiguity seeds are for benchmark design and hidden checks. They should not appear verbatim in the solver-visible task prompt.

- Not every named control with parser or docs presence changes live production behavior.
- At least one control has misleading evidence because one surface suggests reachability while another does not.
- Naming and alias behavior are intentionally confusable enough that grep counts are not trustworthy.
- The prose rollout story is not a reliable substitute for runtime tracing.

## Required Surfaces

- Repo search and code reading across config, env parsing, runtime, docs, and tests.
- Shell for targeted inspection and optional test execution.
- Structured JSON output for the reachability matrix.
- Evidence must come from the provided benchmark bundle only. Workspace-wide analogy hunting or sibling scenario lookups do not count.

## Expected Deliverables

- An audit that classifies each flag correctly and explains why.
- A matrix artifact that makes the classification auditable.
- A cleanup plan that stays within proven evidence and does not over-prescribe implementation work.

## Grader Contract

- Full credit requires:
- Correct live or partial or dead classification for all three named flags.
- Evidence from at least four distinct surfaces across config, parser, runtime, tests, and docs.
- A correct distinction between parser presence and behavioral reachability.
- A correct distinction between a live alias and a dead standalone flag.
- `reachability_matrix.json` names the actual parser symbol and runtime branch symbol for each flag and includes one disproved false-positive path.
- A cleanup recommendation that matches the proven classification.
- Partial credit if two flags are classified correctly and the third is close but unsupported.
- Low credit if the solver infers reachability from grep presence alone or collapses partial and dead into the same bucket.

## Red-Team Traps

- Treat every parsed env var as live behavior.
- Treat every documented flag as production-reachable.
- Miss the alias mapping and double-count it as a separate live control.
- Use tests as the sole source of truth even when they only exercise a stub path.
- Recommend deleting a flag without proving dead runtime reachability.

## Quality Gate

- Hardening applied after adversarial probe:
- The runtime prompt should not reveal the bucket distribution or the exact ambiguity pattern.
- `reachability_matrix.json` now requires parser and runtime symbols plus one disproved false-positive path.
- The bundle boundary is now explicit so metadata-only guessing cannot be laundered with unrelated workspace evidence.
- The hidden grader should verify that a solver cannot score by simply listing file matches; the narrative must align with runtime branching.
- The cleanup deliverable should be judged against overreach. Cleanup that goes beyond proven evidence should lose credit.
- Hidden grader notes may enforce a live or partial or dead distribution, but that fact should stay out of the public prompt.
- Hardening target: a naive GPT-5.4/high solver should not clear 30/100 via grep plus rollout-doc paraphrase.
- Probe record:
- Initial adversarial probe judged the spec too easy at roughly 35-55/100 because the public text leaked the answer skeleton.
- Hardening moved bucket-shape clues into hidden notes and added structured runtime-proof requirements.
- Current expectation after hardening: under 30/100 for a naive GPT-5.4/high solver if the grader validates branch symbols and disproved paths.
