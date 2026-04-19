# Benchmark Run

## Run Metadata
- Family: `sandbox-policy-ci-drift`
- Task: `cnb55-core-sandbox-policy-ci-drift-local-ci-contract`
- Child agent model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da330-d61d-7142-ad35-e3e380637a2b`
- Attempt mode: solver attempt from family bundle context only

## Attempt Prompt
The child agent was told to treat this family directory as its effective workspace root, read `task_spec.md`, `evaluator_contract.md`, `codex/config.toml`, and the family-local skill, then attempt the benchmark task as a solver rather than critique the design.

## Attempt Summary
- Correctly identified the compatibility requirement, canonical-output rule, workflow-preview alignment, and helper-path shortcut risk.
- Named the right parser, config, preview, workflow, and release-note surfaces.
- Did not execute `make ci`, emit preview output, or show any concrete parser changes.

## Scoring
- Parser and config-loading correctness: `8/35`
  - Good plan, but no implementation.
- Workflow preview canonicalization: `4/20`
  - Correct target, no proof or output.
- CI workflow consistency: `3/15`
  - Named the workflow drift, but no concrete fix.
- Release note and Codex config alignment: `3/15`
  - Correct artifacts named, no actual edits.
- Compatibility handling plus anti-shortcut robustness: `2/15`
  - Strong behavioral understanding, no demonstrated protection against hidden cases.

## Cap Applied
- Cap applied: `20/100`
- Reason: the solver produced a solid intent-level repair plan but no executable parser, preview, CI, or artifact evidence.

## Final Score
- `20/100`

## Judgment
- The family remains meaningful.
- The naive `gpt-5.4/high` solver attempt stayed near the target difficulty band.
- No rerun was needed after scoring because the result already landed at the desired ~20.

