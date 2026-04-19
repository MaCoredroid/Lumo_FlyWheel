# Benchmark Run

## Run Metadata
- Family: `transcript-merge-regression`
- Task: `cnb55-core-transcript-merge-regression-reducer-diagnosis`
- Child agent model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da330-d68b-7453-a0e2-84e036d56a43`
- Attempt mode: solver attempt from family bundle context only

## Attempt Prompt
The child agent was told to treat this family directory as its effective workspace root, read `task_spec.md`, `evaluator_contract.md`, `codex/config.toml`, and the family-local skill, then attempt the benchmark task as a solver rather than critique the design.

## Attempt Summary
- Correctly identified that the fix belongs in the merge reducer, not the render layer.
- Named stable event identity, deferred post-completion handling, summary coupling, tests, and incident-note repair.
- Did not inspect real transcript artifacts, patch reducer code, or demonstrate summary preservation.

## Scoring
- Merge-layer repair and event-identity correctness: `8/45`
  - Strong invariant diagnosis, but no code.
- Regression protection across merge, render, and summary tests: `4/20`
  - Good intended tests, none executed.
- Incident-summary preservation: `3/15`
  - Correct concern, no proof.
- Incident note quality: `3/10`
  - Good intended note content, not written.
- Shortcut resistance across hidden traces: `2/10`
  - Recognized bad fixes, but did not prove robustness.

## Cap Applied
- Cap applied: `20/100`
- Reason: the solver named the right invariant but provided no reducer implementation, transcript-backed verification, or incident-summary evidence.

## Final Score
- `20/100`

## Judgment
- The family remains meaningful.
- The naive `gpt-5.4/high` solver attempt stayed near the target difficulty band.
- No rerun was needed after scoring because the result already landed at the desired ~20.

