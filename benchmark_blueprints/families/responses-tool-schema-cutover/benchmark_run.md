# Benchmark Run

## Run Metadata
- Family: `responses-tool-schema-cutover`
- Task: `cnb55-core-responses-tool-schema-cutover-router-replay`
- Child agent model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da330-d46b-7181-a79e-c33e61136bea`
- Attempt mode: solver attempt from family bundle context only

## Attempt Prompt
The child agent was told to treat this family directory as its effective workspace root, read `task_spec.md`, `evaluator_contract.md`, `codex/config.toml`, and the family-local skill, then attempt the benchmark task as a solver rather than critique the design.

## Attempt Summary
- Correctly identified the core repair shape: `call_id`-scoped joins, chronology-preserving reducer behavior, repeated same-name tool-call handling, and non-code updates.
- Named plausible target files and tests in the described benchmark workspace.
- Did not produce executable code, test output, transcript inspection results, or concrete proof from a real `/workspace` repo.

## Scoring
- Functional repair of adapter and reducer semantics: `8/45`
  - Strong intent and file targeting, but no concrete patch or code-level proof.
- Regression protection and test quality: `4/20`
  - Good proposed tests, but none were implemented or executed.
- Non-code contract alignment: `5/15`
  - Correctly called out config and migration note updates.
- Verification quality and evidence-backed reasoning: `2/10`
  - Named checks to run, but supplied no evidence.
- Robustness against hidden traces and shortcut resistance: `1/10`
  - Recognized chronology issues, but provided no demonstrated resistance to hidden traces.

## Cap Applied
- Cap applied: `20/100`
- Reason: the solver produced a plausible repair plan but no real code-level repair or test evidence, which triggers the family cap for artifact-only intent.

## Final Score
- `20/100`

## Judgment
- The family remains meaningful.
- The naive `gpt-5.4/high` solver attempt stayed near the target difficulty band.
- No rerun was needed after scoring because the result already landed at the desired ~20.

