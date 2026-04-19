# Benchmark Run: `candidate-worktree-shootout`

## Run Metadata

- Run type: child-subagent benchmark attempt
- Model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da330-ea57-7572-83a1-d054aa1b799c`
- Context restriction: family-local artifacts only under this family directory

## Attempt Summary

The solver produced two candidate evaluations, a final strategy summary, a comparison note, and a verification note. It followed the intended reasoning pattern well and selected the stronger service-layer strategy. The weakness is that the worktree paths, commands, and outcomes were asserted from family-doc context rather than backed by benchmark-workspace artifacts or captured stdout.

Most important failure mode:

- no actual captured stdout or failure excerpts from either worktree
- no benchmark-workspace patch artifact
- worktree evidence was claimed rather than demonstrated

## Scoring Against `evaluator_contract.md`

### Candidate A Isolated Evaluation Quality: 6 / 15

- worktree path, commands, and intended result were provided
- no concrete stdout or failure excerpt was provided

### Candidate B Isolated Evaluation Quality: 6 / 15

- worktree path, commands, and intended result were provided
- no concrete stdout or success excerpt was provided

### Final Strategy Correctness Across CLI And Non-CLI Callers: 5 / 30

- the chosen strategy was conceptually correct, but no actual benchmark-workspace patch or hidden-path proof was provided

### Service-Level Regression-Test Coverage: 2 / 15

- regression-test intent was named, but no concrete test artifact or diff was supplied

### Comparison Reasoning And Rejected-Candidate Analysis: 5 / 15

- the rejected candidate analysis was coherent and aligned with the family goal

### Workspace Cleanliness And Docs Alignment: 0 / 10

- no actual final diff or docs artifact was available

## Caps Applied

- Effective cap triggered by evaluator intent: no stdout or failure excerpt from either candidate evaluation

## Final Score

- Final score: `24 / 100`

## Final Judgment

- Target judgment: acceptable
- Reason: the task remains coherent, but a solver cannot score well without real per-worktree evidence

