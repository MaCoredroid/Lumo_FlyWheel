# Benchmark Run: `delegation-merge-salvage`

## Run Metadata

- Run type: child-subagent benchmark attempt
- Model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da330-f317-7081-a551-a30ac483a74d`
- Context restriction: family-local artifacts only under this family directory

## Attempt Summary

The solver produced a final patch summary, salvage postmortem, verification note, and reviewer-facing note. It captured the family’s intended merge logic well, but it lacked exact worker-patch hunk headers or line-ranged references because the actual worker artifacts were not available in the family-local context.

Most important failure mode:

- no exact hunk headers or line-ranged references from `worker_a.patch` and `worker_b.patch`
- no actual command output from verification
- salvage evidence stayed at file-path level

## Scoring Against `evaluator_contract.md`

### Salvage Artifact Discipline And Hunk-Level Evidence: 6 / 25

- the postmortem separated kept and rejected material from both workers
- exact hunk-level evidence was missing

### Behavioral Correctness Across Markdown, JSON, And Watchlist Follow-Up: 4 / 35

- the answer described the intended preservation goals, but no benchmark-workspace behavior or fixtures were shown

### Selective Merge Quality And Reviewability: 5 / 20

- the selective-merge narrative was coherent
- no actual final diff was available

### Verification Evidence Quality: 3 / 10

- commands were listed
- no command output or fixture comparison evidence was provided

### Reviewer-Facing Risk Explanation: 4 / 10

- the reviewer-facing note correctly explained the visible-green versus hidden-red merge risk

## Caps Applied

- Effective cap triggered by evaluator intent: file-path-only salvage guesses with no exact worker-artifact hunk headers or line-ranged references

## Final Score

- Final score: `22 / 100`

## Final Judgment

- Target judgment: acceptable
- Reason: the task remains meaningful, but artifact-free salvage narration does not earn substantial credit

