# Evaluator Contract: `candidate-worktree-shootout`

## Purpose

Score a worktree-isolated candidate comparison where the solver must evaluate two bounded implementations, reject the shallower one with evidence, and land the stronger option cleanly.

## Required Submission Artifacts

- final integrated patch
- `artifacts/comparison/candidate_a_eval.md`
- `artifacts/comparison/candidate_b_eval.md`
- comparison note
- verification note

## 100-Point Breakdown

- 15 points: Candidate A isolated evaluation quality
- 15 points: Candidate B isolated evaluation quality
- 30 points: final strategy correctness across CLI and non-CLI callers
- 15 points: service-level regression-test coverage
- 15 points: comparison reasoning and rejected-candidate analysis
- 10 points: workspace cleanliness and docs alignment

## Trusted Evaluation Phases

### Phase 1: Isolation Gate

Cap at 20 if either candidate evaluation is missing a worktree path, exact commands run, touched files, or observed results.

Scoring:

- 15 points: Candidate A evidence quality
- 15 points: Candidate B evidence quality

### Phase 2: Correctness Gate

Award correctness credit only if the landed patch:

- satisfies visible CLI behavior
- satisfies hidden non-CLI caller behavior
- establishes service-layer ownership of normalization
- includes or preserves a service-level regression test covering the non-CLI path

If normalization remains materially duplicated in `cli.py`, cap at 25.

Scoring:

- 20 points: final strategy fixes visible CLI and hidden non-CLI behavior
- 10 points: normalization ownership moved to the correct layer
- 15 points: service-level regression test exists and targets the non-CLI path

### Phase 3: Comparison Gate

Award comparative judgment credit only if the submission:

- explains why the rejected candidate can look attractive on visible checks
- ties rejection to concrete evidence from the isolated evaluation
- avoids cross-candidate contamination in the final patch

Narrative claims without per-worktree evidence do not count.

Scoring:

- 10 points: evidence-backed rejection of the weaker candidate
- 5 points: clean final patch and docs consistency

## Hidden Checks

- hidden non-CLI regression case fails under CLI-local normalization
- diff inspection checks that the final patch does not blend both strategies incoherently
- docs check ensures the contract location matches the chosen design

## Red-Team Defenses

- one-path visible-test fixes cap low
- “I considered both” prose without evidence caps low
- in-place experimentation that ignores worktree isolation caps low

## Score Caps

- Missing one candidate evaluation artifact: max 20
- No per-worktree evidence for both candidates: max 20
- CLI-local quick fix that misses non-CLI path: max 25
- Cross-candidate blended patch: max 30
- No service-level regression test: max 35

## Hardness Judgment

- Current judgment: under 30 likely for a naive `gpt-5.4` / `high` solver
- Why: meaningful credit requires per-worktree evidence plus the hidden non-CLI fix
