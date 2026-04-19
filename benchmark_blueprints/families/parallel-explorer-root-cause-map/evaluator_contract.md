# Evaluator Contract: `parallel-explorer-root-cause-map`

## Purpose

Score a read-heavy orchestration task where the solver must decompose investigation work across non-overlapping explorer questions, synthesize the results, and identify the true root cause without landing speculative code.

## Required Submission Artifacts

- Root-cause brief
- Ranked suspect list
- Evidence table
- Bounded remediation plan
- Verification note

## 100-Point Breakdown

- 20 points: structure and submission completeness
- 35 points: causal accuracy and symbol-level tracing
- 20 points: false-lead disproof and contradictory-artifact handling
- 15 points: orchestration synthesis across non-overlapping subquestions
- 10 points: bounded remediation quality and non-goals

## Trusted Evaluation Phases

### Phase 1: Structure Gate

Reject or cap at 20 if any of these are missing:

- ranked suspect list with at least three candidates
- evidence table with at least four rows
- explicit remediation non-goals
- verification note quoting a failing assertion

Scoring:

- 5 points: ranked suspect list quality
- 10 points: evidence table completeness
- 5 points: verification note and remediation non-goals

### Phase 2: Causal Accuracy Gate

Award core credit only if the submission:

- traces a causal path from normalization into aggregation
- names one adapter symbol, one aggregation symbol, one renderer symbol ruled out, and one test or artifact location
- explains why at least one conflicting artifact is stale, misleading, or downstream-only

Cap at 30 if the answer blames only the renderer, gives grep-level references, or repeats prompt terminology without symbol-level evidence.

Scoring:

- 15 points: correct causal path from normalization into aggregation
- 10 points: correct adapter, aggregation, renderer-ruled-out, and artifact references
- 10 points: explanation of why competing evidence is downstream-only

### Phase 3: Orchestration Gate

Award orchestration credit only if the final brief clearly integrates findings from at least two non-overlapping explorer questions. A single linear read, even if partially correct, cannot score above 30.

Scoring:

- 15 points: synthesis of at least two non-overlapping explorer findings
- 10 points: bounded remediation quality

## Hidden Checks

- hidden fixture confirms the renderer is a downstream symptom rather than the primary cause
- hidden scoring script checks for one adapter symbol, one aggregation symbol, and one ruled-out renderer symbol
- hidden artifact set includes at least one contradictory clue that must be explicitly disambiguated

## Red-Team Defenses

- no substantial credit for naming the most obvious suspect without disproof work
- no substantial credit for broad rewrite plans that skip the bounded remediation requirement
- no orchestration credit for answers that could have been written from the spec alone

## Score Caps

- Missing any required artifact: max 20
- No symbol-level evidence: max 30
- No contradictory-artifact disproof: max 35
- No multi-path synthesis: max 30
- Speculative code rewrite plan instead of bounded remediation: max 40

## Hardness Judgment

- Current judgment: under 30 likely for a naive `gpt-5.4` / `high` solver
- Why: the contract requires falsifiable evidence, symbol-level tracing, and multi-path synthesis before substantial credit
