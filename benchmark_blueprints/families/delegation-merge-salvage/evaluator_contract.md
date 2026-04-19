# Evaluator Contract: `delegation-merge-salvage`

## Purpose

Score a merge-salvage task where the solver must selectively recover value from overlapping worker artifacts instead of choosing a winner wholesale or rewriting the task from scratch.

## Required Submission Artifacts

- final integrated patch
- `artifacts/delegation/salvage_postmortem.md`
- `artifacts/delegation/verification.md`
- reviewer-facing note

## 100-Point Breakdown

- 25 points: salvage artifact discipline and hunk-level evidence
- 35 points: behavioral correctness across Markdown, JSON, and watchlist follow-up
- 20 points: selective merge quality and reviewability
- 10 points: verification evidence quality
- 10 points: reviewer-facing risk explanation

## Trusted Evaluation Phases

### Phase 1: Artifact Discipline Gate

Cap at 30 if the postmortem does not cite:

- one kept hunk from Worker A
- one rejected hunk from Worker A
- one kept hunk from Worker B
- one rejected hunk from Worker B

Generic salvage prose does not count.

Scoring:

- 15 points: Worker A kept and rejected evidence
- 10 points: Worker B kept and rejected evidence

### Phase 2: Correctness Gate

Award correctness credit only if hidden evaluation confirms:

- Markdown path works
- JSON contract is byte-for-byte preserved
- `--include-watchlist` behavior remains correct
- unrelated snapshot fixtures remain untouched

If JSON drifts or the snapshot fixture changes, cap at 20.

Scoring:

- 10 points: Markdown path correctness
- 15 points: byte-for-byte JSON preservation
- 10 points: watchlist follow-up correctness

### Phase 3: Selective Merge Gate

Award salvage credit only if the final diff is selective and reviewable. Applying either worker patch wholesale, or rewriting contested files without hunk-level justification, caps at 30.

Scoring:

- 20 points: selective merge quality and reviewability
- 10 points: verification evidence quality
- 10 points: reviewer-facing regression-risk explanation

## Hidden Checks

- fixture comparison checks JSON output byte-for-byte
- follow-up case exercises the watchlist path
- patch audit checks whether one worker patch was applied wholesale

## Red-Team Defenses

- “pick Worker A and narrate salvage” caps low
- “pick Worker B because it feels safer” caps low
- artifact-ignoring rewrites cap low

## Score Caps

- Generic salvage postmortem with no hunk evidence: max 30
- Wholesale Worker A or Worker B patch: max 30
- JSON drift or unrelated fixture churn: max 20
- Missing watchlist follow-up preservation: max 25

## Hardness Judgment

- Current judgment: under 30 likely for a naive `gpt-5.4` / `high` solver
- Why: meaningful credit depends on hunk-specific salvage plus hidden behavioral preservation
