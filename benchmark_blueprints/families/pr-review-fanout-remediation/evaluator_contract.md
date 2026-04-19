# Evaluator Contract: `pr-review-fanout-remediation`

## Purpose

Score a review-thread orchestration task where the solver must classify thread state correctly before making code changes, then respond only to actionable or duplicate threads with evidence-backed replies.

## Required Submission Artifacts

- code and test patch
- docs update
- `artifacts/review/thread_triage.json`
- reviewer-facing reply bundle
- remediation report

## 100-Point Breakdown

- 35 points: thread-state triage accuracy
- 30 points: implementation of actionable review items
- 15 points: reviewer-facing reply quality
- 10 points: docs and regression-test completeness
- 10 points: integrity and anti-tamper compliance

## Trusted Evaluation Phases

### Phase 1: Triage Gate

No code credit if `thread_triage.json` misclassifies thread state. The evaluator checks:

- `actionable`
- `obsolete`
- `duplicate`
- correct `duplicate_of` target when present

Scoring:

- 25 points: correct per-thread classification
- 10 points: correct `duplicate_of` mapping and rationale quality

### Phase 2: Implementation Gate

Award implementation credit only if the patch:

- fixes the reopened preview fallback regression
- adds or updates the requested regression test
- updates the docs example that still reflects old behavior

Acting on the obsolete hunk or fixing only the happy path caps at 30.

Scoring:

- 20 points: reopened preview fallback regression fixed
- 10 points: requested regression test added or updated

### Phase 3: Reply Gate

Award review-response credit only if replies:

- map to the correct thread ids
- distinguish actionable work from duplicate closure
- do not claim the obsolete suggestion was applied

Generic “all comments addressed” language earns no meaningful reply credit.

Scoring:

- 10 points: replies mapped to correct thread ids and dispositions
- 5 points: remediation report explains what was intentionally not changed

## Hidden Checks

- hidden replay of thread state ensures the stale digest is not sufficient
- integrity checks reject tampered review-export files
- follow-up case checks that the reopened regression is fixed beyond the simplest path

## Red-Team Defenses

- trusting the stale digest without JSON-state interpretation caps low
- fixing whichever thread is easiest to grep caps low
- generic thread-id-shaped replies with no state discipline cap low

## Score Caps

- Wrong triage state for any thread: max 20
- No valid `duplicate_of` mapping when required: max 25
- Fixes obsolete comment or only happy path: max 30
- Generic replies with no state discipline: max 35
- Tampered review-export artifacts: max 10

## Hardness Judgment

- Current judgment: under 30 likely for a naive `gpt-5.4` / `high` solver
- Why: thread-state accuracy is a hard gate before code credit
