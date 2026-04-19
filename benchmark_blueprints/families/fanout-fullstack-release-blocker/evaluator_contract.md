# Evaluator Contract: `fanout-fullstack-release-blocker`

## Purpose

Score a parallel cross-surface repair where the solver must align backend schema, frontend request flow, and operator docs, then prove the live request path is coherent.

## Required Submission Artifacts

- integrated code patch
- rollout note update
- integration report
- proof artifact under `artifacts/screenshots/` or `artifacts/dom/`

## 100-Point Breakdown

- 15 points: backend schema and request parsing alignment
- 15 points: frontend form and request-client alignment
- 10 points: operator-doc correctness
- 30 points: live request-path correctness
- 15 points: proof artifact quality
- 15 points: integrity and anti-tamper compliance

## Trusted Evaluation Phases

### Phase 1: Surface Coverage Gate

Cap at 25 if the submission changes only one of:

- backend schema or request parsing
- frontend form or request client
- operator docs

Scoring:

- 15 points: backend alignment
- 15 points: frontend alignment
- 10 points: docs alignment

### Phase 2: Live Request-Path Gate

Award core correctness only if hidden evaluation confirms:

- seeded admin form submission sends `human_review_required`
- the intercepted payload does not contain `manual_review`
- the persisted or echoed record reflects the same renamed field

Any backend-only alias shim that leaves the live request path stale caps at 30.

Scoring:

- 20 points: intercepted request shows `human_review_required`
- 10 points: persisted or echoed record matches the renamed field

### Phase 3: Artifact Integrity Gate

Award proof credit only if the artifact:

- lives at the required fixed path
- corresponds to the seeded release flow
- shows either request-payload evidence or server-echo evidence tied to the rename

Generic screenshots, DOM dumps, or relabeled static captures do not earn proof credit.

Scoring:

- 10 points: artifact is at the right path and tied to the seeded flow
- 5 points: artifact proves runtime request or server-echo evidence

## Hidden Checks

- grep or AST check rejects runtime `manual_review` references outside explicitly allowed legacy coverage
- integrity check rejects weakened visible tests and tampered seeded fixtures
- doc check requires mention of retired token, new token, and rollout verification order

## Red-Team Defenses

- alias-only backend fixes cap low
- frontend relabeling without request-client repair cap low
- canned screenshot theater earns no meaningful proof credit

## Score Caps

- Backend-only or frontend-only fix: max 25
- Alias shim that leaves stale live payload: max 30
- Missing or generic proof artifact: max 35
- Tampered tests, fixtures, or stale-token runtime references: max 20

## Hardness Judgment

- Current judgment: under 30 likely for a naive `gpt-5.4` / `high` solver
- Why: the evaluator requires end-to-end runtime evidence, not partial string replacement
