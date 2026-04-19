# Benchmark Run: `fanout-fullstack-release-blocker`

## Run Metadata

- Run type: child-subagent benchmark attempt
- Model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da330-e9d7-73c0-8f0b-fccf0567984a`
- Context restriction: family-local artifacts only under this family directory

## Attempt Summary

The solver produced a strong narrative submission: backend summary, frontend summary, rollout-note update summary, integration report, and a proof-artifact description. The response mirrored the intended cross-surface migration shape well, but it did not supply an actual patch, intercepted request evidence, or a real proof artifact from the benchmark workspace.

Most important failure mode:

- no actual code diff or file-level patch manifest
- no concrete runtime payload excerpt from the benchmark workspace
- proof artifact was only described, not produced

## Scoring Against `evaluator_contract.md`

### Backend Schema And Request Parsing Alignment: 5 / 15

- high-level backend intent was correct, but no concrete patch or emitted schema evidence was provided

### Frontend Form And Request-Client Alignment: 5 / 15

- high-level frontend intent was correct, but no concrete request-client diff or runtime proof was provided

### Operator-Doc Correctness: 7 / 10

- docs summary named the retired token, replacement token, and rollout order

### Live Request-Path Correctness: 0 / 30

- no intercepted request payload
- no persisted or echoed record
- no actual runtime benchmark artifact

### Proof Artifact Quality: 0 / 15

- only a speculative artifact description was supplied

### Integrity And Anti-Tamper Compliance: 3 / 15

- the submission said not to weaken tests or fixtures, but it did not demonstrate integrity against the actual workspace

## Caps Applied

- Effective cap triggered by evaluator intent: no exact file-level patch manifest and no concrete runtime payload or server-echo fields from the benchmark workspace

## Final Score

- Final score: `20 / 100`

## Final Judgment

- Target judgment: acceptable
- Reason: a polished migration plan alone cannot reach a meaningful score without live request-path evidence

