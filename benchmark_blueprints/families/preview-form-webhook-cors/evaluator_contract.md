# Evaluator Contract: Preview Form Webhook CORS

## Scoring Breakdown

Total raw score: 100 points.

1. Root-cause investigation quality: 15
   - Identifies preview-browser request failure rather than generic form bug: 5
   - Distinguishes success-path and validation-path cross-origin behavior: 5
   - Uses browser/network observations instead of spec paraphrase alone: 5
2. Concrete workspace edits: 20
   - Names likely frontend, API, config, and test targets with file-level specificity: 10
   - Proposes aligned changes on both client routing and API policy paths: 10
3. Preview/browser evidence: 20
   - Valid submit succeeds end to end from preview: 8
   - Invalid submit still returns browser-visible validation errors: 6
   - Network evidence captures the actual preview API target and policy behavior: 6
4. Regression and anti-shortcut handling: 15
   - Rejects production fallback, fake frontend success, and wildcard CORS: 5
   - Rejects reflected arbitrary-origin policies: 5
   - Preserves validation behavior on error paths: 5
5. Deliverables and docs: 10
   - Deploy note with origin and verification expectations: 5
   - Before/after evidence bundle accounted for: 5
6. Family-specific correctness invariants: 20
   - Preview frontend targets the correct preview API host: 5
   - Preflight behavior is correct: 5
   - Success response path is correct: 5
   - Validation-error response path is correct: 5

## Global Caps

- No concrete file-level patch or diff-backed change set: total capped at 45.
- No trusted preview-browser/network evidence: total capped at 25.
- Response relies only on task text and gives no disambiguating network observations: total capped at 20.
- Any banned shortcut, including production fallback, frontend fake success, wildcard CORS, or reflected arbitrary origins: total capped at 10.

## Automatic Zero Conditions

- The proposal points preview traffic to production.
- The proposal disables validation to make submission appear successful.
- The proposal returns success only in the client without a real network round trip.

## Scoring Notes

- A family-local artifact-only attempt should usually land between 15 and 25 because it cannot supply trustworthy preview-host or browser-network proof.
- The task is correctly hardened when a naive GPT-5.4/high child attempt from this directory stays near 20/100.

## Final Hardness Judgment

- Target naive GPT-5.4/high score: approximately 20/100.
- Current judgment: meaningful and sufficiently hardened if the latest `benchmark_run.md` score is within 15-25 and constrained by the network-evidence cap.
