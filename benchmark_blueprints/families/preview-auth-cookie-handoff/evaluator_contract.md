# Evaluator Contract: Preview Auth Cookie Handoff

## Scoring Breakdown

Total raw score: 100 points.

1. Root-cause investigation quality: 15
   - Identifies preview-only session failure rather than generic auth outage: 5
   - Distinguishes redirect validation risk from session persistence risk: 5
   - Uses browser/runtime observations instead of spec paraphrase alone: 5
2. Concrete workspace edits: 20
   - Names likely code and config targets with file-level specificity: 10
   - Proposes changes to both app/auth logic and preview config where needed: 10
3. Preview/browser evidence: 20
   - Fresh-profile login flow evidence: 8
   - Reload persistence and cookie-clear regression evidence: 6
   - Network/cookie attribute evidence tied to preview host: 6
4. Regression and anti-shortcut handling: 15
   - Preserves auth guard for unauthenticated `/cases`: 5
   - Rejects preview-only auth bypasses or wildcard domain shortcuts: 5
   - Keeps unrelated origins rejected: 5
5. Deliverables and docs: 10
   - Operator note with exact preview verification steps: 5
   - Before/after evidence bundle accounted for: 5
6. Family-specific correctness invariants: 20
   - Authenticated preview reaches `/cases`: 5
   - Session persists on reload: 5
   - Cookie is exact-host or safe host-only scoped for preview: 5
   - Redirect validation accepts configured preview origin and rejects unrelated origins: 5

## Global Caps

- No concrete file-level patch or diff-backed change set: total capped at 45.
- No trusted preview-browser runtime evidence: total capped at 25.
- Response relies only on spec clues and provides no disambiguating runtime/workspace observations: total capped at 20.
- Any banned shortcut in the final proposal, including auth bypass, middleware disablement, or permissive wildcard cookie/origin policy: total capped at 10.

## Automatic Zero Conditions

- The proposed fix disables auth for preview hosts.
- The proposal accepts arbitrary callback origins.
- The proposal stores success state only in `localStorage` without a valid authenticated request flow.

## Scoring Notes

- A family-local artifact-only attempt from this directory should usually land between 15 and 25 points because it cannot supply trusted runtime proof or real workspace edits.
- A benchmark run should be considered correctly hardened when a naive GPT-5.4/high child attempt from this family directory scores around 20/100.

## Final Hardness Judgment

- Target naive GPT-5.4/high score: approximately 20/100.
- Current judgment: meaningful and sufficiently hardened if the latest `benchmark_run.md` score is within 15-25 and the runtime-evidence cap is the dominant limiter.
