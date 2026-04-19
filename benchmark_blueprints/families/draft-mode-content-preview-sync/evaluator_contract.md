# Evaluator Contract: Draft Mode Content Preview Sync

## Scoring Breakdown

Total raw score: 100 points.

1. Root-cause investigation quality: 15
   - Identifies preview-only freshness/status mismatch rather than generic content issue: 5
   - Distinguishes content-body freshness from status-chip derivation: 5
   - Uses preview observations instead of spec paraphrase alone: 5
2. Concrete workspace edits: 20
   - Names likely site, adapter, config, and test targets with file-level specificity: 10
   - Proposes aligned changes for both content selection and status derivation paths: 10
3. Preview/browser evidence: 20
   - Draft article body reflects latest draft revision: 8
   - Status chip matches draft or unpublished state: 6
   - Evidence ties body and status to the same draft revision identity: 6
4. Regression and anti-shortcut handling: 15
   - Preserves published-mode behavior: 5
   - Rejects global cache disablement and hardcoded draft content: 5
   - Rejects forcing every previewed article into draft state: 5
5. Deliverables and docs: 10
   - Operator doc describes verification and cache expectations: 5
   - Before/after evidence bundle accounted for: 5
6. Family-specific correctness invariants: 20
   - Draft content freshness is correct: 5
   - Draft status is correct: 5
   - Published-only article in preview mode remains semantically consistent: 5
   - Body content and status derive from the same revision identity: 5

## Global Caps

- No concrete file-level patch or diff-backed change set: total capped at 45.
- No trusted preview-browser or fixture-linked evidence: total capped at 25.
- Response relies only on task text and gives no cross-article or revision-level observations: total capped at 20.
- Any banned shortcut, including global cache disablement, hardcoded draft text, or forced draft-state rendering for all previewed articles: total capped at 10.

## Automatic Zero Conditions

- The proposal disables caching globally.
- The proposal hardcodes the known article body or title.
- The proposal always renders a draft chip whenever `preview=1`.

## Scoring Notes

- A family-local artifact-only attempt should usually land between 15 and 25 because it cannot prove cross-article consistency or real preview freshness.
- The task is correctly hardened when a naive GPT-5.4/high child attempt from this directory stays near 20/100.

## Final Hardness Judgment

- Target naive GPT-5.4/high score: approximately 20/100.
- Current judgment: meaningful and sufficiently hardened if the latest `benchmark_run.md` score is within 15-25 and constrained by the cross-article evidence cap.
