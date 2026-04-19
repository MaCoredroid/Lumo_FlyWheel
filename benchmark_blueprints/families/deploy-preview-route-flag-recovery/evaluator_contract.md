# Evaluator Contract: Deploy Preview Route Flag Recovery

## Scoring Breakdown

Total raw score: 100 points.

1. Root-cause investigation quality: 15
   - Identifies preview-only route/flag divergence rather than generic page failure: 5
   - Distinguishes route resolution from seeded-content availability: 5
   - Uses preview observations rather than spec paraphrase alone: 5
2. Concrete workspace edits: 20
   - Names likely route, flag, manifest, seed, and test targets with file-level specificity: 10
   - Proposes aligned changes across route, canonical flag key, and preview seed behavior: 10
3. Preview/browser evidence: 20
   - Preview URL resolves without 404: 6
   - Seeded content with known identity is visible: 8
   - Evidence distinguishes real seeded content from a shell wrapper: 6
4. Regression and anti-shortcut handling: 15
   - Rejects global flag enablement and hardcoded route rendering: 5
   - Rejects dual-key compatibility shims as the final state: 5
   - Preserves non-preview gating behavior: 5
5. Deliverables and docs: 10
   - Deploy note documents route, flag, and seed prerequisites: 5
   - Before/after evidence bundle accounted for: 5
6. Family-specific correctness invariants: 20
   - Canonical route segment is correct: 5
   - Canonical feature key is used consistently: 5
   - Preview seed provenance is correct: 5
   - Non-preview access remains gated: 5

## Global Caps

- No concrete file-level patch or diff-backed change set: total capped at 45.
- No trusted preview-browser or seeded-content evidence: total capped at 25.
- Response relies only on task text and gives no route/seed disambiguating observations: total capped at 20.
- Any banned shortcut, including global flag enablement, hardcoded route rendering, shell-only fixes, or dual-key final-state shims: total capped at 10.

## Automatic Zero Conditions

- The proposal turns the feature on globally.
- The proposal uses a catch-all or unconditional route render without respecting gating.
- The proposal replaces real preview data with canned static content.

## Scoring Notes

- A family-local artifact-only attempt should usually land between 15 and 25 because it cannot prove canonical-key consistency or seeded-content provenance from a real preview run.
- The task is correctly hardened when a naive GPT-5.4/high child attempt from this directory stays near 20/100.

## Final Hardness Judgment

- Target naive GPT-5.4/high score: approximately 20/100.
- Current judgment: meaningful and sufficiently hardened if the latest `benchmark_run.md` score is within 15-25 and constrained by the preview-evidence and canonical-key caps.
