# Evaluator Contract: Annotated Mockup Component Port

## Scoring Breakdown

Total raw score: 100 points.

1. Artifact triage and root-cause reasoning: 15
   - Correctly separates authoritative signals from stale or noisy notes: 5
   - Identifies a shared-component-level repair surface instead of a story-only or preview-only surface: 5
   - Recognizes that compatibility must be inferred from the integration or history bundle, not from one note alone: 5
2. Concrete shared-implementation repair: 20
   - Touches the shared component or real helper path used by consumers: 10
   - Repair plausibly addresses multimodal fidelity plus layout behavior rather than a cosmetic one-state tweak: 10
3. Multimodal fidelity across states and presentation modes: 20
   - Core annotated states align with the mockup family: 8
   - Width-sensitive behavior holds for long labels or narrow layouts: 6
   - Alternate density, theme, or equivalent presentation mode remains correct: 6
4. Downstream compatibility preservation: 15
   - Keeps the still-live consumer contract intact: 8
   - Rejects stale-note-driven deletions or prop regressions: 7
5. Verification quality: 15
   - Includes explicit automated coverage for a narrow-width or long-label case: 7
   - Includes automated coverage for a density, theme, or equivalent alternate presentation path: 8
6. Stories, docs, and evidence alignment: 15
   - Stories or examples reflect the repaired implementation: 5
   - Docs or usage contract reflects the true supported behavior: 5
   - Evidence is tied to the real shared implementation rather than a local override: 5

## Global Caps

- No concrete shared-component or shared-helper repair path: total capped at 25.
- No automated verification for both the narrow-width case and the alternate-presentation case: total capped at 30.
- No credible downstream compatibility reasoning or evidence: total capped at 25.
- Response relies only on task-text paraphrase without artifact triage or implementation-level specificity: total capped at 20.
- Story-only, docs-only, snapshot-only, or preview-only patching: total capped at 10.

## Automatic Zero Conditions

- The proposal implements a preview-only or story-only replacement instead of repairing the shared component path.
- The proposal fabricates verification or claims screenshot fidelity without any implementation or test strategy.
- The proposal removes the live compatibility behavior entirely in order to satisfy a stale artifact note.

## Hidden Evaluation Signals

- Hidden checks compare the repaired shared implementation against annotated states across multiple widths and at least one alternate presentation mode.
- Hidden checks exercise the consumer path implied by the integration or history artifacts and fail if the solver drops undocumented-but-live behavior.
- Hidden checks reject one-width hacks, label-length assumptions, and visual patches that collapse outside the easiest screenshot.
- Hidden checks ensure stories and docs describe the actual repaired component rather than local overrides or stale prop semantics.

## Shortcut Failure Modes

- Patch only Storybook stories, docs, screenshots, or baselines.
- Match the mockup in one obvious state while missing another required state family.
- Apply a narrow viewport fix that fails under longer labels or denser presentation.
- Trust the loudest note instead of reconciling notes with the integration surface.
- Describe a good repair plan but provide no trustworthy verification path.

## Scoring Notes

- A family-local artifact-only attempt should usually land between `15` and `25` because it cannot produce real code diffs, runtime evidence, or integration proof.
- The benchmark is correctly hardened when a naive GPT-5.4/high attempt from this directory remains limited by the shared-implementation, compatibility, and verification caps together.

## Final Hardness Judgment

- Target naive GPT-5.4/high score: approximately `20/100`.
- Current judgment: meaningful and sufficiently hardened if the latest `benchmark_run.md` records a family-local attempt in the `15-25` band and the limiting factors are lack of real repo access plus lack of trustworthy verification evidence rather than solver confusion about the task goal.
