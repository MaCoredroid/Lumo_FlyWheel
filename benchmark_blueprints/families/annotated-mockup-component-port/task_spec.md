# Annotated Mockup Component Port

- `family_id`: `annotated-mockup-component-port`
- `task_id`: `annotated-mockup-component-port/shared-comparison-card-port`
- `scenario_type`: `frontend_multimodal_component_repair`

## Task Prompt

You are dropped into a React component-library workspace with a shared comparison-style card that has drifted from the latest annotated design package. The drift is not isolated to one screenshot and the design notes are not fully trustworthy. Repair the shared implementation that downstream consumers actually import, preserve the still-live compatibility path implied by the bundle artifacts, and leave the workspace with code-backed verification that would fail if the fix only worked at one width, one label length, or one presentation mode.

The task is complete only when the real shared component path is repaired, the downstream compatibility contract still holds, stories and docs match the repaired implementation, and trusted automated checks pass for both width-sensitive behavior and an alternate presentation path such as density or theme.

## Runtime Bundle Shape

The official runtime instance is offline and replayable. The family directory you are reading now is only the public benchmark blueprint. The held-out runtime bundle used for real scoring supplies:

- `repo/`
  - React component-library source, shared tokens or layout helpers, Storybook stories, unit tests, visual or DOM-geometry tests, docs, and one downstream integration surface that still imports the shared component.
- `artifacts/mockups/`
  - Annotated PNG or PDF mockups covering multiple states, at least one subtle layout or typography defect, and callouts that are incomplete rather than answer-key explicit.
- `artifacts/notes/`
  - Mixed-authority design notes, including at least one stale recommendation and one incomplete but directionally useful note.
- `artifacts/screenshots/`
  - Current Storybook and integration renders, including distractors that do not correspond to the highest-value defect.
- `artifacts/history/`
  - Review, changelog, or issue fragments that make one compatibility requirement discoverable without stating it cleanly.
- hidden verifier assets outside the family directory
  - `verifiers/annotated-mockup-component-port/verify.sh`
  - milestone inspection scripts
  - held-out hidden test slices
  - verifier data such as screenshot expectations, DOM-geometry assertions, mutation checks, and exploit fixtures

## Seeded Breakage Surfaces

- `shared_component_drift`
  - The real consumer path imports a shared component whose rendered structure or token wiring no longer matches the annotated mockup family across all required states.
- `responsive_layout_instability`
  - One width-sensitive failure lives in shared layout or token logic rather than only in a local story fixture.
- `compatibility_contract_drift`
  - A still-live consumer behavior is under-documented or partially contradicted by the notes and must be inferred from repeated evidence.
- `coverage_illusion`
  - Existing stories, tests, or docs make the surface look covered while leaving at least one important state family under-specified.
- `artifact_noise`
  - Some screenshots and notes point at symptoms or stale recommendations rather than the canonical fix.

## Required Solver Surfaces

- multimodal inspection of the provided mockups and screenshots
- code search and patching in the real shared component path plus true shared dependencies
- story or example updates only after the implementation is coherent
- trusted automated verification
- compatibility reasoning grounded in the downstream integration surface

## Required Deliverables

- code changes in the shared component path and any real shared helper, token, or layout dependency it actually relies on
- story or example coverage for the repaired state family, including the previously under-specified presentation path
- docs or usage-contract updates that describe the repaired behavior instead of stale note language
- automated checks that prove:
  - one narrow-width, long-label, or geometry-stress case, and
  - one density, theme, or equivalent alternate-presentation case
- verification evidence tied to the shared implementation rather than a preview-only substitute

## Official Evaluation Path

Official scoring is code-backed and hidden-test-backed only. Prose quality can clarify reasoning, but it does not replace verifier results.

The trusted evaluation path mirrors the `scenario_families` pattern:

1. `Phase 1: integrity and anti-shortcut inspection`
   - diff inspection confirms the solver repaired the real shared component path
   - grader rejects preview-only forks, story-only patches, docs-only patches, snapshot-only fixes, and verifier tampering
2. `Phase 2: visible command slice`
   - runtime executes the authored visible verification command for the held-out repo
   - at minimum this includes the component test slice and the visible story or visual regression slice
3. `Phase 3: held-out milestone and hidden-test checks`
   - milestone scripts inspect code ownership, state coverage, compatibility preservation, and docs or story alignment
   - hidden tests stress widths, labels, density or theme modes, and downstream consumer behavior
   - hidden checks include DOM or geometry assertions and diff inspections, not only pixel snapshots

## Anti-Cheat Constraints

- Do not patch only stories, docs, screenshots, baselines, or preview-only components.
- Do not disable the live compatibility path to satisfy one loud artifact note.
- Do not gate layout on one exact label string, one exact width, or one screenshot-specific offset.
- Do not weaken, skip, or delete visible or hidden verification hooks.
- Do not introduce artifact-specific conditionals that detect the benchmark fixture instead of fixing the shared behavior.

## Quality Gate

- `target naive GPT-5.4/high score`: about `20/100`
- `acceptable observed band`: `15-25/100`

### Why This Family Is Meaningfully Hard

- Following `scenario_families/normalizer-api-migration/family.yaml`, official scoring is split into visible commands, milestone inspections, and held-out hidden tests rather than rubric-only judging.
- Following [SWE-bench Multimodal](https://arxiv.org/abs/2410.03859), the family targets visual JavaScript repair where strong issue-solving systems still struggle with image-grounded and cross-language software tasks.
- Following [Design2Code](https://arxiv.org/abs/2403.03163), the family emphasizes layout fidelity and visual-element recall, because front-end models often fail on exact structure and layout even when high-level intent is obvious.
- Following [FrontendBench](https://arxiv.org/abs/2506.13832), the family requires executable automatic evaluation with sandboxed tests instead of subjective visual grading.
- Following [OpenAI's SWE-bench Verified analysis](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/), the family keeps the decisive checks hidden and avoids public answer-key leakage, narrow implementation-name tests, and wide tests that demand unstated behavior.

### Hardening Applied In This Revision

- Replaced rubric-like task completion language with an explicit trusted verifier path.
- Shifted evaluation authority to held-out milestone scripts, hidden tests, and diff inspections.
- Made contamination resistance concrete by forbidding fixture-specific hacks and public-answer-key scoring.
- Tightened the task around shared-component ownership, downstream compatibility, and dual-mode verification so a screenshot-matching plan cannot score well.
