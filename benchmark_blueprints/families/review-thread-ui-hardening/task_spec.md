# Review Thread UI Hardening

## Task Prompt
You are on a front-end feature branch with exported PR review threads. The exported review bundle contains multiple icon-like controls and multiple mobile screenshots, but only one combination corresponds to the active unresolved feedback. One resolved thread is mixed into the export and should not be reworked. Investigate the actionable UI feedback from the artifacts, keep the branch coherent, and prepare concise review replies. Do not assume the affected viewport, component, or control from filenames alone, and do not revert or overwrite unrelated branch work.

## Workspace Bundle
- `repo/`: React app or monorepo package on a feature branch, timeline components, snapshot or visual-test config, and a minimal review-reply template.
- `artifacts/review/`: exported review threads in Markdown or JSON, inline screenshot references, resolution metadata, and component names that do not exactly match source names.
- `artifacts/screenshots/`: reviewer screenshots for the active mobile issue plus at least one screenshot tied to a resolved but visually similar issue.
- `artifacts/logs/`: CI snapshot output and one irrelevant lint warning.

## Seeded Visual Or Multimodal Breakage
- `surface_review_triage`: actionable and resolved comments are mixed together, so the solver must respond only to the live issues.
- `surface_mobile_regression`: a mobile layout regression is shown in reviewer evidence, but the exact implementation fault is not disclosed in the spec.
- `surface_accessibility_gap`: an accessibility regression is called out in reviewer feedback, but multiple plausible controls exist in the branch.
- `surface_config_gap`: test coverage is incomplete for the regressed presentation, and only one config path is active in CI.
- `noise`: the irrelevant lint warning and resolved thread create false work.

## Required Surfaces
- Review-thread interpretation.
- Screenshot inspection and repo patching.
- Snapshot or visual-test configuration updates.
- Short review-reply drafting.

## Deliverables
- Code patch fixing the overflow and accessibility issues.
- Visual-test or snapshot-config update that covers the reopened viewport.
- Concise review replies for the unresolved comments, citing the exact thread ids addressed.
- Updated screenshot or test evidence showing the fix for the exact reopened viewport.

## Grader Contract
- Visible checks include the existing branch tests and one reopened mobile viewport snapshot.
- Hidden checks ensure the icon-only control exposes an accessible name and that the mobile card wraps without clipping content.
- Hidden checks verify the solver updated the correct unresolved review artifacts and did not churn resolved-thread text unnecessarily.
- Hidden checks fail if accessible-name changes touch unaffected controls.
- Hidden checks fail if the viewport added to config is not the exact reopened one from the review export.
- Hidden checks fail if the overflow is fixed via clipping, truncation, or screenshot-only updates.
- Hidden checks reject solutions that alter only snapshots or only review replies without code and config fixes.
- Final grading requires review replies that map to the actionable threads.

## Red-Team Traps
- Reply to every thread, including resolved comments, without fixing code.
- Patch evidence or screenshot baselines without fixing runtime behavior.
- Apply blanket accessibility-label changes to every icon control instead of identifying the reviewed surface.
- Add coverage for a plausible but wrong mobile viewport.
- Infer the target route from filenames alone instead of the exported review state.

## Quality Gate
- Naive-solver target: under 30/100 unless the solver can read review state, use screenshot evidence, and coordinate code plus config plus reply artifacts.
- Adversarial GPT-5.4/high probe estimated a naive score around 58/100 before hardening because the original spec telegraphed the overflow, accessibility, and config fixes too directly.
- Hardening applied after probe: removed implementation-level diagnoses, made the artifact bundle more ambiguous, required exact thread-id and viewport mapping, and added hidden checks for overbroad accessibility edits and reply churn on resolved threads.
- Current judgment: under 30/100 for a naive GPT-5.4/high solver if the review-state, control-targeting, and viewport-mapping checks are enforced together.
