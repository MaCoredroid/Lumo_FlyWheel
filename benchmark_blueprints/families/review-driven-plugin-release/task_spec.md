# Review-Driven Plugin Release

- `task_id`: `t9_review_driven_plugin_release_drive_brief`
- `target_family_id`: `review-driven-plugin-release`

## Task Prompt
Address the actionable review feedback on the `drive-brief` plugin release candidate. The visible test suite is not sufficient evidence of correctness. The PR still has unresolved review comments about backward-compatible manifest fields, one screenshot-backed settings regression, and missing release evidence. Land the smallest defensible fix set and produce release-ready artifacts. Do not dismiss the review comments without code or artifact changes, and do not regenerate screenshots from modified baselines unless the underlying UI or manifest contract is actually fixed.

## Workspace Bundle
- Plugin repo with recorded review artifacts and release checklist files.
- Key paths:
  - `.codex-plugin/plugin.json`
  - `drive_brief/settings.py`
  - `drive_brief/ui/settings_panel.tsx`
  - `review/unresolved_threads.json`
  - `review/pr_summary.md`
  - `artifacts/screenshots/settings-panel-before.png`
  - `artifacts/screenshots/settings-panel-expected.png`
  - `docs/release_checklist.md`
  - `docs/release_notes.md`
  - `tests/test_manifest_backward_compat.py`
  - `tests/test_settings_panel.py`
  - `tests/test_release_checklist.py`
- Primary local commands:
  - `pytest -q`
  - `python scripts/render_release_evidence.py`

## Seeded Integration Or Plugin Drift
- The branch dropped a deprecated manifest field that marketplace consumers still require for one release cycle.
- The settings panel visually hides the connector fallback toggle when one optional field is unset; the screenshot evidence in review calls this out.
- Release notes mention the new toggle, but the checklist still claims no UI changes and the evidence renderer omits the screenshot pair.
- Exactly two unresolved review threads and one screenshot note are actionable; one resolved thread in `review/unresolved_threads.json` is a grading-negative red herring if treated as actionable.
- Hidden checks expect the final state to preserve backward compatibility, surface the toggle correctly, and include coherent release evidence.

## Required Surfaces
- `review`
- `plugins_mcp`
- `shell`
- `apply_patch`
- `screenshots`

## Deliverables
- Narrow code patch fixing the actionable review issues.
- Updated release note or checklist artifacts.
- Review response text or status note grounded in the actual fixes.
- Passing manifest, settings, and release-evidence tests.

## Grader Contract
- Phase 1: read the unresolved review thread bundle and render release evidence from the checked-in script.
- Phase 2: run `pytest -q` and require manifest, settings, and release-checklist tests to pass.
- Phase 3 trusted checks:
  - The deprecated manifest field is restored or compatibility-shimmed exactly as the review requires; adding it only in tests or fixtures fails.
  - The settings toggle is visible and labeled in two rendered states, including the optional-field-unset state called out in review.
  - Screenshot baseline filenames and immutable expected artifact names are checksum-guarded; rebaselining without a corresponding UI code fix fails.
  - Release notes, checklist, and generated evidence agree about what changed and include the before and expected screenshot pair.
  - The submitted review reply addresses only the two actionable unresolved thread ids plus the screenshot note, and references the actual changed surface for each.
  - Broad manifest rewrites, closing threads without remediation, or evidence-only syncs without the underlying code fix fail grading.
  - Visible tests are worth at most 10/100 without the trusted review, screenshot, and evidence checks.

## Red-Team Traps
- Mark all review threads resolved and add a generic reply.
- Rebaseline the screenshot to match the broken UI.
- Restore the deprecated field only in tests or fixtures.
- Over-edit the manifest and create fresh backward-compat regressions.
- Update release notes only while leaving generated evidence stale.

## Quality Gate
- Intended naive-solver failure mode: fix the obvious manifest regression, ignore the screenshot-backed UI issue and release-evidence mismatch, then still look superficially complete.
- Hardening built into the task:
  - Review artifacts distinguish actionable from resolved noise.
  - Hidden grading cross-checks code, two-state screenshot evidence, release docs, and generated evidence together.
  - Manifest compatibility and UI evidence must both be correct to pass.
  - Review replies are graded against exact actionable thread ids and changed surfaces.
- GPT-5.4/high probe result: pre-hardening estimate `45/100`; easy path was manifest-only repair plus lightweight docs or evidence edits while skating past the screenshot-backed UI issue.
- Hardening applied after probe:
  - Removed the implication that green visible tests are meaningful.
  - Fixed the review-state ambiguity by naming the exact actionable count and making the resolved thread grading-negative.
  - Added checksum-guarded screenshot evidence, two-state UI verification, and exact review-reply expectations.
  - Capped visible-test credit at 10/100.
- Final under-30 judgment for a naive GPT-5.4/high solver: `Yes`, now looks under 30 because the cheap manifest-only or docs-only path can no longer accumulate enough credit.
- Observed GPT-5.4/high benchmark run: `20/100` on the visible bundle, which lands in the intended `15-25` target band.
