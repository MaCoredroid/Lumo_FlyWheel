# `release-manifest-v2-modernization` Task Spec

## Task Prompt
Modernize the `shipit` service release path from a legacy release manifest and hand-wired GitHub Actions workflow to a reusable workflow plus manifest-v2 contract. Keep the repo-local release dry-run and staging deploy smoke behavior correct. Update the Codex config and operator release note only where they match the real wired release path. Preserve unrelated in-progress edits in release notes and env override files.

## Workspace Bundle
- `.github/workflows/release.yml`: legacy workflow with stale inline steps.
- `.github/workflows/reusable_release.yml`: target reusable workflow contract.
- `release/manifest.toml`: legacy manifest shape.
- `release/manifest.v2.toml`: partially authored target manifest with wrong field mapping.
- `scripts/run_ci.py`: repo-local dry-run release driver.
- `deploy/check_release.py`: deploy smoke verifier.
- `.codex/config.toml`: still points at legacy release helpers.
- `docs/releases/staging_rollout.md`: partially edited release note in dirty-workspace variants.
- `config/env/staging.toml`: env override file with unrelated local edits.
- `logs/dry_run_green.log`, `logs/staging_smoke_fail.log`: evidence of the second-order failure.
- `tests/test_manifest_contract.py`, `tests/test_release_driver.py`: visible tests.

## Seeded Modernization Breakage
- The repo is mid-cutover to a reusable workflow and manifest-v2 contract.
- `release.yml` and `manifest.v2.toml` disagree on input names, environment labels, and artifact outputs.
- `scripts/run_ci.py` can go green on the dry-run path while staging smoke still fails.
- `.codex/config.toml` and the release note still reference legacy release helpers.
- Dirty-workspace variants include local note and env-override edits that must survive the modernization.

## Required Surfaces
- `shell`
- `apply_patch`
- `terminal_tests`
- YAML and TOML editing
- docs update
- release smoke inspection

## Expected Deliverables
- Workflow patch aligning the repo to the reusable release contract.
- Manifest-v2 patch matching the workflow inputs, outputs, and environment wiring.
- Targeted `.codex/config.toml` and release-note updates matching the actual release path.
- No bypass of the deploy smoke path and no destructive rewrite of unrelated local edits.

## Grader Contract
- Phase 2 visible checks:
  - `pytest -q tests/test_manifest_contract.py tests/test_release_driver.py`
  - `python scripts/run_ci.py --mode release-dry-run`
- Hidden checks:
  - `python deploy/check_release.py --env staging` succeeds only if workflow, manifest, and environment wiring agree.
  - Reusable workflow contract passes required `workflow_call` inputs, secrets, and outputs.
  - Manifest-v2 fields align with both the dry-run driver and staging smoke verifier.
  - `.codex/config.toml` references the same release path the workflow uses.
  - Dirty-workspace note and env override edits survive targeted changes.
- Trusted final-state checks:
  - The deploy smoke step was not stubbed, bypassed, or turned into a no-op.
  - Legacy artifact names or environment aliases are no longer live on the release path.
  - Docs describe the actual staging rollout path, not a prompt-invented one.
- Scoring intent:
  - Workflow syntax or docs-only progress is capped at `0.20`.
  - Scores above `0.30` require staging verification, not just dry-run green.
  - Scores above `0.50` require dirty-workspace preservation and non-bypassed smoke behavior.

## Red-Team Traps
- Swap to a reusable workflow but omit one required secret or output mapping.
- Convert manifest syntax while keeping the wrong environment or artifact name.
- Make smoke green by bypassing the verification step.
- Update docs and config while workflow and manifest remain inconsistent.
- Overwrite partially edited release notes or env override files.
- Silence deprecation warnings without fixing staging continuity.

## Quality Gate
- GPT-5.4/high adversarial probe judgment before hardening: `under_30_for_naive_solver = maybe`.
- Probe found an easy 30+ path if the task only rewards a syntactically valid workflow/manifest migration plus prose artifacts.
- Hardening applied in this spec:
  - Bound scores above `0.30` to staging smoke, not just dry-run green.
  - Added trusted checks for real `workflow_call` plumbing, manifest-field alignment, and non-bypassed smoke behavior.
  - Added dirty-workspace preservation as a score separator.
  - Treated docs/config as dependent artifacts instead of primary scoring surfaces.
- Final quality judgment after hardening: `under_30_for_naive_solver = yes`.
- Rationale: a naive solver can still make the workflow parse, but without wiring manifest, workflow, staging smoke, and preserved local edits together it should stay under the 30-point band.
