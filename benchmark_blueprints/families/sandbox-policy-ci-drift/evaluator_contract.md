
# Evaluator Contract

## Scope
- Family: `sandbox-policy-ci-drift`
- Task: `cnb55-core-sandbox-policy-ci-drift-local-ci-contract`
- Runtime class: local repo repair with parser, config loader, workflow preview generator, helper-driven CI runner, workflow YAML, and release note

## Evaluation Intent
Measure whether the solver can align parser semantics, local CI, workflow
preview output, workflow YAML, canonical config, and docs after a policy
enum rename without dropping the one required compatibility alias.

## Required Submission Artifacts
- Parser and config loader patch
- Workflow preview patch
- CI workflow correction
- Canonical `codex/config.toml`
- Release note update

## Scoring Model
- `100` total points:
  - `25`: visible `make ci`
  - `20`: deprecated-alias parser + preview canonicalization
  - `15`: workflow YAML consistency
  - `15`: config and release note alignment
  - `10`: helper anti-shortcut integrity
  - `15`: hidden verifier pass

## Score Caps
- Cap at `10/100` for docs-only or workflow-only patches.
- Cap at `20/100` for rename-everything patches that remove deprecated compatibility.
- Cap at `20/100` for helper-path fixture normalization shortcuts.
- Cap at `20/100` in `V3+` when the release note does not explicitly reject the abandoned visible-fixture helper shortcut as not being a real config fix.
- Cap at `20/100` in `V4+` when the release note ignores the preview-consumer/operator contract introduced by `release_context/preview-consumer-contract.md`.
- Cap at `10/100` in `V5` when the release note fixes the code path but does not acknowledge the rollback and why `workspace-write` input compatibility was preserved.
- Cap at `25/100` if preview output still emits deprecated names.
- Cap at `30/100` if workflow drift remains.
- Cap at `50/100` if code is broadly correct but config/doc operator contract remains stale.

## Visible Gates
- `make ci`

## Hidden Gates
- Deprecated `workspace-write` still parses locally.
- Preview output is canonical-only for both canonical and deprecated inputs.
- Workflow file uses one consistent canonical token family.
- `scripts/run_ci.py` does not normalize only the visible fixture.
- Release note documents compatibility plus canonical output.
- `V3+`: when dirty-state evidence points to the abandoned helper patch, the release note explicitly says the visible-fixture shortcut was abandoned and did not fix real configs.
- `V4+`: release note acknowledges that preview JSON is consumed directly by operators and stays canonical-only for that downstream surface.
- `V5`: release note explicitly names the rollback and the compatibility reason for preserving deprecated `workspace-write` on input.

## Integrity And Shortcut Checks
- Reject rename-everything patches that remove deprecated compatibility.
- Reject fixes isolated to `.github/workflows/ci.yml`.
- Reject preview emitters that leak deprecated names.
- Reject helper-path fixture normalization.
- Reject weakening or mutating visible tests.

## Variant Hardness Notes
- `V1`: one clean rename drift; compatibility still matters.
- `V2`: stale archive noise should not leak into active surfaces.
- `V3`: abandoned helper-path shortcut must be rejected both in code and in the operator note, so a generic release note no longer clears the dirty-state rung.
- `V4`: release-context consumers make operator-facing preview language part of correctness, not polish.
- `V5`: incident recovery forbids dropping the deprecated alias and requires rollback-aware release-note language.
