# Evaluator Contract

## Scope
- Family: `sandbox-policy-ci-drift`
- Task: `cnb55-core-sandbox-policy-ci-drift-local-ci-contract`
- Runtime class: agent CLI repo with parser, config loader, workflow preview generator, CI workflow, and release note

## Evaluation Intent
Measure whether the solver can align parser semantics, local CI, workflow preview output, config, and docs after a policy-enum rename. The evaluator should reward canonical output plus deprecated-spelling compatibility and reject fixture-specific normalizers.

## Required Submission Artifacts
- Parser and config patch
- Workflow preview patch
- CI workflow correction
- Codex config update
- Release note update

## Scoring Model
- `100` total points, awarded by category:
  - `35`: parser and config-loading correctness
  - `20`: workflow preview canonicalization
  - `15`: CI workflow consistency
  - `15`: release note and Codex config alignment
  - `15`: compatibility handling plus anti-shortcut robustness

## Score Caps
- Cap at `10/100` for docs-only, workflow-only, or parser-only rename notes.
- Cap at `20/100` for rename-everything patches that drop deprecated compatibility.
- Cap at `20/100` for plausible-but-unverified alignment plans with no emitted preview guarantees.
- Cap at `25/100` if parser compatibility is handled but preview canonicalization is missing.
- Cap at `30/100` if CI or release note drift remains.
- Cap at `50/100` if code is broadly correct but non-code contract alignment is incomplete.

## Visible Gates
- `make ci`
- Targeted parser tests
- Targeted workflow-preview tests

## Hidden Gates
- Deprecated spelling parses locally
- Preview output remains canonical-only even when deprecated spelling is input
- Workflow jobs use one consistent canonical token family
- Withheld configs cover both canonical and deprecated spellings
- Grep or AST guard rejects fixture-specific helper-path normalization

## Integrity And Shortcut Checks
- Reject rename-everything patches that remove deprecated compatibility
- Reject fixes isolated to `.github/workflows/ci.yml`
- Reject preview emitters that leak deprecated token names
- Reject weakening parser tests

## Variant Hardness Notes
- `V1`: one clear drift, but compatibility still matters
- `V2`: stale docs and lint noise should not distract from parser-preview symmetry
- `V3`: preserve unrelated workflow edits
- `V4`: config and release note are part of correctness
- `V5`: follow-up injects a regression in local fallback parsing after CI goes green

## Current Hardness Judgment
- Actual recorded solver run: `20/100`
- Naive `gpt-5.4/high` above `30/100`: `unlikely under current rubric`
