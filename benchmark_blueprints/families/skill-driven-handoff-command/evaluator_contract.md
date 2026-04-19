# Evaluator Contract

## Scope
- Family: `skill-driven-handoff-command`
- Task: `cnb55-core-skill-driven-handoff-command-report-bundle`
- Runtime class: developer-productivity repo with CLI, renderer, skill doc, automation template, docs, and sample runtime bundles

## Evaluation Intent
Measure whether the solver can extend a real handoff command across code and Codex-native surfaces without breaking JSON export or checking in canned Markdown. The evaluator should reward true command behavior, skill accuracy, automation correctness, and output-contract fidelity.

## Required Submission Artifacts
- CLI and renderer patch
- Tests for Markdown and preserved JSON behavior
- Updated skill doc
- Updated automation template
- Updated docs or generated sample output

## Scoring Model
- `100` total points, awarded by category:
  - `35`: CLI and renderer correctness
  - `20`: JSON preservation and regression safety
  - `15`: skill-doc fidelity
  - `15`: automation-template correctness
  - `15`: sample-output and docs alignment

## Score Caps
- Cap at `10/100` for docs-only, skill-only, or automation-only updates.
- Cap at `20/100` for thin Markdown rendering that only matches the visible sample bundle.
- Cap at `20/100` for plausible command plans with no actual renderer or automation evidence.
- Cap at `25/100` if command behavior is partly right but optional sections or ordering remain weak.
- Cap at `30/100` if JSON export, automation, or sample-output fidelity is broken.
- Cap at `50/100` if code is right but one Codex-native surface remains stale.

## Visible Gates
- CLI tests pass
- Render tests pass
- Skill-contract test verifies `SKILL.md` references the shipped command

## Hidden Gates
- Markdown output includes screenshot evidence and open questions when present
- JSON export behavior remains unchanged
- Automation template invokes the canonical CLI entrypoint
- Sample output matches renderer output on a withheld bundle
- End-to-end automation check runs against real runtime inputs
- Second withheld bundle shape exercises optional sections and ordering

## Integrity And Shortcut Checks
- Reject static-template `summarize` implementations
- Reject doc or skill drift from actual CLI behavior
- Reject automation templates that echo canned output
- Reject removal of JSON-specific tests

## Variant Hardness Notes
- `V1`: straightforward CLI extension but still requires docs accuracy
- `V2`: stale skill text and ambiguous sample artifacts are deliberate noise
- `V3`: partial skill and automation edits must be preserved
- `V4`: code, skill, automation, and sample output all required
- `V5`: follow-up injects evidence that the first implementation drifted from command semantics

## Current Hardness Judgment
- Actual recorded solver run: `20/100`
- Naive `gpt-5.4/high` above `30/100`: `unlikely under current rubric`
