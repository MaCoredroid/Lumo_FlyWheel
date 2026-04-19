# Evaluator Contract

## Evaluation Intent

This family measures whether a solver can repair a repo-local skill contract across docs, config, fixtures, and the live validator path. It must not be scorable above `30/100` through grep-and-rename edits alone.

## Required Inputs

- Final workspace diff
- Final contents of `SKILL.md`, example fixtures, and `codex/config.toml`
- Output from the documented validator flow
- Test results from visible and hidden suites

## Scoring Breakdown

- `20 points`: visible suite passes in `workspace/tests`
- `20 points`: docs and config point at the live validator entrypoint without hardcoding the sample fixture path
- `20 points`: both `workspace/examples/expected_report.md` and `workspace/examples/nightly_expected_report.md` match the real validator output shape for their fixtures
- `20 points`: no retained `skill_smoke` reference remains anywhere in the intended task surface, including docs
- `20 points`: no test weakening, no compatibility shim, and no unintended edits outside the intended repair surface

## Scoring Caps

- Visible checks alone cap the run at `20/100`
- Any config or doc path that still hardcodes the sample fixture caps the run at `20/100`
- Any retained legacy alias, wrapper, or shim caps the run at `10/100`
- Any weakened test or deleted assertion caps the run at `0/100`

## Visible Checks

- `cd workspace && python -m pytest -q tests`

## Hidden Checks

- Extract every executable command example from `workspace/skills/release_handoff/SKILL.md` and run it
- Re-run the documented sample flow and an alternate nightly flow and verify the emitted reports have both required sections in order
- Confirm `workspace/docs/usage.md` matches the live validator path and does not pin the sample fixture
- Confirm `workspace/docs/skill_pack_notes.md` no longer preserves a live `skill_smoke` path
- Reject any retained `skill_smoke` alias, wrapper, or compatibility shim
- Reject changes that weaken tests, fixture assertions, or contract docs
- Reject modifications outside the intended task surface in dirty-workspace variants

## Fail-Fast Conditions

- Any legacy helper alias remains callable from the documented path
- Example fixtures are rewritten to bypass the live validator
- The solver deletes tests or assertion coverage instead of repairing the contract

## Red-Team Expectations

- A shallow solver will try a token rename and fixture text edits
- A stronger but still naive solver may add a temporary shim instead of restoring the contract
- The evaluator must treat both as under-`30/100` outcomes

## Hardness Judgment

- Current target judgment: naive GPT-5.4/high should remain under `30/100`
- Residual risk: a solver could still normalize every referenced surface mechanically and accidentally satisfy the contract if hidden prompt examples are too narrow
