# Evaluator Contract

## Evaluation Intent

This family measures whether a solver can repair an in-thread heartbeat automation across TOML, prompt semantics, skill docs, and operator guidance without creating duplicates. Keyword alignment alone must not score above `20/100`.

## Required Inputs

- Final `automation.toml`
- Final prompt and skill docs
- Final operator notes
- Visible and hidden test results

## Scoring Breakdown

- `20 points`: visible suite passes in `workspace/tests`
- `20 points`: `workspace/automations/release_handoff/automation.toml` encodes a fully specified active heartbeat, including status and a non-generic schedule
- `20 points`: `workspace/automations/release_handoff/prompt.md` requests the exact in-thread handoff artifact with `## Summary` and `## Risks`
- `20 points`: `workspace/skills/release_handoff/SKILL.md` and `workspace/docs/automation_ops.md` forbid duplicate creation and reinforce in-place repair
- `20 points`: no replacement automation bundle is added and unrelated notes remain preserved

## Scoring Caps

- Visible checks alone cap the run at `20/100`
- Any heartbeat without explicit `status = "ACTIVE"` caps the run at `20/100`
- Any generic hourly schedule without a bounded weekday/time cadence caps the run at `20/100`
- Any prompt that omits the exact `## Summary` and `## Risks` sections caps the run at `20/100`
- Any new automation bundle or duplicate schedule entry caps the run at `0/100`

## Visible Checks

- `cd workspace && python -m pytest -q tests`

## Hidden Checks

- Confirm `kind`, `destination`, and cadence semantics match a thread heartbeat
- Confirm the prompt requests the exact bounded in-thread handoff artifact, not a detached file write
- Reject any added replacement automation bundle or duplicate schedule entry
- Confirm docs and skill text warn against duplicate automation creation
- Preserve unrelated operator notes byte-for-byte outside the intended repair surface

## Fail-Fast Conditions

- New automation bundle added
- Prompt still requests the wrong output surface
- Docs or skill text still instruct creating duplicates

## Red-Team Expectations

- A shallow solver will flip `kind`, `destination`, and a few prompt words
- A slightly stronger solver may create a fresh “fixed” automation rather than repairing in place
- Both must remain under `30/100`

## Hardness Judgment

- Current target judgment: naive GPT-5.4/high should remain under `30/100`
- Residual risk: if hidden prompt-artifact checks collapse to token scans, broad text alignment may still overperform
