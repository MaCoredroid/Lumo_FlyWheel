# Benchmark Run

## Final Judgment

- Final calibrated run score: `20/100`
- Target judgment: on target for a naive GPT-5.4/high solver

## Run 1

- Date: `2026-04-18`
- Model: `gpt-5.4`
- Reasoning: `high`
- Agent id: `019da333-e523-7131-8726-e2f71d1d60a7`
- Outcome:
  - The solver repaired the existing automation in place and aligned the prompt, skill doc, and ops doc with in-thread heartbeat behavior.
  - Visible tests passed.
- Pre-hardening assessment:
  - This run overperformed the original bundle because the visible contract did not force a fully specified active heartbeat or exact handoff artifact shape.

## Hardening Applied After Run 1

- Added a visible `status = "ACTIVE"` check.
- Updated `evaluator_contract.md` so generic hourly schedules and prompts without exact `## Summary` / `## Risks` sections cap the run at `20/100`.

## Run 2

- Date: `2026-04-18`
- Model: `gpt-5.4`
- Reasoning: `high`
- Agent id: `019da33a-e297-7641-bbb2-8822693dfe7d`
- Outcome:
  - The solver added `status = "ACTIVE"` to `workspace/automations/release_handoff/automation.toml`.
  - Local verification after the run:
    - `cd workspace && python -m pytest -q tests` -> `4 passed`

## Final Scoring Breakdown

- `20/20`: visible suite passed
- `0/20`: the heartbeat is still not fully specified
  - Evidence: `workspace/automations/release_handoff/automation.toml` keeps `rrule = "FREQ=HOURLY;INTERVAL=1"` instead of a bounded weekday/time cadence
- `0/20`: the prompt still does not require exact `## Summary` and `## Risks` sections
- `20/20`: skill/doc surfaces continue to forbid duplicate creation and reinforce in-place repair
- `20/20`: no replacement automation bundle was added and `workspace/memory.md` stayed intact
- Raw subtotal: `60/100`
- Applied caps:
  - `generic hourly schedule caps the run at 20/100`
  - `prompt that omits exact section headings caps the run at 20/100`
- Final score: `20/100`

## Takeaway

The hardened family now rewards the obvious heartbeat/status fix but withholds deeper credit until cadence semantics and artifact shape are fully specified.
