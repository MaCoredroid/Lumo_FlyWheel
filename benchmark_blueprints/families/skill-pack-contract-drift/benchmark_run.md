# Benchmark Run

## Final Judgment

- Final calibrated run score: `20/100`
- Target judgment: on target for a naive GPT-5.4/high solver

## Run 1

- Date: `2026-04-18`
- Model: `gpt-5.4`
- Reasoning: `high`
- Agent id: `019da333-e33c-7342-a999-07cf3df98f48`
- Outcome:
  - The solver aligned `workspace/skills/release_handoff/SKILL.md`, `workspace/codex/config.toml`, `workspace/docs/skill_pack_notes.md`, and `workspace/examples/expected_report.md`.
  - Visible tests passed.
- Pre-hardening assessment:
  - This run overperformed the original bundle because the task only exercised the primary sample flow and did not penalize sample-bound command wiring.

## Hardening Applied After Run 1

- Added `workspace/docs/usage.md` as a new visible contract surface.
- Added `workspace/fixtures/nightly.json` and `workspace/examples/nightly_expected_report.md` to create a second example path.
- Updated `evaluator_contract.md` to cap any sample-bound command wiring at `20/100`.

## Run 2

- Date: `2026-04-18`
- Model: `gpt-5.4`
- Reasoning: `high`
- Agent id: `019da33a-e1e7-7f42-b919-87814992cc5b`
- Outcome:
  - The solver updated `workspace/docs/usage.md` and `workspace/examples/nightly_expected_report.md`.
  - Local verification after the run:
    - `cd workspace && python -m pytest -q tests` -> `4 passed`

## Final Scoring Breakdown

- `20/20`: visible suite passed
- `0/20`: docs/config are still sample-bound
  - Evidence: `workspace/codex/config.toml`, `workspace/docs/usage.md`, and `workspace/skills/release_handoff/SKILL.md` still pin `fixtures/sample.json`
- `20/20`: primary and nightly expected reports both match the validator's two-section shape
- `20/20`: no retained `skill_smoke` reference remains in the workspace
- `20/20`: no test weakening, shim, or unintended cross-surface edits were observed
- Raw subtotal: `80/100`
- Applied cap:
  - `config or doc path that still hardcodes the sample fixture caps the run at 20/100`
- Final score: `20/100`

## Takeaway

The hardened family now produces the intended benchmark behavior: the solver clears the visible contract, but misses the deeper requirement that the skill contract be generic rather than sample-pinned.
