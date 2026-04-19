# Benchmark Run

## Final Judgment

- Final calibrated run score: `20/100`
- Target judgment: on target for a naive GPT-5.4/high solver

## Run 1

- Date: `2026-04-18`
- Model: `gpt-5.4`
- Reasoning: `high`
- Agent id: `019da333-e483-7b33-ade8-42a960bf4129`
- Outcome:
  - The solver upgraded routing logic, aligned docs/config, and added a new contract test plus an audit note.
  - Visible tests passed.
- Pre-hardening assessment:
  - This run overperformed the original bundle because config-driven behavior and test immutability were not enforced tightly enough.

## Hardening Applied After Run 1

- Added `workspace/docs/skill_router_notes.md` plus a new visible docs check.
- Updated `evaluator_contract.md` so import-time router snapshots cap the score at `20/100`.
- Kept test immutability in the rubric and retained the benchmark expectation that `workspace/config/skill_router.toml` is the live source of truth.

## Run 2

- Date: `2026-04-18`
- Model: `gpt-5.4`
- Reasoning: `high`
- Agent id: `019da33a-e224-7910-9a94-a7eed59d55cf`
- Outcome:
  - The solver updated `workspace/router/skill_router.py` and `workspace/docs/skill_router_notes.md`.
  - Local verification after the run:
    - `cd workspace && python -m pytest -q tests` -> `7 passed`

## Final Scoring Breakdown

- `20/20`: visible suite passed
- `20/20`: list triggers and required-input gating work across the visible/unseen cases sampled during review
- `20/20`: negative triggers suppress false positives in the visible and audit-backed examples
- `0/20`: router still relies on an import-time cached snapshot
  - Evidence: `workspace/router/skill_router.py` defines `ROUTER_CONFIG = _load_router_config()` and `SKILLS = ROUTER_CONFIG.skills` at module import time
- `20/20`: docs/config are current and the rerun did not edit tests
- Raw subtotal: `80/100`
- Applied cap:
  - `import-time cached router snapshot or skill list derived once and reused across decisions caps the run at 20/100`
- Final score: `20/100`

## Takeaway

The hardened family now distinguishes between “config-driven enough to pass visible tests” and “live contract-driven at decision time,” which keeps the naive solver near the target score.
