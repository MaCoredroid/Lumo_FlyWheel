# Benchmark Run

## Final Judgment

- Final calibrated run score: `20/100`
- Target judgment: on target for a naive GPT-5.4/high solver

## Run 1

- Date: `2026-04-18`
- Model: `gpt-5.4`
- Reasoning: `high`
- Agent id: `019da333-e61a-7bf2-89c6-e5bd56626f62`
- Outcome:
  - The solver upgraded selection logic, refreshed manifest/config/docs, and edited the visible test surface.
  - Visible tests passed.
- Pre-hardening assessment:
  - This run overperformed the original bundle because the family did not yet penalize embedded default routing tables strongly enough.

## Hardening Applied After Run 1

- Added `workspace/docs/router_policy.md` plus a new visible docs check.
- Updated `evaluator_contract.md` so any embedded default routing table, capability map, or policy list caps the score at `20/100`.
- Kept test immutability in the rubric for future runs.

## Run 2

- Date: `2026-04-18`
- Model: `gpt-5.4`
- Reasoning: `high`
- Agent id: `019da33a-e2cd-7ac2-aeb9-9d0ccb8f10f3`
- Outcome:
  - The solver refreshed `workspace/tooling/catalog.py`, `workspace/tooling/tool_manifest.yaml`, `workspace/config/tool_router.toml`, `workspace/docs/tool_selection.md`, and `workspace/docs/router_policy.md`.
  - Local verification after the run:
    - `cd workspace && python -m pytest -q tests` -> `4 passed`

## Final Scoring Breakdown

- `20/20`: visible suite passed
- `20/20`: preferred-missing browser cases choose the correct eligible fallback
- `20/20`: preferred-present cases still choose the primary tool
- `0/20`: selection still depends on embedded default tables
  - Evidence: `workspace/tooling/catalog.py` still defines `DEFAULT_TOOL_CAPABILITIES`, `DEFAULT_PREFERRED_TOOLS`, `DEFAULT_CATALOG`, and `DEFAULT_ELIGIBLE_FALLBACK_POLICIES`
- `20/20`: docs are current and the rerun did not edit tests
- Raw subtotal: `80/100`
- Applied cap:
  - `embedded default routing table, capability map, or policy list caps the run at 20/100`
- Final score: `20/100`

## Takeaway

The hardened family now separates “reads enough config to pass visible tests” from “truly derives routing policy from the shipped catalog surfaces,” which holds the naive solver near the target score.
