# Benchmark Run

## Final Judgment

- Final calibrated run score: `20/100`
- Target judgment: on target for a naive GPT-5.4/high solver

## Run 1

- Date: `2026-04-18`
- Model: `gpt-5.4`
- Reasoning: `high`
- Agent id: `019da333-e676-7001-bb74-bd9884062201`
- Outcome:
  - The solver aligned `workspace/.codex-plugin/plugin.json` and `workspace/docs/plugin_quickstart.md`.
  - Visible tests passed.
- Pre-hardening assessment:
  - This run overperformed the original bundle because discovery-index alignment and secondary docs were not yet in scope.

## Hardening Applied After Run 1

- Added `workspace/.agents/plugins/discovery_index.json`.
- Added a visible note-file check in `workspace/tests/test_marketplace_alignment.py`.
- Updated `evaluator_contract.md` so marketplace metadata that omits the manifest skill path caps the score at `20/100`.

## Run 2

- Date: `2026-04-18`
- Model: `gpt-5.4`
- Reasoning: `high`
- Agent id: `019da33a-e3ad-7a62-aa93-a70c80604121`
- Outcome:
  - The solver updated `workspace/.agents/plugins/discovery_index.json` and `workspace/docs/plugin_notes.md`.
  - Local verification after the run:
    - `cd workspace && python -m pytest -q tests` -> `3 passed`

## Final Scoring Breakdown

- `20/20`: visible suite passed
- `20/20`: plugin id, marketplace id, and skill path are internally consistent across the visible bundle
- `20/20`: discovery flow resolves through marketplace metadata, plugin manifest, and discovery index
- `20/20`: operator docs now use stable ids and real paths
- `0/20`: marketplace metadata still omits the manifest skill path
  - Evidence: `workspace/.agents/plugins/marketplace.json` has no `skill_path` field even though the manifest and discovery index do
- Raw subtotal: `80/100`
- Applied cap:
  - `marketplace entry that omits the manifest's skill path caps the run at 20/100`
- Final score: `20/100`

## Takeaway

The hardened family now keeps the naive solver near the target score by requiring marketplace metadata to be fully self-describing, not merely internally consistent enough for the visible checks.
