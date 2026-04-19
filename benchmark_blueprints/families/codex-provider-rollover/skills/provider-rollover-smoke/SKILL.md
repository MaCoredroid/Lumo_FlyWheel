# Provider Rollover Smoke

Use this skill when a Codex provider or local proxy rollover breaks multi-turn Responses-path maintenance checks.

## Inputs
- Repo-local Codex TOML
- Smoke probe or health-check script
- Sample first-turn and follow-up fixtures
- Operator docs for the maintenance profile

## Workflow
1. Identify the selected default provider and its base URL.
2. Confirm the smoke checks exact `previous_response_id` continuity, not just a second success.
3. Preserve unrelated local tuning knobs and comments.
4. Align operator docs only after runtime behavior is correct.

## Avoid
- Rewriting the whole config file from scratch.
- Accepting one-turn success as sufficient.
- Leaving both legacy and new providers selectable by default.

## Expected output
- Correct default provider selection
- Exact follow-up continuity check
- Docs that match the repaired runtime path
