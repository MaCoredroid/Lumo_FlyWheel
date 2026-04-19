# Workspace Approval Rollout

Use this skill when a task adds or repairs `approval_state` across backend, API, CLI, UI, config, and runbook surfaces.

## Focus
- Thread the field from source data through service normalization, serialization, CLI export, and UI rendering.
- Preserve documented fallback behavior for legacy rows missing the new field.
- Keep operator docs and screenshot evidence aligned with actual UI state.

## Required Solver Behavior
- Compare one mixed dataset across service, API, and CLI representations.
- Verify the UI renders the same `approval_state` values shown by the API and CLI.
- Update defaults and runbook text if naming changed.

## Anti-Shortcut Rules
- Do not fabricate `approval_state` only in the frontend.
- Do not alias `approval_state` directly to `risk_level` in the wrong layers.
- Do not treat screenshot filenames as sufficient verification.

## Minimum Verification
- Run backend and frontend tests.
- Confirm one legacy row still renders with the documented fallback.

