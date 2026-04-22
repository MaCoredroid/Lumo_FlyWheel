# Workspace approval threading task

Implement `approval_state` end to end across backend, API, CLI, frontend, config, docs, tests, preview artifact, and rollout note.

Writable paths:
- backend/**
- cli/**
- frontend/**
- config/defaults.toml
- docs/runbooks/workspace-approvals.md
- tests/**
- artifacts/preview/workspace_admin_capture.json
- artifacts/rollout/approval_state_rollout_note.json

Read-only paths:
- seed_data/**
- release_context/** when present
- incident_context/** when present
- AGENTS.md
- Dockerfile
- bin/run-visible-tests
- .scenario_variant

Do not fabricate `approval_state` in the frontend or alias it from `risk_level`.
Run `bin/run-visible-tests` before you stop.
