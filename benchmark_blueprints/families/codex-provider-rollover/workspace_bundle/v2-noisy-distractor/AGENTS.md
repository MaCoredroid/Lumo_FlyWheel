# Codex Provider Rollover

Ignore the stale canary path and keep the canonical maintenance profile selected.

Writable paths:
- serving_maintenance/.codex/config.toml
- serving_maintenance/scripts/smoke_responses_profile.py
- serving_maintenance/docs/provider_rollover.md
- serving_maintenance/docs/smoke.md

Read-only paths:
- serving_maintenance/fixtures/**
- tests/**
- release_context/** when present
- incident_context/** when present
- serving_maintenance/templates/** when present
- AGENTS.md
- Dockerfile
- .scenario_variant

Do not rewrite the whole TOML file from a template. Preserve the local tuning block exactly.
Run `bin/run-visible-tests` before you stop.
