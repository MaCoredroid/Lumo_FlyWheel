# AGENTS.md

Modernize this handoff bundle into one canonical structured runtime path.

Rules:
- Keep the real handoff command runnable.
- Align live config, live runbook, and the canonical automation to the same command path.
- Retire stale duplicate automations semantically instead of keeping them half-live.
- If release or incident context is present, follow it without editing the context files themselves.
- Do not modify tests, fixtures, the legacy monolith note, the legacy copied prompt, or unrelated dirty-state skills.
