# Daily Triage Workflow

The live Codex-facing entrypoint is `make codex-daily-triage`.

The Make target shells out to:

`python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

The automation cadence is pinned in `ops/automation/triage.cron` as:

`0 9 * * 1-5`

Workflow scope:

- Triage blocker incidents for the on-call rotation before any general queue cleanup.
- use `fixtures/open_incidents.json` as the sample evidence surface for local dry runs
- keep task semantics separate from schedule semantics when drafting Codex automation prompts
