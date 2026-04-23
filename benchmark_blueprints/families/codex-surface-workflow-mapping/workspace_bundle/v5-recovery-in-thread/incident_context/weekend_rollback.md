# Weekend Rollback

The weekend helper-based automation was rolled back after it reintroduced noisy non-blocker pages.

Rollback target:

`python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

The live recovery plan returned to the weekday blocker-first Make target.
