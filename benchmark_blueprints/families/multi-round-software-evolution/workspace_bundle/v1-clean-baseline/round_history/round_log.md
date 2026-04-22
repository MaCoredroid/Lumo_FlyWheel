# Round History

- Round 1 focused on reducing cold-start spikes. It helped p95 startup time, but replay determinism remained flaky.
- Round 2 added the rate limiter that contained the pager storm on the legacy fan-out path.
- Round 3 attempted a small restore cleanup, but the writer and restore ordering still disagree on some snapshots.

Current lesson: the loud visible issue is contained; the next round should remove the blocker that keeps later work from landing cleanly.
