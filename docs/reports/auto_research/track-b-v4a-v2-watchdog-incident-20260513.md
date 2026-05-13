# Track B v4a_v2 — watchdog incident, recovery + restart (2026-05-13 00:55 UTC)

## What happened

I wrote a container-age watchdog (`/tmp/codex_container_age_watchdog.sh`)
to reap orphan `codex-runner:v1` containers left behind when
`subprocess.run` timed out. The watchdog had a date-parsing bug:

```bash
docker ps --filter "ancestor=codex-runner:v1" --format '{{.Names}} {{.CreatedAt}}'
# returns lines like:
#   nervous_tharp 2026-05-12 23:59:12 +0000 UTC
started="$(echo "$line" | awk '{print $2}')"
# grabs "2026-05-12" — the date only
age=$(( $(date -u +%s) - $(date -u -d "$started" +%s) ))
# "age" = seconds since midnight of $started's date, not since container creation
```

For a container started TODAY, `$started` is today's date. `date -d
"2026-05-13"` is today's 00:00:00 UTC. `age` ends up = "seconds since
midnight today". Once midnight + 1850s = 00:30:50 UTC passed, the
watchdog started killing every codex-runner:v1 container in sight —
including the live ones for the active attempt.

First scan that found a victim: 00:38:01 UTC, killed
`affectionate_edison` (the live container for task 3 / dead-flag-
reachability-audit / run_01 — which had been doing real work for 8m 49s).
Every subsequent codex spawn was killed within ~30s by the next
watchdog scan. 31 kills total over 15 minutes before I caught it at
00:52:37 and stopped the watchdog.

## Damage

By recorded_at + elapsed_s:

| Family | Run | elapsed_s | rc | Verdict |
|---|---|---:|---:|---|
| responses-sdk-adapter-cutover | run_01..04 | 1438/1339/1800/1479 | 124/124/124/0 | **VALID** (completed before watchdog start) |
| transcript-merge-regression | run_01..04 | 1800/1800/713/1800 | 124/124/0/124 | **VALID** (completed before watchdog start) |
| dead-flag-reachability-audit | run_01 | 529 | 0 | **PARTIAL** — real work but cut short at the watchdog's first kill |
| dead-flag-reachability-audit | run_02..04 | 30 | 0 | **INVALID** — phantom 30s kills |
| sqlalchemy-2-session-modernization | run_01..04 | 30 | 0 | **INVALID** |
| security-audit-hotfix-remediation | run_01..04 | 30 | 0 | **INVALID** |
| responsive-checkout-visual-regression | run_01..04 | 30 | 0 | **INVALID** |
| incident-evidence-synthesis | run_01..04 | 30 | 0 | **INVALID** |
| policy-aware-request-resolution | run_01..04 | 30 | 0 | **INVALID** |
| multi-tool-transaction-repair | run_01..04 | 30 | 0 | **INVALID** |
| release-note-to-plan-translation | run_01..03 | 30 | 0 | **INVALID** |
| fanout-fullstack-release-blocker | – | – | – | not yet started |

Tasks 1+2 valid because they finished before the watchdog ever
started (the watchdog began at 00:36:43 UTC; task 2 run_04 ended at
00:29:12 UTC).

`codex_exit_code=0` on the phantom runs is misleading — `docker stop`
sends SIGTERM, codex inside the container exits gracefully with 0,
and `docker run`'s exit code propagates that 0 back to the host. The
real signal that something is wrong is the missing `turn.completed`
event in `codex_stdout.log` (the phantom runs end mid-execution, no
final usage block).

## Recovery actions taken

1. **Killed the broken watchdog.** Pid `911466` terminated at 00:53 UTC.
2. **Killed the P2 driver and all child codex processes.** Pid `889758`
   plus subprocess.run children + any remaining docker containers.
3. **Archived the valid task 1 + task 2 data plus round-prefix
   artifacts** to
   `output/track_b_e2e_v4a_v2/round_0_phase1_task1_2_PRESERVED/`.
   Contents: `responses-sdk-adapter-cutover__v1-clean-baseline/` (4
   attempts with metadata + workspace + dcgm), `transcript-merge-
   regression__v1-clean-baseline/` (same), plus `preflight_audit.json`,
   `codex_system_prompt.json`,
   `codex_system_prompt_decomposition.json`, `round_warmup_pass.json`.
4. **Deleted the broken `round_0/`** (all attempts from task 3 onward
   that were either killed-at-30s or in-progress when killed).
5. **Truncated `/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl`**
   so the resumed pipeline gets a clean capture from its own warmup.
6. **Patched `run_track_b_e2e_task.py`** to reap any
   `codex-runner:v1` container in its TimeoutExpired handler — proper
   `docker stop -t 5` + `docker rm -f`, which propagates SIGTERM into
   the container instead of leaving a leaked codex process. The
   pipeline runs attempts serially, so stopping every codex-runner:v1
   on timeout is safe (only one legitimate container exists at any
   moment).
7. **Temporarily shrunk `TRACK_B_E2E_TASKS` from 11 to 9** by
   commenting out responses-sdk-adapter-cutover + transcript-merge-
   regression in `scripts/build_track_b_e2e_summary.py`. The resumed
   run only re-does the 9 tasks that need re-doing.
8. **Restarted P2.** New driver pid `923203`. Output goes to a fresh
   `output/track_b_e2e_v4a_v2/round_0/`.

## Merge plan for when the resume completes

When the resumed pipeline finishes:

1. Restore `TRACK_B_E2E_TASKS` to 11 entries (uncomment the 2 lines).
2. `mv output/track_b_e2e_v4a_v2/round_0_phase1_task1_2_PRESERVED/
   {responses-sdk-adapter-cutover,transcript-merge-regression}__v1-clean-baseline/
   output/track_b_e2e_v4a_v2/round_0/`.
3. Regenerate `round_summary.json` manually against the now-merged
   11-task set (the resume's `round_summary.json` only aggregates 9
   tasks).
4. Run graders on each task's workspace and store
   `grader_result.json` per attempt.
5. Write the v4a_v2 closeout report with the merged headline numbers.

## Lessons

- **Never trust ad-hoc shell timestamp parsing without testing both
  yesterday-and-today timestamps.** A second of pair-coding with
  `docker inspect --format '{{.State.StartedAt}}'` (which returns a
  proper RFC3339 timestamp + nanoseconds) would have shipped a
  correct watchdog the first time.
- **Always run a destructive watchdog in dry-run mode for one scan
  cycle first.** Logging the candidate kills before issuing them
  would have caught the bug in 30 seconds.
- **Container reaping belongs in the runner, not in a sidecar
  watchdog.** The runner knows which container belongs to the current
  attempt (because it spawned it). A separate watchdog has to guess.
  The patched task runner does this correctly now.
- **`subprocess.run(..., timeout=)` with a docker run command requires
  explicit container cleanup.** The default behavior is to leak the
  container. Document this widely in any harness that wraps codex
  through docker.

## What this means for the broader audit

- The §13–§17 proxy stack remains valid. Task 1 + task 2 with their
  fully-populated `vllm_request_metrics.jsonl` (0% `None`-tokens, real
  spec_decode metrics) demonstrate the proxy patches still hold.
- The orphan-contamination story (the prior doc, `track-b-v4a-v2-
  orphan-container-contamination-20260513.md`) was correct in its
  cause — but the attempted fix (the broken watchdog) was worse than
  the disease. The proper fix (in-runner docker stop) is now in
  place.
- The v4a_v2 baseline closeout will land after the resume completes
  (~6-9 more hours from this restart).
