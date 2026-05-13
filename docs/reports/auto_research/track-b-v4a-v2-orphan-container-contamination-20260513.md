# Track B v4a_v2 — docker orphan container contamination (2026-05-13 00:36 UTC)

Observed: `subprocess.run(timeout=1800)` in `run_track_b_e2e_task.py`
kills the **`docker run`** client when an attempt hits the wall budget,
but the **container itself** is not always reaped — Docker doesn't
reliably propagate the client's SIGKILL into the container, so the
codex process inside keeps running, keeps making requests to the proxy
on `127.0.0.1:8022`, and keeps competing for vLLM resources.

## What was observed directly

At 00:36 UTC on 2026-05-13 I caught one orphan red-handed:

| Container | Start | Age when observed | Status |
|---|---|---:|---|
| `nervous_tharp` | 23:59:12 UTC | 37 min | Still running, still bind-mounted RW to `transcript-merge-regression/run_04/workspace`, while `affectionate_edison` (task 3 run_01) was 7 min into its attempt |

Task 2 run_04's `runner_metadata.json` was already written
(`recorded_at=2026-05-13T00:29:12`, `codex_exit_code=124`,
`elapsed_s=1800`) — the **runner thought it was done** while the
container was still alive. The runner had already moved on to spawn
task 3 run_01 in `affectionate_edison`. From 00:29 to 00:36 (and
possibly slightly past) both containers were active simultaneously.

## Why the runner could not see this

The `subprocess.run(..., timeout=args.timeout_s)` call in
`run_track_b_e2e_task.py:696` raises `TimeoutExpired` (now caught by
my patch in this session). On timeout, Python sends `SIGKILL` to the
`docker run` client. The Docker client dies; the container does not.
There is no `docker stop <name>` cleanup at the runner level — the
patch I landed only catches the exception and writes metadata, it
does **not** stop the container.

## Which attempts likely had a co-orphan running at the start of the next attempt

Every prior attempt that ended with `rc=124` (timeout) is a suspect.
The next attempt in line starts immediately after `subprocess.run`
returns, so its early phase coexists with whatever the prior
container was still doing.

| Prior attempt | End | Next attempt starts | Risk window |
|---|---|---|---|
| task 1 run_01 | 21:22:40 rc=124 | task 1 run_02 starts 21:22:40 | early minutes of run_02 |
| task 1 run_02 | 21:52:40 rc=124 | task 1 run_03 starts 21:52:40 | early minutes of run_03 |
| task 1 run_03 | 22:22:40 rc=124 | task 1 run_04 starts 22:22:40 | early minutes of run_04 |
| task 1 run_04 | 22:47:19 rc=0 (clean) | task 2 run_01 starts 22:47:19 | none (clean exit) |
| task 2 run_01 | 23:17:19 rc=124 | task 2 run_02 starts 23:17:19 | early minutes of run_02 |
| task 2 run_02 | 23:47:19 rc=124 | task 2 run_03 starts 23:47:19 | early minutes of run_03 |
| task 2 run_03 | 23:59:12 rc=0 (clean) | task 2 run_04 starts 23:59:12 | none |
| task 2 run_04 | 00:29:12 rc=124 | **task 3 run_01 starts 00:29:12** | **confirmed orphan to 00:36** |

So **6 attempts** in the corpus so far MAY have started with a
co-running orphan from the prior attempt:

- responses-sdk-adapter-cutover: run_02, run_03, run_04
- transcript-merge-regression: run_02, run_03
- dead-flag-reachability-audit: run_01 (current — observed)

The clean-exit attempts (task 1 run_04, task 2 run_03) cannot have
left orphans because they exited their own subprocess naturally.

## What the contamination actually changes

For each affected next-attempt, during the orphan's lingering window
(roughly first 1–15 minutes of the next attempt, depending on how
fast the orphan finally died on its own):

1. **Proxy capture rows interleave.** `/tmp/track_b_e2e_proxy_capture/
   request_metrics.jsonl` is shared. The task runner slices by byte
   offset between subprocess start and end, so all rows the orphan
   wrote during the next attempt's window are attributed to the next
   attempt. Some rows in run_02/03/04's `vllm_request_metrics.jsonl`
   are not actually from that attempt's codex — they're from the
   previous attempt's orphaned codex still talking to the proxy.

2. **vLLM batching dilutes decode share.** vLLM batches concurrent
   requests; the orphan's tokens compete with the legitimate
   attempt's tokens for KV cache and decode slots. Decode_tps and
   spec_decode_acceptance numbers from the affected attempts'
   metrics rows are **understated** in proportion to orphan share.

3. **Wallclock comparison is fair within each attempt** (subprocess
   timer on the host is independent of container scheduling) but
   **per-regime efficiency comparisons** across attempts are not
   directly comparable when one attempt was the only consumer and
   another shared the GPU with an orphan.

What is **not** contaminated:

- The legitimate attempt's workspace edits (each container has its
  own bind-mount; the orphan can only mutate the previous attempt's
  workspace, not the active one).
- `runner_metadata.json` `elapsed_s` (host clock, not container clock).
- `codex_stdout.log` of the legitimate attempt (logged from inside
  its own container).
- Grader scores (computed post-hoc against the workspace, not against
  the metrics).

## What's been done

1. Killed `nervous_tharp` at 00:36 UTC. `affectionate_edison` (the
   legitimate task 3 run_01) continues without interference.
2. Started a container-age watchdog at `/tmp/codex_container_age_
   watchdog.sh`, pid `911466`. Scans every 30 seconds, runs
   `docker stop -t 5 && docker rm -f` on any `codex-runner:v1`
   container whose age exceeds 1850s (1800s budget + 50s grace).
3. Container leaks from any subsequent attempt's timeout will be
   reaped within 30s of crossing the budget, so the next attempt's
   "early window" contamination is now bounded to seconds rather
   than minutes.

## What's still needed

- **Real fix**: the task runner should `docker stop -t 5
  <container-name>` after `subprocess.run` times out (or on any
  non-zero return), and should `--name` each container so it can
  target the right one. Right now containers get random Docker names
  like `nervous_tharp` — the runner can't identify "its" container
  to stop. Tracking as a follow-up patch; current P2 in-memory
  process won't pick up code changes, so the watchdog is the
  operational fix for this round.
- **Optional re-attempt**: if per-regime numbers from the 5
  contaminated attempts need to be precise (e.g. to compare against
  the ablation rounds), those attempts should be re-run after the
  task-runner fix lands. For headline wallclock and pass-rate
  numbers, they are still usable since those metrics aren't
  affected.

## Note for task 3 specifically

Task 3 run_01 in `affectionate_edison` was active during the
nervous_tharp orphan window (00:29-00:36 → ~7 minutes of overlap).
Its `vllm_request_metrics.jsonl` will likely include some rows that
are actually nervous_tharp's. Cross-check: when task 3 run_01
finishes, count rows in its metrics file with `oracle_session_id`
mismatching task 3's session — those are the orphan-attributed rows.
If the count is small (say <10), treat as nuisance; if it's a
meaningful fraction of the attempt's turns, treat run_01 as
contaminated and rely on run_02/03/04 for task 3 stats.
