# Track B v4a_v2 baseline — mid-pipeline audit (2026-05-12 ~20:30 UTC)

This report documents what has been run since the §19.10 closeout, what
patches were required to get truthful instrumentation under the §13-§17
proxy stack, and what numbers are flowing in right now. P2 is still
running and continues to write artifacts as I type — those follow-up
numbers will arrive over the next few hours.

## Scope of the run

| Item | Value |
|---|---|
| Output dir | `output/track_b_e2e_v4a_v2/round_0/` |
| Corpus | 11 active v4a tasks (excluded plugin-scaffold-alignment + skill-router-contract-upgrade — missing AGENTS.md and seeded drift) |
| Ablation point | D (all techniques on: T1+T2+T3+T4) |
| Repeats / task | 4 |
| Per-attempt budget | 1800 s (30 min) |
| Runtime config hash | `sha256:5ae88ac4e10201f83a617e2bda3f1c07da4c7217c80db5482d317a79dd93b43a` |
| Proxy stack | §13 + §14 + §16 + §17 patches active |
| Proxy env | `LUMO_PROXY_NONSTREAM_BYPASS=1`, `LUMO_PROXY_AUTO_CONTINUE=1`, `LUMO_PROXY_AUTO_CONTINUE_MAX_RETRIES=5`, `LUMO_TRACK_B_REQUEST_METRICS_OUT=/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl` |
| Endpoint | `http://127.0.0.1:8022/v1` (bench proxy) → `http://127.0.0.1:9950` (vLLM) |
| Model | qwen3.5-27b |
| Launcher | `scripts/launch_v4a_v2_round_0_baseline.sh` |
| Started at | 2026-05-12T17:43Z (current attempt; two prior attempts were killed and restarted after each patch landed) |

## Headline numbers so far

Two tasks (out of 11) have completed at least one run.

### Task 1 — responses-sdk-adapter-cutover (all 4 attempts done)

| Attempt | rc | elapsed | normalized rows |
|---|---:|---:|---:|
| run_01 | 1 | 1438 s (24.0 min) | 113 |
| run_02 | 0 | 1339 s (22.3 min) | 61 |
| run_03 | 124 | 1800 s (30.0 min) | 81 |
| run_04 | 124 | 1800 s (30.0 min) | 91 |

Median wallclock: **24.0 min**. 2 of 4 attempts hit the 1800s wall budget
(rc=124) — the patched `subprocess.TimeoutExpired` handler still wrote
the metadata, trace, and per-turn artifacts on those.

### Task 2 — transcript-merge-regression (2 of 4 attempts done, run_03 in flight)

| Attempt | rc | elapsed | normalized rows |
|---|---:|---:|---:|
| run_01 | 0 | 938 s (15.6 min) | 38 |
| run_02 | 124 | 1800 s (30.0 min) | 115 |
| run_03 | (in flight) | — | — |

### Pacing read

Task 1 was the §18 outlier (responses-sdk-adapter-cutover wrote 64 files
under validation). At ~26 min mean per attempt × 4 attempts = ~108 min
per task wallclock for this kind of work. Task 2's first attempt at 15.6
min is more representative of the rest of the corpus. A reasonable
projection for the full D-point round is **6-10 hours** of remaining
wallclock from the 17:43Z start.

## Patches that had to land first

These 5 patches all came from chasing harness defects exposed only once
the proxy stack actually let the model do real work. Without each, the
v4a_v2 baseline would have been silently invalid.

### 1. `_normalize_vllm_request_metrics`: skip-not-raise

`scripts/run_track_b_e2e_task.py:213` previously raised on any row
missing `prompt_tokens` / `completion_tokens`. Auto-continue retry
envelopes from the proxy come through without populated `usage` (see
patch 4), and that single bad row killed the entire post-codex artifact
write path.

Patched to return `None` so the downstream "no normalized rows" branch
writes a deferred per-turn record. Caller no longer crashes mid-task.

### 2. Task runner: catch `TimeoutExpired`

`scripts/run_track_b_e2e_task.py:696` ran `subprocess.run(..., timeout=)`
inside a `try`/`finally` that only stopped the DCGM sampler. When codex
hit the per-attempt wall budget, `TimeoutExpired` propagated out of
`run_one` *before* `runner_metadata.json` / `codex_trace.jsonl` /
`vllm_request_metrics.jsonl` were written.

Patched to synthesize a `subprocess.CompletedProcess(returncode=124)` on
timeout and fall through to the artifact write. Verified live —
run_03 and run_04 of responses-sdk-cutover both wrote full metadata
despite rc=124.

### 3. Proxy: propagate `usage` into `capture_state`

`src/lumo_flywheel_serving/inference_proxy.py:1158` updated
`capture_state` with `response_id`, `model`, `saw_response_completed`,
but dropped the upstream `usage` dict. The downstream
`_build_request_metrics_row` read `response_observed["usage"]` and
came up empty, so 96% of captured rows had `prompt_tokens=None /
completion_tokens=None`.

Patched to copy `usage` through. After restart, the new capture is
running at **0% None-token rate** (verified at 19 / 19 rows and ongoing).

### 4. Round driver: allow deferring `codex_command_smoke`

`scripts/run_track_b_e2e_round.py:14` enumerated only 3 deferrable
preflight checks. Under the §13-§17 proxy stack the smoke prompt
"complete it in this workspace" no longer short-circuits — codex now
performs real tool calls and auto-continue holds it past any reasonable
smoke window. §18 already established substrate trust.

Patched to include `codex_command_smoke` in `DEFERABLE_PREFLIGHT_CHECKS`.

### 5. Preflight: skip the smoke when deferred

`scripts/preflight_track_b_e2e.py:388` ran `_codex_command_smoke(args)`
unconditionally even when the check was marked deferred via
`--defer-checks`. That wasted ~10 minutes of preflight wallclock per
round (the smoke ran to its 600s timeout).

Patched to short-circuit when `codex_command_smoke` is in
`--defer-checks`: returns `{"ok": False, "reason": "deferred",
"skipped": True}` immediately. Preflight now completes in ~1-2 minutes.

## Corpus change

`scripts/build_track_b_e2e_summary.py` `TRACK_B_E2E_TASKS` shrunk from 13
to 11. The two removed tasks (plugin-scaffold-alignment +
skill-router-contract-upgrade) both lacked `AGENTS.md` AND lacked the
seeded scaffold drift their `task_spec.md` describes — codex correctly
reports "task already complete" in those workspaces with no real
artifacts to write.

Drafted `AGENTS.md` and `.scenario_variant` for each so re-inclusion is
one step once their `workspace_bundle/v1-clean-baseline/` content gets
real drift restored. Both fixtures are in this commit, but the corpus
remains at 11 — re-add only after a validation run shows the model
writes real edits on them.

New `SAMPLE_HASH` after the shrink:
`sha256:91e35b265ff94d0e892456a958224636f651e64e037e723205c8f339d70079a3`.

## Harness incident: codex workspace leak to main repo

While P2 was running, the codex subprocess inside one of the workspaces
walked up the directory tree (workspaces have no `.git`), found the main
repo's `.git`, and committed its task work directly to `main` as the
local user. Four commits landed on origin (`e79cf77`..`b4643c0`) mixing
real harness patches with output workspace artifacts and the wrong
commit messages.

Reconciled non-destructively in `cee6574` — kept the real patches that
were in those commits, removed the leaked workspace files and a stale
`.claude/scheduled_tasks.lock`.

The root cause — codex finding the parent `.git` — is unfixed in the
running pipeline. Active workspace was `git init`'d as a stopgap so
new commits stay local. Proper fix: the task runner should
`git init -q` each workspace before spawning codex. Tracking this
separately; not in P2's current run.

## Live state at audit time

| Field | Value |
|---|---|
| P2 driver pid | `848597` |
| Total elapsed | ~2h 42min |
| Tasks done | 1 / 11 (responses-sdk-adapter-cutover, all 4 attempts) |
| Tasks in flight | transcript-merge-regression (run_03 active) |
| Tasks pending | 9 |
| Metrics capture rows | growing; current run is fully populated (`prompt_tokens` + `completion_tokens` present on every row) |
| Cron loop | every 5 min (`81c57a06`) — auto-monitors and reports |

## What this round will produce

When P2 completes:

- `runner_metadata.json` per attempt (44 of them) with real `elapsed_s`,
  `codex_exit_code`, `vllm_request_metrics_capture` summary, `warmup_pass`
- `codex_trace.jsonl` per attempt — synthesized from proxy capture
- `vllm_request_metrics.jsonl` per attempt — real tokens, no `None`s
- `vllm_per_turn.json` per attempt
- `codex_stdout.log` / `codex_stderr.log` per attempt
- `dcgm_samples.jsonl` per attempt
- Per-task aggregated summary via `build_track_b_e2e_summary.py task ...`
  (the round driver auto-invokes this)
- A `round_summary.json` once the runner finishes the last attempt

After that, post-hoc grading via `scripts/run_v4a_graders_on_validation.py`
(adapted to walk `round_0/<task>__variant/run_NN/workspace/`) yields
real `task_score` per attempt — feeds the §18 → §19 → v4a_v2 score-rate
comparison.

## What this round does NOT cover

- Ablation a / b / c (T2/T3/T4 individual contributions) — P3 starts only
  after P2 lands. Same corpus, same patched harness, runtime flags written
  to `/tmp/lumo_track_b_runtime_flags.json` per ablation point.
- Grader scoring is *not* in-line in the round driver yet. The summary
  builder accepts `task_score` from the per-attempt trace, but
  `run_track_b_e2e_task.py` always passes `task_score=None` to
  `_synthesize_codex_trace_from_proxy_rows`. Wiring graders in-line is
  task #51 follow-up; for now we post-hoc grade the workspaces.

## Audit checklist

- [x] Corpus shrunk to 11 with rationale recorded in `TRACK_B_E2E_TASKS` header
- [x] 4 harness instrumentation patches landed and verified live
- [x] Stray-commit leak from codex agent reverted; cleanup commit on `main`
- [x] Real per-attempt metadata flowing on the active run
- [x] Token-count fields present on 100% of post-restart capture rows
- [ ] Per-task `task_score` populated (deferred — post-hoc grader pass)
- [ ] All 11 tasks × 4 attempts completed (in progress)
- [ ] Round summary regenerated and committed
- [ ] P3 ablation launched against same corpus
