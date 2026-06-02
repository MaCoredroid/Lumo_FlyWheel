# FR9 Swap / B4 Temp 0.6 Independent-Row Closeout

Status: **REOPENED_IN_PROGRESS** as of 2026-06-02. The earlier clean
`lowmem088_mtp5_s1` arm remains accepted, but the swap experiment is not
closed until every required arm below either has clean evidence or is
explicitly invalidated as infra-blocked evidence rather than a benchmark
result.

This file replaces the stale June 1 closeout. The earlier version marked
`fr9_b4temp06_mtp5_s1_20260601T230213Z` and
`fr9_b4temp06_mtp5_s2_20260601T233000Z` as clean. That was wrong after the
later capture audit: every per-task `vllm_request_metrics.jsonl` under both
old tags is zero bytes, so capture did not advance into SWE. Those tags are
invalid and are not accepted evidence here.

## Requested Scope

- Track: FR9 Swap, independent-row MTP/spine tuning before enhanced tree work.
- Serving config: B4/Fb, `row_mode=independent`, `temp=0.6`.
- Dataset: the same SWE-bench Verified 16-instance subset:
  `docs/reports/auto_research/swe-bench-concprobe16-verified-instances-20260522.json`.
- SWE shape: x86 SWE box, `concurrency=4`, agent wall 1800 s, eval timeout
  1800 s.
- Evidence rule: no contaminated or zero-request-metrics tag is counted as an
  accepted arm.
- Reopened run policy: infra failures are fix-and-rerun events, not accepted
  benchmark results. Any restarted tag must keep the same subset, B4/Fb temp
  0.6 serving shape, x86 SWE execution, concurrency/wall settings, and evidence
  standard as the accepted `lowmem088_mtp5_s1` arm.

## Required Arm Matrix

The reopened FR9 swap goal requires the following arm variants before enhanced
tree work can consume this experiment as baseline evidence:

| Required arm | Status | Required action / acceptance bar |
|---|---|---|
| `mtp=5`, `spines=1`, `gpu_memory_utilization=0.88` | accepted | Keep `fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z` as the clean baseline: 16/16 x86 tasks, all per-task request-metrics files nonzero, full campaign and agentic artifacts present. |
| `mtp=5`, `spines=2`, `gpu_memory_utilization=0.88` | invalidated; rerun required | Latest retry tag `fr9_b4temp06_lowmem088_mtp5_s2_20260602T051155Z` passed the hardened request-metrics smoke preflight, but aborted when the second batch produced 0-byte per-task request metrics. Do not count it as a result. Rerun only after the x86 mirror/streamer failure mode is fixed and all 16 tasks can complete on x86 with nonzero per-task request metrics, complete summaries/traces, and a verified `independent_winner_trace.jsonl` with no winner-superset violations and copy-missing sum 0. |
| `mtp=3`, `spines=2`, lowmem retry if strict 0.90 cannot prelaunch | required pending | Launch only after the `mtp=5`, `spines=2` arm is either accepted or invalidated for fix-and-rerun. Accept under the same x86, metrics, artifact, and winner-trace rules; strict 0.90 prelaunch memory failures are infra-blocks and do not count as SWE evidence. |

Every accepted arm must include `driver.log`, top-level and nested
`campaign_summary.json`, `predictions.jsonl`, `agentic_summary.json`,
`dgx_steptrace.jsonl`, `per_req_spec_trace.jsonl`, per-task
`runner_metadata.json`, per-task nonzero `vllm_request_metrics.jsonl`,
`vllm_per_turn.json`, eval artifacts, and speed/overhead metrics comparable to
the accepted `lowmem088_mtp5_s1` report. Multi-spine arms additionally require
winner-trace validation.

## Decision Summary

| Arm / tag | Decision | Result evidence |
|---|---|---|
| `fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z` | accepted clean arm | lowmem088, `mtp=5`, `spines=1`, B4/Fb, temp 0.6, x86 SWE; 8/16 resolved, 8/16 failed; all 16 runner metadata files have nonzero request-metrics bytes. |
| `fr9_b4temp06_mtp5_s1_20260601T230213Z` | invalid | stale June 1 tag; all 16 per-task request-metrics files are 0 bytes. Do not use its 1/16 stale summary as accepted evidence. |
| `fr9_b4temp06_mtp5_s2_20260601T233000Z` | invalid | stale June 1 tag; all 16 per-task request-metrics files are 0 bytes. Do not use its stale summary as accepted evidence. |
| strict `mtp=5`, `spines=2`, gpu memory util 0.90 | no valid campaign | prelaunch failed because free memory was below the 0.90 requested threshold. |
| `fr9_b4temp06_lowmem088_mtp5_s2_20260602T033500Z` | invalid contaminated | first launched lowmem `spines=2` attempt; request metrics are 0 bytes for every inspected task, including three committed failed task rows and five other zero-metric task dirs. |
| `fr9_b4temp06_lowmem088_mtp5_s2_20260602T035600Z` | invalid partial evidence | first four tasks had nonzero metrics and commits, but the next batch had 0-byte metrics and triggered the missing-request-metrics guard. Whole tag is invalid. |
| `fr9_b4temp06_lowmem088_mtp5_s2_20260602T041200Z` | invalid contaminated | smoke capture passed 1378/1378, but the first four x86 SWE tasks all had 0-byte request metrics; capture stayed smoke-only. No `spines=2` SWE result is accepted. |
| `fr9_b4temp06_lowmem088_mtp5_s2_20260602T051155Z` | invalid partial evidence | hardened smoke capture passed 1359/1359, and the first four x86 tasks had nonzero metrics and commits, but the next batch had 0-byte request metrics; the hardened guard aborted at `astropy__astropy-13579`. Whole tag is invalid. |
| strict/lowmem `mtp=3`, `spines=2` | no valid campaign | only old strict prelaunch failures are documented; no local `fr9_b4temp06*mtp3*s2*` output directory exists. |

## Accepted Arm: `lowmem088_mtp5_s1`

Tag:
`fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z`.

Artifacts verified:

- `output/fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z/driver.log`
- `output/fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z/campaign_summary.json`
- `output/fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z/agentic_summary.json`
- `output/fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z/dgx_steptrace.jsonl`
- `output/fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z/per_req_spec_trace.jsonl`
- `output/fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z/fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z/campaign_summary.json`
- `output/fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z/fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z/predictions.jsonl`

Result counts from `campaign_summary.json`:

| Metric | Value |
|---|---:|
| instances_total | 16 |
| resolved | 8 |
| failed | 8 |
| resolved_rate | 0.5 |
| failure_mode_counts | tests_passed=8, tests_failed=5, patch_apply_failed=3 |
| codex_wall_seconds p50 / p90 / p99 / min / max | 1800.102 / 1800.136 / 1800.159 / 674.202 / 1800.159 |
| eval_wall_seconds p50 / p90 / p99 / min / max | 51.861 / 65.036 / 67.204 / 0.0 / 67.204 |
| campaign window | 2026-06-02T00:59:17Z to 2026-06-02T03:24:28Z |

Task verdicts:

| Resolved | Failed |
|---|---|
| `astropy__astropy-12907` | `astropy__astropy-13033` |
| `astropy__astropy-13453` | `astropy__astropy-13236` |
| `astropy__astropy-14096` | `astropy__astropy-13398` |
| `astropy__astropy-14309` | `astropy__astropy-13579` |
| `astropy__astropy-14365` | `astropy__astropy-13977` |
| `astropy__astropy-14508` | `astropy__astropy-14182` |
| `astropy__astropy-14539` | `astropy__astropy-14369` |
| `astropy__astropy-14995` | `astropy__astropy-14598` |

Agentic/spec summary from `agentic_summary.json`:

| Metric | Value |
|---|---:|
| spec_events | 86,355 |
| accepted_tokens | 261,362 |
| draft_tokens | 431,775 |
| accept_per_event | 3.026599502055469 |
| accept_per_draft | 0.6053199004110937 |
| steptrace generation_tokens | 347,914 |
| steptrace decode_tps | 39.90650600912407 |
| steptrace mean_gpu_util | 95.5345474022496 |
| steptrace request_decode_time_s | 23,575.919428933877 |
| steptrace request_prefill_time_s | 2,539.956661415752 |
| steptrace window_s | 8,718.227547168732 |
| tree_accept_paths | unavailable |

Capture parity and x86 evidence:

- `per_req_spec_trace.jsonl`: 86,475 lines.
- `dgx_steptrace.jsonl`: 49,909 lines.
- All 16 accepted runner metadata files report nonzero
  `vllm_request_metrics_bytes`, ranging from 78,827 to 275,281 bytes.
- Runner/eval metadata reports `eval_host=mark-Alienware-Aurora-ACT1250` and
  `arch=x86_64` for every accepted task checked locally and by read-only SSH.
- The runner metadata records the Codex container names for all accepted tasks.
  The campaign summary and most task eval reports record
  `model_id=qwen3.6-27b-fp8::codex-cli-0.128.0::q36-a`; three task eval
  reports omit that field. The metadata does not record a separate x86 Codex
  binary path.
- `spines=1` has no `independent_winner_trace.jsonl`; that is expected because
  no multi-spine winner branch exists for this arm.

Accepted task commits:

`234c393d`, `b6ce92ef`, `b9131b27`, `7de1f6b6`, `731c0bcd`,
`7f5e661c`, `251b26ca`, `2eda1414`, `343a18ca`, `f7b5d156`,
`fc3206ed`, `c557aa7b`, `ec91eeb3`, `0ded18e1`, `c34bf75d`,
`83ac64c4`.

Summary commit for this accepted arm: `e83acde8`.

## Invalidated `spines=2` Evidence

No `spines=2` SWE result is accepted.

Strict `mtp=5`, `spines=2` at gpu memory util 0.90 failed before a campaign
because vLLM did not satisfy the required free-memory threshold. The lowmem088
salvage attempts below also failed the evidence rules.

`fr9_b4temp06_lowmem088_mtp5_s2_20260602T033500Z`:

- `driver.log` launched 16 tasks at `concurrency=4`.
- Local request-metrics files are 0 bytes for all eight task dirs inspected:
  first batch `12907`, `13033`, `13236`, `13398` and second batch `13453`,
  `13579`, `13977`, `14096`.
- Git contains three committed failed task rows for this contaminated tag:
  `fcf4ef55`, `9803cd9f`, `b55149a6`. They are not accepted evidence.
- `independent_winner_trace.jsonl` has 11 lines, but the SWE capture is
  contaminated, so winner trace content cannot make the arm valid.

`fr9_b4temp06_lowmem088_mtp5_s2_20260602T035600Z`:

- The first four x86 tasks had nonzero request metrics:
  `12907`, `13033`, `13236`, `13398` each had 34,204-byte
  `vllm_request_metrics.jsonl` files.
- Those first four failed tasks were committed as `0768d304`, `984d679f`,
  `59a796a8`, and `a1b3eccb`.
- The next batch `13453`, `13579`, `13977`, `14096` had 0-byte request metrics,
  triggering the guard; whole tag is invalid partial evidence.
- Commit `08efa312` records the invalid abort in `driver.log`.
- Commit `d2c24374` records the attempted self-copy fix in independent winner
  state sync.
- `independent_winner_trace.jsonl` has 3,504 lines, but this tag is not a valid
  SWE result because capture failed mid-campaign.

`fr9_b4temp06_lowmem088_mtp5_s2_20260602T041200Z`:

- `/tmp/fr9_b4temp06_lowmem088_mtp5_s2_20260602T041200Z_runner.log` records
  request-metrics smoke parity: `local_before=0 remote_before=0`, then
  `local_after=1378 remote_after=1378`.
- The same runner log aborts before commit when task
  `astropy__astropy-12907` finishes with 0-byte request metrics.
- Local and remote checks both show the first four SWE tasks
  `12907`, `13033`, `13236`, `13398` had 0-byte
  `vllm_request_metrics.jsonl` files.
- Capture therefore stayed smoke-only at 1378/1378; no SWE request metrics were
  accepted for this tag.
- Handoff evidence identifies the EngineCore root cause in
  `/tmp/lumo-l0c-fp8-cutlass-run30-logs/vllm_qwen3.6-27b.log` around lines
  329-368: `torch.AcceleratorError` / CUDA illegal memory access in
  `_lumo_ir_winner_update_states_after_model_execute` at
  `output_token_ids.detach().cpu().tolist()`. At this closeout, the local file
  at that exact path is 0 bytes, so the crash text is recorded as handoff/root
  cause evidence rather than locally re-quoted log evidence.
- `independent_winner_trace.jsonl` has 13 lines, but the tag is invalid.

`fr9_b4temp06_lowmem088_mtp5_s2_20260602T051155Z`:

- Launch settings matched the requested arm: B4/Fb, `row_mode=independent`,
  `mtp=5`, `spines=2`, `LUMO_GPU_MEMORY_UTILIZATION=0.88`, `temp=0.6`,
  SWE Verified `concprobe16`, `concurrency=4`, 1800 s agent/eval limits, and
  nsight off.
- Prelaunch gates were clean: local `main` matched `origin/main` at hardening
  commit `b9159786` before launch, `swe_infra` was restarted from the patched
  streamer script, and x86 identity was `mark-Alienware-Aurora-ACT1250` /
  `x86_64`.
- The hardened request-metrics smoke preflight passed before SWE launch:
  `local_before=0 remote_before=0`, then
  `local_after=1359 remote_after=1359`.
- The first four x86 tasks completed and had nonzero request metrics:
  `12907` = 1,464 bytes, `13033` = 7,320 bytes, `13236` = 46,861 bytes, and
  `13398` = 46,861 bytes. All four runner metadata files report
  `eval_host=mark-Alienware-Aurora-ACT1250` and `arch=x86_64`.
- The first four failed task artifacts were committed as `15d46fe6`,
  `53d70a9e`, `7a44dde5`, and `b47e4ac7`. The first and fourth per-task push
  attempts hung up, but later/report pushes carry the local commits on `main`;
  they remain partial invalid evidence, not accepted benchmark rows.
- The second batch completed with 0-byte request metrics for `13453`, `13579`,
  `13977`, and `14096`. The local hardened guard aborted the whole tag at
  `astropy__astropy-13579` with:
  `SWE task finished without nonzero vLLM request metrics`.
- No top-level or nested `campaign_summary.json`, campaign-level
  `predictions.jsonl`, or `agentic_summary.json` exists for this tag because
  the runner aborted before campaign finalization. Per-task eval
  `predictions.jsonl` files exist for the completed task dirs: 7 mirrored
  locally before abort and 8 present on the x86 runner.
- Mirrored trace artifacts before abort: `dgx_steptrace.jsonl` has 50,813
  lines, `per_req_spec_trace.jsonl` has 2,472 lines, and
  `independent_winner_trace.jsonl` has 1,307 lines. The tag is still invalid
  because request-metrics capture failed mid-campaign.
- After abort, four orphan remote Codex containers for the same invalid tag
  (`14182`, `14309`, `14365`, `14369`) were stopped. A follow-up x86 process
  check showed no active `run_swe_bench_q36_a.py`, Codex, eval worker, or
  `swe-codex-*` container.

## Invalidated June 1 Tags

The stale June 1 report summaries remain useful only as examples of why the
zero-metric guard was needed.

| Tag | Stale summary | Audit result |
|---|---|---|
| `fr9_b4temp06_mtp5_s1_20260601T230213Z` | 16 tasks, 1 resolved, 15 failed | invalid: all 16 per-task request-metrics files are 0 bytes. |
| `fr9_b4temp06_mtp5_s2_20260601T233000Z` | 16 tasks, 0 resolved, 16 failed | invalid: all 16 per-task request-metrics files are 0 bytes. |

Do not compare these stale counts against the accepted `lowmem088_mtp5_s1`
result.

## `mtp3/s2` Status

No valid `mtp=3`, `spines=2` campaign exists. The only documented attempts are
the old strict prelaunch failures:

- `fr9_b4temp06_mtp3_s2_20260601T234537Z`
- `fr9_b4temp06_mtp3_s2_20260601T234919Z`
- `fr9_b4temp06_mtp3_s2_20260601T235241Z`

The stale report recorded those as vLLM prelaunch memory failures and no local
`output/fr9_b4temp06*mtp3*s2*` directory exists. No lowmem `mtp3/s2` campaign
was launched for this report.

## Honesty Audit

- Zero-metric guard commits are on `main`:
  `9b2ddb67` (`Abort SWE commit on missing request metrics`) and `0ce9cf4e`
  (`Stop SWE campaign on missing request metrics`).
- The report did not launch any new SWE/vLLM campaign. It used read-only
  inspection of git history, local artifacts, `/tmp` logs, process state, and
  remote x86 metadata over SSH.
- A process check for active `python ... scripts/run_codex_experiment.py`,
  `run_swe_bench_q36_a.py`, `vllm serve`, `EngineCore`, and Docker SWE worker
  commands found no actual active runner process after excluding the current
  reporting Codex prompt and an existing OOM guard shell.
- Existing tmux/OOM-guard infrastructure was observed but not started,
  restarted, or modified.
- The local `qwen3.6` vLLM crash log path named in the handoff is currently
  0 bytes, so this report does not pretend to have re-read the crash traceback
  from that file. The invalidation does not depend on the traceback: zero SWE
  request metrics are sufficient to reject the `spines=2` tags.

## Closeout

Accepted evidence is limited to
`fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z`: `mtp=5`, `spines=1`,
lowmem088, B4/Fb, independent row mode, temp 0.6, SWE Verified 16 on the x86
box, 8/16 resolved.

There is no accepted `spines=2` SWE result for FR9 B4/Fb temp 0.6, and there is
no valid `mtp3/s2` campaign. Enhanced tree work should not consume any of the
invalidated June 1 or lowmem `spines=2` tags as baseline evidence.
