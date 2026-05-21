# SWE-Bench Q36-A Campaign — Live Progress Log

**Started:** 2026-05-21
**Pre-registration:** [`swe-bench-q36-a-campaign-pre-registration-20260520.md`](swe-bench-q36-a-campaign-pre-registration-20260520.md)
**Per-task agent budget:** 1800s (30 min)
**Concurrency:** 1 (forced by 7.9 GB MemAvailable on 117 GB unified DGX Spark host)
**Eval namespace:** auto (Docker Hub prebuilt arm64 → local build fallback)
**No grading:** verdicts are reported as the harness emits them; pass-rate aggregates appear in the closeout, not here.

This document is updated by the cron supervisor (`07385536`, every 20 min)
after each instance completes. Per-task artifacts (patch.diff, codex_trace.jsonl,
vllm_request_metrics.jsonl, eval/) live under `output/swe_bench_q36_a_temp06/`
and are gitignored.

---

## Per-instance verdicts — Verified

| Instance | Repo | Verdict | Failure mode | Codex s | Eval s | Patch bytes | Namespace | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| `astropy__astropy-14508` | astropy/astropy | resolved | tests_passed | 1007.4 | 89.1 | 1907 | none (local build) | codex_rc=1 (turn.failed on final turn after 130 OK turns — BadRequestError JSON parse; agent recovered enough to ship a passing patch) |
| `django__django-11119` | django/django | resolved | tests_passed | 1800.2 | 119.8 | 485 | none (local build) | codex hit 30-min wall (timed_out=True, rc=-1); patch produced before kill resolves the task |
| `django__django-12754` | django/django | resolved | tests_passed | 1800.2 | 124.6 | 1364 | none (local build) | codex hit 30-min wall; agent produced a 1364-byte patch that resolves the task |
| `django__django-13741` | django/django | failed | tests_failed | 296.8 | 111.4 | 446 | swebench (prebuilt) | agent self-stopped clean (rc=0) at 5 min thinking the task was done; 446B patch applies but tests don't pass |

## Per-instance verdicts — Pro

_(none yet)_

---

## Hardware-resource hygiene log

| Timestamp UTC | MemAvailable (GiB) | swap_used | Note |
|---|---:|---|---|
| 2026-05-21T06:43Z | 7.8 | 6.4/15 GiB | Verified Tier 0 launched |
| 2026-05-21T07:00Z | 7.8 | 6.4/15 GiB | astropy-14508 eval in progress |
| 2026-05-21T07:02Z | — | — | astropy-14508 resolved (instance 1/20) |
| 2026-05-21T07:20Z | 7.9 | 6.4/15 GiB | django-11119 codex agent at ~18 min into 30-min budget |
| 2026-05-21T07:34Z | — | — | django-11119 resolved (instance 2/20); Codex hit 30-min wall, watchdog killed cleanly |
| 2026-05-21T08:00Z | 7.9 | 6.4/15 GiB | django-12754 codex agent at ~26 min into 30-min budget |
| 2026-05-21T08:06Z | — | — | django-12754 resolved (instance 3/20) |
| 2026-05-21T08:13Z | — | — | django-13741 failed/tests_failed (instance 4/20); agent self-stopped at 5 min |
