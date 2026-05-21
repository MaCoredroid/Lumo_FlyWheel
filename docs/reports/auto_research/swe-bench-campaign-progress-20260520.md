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
| `django__django-13809` | django/django | resolved | tests_passed | 724.2 | 96.2 | 1841 | swebench (prebuilt) | agent self-stopped clean (rc=0) at 12 min; 1841B patch passes |
| `django__django-14170` | django/django | failed | tests_failed | 1800.2 | 69.4 | 581 | swebench (prebuilt) | agent hit 30-min wall; 581B patch applies but tests fail |
| `django__django-14373` | django/django | resolved | tests_passed | 346.8 | 135.9 | 418 | none (local build) | agent self-stopped clean (rc=0) at 6 min; 418B patch passes |
| `django__django-16100` | django/django | failed | tests_failed | 893.5 | 73.9 | 3761 | swebench (prebuilt) | agent self-stopped clean (rc=0) at 15 min; large 3761B patch applies but tests fail |
| `django__django-16256` | django/django | failed | patch_apply_failed | 31.8 | 0 | 0 | n/a | **FLAKE — re-run candidate.** Proxy/vLLM emitted `BadRequestError: Unterminated string at column 89` on turn 3 (2nd recurrence; first was astropy-14508 final turn). Codex CLI crashed rc=1, no patch produced. Same root cause as the prior incident; not representative of agent capability. |
| `django__django-17084` | django/django | failed | tests_failed | 1800.1 | 75.9 | 1017 | swebench (prebuilt) | agent hit 30-min wall; 1017B patch applies but tests fail |
| `matplotlib__matplotlib-24637` | matplotlib/matplotlib | resolved | tests_passed | 1800.2 | 465.3 | 808 | none (local build) | agent hit 30-min wall; eval_s=465 includes first-time matplotlib env image build (~7-8 min); 808B patch passes |
| `pydata__xarray-6721` | pydata/xarray | crash | infra_error | 1800.2 | 63.5 | 484 | none (local build attempt) | **ARM64 UNSUPPORTED.** Env image build fails: `cdms2` conda package has no `linux-aarch64` build. Codex agent produced a 484B patch but it can never be evaluated on this host. Pre-reg §7 decision: log as infra_error, no fallback budget; carve out of pass-rate denominator. |
| `pylint-dev__pylint-6528` | pylint-dev/pylint | resolved | tests_passed | 1800.2 | 62.7 | 3174 | swebench (prebuilt) | agent hit 30-min wall; 3174B patch passes |
| `pytest-dev__pytest-8399` | pytest-dev/pytest | resolved | tests_passed | 399.6 | 72.0 | 2460 | none (local build) | codex_rc=1 (turn.failed late on proxy JSON; agent had already shipped a 2460B patch that passes); fast self-stop at 7 min |

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
| 2026-05-21T08:20Z | 7.9 | 6.4/15 GiB | django-13809 codex agent at ~7 min (~4 instances completed, memory stable — no leak) |
| 2026-05-21T08:27Z | — | — | django-13809 resolved (instance 5/20); fast self-stop at 12 min |
| 2026-05-21T08:40Z | 7.8 | 6.4/15 GiB | django-14170 at ~13 min; memory ticked down 0.1 GiB (within noise) |
| 2026-05-21T08:58Z | — | — | django-14170 failed/tests_failed (instance 6/20) |
| 2026-05-21T09:00Z | 7.8 | 6.4/15 GiB | django-14373 codex agent at ~2 min |
| 2026-05-21T09:06Z | — | — | django-14373 resolved (instance 7/20); fast self-stop at 6 min |
| 2026-05-21T09:20Z | 7.8 | 6.4/15 GiB | django-16100 codex agent at ~14 min |
| 2026-05-21T09:22Z | — | — | django-16100 failed/tests_failed (instance 8/20); large 3.7K patch applied but tests fail |
| 2026-05-21T09:23Z | — | — | django-16256 FLAKE (instance 9/20); proxy JSON BadRequestError on turn 3 → empty patch (2nd recurrence) |
| 2026-05-21T09:40Z | 7.8 | 6.4/15 GiB | django-17084 codex agent at ~17 min |
| 2026-05-21T09:54Z | — | — | django-17084 failed/tests_failed (instance 10/20); end of the 9-django block |
| 2026-05-21T10:00Z | 7.4 | 6.4/15 GiB | matplotlib-24637 at ~5 min; memory ticked down 0.4 GiB (transient — repo clone + image build using page cache) |
| 2026-05-21T10:20Z | 7.8 | 6.4/15 GiB | matplotlib-24637 at ~25 min; memory **recovered** to 7.8 — no leak |
| 2026-05-21T10:32Z | — | — | matplotlib-24637 resolved (instance 11/20); first new repo since django, env image cached for future |
| 2026-05-21T10:40Z | 8.0 | 6.5/15 GiB | xarray-6721 codex agent at ~7 min |
| 2026-05-21T11:00Z | 7.9 | 6.5/15 GiB | xarray-6721 codex agent at ~27 min |
| 2026-05-21T11:03Z | — | — | xarray-6721 crash/infra_error (instance 12/20); cdms2 unavailable on linux-aarch64 — first ARM64-unsupported instance |
| 2026-05-21T11:20Z | 8.0 | 6.5/15 GiB | pylint-6528 codex agent at ~16 min |
| 2026-05-21T11:34Z | — | — | pylint-6528 resolved (instance 13/20) |
| 2026-05-21T11:40Z | 7.9 | 6.5/15 GiB | pytest-8399 codex agent at ~5 min |
| 2026-05-21T11:42Z | — | — | pytest-8399 resolved (instance 14/20); fast self-stop ~7 min |

## Re-run queue (flake recovery)

Instances tagged for re-run after Tier 0 completes (per pre-reg §5 "interrupted" policy):

| Instance | Reason | First-attempt artifacts kept at |
|---|---|---|
| `django__django-16256` | Proxy/vLLM `BadRequestError: Unterminated string at column 89` on turn 3 — empty patch | `output/swe_bench_q36_a_temp06/verified/per_task/django__django-16256/` |

## ARM64-unsupported queue (denominator carve-out)

Instances whose SWE-Bench env image cannot be built on linux-aarch64. Per pre-reg §7
these are logged as infra_error and carved out of the pass-rate denominator in the
closeout. Re-running cannot fix them — needs x86 emulation or an x86 host.

| Instance | Reason |
|---|---|
| `pydata__xarray-6721` | conda-forge has no `linux-aarch64` build of `cdms2` |
