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

## Per-instance verdicts — Pro

_(none yet)_

---

## Hardware-resource hygiene log

| Timestamp UTC | MemAvailable (GiB) | swap_used | Note |
|---|---:|---|---|
| 2026-05-21T06:43Z | 7.8 | 6.4/15 GiB | Verified Tier 0 launched |
| 2026-05-21T07:00Z | 7.8 | 6.4/15 GiB | astropy-14508 eval in progress |
| 2026-05-21T07:02Z | — | — | astropy-14508 resolved (instance 1/20) |
