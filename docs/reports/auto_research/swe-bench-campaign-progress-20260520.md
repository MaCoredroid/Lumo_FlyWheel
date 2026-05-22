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
| `scikit-learn__scikit-learn-13496` | scikit-learn/scikit-learn | crash | infra_error | 1800.2 | 141.9 | 1383 | none (local build attempt) | **ARM64 UNSUPPORTED.** Env image build fails: scipy can't be built from source on aarch64 (no Fortran compiler in base image, no prebuilt arm64 wheel for the pinned old scipy version). Agent shipped a 1383B patch but it can't be evaluated on this host. |
| `sphinx-doc__sphinx-7440` | sphinx-doc/sphinx | resolved | tests_passed | 1800.2 | 58.4 | 2608 | swebench (prebuilt) | agent hit 30-min wall; 2608B patch passes |
| `sphinx-doc__sphinx-9230` | sphinx-doc/sphinx | resolved | tests_passed | 1352.0 | 57.0 | 541 | swebench (prebuilt) | agent self-stopped clean (rc=0) at 22.5 min; 541B patch passes |
| `sympy__sympy-13757` | sympy/sympy | failed | patch_apply_failed | 1651.4 | 0 | 0 | n/a | agent ran 60 tool calls clean (rc=0, turn.completed) but produced an empty diff. Real "agent gave up" failure — NOT a proxy flake (no error events in trace). Not in re-run queue. |
| `sympy__sympy-13974` | sympy/sympy | failed | patch_apply_failed | 1410.6 | 0 | 0 | n/a | agent ran 23.5 min clean (rc=0) but produced an empty diff. Same shape as sympy-13757 — agent gave up. Not in re-run queue. |
| `sympy__sympy-17630` | sympy/sympy | failed | tests_failed | 1800.1 | 61.1 | 1244 | swebench (prebuilt) | agent hit 30-min wall; 1244B patch applies but tests fail |

## Tier 0 closeout

- **20/20 instances complete** in 7h 49min wall-clock (06:43:36Z → 14:32:25Z).
- **Resolved: 10 (raw 50%)** — astropy-14508, django-11119/-12754/-13809/-14373, matplotlib-24637, pylint-6528, pytest-8399, sphinx-7440/-9230.
- **Failed: 8** — django-13741, django-14170, django-16100, django-17084, sympy-13757, sympy-13974, sympy-17630 (tests_failed/agent-gave-up), django-16256 (proxy flake — re-run queued).
- **ARM64-unsupported: 2** — xarray-6721 (cdms2), sklearn-13496 (scipy fortran).
- **Pre-reg G0 gate (≥4/20 = 20%): MET** (10/20 = 50%, 10x the gate threshold).
- **Adjusted pass-rate after carve-out**: 10/18 = 55.6% (if the flake re-runs successfully, 11/18 = 61.1%).
- **Per-repo**: 9-django at 4/9 (44%), 3-sympy at 0/3, 2-sphinx at 2/2, single repos (astropy/matplotlib/pylint/pytest) all 1/1.

Per-instance medians: codex_s 1800 (most hit the 30-min wall), eval_s 74s, total 22 min/instance. No memory leak observed across 8 hours of continuous execution (host MemAvailable stable at 7.8-8.0 GiB).

## Campaign reset: pre-fix vs post-fix (2026-05-21T20:20Z)

On 2026-05-21T17:25Z the proxy was patched (commit `d458d794`) so that upstream
4xx responses now propagate as JSON instead of being wrapped in synthetic SSE
`response.created` events. The 25 instances completed before that timestamp
(20 Tier 0 + 5 Tier 2 astropy) carry the pre-fix bias: each was exposed to the
mid-run proxy JSON crash that destroyed django-16256 with zero patch, plus
they ran with the 100Hz DCGM sampling and the 5-retry auto-continue.

To produce a clean homogeneous dataset, all 24 of those pre-fix instance
directories (django-16256 omitted — running in a parallel side slot at the
time) were deleted at 2026-05-21T20:18Z and the orchestrator was restarted.
The 5 post-fix Tier 2 instances (`astropy-13579`, `-13977`, `-14096`, `-14182`,
`-14309`) were kept; they already ran under the post-fix stack.

The verdict tables above are preserved as historical record. The
post-restart sweep below is the campaign of record for the closeout.

### New failure-mode classification (added 2026-05-21)

In addition to the harness's four `failure_mode` values
(`tests_passed`, `tests_failed`, `patch_apply_failed`, `infra_error`),
we tag two campaign-level sub-modes when the orchestrator's
`patch_apply_failed` row carries `patch_bytes == 0`:

- **`empty_diff__agent_gave_up`**: Codex exit clean (`rc == 0`),
  `turn.completed` in trace, NO error events. Agent finished without
  emitting a patch. Examples in pre-fix data: sympy-13757, sympy-13974.
- **`empty_diff__setup_loop`**: Codex used the full 30-min wall in
  what turned out to be a setup-fight loop (e.g. astropy-14096 spent
  30 min on `pip install setuptools` permutations, hitting the same
  `setuptools.dep_util` import error each retry). Wall-budget extension
  would NOT help — agent makes the same wrong attempt repeatedly.
- **`empty_diff__proxy_flake`** (legacy, eliminated): the proxy-fix
  class. Will not recur post-2026-05-21T17:25Z.

These sub-modes are reported in addition to the harness verdict and are
counted in the campaign closeout's pass-rate denominator-adjustment section.

## Tier 2 verdicts (incremental — Tier 0 reused via --skip-existing)

| Instance | Repo | Verdict | Failure mode | Codex s | Eval s | Patch bytes | Namespace | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| `astropy__astropy-12907` | astropy/astropy | resolved | tests_passed | 1800.2 | 57.6 | 2208 | swebench (prebuilt) | first Tier 2 instance; hit 30-min wall; 2208B patch passes |
| `astropy__astropy-13033` | astropy/astropy | failed | tests_failed | 1800.1 | 114.4 | 1091 | none (local build) | hit 30-min wall; 1091B patch applies but tests fail |
| `astropy__astropy-13236` | astropy/astropy | failed | tests_failed | 1800.2 | 74.3 | 1617 | swebench (prebuilt) | hit 30-min wall; 1617B patch applies but tests fail |
| `astropy__astropy-13398` | astropy/astropy | failed | tests_failed | 1800.2 | 74.0 | 580 | swebench (prebuilt) | hit 30-min wall; 580B patch applies but tests fail |
| `astropy__astropy-13453` | astropy/astropy | resolved | tests_passed | 1800.2 | 111.9 | 427 | none (local build) | hit 30-min wall; 427B patch passes |
| `astropy__astropy-13579` | astropy/astropy | failed | patch_apply_failed | 1128.8 | 0 | 0 | n/a | first instance after proxy fix + full artifact bundle. Agent ran 40 tool calls clean (rc=0) but spent 19 min on setup (`pip install setuptools>=68.0`) and never wrote a fix — agent strategy failure, NOT a proxy/JSON flake. NOT in re-run queue. |
| `astropy__astropy-13977` | astropy/astropy | failed | tests_failed | 1713.8 | 113.5 | 1462 | none (local build) | codex_s=28.6 min (close to wall), rc=1 (likely late turn.failed but patch already shipped — astropy-14508 pattern); 1462B patch applies but tests fail |
| `astropy__astropy-14096` | astropy/astropy | failed | patch_apply_failed | 1800.2 | 0 | 0 | n/a | first instance with 10 Hz dcgm (5.9 MB samples, 10× under 100Hz). Codex hit 30-min wall, watchdog kill → no patch shipped. Astropy continues to be hard for Q36-A. |
| `astropy__astropy-14182` | astropy/astropy | failed | tests_failed | 1800.2 | 112.9 | 882 | none (local build) | hit 30-min wall; 882B patch applies but tests fail |
| `astropy__astropy-14309` | astropy/astropy | resolved | tests_passed | 1800.1 | 60.9 | 1689 | swebench (prebuilt) | hit 30-min wall; 1689B patch passes — first astropy resolve in Tier 2 |
| `astropy__astropy-12907` | astropy/astropy | resolved | tests_passed | 1530.1 | 52.3 | 504 | swebench (prebuilt) | (clean post-fix re-run after pre-fix sweep deleted). codex_rc=1 (late turn.failed, patch already shipped). 504B patch passes. |
| `astropy__astropy-13033` | astropy/astropy | failed | tests_failed | 1800.2 | 111.3 | 1233 | none (local build) | (post-fix re-run; pre-fix also failed). hit 30-min wall; 1233B patch applies but tests fail |
| `astropy__astropy-13236` | astropy/astropy | resolved | tests_passed | 1800.2 | 76.4 | 819 | swebench (prebuilt) | **FLIPPED**: pre-fix=failed (1617B patch tests_failed) → post-fix=resolved (819B patch). First verdict flip; smaller patch passes — proxy fix + tightened auto-continue likely steering agent toward cleaner fixes. |
| `astropy__astropy-13398` | astropy/astropy | failed | tests_failed | 1800.2 | 77.5 | 580 | swebench (prebuilt) | (post-fix re-run; pre-fix also failed with identical 580B patch — same fix attempted, same fail). |
| `astropy__astropy-13453` | astropy/astropy | resolved | tests_passed | 1801.1 | 112.9 | 551 | none (local build) | (post-fix re-run; pre-fix also resolved). hit wall; 551B patch passes. **MEM SPIKE during this eval**: MemAvailable dipped 7.8→0.6 GiB during env-build+pytest, absorbed by swap (no OOM), recovered to 10 GiB after eval container exited. |

### Re-run summary (pre-fix astropy cells, 5 of 24 done)

| Instance | Pre-fix | Post-fix | Change |
|---|---|---|---|
| astropy-12907 | resolved | resolved | same |
| astropy-13033 | failed | failed | same |
| astropy-13236 | failed | **resolved** | **FLIP +1** |
| astropy-13398 | failed | failed | same |
| astropy-13453 | resolved | resolved | same |

Net post-fix change on astropy re-runs so far: **+1 resolved** (13236 flip).
Astropy post-fix sweep standing: 3/5 resolved (vs pre-fix 2/5).

### KEY FINDING: ARM64 carve-out is Python-version-driven, not repo-driven

`django-10097` (an old django needing `python=3.5*`) crashed with the same
conda PackagesNotFoundError. **The arm64 build failures are not specific to
scientific repos — they affect ANY SWE-Bench instance whose environment pins
Python ≤3.5/3.6 or old deps that conda-forge never built for linux-aarch64.**
SWE-Bench Verified spans ~2018-2023; the older cohort (django 2.x, astropy
3.x, etc.) will carve out broadly on this aarch64 host. Modern instances
(python ≥3.8) build clean. Expect the full-benchmark arm64 carve-out to be
material (rough early estimate 20-35%); the closeout MUST report an
arm64-evaluable denominator separately from the headline. An x86-offload or
emulation path is the only way to recover these cells.

### Fresh instances (never run pre-fix)

| Instance | Repo | Verdict | Failure mode | Codex s | Eval s | Patch B | Namespace | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| `astropy__astropy-14365` | astropy/astropy | failed | tests_failed | 1758.9 | 93.5 | 617 | none (local build) | self-stopped (rc=1) ~29 min; 617B patch applies but tests fail |
| `django__django-10097` | django/django | crash | infra_error | 1800.2 | 42.8 | 596 | none (local build attempt) | **ARM64 UNSUPPORTED** (8th). Needs `python=3.5*` — no conda linux-aarch64 build. First non-scientific-repo carve-out; confirms the issue is Python-version-driven. **RECOVERED via x86 offload -> failed.** |

## ARM64 carve-out RESOLVED — x86 eval-offload (2026-05-22)

The carve-out is fixed by offloading the **eval step** to a native x86_64
box (alienware) over SSH. Agent runs on the DGX Spark (Q36-A vLLM); eval
runs on x86 where every instance's conda env builds. Infra: scripts/
swe_eval_x86_worker.py (remote) + scripts/swe_eval_offload.py (driver) +
orchestrator `--eval-host`. Network-tolerant (backoff retries, keepalives,
network-issue log), docker relocated to alienware /home (85 GB).

**Validity (answers the contamination question): 17/17 buildable instances
agree between arm64-native and x86 eval — 0 flips.** The arm64 verdicts we
already had were correct. The 6 arm64-crashes were *blocked*, not
contaminating; all recovered on x86 (2 -> resolved, 4 -> failed). The 23
on-disk instances are now x86-authoritative single-arch; runner_metadata
keeps eval_report_arm64 for the comparison.

From 2026-05-22T07:19Z the resumed Verified sweep runs eval on x86 for
every fresh instance (single-arch, no carve-outs, and the DGX eval-phase
memory spikes are gone since eval is remote).

### Fresh instances — x86 eval-offload (resumed sweep)

| Instance | Repo | Verdict | Failure mode | Codex s | Eval s | Patch B | Eval host | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| `django__django-10554` | django/django | failed | patch_apply_failed | 1800.1 | 1.1 | 0 | alienware (x86) | first resumed instance; agent hit 30-min wall, empty patch; eval offloaded + fast-pathed (arch=x86_64 confirmed) |
| `django__django-10880` | django/django | resolved | tests_passed | 1800.1 | 64.7 | 639 | alienware (x86) | first RESOLVED via offload; native x86 django eval 65s (vs >18 min emulated); 639B patch passes |
| `django__django-10914` | django/django | resolved | tests_passed | 1800.1 | 67.1 | 625 | alienware (x86) | hit wall; 625B patch passes |
| `django__django-10973` | django/django | resolved | tests_passed | 1800.1 | 64.9 | 2451 | alienware (x86) | hit wall; 2451B patch passes |
| `django__django-10999` | django/django | failed | tests_failed | 975.8 | 65.1 | 497 | alienware (x86) | self-stop ~16 min; 497B patch applies but tests fail |
| `django__django-11066` | django/django | failed | patch_apply_failed | 38.3 | 1.0 | 0 | alienware (x86) | **`flake__proxy_json` (re-run candidate).** Early-turn `BadRequestError: Unterminated string at column 149` + `unsupported call: apply_patch` (rc=1, empty patch, 38s). Same proxy/vLLM malformed-tool-call class as django-16256; the 4xx-passthrough fix made the error honest but didn't eliminate it. NOT a real capability failure. |
| `astropy__astropy-14369` | astropy/astropy | failed | patch_apply_failed | 1481.7 | 0 | 0 | n/a | `empty_diff__agent_gave_up`: clean rc=0 at 24.7 min, no patch emitted |
| `astropy__astropy-14508` | astropy/astropy | resolved | tests_passed | 1800.2 | 91.9 | 2830 | none (local build) | (was Tier 0's resolved instance; consistent re-run). hit wall; 2830B patch passes |
| `astropy__astropy-14539` | astropy/astropy | failed | patch_apply_failed | 847.5 | 0 | 0 | n/a | `empty_diff__agent_gave_up`: clean rc=0 at 14 min, no patch emitted |
| `astropy__astropy-14598` | astropy/astropy | failed | tests_failed | 1488.5 | 88.6 | 95 | none (local build) | self-stop clean (rc=0) ~25 min; tiny 95B patch applies but tests fail |
| `astropy__astropy-14995` | astropy/astropy | resolved | tests_passed | 1794.5 | 65.5 | 669 | swebench (prebuilt) | rc=1 late, patch already shipped; 669B patch passes |
| `astropy__astropy-7166` | astropy/astropy | crash | infra_error | 1800.2 | 29.7 | 3998 | none (local build attempt) | **ARM64 UNSUPPORTED** (3rd). Env build fails: conda has no `setuptools==38.2.4` for linux-aarch64 (ancient pin in old astropy). Agent shipped 3998B patch, uneval-able. Carve out. |
| `astropy__astropy-7336` | astropy/astropy | crash | infra_error | 1800.2 | 40.5 | 6375 | none (local build attempt) | **ARM64 UNSUPPORTED** (4th). Same `setuptools==38.2.4` arm64 gap as -7166. 6375B patch uneval-able. Carve out. |
| `astropy__astropy-7606` | astropy/astropy | crash | infra_error | 972.1 | 28.5 | 7428 | none (local build attempt) | **ARM64 UNSUPPORTED** (5th). Same `setuptools==38.2.4` gap. 7428B patch uneval-able. Carve out. |
| `astropy__astropy-7671` | astropy/astropy | crash | infra_error | 1800.2 | 37.0 | 3239 | none (local build attempt) | **ARM64 UNSUPPORTED** (6th). Same old-astropy conda gap. 3239B patch uneval-able. Carve out. |
| `astropy__astropy-8707` | astropy/astropy | failed | patch_apply_failed | 625.3 | 0 | 0 | n/a | **`flake__token_truncation`** (NEW sub-mode, re-run candidate). Codex `turn.failed` rc=1: model emitted an ~80 KB single tool-call argument that hit the `LUMO_PROXY_MAX_OUTPUT_TOKENS=32768` cap and truncated mid-string → `EOF parsing function arguments at column 81828` → unparseable → BadRequestError. Distinct from the django-16256 SSE-wrap flake (fixed). Proxy fix made the error honest (`type:error` not fake `response.created`) but the truncation root cause is separate. |

| `astropy__astropy-8872` | astropy/astropy | crash | infra_error | 1800.2 | 22.5 | 1295 | none (local build attempt) | **ARM64 UNSUPPORTED** (7th). Build fails: `ModuleNotFoundError: No module named 'pkg_resources'` (setuptools-era arm64 build incompat). 1295B patch uneval-able. Carve out. |

### NEW flake sub-mode: token truncation of large tool calls

`flake__token_truncation` — when the model emits a single tool-call argument
larger than ~80 KB (≈ the 32768-token output cap × ~2.5 chars/token), the
JSON serialization truncates mid-string and becomes unparseable. Codex sees
`failed to parse function arguments: EOF ... column ~81828` and crashes the
turn with an empty patch. Typically a huge `apply_patch`/file-write the model
tried to emit in one shot.

- Root cause: `LUMO_PROXY_MAX_OUTPUT_TOKENS=32768` truncates large tool calls.
- NOT fixed by the d458d794 4xx-passthrough patch (that addressed SSE-wrapping
  of upstream errors; this is a model-output-boundary truncation).
- Candidate fixes (deferred, not applied mid-campaign): (a) raise the output
  cap; (b) detect truncated tool-call JSON in the proxy and auto-retry with a
  "split your edit into smaller chunks" continue message; (c) accept as a real
  capability limit of the harness on instances needing huge single edits.
- Re-run queue: these instances are tagged for re-run after a proxy fix lands.

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
| 2026-05-21T12:00Z | 7.9 | 6.5/15 GiB | sklearn-13496 codex agent at ~17 min |
| 2026-05-21T12:15Z | — | — | sklearn-13496 crash/arm64-unsupported (instance 15/20); scipy fortran build failure — second ARM64 carve-out |
| 2026-05-21T12:20Z | 7.9 | 6.5/15 GiB | sphinx-7440 codex agent at ~5 min |
| 2026-05-21T12:46Z | — | — | sphinx-7440 resolved (instance 16/20) |
| 2026-05-21T13:00Z | 7.9 | 6.5/15 GiB | sphinx-9230 codex agent at ~14 min; 20 ticks of memory stability → no leak proven |
| 2026-05-21T13:10Z | — | — | sphinx-9230 resolved (instance 17/20) |
| 2026-05-21T13:20Z | 7.9 | 6.5/15 GiB | sympy-13757 codex agent at ~10 min |
| 2026-05-21T13:37Z | — | — | sympy-13757 failed/empty-diff (instance 18/20); agent ran 27 min clean but produced no patch — NOT a flake |
| 2026-05-21T14:00Z | 7.9 | 6.5/15 GiB | sympy-13974 codex agent at ~22 min |
| 2026-05-21T14:01Z | — | — | sympy-13974 failed/empty-diff (instance 19/20); same shape as sympy-13757 |
| 2026-05-21T14:20Z | 7.9 | 6.5/15 GiB | sympy-17630 codex agent at ~19 min |
| 2026-05-21T14:32Z | — | — | sympy-17630 failed/tests_failed (instance 20/20); **TIER 0 COMPLETE** |
| 2026-05-21T14:33Z | — | — | Verified Tier 2 (full 500) launched with --skip-existing (480 remaining) |
| 2026-05-21T15:00Z | 7.9 | 6.5/15 GiB | astropy-12907 codex agent at ~26 min |
| 2026-05-21T15:04Z | — | — | astropy-12907 resolved (Tier 2 instance 1/480 = overall 21/500) |

## Re-run queue (flake recovery)

Instances tagged for re-run after Tier 0 completes (per pre-reg §5 "interrupted" policy):

| Instance | Reason | First-attempt artifacts kept at |
|---|---|---|
| `django__django-16256` | Proxy/vLLM `BadRequestError: Unterminated string at column 89` on turn 3 — empty patch (SSE-wrap flake, fixed by d458d794; re-collected naturally in the post-fix sweep) | `output/.../django__django-16256/` |
| `astropy__astropy-8707` | `flake__token_truncation`: 80 KB tool-call arg hit the 32768-token cap, truncated mid-string. Needs a token-cap or split-edit fix before re-run. | `output/.../astropy__astropy-8707/` |

## ARM64-unsupported queue (denominator carve-out)

Instances whose SWE-Bench env image cannot be built on linux-aarch64. Per pre-reg §7
these are logged as infra_error and carved out of the pass-rate denominator in the
closeout. Re-running cannot fix them — needs x86 emulation or an x86 host.

| Instance | Reason |
|---|---|
| `pydata__xarray-6721` | conda-forge has no `linux-aarch64` build of `cdms2` |
| `scikit-learn__scikit-learn-13496` | scipy source build needs Fortran compiler (absent in SWE-Bench base); no arm64 wheel for pinned old scipy |
| `astropy__astropy-7166` | conda has no `setuptools==38.2.4` for linux-aarch64 (ancient pin in old astropy version) |
| `astropy__astropy-7336` | same `setuptools==38.2.4` arm64 gap as -7166 |
| `astropy__astropy-7606` | same `setuptools==38.2.4` arm64 gap |
| `astropy__astropy-7671` | same old-astropy conda arm64 gap |
| `astropy__astropy-8872` | `pkg_resources` missing during arm64 build (setuptools-era incompat) |
