# SWE-Bench Q36-A Campaign Pre-Registration

**Generated:** 2026-05-21 (effective 2026-05-20 spec)
**Spec:** [`swe-bench-bounded-time-spec-20260520.md`](swe-bench-bounded-time-spec-20260520.md)
**Anchor product:** Q36-A on Codex CLI 0.128.0 (Track B shipping config)
**Status:** Pre-registered — committed before any measurements are taken.

---

## 1. Why this document

Spec §3 requires gates, stratification, and per-task budgets to be locked
in writing before reading any data. This document is that lock. Any
change after a tier launches is a tracked deviation that must be called
out in the closeout.

## 2. System under test

| Component | Configuration |
|---|---|
| Model | `Qwen/Qwen3.6-27B-FP8` served as `qwen3.6-27b` |
| Inference | vLLM 0.19 + Arctic Inference 0.1.2, FP8 weights, KV cache `auto` |
| Spec-decode (point A) | `method=suffix, num_speculative_tokens=12, suffix_decoding_max_tree_depth=32, suffix_decoding_max_spec_factor=2.0, suffix_decoding_min_token_prob=0.05, suffix_decoding_max_cached_requests=1000, rejection_sample_method=probabilistic` |
| Sampling | temp=0.6, top_p=0.95 (Qwen 3.6 precise-coding rec) |
| Tool / reasoning parsers | `--tool-call-parser qwen3_xml --reasoning-parser qwen3` |
| Agent | Codex CLI 0.128.0, `model_reasoning_effort="high"` |
| Endpoint | `http://127.0.0.1:8022/v1` (codex-bench-proxy) |
| Bundle file | `/tmp/lumo-track-b-bundle-qwen36/bundle.yaml` |
| Relaunch driver | `/tmp/relaunch_qwen36_AD.py` (A and D share this bundle; D adds T2+T3+T4 runtime flags) |
| Container | `lumo-vllm-track-b-suffix` |
| Host | NVIDIA GB10 (DGX Spark, sm_120, aarch64) |
| Wall-clock per-task budget | 25 min Codex agent + 5 min eval buffer (spec §6) |

The vLLM instance was already up and serving with this exact
configuration as of 2026-05-21 (init log timestamp 2026-05-19 16:42:31
UTC); no relaunch is required prior to Tier 0.

## 3. Subset pre-registration

| Tier | Dataset | Subset | Manifest |
|---|---|---:|---|
| 0 | Verified | 20 | [`swe-bench-tier0-verified-instances-20260520.md`](swe-bench-tier0-verified-instances-20260520.md) |
| 1 | Verified | 100 | [`swe-bench-tier1-verified-instances-20260520.md`](swe-bench-tier1-verified-instances-20260520.md) |
| 1 | Pro | 100 | [`swe-bench-tier1-pro-instances-20260520.md`](swe-bench-tier1-pro-instances-20260520.md) |
| 2 | Verified | 500 (full) | n/a — full benchmark |
| 2 | Pro | 731 (full) | n/a — full benchmark |

Stratification: proportional by repo, largest-remainder rounding,
seed=0, deterministic per-repo `random.sample`. The spec §5
illustrative table is superseded by the actual dataset distributions at
the pinned commits (Verified commit `c104f840`, Pro commit recorded by
HF resolver at fetch time on 2026-05-21).

### 3.1 Actual repo distributions

**Verified** (500): django/django 231 (46%), sympy/sympy 75, sphinx-doc/sphinx 44, matplotlib/matplotlib 34, scikit-learn/scikit-learn 32, astropy/astropy 22, pydata/xarray 22, pytest-dev/pytest 19, pylint-dev/pylint 10, psf/requests 8, mwaskom/seaborn 2, pallets/flask 1.

**Pro** (731): ansible/ansible 96, internetarchive/openlibrary 91, flipt-io/flipt 85, qutebrowser/qutebrowser 79, gravitational/teleport 76, protonmail/webclients 65, future-architect/vuls 62, navidrome/navidrome 57, element-hq/element-web 56, NodeBB/NodeBB 44, tutao/tutanota 20. No `category` tag is present (spec §15 Q6 resolved: stratify by repo).

## 4. Gates (pre-registered)

| Gate | Tier | Criterion | Decision if met / not met |
|---|---|---|---|
| **G0** smoke | 0 | Tier 0 Verified resolves ≥ 4/20 (≥20%) | met → continue to Tier 1; not met → STOP and debug harness |
| **G1a** subset baseline V | 1 | Tier 1 Verified ≥ 72/100 (published 77.2 − 5) | met → continue to Tier 2; not met → pause, sample 5 failures, decide |
| **G1b** subset baseline P | 1 | Tier 1 Pro ≥ 48/100 (published 53.5 − 5) | met → continue to Tier 2; not met → same pause-and-investigate as G1a |
| **G2a** full baseline V | 2 | Full Verified ≥ 75/100 (within 2.2 of published 77.2) | met → Round 5 anchor accepted |
| **G2b** full baseline P | 2 | Full Pro ≥ 51/100 (within 2.5 of published 53.5) | met → Round 5 anchor accepted |
| **G3** harness gap | 2 | Full Verified > 5 absolute below published 77.2 | met → STOP; investigate Codex-vs-Qwen-scaffold harness gap before Round 5 |
| **G4** lossy before-after | per Round 5 sub-change | (lossy_after − lossy_before) ≥ −1.0 absolute on both V and P | met → lossy sub-change is shippable |

Pass-rate is computed on **completed instances only** (denominator
excludes infra_error and hydration_failed). Per spec §3 these latter
counts are reported separately in the closeout so the reader can audit
the denominator.

## 5. Stop-early triggers (pre-registered)

| Trigger | Action |
|---|---|
| Infra failure rate > 10% in first 50 tasks | STOP — investigate harness before continuing |
| Codex CLI self-stop rate > 80% (rc=0 with empty patch) in first 50 tasks | STOP — agent loop broken |
| Tier 1 pass-rate at 50/100 falls > 15 absolute below published baseline | PAUSE — sample 5 failed tasks, document and decide |
| vLLM crash or restart mid-run | PAUSE — relaunch via `/tmp/relaunch_qwen36_AD.py`, resume from next instance; tag affected tasks as `interrupted` for re-run |
| Wall-clock projection exceeds budget by > 50% | PAUSE — pick: shrink remaining task set, or extend budget explicitly |

All pauses are logged in `docs/reports/auto_research/swe-bench-run-log-20260520.md` (created on first pause).

## 6. Concurrency policy

- **Codex agent step:** start at concurrency=1 (LLD-05 §4.6 default). Bump to 4 only after Sprint-1 validation against Docker daemon contention, cache reuse, cleanup behavior.
- **`codex-bench-eval-swe` step:** strict 1 per host (LLD-05 §4.7).

## 7. ARM64 fallback decision (Sprint-1 gate, LLD-05 §4.6)

**Validated 2026-05-21:** native ARM64 SWE-Bench evaluation works on DGX
Spark. Gate evidence: `swebench/sweb.eval.arm64.astropy_1776_astropy-12907:latest`
pulled cleanly, gold-patch instance resolved in 141s on first run and
54s on cache-hit re-run. `codex-bench-eval-swe` (this repo) applies the
required `arch="arm64"` shim because upstream `make_test_spec` defaults
to `x86_64` regardless of host arch and the CLI surface does not expose
the knob.

**Decision:** ship the native ARM64 path. No fallback budget reserved.
If a later instance fails to pull / build for arm64, the failure is
logged as `infra_error` and surfaced in the closeout, not silently
patched around.

## 8. Pinned dependencies

- `swebench==4.1.0` (installed 2026-05-21 into `.venv`)
- `datasets==4.8.5`
- Codex CLI image: `codex-runner:v1`
- Docker base namespace: `swebench` (for `sweb.eval.arm64.*:latest` pulls)
- HF dataset commit Verified: `c104f840cc67f8b6eec6f759ebc8b2693d585d4a` (pinned by `datasets.load_dataset` resolver at 2026-05-21)

## 9. Artifact layout

```
output/swe_bench_q36_a_temp06/
  verified/
    campaign_summary.json
    predictions.jsonl
    per_task/<instance_id>/
      runner_metadata.json
      codex_stdout.log
      codex_trace.jsonl
      patch.diff
      eval_invocation.log
      eval/
        predictions.jsonl
        eval.log
        eval_report.json
        normalized_eval.json
        logs/run_evaluation/<run_id>/<model>/<id>/  (upstream harness per-instance dir)
  pro/
    (same layout)
```

Per-task workspaces (the worktree under `per_task/<id>/workspace/`) are
removed at the end of each instance to free disk; the patch and trace
remain. Per spec §17 trajectories are not committed (gitignored under
`output/`).

## 10. Open spec questions and decisions made here

| Spec §15 Q | Decision | Rationale |
|---|---|---|
| Q1 concurrency cap | start=1, raise only after Sprint-1 validation | LLD-05 §4.6 default |
| Q2 per-task wall budget | 25 min Codex + 5 min eval buffer | spec §6 cost discipline |
| Q3 ARM64 fallback | native (validated) | gate passed; documented above |
| Q4 Q35-D anchor | not run in this campaign | cost +150 wall-hours; revisit only if Tier 2 closes early |
| Q5 temperature | only temp=0.6 (shipping config) | spec §15 recommendation; halves cost; document the temp-vs-published-1.0 gap in closeout |
| Q6 Pro stratification | by repo (no category tag in dataset) | confirmed empirically at fetch time |
| Q7 trajectory retention | gitignore under `output/`; archive locally only | spec §17 recommendation; ~1.2 GB acceptable |
| Q8 CI for pass-rate | bare CI for Tier 1; bootstrap deferred to LLD-12 owner | spec §15 placeholder |

## 10b. Addendum (2026-05-21): ARM64 prebuilt image coverage

After §7 was committed, a Docker Hub probe over all 20 Tier 0 Verified
instance IDs found that **only 11/20 (55%) have a published
`swebench/sweb.eval.arm64.<id>:latest` manifest**; the other 9 do not.
The 12907 instance used for the original Sprint-1 gate was a
fortunate hit, not the median case.

**Operational decision:** `codex-bench-eval-swe` defaults to
`--namespace auto`, which probes the Docker Hub manifest per instance
and falls back to local build (`namespace=None`) when missing. Local
build is functionally equivalent to the prebuilt path (same Dockerfile);
the cost is ~10-15 min for the first build per `(repo, version)`
combination, cached on disk thereafter. For Tier 1+, the warm-cache hit
rate climbs quickly because the dataset only has ~12 distinct repos.

This change is **operationally permissive** — it does NOT relax any
pre-registered gate or alter the verdict surface. Every fallback build
is logged in `eval.log` under `effective_namespace=none`; the run log
will report per-tier counts so the reader can audit how many instances
went through each path.

No fallback budget shift required; the per-task wall budget already
accommodates a ~15 min eval-build phase before the per-task 25 min
agent + 5 min eval buffer kicks in.

## 11. Sign-off

Pre-registration owner: Track B team. This document is committed as part
of the same git revision that introduces the Tier 0 instance manifest.
Any measurement-time deviation must:
1. Be logged in `swe-bench-run-log-20260520.md` with a timestamp.
2. Be flagged in the closeout's "deviations from pre-registration" section.

End of pre-registration.
