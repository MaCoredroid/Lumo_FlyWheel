# SWE-Bench Verified + Pro on Qwen 3.6-27B-FP8 + vLLM + Codex CLI — Bounded-Time Measurement Spec

**Generated:** 2026-05-20
**Audience:** Track B team
**Status:** Spec (no measurements yet)
**Constraint:** bounded wall-time; design for partial-coverage results to be defensible.

---

## 1. Why this spec

The Round 5 R&D spec (`round5-rd-spec-mtp-suffix-harness-codesign-20260520.md` §9) calls for SWE-Bench Verified + SWE-Bench Pro as the **external quality gate** for every lossy Path 4 harness change. That gate has two prerequisites:

1. **Order 0b baseline reproduction** — run Verified + Pro on our current Q36-A stack and confirm we land within ±2 absolute points of Qwen team's published 77.2 / 53.5. Without this, every subsequent before-after comparison is anchored to an undefined reference.
2. **A repeatable measurement protocol** — bounded enough to fit in real engineering schedules, stratified enough to give defensible numbers from partial coverage, instrumented enough to support the lossy-change before-after comparisons that come later.

This spec defines that protocol.

## 2. The three things that make this trickier than internal v4a_v2

| Issue | Implication |
|---|---|
| **Wall-time per task isn't standardized.** Community convention is 30 min; Qwen team's published numbers don't document their cap. We choose ours. | Per-task budget decision affects tail-completion tasks and aggregate score. Document it explicitly and stick to one number across before/after pairs. |
| **Codex CLI ≠ Qwen team's internal scaffold.** Their published numbers used bash + file-edit tools with their own agent loop. Ours uses Codex CLI 0.128.0. | Report our numbers as "Qwen-3.6-27B on Codex CLI" — not as a direct attempt to match the published headline. Different agent loops can swing 5-15 points. |
| **The benchmark harness is heavy.** Per-task Docker env setup, gold test suite, ARM64 considerations on DGX Spark. | Use the existing LLD-05 infrastructure (`codex-bench-eval-swe`); don't rebuild from scratch. |

## 3. Goal and gate criteria

### Primary goal

Produce a defensible, reproducible benchmark number for Qwen 3.6-27B-FP8 + Q36-A spec-decode + Codex CLI on:

- **SWE-Bench Verified** — 500 instances, dataset `princeton-nlp/SWE-bench_Verified`
- **SWE-Bench Pro** — ~700 instances, dataset `ScaleAI/SWE-bench_Pro`

### Pre-registered gates

| Gate | Criterion | Decision |
|---|---|---|
| **G0** smoke pass | Tier 0 (20-task Verified subset) lands ≥ 4/20 = 20% PASS | Continue to Tier 1; harness not catastrophically broken |
| **G1** subset baseline | Tier 1 (100-task stratified subset) lands within ±5 absolute of published baselines | Continue to Tier 2; we're in the right neighborhood |
| **G2** full baseline | Tier 2 full benchmark lands within ±2 absolute of published 77.2 / 53.5 | Round 5 lossy work can use this as the reference |
| **G3** harness gap escalation | Tier 2 lands > 5 absolute below published | **STOP** — Codex CLI harness gap is bigger than spec-decode work can fix; investigate harness before Round 5 |
| **G4** before-after | (lossy change after − before) ≥ −1.0 absolute on both Verified and Pro | Round 5 lossy Path 4 change is shippable |

Pre-register all gates before measurement. Document the exact stratification of any subset before reading any data from it.

## 4. The bounded-time design — three tiers

The spec is built around three execution tiers, each with a clear stop/continue decision. Time budget per tier escalates as confidence in the harness grows.

### Tier 0 — Smoke (overnight, ~10 wall-hours)

| Parameter | Value |
|---|---|
| Tasks | 20 stratified-random Verified instances |
| Per-task wall budget | 30 min Codex agent + 5 min eval |
| Concurrency | 1 (single-stream; debug-friendly) |
| Cost | 20 × ~35 min = ~12 wall-hours |
| Stratification | 2 per repo across 10 repos (django, sympy, scikit-learn, requests, flask, sphinx, astropy, pylint, pytest, matplotlib — or top-10 by instance count in Verified) |
| Pass criterion | ≥ 4/20 = 20% PASS (well below published 77% — catches catastrophic breakage only) |

Goal: catch infrastructure failures before committing to a longer run. If any of the following happen, stop and fix:

- Docker per-task env setup fails on ARM64 (LLD-05 §4.6 fallback decision)
- `codex-bench-eval-swe` exit code 2 (infra error) on > 3 tasks
- `predictions.jsonl` schema mismatch with upstream harness
- vLLM connectivity issue mid-run
- Codex CLI Docker container OOM
- Wall-budget-exceeded on > 75% of tasks (signals per-task budget is too tight or agent loop is broken)

### Tier 1 — Subset baseline gate (1-2 days, ~30-50 wall-hours)

| Parameter | Value |
|---|---|
| Tasks | 100 stratified Verified + 100 stratified Pro = 200 total |
| Per-task wall budget | 25 min Codex + 5 min eval (a small tightening from Tier 0 to fit more) |
| Concurrency | 2-4 concurrent Codex agents (validate per LLD-05 §4.6 host-local semaphore) |
| Cost (4×) | 200 × 30 min / 4 ≈ 25 wall-hours |
| Cost (2×) | 200 × 30 min / 2 ≈ 50 wall-hours |
| Stratification | By repo + difficulty tier (when tags available). For Pro, use ScaleAI's category tags if available; else stratified-random by repo. |
| Pass criterion (G1) | Subset Verified ≥ 72 (published 77.2 − 5) AND subset Pro ≥ 48 (published 53.5 − 5) |

Goal: directionally confirm we're in the right neighborhood before committing to the full benchmark. Subset is large enough to give 95% confidence intervals of ±4-5 points on a 77% baseline.

If G1 fails, halt and investigate:
- Compare per-task PASS list to the published Qwen team's per-task list (if available). Identify divergent categories.
- Check Codex CLI version / wire format mismatch.
- Verify Codex's emitted patches are syntactically valid before passing to the eval harness.
- Sample a handful of fails and read the trajectory — is the agent making sensible attempts and the patch failing the test, or is something else broken?

### Tier 2 — Full benchmark (5-7 days, ~120-150 wall-hours)

| Parameter | Value |
|---|---|
| Tasks | 500 Verified + ~700 Pro = 1200 total |
| Per-task wall budget | 25 min Codex + 5 min eval |
| Concurrency | 4 (assuming Tier 1 validated this concurrency) |
| Cost | 1200 × 30 min / 4 ≈ 150 wall-hours |
| Pass criterion (G2) | Verified ≥ 75 (within 2.2 of published 77.2) AND Pro ≥ 51 (within 2.5 of published 53.5) |

Goal: produce the official Round 5 baseline number with publication-grade coverage.

If G2 fails (G3 escalation):
- Document the gap explicitly (e.g., "Codex CLI agent loop achieves 70.0 on SWE-Bench Verified vs Qwen team's 77.2 — a 7.2-point harness gap").
- Investigate the gap before committing engineering time to Round 5 spec-decode hybrids. The published Qwen numbers were achieved by Qwen's scaffold; closing that gap may be the highest-leverage Round 5 task.

### Tier 3 — Lossy-change gate (per Path 4 lossy sub-change)

Repeats Tier 1 (subset, ~25-50 wall-hours) before and after each lossy sub-change, plus Tier 2 (full) before shipping a lossy bundle in a release. See Round 5 R&D spec §9 for the full per-sub-change matrix.

## 5. Sample design — stratification

### Verified stratification (100-task subset for Tier 1)

| Stratum | Count | Notes |
|---|---:|---|
| `django/django` | 20 | Largest single repo in Verified; over-sampled to match relative weight |
| `sympy/sympy` | 12 | |
| `scikit-learn/scikit-learn` | 10 | |
| `matplotlib/matplotlib` | 8 | |
| `astropy/astropy` | 7 | |
| `sphinx-doc/sphinx` | 6 | |
| `pytest-dev/pytest` | 6 | |
| `pylint-dev/pylint` | 5 | |
| `psf/requests` | 4 | |
| `pallets/flask` | 4 | |
| Other 8 repos | 18 | 2-3 each |
| **Total** | **100** | |

Sample selection: deterministic with seed `0` for reproducibility. Document the exact list of 100 instance IDs in `docs/reports/auto_research/swe-bench-tier1-verified-instances-20260520.md` before running.

### Pro stratification (100-task subset for Tier 1)

ScaleAI's SWE-Bench Pro has category tags. Stratify by category if available; else stratified-random by repo with similar approach to Verified.

### Full benchmark (Tier 2)

All instances, no stratification needed.

## 6. Per-task wall-budget — why 25 minutes

Three considerations:

- **Codex CLI 0.128.0 self-stop behavior.** We've seen Codex self-stop around 17-20 minutes on Qwen 3.6 tasks even when the agent has more work to do. The 30-min budget is mostly used by the auto-continue loop.
- **Q36-A throughput.** At 22.46 tps median, a 25-min budget delivers ~33K decode tokens (assuming continuous decode, no tool exec time). That's enough for ~50-80 tool-call turns for typical SWE-Bench tasks. Sufficient.
- **Cost discipline.** 25 min × 1200 tasks / 4 concurrent = 125 wall-hours. 30 min would be 150. The 17% saving matters at full-benchmark scale.

Risk: tasks near the 25-min wall might just barely complete on 30 min. Mitigation: collect both `elapsed_s` and `codex_exit_code` per task; if > 15% of failures are wall-hit (rc=124) with non-trivial `changed_paths`, we may need to raise to 30 min for Tier 2.

## 7. Concurrency strategy

### Per-host concurrency (LLD-05 §4.6 host-local semaphore)

Default per LLD-05: **1 concurrent SWE-Bench evaluator subprocess per host.** Going higher requires Sprint-1 validation against:

- Docker daemon contention (each task spawns a fresh container)
- Cache reuse behavior (per-task Python env builds shouldn't fight for the same cache dirs)
- Cleanup behavior (concurrent failures could leave orphan containers)

### What can run concurrently safely

- **Codex agent step:** Yes, multiple Codex CLI Docker containers can run simultaneously against the same vLLM instance. vLLM continuous batching handles the inference side. Default cap: 4 concurrent Codex agents — validated on similar workloads but worth a Sprint-1 check.
- **Eval step:** Maybe. Eval spawns per-task pytest containers. LLD-05 default is 1 concurrent. We could run 2 concurrent eval subprocesses if Sprint-1 validation shows clean cache + cleanup behavior. **For Tier 1, keep eval at 1× to be safe.**

### Practical schedule

```
  Time → 0h        1h        2h        3h        4h         5h
Agent  [task001][task002][task003][task004][task005][task006]...
Agent  [task007][task008][task009][task010][task011][task012]...
Agent  [task013][task014][task015][task016][task017][task018]...
Agent  [task019][task020][task021][task022][task023][task024]...
Eval        [task001-eval][task002-eval][task003-eval]...
```

The eval queue trails behind the agent queue by one task; the eval is single-stream by default. Agent throughput is the bottleneck; eval keeps up so long as it's not too far behind.

### Multi-host (if available)

If we have 2 DGX Sparks, the natural split is:
- Spark A: Verified subset / full
- Spark B: Pro subset / full

Each runs independent agent + eval pipeline. Roughly halves wall time.

## 8. Stop-early criteria — when to bail mid-run

Set criteria up front; pre-register them; don't change them mid-run.

| Trigger | Action |
|---|---|
| Infra failure rate > 10% (Docker setup fails, eval exit code 2) in first 50 tasks | **Stop** — investigate harness before continuing |
| Codex CLI self-stop rate > 80% (rc=0 with no changed_paths) in first 50 tasks | **Stop** — auto-continue is broken; fix before continuing |
| Tier 1 pass-rate falls > 15 absolute below published baseline at 50/100 mark | **Pause** — sample 5 failed tasks, read trajectories, decide whether to continue and document the gap, or stop |
| vLLM crash or restart mid-run | **Pause** — re-launch, resume from next task; tag the affected tasks as `interrupted` for re-run |
| Wall-clock projection exceeds budget by > 50% | **Pause** — pick: shrink the remaining task set, or extend the budget explicitly |

Document every pause-or-stop in `docs/reports/auto_research/swe-bench-run-log-<date>.md` for post-hoc traceability.

## 9. Integration with existing LLD-05 infrastructure

LLD-05 §4 already specifies:

- **Entry point:** `codex-bench-eval-swe`
- **Input:** Codex patch artifact path (per-attempt patch.diff from the Codex trace)
- **Conversion:** patch → upstream `predictions.jsonl` schema
- **Invocation:** official SWE-Bench harness CLI
- **Output:** `eval_report.json` with failure-mode enum (`tests_passed`, `tests_failed`, `patch_apply_failed`, `infra_error`)
- **No-patch terminal:** synthesized from run record when Codex never emits a patch (e.g., wall-budget timeout or self-stop with empty diff)

What we add for this campaign:

1. **Per-attempt artifact layout** matching LLD-05 §4.4:
   ```
   output/swe_bench_q36_a_temp06/
     verified/
       run_log.md                      # human-readable run log
       predictions.jsonl               # all per-task predictions in one file
       per_task/
         <instance_id>/
           prompt.md                   # synthesized from SWE-Bench problem_statement + test_patch hint
           codex_trace.jsonl
           codex_stdout.log
           workspace/                  # post-Codex workspace
           patch.diff                  # extracted patch
           predictions.jsonl           # single-task predictions file
           eval_report.json            # SWE-Bench harness verdict
           runner_metadata.json        # elapsed_s, codex_exit_code, etc.
           vllm_request_metrics.jsonl  # for throughput correlation
           dcgm_samples.jsonl          # for power correlation
     pro/
       (same layout)
   ```
2. **Driver script** `scripts/run_swe_bench_q36_a.py` — orchestrates per-task execution, manages concurrency, writes predictions.jsonl per task, invokes the eval CLI, aggregates.
3. **Stratification driver** `scripts/build_swe_bench_subset.py` — produces the deterministic Tier 1 subset list.

## 10. Comparison anchors

Multiple anchor points; report all of them.

| Anchor | Source | Verified | Pro |
|---|---|---:|---:|
| Qwen team published (Q36-27B-on-Qwen-scaffold) | Qwen model card April 2026 | 77.2 | 53.5 |
| Qwen team published (Q35-27B-on-Qwen-scaffold) | Qwen model card | 75.0 | 51.2 |
| Anthropic published (Claude 4.5 Opus) | Qwen model card | 80.9 | 57.1 |
| **Ours (Q35-D-on-Codex)** — old shipping config | this campaign optional addendum | (TBD) | (TBD) |
| **Ours (Q36-A-on-Codex)** — new shipping config, primary anchor | this campaign | (TBD) | (TBD) |

The Q35-D-on-Codex anchor is optional but valuable: it tells us how much of any Q36-A improvement comes from "model upgrade" vs "spec-decode upgrade." Costs ~150 wall-hours additional. Recommend running ONLY IF Tier 2 closes early and we have hardware budget remaining.

## 11. Per-task execution protocol

Per task:

1. **Setup:** Pull the SWE-Bench instance bundle (problem_statement + base repo + test_patch).
2. **Codex agent:** Launch `codex-runner:v1` Docker against the prepared workspace. Send the problem_statement as the operator prompt. Codex iterates with tool calls; emits a patch.
3. **Patch extraction:** Diff `workspace/` against base repo → `patch.diff`.
4. **Predictions.jsonl synthesis:** Convert patch.diff into upstream-compatible record:
   ```jsonc
   {
     "instance_id": "<repo>__<repo>-<issue>",
     "model_name_or_path": "qwen3.6-27b-fp8::codex-cli-0.128.0",
     "model_patch": "<patch.diff contents>",
     "trajectory_path": "<absolute path to codex_trace.jsonl>"
   }
   ```
5. **Eval:** Invoke `codex-bench-eval-swe --instance-id <id> --predictions-path predictions.jsonl --dataset-name princeton-nlp/SWE-bench_Verified` (or Pro variant). Returns exit code + `eval_report.json`.
6. **Verdict normalization:** Parse `eval_report.json`, emit per-task `{instance_id, outcome, failure_mode}` record.

## 12. Artifact protocol

Per task, save the artifacts listed in §9. Per-campaign, additionally:

- `predictions.jsonl` — aggregated across all per-task records (the upstream-compatible single file)
- `campaign_summary.json` — aggregate score, per-repo breakdowns, per-failure-mode counts, time budget consumed, wall-hours per task percentiles
- `swe-bench-run-log-<date>.md` — narrative log of any pauses, infra failures, escalation decisions

Commit the campaign summary and run log; gitignore the per-task workspaces (consistent with existing pattern in `output/`).

## 13. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ARM64 SWE-Bench eval fails on DGX Spark | medium | high — blocks measurement | Per LLD-05 §4.6: fallback options pre-pinned: (a) x86 emulation; (b) offload eval to a x86 host; (c) wait for SWE-Bench ARM64 native fix. Validate in Tier 0. |
| Concurrency causes Docker daemon contention | medium | medium — slows or breaks runs | Cap at 4× concurrent agents + 1× eval initially. Sprint-1 validate higher if needed. |
| Per-task wall budget too tight, tail tasks fail | medium | low-medium | Collect rc=124 rate; if > 15%, raise to 30 min before Tier 2. |
| Codex CLI patch format inconsistent with SWE-Bench expectations | low | high | LLD-05 already owns the conversion; validate in Tier 0. |
| vLLM crash mid-run loses ~30 min of work | low | low | Per-task atomic artifacts; restart resumes from next task. |
| Tier 2 lands far below published (G3 escalation) | medium | high — invalidates Round 5 plan | This IS what we want to find out. Treat as data, not as a failure to fix the spec — report the harness gap and adjust Round 5 priorities. |
| Per-task cost varies wildly (5 min vs 25 min) | high | low | Concurrency hides this; the slowest task in a 4-batch dominates batch latency but overall throughput is fine. |
| ScaleAI Pro dataset gating / access | low | medium | Verify dataset accessible before Tier 1. The dataset is open on Hugging Face per the source. |

## 14. Decision matrix — which tier should we run first

Given different hardware/time budgets:

| You have... | Run | Get |
|---|---|---|
| 1 day overnight | Tier 0 (smoke, 20 tasks) | Catch-or-greenlight signal. ±20 abs CI on a 20-task sample. |
| 1 weekend (~48 hours) | Tier 1 subset (200 tasks) | Directional baseline ±4-5 abs. G1 gate. Sufficient to greenlight Tier 2. |
| 1 week (~150 wall-hours) | Tier 2 full (1200 tasks) | Publication-grade baseline ±2 abs. G2 gate. **This is the Round 5 prerequisite.** |
| 2-3 weeks | Tier 2 + Q35-D-on-Codex anchor | Headline number + model-vs-spec-decode attribution |

**Recommended sequence:** Tier 0 → Tier 1 (gate-pass required) → Tier 2.

If we want a defensible Round 5 anchor in the shortest possible time, the sequence is **Tier 0 over a weekend + Tier 1 the following week + Tier 2 the week after**. End-to-end: ~14 days calendar time, ~180 wall-hours hardware, with three gates that let us bail early if the harness is broken.

## 15. Open design questions

1. **Concurrency cap.** LLD-05 default is 1 concurrent eval per host. Can we safely go to 2 on DGX Spark? Sprint-1 validation gate. Resolve before Tier 1.
2. **Per-task wall budget.** 25 min is the proposed default. If Tier 0 shows > 15% wall-hit rate, raise to 30. Document before Tier 1.
3. **ARM64 SWE-Bench eval.** Does native ARM64 work on DGX Spark? If no, which fallback? Decide before Tier 0 finishes.
4. **Q35-D anchor.** Do we run Q35-D-on-Codex SWE-Bench as a comparison? Cost +150 wall-hours. Recommended ONLY if Tier 2 closes early.
5. **Temperature.** Q36-A campaign is at temp=0.6 (precise-coding rec). Qwen's published 77.2 was at temp=1.0. **Should we measure both, or only temp=0.6 (our shipping config)?** Recommendation: only temp=0.6 — measuring two temperatures doubles cost without changing the production decision. Document the temperature in the campaign metadata and note it as a known gap vs Qwen's published number.
6. **Sampling for Pro stratification.** Does Pro have published category tags we can use? If not, stratify by repo and document. Resolve before Tier 1 subset selection.
7. **Trajectory artifact retention.** Codex traces can be 500KB-2MB per task. 1200 tasks × 1MB ≈ 1.2GB. Acceptable on local disk. Decide whether to commit (probably no, gitignore) or just archive locally.
8. **Pass-rate confidence intervals.** Tier 1 100-task subset on a 77% baseline has 95% CI of ±8 points worst-case. Tier 2 500-task on the same has ±4. For publication, do we need bootstrapping or is the bare CI sufficient? Resolve before Tier 2 closeout.

---

## 16. What to do this week

If we want to start measuring this week:

1. **Day 1** — confirm Tier 0 setup. Validate `codex-bench-eval-swe` runs natively on DGX Spark (ARM64 gate). Verify dataset access for both Verified and Pro. Build the stratification driver. Pre-register Tier 0 task list.
2. **Day 2** — kick off Tier 0 (overnight). Single-stream, 20 Verified tasks. Sleep on it.
3. **Day 3** — read Tier 0 results. If G0 passes, lock the stratification + concurrency decisions and kick off Tier 1 (24-48 hours).
4. **Day 4-5** — Tier 1 running. Use the wait to start prep for Tier 2 (containers, scratch space, etc.).
5. **Day 6** — Tier 1 closeout. Check G1 gate. Decide Tier 2 scope.
6. **Days 7-13** — Tier 2 full benchmark, both Verified and Pro.
7. **Day 14** — Tier 2 closeout, write headline numbers report, decide Round 5 next steps.

## 17. Files this spec touches

- This spec: `docs/reports/auto_research/swe-bench-bounded-time-spec-20260520.md`
- Round 5 R&D spec: `docs/reports/auto_research/round5-rd-spec-mtp-suffix-harness-codesign-20260520.md`
- Q36 closeout report: `docs/reports/auto_research/qwen36-off-a-d-closeout-20260520.md`
- LLD-05 evaluator dual track: `docs/LLD-05-Evaluator-Dual-Track-v0_1.md`
- Drivers to add: `scripts/run_swe_bench_q36_a.py`, `scripts/build_swe_bench_subset.py`
- Future artifacts: `output/swe_bench_q36_a_temp06/{verified,pro}/...`

## 18. External sources

- [SWE-Bench Verified dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
- [SWE-Bench Pro dataset](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro)
- [SWE-Bench harness repo](https://github.com/SWE-bench/SWE-bench)
- [Qwen 3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B) — published baselines
- [Codex CLI 0.128.0](https://github.com/openai/codex)

---

**End of spec.**
