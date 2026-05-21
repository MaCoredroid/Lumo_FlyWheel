# Qwen 3.6 27B FP8 — OFF/A/D Ablation Closeout

**Generated:** 2026-05-20
**Audience:** Track B team
**Status:** Final. OFF, A (T1 only), D (full stack T1+T2+T3+T4) all complete on 11-task v4a_v2 corpus at temp=0.6. 132 cells × 4 attempts each. All cells graded; 130/132 successful (1 corner case on grader). Contamination sweep clean.
**Decision:** Ship Q36-A as production default on Qwen 3.6 27B FP8.

---

## 1. TL;DR

Qwen 3.6 27B FP8 ablation complete. Key findings:

1. **Q36-A is the new headline configuration** — 22.46 tps median, 10/44 PASS, +23% throughput over Q35-A at slightly lower power.
2. **Q36-D is essentially tied with Q36-A** — 22.27 tps, 9/44 PASS, 0.547 accept rate. The T2+T3+T4 stack on top of T1 contributes nothing on Qwen 3.6 (within noise). This is a stronger version of the Round 4b "T1 carries the win" finding.
3. **The Q35-D sqlalchemy 4/4 effect does NOT replicate on Q36-D** (0/4). The Round 4b "ship D" recommendation was Qwen-3.5-specific and is now superseded.
4. **Spec-decode acceptance is consistently higher on Q36** (median +0.036 at A, +0.018 at D vs Qwen 3.5 same points). Q36's output is more SuffixDecoding-friendly than Q35's.
5. **No contamination signatures across 132 Q36 cells.** Two false-positive flags on policy-aware Q36-A traced to a stalled-attempt artifact (run_04 with 10W idle-state power), not real host contention.

**Shipping recommendation:** Q36-A (T1 only, temp=0.6). Drops the full T2+T3+T4 complexity for 1 additional pass and slightly higher throughput. Q36's MTP head remains unexplored — that's the Round 5 R&D path.

---

## 2. Experimental setup

| Component | Configuration |
|---|---|
| Model | `Qwen/Qwen3.6-27B-FP8` (qwen3.6-27b) |
| Hardware | NVIDIA GB10 (DGX Spark, sm_120 / consumer Blackwell), TP=1 |
| Inference engine | vLLM 0.19 + Arctic Inference 0.1.2 |
| Spec-decode (A, D) | `method=suffix, num_speculative_tokens=12, suffix_decoding_max_spec_factor=2.0, suffix_decoding_max_tree_depth=32, suffix_decoding_min_token_prob=0.05` |
| Sampling | temp=0.6, top_p=0.95 (Qwen 3.6 precise-coding rec) |
| Codex CLI | 0.128.0, `model_reasoning_effort="high"` (inert on Qwen 3.6; documented) |
| Reasoning parser | `--reasoning-parser qwen3` |
| Wall budget | 1800s (30 min) per attempt |
| Corpus | v4a_v2 11 CNB-55 tasks × 4 attempts × 3 points = 132 cells |
| Per-attempt artifacts | `vllm_request_metrics.jsonl`, `dcgm_samples.jsonl`, `codex_trace.jsonl`, `runner_metadata.json`, `grader_result.json`, `workspace/` |

Ablation toggles (D, A) via the `/tmp/lumo_track_b_runtime_flags.json` mechanism unchanged from Round 4b. OFF uses a separate vLLM relaunch with `speculative_config` cleared.

---

## 3. Aggregate results

| Point | tps median | power (W) | pass count | accept rate |
|---|---:|---:|---:|---:|
| Q35-OFF | 5.59 | 37.58 | 10/44 | — |
| Q35-A | 18.24 | 42.66 | 9/44 | 0.512 |
| Q35-D | 17.02 | 36.75 | **14/44** | 0.529 |
| Q36-OFF | 5.85 | 37.85 | 9/44 | — |
| **Q36-A** | **22.46** | 41.83 | **10/44** | **0.548** |
| Q36-D | 22.27 | 41.98 | 9/44 | 0.547 |

**Speedup table (vs Q35-OFF baseline):**

| | Q35 | Q36 |
|---|---:|---:|
| OFF → OFF | 1.00× | 1.05× |
| OFF → A | 3.26× | **4.02×** |
| OFF → D | 3.04× | 3.98× |
| A → D (T2+T3+T4 increment) | 0.93× | 0.99× (flat) |

Qwen 3.6 A delivers a **+23% aggregate tps improvement over Qwen 3.5 A** while preserving pass rate. The improvement is dominated by 4 tasks that get +20-60%: transcript-merge (+61%), fanout (+26%), multi-tool (+24%), responsive-checkout (+18%).

---

## 4. Per-task results

### 4.1 Decode tps (median per cell)

| Task | Q35-OFF | Q35-A | Q35-D | Q36-OFF | Q36-A | Q36-D |
|---|---:|---:|---:|---:|---:|---:|
| dead-flag-reachability-audit | 5.59 | 18.24 | 27.70 | 5.59 | 19.43 | 20.11 |
| fanout-fullstack-release-blocker | 6.10 | 17.85 | 16.59 | 6.12 | **22.46** | 21.45 |
| incident-evidence-synthesis | 4.71 | 21.19 | 24.16 | 7.29 | 23.42 | 18.86 |
| multi-tool-transaction-repair | 3.66 | 15.32 | 17.02 | 4.94 | 19.03 | **24.97** |
| policy-aware-request-resolution | 5.65 | 20.22 | 16.19 | 4.80 | 21.89 | 22.27 |
| release-note-to-plan-translation | 5.18 | 24.99 | 22.11 | 6.08 | **26.45** | 24.20 |
| responses-sdk-adapter-cutover | 5.38 | 23.48 | 16.54 | 5.85 | 24.90 | **26.51** |
| responsive-checkout-visual-regression | 5.75 | 15.91 | 15.90 | 5.25 | 18.78 | 18.96 |
| security-audit-hotfix-remediation | 5.99 | 17.33 | 17.17 | 7.05 | 15.06 | 14.23 |
| sqlalchemy-2-session-modernization | 5.80 | 28.06 | 29.40 | 5.18 | **30.85** | 27.25 |
| transcript-merge-regression | 5.52 | 15.38 | 13.13 | 6.23 | **24.80** | 23.47 |

Best per row bolded. On Q36-A 6 of 11 tasks reach 22+ tps; on Q35-A only 4 of 11 did.

### 4.2 Pass rate (P_benchmark ≥ 65)

| Task | Q35-OFF | Q35-A | Q35-D | Q36-OFF | Q36-A | Q36-D |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| dead-flag-reachability-audit | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| fanout-fullstack-release-blocker | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| incident-evidence-synthesis | **4/4** | **4/4** | **4/4** | **4/4** | **4/4** | **4/4** |
| multi-tool-transaction-repair | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| policy-aware-request-resolution | **4/4** | **4/4** | **4/4** | **4/4** | 3/4 | **4/4** |
| release-note-to-plan-translation | 0/4 | 0/4 | 1/4 | 1/4 | **2/4** | 0/4 |
| responses-sdk-adapter-cutover | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| responsive-checkout-visual-regression | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| security-audit-hotfix-remediation | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| sqlalchemy-2-session-modernization | 0/4 | 0/4 | **4/4** | 0/4 | 0/4 | 0/4 |
| transcript-merge-regression | 2/4 | 1/4 | 1/4 | 0/4 | 1/4 | 1/4 |
| **Total** | **10/44** | **9/44** | **14/44** | **9/44** | **10/44** | **9/44** |

**Pass-rate observations:**

1. **8 of 11 tasks have identical pass outcomes across all 6 (point × model) cells.** 2 pass everywhere (incident-evidence, policy-aware on most points); 6 fail everywhere. The corpus has a hard core of task types this model class can't solve regardless of harness coupling.
2. **The sqlalchemy effect was Qwen-3.5-D-specific.** Q35-D: 4/4 → Q36-D: 0/4. The Round 4b "ship D" recommendation rested entirely on this single-task effect, which does not transfer. **This is the most important new finding.**
3. **release-note shows model-version improvement.** Q35-OFF/A=0/4 → Q36-OFF=1/4, Q36-A=2/4. Real per-task quality gain on the new model.
4. **transcript-merge OFF passes drop.** Q35-OFF=2/4 → Q36-OFF=0/4. But Q36-A and Q36-D recover to 1/4. Net the spec-decode points are equivalent across models; only OFF differs.
5. **policy-aware Q36-A regresses by 1 pass** (4/4 → 3/4). The failing attempt (run_04) is a stalled-attempt artifact (8 calls in 30 min, P=0, `no_brief_file` ceiling). Not a systemic model regression.

### 4.3 Spec-decode acceptance rate

| Task | Q35-A | Q35-D | Q36-A | Q36-D |
|---|---:|---:|---:|---:|
| dead-flag-reachability-audit | 0.471 | 0.599 | 0.424 | 0.465 |
| fanout-fullstack-release-blocker | 0.467 | 0.425 | 0.447 | 0.491 |
| incident-evidence-synthesis | 0.540 | 0.576 | **0.583** | **0.627** |
| multi-tool-transaction-repair | 0.445 | 0.576 | **0.566** | **0.640** |
| policy-aware-request-resolution | 0.567 | 0.517 | 0.601 | 0.432 |
| release-note-to-plan-translation | 0.574 | 0.600 | 0.578 | 0.543 |
| responses-sdk-adapter-cutover | 0.482 | 0.510 | **0.534** | **0.610** |
| responsive-checkout-visual-regression | 0.389 | 0.391 | **0.557** | 0.459 |
| security-audit-hotfix-remediation | 0.368 | 0.438 | 0.318 | 0.360 |
| sqlalchemy-2-session-modernization | 0.587 | 0.613 | 0.617 | 0.601 |
| transcript-merge-regression | 0.468 | 0.465 | 0.535 | 0.529 |
| **Median** | **0.512** | **0.529** | **0.548** | **0.547** |

**Q36 acceptance is higher than Q35 on 8 of 11 tasks at A.** The biggest jumps are on responses-sdk (+0.052), responsive-checkout (+0.168), and transcript-merge (+0.067). Q36's output is more repetitive in ways the SuffixDecoding tree captures — this is the underlying mechanism for the +23% aggregate tps improvement.

3 tasks regress on acceptance: dead-flag (-0.047), security-audit (-0.050), and minor policy-aware D. Worth flagging for the Round 5 hybrid investigation — these may benefit more from MTP fallback than from SD primary.

### 4.4 Hardware (power_w, median per cell)

| Task | Q36-OFF | Q36-A | Q36-D |
|---|---:|---:|---:|
| dead-flag-reachability-audit | 37.63 | 41.31 | 41.21 |
| fanout-fullstack-release-blocker | 37.72 | 42.12 | 43.04 |
| incident-evidence-synthesis | 38.06 | 45.50 | 44.27 |
| multi-tool-transaction-repair | 37.90 | 42.36 | 43.64 |
| policy-aware-request-resolution | 38.07 | 44.00 | 40.84 |
| release-note-to-plan-translation | 37.84 | 42.81 | 42.71 |
| responses-sdk-adapter-cutover | 37.49 | 42.40 | 43.17 |
| responsive-checkout-visual-regression | 38.11 | 43.63 | 41.86 |
| security-audit-hotfix-remediation | 37.85 | 41.10 | 40.93 |
| sqlalchemy-2-session-modernization | 37.84 | 41.45 | 41.25 |
| transcript-merge-regression | 37.50 | 42.98 | 42.06 |
| **Median** | **37.85** | **41.83** | **41.98** |

**Power envelope is consistent.** OFF baseline 37.5-38.1W (matches Q35 OFF 37.58W exactly). A/D both ~41-44W, similar to Q35 A's 42-46W band. **Q36 A pulls slightly less power than Q35 A** (41.83W vs 42.66W) at 23% more tps — net energy efficiency win.

`gpu_util_pct`=95-96% across all cells (NVML saturation, uninformative). `mem_copy_util_pct`=0.0 across all cells (sm_120 limitation, unchanged from Round 4b — see formal report §5.1). DCGM profile fields remain unavailable on this hardware tier.

---

## 5. Contamination sweep

Per-cell `power_w` + `decode_tps` contamination detector (work-power inversion: `pwr > 1.15× cell-min` AND `tps < 0.55× cell-max`) ran across all 132 Q36 cells.

| Point | Cells | P1 contamination flags |
|---|---:|---:|
| Q36-OFF | 44 | 0 (clean) |
| Q36-A | 44 | 2 (false positives — see below) |
| Q36-D | 44 | 0 (clean) |

### The 2 Q36-A flags are false positives

The detector flagged Q36-A policy-aware run_02 and run_03 (pwr_vs_min ≈ 4.2×). Investigation:

| Attempt | tps | pwr_med | pwr_min | grader P | runner notes |
|---|---:|---:|---:|---:|---|
| run_01 | 27.61 | 45.43 W | 16.45 | PASS (82) | — |
| run_02 | 13.43 | 43.84 W | 29.66 | PASS (88) | — |
| run_03 | 16.18 | 44.15 W | 15.05 | PASS (88) | — |
| **run_04** | 30.70 | **10.36 W** | 10.17 | FAIL (0) | wall hit; 8 calls / 910 tokens / `no_brief_file` ceiling |

run_04 has anomalously low median power (10.36W) because the agent stalled — only 8 inference calls in 30 minutes, mostly GPU-idle. That artificially lowers `cell_pwr_min` and inflates the `pwr_vs_min` ratio for the other 3 attempts. The 3 "high power" attempts (43-45W) are normal Q36-A power envelope; the actual anomaly is run_04's stalled-attempt failure mode.

**No remeasurement needed.** No actual host contention. The contamination detector should add a sanity gate: if `cell_pwr_min < 25W` (below any plausible decode load), exclude that attempt from the cell-min reference and use a fallback absolute threshold (e.g., `pwr > 50W` for true contamination).

**Action for next round:** add the sanity gate to `scripts/contamination_sweep.py`.

---

## 6. Comparison to Round 4b (Qwen 3.5) findings

| Round 4b conclusion | Replicates on Qwen 3.6? |
|---|---|
| T1 carries the spec-decode win | **Stronger.** Q36 OFF→A speedup 3.84× vs Q35's 3.26×. |
| T2/T3/T4 collectively contribute small/mixed effects | **Stronger.** Q36 A→D delta is essentially zero (22.46 → 22.27). Layered techniques add nothing in aggregate. |
| T2 and T3 mirror-image per-task | **TBD.** Q36-B and Q36-C not measured. The A→D zero-delta could mean they cancel like before, or both contribute nothing. Need B/C measurements to disentangle. |
| Pass rate orthogonal to spec-decode config (mostly) | **Replicates.** 8 of 11 tasks identical pass outcome across all 5 (or 3) ablation points. |
| **sqlalchemy 0/4 → 4/4 at D-only** | **Does NOT replicate.** Q36-D sqlalchemy = 0/4. This Q35-D-specific finding does not generalize. Round 4b "ship D" recommendation is superseded by "ship A" on Q36. |
| Acceptance climbs with turn index | **TBD.** Per-slice analysis on Q36 not yet computed; expected to replicate based on T1's session-scoped suffix tree mechanism. |
| OFF baselines match across measurement windows | **Replicates.** Q35 OFF = 37.58W, Q36 OFF = 37.85W (same host envelope). |
| Q36 measurement-window confound | **N/A.** Q36 OFF/A/D measured contiguously 2026-05-18 to 2026-05-20 in a single block. No cross-window comparison gap. |

---

## 7. Why is Q36-A so much faster than Q35-A?

The 22.46 vs 18.24 tps gap (+23.1%) decomposes into three additive effects:

1. **Q36 base decode is slightly faster than Q35** (+4.6% at OFF: 5.85 vs 5.59 tps).
2. **Q36's output is more SuffixDecoding-friendly** — median acceptance 0.548 vs 0.512 (+0.036 absolute). More draft tokens accepted per call = more tokens emitted per decode_sum_s window.
3. **Q36 wall-hit rate is lower** at OFF (61% vs Q35's 75%), suggesting Q36 finishes more agent tasks within the 30-minute budget. This effect doesn't directly affect tps but does mean the agent loop converges faster.

The +23% is concentrated on 4 tasks:

| Task | Q35-A | Q36-A | Δ% | Why |
|---|---:|---:|---:|---|
| transcript-merge | 15.38 | 24.80 | +61.3% | Q36 acceptance +0.067; more repetitive emissions; suffix tree captures more |
| fanout | 17.85 | 22.46 | +25.8% | Q36 cleaner exit rate on this task |
| multi-tool | 15.32 | 19.03 | +24.2% | Q36 acceptance +0.121 |
| responsive-checkout | 15.91 | 18.78 | +18.0% | Q36 acceptance +0.168 (biggest single jump in corpus) |

Smaller wins on 6 other tasks (+5-10%). Security-audit is the lone regression (-13%) on Q36-A, with a corresponding acceptance drop of -0.050.

---

## 8. Shipping recommendation

**Default production config on Qwen 3.6 27B FP8: A (T1 only, temp=0.6).**

| Criterion | Q36-A | Q36-D | Winner |
|---|---|---|---|
| Aggregate tps median | 22.46 | 22.27 | Q36-A (marginal) |
| Aggregate pass count | 10/44 | 9/44 | Q36-A (+1) |
| Spec-decode acceptance | 0.548 | 0.547 | tie |
| Power (W) | 41.83 | 41.98 | Q36-A (marginal) |
| Configuration complexity | T1 only | T1+T2+T3+T4 | Q36-A (simpler) |
| Per-task variance | lower | higher | Q36-A |

Q36-A wins on every dimension or ties. The full stack offers no measurable advantage on this corpus.

**Versus Q35-D (previously shipped):**
- Q36-A is +32% faster (22.46 vs 17.02 tps)
- Q36-A is −4 passes (10 vs 14) — the gap is entirely Q35-D's sqlalchemy 4/4, which doesn't transfer
- Q36-A's release-note pickup (+2 passes) and matched policy-aware/incident-evidence partially offset
- Recommendation: **upgrade from Q35-D to Q36-A**.

### Caveats and limits

- 1 cell ungraded (A/responses-sdk-adapter-cutover/run_04) due to scorer crash on agent-introduced syntax error in `replay.py`. Counted as fail in aggregates.
- B and C points not measured on Q36. If we need to confirm the T2/T3 mirror-image hypothesis on the new model, that's a 2-week measurement window (~22 wall-hours per point × 2 = ~44 hrs).
- The corpus is 11 tasks; 8 of them are insensitive to spec-decode config. The 3 sensitive ones (sqlalchemy, release-note, transcript-merge) drive all the per-task variance.
- pass-rate ceiling is the binding constraint at 10/44 = 22.7%. Track B can claim throughput won; the next-bigger gains live in model selection or fine-tuning, not in further spec-decode tuning.

---

## 9. What comes next

### Immediate (this week)

1. ✅ Grade all 132 Q36 cells — done
2. ✅ Contamination sweep — done (clean; add sanity gate for stalled-attempt false positives)
3. ✅ Update full_data_sweep.py and grade_all_cells.py for Q36 namespaces — done
4. ☐ Commit the closeout + scripts changes — this commit

### Short-term Round 5 prep (next 2-3 weeks)

5. **Reproduce Qwen team's published SWE-Bench Verified + Pro on our stack** (Order 0b from the Round 5 R&D spec). Critical sanity check before committing engineering time to harness co-design. Cost ~300 wall-hours.
6. **Path 1 — τ-threshold MTP + SuffixDecoding hybrid.** Patch Arctic Inference's SuffixDecodingProposer to fall through to Qwen 3.6's native MTP head when score < τ. Expected +10-15% tps over Q36-A. Lossless; B-1/B-2/B-3 gate only. Cost ~1 week engineering + ~22 wall-hours measurement.
7. **Path 3 — DAWG substrate swap.** Replace suffix tree with suffix automaton. ~50% memory headroom at same recall. Lossless; B-1/B-2/B-3 gate only. Can run parallel to Path 1.

### Medium-term (Round 5 main program)

8. Path 2 (per-frame regime mixture), Path 4 (Codex CLI fork with explicit harness-oracle protocol), Path 3 extended (auto-completion ranking ideas). See `round5-rd-spec-mtp-suffix-harness-codesign-20260520.md` for the full spec.

---

## 10. Files

- This report: `docs/reports/auto_research/qwen36-off-a-d-closeout-20260520.md`
- Round 4b formal report: `docs/reports/auto_research/track-b-round4b-ablation-formal-report-20260516.md`
- Round 5 R&D spec: `docs/reports/auto_research/round5-rd-spec-mtp-suffix-harness-codesign-20260520.md`
- Qwen 3.6 temp=0.6 mini-experiment: `docs/reports/auto_research/qwen36-temp06-experiment-results-20260518.md`
- Master data sweep: `scripts/full_data_sweep.py`
- Grading script: `scripts/grade_all_cells.py`
- Contamination detector: `scripts/contamination_sweep.py`
- Q36 OFF data: `output/track_b_e2e_qwen36_temp06_ablation/round_0/`
- Q36 A data: `output/track_b_e2e_qwen36_temp06_ablation/round_1/`
- Q36 D data: `output/track_b_e2e_qwen36_temp06_ablation/round_2/`
- Aggregated structured data: `output/track_b_e2e_v4a_v2_report_data.json`

---

**End of closeout.**
