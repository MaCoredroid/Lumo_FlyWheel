# Track B Round 4b — E2E Ablation Against v4a Baseline

Generated: 2026-05-11
Status: **NULL RESULT** — techniques T2/T3/T4 do not contribute measurable
e2e wallclock or acceptance improvement on the 13-task v4a corpus. The
Round 3 synthetic microbench gradient (T1 33.5 % → all-on 78.9 %) does
**not** generalize to real Codex CLI traffic.

Companion to:
- `track-b-round4a-closeout-20260510.md` (canonical baseline; defines the §11 two-number framing)
- `track-b-round4b-per-regime-acceptance-20260511.md` (per-regime decomposition on v4a baseline)
- `scripts/run_track_b_round3_ablation.py` (the synthetic microbench whose gradient this report falsifies)

## 1. Headline

Four ablation points on the v4a measurement protocol (round-start
warmup, `--repeat 4`, `--zero-token-retries 3`). Point D is the
existing v4a baseline; points A, B, C are new measurements with the
runtime ablation flag file flipping T2/T3/T4 to disabled.

| Point | Flags disabled | Clean median | Δ vs D | Op median | Clean agg | Tool-call accept | Reasoning accept | Cohort clean/rec/exh |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **D** (all on, v4a baseline) | none | **15.60 s** | — | 19.37 s | 193.5 s | 0.532 | 0.230 | 18 / 16 / 5 |
| **A** (T1 only) | T2,T3,T4 | 16.19 s | +3.8 % | 19.42 s | 170.6 s | 0.556 | 0.214 | 16 / 15 / 8 |
| **B** (T1+T2) | T3,T4 | 16.17 s | +3.7 % | 19.62 s | 141.5 s | 0.562 | 0.267 | 17 / 15 / 7 |
| **C** (T1+T2+T3) | T4 | **14.85 s** | **−4.8 %** | 20.30 s | 172.6 s | 0.551 | 0.182 | 17 / 16 / 6 |

The 4 points span clean medians **14.85 s – 16.19 s** — a 1.34 s range,
~8.5 % of the lowest value. Sample medians over 16-18 measurements each
have an inherent variance comparable to that range, so the differences
between points are **indistinguishable from sample noise**. Notably,
**point C (one technique off) has a lower clean median than point D
(all on)**, which is the opposite of what the technique-composition
hypothesis predicts.

Tool-call acceptance spans **0.532 – 0.562** across all four points — a
0.030 range over 78-89-row samples per regime. Reasoning acceptance
spans 0.182-0.267 over 8-12 rows; that range is dominated by small-N
variance and shows no monotone gradient with technique enablement.

**Conclusion:** the harness-coupled techniques T2, T3, T4 contribute no
measurable e2e wallclock or acceptance improvement on the v4a 13-task
Codex corpus. The Round 3 synthetic microbench's large monotone gradient
was a property of the synthetic workload, not of the techniques.

## 2. Ablation matrix and flag semantics

| Point | T2 disabled | T3 disabled | T4 disabled | Output dir |
|---|---:|---:|---:|---|
| A (T1 only) | ✓ | ✓ | ✓ | `output/track_b_e2e_v4a_ablation/round_1/` |
| B (T1+T2) | | ✓ | ✓ | `output/track_b_e2e_v4a_ablation/round_2/` |
| C (T1+T2+T3) | | | ✓ | `output/track_b_e2e_v4a_ablation/round_3/` |
| D (all on) | | | | `output/track_b_e2e_v4a/round_0/` (existing baseline) |

Flags are written by `scripts/run_track_b_v4a_e2e_ablation.py` to
`/tmp/lumo_track_b_runtime_flags.json` inside the
`lumo-vllm-track-b-suffix` container. The `_lumo_track_b_disabled`
patch in `vllm/v1/spec_decode/suffix_decoding.py` reads this file and
gates the techniques. `True` = technique disabled.

Container, sample hash, runtime config hash, system prompt, codex CLI
version, and codex command template were all held constant. The only
varying input across A/B/C/D is the runtime flags file.

## 3. The Round 3 microbench gradient does not reproduce

Round 3's ablation (`scripts/run_track_b_round3_ablation.py`) ran a
synthetic 5×3 microbench (5 canned sessions × 3 history depths) at
`/v1/responses` directly. It reported (per the prior session context):

- T1 only: 33.5 % acceptance
- All on: 78.9 % acceptance

In the same flag configurations on the real v4a Codex corpus:

- T1 only (point A): 0.545 aggregate acceptance, tool-call 0.556
- All on (point D): 0.521 aggregate acceptance, tool-call 0.532

The Round 3 microbench's 33-pp gradient becomes **+2.4 pp** in real
e2e, and the sign is inconsistent across techniques. The synthetic
microbench was measuring something other than the techniques' real-
traffic impact.

Plausible reasons the synthetic microbench was misleading:

1. **Canned session content.** The Round 3 sessions were 5 stylized
   shell+apply_patch tool-call traces with `FILE_BLOB` × 25 padding.
   These are dense in the exact n-gram patterns T2/T3/T4 were designed
   to catch. Real Codex traffic has higher entropy per turn and a
   different tool-call sequence shape.
2. **Short responses.** `max_output_tokens=64` per turn. The acceptance
   measurement was dominated by 1-2 speculative-decode windows per
   call, where T2/T3/T4's fallback heuristics had outsized weight.
3. **Reset-per-bench prefix cache.** Each microbench point reset the
   prefix cache, so every measurement was cold-prefill dominated. The
   acceptance signal was being mixed with prefill-shape signal.
4. **Suffix index state mismatch.** The microbench's suffix index was
   built fresh for each point with no historical traffic. Real-traffic
   T1 has thousands of prior turns of suffix state to draw from; it
   already covers most of what T2/T3/T4 would add on top.

In short, the synthetic microbench measured "in an artificial
distribution where the suffix index is empty and the patterns are
crafted to exercise the fallback paths, what do the fallback paths
contribute." That is not what real Codex inference looks like.

## 4. Per-regime decomposition across ablation points

Decode-time share is the upper bound on translatable wallclock impact
from any drafter improvement on that regime. All four points have
~93-95 % of decode-time in tool-call regime.

| Point | Tool-call decode_s | Reasoning decode_s | Tool-call rows | Reasoning rows |
|---|---:|---:|---:|---:|
| D (all on) | 317.3 s (93.2 %) | 23.3 s (6.8 %) | 87 | 8 |
| A (T1 only) | 269.9 s (94.4 %) | 15.9 s (5.6 %) | 78 | 8 |
| B (T1+T2) | 282.6 s (94.9 %) | 15.2 s (5.1 %) | 85 | 9 |
| C (T1+T2+T3) | 283.8 s (92.3 %) | 23.8 s (7.7 %) | 89 | 12 |

Per-regime acceptance across points:

| Point | Tool-call accept | Reasoning accept | Aggregate accept |
|---|---:|---:|---:|
| D (all on) | 0.532 | 0.230 | 0.521 |
| A (T1 only) | 0.556 | 0.214 | 0.545 |
| B (T1+T2) | 0.562 | 0.267 | 0.551 |
| C (T1+T2+T3) | 0.551 | 0.182 | 0.534 |

Tool-call acceptance range: 0.030 over 4 points × ~80 rows each. With
binomial sampling variance σ ≈ √(p(1-p)/n) ≈ 0.06 per point, a 0.030
between-point delta is well within ±1 σ. Treat as flat.

Reasoning acceptance range: 0.085 over 4 points × ~10 rows each. With
n=10 and p≈0.2, σ ≈ √(0.2×0.8/10) ≈ 0.13. The observed range is
within ±1 σ. Treat as flat with high uncertainty.

## 5. Cohort soft-trend on quirk recovery

The clean/recovered/exhausted cohort counts show a soft monotone trend
that the wallclock numbers do not:

| Point | clean | retry_recovered | retry_exhausted | tasks_with_no_clean |
|---|---:|---:|---:|---:|
| D (all on) | 18 | 16 | 5 | 2 |
| C (T1+T2+T3) | 17 | 16 | 6 | 2 |
| B (T1+T2) | 17 | 15 | 7 | 4 |
| A (T1 only) | 16 | 15 | 8 | 3 |

Adding techniques shifts attempts toward "clean on first try" and away
from "retry exhausted" by ~2-3 attempts per direction. This is the
**only** measurable effect of T2/T3/T4 in this study, and it's at the
level of a few attempts out of 39 measured.

Mechanism is unclear — possibly the techniques shape the SSE-stream
event ordering in a way that's less likely to trigger Codex 0.128.0's
zero-token quirk. But the effect is too small (and the underlying
Codex bug too dominant) to be load-bearing.

## 6. Caveats

- **One sweep per ablation point, 4 attempts/task = ~16-18 clean
  measurements per point.** Sample-median noise floor is ~10 %. The
  observed clean-median range (8.5 %) does not exceed that. A 4×
  larger sweep might tighten the bounds but is unlikely to invert the
  conclusion: tool-call acceptance is bounded above by ~0.56 across
  all configurations and is already there with T1 alone.
- **13 hard-coded SWE-style tasks.** Generalization to other Codex
  workloads (Q&A, long-form planning, multi-modal, etc.) is not
  assured. The reasoning regime is under-sampled here (5-7 % of decode
  time); a more reasoning-heavy corpus might give T2/T3/T4 different
  leverage. But Round 3 also used SWE-style synthetic traces, so on
  the corpus the techniques were tuned for, they do not deliver.
- **Codex 0.128.0 zero-token quirk persists** (5-8 attempts per point
  exhausted retries). Mitigated by `--zero-token-retries=3` so all
  13/13 tasks achieved at least 1 successful attempt per ablation
  point. Two tasks remained quirk-fragile across points D and C; one
  more (multi-tool-transaction-repair) was added in A and B,
  fanout-fullstack-release-blocker was added in A. The clean-aggregate
  comparison excludes these tasks per point.
- **No spec_decode config change between points.** Same
  `runtime_speculative_config_suffix` activation sentinel across all 4
  points; the suffix drafter is identically configured. The only
  varying parameter is the file-based flag set inside the container.

## 7. What this changes about Round 4b strategy

### 7.1 The harness-coupled techniques (T2/T3/T4) should not be the
default measurement story for Round 4b drafter work

If T2/T3/T4 contribute zero measurable wallclock on the v4a corpus,
then any future drafter comparison that anchors on "all on" vs "T1
only" is anchoring on noise. The clean apples-to-apples comparison
should be **the new drafter (MTP-N, learned-drafter, alternative
suffix configs) vs T1-baseline-on-v4a**, not vs "all on".

### 7.2 Tool-call acceptance has a hard ceiling around 0.56 with the
current suffix drafter, regardless of harness-oracle help

If 0.532 (D) → 0.562 (B) is the entire envelope of harness-oracle
contribution to tool-call acceptance on real Codex traffic, then
**lifting tool-call acceptance further requires a different drafter
architecture entirely**, not more harness-oracle features bolted on.
This matches the parent agentic-saturation plan's §6.5 framing: T1
already covers most of the achievable acceptance in the regime it was
designed for.

### 7.3 MTP-1 test priority is unchanged from §3 of the per-regime
report — still deprioritized

Reasoning regime is 5-8 % of decode-time across all 4 points; a
reasoning-targeted drafter (MTP-1) has the same wallclock leverage
ceiling regardless of T-flags. The per-regime report's
deprioritization recommendation stands.

### 7.4 The real next experiment is corpus expansion, not drafter swap

If T2/T3/T4 are flat on the v4a corpus but Round 3 microbench claimed
33-pp gradient, the missing piece is the workload distribution. Two
candidate diagnostics for what to measure next:

1. **A reasoning-heavy corpus.** Add 3-5 tasks dominated by Codex's
   reasoning-mode output (planning, root-cause analysis, multi-step
   inference). Re-run the 4-point ablation. If T2/T3/T4 contribute
   measurably on this corpus, the v4a corpus is the wrong shape; if
   they still don't, the techniques don't carry their weight.
2. **A traffic-replay corpus.** Capture 1000+ real production Codex
   turns and replay them through suffix decoding at each ablation
   point. Acceptance per regime over the replay tells us what the
   suffix drafter sees in the wild, independent of any 13-task suite.

Either gives Round 4b a sturdier predicate than "Round 3 microbench
said 33→79 so techniques are good."

## 8. Implementation diff (files touched)

**New scripts:**
- `scripts/run_track_b_v4a_e2e_ablation.py` — driver that sets flags,
  runs sweep, recomputes clean wallclock, slices proxy capture by
  point window, runs per-regime aggregation. Restores flags at end.

**Bug fix:**
- `scripts/build_track_b_per_regime_acceptance.py` — fixed
  `format_table` TypeError when a regime has `None` aggregate
  acceptance (occurs when a regime has 0 drafted tokens). Now prints
  `n/a` in such cells. Caught when sweep B produced one `unknown`-
  regime row with 0 spec_decode draft tokens.

**New docs:**
- `docs/reports/auto_research/track-b-round4b-e2e-ablation-20260511.md`
  (this file)

**Artifacts (gitignored under `output/`):**
- `output/track_b_e2e_v4a_ablation/round_{1,2,3}/round_summary.json`
- `output/track_b_e2e_v4a_ablation/round_{1,2,3}/round_summary_clean.json`
- `output/track_b_e2e_v4a_ablation/round_{1,2,3}/per_regime_acceptance.json`
- `output/track_b_e2e_v4a_ablation/round_{1,2,3}/request_metrics.<label>.jsonl`
- `output/track_b_e2e_v4a_ablation/ablation_run_log.json`

## 9. Reproduce

```bash
# 1. Confirm container live + ablation patch present
docker exec lumo-vllm-track-b-suffix bash -lc \
  "grep -c '_lumo_track_b_disabled' /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/suffix_decoding.py"

# 2. Run all 3 ablation points (D = existing v4a baseline, no re-run)
mkdir -p output/track_b_e2e_v4a_ablation
.venv/bin/python scripts/run_track_b_v4a_e2e_ablation.py \
  --out-root output/track_b_e2e_v4a_ablation \
  --repeat 4

# 3. Inspect artifacts
ls output/track_b_e2e_v4a_ablation/round_{1,2,3}/round_summary_clean.json
ls output/track_b_e2e_v4a_ablation/round_{1,2,3}/per_regime_acceptance.json
cat output/track_b_e2e_v4a_ablation/ablation_run_log.json
```

Total wall: ~73 minutes (3 sweeps × ~24 min/sweep + flag-flip overhead).

## 10. Key files

- Driver: `scripts/run_track_b_v4a_e2e_ablation.py`
- Per-point round dirs: `output/track_b_e2e_v4a_ablation/round_{1,2,3}/`
- Baseline (point D): `output/track_b_e2e_v4a/round_0/`
- Per-regime aggregator (bug-fixed): `scripts/build_track_b_per_regime_acceptance.py`
- Run log: `output/track_b_e2e_v4a_ablation/ablation_run_log.json`
- Round 3 synthetic microbench (for comparison): `scripts/run_track_b_round3_ablation.py`
