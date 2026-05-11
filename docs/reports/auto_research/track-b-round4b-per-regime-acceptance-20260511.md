# Track B Round 4b — Per-Regime Acceptance on v4a Baseline

Generated: 2026-05-11
Status: ARTIFACT LANDED — diagnostic only, no measurement runs

Companion to:
- `track-b-round4a-closeout-20260510.md` (canonical baseline; defines the 95-turn v4a window)
- `track-b-e2e-round4a-measurement-protocol-spec-20260510.md` §15 (round-start warmup architecture)

## 1. Headline

Per-regime acceptance recompute on the v4a baseline's 95-turn proxy-capture cohort. Zero new measurement — pure recompute over `request_metrics.jsonl` filtered to the v4a `ts_request_received` window.

| Regime | rows | agg accept | p50 accept | p90 accept | p50 decode tps | completion tokens | decode_sum_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| **reasoning** | 8 | **0.230** | 0.265 | 0.312 | 9.47 | 228 (2.5 %) | 23.3 s (6.8 %) |
| **tool-call** | 87 | **0.532** | 0.560 | 0.752 | 33.03 | 9,057 (97.5 %) | 317.3 s (93.2 %) |
| **AGGREGATE** | 95 | 0.521 | — | — | — | 9,285 | 340.6 s |
| summary | 0 | — | — | — | — | — | — |

Sample window: `ts_request_received` ∈ [`2026-05-10T21:55:16Z`, `2026-05-10T22:15:43Z`]. Matches closeout §4 "95 v4a turns" cohort exactly.

Artifact: `output/track_b_e2e_v4a/round_0/per_regime_acceptance.json` (schema `lumo.track_b.per_regime_acceptance.v1`).

## 2. Change against v2 (last time per-regime acceptance was measured directly)

The v3 round did not capture per-turn metrics (the closeout headline lists "Spec-decode token acceptance: deferred, no per-turn capture"). v2's per-regime numbers are the prior data point.

| Regime | v2 acceptance | v4a acceptance | Δ |
|---|---:|---:|---:|
| reasoning | 0.209 | 0.230 | +0.021 |
| tool-call | 0.521 | 0.532 | +0.011 |

Both regimes are statistically unchanged. The §6.5 low-acceptance floor for reasoning is persistent: the suffix drafter does not learn the regime, and the harness-oracle techniques (T2/T3/T4) that landed since v2 have not lifted it.

## 3. What this changes about Round 4b priorities

### 3.1 Reasoning regime has acceptance headroom but minimal decode-time share

The gap between reasoning (0.230) and tool-call (0.532) acceptance is 2.3×. That's the same gap v2 showed and the same gap that motivated the MTP-vs-suffix hypothesis: maybe a learned drafter (MTP-1) could close it where the prefix-matching suffix drafter cannot.

But **reasoning is only 6.8 % of total decode-time** in v4a's 13-task corpus (23.3 s of 340.6 s). Even at perfect acceptance (1.0 vs. 0.230), the maximum possible decode-time shrink from a reasoning-only drafter improvement is bounded by that 6.8 % share. At a realistic intermediate (e.g., MTP-1 lifts reasoning to 0.45), the decode-time win is far smaller. After translating decode time → clean wallclock at the v4a decode share (66.8 %), this is well below the 4-attempt sample-median noise floor.

**Implication for Priority 2-i (MTP-1 test):** The MTP-vs-suffix comparison should not be the binding next step. The reasoning regime is too small a share of this 13-task corpus to give MTP a chance of moving the headline. Deprioritize until either (a) the corpus is expanded to exercise reasoning more heavily, or (b) Priority 1 ablation shows tool-call decode time itself is the binding constraint and MTP can plausibly beat suffix there too (much less likely a priori).

### 3.2 Tool-call regime is already well-served by suffix

At 0.532 aggregate / 0.560 p50 / 0.752 p90, suffix is doing what it was designed to do. MTP would have to beat 0.53 in the regime suffix was specifically chosen for. There is no a priori reason to expect that.

### 3.3 Zero summary-regime entries

The 13-task v4a corpus produced zero turns classified as `summary` regime. Either the classifier is conservative or the corpus does not exercise short-summary outputs. Either way, summary regime is invisible to Round 4b drafter measurements on this sample and any technique designed for summary regime would need a different corpus.

## 4. Decode-time share breakdown

The intra-regime decode share figures in the v4a headline (tool-call 66.2 %, reasoning 76.1 %) are decode_sum / (decode_sum + prefill_sum) within each regime — i.e., "of the time spent in turns of regime X, how much is decode." The inter-regime share computed here is decode_sum_X / total_decode_sum — i.e., "of all decode-time, how much was spent in regime X." Both are real; the former tells us where prefill optimization helps, the latter tells us where drafter optimization helps.

| Regime | decode_sum_s | inter-regime decode-time share |
|---|---:|---:|
| reasoning | 23.3 s | 6.85 % |
| tool-call | 317.3 s | 93.15 % |

For drafter techniques (which act on decode), inter-regime share is the upper bound on translatable wallclock impact.

## 5. Caveats

- **Sample size for reasoning is small** (8 rows / 228 completion tokens). The acceptance estimate has wide variance. Worth re-checking after Priority 1 ablation produces a 4× larger v4a sample.
- **Decode tps figures conflate model speed with draft-acceptance.** Tool-call's 33 tps vs reasoning's 9.5 tps gap is partly the lower acceptance (0.230 vs 0.532 → fewer accepted speculative tokens per step → more compute steps per token) and partly that reasoning outputs are sequentially harder for the suffix drafter to predict (longer effective entropy per token). The artifact reports both ratios separately.
- **Sample window is the v4a baseline only.** Round 1-3 historical acceptance numbers (v2 0.209 reasoning / 0.521 tool-call) come from a different proxy-capture window with different runtime config. Direct comparison is approximate.

## 6. Reproduce

```bash
# Filter proxy capture to v4a window
python3 -c "
import json
lo, hi = '2026-05-10T21:55:16Z', '2026-05-10T22:15:43Z'
with open('/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl') as fi, \
     open('/tmp/track_b_e2e_proxy_capture/request_metrics.v4a_round_0.jsonl', 'w') as fo:
    for line in fi:
        s = line.strip()
        if not s: continue
        r = json.loads(s)
        ts = r.get('ts_request_received')
        if ts and lo <= ts <= hi:
            fo.write(s + '\n')
"

# Aggregate
.venv/bin/python scripts/build_track_b_per_regime_acceptance.py \
  /tmp/track_b_e2e_proxy_capture/request_metrics.v4a_round_0.jsonl \
  --out output/track_b_e2e_v4a/round_0/per_regime_acceptance.json \
  --text
```

## 7. What this enables

For Priority 1 (Round 1-3 ablation against v4a baseline): when ablation points produce per-regime acceptance numbers via the same aggregator, the contribution of T2/T3/T4 to each regime separately becomes visible. Expectation: T2/T3/T4 land in tool-call regime (harness oracle is tool-coupled), so reasoning acceptance should be approximately constant across ablation points. If reasoning acceptance moves, that's an unexpected coupling worth investigating.

For Round 4b drafter work overall: the next high-leverage target is not MTP-vs-suffix on reasoning (too small a share). The next high-leverage target is whatever can lift **tool-call acceptance from 0.53 toward 0.75 (current p90)** — a learned drafter, a richer suffix index, or harness-oracle expansion. The 0.53 → 0.75 envelope on 93 % of decode-time is ~5-15× the wallclock leverage of reasoning-regime drafter work on this corpus.
