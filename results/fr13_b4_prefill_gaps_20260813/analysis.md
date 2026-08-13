# Width-4 out-of-decode wall: per-gap attribution (2026-08-13)

Companion prose to `gaps.json` (reducer: `scripts/fr13_b4_prefill_gaps_reduce.py`, commit c28f76a62).
Evidence: the 539-step width-4 nsys capture (results/fr13_b4_width4_nsys_20260813) + census + admission/orchestrator ledgers.

## HEADLINE: the 40.7% is 95.5% GPU-BUSY. It is chunked-prefill compute, not idle.

Across the 360.19 s window the GPU is idle 18.50 s (5.1%). Longest idle interval anywhere: 584 ms; longest inside a gap: 197.7 ms. The "agent thinking on alienware" reading of the gaps is falsified — with 4 co-resident agents, one agent's think time is covered by the other three. It costs throughput as width dilution inside decode steps (effective_concurrency 2.48 against 4 slots), never as wall.

Why the gaps exist: `fr13.fixed32.step` NVTX is pushed only on a pure-decode pass. The 33 gaps hold 144 mixed forward passes — chunked prefill co-batched with resident spec-decode rows. Identity is exact: `sample_readback` fires 684 times = 540 step ranges + 144 gap passes, and all 144 land inside a >1 s gap.

## Classification

| class | s | % of window (360.19 s) | % of the 40.7% |
|---|---:|---:|---:|
| (a) chunked-prefill compute | 112.43 | 31.2% | 76.8% |
| (d) co-batched decode compute | 27.45 | 7.6% | 18.7% |
| (b)+(c) GPU idle in gaps | 6.55 | 1.8% | 4.5% |
| out-of-decode total | 146.44 | 40.7% | 100% |
| inside pure-decode steps | 213.75 | 59.3% | — |

(c) is not separable from (b); bounded jointly. (d) is decode progress the pure-decode census declines to count — not a cost. Classification is data-driven: kernel classes with zero in-step instances are prefill by construction (97.19 s floor); shared classes give decode its own in-step per-pass rate and prefill only the excess (15.35 s). Plain sum 140.01 s vs union 139.89 s — serial execution, so kernel time removed IS window wall removed.

Per-gap table: see gaps.json. Every gap is 88–96% busy; not one is a stall. The three biggest gaps (20.58 s / 14.70 s / 11.75 s) are the three in-window task admissions (cold prefills 10–24 k tokens, 47.03 s = 32% of the out-of-decode mass); the other 30 are per-turn suffix re-prefills (1.4–4.8 k tokens, 2–5 passes).

## Class-(a) priced

137,128 prefill tokens in-window; 112.43 s → 0.820 ms/token, 1,220 tok/s (CUPTI-perturbed). Per cold admission: 10–24 k tokens, 8.6–15.9 s. Per agent turn thereafter: 1.4–4.8 k tokens, 1.0–5.5 s. Token counts read off `silu_and_mul_per_block_quant_kernel` gridX (== batch token count; shows `long_prefill_token_threshold=1024` directly).

Prefill neither delays decode nor fills idle: it runs IN THE SAME forward pass (all 144 mixed passes carry chunked GDN prefill kernels AND the recurrent decode update, 48/pass each; zero pure-prefill passes). vLLM chunked prefill already fuses at the maximum rate.

What prefill isolation would buy: a 5.6% REGRESSION. The 144 mixed passes carry 10,720 decode rows for 27.45 s riding the prefill chunk's weight reads; isolated they need 144 own passes at the width-2.33 interpolated wall = 47.7 s. Net +20.3 s.

## Lever-transfer table (D = transfer × 0.5934)

| family | in-step s | out-of-step s | window s | % win | transfer | D |
|---|---:|---:|---:|---:|---:|---:|
| GEMM fp8 blockwise | 70.77 | 62.06 | 132.82 | 36.9% | 1.877 | 1.114 |
| other | 38.38 | 26.05 | 64.43 | 17.9% | 1.679 | 0.996 |
| FA2 splitkv tree (gqa_pair target) | 37.70 | 7.07 | 44.76 | 12.4% | 1.188 | 0.705 |
| FA2 splitkv causal (prefill + MTP) | 2.33 | 24.66 | 26.99 | 7.5% | 11.60 | 6.885 |
| GDN tree scan (single_launch) | 22.15 | 4.28 | 26.43 | 7.3% | 1.193 | 0.708 |
| bf16 GEMM (LM head / misc) | 16.26 | 4.98 | 21.23 | 5.9% | 1.306 | 0.775 |
| unified attention | 8.43 | 2.27 | 10.70 | 3.0% | 1.269 | 0.753 |
| GDN chunked (prefill-only) | 0.00 | 7.45 | 7.45 | 2.1% | ∞ | ∞ |
| GDN delta-rule update | 5.91 | 1.20 | 7.11 | 2.0% | 1.203 | 0.714 |

Key: the banked gqa_pair candidate is scoped final_fixed32_b4_full_graph_only — mixed passes dispatch PIECEWISE/NONE, so its 7.07 s out-of-step FA2 tree headroom is unreached. The causal FA2 instantiation (24.66 s, 6.8% of window) is the SAME kernel body differing in one bool template argument (Is_causal). If the gqa_pair KV-layout mapping transfers, the FA2 family total is 71.75 s (19.9%) at combined D = 1.06. Realized rank-1 gain today: candidate engages on 225/540 steps (26.8% of window wall) → −7.17% there = 1.92% of window wall.

## Window vs arm (honesty)

The capture is a deliberately prefill-heavy 360 s slice of a 4,431 s arm. Arm-level: inside pure decode 67.7%, mixed-pass wall ~29.4%, residual idle ≤2.9%. Window overstates prefill share by ~8 pp; direction unchanged.

## VERDICT

Prefill contention is RETIRED as a serving lever and restated as a workload constant:
- chunked-prefill compute 112.43 s: recoverable only by fewer tokens (harness) or cheaper prefill kernels — not scheduling.
- prefill FA2 (causal) 24.66 s: OPEN — the largest single unreached block (needs Is_causal=1 reach).
- prefill GEMM ~59 s: genuine FLOPs, floor.
- co-batched decode 27.45 s: not a cost (cheaper than isolated).
- idle 6.55 s: below threshold.
- prefill isolation: −5.6% regression. Do not build.
- oversubscription at the out-of-decode wall: wrong target (no idle to fill). Demand-side value lives as width dilution INSIDE decode steps (eff 2.48 vs 4) — raises aggregate, does not lower step wall.

## RECOMMENDED NEXT ACTION

Test whether the width-4 gqa_pair mapping reaches Is_causal=1. (1) Offline: read the forked FA2 source; establish whether the gqa_pair path instantiates for Is_causal=1 and whether candidate_scope can widen past final_fixed32_b4_full_graph_only without touching the sealed decode dispatch. If structurally decode-only, this dies free. (2) If it survives: a paired arm with the candidate additionally bound to the prefill dispatch, read on WINDOW WALL / aggregate tok/s (a prefill-side gain cannot appear in pure-decode step wall). Falsification: 10% of 24.66 s = 0.69% of window wall; needs the aggregate basis and several passes.

## DOES NOT CLAIM

No acceptance/regression reading (CUPTI-attached arm; ~4.3% inflation on the tok/s figures). The (a)/(d) split inside shared kernel classes is an attribution rule (conservative toward (d); rule-free prefill-exclusive floor 97.19 s = 27.0%). Arm extrapolation assumes window mixed-pass cost, errs safe. Sealed denominators and the gqa_pair re-test stand as banked.
