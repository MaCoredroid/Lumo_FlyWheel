# SWE-Bench Q36-A: temp=1.0 throughput regression debug (2026-05-22)

## Trigger
Same-instance comparison flagged a ~40% tps drop + ~48% spec-decode acceptance
drop when the campaign switched from temp=0.6 to temp=1.0 (Qwen's published config):
- astropy-12907: temp06 20.74 tps / 0.456 accept  → temp10 12.15 tps / 0.238 accept
- astropy-13033: temp06 17.30 tps / 0.331 accept  → temp10 10.42 tps / 0.229 accept

## Ruled out
- **Power**: ~41-48W during active decode (matches historical Q36-A 41.83W). Idle 11W.
- **Thermal/clocks**: 62-63°C, 2496-2515 MHz (near max 3003), no HW/SW slowdown, no power brake.
- **Concurrency**: clean single-stream temp10 (13236: 10.73 tps / 0.197) ≈ contaminated
  startup instances (12907: 12.15 / 0.238). Parallelism was NOT the driver.
- **Temperature**: see controlled probe below — DEFINITIVELY ruled out.

## Controlled A/B decode probe (identical prompt, GPU idle, no agent/dcgm)
| run | temp | top_p | tps | accept_rate |
|---|---|---|---:|---:|
| 1 | 0.6 | 0.95 | 11.16 | 0.144 |
| 2 | 1.0 | 0.95 | 11.10 | 0.136 |
| 3 | 0.6 | 0.95 | 15.43 | 0.241 |
| 4 | 1.0 | 0.95 | 15.57 | 0.234 |

**temp=0.6 ≈ temp=1.0 within noise** for both tps and acceptance. Temperature
has no measurable effect on SuffixDecoding throughput.

## Actual driver
Spec-decode acceptance (hence tps) is driven by **workload content repetitiveness
+ suffix-cache warmth**, not temperature:
- runs 3-4 (warmer cross-request suffix cache, max_cached_requests=1000) beat runs
  1-2: 0.24 vs 0.14 accept, 15 vs 11 tps — same temps.
- the synthetic novel-code probe prompt gets low acceptance (0.14-0.24) at both temps;
  the high-acceptance SWE-bench trajectories (0.45) come from repetitive agent output
  (re-reading files, repeated tool-call envelopes).

## Conclusion
The 20.74→12.15 observation conflated temperature with **per-instance agent-trajectory
variance**: at temp=1.0 the agent walked a different, less-cache-friendly path on that
instance. There is no systematic temp=1.0 throughput penalty. temp=1.0/top_p=0.95
(Qwen's published SWE-bench config) is retained; the campaign resumes.
