# Track B E2E Round 0 v2 Report

Generated: 2026-05-08T21:37:59Z

## Summary
- v2 tasks_completed: **12**, trusted_task_count: **12**, median_wallclock_s: **109.070821**, aggregate_wallclock_s: **1309.666557**
- v2 runtime_config_hash: `sha256:841fb0ea93184839dc7e85f93911f65ff385a3ed0fb9d9ff1250c4c510c4d542`
- v1 (reduced-contract) tasks_completed: 13, median_wallclock_s: 95.023, aggregate_wallclock_s: 1263.267
- v2 median wallclock vs v1: **109.07s vs 95.02s (+14.8%)**

## Per-task wallclock (measured attempts only; cold run_01 discarded)

| task | attempts | measured | median elapsed | p90 elapsed | retries | non-zero exit |
|---|---:|---:|---:|---:|---:|---:|
| dead-flag-reachability-audit/v1-clean-baseline | 4 | 3 | 108.46s | 110.16s | 0 | 0 |
| fanout-fullstack-release-blocker/v1-clean-baseline | 4 | 3 | 113.53s | 115.34s | 6 | 0 |
| incident-evidence-synthesis/v1-clean-baseline | 4 | 3 | 111.36s | 118.30s | 3 | 0 |
| multi-tool-transaction-repair/v1-clean-baseline | 4 | 3 | 107.07s | 112.37s | 7 | 0 |
| plugin-scaffold-alignment/v1-clean-baseline | 4 | 3 | 114.11s | 139.51s | 6 | 0 |
| policy-aware-request-resolution/v1-clean-baseline | 4 | 3 | 123.31s | 126.07s | 3 | 0 |
| release-note-to-plan-translation/v1-clean-baseline | 4 | 3 | 112.11s | 114.21s | 5 | 0 |
| responses-sdk-adapter-cutover/v1-clean-baseline | 4 | 3 | 103.41s | 106.61s | 4 | 0 |
| responsive-checkout-visual-regression/v1-clean-baseline | 4 | 3 | 97.55s | 99.59s | 1 | 0 |
| security-audit-hotfix-remediation/v1-clean-baseline | 4 | 3 | 109.68s | 110.82s | 4 | 0 |
| skill-router-contract-upgrade/v1-clean-baseline | 4 | 3 | 107.87s | 112.35s | 7 | 1 |
| sqlalchemy-2-session-modernization/v1-clean-baseline | 4 | 3 | 102.28s | 108.44s | 3 | 0 |
| transcript-merge-regression/v1-clean-baseline | 4 | 3 | 106.79s | 107.61s | 6 | 0 |

## Per-regime acceptance (proxy-capture aggregation)

Total captured rows: **94**
Aggregate accepted/draft: **0.4838226108682074**

| regime | rows | agg accept | p50 accept | p90 accept | p50 decode_tps | tokens out |
|---|---:|---:|---:|---:|---:|---:|
| reasoning | 10 | 0.209 | 0.231 | 0.303 | 10.24 | 858 |
| tool-call | 84 | 0.521 | 0.550 | 0.711 | 33.61 | 8882 |

## §6.5 diagnosis (regime-level)
- `reasoning`: moderate (agg = 0.209); SuffixDecoding has some traction but is not yet pulling its weight — Techniques 1/3 are reasonable next bets
- `tool-call`: strong (agg = 0.521); SuffixDecoding carrying its weight here

## Runtime config

- runtime_config_hash: `sha256:841fb0ea93184839dc7e85f93911f65ff385a3ed0fb9d9ff1250c4c510c4d542`
- live spec_decode: `method=suffix, num_speculative_tokens=12, suffix_decoding_max_tree_depth=32`
- arctic-inference==0.1.2 installed via prelaunch hook
- proxy capture path: /tmp/track_b_e2e_proxy_capture/request_metrics.jsonl
- proxy capture writer: TrackBRequestMetricsCapture (lumo.track_b.vllm_request_metrics.v1)

