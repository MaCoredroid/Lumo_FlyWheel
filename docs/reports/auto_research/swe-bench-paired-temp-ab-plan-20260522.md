# SWE-Bench Q36-A — Paired temperature A/B plan (2026-05-22)

Clean isolation experiment: **only temperature varies**, all else held constant.
Same 55 Verified instances run twice. Subset:
`docs/reports/auto_research/swe-bench-paired55-verified-instances-20260522.json`
(first 55 of tier2 verified; last id = `django__django-11555`).

Held constant both rounds: Q36-A (config A suffix spec-decode) + Bundle B fixes,
top_p=0.95, max_output_tokens=80000, parallel=1, agent-wall 1800s, eval-timeout
1800s, eval offloaded to alienware (x86), proxy auto-continue + retry-400.

## Round 1 — temp10 (RUNNING)
- **temp=1.0, top_p=0.95**
- out: `output/swe_bench_q36_a_temp10/verified`
- subset: paired55 (skip-existing; instances #1-6 already done from the 500-run)
- Stop condition: after instance #55 (`django__django-11555`). Orchestrator exits
  on its own (subset is exactly 55).

## Round 2 — temp06 + Qwen-top_p (PENDING — starts after Round 1 finishes)
- **temp=0.6, top_p=0.95**
- out: `output/swe_bench_q36_a_temp06_tp095/verified` (NEW dir — do NOT reuse the
  old `swe_bench_q36_a_temp06`, which had top_p=1.0 unmanaged and is a different
  config).
- Same paired55 subset, all 55 from scratch in the new dir.
- **Switch procedure**: restart codex-bench-proxy with
  `LUMO_PROXY_FORCE_TEMPERATURE=0.6` (keep `LUMO_PROXY_FORCE_TOP_P=0.95` and all
  other knobs). vLLM is NOT restarted. Verify the proxy env shows 0.6 before launch.
- Budget: ~24h. Stop after #55, then report the paired comparison.

> NOTE: Round 2 differs from the original Round 4b temp06 (top_p=1.0 unmanaged).
> Document it as **"temp06 + Qwen-top_p"** for the paired comparison.

## Supervise-loop state machine
- **phase=round1_running**: monitor `output/swe_bench_q36_a_temp10/verified`; commit
  each finished instance; restart orchestrator on crash (same bounded 55-subset cmd).
  When orchestrator process is gone AND 55 instances have runner_metadata →
  advance to round2_setup.
- **phase=round2_setup**: confirm Round 1 fully committed; restart proxy with
  FORCE_TEMPERATURE=0.6 / FORCE_TOP_P=0.95; verify env; launch orchestrator on the
  paired55 subset → `output/swe_bench_q36_a_temp06_tp095/verified` (NO skip needed,
  fresh dir). Set phase=round2_running.
- **phase=round2_running**: monitor temp06_tp095 dir; commit each instance; restart
  on crash. When 55 done → phase=done.
- **phase=done**: stop loop; write paired comparison report (resolved-rate +
  tps/accept per round); do NOT grade beyond the harness verdicts already emitted.

Live phase tracked in: `output/swe_bench_q36_a_paired_state.txt`
