# FR13 — chain5 (forked-fa2 tree-attn) vs native-MTP: char-8 localization plan

**Date:** 2026-07-01

## Bank: terminated run `run_20260701T072605Z` (killed to redesign)
| arm | config | resolved |
|---|---|---|
| m_e5_ON | chain5 forked-fa2, cache-ON, cap500 | **1/16** |
| m_cat8_OFF | cat8 forked-fa2 tree, cache-OFF, cap500 | **4/16** |
| m_cat8_ON | (booting when killed) | — |
Raw artifacts preserved under the timestamped run dir. Findings in FR13_CHAR8_REGRESSION_FINDINGS.md,
FR9_VS_CURRENT_CHAR8_STATS.md.

## Why this experiment
fr9 resolved **8/16** on these tasks with **native `qwen3_5_mtp`** (stock MTP, no fork, no tree).
Current pipeline gets 1–2/16. Ruled OUT as drivers: model (SAME qwen3.6-27b-fp8), codex (SAME
codex-cli-0.128.0), cache (cache-OFF has MORE char-8, not less), tunnel (payloads intact 1KB–1MB).
**Top remaining suspect = the forked-fa2 TREE_ATTN decode kernel.** `chain5` is the forked kernel running
a *chain* (no branches), so chain5-vs-native isolates the KERNEL as the single variable.

## The A/B (2 arms, single variable = decode kernel)
| axis | Arm A: native-MTP-5 | Arm B: chain5 (forked) |
|---|---|---|
| decode kernel | **stock vLLM `qwen3_5_mtp`, no forked-fa2, no tree** | forked-fa2 TREE_ATTN, chain-5 topology |
| cache | OFF | OFF |
| thinking cap | 500 (held constant) | 500 (held constant) |
| model / codex / harness / subset | identical | identical |
Everything else identical → any char-8 / solve-rate delta is attributable to the kernel.

## Small localization task set (the char-8 smoking-guns)
`subset_char8_localize.json` = **13453, 14508, 14539, 14995** (+ optional 14365) — all fr9-RESOLVED but
current-FAILED via `patch_apply_failed` / char-8. 4–5 tasks × 2 arms ≈ fast (~1–1.5 h/arm cache-off).

**Read-out:**
- native-MTP resolves these (≥3/4) with ~0 char-8 → **the forked-fa2 kernel is the char-8 driver** → reframes
  FR13 (the tree-spec kernel, added for the speed win, broke tool-call generation).
- native-MTP is ALSO ~0/4 with char-8 → kernel exonerated → driver is thinking-cap / vLLM-version / offload;
  next A/B = cap-off, then on-GB10.

## Setup gap (the one build task)
No native-MTP launcher exists in the current script set (all forked/locked, SO-required). Need a
**native-MTP arm**: a launcher = stock `vllm serve /models/qwen3.6-27b-fp8 --speculative-config
'{"method":"qwen3_5_mtp","num_speculative_tokens":5}'` (NO `speculative_token_tree`, NO `FORKED_FA2_SO`
mount, stock attention) wired into `fr13_bigdenom_swe_serve_variant.sh` as a new kind (e.g. `nativemtp5:
LAUNCHER=native`). Instrument BOTH arms with docker_full.log + proxy_request_dumps for per-event char-8
attribution (the prior run's cache-OFF arm lacked these).

## Instrumentation must-haves (from wow45pfqn)
- capture `docker_full.log` per arm (char-8 400s land here)
- capture proxy request/pair dumps on BOTH arms
- tally char-8-per-turn + char-8-on-apply_patch per arm
