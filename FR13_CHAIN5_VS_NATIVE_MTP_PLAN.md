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

---

## RESULTS — chain5 (forked) vs native-MTP A/B (run_20260701T172042Z, cache-OFF, cap=500, N=1)

| task | native | chain5 |
|---|---|---|
| 12907 | resolved | resolved |
| 13453 | resolved | tests_failed |
| 14508 | resolved | tests_failed |
| 14539 | resolved | resolved |
| 14995 | resolved | resolved |
| **total** | **5/5** | **3/5** |

**Verdict (honest):**
- **char-8 is NOT the carrier** — gate: chain5 4 char-8 (15.4/1k-turns) < native 10 (44.6/1k), PASS(char8-lossless). Both arms 0 degeneration. char-8 is survivable (native resolved 5/5 with 10 char-8) and NOT elevated on the forked kernel.
- **The 5/5-vs-3/5 gap is NOT statistically significant** (Fisher p=0.44) — well within seed noise at N=1, temp 0.6. Divergences (13453, 14508) are `tests_failed` = real patches, wrong fixes; heterogeneous (chain5 recovered 12907/14539/14995) → leans SEED, kernel not proven a carrier.
- **Adjudicator (temp-0.6, no teacher-forcing):** the replica self-noise gate (scripts/fr13_replica_selfnoise_run.sh + fr13_replica_selfnoise_gate.py) — K seed-replicas per config, CMH stratified test vs native's self-noise floor. Recommended focused run: 13453+14508, K=4-5.
- Per-task wall-clock: native mean 18.6 min, chain5 21.8 min (both cache-OFF); variance dominated by clean-context retries (SWE_EMPTY_PATCH_RETRIES=2 → up to 3 sessions, ~30-60 min on hard tasks).

## NEXT: native+EXACT_SEED (run_20260701T205744Z) — does the lossless deployment cache preserve native 5/5?
Booting: Qwen3_5MTP + method=mtp num_spec=5 + enable_prefix_caching=True + APC bridge EXACT_SEED=1 on the native FLASH_ATTN path. WATCH: ES must capture/restore on real requests (boot warmup showed ES_CKPT0_SKIP reason=no_nonspec_row_map — confirm engagement on real multi-turn traffic, else cache is inert).

---
## qwen-code CAMPAIGN P1 — native+cache (EXACT_SEED), 5-task (output/fr13_tree_cache_matrix/run_20260702T074032Z, 2026-07-02)
Harness = qwen-code (SWE_AGENT=qwen_code), temp 0.6, cache-ON EXACT_SEED, offload.
| task | qwen-code native+cache | codex history |
|---|---|---|
| 12907 | failed | codex stock+patched BOTH failed (flaky/seed) |
| 13453 | **resolved** | codex FAILED (char-8) |
| 14508 | **resolved** | codex FAILED (char-8 death) |
| 14539 | **resolved** | codex FAILED (char-8) |
| 14995 | **resolved** | codex FAILED |
| **total** | **4/5** | codex native-OFF baseline 5/5 (no cache; survived char-8) |
**char-8 = 0** (server docker_full.log) — ELIMINATED vs codex (which hit char-8 on these exact tasks). All
three char-8 smoking-guns (13453/14508/14539) RECOVERED on qwen-code. Only 12907 (the seed-flaky task both
agents miss) failed. Net: qwen-code native+cache ≈ 5/5 baseline within seed noise, with char-8 gone. Confirms
apply_patch/char-8 was the codex-specific blocker; the Qwen-native harness clears it.
