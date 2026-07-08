# FR13 — B=4 ALL-CACHE-ON Speed/Quality Matrix (living results doc)

**Status:** IN PROGRESS. `native+cache` running (13/16 graded); `cat8+cache` DONE (16/16); `cat6+cache` QUEUED.
**Maintained by** the 20-min monitor loop (cron `0dd3b309`) — updated + committed each tick with the artifact archive under `results/fr13_b4_cache_matrix/`.
**Facts below are independently extracted from the artifact files** (read-only verification workflow `wf_12862543-571`), not transcribed from memory.

---

## 0. What this matrix answers

Isolate the **tree decode superset** against **native MTP-5**, both with **our forked lossless prefix cache** (all arms cache-ON), on a live agentic SWE-bench-Verified gate. Three cache-ON arms:

| arm | decode | tree | status |
|---|---|---|---|
| **native+cache** | native MTP-5 (linear, 5-token) | none | RUNNING (13/16) |
| **cat8+cache** | branch tree, 8-node depth-5 | TREE_ATTN | **DONE (16/16)** |
| **cat6+cache** | branch tree, 6-node depth-5 | TREE_ATTN | QUEUED |

The two prior **no-cache** arms and **spine5** were **cancelled** (user 2026-07-08 pivot); cat6 was added to test the smaller-tree speed/accept tradeoff.

---

## 1. Configuration (ALL facts)

**Hardware / model:** NVIDIA GB10 (unified memory), Qwen3.6-27B-fp8 (GDN/mamba-hybrid), patched vLLM fork. Agents offloaded to alienware (x86) via the offload proxy; vLLM serves on GB10.

**git HEAD:** `d8af472d` (matrix pivot). Patcher `fr10_phase4_patch_vllm_tree_gdn.py` last committed `37bd90e2` (leak-fix deprecation).

### Common (every arm)
| knob | value | source |
|---|---|---|
| B (max-num-seqs) | **4** | `BSIZE=4` → `MAX_NUM_SEQS_OVR` |
| SWE concurrency | **4** | `CONC=4` → `SWE_CONCURRENCY` |
| **effective decode batch** | **~1** (Running:1 dominant) | `max_num_batched_tokens=1024` prefill throttle + agent tool-gaps |
| agent | **qwen-code** | `SWE_AGENT=qwen_code` + `run_swe_bench_q36_a.py` default |
| nudge | **OFF** | `LUMO_PROXY_AUTO_CONTINUE=0` (both relaunch scripts) |
| wall | **0 (none)** | `WALL=0`/`AGENT_WALL_S=0`; backstop = 600s stream-idle stall watchdog only |
| turn limit | **100000 (effectively none)** | qwen `--max-session-turns 100000` |
| temperature | **0.6** | `LUMO_PROXY_FORCE_TEMPERATURE=0.6` (proxy-forced), seed 0 |
| subset | **subset_b4_sixteen** | SWE-bench_Verified, 16 astropy tasks, official per-instance images |
| APC | `--enable-prefix-caching --enable-chunked-prefill --mamba-block-size 1024 --mamba-ssm-cache-dtype float32 --max-num-batched-tokens 1024 --block-size 1024` | #45238 overshoot fix |
| cudagraph | **FULL_AND_PIECEWISE** (GRAPH mode, not eager) | launcher default |
| gpu_mem_util | 0.78 | variant serve override |
| max_model_len | 131072 | launcher default |

**Deprecated-OFF (code-level, can't be re-enabled):** `FR13_APC_EXACT_SEED`, `FR13_APC_HIT_RECURRENT_SUFFIX`, `FR13_APC_BLOCK_REFOLD`, `FR13_APC_REFOLD_TO_SNAPSHOT` are force-set to `"0"` at GDN import (patcher L1037-1038, before every request-time gate; TP=1 so one site neutralizes all). Reason: block-hash-keyed `_fr13_es_ckpt` store's only reaper is block-eviction which never fires → host OOM ~3.5h (killed the earlier es=1 native+cache at 8/16).

### Per-arm decode/tree config
| | native+cache | cat8+cache | cat6+cache |
|---|---|---|---|
| KIND | `nativemtp5_exseed` | `cat8` | `cat6root` |
| launcher | forked (patcher only) | forked | forked |
| decode | naive_mtp (linear) | tree | tree |
| spec method | qwen3_5_mtp, **5-token** | qwen3_5_mtp, **8-node tree** | qwen3_5_mtp, **6-node tree** |
| tree shape | — (no tree) | `[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]` | `[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]` |
| attention | **FLASH_ATTN** | **TREE_ATTN** | **TREE_ATTN** |
| EXPECT ratio | 5 | 8 | 6 |
| cache flags | `FR13_ENABLE_APC=1` (base forked cache) | `FR13_ENABLE_APC=1` | `FR13_ENABLE_APC=1` |
| stateless-tree flags | **none** | `COMMIT_TO_RUNNING_ROW=1 TREE_RUNROW_INIT=1 BURN_NODE_BANK=1` | same as cat8 |

native's tree machinery is explicitly un-leaked via XFLAGS (`FR13_FA2_TREE_BIAS=0 FR13_TREE_SAMPLE_ROW=0 FR13_CONV_COMMITTED_PATH=0` …), asserted live in `container_env.txt`.

---

## 2. Speed (the CLEAN metric = per-committed-token = `s_per_fwd_gpu / accept_per_event`)

Per-request decode-TPS is **confounded** (prefill share, concurrency, task-mix) — not used for conclusions.

| arm | accept/event | s_per_fwd_gpu (window) | **per-committed-token** | s_per_fwd_gpu (cumulative) | per-req TPS |
|---|---|---|---|---|---|
| **cat8+cache** (FINAL) | **3.588** | **131.2 ms** | **36.6 ms/tok** | 163.7 ms (8936.56s / 54575) | 15.04 |
| **native+cache** (es=0, PENDING) | pending reduce | pending | **pending** | 147.0 ms (4820.88s / 32786) | pending |
| native+cache (es=1 OOM-partial, 8-task **preview only**) | 3.465 | 118.5 ms | 34.2 ms/tok | — | 16.33 |

**Preliminary read (not final — native es=0 reduce not yet run):**
- Tree per-forward overhead: **+10.7%** window (131 vs 118 ms) / +11.4% cumulative (164 vs 147 ms) — verifying a wider tree (TREE_ATTN irregular access).
- Tree accept edge: **+3.6%** (3.588 vs 3.465).
- Net: native ~**7% faster per committed token** (36.6 vs 34.2). The overhead exceeds the accept gain.

**Why this is the strong form:** both arms run at **effective batch ~1** (cat8 measured: mean 1.31, 80% at Running=1, dist `R0=28 R1=1466 R2=150 R3=110 R4=72`, n=1826; native live-observed ~1.2, saved dist lands at teardown). Batch-1 is HBM-bound — the tree's *favorable* regime (weight-read floor dominates, accept is the lever, extra compute nearly free). The tree losing per-committed-token *there* means it's the tree-attn kernel overhead vs a thin accept edge, **not** a batch-amortization artifact.

**Caveat:** accept is task-mix dependent (same native config: 3.045 in a different campaign vs 3.465 here; live accept swings 2.58↔5.38 within minutes). Only the **full-run matched 16-task** cat8-vs-native comparison is apples — that's what native+cache finishing produces.

---

## 3. Quality (live SWE-bench-Verified, honest give-up gate)

### Grades
| arm | graded | PASS | give-ups (empty patch) | fail (wrong patch) |
|---|---|---|---|---|
| **cat8+cache** | 16/16 | 6 | **5** | 5 |
| **native+cache** | 13/16 | 7 | **1** | 5 |

- cat8 PASS: 12907, 13236, 13453, 14096, 14309, 14995 · GU: 13398, 13579, 14508, 14539, 14598 · fail: 13033, 13977, 14182, 14365, 14369
- native PASS: 12907, 13236, 13453, 13579, 14309, 14508, 14539 · GU: 14096 · fail: 13033, 13977, 14182, 14365, 14369 · **pending: 13398, 14598, 14995**

### Head-to-head (13 common tasks)
| divergence | tasks | detail |
|---|---|---|
| **cat8 GU → native PASS** | **13579, 14508, 14539** | native solves all 3, and **much faster**: 22m / 42m / 19m vs cat8's give-up grind **2.8h / 52m / 51m** |
| native GU → cat8 PASS | 14096 | native 11m give-up vs cat8 33m pass |
| both fail | 13033, 13977, 14182, 14365, 14369 | genuinely hard (both wrong) |
| both pass | 12907, 13236, 13453, 14309 | agreement |

### Tree-agentic-degradation — verdict PENDING on 13398 + 14598
Signal so far is strong: the tree (cat8) not only gives up where native solves, it **wastes hours meandering** — cat8 burned **9.6 agent-hours across its 5 give-ups** (13398 alone ran **4.5h**), while native converges on the same tasks in minutes. But the two remaining discriminating give-up tasks (13398, 14598) haven't graded on native yet:
- **native passes both** → degradation confirmed (5/5 of cat8's give-ups solved by native).
- **native also gives up** → those two are genuinely hard (task difficulty), gap narrows.

The tree's PASS-rate deficit is mostly the **give-up (non-convergence)** gap, not wrong-patches: excluding give-ups, cat8 was 6/11 vs native 7/12 patched — much closer. The tree degrades by making the agent *meander*, not by producing wrong code.

---

## 4. Red-team ledger (corrections included)

1. **Metric fixed:** per-committed-token, not per-request-TPS. Both arms are forked so both carry the `s_per_fwd_gpu` GPU timer (native es=0 window value pending its reduce).
2. **Mechanism corrected twice:** it is *not* "B=4 amortizes weight-reads" — effective batch is ~1. The tree's +10.7% per-forward cost is TREE_ATTN kernel overhead, batch-independent.
3. **Effective batch is apples:** cat8 mean 1.31 vs native ~1.2; the `max_num_batched_tokens=1024` throttle applies to all arms. It also means "B=4" buys agent-level parallelism, not a decode batch of 4 — a **cache-ON cost** (the lossless-cache overshoot fix throttles concurrency).
4. **Verdict reserved** on tree-degradation pending 13398/14598 (n=1/task at temp 0.6; but the direction + the hours-wasted magnitude both favor native).
5. **Not-a-stall confirmed:** GB10-side dcgm telemetry froze at 02:31 while alienware agent containers kept running — source-of-truth liveness is `ssh alienware docker ps`.

---

## 5. Contribution honesty

The novel piece is the **stateless-tree lossless prefix cache** for branched/tree GDN spec-decode (`COMMIT_TO_RUNNING_ROW + TREE_RUNROW_INIT + BURN_NODE_BANK`; extension of open #39273). `EXACT_SEED`/`HRS`/`refold`/`SNAP_FIX-leafmaps` are **deleted scaffolding, not contribution** (now code-deprecated). The remaining APC issues (#45238 overshoot, #39809, #43995) are known-open upstream that we hit/mitigate. Within-floor lossless, not bit-exact. Committer at temp 0.6 = canonical-multidraft SAMPLED (distribution-preserving), not greedy-LCP.

---

## 6. Artifact archive (`results/fr13_b4_cache_matrix/`)

Full run artifacts committed (incl. raw `dcgm_samples.jsonl` GPU traces, per user request):
- `sl_cat8_cache_qc4/` — complete arm (deploy_speed, all 16 eval_reports, runner_metadata, per-turn metrics, dcgm, docker_full.log, offload logs, config).
- `native_ourcache_qc4/` — in-progress snapshot (13/16); refreshed each loop tick.
- `sfwd_sidecar/` — the `s_per_fwd_gpu` GPU-timer sidecars for both arms.

---

## 7. Open items
- native+cache → 16/16 + `deploy_speed_qc4.json` reduce → lock the clean matched-subset per-committed-token comparison.
- **13398 + 14598** grade → tree-degradation verdict.
- cat6+cache runs (boundary catch: cat6root vs stale-nocache) → complete the 3-arm matrix.
- Then task #14 cleanup (delete dead deprecated code, rename `codex_trace.jsonl` → `qwen_trace.jsonl`).
