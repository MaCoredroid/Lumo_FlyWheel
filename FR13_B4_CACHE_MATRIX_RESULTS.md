# FR13 — Cache-ON Speed/Quality Matrix (B=4 + B=1) — living results doc

**Status:** IN PROGRESS. B=4: `native+cache` running (14/16 graded); `cat8+cache` DONE (16/16); `cat6+cache` QUEUED. B=1 pair (`cat6+cache`, `native+cache`) queued behind the B=4 cat6 arm.
**Maintained + committed each tick** by monitor loop cron `1a91cac8`, with the artifact archive under `results/`.
**Every number below is independently extracted from the artifact files and cross-verified** (workflow `wf_12862543-571`, adversarial completeness critic). Corrections from that pass are folded in.

> ⚠️ **The matrix spans three different binaries.** Per-arm `git_head.txt`: **cat8 = `8c27f454`** (2026-07-07, PRE leak-fix), **native = `ecd7aedd`** (POST leak-fix `37bd90e2`), **cat6 = pending (≥ `d8af472d`)**. The leak-fix code-deprecation of EXACT_SEED/HRS/refold is **NOT in cat8's binary** — cat8 avoided the `_fr13_es_ckpt` host-OOM leak only via explicit env `FR13_APC_EXACT_SEED=0` **and** being a tree arm that never engages the es_ckpt path. Reproduce each arm from its own head, not the doc HEAD.

---

## 0. What this matrix answers

Isolate the **tree decode superset** vs **native MTP-5**, both with **our forked lossless prefix cache**, on a live agentic SWE-bench-Verified gate. Then repeat at **true B=1** (the tree's favorable HBM-bound regime).

| arm | decode | tree cache | git head | status |
|---|---|---|---|---|
| **native+cache** | native MTP-5 (linear, 5-tok) | base forked APC (linear-prefix), **no** stateless-tree flags | `ecd7aedd` | RUNNING 14/16 |
| **cat8+cache** | branch tree, 8-node depth-5 | **stateless-tree** trio | `8c27f454` | DONE 16/16 |
| **cat6+cache** | branch tree, 6-node depth-5 | stateless-tree trio | pending | QUEUED |

**native+cache is the behavioral apples bar, NOT the same cache mechanism.** The lossless novelty is the **tree** cache (`FR13_APC_COMMIT_TO_RUNNING_ROW + FR13_TREE_RUNROW_INIT + FR13_APC_BURN_NODE_BANK`); native uses the plain linear-prefix forked APC.

---

## 1. Configuration (ALL facts)

**Hardware/model:** GB10 (unified memory), Qwen3.6-27B-fp8 (GDN/mamba-hybrid), patched vLLM fork. Agents+proxy+eval offloaded to alienware (`OFFLOAD_AGENT=1`); GB10 is vLLM-only.

### Common (every arm)
| knob | value |
|---|---|
| B (`--max-num-seqs`) | 4 (B=4 campaign) / 1 (B=1 campaign) |
| SWE concurrency | 4 / 1 |
| **effective decode batch (measured, cat8)** | **~1.3** (Running==1 for 80.3% of 1826 ticks; dist R0=28,R1=1466,R2=150,R3=110,R4=72). native's not yet measured (docker_full.log lands at teardown). |
| agent | **qwen-code** (honest give-up gate; codex nudge confounded all prior data) |
| nudge | **OFF** (`LUMO_PROXY_AUTO_CONTINUE=0`) |
| wall / turn limit | **0 (none)** / **100000 (effectively none)**; only backstop = 600s stream-idle stall watchdog |
| temperature / seed | **0.6** (proxy-forced) / **0** (`--seed 0`) |
| subset | **subset_b4_sixteen** — 16 astropy SWE-bench_Verified, chosen to match the fr9 native-MTP5 `decode_tps≈39.9` baseline |
| APC serve flags (all 6) | `--enable-prefix-caching --enable-chunked-prefill --mamba-block-size 1024 --mamba-ssm-cache-dtype float32 --max-num-batched-tokens 1024 --block-size 1024` (#45238 overshoot fix) |
| cudagraph | **FULL_AND_PIECEWISE** (GRAPH mode, `enforce_eager=False`) |
| gpu_mem_util / max_model_len | 0.78 / 131072 |
| instruments | `FR13_SFWD_GPU_TIMER=1`, `FR13_DEVICE_MULTIDRAFT=1` |

**Deprecated-OFF (code-level, in native's `ecd7aedd` build; NOT in cat8's `8c27f454`):** `FR13_APC_EXACT_SEED`, `FR13_APC_HIT_RECURRENT_SUFFIX`, `FR13_APC_BLOCK_REFOLD`, `FR13_APC_REFOLD_TO_SNAPSHOT` — force-`"0"` at GDN import (patcher L1037-1038). Reason: block-hash-keyed `_fr13_es_ckpt` reaper never fires → host OOM ~3.5h (killed the earlier es=1 native+cache at 8/16).

### Per-arm decode/tree config
| | native+cache | cat8+cache | cat6+cache |
|---|---|---|---|
| KIND | `nativemtp5_exseed` | `cat8` | `cat6root` |
| launcher | forked (patcher only) | forked | forked |
| decode | naive_mtp (linear) | tree | tree |
| spec | qwen3_5_mtp, **5-tok** | qwen3_5_mtp, **8-node tree** `[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]` | qwen3_5_mtp, **6-node tree** `[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]` |
| attention | **FLASH_ATTN** | **TREE_ATTN** | **TREE_ATTN** |
| stateless-tree flags | **none** | `COMMIT_TO_RUNNING_ROW=1 TREE_RUNROW_INIT=1 BURN_NODE_BANK=1` | same as cat8 |

native's tree machinery is explicitly un-leaked via XFLAGS (`FR13_FA2_TREE_BIAS=0 FR13_TREE_SAMPLE_ROW=0 FR13_CONV_COMMITTED_PATH=0`…), asserted live in `container_env.txt`.

---

## 2. Speed — use the file's own clean metric `derived_tps_gpu` (committed-tok / GPU-sec)

**Metric definitions (pin these — the deploy_speed file has multiple GPU numbers):**
- `committed_per_event = accept_per_event + 1` (the +1 bonus token). cat8: 3.588 → **4.588**.
- `s_per_fwd_gpu` = GPU-sec **per matched pure-decode DRAFT** (`_basis` field: `/drafts_total`). cat8: **0.131** — this is per-draft, **NOT** per-forward.
- `s_per_fwd_gpu_per_forward` = GPU-sec **per pure-decode STEP** = the sidecar cumulative. cat8: **0.164** (= 8936.56 s / 54575 steps).
- **`derived_tps_gpu = committed_per_event / s_per_fwd_gpu`** = the clean, prefill-independent decode throughput. **This is the cross-arm comparison number.**
- `s_per_fwd` = 0.288 = wall-span, **prefill-confounded at B>1** — do not use. `prefill_frac` must match before any aggregate/derived-TPS cross-arm compare.

| arm | accept/event | committed/event | **derived_tps_gpu (committed-tok/GPU-s)** | prefill_frac | basis |
|---|---|---|---|---|---|
| **cat8+cache** (clean es=0, FINAL) | 3.588 | 4.588 | **34.98** | 0.169 | 16 tasks |
| **native+cache** (clean es=0) | pending | pending | **PENDING (deploy_speed absent, arm running)** | — | — |
| native+cache (es=1 OOM-partial, **PREVIEW ONLY**) | 3.465 | 4.465 | 37.67 | 0.276 | 8 tasks |

**Speed verdict: PENDING.** cat8's clean number is **34.98**. Native's clean es=0 number does **not exist yet** (its `deploy_speed_qc4.json` reduce runs at arm completion; its sidecar is live and still climbing, per-step 0.147→0.148). The only native number today is the **es=1 8-task preview (37.67)**, which is **triply confounded** vs cat8: (a) deprecated es=1 config, (b) 8 vs 16 tasks, (c) prefill_frac 0.276 vs 0.169. It hints native is slightly higher-throughput but is **not apples** and must not be reported as "the native result."

**Caveats:** accept is task-mix dependent (live swings 2.58↔5.38; same native config gave 3.045 in a different campaign). Per-task deploy_speed brackets are **counter-wrap corrupted** for cat8 {14369,14508,14539,14598,14995} and native-es1 {13977} (negative drafts / null accept / duplicated tps) — the **top-level aggregate is computed separately and is fine**, but the per-task speed instrument is unreliable for those.

---

## 3. Quality (live SWE-bench-Verified, honest give-up gate) — the headline finding

### Grades (one convention: give-ups = `empty_patch`, reported separately from genuine `tests_failed`)
| arm | graded | PASS | give-ups | genuine fail |
|---|---|---|---|---|
| **cat8+cache** | 16/16 | 6 | **5** | 5 |
| **native+cache** (provisional) | 14/16 | 8 | **1** | 5 |

native pending: **13398, 14598** (the two remaining discriminating give-up tasks). WALL=0 so no wall-censoring; counts can still change.

### Head-to-head — EQUAL DENOMINATOR, the 13 tasks both had graded (not 6/16 vs 7/13)
cat8 = **5 pass / 3 giveup / 5 fail** · native = **7 pass / 1 giveup / 5 fail** · **the same 5 tests fail on both** (13033, 13977, 14182, 14365, 14369 — genuinely hard).

| divergence | tasks | detail |
|---|---|---|
| **cat8 GU → native PASS** | **13579, 14508, 14539** | native solves all 3, and **far faster**: 22m / 42m / 19m vs cat8's give-up grind **2.8h / 52m / 51m** |
| native GU → cat8 PASS | 14096 | native 11m give-up vs cat8 33m pass |

**The tree (cat8) gives up MORE** — native resolves 3 tree-giveups while cat8 recovers only 1. And cat8 burned **9.6 agent-hours across its 5 give-ups** (13398 alone ran **4.5h**) vs native's minutes. The deficit is **non-convergence (meander → empty patch)**, not wrong code: excluding give-ups, cat8 5-pass/8-patched vs native 7-pass/12-patched is far closer. **This tree-agentic-degradation delta is the headline** — but it's **provisional** (native 14/16; verdict finalizes when 13398/14598 grade: native-passes-both ⇒ confirmed; native-also-gives-up ⇒ narrows to task-difficulty).

---

## 4. Red-team ledger (corrections included)

1. **Metric fixed (twice).** per-request-TPS is confounded → use `derived_tps_gpu`. And `s_per_fwd_gpu` is **per-draft**, not per-forward — my earlier "36.6 ms/committed-token" mixed denominators; the file's clean number is `derived_tps_gpu` (cat8 34.98). Native's is pending.
2. **Effective-batch cause CORRECTED.** Batch is ~1.3, not 4 — but the cause is **NOT** the `max_num_batched_tokens=1024` throttle (4 decode tokens ≪ 1024). It's **agentic request sparsity**: 4 offloaded agents rarely have simultaneous in-flight LLM requests (sequential turns + tool execution + alienware round-trip). The 1024 prefill-admission budget contributes only when a new turn's long prompt must prefill. Net: "B=4" buys agent-level parallelism, not a decode batch of 4 — measured speed is really ~B=1.3.
3. **Tree-degradation is provisional** (native 14/16; n=1/task at temp 0.6), but the direction + the hours-wasted magnitude both favor native. Verdict pends 13398/14598.
4. **Not-a-stall confirmed:** GB10 dcgm telemetry froze at 02:31 while alienware agent containers kept running — source-of-truth = `ssh alienware docker ps`.
5. **B=1 is the decisive speed test:** at true single-stream (HBM-bound) the tree *should* win per committed token. If it still loses there, that's decisive against the tree.

---

## 5. Contribution honesty
Novel = the **stateless-tree lossless prefix cache** for branched/tree GDN spec-decode (`COMMIT_TO_RUNNING_ROW + TREE_RUNROW_INIT + BURN_NODE_BANK`; extension of open #39273). `EXACT_SEED`/`HRS`/`refold`/`SNAP_FIX-leafmaps` = deleted scaffolding, not contribution. Remaining APC issues (#45238/#39809/#43995) are known-open upstream we hit/mitigate. Within-floor lossless, not bit-exact. Committer @ temp 0.6 = canonical-multidraft SAMPLED (distribution-preserving), not greedy-LCP.

---

## 6. Artifact archive (`results/`)
`results/fr13_b4_cache_matrix/` — full run artifacts **incl. raw `dcgm_samples.jsonl`** for `sl_cat8_cache_qc4` (done) + `native_ourcache_qc4` (in-progress snapshot, refreshed each tick) + `sfwd_sidecar/`. B=1 arms will archive under `results/fr13_b1_cache_matrix/`.

---

## 7. Open items
- native+cache → 16/16 + `deploy_speed_qc4.json` reduce → lock the clean matched-subset `derived_tps_gpu` comparison (the pending speed verdict).
- **13398 + 14598** grade → finalize tree-degradation verdict.
- cat6+cache (B=4) runs (boundary catch: cat6root vs stale-nocache).
- **B=1 pair** (`sl_cat6_cache_qc1`, `native_ourcache_qc1`) launches after B=4 cat6 → the decisive HBM-bound speed test.
- Then task #14 cleanup (delete dead deprecated code, rename `codex_trace.jsonl`→`qwen_trace.jsonl`).
