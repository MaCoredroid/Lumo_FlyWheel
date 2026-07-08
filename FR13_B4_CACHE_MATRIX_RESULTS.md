# FR13 — Cache-ON Speed/Quality Matrix (B=4 + B=1) — living results doc

**Status:** B=4 `native+cache` **DONE 16/16**; `cat8+cache` **DONE 16/16**; `cat6+cache` **RUNNING**. B=1 pair queued behind B=4 cat6.
**Maintained + committed each tick** by monitor loop cron `1a91cac8`, artifacts under `results/`.
**Every number independently extracted from artifact files + adversarially verified** (workflow `wf_12862543-571`); the B=4 speed verdict below **reverses** an earlier preliminary claim that was based on a confounded es=1 preview.

> ⚠️ **Matrix spans 3 binaries.** cat8 = `8c27f454` (PRE leak-fix — the EXACT_SEED/HRS/refold code-deprecation `37bd90e2` is NOT in cat8's build; it avoided the es_ckpt OOM via explicit env `es=0` + tree-path never engaging es_ckpt). native = `ecd7aedd` (POST fix). cat6 = pending (≥ `d8af472d`).

---

## THE B=4 VERDICT (both arms 16/16, apples): the tree is a SPEED WIN and a QUALITY LOSS

| axis | winner | margin | why |
|---|---|---|---|
| **Speed** (per committed token) | **cat8 (tree)** | **+10.2%** | tree accept 3.588 vs native 3.050 (**+17.6%**) beats its tiny per-forward cost (+2.7%) |
| **Quality** (give-ups) | **native** | 5 vs 1 give-ups | tree meanders → empty patch where native converges or at least attempts |

**The tree buys decode speed at the cost of agentic reliability.** Its higher accept-per-forward (more tree candidates) makes it faster per committed token; but the same tree, though token-lossless-within-floor, shifts agent trajectories enough to cause **5× more give-ups**. Token-lossless ≠ agentic-parity. B=1 tests whether the speed win grows (HBM-bound) and whether the quality gap persists.

---

## 0. Arms
| arm | decode | tree cache | git head | status |
|---|---|---|---|---|
| **native+cache** | native MTP-5 (linear, 5-tok) | base forked APC (linear-prefix), **no** stateless-tree flags | `ecd7aedd` | DONE 16/16 |
| **cat8+cache** | branch tree, 8-node depth-5 | **stateless-tree** trio | `8c27f454` | DONE 16/16 |
| **cat6+cache** | branch tree, 6-node depth-5 | stateless-tree trio | pending | RUNNING |

native+cache is the behavioral **apples bar, NOT the same cache mechanism**. Novelty = the **tree** cache (`COMMIT_TO_RUNNING_ROW + TREE_RUNROW_INIT + BURN_NODE_BANK`); native uses plain linear-prefix forked APC.

---

## 1. Configuration (ALL facts)
**HW/model:** GB10 unified memory, Qwen3.6-27B-fp8 (GDN/mamba-hybrid), patched vLLM fork. Agents+proxy+eval on alienware (`OFFLOAD_AGENT=1`); GB10 vLLM-only.

| knob | value |
|---|---|
| B / concurrency | 4 / 4 (B=4 campaign); 1 / 1 (B=1) |
| **effective decode batch (cat8 measured)** | **~1.3** (Running==1 80.3% of 1826 ticks) — cause is **agentic request sparsity + offload round-trip latency**, NOT the `max_num_batched_tokens=1024` throttle (4 decode tokens ≪ 1024) |
| agent / nudge | qwen-code (honest give-up gate) / **OFF** (`LUMO_PROXY_AUTO_CONTINUE=0`) |
| wall / turn limit | **0 (none)** / 100000 (none); only 600s stall-watchdog |
| temp / seed | 0.6 (proxy-forced) / 0 |
| subset | subset_b4_sixteen — 16 astropy SWE-bench_Verified (fr9-matched to native decode_tps≈39.9) |
| APC serve flags (6) | `--enable-prefix-caching --enable-chunked-prefill --mamba-block-size 1024 --mamba-ssm-cache-dtype float32 --max-num-batched-tokens 1024 --block-size 1024` (#45238 overshoot fix) |
| cudagraph / gpu_util / max_len | FULL_AND_PIECEWISE (GRAPH) / 0.78 / 131072 |

**Deprecated-OFF (in native `ecd7aedd`, NOT cat8 `8c27f454`):** EXACT_SEED / HIT_RECURRENT_SUFFIX / BLOCK_REFOLD / REFOLD_TO_SNAPSHOT force-`"0"` at GDN import (patcher L1037-1038).

| | native+cache | cat8+cache | cat6+cache |
|---|---|---|---|
| KIND / decode | nativemtp5_exseed / naive_mtp | cat8 / tree | cat6root / tree |
| spec | qwen3_5_mtp 5-tok | qwen3_5_mtp 8-node tree | qwen3_5_mtp 6-node tree |
| attention | FLASH_ATTN | TREE_ATTN | TREE_ATTN |
| stateless-tree flags | none | COMMIT_TO_RUNNING_ROW+TREE_RUNROW_INIT+BURN_NODE_BANK | same |

---

## 2. Speed — clean metric `derived_tps_gpu` = committed_per_event(=accept+1) / s_per_fwd_gpu (committed-tok/GPU-s)

Definitions: `s_per_fwd_gpu` is per-DRAFT (`_basis=/drafts`); `s_per_fwd_gpu_per_forward` (=sidecar cumulative) is per-STEP; `s_per_fwd` (wall) is prefill-confounded. Use `derived_tps_gpu`.

| arm | accept/event | committed/event | **derived_tps_gpu** | s_per_fwd_gpu | prefill_frac | basis |
|---|---|---|---|---|---|---|
| **cat8+cache** (es=0 FINAL) | 3.588 | 4.588 | **34.98** | 0.1312 | 0.169 | 16 tasks |
| **native+cache** (es=0 FINAL) | 3.050 | 4.050 | **31.74** | 0.1276 | 0.151 | 16 tasks |
| native+cache (es=1 8-task preview) | 3.465 | 4.465 | 37.67 | — | 0.276 | **SUPERSEDED** |

**Speed verdict: cat8 (tree) is ~10.2% faster per committed token (34.98 vs 31.74).** Both es=0, both 16 tasks, prefill_frac close (0.169 vs 0.151), and `derived_tps_gpu` is prefill-independent by construction → apples. The tree's accept-per-forward (3.588) is **+17.6%** over native (3.050) because the 8-node branch tree gives more candidates to match; per-forward GPU cost is nearly equal (0.131 vs 0.128, +2.7%) since decode is HBM-bound (weight-read floor dominates, tree's extra compute nearly free). **Accept edge dominates → tree wins.**

**Why the earlier "native faster" was wrong:** it used the es=1 8-task preview (37.67), whose 8 tasks happened to be higher-accept (3.465); on the full matched 16 (incl. hard low-accept tasks) native accept drops to 3.050 and the tree wins. Lesson: task-mix-dependent accept makes partial-subset speed numbers misleading — only the full matched subset is apples ([[reference_deploy_speed_metric_definitions]]).

**Caveats:** per-task deploy_speed brackets are counter-wrap corrupted for some tasks (top-level aggregate fine).

---

## 3. Quality (live SWE-bench-Verified, honest give-up gate) — tree DEGRADES

| arm | graded | PASS | give-ups | genuine fail |
|---|---|---|---|---|
| **cat8+cache** | 16/16 | 6 | **5** | 5 |
| **native+cache** | 16/16 | **8** | **1** | 7 |

**Same-16 head-to-head:** the same 5 tests fail-with-patch on both (13033, 13977, 14182, 14365, 14369). Divergences:

| divergence | tasks | detail |
|---|---|---|
| **cat8 GU → native PASS** | **13579, 14508, 14539** | native solves, **far faster**: 22m/42m/19m vs cat8 give-up grind 2.8h/52m/51m |
| **cat8 GU → native WRONG-PATCH** | **14598, 13398** | native *produces* a patch (tests fail); cat8 **gave up** (empty). Both fail to solve, but native attempts where the tree bails. |
| native GU → cat8 PASS | 14096 | native 11m give-up vs cat8 33m pass |

**TREE-AGENTIC-DEGRADATION CONFIRMED.** Of cat8's **5 give-ups, native gives up on ZERO** — 3 solves + 2 wrong-patches. Native gives up on exactly **1** task total (14096). cat8 burned **9.6 agent-hours across its 5 give-ups** (13398 4.5h, 13579 2.8h). The deficit is **non-convergence (meander → empty patch)**, not wrong code — the tree's give-ups are tree-*induced* (native proves the agent can at least attempt a patch on every one of them). Extends [[project_fr13_tree_agentic_degradation]].

---

## 4. Red-team ledger
1. **Speed verdict REVERSED** (es=1 preview → es=0 final): cat8 wins +10.2%, not native. Confounded partial-subset preview was the trap.
2. **Metric:** `derived_tps_gpu` (committed/s_per_fwd_gpu); `s_per_fwd_gpu` is per-draft not per-forward.
3. **Effective-batch ~1.3 cause:** agentic sparsity + offload latency, NOT the 1024 throttle.
4. **Quality verdict:** tree degrades (5 vs 1 give-ups) — token-lossless-within-floor ≠ agentic parity.
5. **Dual finding:** tree = speed win + quality loss. B=1 (HBM-bound) tests if speed win grows + quality gap persists.

---

## 5. Contribution honesty
Novel = stateless-tree lossless prefix cache for branched/tree GDN spec-decode (ext of open #39273). EXACT_SEED/HRS/refold = deleted scaffolding. Remaining APC issues (#45238/#39809/#43995) known-open upstream. Within-floor lossless, not bit-exact. Committer @ temp 0.6 = canonical-multidraft SAMPLED.

## 6. Artifact archive (`results/`)
`results/fr13_b4_cache_matrix/` — full artifacts incl raw `dcgm_samples.jsonl` for cat8 (done) + native (done); refreshed at completion. B=1 → `results/fr13_b1_cache_matrix/`.

## 7. Open items
- **cat6+cache (B=4)** running now → 3rd cell (does the smaller 6-node tree keep the speed win with fewer give-ups?).
- **B=1 pair** launches after cat6 → decisive HBM-bound speed test.
- Then task #14 cleanup (delete dead deprecated code, rename `codex_trace.jsonl`→`qwen_trace.jsonl`).
