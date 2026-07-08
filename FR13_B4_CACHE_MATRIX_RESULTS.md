# FR13 — Cache-ON Speed/Quality Matrix (B=4 + B=1) — living results doc

**Status:** B=4 **3-arm matrix COMPLETE** — `native+cache`, `cat8+cache`, `cat6+cache` all DONE 16/16. Now running: **`native+nocache`** control (KIND flash_ns5_nocache, isolates the cache). B=1 CANCELLED. Serialization-fix shot (§7) queued after native+nocache.
**Every number independently extracted from artifact files + adversarially verified** (workflow `wf_12862543-571`).

> ⚠️ **Matrix spans 3+ binaries.** cat8 = `8c27f454` (PRE leak-fix — the EXACT_SEED/HRS/refold deprecation `37bd90e2` is NOT in cat8's build; it avoided the es_ckpt OOM via explicit env `es=0`). native = `ecd7aedd` (POST fix). cat6 = `d8af472d`+.

---

## THE B=4 VERDICT (3 arms × 16/16): the tree buys DECODE SPEED, pays AGENTIC GIVE-UPS; the two tree sizes are quality-equal (which is faster = UNRESOLVED, cat8⊃cat6)

| arm | accept | s_per_fwd_gpu | **derived_tps_gpu** | resolved | give-ups |
|---|---|---|---|---|---|
| native+cache (MTP-5) | 3.050 | 0.1276 | 31.74 | **8/16 (50%)** | **1** |
| cat8+cache (8-node) | 3.588 | 0.1312 | 34.98 | 6/16 (38%) | 5 |
| **cat6+cache (6-node)** | 3.850 | **0.1209** | **40.12** | 6/16 (38%) | 5 |

**Speed — PROVISIONAL. cat6-vs-cat8 is UNRESOLVED; cat6-vs-native is directionally real.** As measured the trees beat native (cat6 +26%, cat8 +10%). BUT:
- **cat8 is a strict SUPERSET of cat6** (verified from code: `cat8 = cat6 ∪ {(0,1),(0,0,1)}`, same depth-5 spine + `(1,)` root sibling). A superset tree **must accept ≥ its subset on identical draft/target tokens** (cat6's best path is available to cat8, plus extras). So cat6's measured accept **3.850 > cat8's 3.588 is structurally impossible on matched inputs → pure trajectory noise** (temp 0.6, different tokens per arm).
- Therefore **"cat6 faster than cat8" is NOT established** (I earlier over-claimed it). On matched trajectories cat8 accepts ≥ cat6; the real question is whether cat6's fewer-nodes **per-forward saving** (6 vs 8 positions) outweighs cat8's genuine (small, diminishing-returns) **extra accept** from the 2 added siblings. Net sign unknown → needs a **same-trajectory** measurement.
- **concurrency-confounded** too — all arms ran at effective batch ~1.3 (§7 serialization bug), not true B=4.
- What DOES hold: **both trees out-accept native's linear chain**, so the tree>native speed edge is directionally real (cat8's +10% is on matched 16-task apples; cat6>native likewise but accept-inflated). Re-measure cat6-vs-cat8 same-trajectory after the §7 fix before ranking the trees.

**Quality:** native best (8/16, 1 give-up). **Both trees identical: 6/16 resolved, 5 give-ups** — the tree-degradation is **SIZE-ROBUST** (shrinking 8→6 nodes did not reduce give-ups; cat6's give-up *set* differs — 2 cat6-specific — but the count matches). Resolve is on-par between trees, ~2 below native.

**Net:** the tree trades ~2 resolves (give-up cost, size-robust) for a real decode-speed edge over native. **cat6 and cat8 are quality-equal** (6/16, 5 give-ups each). **Which tree is faster is UNRESOLVED** — cat8 ⊃ cat6, so cat8 accepts ≥ cat6 on matched tokens; cat6's cheaper per-forward vs cat8's (small) extra accept is a genuine open question needing a same-trajectory measure. So the shippable-tree choice (cat6 vs cat8) is TBD; both carry the same give-up cost, which must be attacked at the trajectory level (not tree size). `native+nocache` (running) isolates whether the cache itself moves give-ups/speed; the §7 serialization fix could lift the matrix's decode concurrency and enable a clean tree-vs-tree re-measure.

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
| **effective decode batch (measured, GENERAL)** | **~1.3** (cat8 Running==1 80% of 1826 samples; native even more serial at 85%) — vLLM **serializes** the 4 queued requests. NOT sparsity (`Waiting>0` 94% of samples — 4 requests ARE queued), NOT the 1024 throttle, NOT preemption (≤2). Waiting reqs mostly un-prefilled (`prompt_tput=0` ~78%); Running spikes to 2–4 only in prefill bursts, then drains to 1. Exact V1-scheduler reason TBD (needs source read). |
| agent / nudge | qwen-code (honest give-up gate) / **OFF** (`LUMO_PROXY_AUTO_CONTINUE=0`) |
| wall / turn limit | **0 (none)** / 100000 (none); only 600s stall-watchdog |
| temp / seed | 0.6 (proxy-forced) / 0 |
| subset | subset_b4_sixteen — 16 astropy SWE-bench_Verified (fr9-matched to native decode_tps≈39.9) |
| APC serve flags (6) | `--enable-prefix-caching --enable-chunked-prefill --mamba-block-size 1024 --mamba-ssm-cache-dtype float32 --max-num-batched-tokens 1024 --block-size 1024` (#45238 overshoot fix) |
| cudagraph / gpu_util / max_len | FULL_AND_PIECEWISE (GRAPH) / 0.78 / 131072 |

**Two flag categories (do not conflate):**
- **Deprecated / force-OFF (DEAD)** — `EXACT_SEED`, `HIT_RECURRENT_SUFFIX`, `BLOCK_REFOLD`, `REFOLD_TO_SNAPSHOT` force-`"0"` at GDN import (patcher L1037-1038; present in native `ecd7aedd`, NOT in cat8 `8c27f454`).
- **Baked-ON working cache-correctness fixes** — `FR13_APC_SNAP_FIX=1` (the SSM node-bank snapshot fix: on commit, publish the committed accepted-LEAF SSM state to the node-bank so cache restore is faithful — verified FAITHFUL 240/240, baked 2026-06-24, the working fix), `SNAP_FIX_ZEROACCEPT=1` (same for accepted_len==0), `CONV_FIX=1`. **All forked tree arms (cat8, cat6) + native boot with these =1** (verified identical in cat8's `docker_full.log`). SNAP_FIX is in #14 cleanup only to *review* whether it's now redundant with the stateless-tree flags — it is correct and required as configured.

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

**These are EXHAUSTIVE give-ups, NOT quick bails.** Every cat8 give-up **tried 39 min – 4.5 h** before emitting an empty patch: 13398 **4.5h**, 13579 **2.8h**, 14508 **52m**, 14539 **51m**, 14598 **39m** — minimum effort **39 min**. This is the **honest give-up gate**: nudge is OFF (`LUMO_PROXY_AUTO_CONTINUE=0`), so an empty patch stands as a give-up. **Before the fix** (codex + nudge), empty patches were auto-continued 3× ("you MUST apply_patch"), which *masked* give-ups — so a give-up here means the model genuinely ground for **30 min+** (often hours) and could not converge, not a lazy quick bail. (native's single give-up, 14096, was quicker at **11 min**.)

**TREE-AGENTIC-DEGRADATION CONFIRMED.** Of cat8's **5 give-ups, native gives up on ZERO** — 3 solves + 2 wrong-patches. Native gives up on exactly **1** task total (14096). cat8 burned **9.6 agent-hours across its 5 give-ups**. The deficit is **non-convergence (meander → empty patch)**, not wrong code — the tree's give-ups are tree-*induced* (native proves the agent can at least attempt a patch on every one of them). Extends [[project_fr13_tree_agentic_degradation]].

> **Scaffold note:** the 77.2% official Qwen3.6-27B SWE-Verified is temp 1.0 on Qwen's internal scaffold, which likely includes nudge/retry-style continuation. We run **qwen-code with nudge OFF** at temp 0.6 on a hard 16-subset — deliberately un-masked, so these numbers are a stricter honest floor, not comparable to the headline.

---

## 4. Red-team ledger
1. **Speed verdict REVERSED** (es=1 preview → es=0 final): cat8 wins +10.2%, not native. Confounded partial-subset preview was the trap.
2. **Metric:** `derived_tps_gpu` (committed/s_per_fwd_gpu); `s_per_fwd_gpu` is per-draft not per-forward.
3. **Effective-batch ~1.3 — investigated, both prior causes wrong:** NOT the 1024 throttle, NOT agentic sparsity (`Waiting>0` 94% — requests ARE queued), NOT preemption. vLLM serializes the 4 queued requests (bursty prefill admission → drain to 1); general to native (85% R1) + cat8 (80% R1). Exact V1-scheduler policy TBD.
4. **Quality verdict:** tree degrades (5 vs 1 give-ups) — token-lossless-within-floor ≠ agentic parity.
5. **Dual finding:** tree = speed win + quality loss. B=1 (HBM-bound) tests if speed win grows + quality gap persists.

---

## 5. Contribution honesty
Novel = stateless-tree lossless prefix cache for branched/tree GDN spec-decode (ext of open #39273). EXACT_SEED/HRS/refold = deleted scaffolding. Remaining APC issues (#45238/#39809/#43995) known-open upstream. Within-floor lossless, not bit-exact. Committer @ temp 0.6 = canonical-multidraft SAMPLED.

## 6. Artifact archive (`results/`)
`results/fr13_b4_cache_matrix/` — full artifacts incl raw `dcgm_samples.jsonl` for cat8 (done) + native (done); refreshed at completion. B=1 → `results/fr13_b1_cache_matrix/`.

## 7. The "B=4 ≈ B=1" serialization: root cause FOUND (source-verified), fix PROPOSED (UNTESTED)

Investigated via workflow `wf_1c4af669-5c7` (web + live-container source read + adversarial synthesis). Both my earlier explanations were wrong; this is the source-verified answer.

**Root cause** (live container `vllm/v1/core/sched/scheduler.py`): `max_num_batched_tokens=1024` is set **exactly equal** to `mamba_block_size=1024` (the APC #45238 overshoot weld). Each step, the scheduler deducts the running decodes from the single 1024-token budget first (~28 tok for 4 spec-decodes; each decode = 1+num_lookahead), leaving **<1024** for the waiting loop. With `mamba_cache_mode='align'` (auto for hybrid+prefix-caching), `_mamba_block_aligned_split` rounds any non-final prefill chunk **down** to a 1024 multiple → residual <1024 → chunk=0 → **`break`** → **no waiting request is prefilled while any decode runs**. Prefill and decode become mutually exclusive per step; long agentic prompts (~24 blocks) serialize one-at-a-time → effective decode batch pinned ~1. **General** (native MTP-5 + tree), **not structural** — the Running=2→4 prefill *bursts* prove co-residence is architecturally allowed, just budget-starved. Corroborated by vLLM issue #36697 (Qwen3.5 Mamba). Likely the same root as the historical "carrier-B = concurrency" agentic degradation.

**Proposed fix (two flags, BOTH required):**
1. ADD `--long-prefill-token-threshold 1024` (= mamba block; currently unset=0) — caps *every request's* per-step prefill chunk to ≤1 mamba block on both scheduler paths.
2. RAISE `--max-num-batched-tokens 1024 → 4096` (keep `mamba-block-size 1024` + `block-size 1024` unchanged).

Effect: 4 decodes (~28 tok) + one-to-three full 1024-token prefill blocks (distinct waiting requests) co-schedule in one forward pass → requests accumulate into the decode set instead of serializing.

**APC losslessness — NO conflict** (verified): #45238 requires only (i) `mamba_block_size=1024` (untouched) and (ii) ≤1 block-boundary per request per step. The `max_num_batched=1024` value was only a *proxy* for (ii); the threshold re-enforces (ii) **directly + per-request**, so per-request chunk boundaries stay at the *same* 1024 positions → align fp-profile unchanged. ⚠️ Raising `max_num_batched` **alone** (the naive fix) WOULD re-poison (#45238) — the threshold is what makes it safe.

**Status: UNTESTED.** Batch-*composition* change (co-scheduling a prefill chunk with the spec-decodes) can perturb non-batch-invariant GEMM tiling ~1 ULP → must pass the temp-0.6 recurrent-oracle lossless gate before use. **Synthetic 4-concurrent shot** (Running histogram + lossless gate) queued to run **after cat6 frees the GPU** (`scripts/fr13_serialization_shot.sh`). Fallbacks if lossless regresses: `LUMO_BATCH_INVARIANT_VLLM=1`, conservative 2048, or scope to cache-OFF only — none touch `mamba_block_size`.

## 8. Open items
- **cat6+cache (B=4)** running → 3rd matrix cell (smaller tree: keep speed win with fewer give-ups?).
- **Serialization-fix shot** after cat6 (§7).
- B=1 pair **CANCELLED** (user 2026-07-08 — "B=4" is already ~B=1; fix the config instead).
- Task #14 cleanup (delete dead deprecated code, rename `codex_trace.jsonl`→`qwen_trace.jsonl`).
