# FR13 APC EXACT_SEED — lossless-cache pivot CHECKPOINT (2026-06-29, autonomous build loop)

Status: **architecture VALIDATED + storage/restore plumbing built and boots clean; the prefill-capture COMPUTE never fires (`ES_PREFILL_CAPTURE=0`) — one or more capture-gate variables never become available. Not closed. Bounded plumbing remains.**

## 1. VALIDATED ARCHITECTURE (the durable result — do not re-derive)
The goal is lossless mamba/GDN prefix caching at a small `mamba_block_size` (so spec+cache keeps the TTFT win without the 8192 band-aid). Across 4 iterations + 3 workflows the design space collapsed to one answer, each step empirically proven:

1. **Per-(req_id or slot) keying is DOOMED.** No physical identity survives prefill→decode→turn: align-mode reallocates the running-state block every step, the col-0 window-anchor `(seq_len-1)//block_size` advances within a request, and per-request slots are freed/recycled across turns. Re-keying req_id→slot just moved the failure "from a different string each turn to a different integer each step" (0/4 store∩drain overlap, 91k `ES_DRAIN_NOCKPT`).
2. **The only turn-stable identity is the APC prefix block hash** (`BlockHashWithGroupId` = `block_hash + group_id`, what `find_longest_cache_hit` matches; `cache_salt` confirmed unset → hash stable across turns).
3. **Store the chunked-realization checkpoint ON the prefix-hash-keyed cache block** (SGLang MambaRadixCache analogue), NOT a per-request Python dict.
4. **Compute via PREFILL-CAPTURE, not the incremental decode drain.** The drain reaches only ~pos 896 (a turn decodes ~128 tokens) and resets per-turn → never crosses a `block_size` (1024) boundary → never produces a storable checkpoint. The chunked kernel runs in *prefill*; capture per `block_size`-segment there (seeded by the restored chunked base; cache-hit re-prefill extends the chain cross-turn). Bit-exact by Sparse-Prefix-Caching Remark-1 (1024 is a multiple of `FLA_CHUNK_SIZE`=64); works with FlashInfer (needs only `output_final_state`).

Separately settled this session: the cuda-graph carrier is real but separate (`cudagraph_mode=PIECEWISE` banked, commit 2dfbb334); cause-C (spec node-bank) is fixed (`SNAP_FIX`, baked); char-8 is a tool-call-format flaky-decode artifact, largely cache-independent (memory `project_fr13_char8_attribution_open`).

## 2. WHAT IS BUILT + BOOTS (in the working-dir patcher `scripts/fr10_phase4_patch_vllm_tree_gdn.py`, all gated `FR13_APC_EXACT_SEED=1`, default-OFF byte-identical / AST-proven)
- **3 new patch targets apply cleanly to the real in-container vLLM 0.19.2**: `_patch_block_pool_exact_seed`, `_patch_kv_cache_manager_exact_seed`, `_patch_worker_mamba_exact_seed`. Boot survives all of them (serving OK).
- **STORAGE**: `BlockPool._fr13_es_ckpt` side-dict keyed by `BlockHashWithGroupId`; write at `cache_full_blocks` insert anchor; eviction hooks in `_maybe_evict_cached_block` (before `reset_hash`) + `reset_prefix_cache`. In-process heap (UniProcExecutor TP=1) makes module-globals visible across scheduler↔worker↔model hops (no IPC).
- **RESTORE plumbing FIRES** (`ES_RESTORE=39` on a full seq49 replay): manager (`kv_cache_manager:203`, has `request`) stashes hit `BlockHashWithGroupId` (reconstructed via `BlockHashListWithBlockSize` for the block_size-granular merged hash) → worker `preprocess_mamba` reads `block_pool._fr13_es_ckpt[hit_hash]` → GDN seed at `gdn_linear_attn:984` via `_LUMO_FA_NONSPEC_ROW_REQ_IDS`.
- **PREFILL-CAPTURE block** implemented (per-`block_size`-segment `chunk_gated_delta_rule`, seeded by restored base / prior segment) at the prefill site, with abs-base bridged via a per-forward `{row→abs_base}` stash from the restore loop. Drain's cache-block write disabled (`if False`).

## 3. THE REMAINING BLOCKER (precise, reproducible)
On a full eager state-diff @1024 with EXACT_SEED=1: **`ES_PREFILL_CAPTURE=0`** (never fires) → `ES_PENDING=0` → `ES_WRITE=0` → `block_pool._fr13_es_ckpt` empty → 39 `ES_RESTORE` all `seeded=False` → write∩restore hash overlap=0 → **drift stays 77.96 (no fix)**.
The capture gate `if (_fr13_es_bs is not None and int(_fr13_es_bs)>0 and _fr13_es_bs%64==0 and _fr13_es_pend_by_req is not None):` never passes. Candidates (NOT yet disambiguated — this is what burned the iterations, one-per-boot):
- **(a) cross-module global**: manager sets `gdn_linear_attn._FR13_ES_BLOCK_SIZE`; capture reads `globals().get("_FR13_ES_BLOCK_SIZE")` — confirm the capture literal's `globals()` IS the gdn module dict (it should be, but unverified at runtime).
- **(b) `_FR13_ES_PENDING_BY_REQ` is None** at the capture (init only happens in the manager hit path; iter-4 added eager init there but it's still gated on `num_new_computed_tokens>0`).
- **(c) turn-0 skip**: block_size publishes on the first *hit* (`num_new_computed_tokens>0`); turn-0 (the fresh prefill that ESTABLISHES the cached prefix, spanning ~11 boundaries) is a miss → block_size None → its boundaries never captured → later restores find nothing even if (a)/(b) are fixed.
- **(d) capture block reachability**: confirm the per-segment block is on the executed prefill path (not behind an untaken branch).

## 4. RECOMMENDED FOCUSED FINISH (bounded — ~1-2 iterations, NOT open-ended)
1. **Add ONE all-gates diagnostic** at the capture entry: log `block_size`, `pend_by_req is None`, `segbase`, per-row `hit/b0`, and "block reached" — so a SINGLE boot reveals every failing gate at once (the 4 iterations each surfaced one gate per 25-min boot; instrument to collapse that).
2. **Publish block_size truly pre-turn-0 / unconditional** — at engine init or on every `get_computed_blocks` (drop the `num_new_computed_tokens>0` gate for the block_size publish), AND init `_FR13_ES_PENDING_BY_REQ` at module load — so turn-0's fresh prefill captures.
3. Re-boot the eager state-diff @1024; success criteria: `ES_PREFILL_CAPTURE>0 → ES_WRITE>0 → write∩restore hash overlap>0 → ES_RESTORE seeded=True → drift off 77.96 (toward fp)`.
4. Then the L1 1-task proxy (12907, temp 0.6, PIECEWISE) and the L0→L3 ladder (task #7).

## 5. STATE / RECOVERY
- The EXACT_SEED pivot code lives UNCOMMITTED in the working-dir patcher (staged from worktree `wf_4f4d8bf1` + this session's 3 new patch targets + prefill-capture). Committed main (`fr13-apc-ssm-shadow` HEAD `2dfbb334`) is clean: PIECEWISE baked, experimental flags removed, findings docs.
- Test harness: `scripts/fr13_apc_exactseed_statediff.sh` (eager state-diff @1024, reuses the 193-capture continuous REF). ENG_LOG markers: `ES_PREFILL_CAPTURE`/`ES_WRITE`/`ES_RESTORE`/`ES_PENDING`/`ES_DRAIN_NOCKPT`.
- Real vLLM 0.19.2 sources extracted at `/home/mark/.claude/jobs/22c39bb9/tmp/vllm_real_0192/` (gdn_linear_attn, gdn_attn, block_pool, kv_cache_utils, kv_cache_manager, single_type_kv_cache_manager, worker/mamba_utils, gpu_input_batch).
