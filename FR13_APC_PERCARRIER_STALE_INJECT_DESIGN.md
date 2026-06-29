# FR13 APC per-carrier LIVE-SWE STALE-INJECT localization instrument

Goal: localize WHICH cached carrier (conv K-1 window / SSM recurrent state /
full-attn KV / position) makes the stale APC cache derail a live SWE run.

## Architectural constraint (verified, respected)

`num_computed_tokens` couples ALL carriers. On a prefix-cache HIT the scheduler
hands the request the cached physical blocks and sets
`num_new_local_computed_tokens > 0`; the worker then SKIPS recomputing those
positions and READS the cached carriers (SSM/conv via the gathered
`non_spec_state_indices_tensor`, KV via the block-table-referenced blocks, RoPE
via `num_computed_tokens`). You therefore CANNOT make one carrier fresh while
keeping the others cached — the recurrent SSM state and per-position KV need the
WHOLE prefix recomputed to be fresh, and that recompute is all-or-nothing
(driven by `num_new_local_computed_tokens`).

Verified call chain:
- Scheduler hit: `vllm/v1/core/sched/scheduler.py` `get_computed_blocks(request)`
  (patcher anchor `fr10_phase4_patch_vllm_tree_gdn.py:6395`).
- `vllm/v1/core/kv_cache_manager.py:176 get_computed_blocks` -> records the hit
  (`prefix_cache_stats.record`, L208-214) BEFORE the shadow zeroing, then returns
  `(create_kv_cache_blocks(computed_blocks), num_new_computed_tokens)`.
- Global shadow (patcher `_patch_scheduler_apc_shadow`, L6426-6428) sets
  `new_computed_blocks = new_computed_blocks.new_empty()` +
  `num_new_local_computed_tokens = 0` => the scheduler allocates FRESH blocks for
  the whole prefix => the request re-prefills (takes the `num_prefills > 0`
  branch in `gdn_linear_attn._forward_core` and the full-attn prefill), and the
  cached blocks are simply not attached to THIS request (still ref-counted in the
  pool, content-addressed).

So isolation is the INVERSE of "one fresh, rest cached":
1. global shadow ON => ALL carriers fresh (re-prefill the whole prefix).
2. CAPTURE the stale value for carrier X at the moment of the HIT, keyed by
   request, BEFORE the shadow `new_empty()` discards the cached block handle.
3. INJECT: after the fresh forward produces carrier X's state, OVERWRITE the
   cached-prefix region of X with the captured stale value.

A run with "only X stale, others fresh" that DERAILS like full-cache-ON => X is
the carrier. (If multiple Xs individually derail, the carrier set is the union.)

The master flag is `FR13_APC_INJECT_STALE ∈ {none, conv, ssm, kv, pos}` (default
`none`). It is GATED UNDER global shadow (`FR13_APC_SHADOW=1` or
`FR13_APC_SHADOW_RUNTIME=1`): inject only makes sense when the base is the
all-fresh re-prefill. With `FR13_APC_INJECT_STALE=none` every new code path is
dead => byte-identical to the current shadow-only patcher.

---

## Where the carriers live (verified file:line in /tmp/vllm_cu130_src)

| Carrier | Cached value read (restore) | Fresh value produced (inject site) |
|---|---|---|
| CONV | `gdn_linear_attn.py:890` `causal_conv1d_fn(..., conv_states=conv_state, cache_indices=non_spec_state_indices_tensor)` reads the restored K-1 window in-place | same call mutates `conv_state[indices]` in place (prefill) |
| SSM  | `gdn_linear_attn.py:984` `initial_state = ssm_state[non_spec_state_indices_tensor]` | `gdn_linear_attn.py:1004` `ssm_state[non_spec_state_indices_tensor] = last_recurrent_state` |
| KV   | cached blocks referenced by `block_table`; forward reads them | `flash_attn.py:869 reshape_and_cache_flash(key,value,key_cache,value_cache,slot_mapping,...)` writes fresh KV into the (fresh, under-shadow) blocks |
| POS  | `num_computed_tokens` -> RoPE base | patcher `_fr10_mrope_base` `fr10_phase4_patch_vllm_tree_gdn.py:10342` (mrope base = `num_computed_tokens_cpu[i]`) |

The patcher already injects the gdn prefill conv/ssm region as STRING blocks:
- prefill conv: `prefill_conv_replacement` (patcher L5475-5515) — already CAPTURES
  the restored conv rows under `FR13_APC_CONV_RESTORE_CAPTURE`.
- prefill ssm writeback: `prefill_scan_replacement` ends with
  `ssm_state[non_spec_state_indices_tensor] = last_recurrent_state` (patcher
  L5868-5870).

---

## CAPTURE: the stale value, at the hit, before shadow zeroing

The shadow zeroing happens in the SCHEDULER (pid 1). The carrier tensors
(conv_state/ssm_state/kv_cache) live in the WORKER. The scheduler has no handle
to them. So the capture cannot be a tensor read in the scheduler.

Resolution — capture the stale value where the carrier tensor is live, on the
FIRST forward of the request when the cached blocks are STILL referenced:
this is exactly the NON-shadow forward. But under shadow the cached blocks are
never attached, so there is no in-worker moment where the stale value is read.

Two viable capture strategies:

### Strategy A (chosen, clean) — "stale = what the deployed (cache-ON, no-shadow) run restored"
Run the instrument in TWO phases per turn boundary via the existing RUNTIME flag
file (`FR13_APC_SHADOW_RUNTIME`):
- Phase CAPTURE (flag file = "0", shadow OFF): the deployed cache-ON path runs;
  the request restores the cached carrier. We TAP the restored carrier rows at
  the read sites above and stash them in a worker module global keyed by req_id
  + layer (`gdn_linear_attn._FR13_APC_STALE_CAP_<carrier>` dict). This is the
  EXACT stale value a cache-ON hit serves.
- Phase INJECT (flag file = "1", shadow ON): the request re-prefills fresh; at
  the inject site we OVERWRITE carrier X with the stashed capture.

This needs the SAME request/prefix to pass through both phases — true for the
SWE multi-turn replay where each turn re-hits the prior turn's prefix.

### Strategy B (simpler, weaker) — "stale = the cached block-pool row the snapshot persisted"
Reuse the SNAP_FIX write-side machinery: the value the cache WOULD restore is the
block-pool row `block_ids[src_block_idx + accept_token_bias]` recorded at
`collect_mamba_copy_meta` (mamba_utils). Stash that row index at snapshot time;
on the next hit's fresh prefill, copy that row into the fresh running-state row.
This is purely intra-worker (no two-phase coordination) but only covers SSM/conv
(KV/pos have no mamba snapshot).

This instrument implements **Strategy A** for conv+ssm+kv (capture at the restore
read on a shadow-OFF step, inject at the fresh-write on a shadow-ON step) and a
DIRECT inject for pos (no capture needed — the stale position base is a pure
function of the block-aligned cached length, computable at inject time).

---

## Per-carrier feasibility + sites + patch

### 1. CONV — FEASIBLE (cleanest; reuses existing capture)
- Capture site: `gdn_linear_attn.py:890` prefill `causal_conv1d_fn` reads the
  restored conv window from `conv_state[non_spec_state_indices_tensor]`. The
  patcher already snapshots it (`_fr13_prefill_conv_restore_capture`, patcher
  L5491-5499) under `FR13_APC_CONV_RESTORE_CAPTURE`. Extend that capture to ALSO
  stash per (req_id, layer-prefix) into a worker global on a shadow-OFF step.
- Inject site: same prefill block, AFTER `causal_conv1d_fn` returns (patcher
  L5513). On a shadow-ON step, OVERWRITE `conv_state[indices]` with the stashed
  capture for the cache-hit rows (`has_initial_state` True).
- Patch: extend `prefill_conv_replacement` (patcher L5475) — see implementation.

### 2. SSM — FEASIBLE (reuses the prefill writeback)
- Capture site: `gdn_linear_attn.py:984` `initial_state = ssm_state[indices]`
  (the restored recurrent seed) on a shadow-OFF step — but the value we want as
  "stale" is the FINAL restored recurrent state the next decode consumes, which
  on a cache-ON hit equals `last_recurrent_state` written at L1004. Capture
  `ssm_state[indices]` rows AFTER the writeback on a shadow-OFF step.
- Inject site: `gdn_linear_attn.py:1004`
  `ssm_state[non_spec_state_indices_tensor] = last_recurrent_state` (patcher
  L5868). On a shadow-ON step, after the fresh write, OVERWRITE the cache-hit
  rows with the stashed capture.
- Patch: extend `prefill_scan_replacement` end (patcher L5868).

### 3. KV (full-attn) — FEASIBLE (separate flash_attn patch)
- Capture site: `flash_attn.py:869 reshape_and_cache_flash`. On a shadow-OFF
  step, after the cache write, read back the KV rows for the cached-prefix slots
  (`slot_mapping` covers the re-written positions). Stash per req/slot. NOTE:
  on a cache-ON hit the prefix slots are NOT re-written (slot_mapping only covers
  NEW tokens) — so the "stale KV" is what is ALREADY in the cached blocks. The
  faithful capture is therefore: read the cached-prefix KV rows from the cache
  tensor directly (via the request's block_table) on the hit step.
- Inject site: `flash_attn.py:869`, AFTER `reshape_and_cache_flash` on a
  shadow-ON re-prefill step (slot_mapping now covers the whole prefix => fresh KV
  was just written). OVERWRITE the cached-prefix slots with the stashed capture.
- Feasibility note: this requires the per-request block_table + the cached-prefix
  length at the worker. The flash_attn `do_kv_cache_update` does NOT carry req_id
  or block_table directly; it gets `slot_mapping`. Capturing/injecting at row
  granularity needs the slot list of the cached prefix. This is the HARDER
  carrier — implemented as a SLOT-RANGE capture/restore keyed off slot_mapping +
  a worker-published per-step "cached prefix slot set". See KV section: shipped
  as a DESIGN + a guarded scaffold (default-OFF, slot-set publish required);
  flagged `FR13_APC_INJECT_STALE=kv` with an explicit "needs slot-set publish"
  runtime guard so it never silently no-ops a faithful inject.

### 4. POSITION — FEASIBLE (direct, no capture)
- Inject site: patcher `_fr10_mrope_base` `fr10_phase4_patch_vllm_tree_gdn.py:10342`.
  The stale position base is the BLOCK-ALIGNED cached length (what a cache-ON hit
  would use as the RoPE base for the first decode), not the true generated
  length. Under shadow the re-prefill uses the TRUE length; inject the
  block-aligned stale base instead, for spec rows on the first decode after a
  hit. Computable directly: `stale_base = (num_computed_tokens // block_size) *
  block_size` is the cached-block-aligned base. No capture needed.
- Patch: guard the `_fr10_mrope_base` assignment under
  `FR13_APC_INJECT_STALE==pos`.

---

## Wiring

- New flags: `FR13_APC_INJECT_STALE` (master) + per-carrier helper flags reuse
  existing capture flags where possible.
- Launcher `-e` (fr13_launch_forked_fa2_tree_server.sh): add
  `FR13_APC_INJECT_STALE`.
- Worker bridge (`_fr13_write_apc_env_sidecar` keys[], patcher L17228): add
  `FR13_APC_INJECT_STALE`. The worker reinject is keyed on the `FR13_APC_`
  prefix (patcher L1335), so the new key auto-bridges into the worker os.environ.
- Scheduler/pid-1: the inject sites are all in the WORKER (gdn/flash) except POS
  (gpu_model_runner, also worker). No new pid-1-only flag is needed; the master
  reaches the worker via the sidecar bridge + launcher -e.

## Validation
- `ast.parse` the patcher.
- sim-apply each new string patch against the real
  /tmp/vllm_cu130_src source (assert anchors unique, replacement parses).
- `py_compile` a render of each patched file.
- NO GPU run (this is a code-study + patcher edit; correctness over speed).
