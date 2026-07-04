# FR13 Serving-Phase Unified-Mem Leak — Archaeology & Decisive Test Ladder

Date: 2026-07-04. Synthesis of 4 investigation digs + 1 adversarial verifier pass.
Leaking run: `output/fr13_leak_main/mainleak/`, git **1053c604**, arm **`nativemtp5_exseed`**
(native MTP-5 decode: `FR10_DECODE_MODE_DEFAULT=naive_mtp`, `ATTENTION_BACKEND=FLASH_ATTN`,
no `speculative_token_tree`, + APC + `FR13_APC_EXACT_SEED=1`). OOM-137 (rc=143, SIGTERM by
`gpu_oom_guard`) ~7 min into serving on task astropy-13453.

> **Headline (verifier-overturned framing):** This is **NOT a code regression** and **NOT the
> util/floor config drift** the first digs proposed. A same-config, same-task, *descendant*
> commit ran clean for 46 min the day before. The differentiator is **run-date / workload /
> environment** (the offloaded agent drove a ~9× denser re-prefill stream on 07-04), sitting on
> top of a **pre-existing, bounded but thin-margin** memory tax. Stop bisecting the 12 commits.

---

## 1. Confirmed leak profile

**Phase segmentation** (from `output/fr13_leak_main/memavail.log`, 10 s samples, boot 07:26:48Z;
cross-checked vs run.log + docker_full.log metrics):

| Phase | Window | MemAvailable | Note |
|---|---|---|---|
| idle | s1–4 | 117 → 114 GB | — |
| weights load | s5 | 114 → 83 GB | one-time |
| model-resident / profile | s5–39 | 83 → 75 GB | one-time |
| KV-pool touch-in + graph capture | s40 | 75 → 35 GB | one-time (0.82 util pool) |
| graph + warmup | s40–51 | 35 → ~20 GB | one-time |
| **SERVING fill (SWE 13453)** | **s52–67** | **~20 → ~10 GB** | **~4 GB/min, front-loaded** |
| **PLATEAU** | **s68–93** | **~9.2–9.6 GB** | **residual ~0.09 GB/min** |
| process death | s94 | jump → 15.7 GB | guard SIGTERM; engine was alive+healthy |

- **Rate:** front-loaded fill measured **4.12 GB/min** (s52 20.40 GB → s67 10.10 GB, 10.31 GB /
  150 s); consistent with the established "~3.5 GB/min" average-to-OOM. It is **not monotonic to
  zero** — it **plateaus** and the residual is only ~0.09 GB/min (== the documented ~87 MiB/min
  host-RSS ES-ckpt tail).
- **The kill was a trip-wire GRAZE, not exhaustion.** `gpu_oom_guard.log`: `avail=8984MiB <
  floor=9000` → `docker kill`. KV cache usage flat at **5.1–5.2 %** the whole serving phase, no
  CUDA-OOM traceback, rc=143. The working set had essentially **stabilized just below the 9000 MiB
  floor**.
- **Unit-fit (strong):** plateau floor **~9.0 GiB == 64 × 144 MiB**, where 144 MiB = one full
  48-GDN-layer EXACT-SEED checkpoint (48 layers × `[num_v_heads=48, head_k=128, head_v=128]` fp32
  CPU tensor = 3.00 MiB/layer). `_fr13_es_ckpt` is an `OrderedDict` LRU-**capped at 64**
  (`FR13_ES_CKPT_CAP`, patcher ~L885/17925). The cap fires (the flat plateau proves it); the cap
  **value** is the tax. On GB10 unified memory these `.cpu()` fp32 tensors count against
  MemAvailable as **host RSS** (so `cuda_alloc_mb` looks flat — the leak is host-side).
- **Fill driver:** `logs/fr13_apc_exact_seed_eng.log` shows 7152 `ES_WRITE` lines / **138 distinct
  block-hashes** in ~7 min (28.6 new hashes/min → × 144 MiB ≈ 4 GB/min). Each long-context
  re-prefill captures ~context_tokens/1024 block checkpoints (`MAMBA_BLOCK_SIZE=1024`).
- **Residual note:** serving consumed ~10–25 GB (boundary-dependent: ~20 GB from KV-touch-in start,
  larger from first-prefill start) but the capped store is only ~9 GiB, so **~10–16 GB of the peak
  is NOT the store** — it is transient re-prefill working set + per-request accumulator maps
  co-resident at peak (see Suspect 3).

---

## 2. Ranked surviving suspects (post-verifier)

### S1 — Workload / environment drift 07-03 → 07-04 (**the carrier**) — HIGH
- **Mechanism:** the offloaded codex/agent on alienware (auto-continue on, proxy forces temp 0.6)
  drove a **~9× denser / burstier** request stream on 07-04 — more/faster re-prefills, longer
  held-open context. That fills the cap-64 ES store to its 9 GiB high-water fast **and** co-resides
  transient re-prefill buffers + per-request accumulator maps at peak, pushing the working-set
  plateau down to graze the 9000 floor. On 07-03 the slacker trajectory let LRU eviction +
  between-turn request frees keep peak safely above the floor.
- **Evidence (decisive control):** `output/fr13_native_exseed_ctrl/m_nat_exseed_local`, git
  **0d12cdbf** (a *descendant* of 1053c604 with **more** code, incl. the leak-fixes), ran the
  **identical** `nativemtp5_exseed` arm on the **same** task 13453 on **07-03 for 46 min, swerc=0,
  no OOM, recovered to 110 GB** — with **util=0.82, floor=9000, CONV_SNAP_FIX=1, EXACT_SEED=1,
  SERVE_LOG=1** (identical config to the leak; only inert `FR13_LEAK_PROBE` differs). Clean-vs-leak
  sorts by **run date**, not code or config. Hash-capture rate: **2.2/min (07-03) vs 19.7/min
  (07-04)**, same code/config/task.
- **Test:** Rung 0 (artifact diff) + Rung 3 (commit-pinned ancestry) below.

### S2 — `BlockPool._fr13_es_ckpt` store: cap 64 × 144 MiB = 9.0 GiB host-RSS high-water — HIGH
- **Mechanism:** every re-prefill block boundary captures the full 48-layer GDN SSM state as a
  144 MiB CPU fp32 tensor into the block-hash-keyed store; it must outlive the request (cross-turn
  restore) and is pruned only by the LRU cap of 64. At 0.82 util the 9 GiB high-water sits right at
  the 9000 MiB guard floor once boot headroom is spent. **Thin margin, not a regression.**
- **Evidence:** plateau == exactly 64 × 144 MiB; cap 64 **byte-identical** at 703f9af4 / 4b68c8af /
  1053c604 (present in the clean 46-min and 89-min anchors too). This is a pre-existing bounded tax.
- **Test:** Rung 1 (`FR13_ES_CKPT_CAP=8`) — mitigation/margin, confirms dominance but does **not**
  explain why 07-03 survived at cap 64.

### S3 — Unbounded per-req / per-layer accumulator maps, reaped ONLY by `Scheduler._free_request`,
defeated by auto-continue holding one request open all session — MEDIUM
- **Mechanism:** with a session-long held request the per-turn maps accumulate `{pos → {48 layers →
  state}}`; under the denser 07-04 trajectory they grow enough (~10–16 GB) that, stacked on the
  9 GiB store, the working set crosses the floor. On 07-03 requests were freed between turns so the
  reaper fired and the maps stayed bounded. This is the **variable component** that turns a
  survivable 07-03 into an OOM 07-04.
- **Evidence:** reaper docstring (`fr10_phase4_patch_vllm_tree_gdn.py:7238-7291`) documents
  `_FR13_ES_PENDING_BY_REQ/_HASH_BY_REQ/_RESTORE_BY_REQ` + per-layer
  `_fr13_apc_pending_kvab/_fr13_apc_chunked_ckpt_by_req/_fr13_apc_abs_pos_by_req` as "populated
  EVERY turn but never popped when vLLM frees the request → ~0.7 GiB/min"; block-eviction reaper
  "never fires at low KV pressure". `run.log` = "auto-continue on".
- **Test:** Rung 4 (instrumented run) — watch per-map `len`/`g<bytes>`/rss climb; + whether
  `_free_request` fires mid-session under auto-continue. Rung 2 (`EXACT_SEED=0`) brackets the whole
  ES/APC family.

### S4 — Never-popped small-value maps + wrong-module pop bug — LOW
- `_FR13_ES_HIT_HASHES`, `_FR13_APC_SSM_CHUNKED_PTR_BY_REQ`, `_FR13_APC_SSM_CHUNKED_POS_BY_REQ`,
  `_LUMO_FA_TREE_ACCEPT_BY_REQ` are omitted from the reaper pop-list; and
  `_FR10_TREE_ACCEPTED_PATH_BY_REQ_ID` is popped from the **wrong module** (reaper does
  `getattr(gdn, ...)` but it lives in `vllm.v1.worker.mamba_utils`) → silent no-op. **Real bug**,
  but these hold ints/paths/hashes (not GB-scale) and on the native-MTP arm the tree maps stay
  empty. Passing glance in the probe only.

### S5 — torch caching-allocator fragmentation — LOW
- Reserved-climbs-while-allocated-flat is possible, but the ES store is host-side `.cpu()` fp32
  (shows as RSS, not `cuda_reserved`), so this is unlikely the carrier and the probe's
  `cuda_reserved` column would not even catch the store. Low.

---

## 3. DECISIVE TEST LADDER (cheapest-first)

Base invocation = however `fr13_leak_main` drove `scripts/fr13_bigdenom_swe_serve_variant.sh` with
arm kind `nativemtp5_exseed`, `subset_one_13453.json`, + the memavail sampler. Deltas below are
expressed as **env prepends / added flags** vs that base (the leak run overrode nothing, so it took
`serve_variant`'s `GPU_UTIL=0.82` and `gpu_oom_guard`'s `FLOOR_MIB=9000` defaults).
**LOCALIZE vs MASK is flagged per rung** — a rung that only moves the wall is a mitigation, not a
root-cause discriminator.

### Rung 0 — FREE, no GPU: artifact / trajectory diff (do this FIRST)
Compare the 07-03 clean control vs the 07-04 leak, no container:
```
diff <(sort output/fr13_native_exseed_ctrl/m_nat_exseed_local/container_env.txt) \
     <(sort output/fr13_leak_main/mainleak/container_env.txt)
# proxy / offload env + agent version:
diff  output/fr13_native_exseed_ctrl/*/proxy_env.txt      output/fr13_leak_main/*/proxy_env.txt
diff  output/fr13_native_exseed_ctrl/*/offload_*env.txt   output/fr13_leak_main/*/offload_*env.txt
# workload density (distinct block-hashes/min, turn count, max context pos):
grep -c ES_WRITE   output/fr13_native_exseed_ctrl/m_nat_exseed_local/logs/fr13_apc_exact_seed_eng.log
grep -oE 'hash=[0-9a-f]+' <log> | sort -u | wc -l     # 100 (07-03) vs 138 (07-04)
grep -oE 'pos=[0-9]+' <log> | sort -n | tail -1        # max re-prefill span
```
- **Discriminates:** confirms the 07-04 trajectory is denser (verifier already measured
  2.2 → 19.7 hashes/min) and pins the *environmental* delta (proxy/agent version, turn count).
  **LOCALIZES** the carrier to workload without spending a GPU minute. If density is ~equal, the
  carrier is a lower-level env change (docker image / allocator / co-resident proc) → weight Rung 3.

### Rung 1 — GPU, cheapest flip: ES store cap knockdown (current build 1053c604)
```
FR13_ES_CKPT_CAP=8   <base leak invocation, unchanged 0.82 util / 9000 floor>
```
- **Expected:** store high-water 64×144 MiB → 8×144 MiB ≈ 1.15 GiB; plateau rises ~8 GiB above the
  floor; 13453 completes without the guard kill.
- **Discriminates:** proves S2 (store) is the *dominant* bounded tax. **MITIGATION/MASK**, not a
  root cause — it does not explain why 07-03 survived at cap 64. If it still OOMs → a second
  unbounded structure (S3) dominates → jump to Rung 4.

### Rung 2 — GPU: EXACT_SEED off (bracket the whole ES/APC family)
```
FR13_APC_EXACT_SEED=0   <base leak invocation>
```
- **Expected / discriminates:** leak vanishes → the ES/APC machinery (S2+S3) is the accumulator
  family. Leak persists → carrier is *outside* EXACT_SEED (allocator/env, S5) → Rung 3/4.
  (Changes the arm's semantics — diagnostic only, not a shippable config.)

### Rung 3 — GPU: commit-pinned ancestry in TODAY's environment (decisive for code-vs-env)
Re-run a **clean anchor** with config **matched to the leak** so config is not a confound, driving
the **same offload trajectory**:
```
# checkout 703f9af4 (or 4b68c8af), rebuild patched image, same arm/task/sampler:
GPU_UTIL=0.82  GPU_GUARD_FLOOR_MIB=9000   <base invocation at 703f9af4>
```
- **Expected / discriminates:**
  - **Now OOMs (~GB/min)** → the suspect window 4b68c8af..1053c604 is **fully exonerated**; the
    carrier is environment/workload (matches the ancestry evidence — a descendant with *more* code
    ran clean). **This is the decisive LOCALIZER: code vs date.** Stop bisecting.
  - **Stays clean** → contradicts the byte-identical static diff; something in the window matters
    after all → then and only then bisect the 12 commits.
- **Caveat:** the offloaded agent is stochastic — to make this a fair test the **same trajectory
  must be replayed/pinned** (else a lighter agent run masks). Pin the request stream or replay the
  07-04 trajectory.
- **Do NOT** run the clean anchor at its *original* 0.6 util / 3000 floor as the "repro" — that
  **MASKS** (more headroom + lower trip-wire) and would falsely conclude "config drift".

### Rung 4 — GPU: instrumented run (exact structure localization)
Apply the validated diff (Section 4), then:
```
FR13_MEM_DUMP=1  FR13_MEM_DUMP_EVERY=50  FR13_MEM_DUMP_LOG=/logs/fr13_memdump.log \
    <base leak invocation at 1053c604 + sampler>
# read:
grep -a 'FR13MEM ' output/fr13_leak_main/<arm>/docker_full.log
grep -a 'FR13MEM ' output/fr13_leak_main/<arm>/logs/fr13_memdump.log
```
- **Discriminates (which token climbs monotonically across the 7-min window):**
  - `_fr13_es_ckpt=<len>/…/cap64` **pins at 64** while `rss_mb` keeps rising → a **second**
    host-side structure beyond the store (S3).
  - a per-req map `len` climbs unbounded (`_FR13_ES_PENDING_BY_REQ`, `_HASH_BY_REQ`) → reaper
    **defeated** by the held-open request (S3 confirmed).
  - `cuda_reserved_mb` climbs while all map lens + `cuda_alloc_cur_mb` stay flat → allocator
    fragmentation (S5).
- **INSTRUMENT CAVEAT (native-MTP arm):** the tree-path maps read via `gdn._FR13_REPLAY_LAYERS`
  (`_fr13_apc_chunked_ckpt_by_req`, `_fr13_apc_pending_kvab`) and
  `mamba_utils._FR10_TREE_ACCEPTED_PATH_BY_REQ_ID` will read **EMPTY** on `nativemtp5_exseed` (no
  tree kernel) — **do not misread empty as "no leak."** The informative columns on this arm are
  `_FR13_ES_PENDING_BY_REQ` / `_HASH_BY_REQ` len, `_fr13_es_ckpt` len (should pin at 64), and
  **`rss_mb`** (host-RSS is the unified-mem signal; `cuda_alloc_mb` looks flat).

---

## 4. Instrumentation diff (verbatim — SURVIVED verification)

Verifier verdict: **safe to apply.** Anchor `_fr13_sfwd_end(_fr13_sfwd_ev)` +
`record_function_or_nullcontext("gpu_model_runner: postprocess")` exists on main (patcher
L17509-17511 via `_patch_gpu_model_runner_sfwd_gpu_timer`, registered in `patch_steps` **before**
the new fn); all runtime symbols exist; guards verified: **default-OFF** (one module-bool/step),
**throttle every 50 steps**, **cuda-graph-safe** early-return on `is_current_stream_capturing`,
host-side allocator/`/proc` reads only (no device sync / DtoH / kernel launch), **fails-loud**
RuntimeError if anchor absent, bounded 20000-node tensor walk.

> **Authoritative artifact:** apply the on-disk, apply/compile-tested file
> `/tmp/claude-1000/-home-mark-shared/1297dd77-e0da-41fe-aceb-175500c156f5/scratchpad/memdump.diff`
> (248 lines; `git apply` / `patch -p1` clean against main @1053c604, patcher + injected code both
> `py_compile`). The listing below is that exact file, reproduced verbatim.

```diff
--- a/scripts/fr10_phase4_patch_vllm_tree_gdn.py
+++ b/scripts/fr10_phase4_patch_vllm_tree_gdn.py
@@ -17521,6 +17521,237 @@
     return True
 
 
+def _patch_gpu_model_runner_fr13_mem_dump() -> bool:
+    """FR13_MEM_DUMP (DEFAULT-OFF): per-engine-step in-worker memory + FR13-map
+    size probe for the serving-phase unified-mem leak hunt.
+
+    Docker-exec introspection is useless here (a fresh CUDA context sees none of
+    the worker's allocator state), so this MUST run inside the EngineCore/worker
+    process. Every FR13_MEM_DUMP_EVERY (default 50) execute_model steps it prints
+    ONE line ("FR13MEM ...") to stdout (-> docker_full.log) carrying:
+      - torch.cuda.memory_allocated / memory_reserved (MiB) + memory_stats
+        allocated/reserved current,
+      - process RSS from /proc/self/status VmRSS (the GB10 unified-mem signal),
+      - len + (cpu,gpu) tensor-bytes of every FR13 module-level req-keyed map in
+        gdn_linear_attn, the per-layer APC dicts summed over _FR13_REPLAY_LAYERS,
+        the block_pool EXACT_SEED checkpoint OrderedDict, and mamba_utils'
+        _FR10_TREE_ACCEPTED_PATH_BY_REQ_ID.
+
+    Host-side reads ONLY (no device sync, no DtoH, no kernel launch); skipped
+    while a cuda graph is capturing; wrapped so it can NEVER raise into serving.
+    Flag OFF => a single module-bool check per step => byte-identical default
+    path. Hooks the existing per-step host callback site right after
+    _fr13_sfwd_end(_fr13_sfwd_ev) in execute_model (injected by the SFWD timer
+    patch, which is registered immediately before this one)."""
+    text = GPU_MODEL_RUNNER_PATH.read_text()
+    sentinel = "# FR13_MEM_DUMP"
+    if sentinel in text:
+        return False
+
+    module_block = '''
+
+# FR13_MEM_DUMP: flag-gated per-engine-step memory + FR13-map-size probe.
+# Inert unless FR13_MEM_DUMP=1 => byte-identical to the unpatched step path
+# (one module-bool check per step). See the patch docstring for the design.
+_FR13_MEMDUMP_ON = __import__("os").environ.get("FR13_MEM_DUMP", "0") == "1"
+try:
+    _FR13_MEMDUMP_EVERY = int(
+        __import__("os").environ.get("FR13_MEM_DUMP_EVERY", "50")
+    )
+    if _FR13_MEMDUMP_EVERY < 1:
+        _FR13_MEMDUMP_EVERY = 50
+except Exception:
+    _FR13_MEMDUMP_EVERY = 50
+_FR13_MEMDUMP_STEP = 0
+_FR13_MEMDUMP_GDN_MAPS = (
+    "_FR13_ES_PENDING_BY_REQ",
+    "_FR13_ES_HASH_BY_REQ",
+    "_FR13_ES_RESTORE_BY_REQ",
+    "_FR13_ES_HIT_HASHES",
+    "_FR13_ES_BLOCK_PENDING",
+    "_FR13_BOUNDARY_LAST_WRITTEN_BY_REQ",
+    "_FR13_APC_SSM_LEAF_BY_REQ",
+    "_FR13_APC_CONV_LEAF_BY_REQ",
+    "_FR13_APC_SSM_ALIGNED_POS_BY_REQ",
+    "_FR13_APC_SSM_CHUNKED_PTR_BY_REQ",
+    "_FR13_APC_SSM_CHUNKED_POS_BY_REQ",
+    "_LUMO_FA_TREE_ACCEPT_BY_REQ",
+)
+_FR13_MEMDUMP_LAYER_MAPS = (
+    "_fr13_apc_pending_kvab",
+    "_fr13_apc_chunked_ckpt_by_req",
+    "_fr13_apc_abs_pos_by_req",
+)
+
+
+def _fr13_memdump_bytes(obj, budget):
+    """Bounded nested walk summing torch-tensor storage as (cpu_bytes,
+    gpu_bytes, budget_left). The node budget caps the probe cost so a huge
+    leaking map cannot make the diagnostic itself expensive."""
+    import torch as _t
+    cpu_b = 0
+    gpu_b = 0
+    stack = [obj]
+    while stack and budget > 0:
+        cur = stack.pop()
+        budget -= 1
+        try:
+            if isinstance(cur, _t.Tensor):
+                nb = cur.numel() * cur.element_size()
+                if cur.is_cuda:
+                    gpu_b += nb
+                else:
+                    cpu_b += nb
+            elif isinstance(cur, dict):
+                stack.extend(cur.values())
+            elif isinstance(cur, (list, tuple, set)):
+                stack.extend(cur)
+        except Exception:
+            continue
+    return cpu_b, gpu_b, budget
+
+
+def _fr13_memdump_step():
+    global _FR13_MEMDUMP_STEP
+    if not _FR13_MEMDUMP_ON:
+        return
+    try:
+        _FR13_MEMDUMP_STEP += 1
+        if (_FR13_MEMDUMP_STEP % _FR13_MEMDUMP_EVERY) != 0:
+            return
+        import os as _os
+        import sys as _sys
+        import torch as _t
+        try:
+            if _t.cuda.is_available() and _t.cuda.is_current_stream_capturing():
+                return
+        except Exception:
+            pass
+        parts = ["FR13MEM", "step=" + str(_FR13_MEMDUMP_STEP)]
+        try:
+            parts.append(
+                "cuda_alloc_mb=%.1f" % (_t.cuda.memory_allocated() / 1048576.0)
+            )
+            parts.append(
+                "cuda_reserved_mb=%.1f" % (_t.cuda.memory_reserved() / 1048576.0)
+            )
+            try:
+                _ms = _t.cuda.memory_stats()
+                parts.append(
+                    "cuda_reserved_cur_mb=%.1f"
+                    % (_ms.get("reserved_bytes.all.current", 0) / 1048576.0)
+                )
+                parts.append(
+                    "cuda_alloc_cur_mb=%.1f"
+                    % (_ms.get("allocated_bytes.all.current", 0) / 1048576.0)
+                )
+            except Exception:
+                pass
+        except Exception:
+            pass
+        try:
+            with open("/proc/self/status", "r") as _fh:
+                for _ln in _fh:
+                    if _ln.startswith("VmRSS:"):
+                        parts.append("rss_mb=%.1f" % (float(_ln.split()[1]) / 1024.0))
+                        break
+        except Exception:
+            pass
+        _budget = 20000
+        try:
+            _gdn = _sys.modules.get(
+                "vllm.model_executor.layers.mamba.gdn_linear_attn"
+            )
+            if _gdn is not None:
+                for _nm in _FR13_MEMDUMP_GDN_MAPS:
+                    _d = getattr(_gdn, _nm, None)
+                    if isinstance(_d, dict):
+                        _cb, _gb, _budget = _fr13_memdump_bytes(_d, _budget)
+                        parts.append(
+                            _nm + "=" + str(len(_d)) + "/c" + str(_cb) + "/g" + str(_gb)
+                        )
+                _lys = getattr(_gdn, "_FR13_REPLAY_LAYERS", None)
+                if isinstance(_lys, dict):
+                    parts.append("_FR13_REPLAY_LAYERS=" + str(len(_lys)))
+                    for _lnm in _FR13_MEMDUMP_LAYER_MAPS:
+                        _tot = 0
+                        _cb = 0
+                        _gb = 0
+                        for _ly in _lys.values():
+                            _ld = getattr(_ly, _lnm, None)
+                            if isinstance(_ld, dict):
+                                _tot += len(_ld)
+                                _c2, _g2, _budget = _fr13_memdump_bytes(_ld, _budget)
+                                _cb += _c2
+                                _gb += _g2
+                        parts.append(
+                            _lnm + "=" + str(_tot) + "/c" + str(_cb) + "/g" + str(_gb)
+                        )
+                _bp = getattr(_gdn, "_FR13_ES_BLOCK_POOL", None)
+                _ck = getattr(_bp, "_fr13_es_ckpt", None) if _bp is not None else None
+                if isinstance(_ck, dict):
+                    _cb, _gb, _budget = _fr13_memdump_bytes(_ck, _budget)
+                    parts.append(
+                        "_fr13_es_ckpt=" + str(len(_ck)) + "/c" + str(_cb)
+                        + "/g" + str(_gb) + "/cap"
+                        + str(getattr(_bp, "_fr13_es_ckpt_cap", "?"))
+                    )
+        except Exception:
+            pass
+        try:
+            _mu = _sys.modules.get("vllm.v1.worker.mamba_utils")
+            _md = (
+                getattr(_mu, "_FR10_TREE_ACCEPTED_PATH_BY_REQ_ID", None)
+                if _mu is not None else None
+            )
+            if isinstance(_md, dict):
+                _cb, _gb, _budget = _fr13_memdump_bytes(_md, _budget)
+                parts.append(
+                    "_FR10_TREE_ACCEPTED_PATH_BY_REQ_ID=" + str(len(_md))
+                    + "/c" + str(_cb) + "/g" + str(_gb)
+                )
+        except Exception:
+            pass
+        _line = " ".join(parts)
+        try:
+            print(_line, flush=True)
+        except Exception:
+            pass
+        _lp = _os.environ.get("FR13_MEM_DUMP_LOG")
+        if _lp:
+            try:
+                with open(_lp, "a", buffering=1) as _fh2:
+                    _fh2.write(_line + chr(10))
+            except Exception:
+                pass
+    except Exception:
+        pass
+'''
+    text = text + module_block
+
+    anchor = (
+        "        _fr13_sfwd_end(_fr13_sfwd_ev)\n"
+        "\n"
+        "        with record_function_or_nullcontext(\"gpu_model_runner: postprocess\"):\n"
+    )
+    if anchor not in text:
+        raise RuntimeError(
+            "FR13_MEM_DUMP: per-step hook anchor (_fr13_sfwd_end + postprocess "
+            "record_function) not found in gpu_model_runner.py -- the "
+            "FR13_SFWD_GPU_TIMER patch must be registered before this one."
+        )
+    inject = (
+        "        _fr13_sfwd_end(_fr13_sfwd_ev)\n"
+        "        _fr13_memdump_step()  # FR13_MEM_DUMP per-step host probe (inert unless FR13_MEM_DUMP=1)\n"
+        "\n"
+        "        with record_function_or_nullcontext(\"gpu_model_runner: postprocess\"):\n"
+    )
+    text = text.replace(anchor, inject, 1)
+
+    GPU_MODEL_RUNNER_PATH.write_text(text)
+    return True
+
+
 def _patch_fp8_utils_gb10_gemv_cfg() -> bool:
     """OPT-A: GB10/sm_121-tuned fp8 w8a8 block-scaled-mm decode config.
 
@@ -18455,6 +18686,7 @@
         (GPU_MODEL_RUNNER_PATH, _patch_gpu_model_runner_fr13_det_warn()),
         (GPU_MODEL_RUNNER_PATH, _patch_gpu_model_runner_replay_boundary_tap_d()),
         (GPU_MODEL_RUNNER_PATH, _patch_gpu_model_runner_sfwd_gpu_timer()),
+        (GPU_MODEL_RUNNER_PATH, _patch_gpu_model_runner_fr13_mem_dump()),
         (MAMBA_UTILS_PATH, _patch_mamba_utils_tree_accept_bias()),
         (MAMBA_UTILS_PATH, _patch_mamba_utils_boundary_log()),
         (MAMBA_UTILS_PATH, _patch_mamba_utils_preprocess_context_flag()),
```

---

## 5. EXCLUDED (with reason — do not re-chase)

| Excluded | Why refuted |
|---|---|
| **12-commit window 4b68c8af..1053c604 as a code regression** | Runtime patcher **byte-identical** clean↔leak except one hunk (60d7170c) that is **dormant** on the native arm (inside the tree branch, deref `attn_metadata.fr10_tree_parent`). 8/12 commits docs/output-only; 3 add uninvoked standalone scripts. A **descendant** (0d12cdbf, *more* code) ran clean on 07-03. |
| **`gpu_memory_utilization` 0.6 → 0.82 as the differentiator** | The 07-03 clean control used **0.82** (same as leak) and ran clean 46 min. The DIG that flagged this compared against a poorly-matched 07-02-morning 0.6 baseline. 0.82 is a real margin-reducer but is **held constant** clean↔leak. |
| **`gpu_oom_guard` floor 3000 → 9000 as the differentiator** | 07-03 control armed the guard at **9000** and never tripped for 46 min. Held constant clean↔leak. |
| **`FR13_APC_CONV_SNAP_FIX` 0 → 1** | **=1 in BOTH** the 07-03 clean control and the 07-04 leak → cannot explain a clean-vs-leak split. Its `_FR13_APC_CONV_LEAF_BY_REQ` is req-keyed and IS in the reaper cleanup list (bounded). Also user-excluded (=0 still leaked). |
| **`FR13_LEAK_PROBE` 0 → 1** | Used only at patcher L7277 as a **log-only** branch INSIDE the `_free_request` reaper (after the pop); read-only, cannot retain refs. Sole env delta vs the clean control — exonerated. |
| **`FR13_SERVE_LOG` 0 → 1** | =1 in BOTH clean control and leak. |
| **60d7170c REPLAY_ROUTE un-bake** | Inside the tree GDN kernel branch; native-MTP arm never enters it. Default value also leaves serving unchanged. |
| **2e1bbf53 / 47db4ecc / 7307b808 probe+driver commits** | CAPTURE_ONLY block gated on unset flag; standalone drivers not invoked by the bigdenom serving path (run used `fr13_bigdenom_swe_serve_variant.sh`, not `fr13_apc_multiturn_one_arm.sh`). Zero effect. |
| **GPU KV pool growth** | vLLM v1 `BlockPool` is fixed-size (no grow path); `kv_cache_usage_perc` = 5.2 % flat. The growth is **host RSS** (CPU tensors). |
| **LRU broken / not firing** | Refuted — plateau sits at **exactly** 64 × 144 MiB, proving the cap fires. The bug is the cap **value** (64 too high) + 144 MiB/entry, not a missing LRU. |
| **`_FR13_ES_PENDING_BY_REQ` as the classic ~0.7 GiB/min leak** | That specific map IS popped by the reaper *when `_free_request` fires*; the open question (S3) is whether auto-continue's held-open request prevents it firing mid-session — measured by Rung 4, not assumed. |
| **The steep ~4 GB/min segment = "the leak"** | That 390–690 s segment is one-time working-set fill; the sustained tail is ~0.09 GB/min. The kill is a **graze** of the fill's plateau against the 9000 floor. |
| **FIX A/B/C (ckpt LRU / PENDING map / GPU-resident ckpts)** | User already reverted all three and the leak **persisted** on the reverted build. Probe still measures them so flatness is positively confirmed, but they are not the fix. |

**Net:** the code and config are exonerated as *differentiators*; the leak is a **pre-existing
thin-margin memory tax** (S2 store + S3 accumulator maps) tripped by a **denser 07-04 workload /
environment** (S1). Rung 0 + Rung 3 settle code-vs-env; Rung 4 names the exact structure.
