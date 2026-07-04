# FR13 tree+cache give-up campaign — 2026-07-04

**End goal (user):** tree (cat8 or spine) + EXACT_SEED cache, NO agent give-up, nudge-free qwen-code.
Assumed-working baselines: native-MTP-5+cache, tree+no-cache. Nudge/retry = banned (lossy behavior).

## 1. Clean baseline matrix — main @ 1053c604, task astropy-13453, nudge-free qwen-code (offloaded), temp 0.6

| arm | KIND / env | verdict | dur | patch | memory | cache engaged |
|---|---|---|---|---|---|---|
| m_native_cache (mainleak) | nativemtp5_exseed | fence-graze kill (agent mid-work, NOT give-up) | ~14min | — | asymptote ~8.9GB | ES_SEED 3216, bs=1024 |
| m_tree_nocache | cat8 + APC CONFIG_ONLY | **resolved** | 1719s | 551B | **flat 14.6GB**, 29min, zero drift | none (correct) |
| m_tree_cache_base | cat8 + EXACT_SEED, MAMBA_BLOCK 1024, fp32 | **GIVE-UP** (`failed`, patch_bytes=0) | 843s | 0 | flat 12.3GB, **no OOM** | ES_SEED 1008, bs=832 |

Load-bearing control: the give-up reproduced with ZERO memory pressure → the give-up is a real
cache-ON-tree serving defect, not a memory artifact. Baselines match the charter's assumptions.

## 2. The "~10GB/min leak" is RESOLVED as a bounded footprint grazing our own fence

Full evidence: `research/fr13_workflows/FR13_LEAK_ARCHAEOLOGY.md` (adversarially verified).

- The 07-04 native+cache run was `docker kill`ed by **our gpu_oom_guard at avail=8984MiB vs floor=9000MiB**
  — a 16MiB graze of a **decelerating** curve, not a runaway (guard log 07:42:12).
- Trajectory = front-loaded fill (~4GB/min touch-in) → **plateau ~9.2GB**, residual ~0.09GB/min.
  Plateau ≈ 64 × 144MiB = the ES checkpoint store at its LRU cap (48 GDN layers × fp32, per block boundary)
  — a designed, bounded, pre-existing tax.
- **Code + config exonerated**: 0d12cdbf (descendant of main tip, MORE code, identical arm/task/config)
  ran 46min clean on 07-03. Clean-vs-leak sorts by RUN DATE → workload/environment density
  (07-04 agent traffic ≈ 9× denser re-prefill block-hash rate: 19.7 vs 2.2/min).
- FIX A/B/C "identical failure trajectories" explained: same legitimate curve, same fence.
- Real secondary bug (keep): per-request accumulator maps are reaped only by `_free_request`,
  which the proxy's auto-continue held-open request defeats → unbounded growth on long sessions
  (patcher :7238-7291 docstring). Fix later; not today's dominant term.
- Tree+no-cache: flat 14.6GB for 29min → base stack leak-free.

**Durable fix (proposed, not yet applied): GPU_UTIL 0.82 → ~0.78** (+4.7GB fence clearance, no cache-machinery change).
Diagnostic setting used today: `GPU_GUARD_FLOOR_MIB=6500` (guard purpose preserved; kernel-OOM danger is ~1-2GB).
NOT chosen: FR13_ES_CKPT_CAP cut (changes restore behavior → could confound give-up work).
Instrumentation ready if growth ever looks unbounded: `research/fr13_workflows/fr13_memdump.diff` (FR13_MEM_DUMP=1).

## 3. Cherry-picked build — branch `fr13-mainpick` @ 08b629ef (pushed)

10 commits from fr13-apc-ssm-shadow onto main 1053c604, zero conflicts, adversarially verified
byte-exact to the GPU-validated branch state (patcher==837236d0, launcher/serve_variant==e6d0214a);
both features default-OFF-inert; default path byte-identical to main.
Features: SCAN_ALIGN=recompute + FR13_RECOMPUTE_NODE_PARALLEL (grid-z, GPU-untested) and
Path A FR13_APC_BLOCK_REFOLD + 832-boundary bind fix (prior: fires-but-inert for the give-up).
Plan detail: `research/fr13_workflows/FR13_SESSION_DIGEST_AND_PLAN.md`.

## 4. Phase 4 (in flight): fix arms on fr13-mainpick, same task/harness/guard-6500

- m_tree_recompute_np: cat8 + CACHE + FR13_SCAN_ALIGN=1 MODE=recompute ALLOW=1 NP=1 (auto-fallback NP=0 if boot fails)
- m_tree_patha: cat8 + CACHE + FR13_APC_BLOCK_REFOLD=1 (expected-null confirmatory)
- NOT-give-up bar: engagement well past 843s/8-turn baseline + patch_bytes>0; resolve = win.
- Interpretation guards: CONV_SNAP_FIX=1 baked everywhere (baseline already shifted 2→8 turns);
  n=1 per arm at screen stage — any positive gets a repeat (n≥2) + base repeat before declaring.

Results: `output/fr13_phase4_mainpick/PHASE4_SUMMARY.txt` (output/ gitignored — numbers get banked here on close).

## 5. Phase-4 RESULTS (2026-07-04): both fixes NULL — give-up persists

| arm (fr13-mainpick, task 13453, guard 6500) | engaged proof | verdict | dur | patch |
|---|---|---|---|---|
| m_tree_cache_base (main) | ES_SEED 1008 | give-up | 843s | 0 |
| m_tree_recompute_np (NP=1) | env: SCAN_ALIGN=1 MODE=recompute NP=1; ES_SEED 240 | give-up | 751s | 0 |
| m_tree_recompute (NP=0) | env: SCAN_ALIGN=1 MODE=recompute NP=0; ES_SEED 240 | give-up | 624s | 0 |
| m_tree_patha | REFOLD_APPLIED=432 ENTER=8 (docker_full.log); RESTORE_OTHER=30 | give-up | 613s | 0 |
| m_tree_nocache (control) | no eng log (correct) | **resolved** | 1719s | 551B |

- SCAN_ALIGN=recompute does NOT fix the give-up (either NP variant). The 6c70ed8c "engages 46min"
  anecdote is dead (memory-error/n=1 confound, as the bbd9619c close-out suspected).
- Path A engages (fold fires 432x) but REFOLD_RESTORE_OTHER=30 with no RESTORE_USED → the bind still
  doesn't land the fold into the restore path; give-up unchanged. "Bind-fixed but inert" CONFIRMED.
- Positive side-finding: NP=1 boots, cuda-graph-captures, serves (was GPU-untested).
- Carrier therefore OUTSIDE the SSM scan/re-fold family. Leading untouched suspect: full-attn
  POSITION base on cache-hit boundary turns (patcher _fr10_mrope_base, 2026-06-28 finding, never
  verified fixed for the tree path). Give-up autopsy workflow in flight.

## 6. Native+cache completion run + give-up autopsy (2026-07-04, later)

- **m_native_cache under guard-6500: dur=997s, patch_bytes=398, verdict=failed, mem min 8.86GB, COMPLETED (no OOM).**
  - Boundary story CLOSED: plateau ~8.86GB < old 9000MiB fence; the fence was killing healthy runs. Guard floor
    or GPU_UTIL must change permanently (proposal stands: GPU_UTIL 0.82→0.78, keep floor 9000).
  - NO give-up: full 17min engagement + real patch (vs Jul-2 resolve — today's patch failed eval, n=1, temp 0.6).
- Give-up phenotype ledger: 0-byte/no-tool-call/off-task-drift = 4/4 tree+cache arms, 0/2 elsewhere → TREE×CACHE interaction.
- Autopsy (research/fr13_workflows/FR13_GIVEUP_AUTOPSY.md, adversarially verified): give-up = coherent off-task
  topic-hijack (web-dev attractor keyed on "HTML write" tokens), NOT char-8/garble (0 XML warnings in give-ups);
  NO cache hits ever fired (row0_hit=False ×50 all arms) → not discrete hit-corruption; first divergence = turn-1
  route flip at COLD prefill (cache-ON numerics: align-mode chunked prefill + fp32 SSM dtype), deterministic per config.
- In flight: m_tree_cache_base_r2 (B=1 same-config repeat = playbook gate). Then full-attn capture on first
  post-restore turn (the unmeasured subsystem, H3).

## 7. Repeat gate PASSED (2026-07-04): give-up is reproducible

- m_tree_cache_base_r2 (identical config, fresh boot): dur=530s, patch_bytes=0, verdict=failed, mem min 14GB.
- Day tally: tree+cache give-up **5/5** (base x2, recompute NP=0/1, patha) vs **0/3** non-tree-cache
  (nocache resolved; native+cache engaged 17min w/ 398B patch, failed eval, n=1).
- Conclusion: reproducible config-deterministic TREE x CACHE defect. Next = staged diagnostics:
  turn-1 route-flip replay probe (N=8/config) + H3 full-attn capture under cache.
