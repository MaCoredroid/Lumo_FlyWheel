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

## 8. align1024 A/B: NULL — give-up survives perfect grid geometry

- m_tree_cache_align1024 (APC_BLOCK_SIZE=1024: bs=1024==block==max_num_batched confirmed in boot+ES_GATE):
  dur=554s patch_bytes=0 verdict=failed. Same give-up band. ES_SEED 816.
- Excludes: 832-grid / overshoot-invariant violation as the give-up carrier (still a hygiene fix:
  docs require block 64-multiple + >=816 + ==max_num_batched; tree arms must pass APC_BLOCK_SIZE).
- Tally: 6/6 tree+cache give-ups (base x2 @832, recompute NP0/NP1 @832, patha @832, align @1024).
- Survivors (must be TREE x CACHE interactions): (1) full-attn position base on restored prefixes x
  tree depth-positions (_fr10_mrope_base class, named 06-28, never verified fixed for tree) -> H3 capture;
  (2) ES write-side row selection during TREE decode (wrong-bank-row class) -> code read + targeted probe.

## 9. BAKED (2026-07-04, user): APC_BLOCK_SIZE defaults to 1024 on mainline

- Launcher now defaults `APC_BLOCK_SIZE:=$APC_MAX_NUM_BATCHED_TOKENS` (=MAMBA_BLOCK_SIZE=1024) in the
  cache-ON branch + FAIL-LOUD guard enforcing the documented invariant: 64-multiple AND >=816 AND
  == max-num-batched (FR13_APC_EXACT_SEED_SUCCESS.md constraints; overshoot fix d228c76b/#45238).
- Closes the config hole where tree cache arms booted at vLLM's native 832 rounding while
  max_num_batched stayed 1024 (overshoot-invariant violation, silent).
- Scope: cache-ON boots only; CONFIG_ONLY + non-APC paths byte-identical. NOT the give-up fix
  (§8: give-up is geometry-independent) — this is cache-correctness hygiene.

## 10. Fix-arm dispositions after the geometry bake (2026-07-04)

**Recompute (SCAN_ALIGN=recompute, ±NODE_PARALLEL): RETIRED for the give-up — wrong-in-theory.**
1. Target = SSM scan/seed realization content; Track A measured the restored SSM state BIT-EXACT
   (min_dist=0.0 x 48 layers) — it aligns something already exact at the boundaries.
2. Decisive control 0d12cdbf: native+EXACT_SEED has HIGHER first-token TV than cat8 yet resolves
   => seed-realization drift does not predict give-up => aligning it cannot fix give-up.
3. SCAN_ALIGN/K1 family separately refuted as a lossless lever (moved deployed scan state 22.8x
   AWAY from native).
4. Both NP variants gave up (fastest arms, turn 2); give-up is geometry-independent so 1024 does
   not rescue the mechanism. Positive side-finding kept: NP=1 boots + cuda-graph-captures + serves.

**Path A (FR13_APC_BLOCK_REFOLD): theory NOT refuted — today's null was VACUOUS. RETRY QUEUED.**
1. Fold fired 432x but RESTORE_USED=0 / RESTORE_OTHER=30: its output was never consumed; the
   theory (faithful write-side checkpoint on the accepted path) went untested.
2. Bind publishes at runtime block size: at 832 the bind grid was tangled (0/60 bind diagnostic);
   at the baked 1024 the bind grid coincides with the checkpoint grid — a different experiment.
3. Overlaps live suspect #2 (ES write-row selection under tree decode; code-read in flight).
4. Retry conditions: geometry=1024 (baked default), engagement gate = RESTORE_USED>0 (REFOLD_APPLIED
   alone is insufficient — proven today); run ONLY if the code-read confirms the write-side seam is
   broken in a way the fold would fix; if the write path reads clean, Path A stays retired too.
   Skeptical prior stays on record: c9deb112 ("restored boundaries already faithful via
   prefill-capture") — drawn at 832 with the overshoot hole open, so re-evaluate at 1024.

**HIT_RECURRENT_SUFFIX=0 confirmed deliberate** (launcher :286 un-bake 2026-06-27): HRS re-prefills
the hit remainder through the recurrent kernel (≠ chunked realization, cannot be bit-exact);
superseded by EXACT_SEED; empirically identical give-up with HRS on/off. No action.

## 11. Code-read verdicts (workflow wv4iajyw8, adversarially cross-examined): BOTH remaining seams REFUTED — sole survivor = H1 cold-prefill amplification

- **Position base: REFUTED.** _fr10_mrope_base = unconditional num_computed_tokens_cpu (patcher :11828-30),
  no cache/restore branch; remap fires ONLY on tree decode rows (:11831) so it cannot touch turn-1 prefill;
  KV slots computed flat BEFORE remap (:11692-96) = cache-correct; depth-collapse is pure topology, proven
  lossless (tree+no-cache resolves with identical remap). Jun-28 finding was diagnosis-only, its trigger
  (real hit) never fires here. H3 capture = positive closer only (predict positions[root]==num_computed_tokens, 0.0 vs oracle).
- **SSM write-row: REFUTED.** SNAP_FIX redirects the decode snapshot to the committed accepted-leaf
  (:13715-20 via :8439 publish; banked FAITHFUL 240/240). The EXACT_SEED SSM-snapshot override
  (_FR13_APC_SSM_CHUNKED_PTR_BY_REQ) is DEAD CODE — zero producers — which EXPLAINS the every-gate
  ES_REDIRECT_FALLBACK. ES prefill/block-hash captures recompute from tokens (row-independent);
  decode-drain relay is a disabled no-op (:18922). Residual real defect: CONV_SNAP_FIX still PARTIAL
  (wrong-row conv write possible at num_accepted>1) — inert here (no hit consumes it), hygiene-fix later.
- **Sole survivor H1:** cache-ON boot flags (align-mode prefill machinery + fp32 SSM dtype) perturb the
  turn-1 COLD-prefill GDN realization (~1e-2 class); TREE decode amplifies it into a deterministic turn-1
  route flip (monolithic vs Explore-subagent, byte-identical input) -> long single-context drift onto the
  web-dev attractor -> no tool call. Explains all 6 facts incl. native robustness + geometry-independence.
  RESIDUAL (medium): cache-root-defect vs benign trajectory-selection — but the engineering target is the
  same either way: **cold-prefill realization parity** (cache-ON turn-1 bit-exact to CONFIG_ONLY turn-1).
- **Rescue-path gap noted and REJECTED per charter:** terminal completions end stop_reason=null so
  auto-continue never fires; injecting a nudge there would convert stops to resolves — BANNED (nudge=lossy,
  user 2026-07-04). Recorded only so nobody re-proposes it as a fix.
- **Decisive next test (flag-only): FR13_APC_EXACT_SEED=0** keeping prefix-caching+chunked+fp32+1024.
  Persists => ES machinery exonerated, carrier = plain cache-config x tree (H1 confirmed).
  Resolves => EXACT_SEED prefill-capture side-computation implicated.
- Path A retry (§10) condition FAILED: write side reads clean => nothing for the fold to fix. Parked
  unless user overrides.

## 12. Flag-decomposition ladder (H1 chase)

| arm | flags (delta vs resolving CONFIG_ONLY) | verdict |
|---|---|---|
| m_tree_cache_es0 | +prefix-caching(align) +mamba-block-size +fp32-ssm-dtype +block-size-1024, ES=0 | **give-up** (637s/0B, es_seed_lines=0) — 7/7; EXACT_SEED machinery EXONERATED |
| m_tree_cache_dtypeauto (running) | same minus fp32 (dtype=auto) | ? |

- es0 give-up => carrier is in the BASE cache config, not the ES seed/restore machinery.
- dtype=auto resolves => fp32 SSM cache dtype is the perturber. Still gives up => carrier =
  align-mode prefill machinery itself (inseparable from prefix caching; mamba_cache_mode='all'
  hard-blocked for Qwen3-Next+spec) => fix = bit-exact align-mode prefill parity.
- Note: decomposition must run from the give-up side — vLLM rejects the mamba dtype/block flags
  without --enable-prefix-caching, so fp32 cannot be added to CONFIG_ONLY.
- Determinism note (user challenge, answered): launcher pins --seed 0 every boot => per-config
  trajectories are REPRODUCIBLE; and 6/7 give-up variants differ numerically from each other =>
  the route-decision shift is SYSTEMATIC (larger than noise), leaning corrupt over select.
  Margin probe (teacher-forced, first-divergent-token, tree+cache vs tree+no-cache) promoted to
  the decisive instrument after the flag ladder.

## 13. CORRECTION (Fable first-hand re-read of code+artifacts): "no cache hits" premise FALSE — conv wrong-row carrier RESURRECTED as lead

- Prometheus per-task brackets (definitive, not throttled): tree+cache base = **138,944 prefix-cache HITS**
  (77% of 180,635 queries); es0 = 239,616; native = 524,288 (engages!); nocache = 0. The autopsy's
  "row0_hit=False x50 => no hit ever fired" was the 50-forward throttle window (all cold turn-1) —
  scalar-blindspot class. Every hit-dependent inference downstream (incl. code-read "conv write inert",
  cross-exam "H1 sole survivor") inherited the false premise.
- **Lead hypothesis (E): conv wrong-row write -> hit-restore corruption.** CONV_SNAP_FIX is PARTIAL
  (code-read confirmed in code): at num_accepted>1 the conv snapshot can capture base col-0 instead of the
  committed accepted-leaf. Run-level: base arm logged **ES_REDIRECT_FALLBACK x384** (wrong-row copied)
  vs native **0**. Commit trail: 4063b346 "conv carrier confirmed, source-rewrite next" — the source-rewrite
  was NEVER WRITTEN (branch pivoted to Track A/B; 1053c604 only baked the flag).
- Explains the full 2x2 + all nulls: tree-only (wrong leaf among co-resident rows), cache-only (consumed
  on hit), ES-independent (conv restore in es0 too), geometry-independent, recompute/PathA-null (SSM-side),
  degrade turns 2/3/6 = at/after first hit turn. The turn-1 route flip remains a separate (possibly benign)
  cold-config effect — dtypeauto arm splits its dtype term; margin probe quantifies it.
- **Fix of record to implement: the conv source-rewrite** (finish 4063b346's "next"): publish the conv
  accepted-leaf row for EVERY snapshot boundary (incl. zero-accept + commit-timing gaps) and rewrite BOTH
  src_ptrs AND sizes to the committed-leaf row so the redirect can never fall back. Then tree+cache give-up
  gate; then lossless gates; then the 16-task matrix.

## 14. CORRECTION of §13's marker reading + refined lead (E')

§13 conflated markers. Corrected semantics (from patcher :13722-13760 + :8362-8444):
- ES_REDIRECT_FALLBACK is the **SSM snapshot** redirect (not conv); its fallback target is the
  SNAP_FIX committed-leaf = CORRECT ROW but the RECURRENT-decode realization — the exact realization
  EXACT_SEED v2 was designed to replace with chunked checkpoints, whose producer
  (_FR13_APC_SSM_CHUNKED_PTR_BY_REQ) is DEAD CODE. So 384 fallbacks = "tree snapshots stored
  recurrent-realization states 384 times", not wrong-row evidence.
- The marker only logs under EXACT_SEED=1 (es0 silent ≠ clean) and only fires when the tree
  committer publishes the leaf map (native zero lines = NOT APPLICABLE, not clean). Three arms,
  three different meanings of "no lines" — the uniform-instrumentation fix (§15) addresses this class.
- **Refined lead E′ (coherent with the whole June arc):** turn-2+ hits consume mamba-cache states
  written by DECODE-side snapshots. For TREE these are (a) recurrent-realization (chunked producer dead)
  and (b) branch-CONTAMINATED committed-leaf states (STree 2505.14969; spine+APC was PROVEN lossless
  while tree was 0/4 — the June finding). HRS partially fixed this family (17/48→5/48), was un-baked in
  favor of EXACT_SEED, whose decode-side half never got wired ⇒ the June carrier is UNFIXED in today's
  builds. Native immune: single-spine states are never branch-contaminated. Explains every cell + null.
- Fix of record (supersedes conv-only §13 fix): **wire EXACT_SEED's decode-side chunked-realization
  checkpoint producer** (or SGLang-style recompute-clean-linear-scan) for boundaries crossed during
  decode, + the conv twin (incl. the wrong-row gap). Conv-only rewrite would leave the SSM realization
  term in place.

## 15. Flag ladder CLOSED: 8/8 — carrier is prefix-caching x tree decode itself

| variant (all cat8+cache on 13453, nudge-free) | give-up |
|---|---|
| base @832 x2, recompute NP0/NP1 @832, patha @832 | 5/5 |
| align1024 (ES=1, fp32) | give-up |
| es0 (ES=0, fp32) | give-up (240k hits) |
| dtypeauto (ES=0, auto dtype) | give-up (344k hits) |

Eliminated by the ladder: geometry (832/1024), EXACT_SEED machinery, fp32 SSM dtype, SSM-content fixes
(recompute), Path A fold. Common to all 8: --enable-prefix-caching + tree decode + hits consumed.
=> E' stands as the lead: decode-side snapshot states (recurrent-realization; tree branch-contaminated
committed-leaf per STree/June evidence) written into the mamba cache and consumed by turn-2+ hits.
Cold-route-flip (A) present in all variants but cannot be sufficient alone (native+cache engages with
the same cold config; every cold-config permutation still gave up only WITH tree).
NOTE: per-turn reset_prefix_cache as an A-vs-E' splitter is REJECTED — reset itself is a known
corruption artifact (project_fr13_apc_cache_causes_corruption).
NEXT = fix of record (§14): wire EXACT_SEED's decode-side chunked-realization producer + conv twin;
gated behind the FR13_OBS instrumentation land (in flight) so the fix run has trustworthy counters.

## 16. HRS vs EXACT_SEED, the missing producer, and the Path A reframe (fix spec for task #7)

**The gap both addressed:** a hit needs the GDN state at the hit boundary; stored states come from two
kernels that are R-equal but bit-different (chunked-prefill vs sequential-recurrent, ~0.0078/boundary).

- **HRS** = accept mixed lineage, roll the nearest checkpoint forward over the suffix with the RECURRENT
  kernel. Replaced the stock restart-fold (the gross-corruption carrier). Partial by construction: right
  continuation, wrong realization -> residual 5/48 layers mismatched (from 17/48, some 27x, pre-HRS).
- **EXACT_SEED v2** = keep the cache PURE chunked-lineage: capture only chunked-kernel realizations at
  64-aligned boundaries; restore remainders THROUGH the chunked kernel; restored state bit-exact to
  cache-OFF (min_dist=0.0 proven). Better design; HRS correctly un-baked in its favor.
- **The defect: only EXACT_SEED's PREFILL half was wired.** Decode-crossed blocks enter the cache via the
  decode-side snapshot = recurrent-kernel bank state, and for TREE a branch-co-resident bank state
  (June: spine+APC lossless, tree+APC 0/4). The snapshot redirect CONSUMER exists
  (_FR13_APC_SSM_CHUNKED_PTR_BY_REQ read, patcher ~:13644) but NO PRODUCER assigns it -> falls back
  every time (384x logged). Native tolerates its own recurrent-lineage writes (single spine = eps-off
  only); tree's are contaminated (grossly off) -> only tree x cache dies.
- **Path A REFRAME:** the fold IS the orphaned producer — an accepted-path chunked-FLA re-fold of the
  decoded 64-block (fired 432x) — but it publishes to _FR13_ES_PENDING_BY_REQ/try_bind, a channel the
  snapshot never reads (hence REFOLD_APPLIED=432 / RESTORE_USED=0). Two halves of the same fix, built
  at different times, never connected.

**Task #7 fix spec:** connect the fold output to _FR13_APC_SSM_CHUNKED_PTR_BY_REQ (+ conv twin, + close
the conv wrong-row gap). Flag-gated default-OFF, byte-identical OFF. Caveats to verify in implementation:
(a) fold FIDELITY never end-to-end validated (firing != correct output) — validate vs a token-recompute
oracle on the first wired boundary; (b) fold inputs must be tokens+prior-chunked-state (lineage-clean),
not bank rows. Engagement gate on the fix run (FR13_OBS counters): snapshot redirect_used>0 AND
redirect_fallback==0 AND conv_leafmap_miss==0; then the give-up gate.

**Sequencing decision (user):** WAIT for FR13_OBS to land before implementing — same-file edits, and the
fix gate depends on the new counters. GPU idles in the interim by design.

## 17. Fix v1 gate result: PHENOTYPE CHANGED — drift give-up GONE, new wall = context compression

m_tree_cache_fixv1 (cf060312: BLOCK_REFOLD=1 REFOLD_TO_SNAPSHOT=1 CONV_LEAF_COMPLETE=1, 13453):
- dur=872s, 16 turns (free_request_fired=16) vs 2-6-turn give-ups — ~3x engagement; coherent throughout;
  terminal = qwen-code CONTEXT COMPRESSION FAILURE at ~49k tokens (COMPRESSION_FAILED_EMPTY_SUMMARY),
  NOT the off-task drift. patch_bytes=0/failed formally, but the old give-up phenotype is GONE (n=1).
- CONV HALF ENGAGED PERFECTLY: conv_snapshot_events=25200, conv_leafmap_hit=25200, miss=0 — wrong-row
  conv write eliminated. Dose-response holds: conv partial 2->8 turns (4063b346), conv complete ->16.
- SSM HALF NOT ENGAGED: REFOLD_APPLIED=1344 (fold runs) but refold_published=0; snapshot_events=48,
  redirect_engaged=96, ALL fallback (redirect_used=0). Cause: fold abs-base (_FR13_REFOLD_ABS) only
  seeded on E3 hit-restore and evidently never seeded here -> _rf_blk_end never matches the aligned
  boundary -> publish dropped (STRUCTURAL#1 generalized).
- First run judged on unthrottled OBS counters (es_row0_hit_true=4368 — honest hit counting throughout).
- ITERATION 2: (a) seed _FR13_REFOLD_ABS at prefill/E0 from absolute committed length (covers cold +
  hit turns) -> refold_published>0 -> redirect_used>0 -> SSM snapshots become chunked-realization;
  (b) investigate the compression failure source (model-empty summary = serving residue at high ctx
  vs harness artifact; alienware-side proxy logs — GB10-side pair dumps empty for offloaded arms);
  (c) re-run gate; the context-wall question may also re-implicate the monolithic route (context bloat).

## 18. Route mini-probe (free, from today's 8 traces): the turn-1 route flip is SYSTEMATIC, tree x cache

first tool call at turn 1: agent(subagent) = tree+nocache, native+cache (2/2); read_file(monolithic) =
ALL SIX tree+cache variants (base x2, align1024, es0, dtypeauto, fixv1) — spanning geometry, ES on/off,
dtype, conv/refold fixes. Config-independent coin-flip explanation ~1% class => systematic logit shift
pinned to tree x cache. Carrier must be what all six share vs native: ALIGN-MODE CACHE PREFILL feeding
the TREE decoder's first steps (third seam, separate from the write-side v2 work — es0 in the set proves
v2 cannot move it). Consequences: (a) route seam = cold-prefill realization parity (candidate fix class:
make align prefill end-state match CONFIG_ONLY's realization); (b) route choice decides context economics
(subagent quarantine vs monolithic 49k wall) -> the context track covers it harness-side regardless;
(c) full N=8 route probe still queued for true per-config distributions (task #1).

## 19. BAKED (user): FR13_APC_CONV_LEAF_COMPLETE default ON

The drift-give-up fix is now default for every cache-ON boot (launcher :308 block). Rationale §17-18;
correctness-directional, native auto-no-op, cache-OFF byte-identical. REFOLD_TO_SNAPSHOT stays default
OFF until the v2 gate proves the SSM half (refold_published>0, redirect_used>0, lossless gates).

## 20. Bake policy for the refold half (user 2026-07-04): measure quality contribution vs speed tax first

v1 (CONV_LEAF_COMPLETE) is baked: zero-compute (row-index publish), proven give-up carrier. The refold
publish (v2/v3, FR13_APC_REFOLD_TO_SNAPSHOT) is a COMPUTE fix with a real cost (fold = chunked FLA
re-fold per 64-block + clones/stashes; Path A history estimated ~0.8-1% decode tax class) and its
behavioral benefit is UNPROVEN — v1 alone already yields coherent 16-turn engagement, and the SSM
realization epsilon may be behaviorally negligible (native ships the same epsilon class and works).

DECISION GATE before any bake of v2/v3 — A/B (cat8+cache conv-only vs conv+refold), measure:
 (i) agent quality: turns / patch / resolve on 13453 (+1-2 more tasks), route distribution (probe metric);
 (ii) losslessness: per-token argmax-flip vs each arm's own no-spec oracle (binding instrument);
 (iii) speed: canonical fr13_measure deploy-speed s/fwd + derived TPS (tax must be quantified).
BAKE only if (i) or (ii) improves materially AND the tax is acceptable (~<=1%); otherwise v2/v3 remain a
documented opt-in correctness lever (the EXACT_SEED-complete configuration) and the DEFAULT deployed
tree+cache config = conv-only. The 16-task speed matrix (task #4) runs cat8 = conv-only default unless
this gate says otherwise.

## 21. Context track landed (workflow wvwfyxud7, adversarially audited): 49k wall root-caused + R1/fence SHIPPED

- ROOT CAUSE (verified in-image, qwen-code v0.19.4): hard limit 48875.2 = computeThresholds over a
  65536 context budget = 131072 (served) minus a 65536 OUTPUT RESERVE the agent can never use (proxy
  caps output at 32768). Audit re-derived the exact formula from the minified chunk and fixed two
  math bugs in the draft.
- SHIPPED R1: QWEN_CODE_MAX_OUTPUT_TOKENS=32768 threaded into QWEN_CODE_TEMPLATE -> hard limit 75304
  (+54%, clears the 13453 class with ~26k headroom). Uniform across arms.
- SHIPPED fence fix (§2 durable): serve_variant GPU_UTIL default 0.82 -> 0.78 (+4.7GB guard clearance;
  orthogonal to R1 — KV pool pre-allocated at boot, R1 is footprint-neutral).
- RE-BASELINE REQUIRED (audit): post-R1 runs are a different regime — do not compare to pre-R1 banked
  numbers; re-run all four arms before cross-arm claims.
- AUDIT OVERRULED R2a auto-exclusion: labeling compression-abort terminals is fine; EXCLUDING them
  selectively rescues tree+cache (§18 route flip makes it arm-coupled). Exclusion only after the R3b
  serving-health probe, identical rule all arms.
- Compression side-query facts: served by the arm under test at temp 0.6 + presence_penalty 1.0 (!)
  asking 9-section XML over 49k tokens, maxAttempts~1, hard-stop by design; auto-continue confirmed
  inert on qwen-code AND a banned nudge anyway. R3b probe (2x2 boot x sampling) queued GPU-gated;
  R2b (runner image v2: retry + deterministic elision) + R4 (structured compaction + subagent
  quarantine) = later engineering cycle. Prompt caching OFF on proxy (cache_read=0) = separate
  gated experiment. Full doc: FR13_CONTEXT_COMPRESSION_DESIGN.md.

## 22. Route probe (paired seeds, N=16/arm): SYSTEMATIC turn-1 behavioral collapse under tree+cache

Per-request seed=k (common random numbers), byte-identical real turn-1 request, cold prefill enforced:
| arm | delegation | read_file | NO_TOOL |
|---|---|---|---|
| cat8 no-cache | 16/16 | 0 | 0 |
| native+cache | 16/16 (15 Explore + 1 todo_write) | 0 | 0 |
| cat8+cache (conv baked) | **4/16** | 6/16 | **6/16** (3 finish=length, 3 GENUINE finish=stop) |

- Route luck REFUTED at N=16: P(delegate) collapses 1.0 -> 0.25 only in tree x cache; native+cache is
  as healthy as no-cache => the cold term is the INTERACTION (align-prefill x tree decode), not cache alone.
- Divergence point precision: think channel byte-identical across all 16 seeds in EVERY arm; the fork is
  exactly at the first <answer> token — cache arms fork there immediately, no-cache forks ~10 tokens deeper.
  The distortion concentrates at the high-entropy route-decision position.
- 3/16 genuine tool-less stops at turn 1 = the give-up class exists at turn 1 under tree+cache only.
- Full reduce: research/fr13_workflows/fr13_route_probe_3arm_reduce.txt. 2x2 control (native_nocache)
  running; cold-prefill localization ladder staged (task #10) = the fix path.

## 23. Route probe 2x2 COMPLETE: pure interaction, attribution closed

native_nocache control (N=16 paired seeds): 14x Explore + 2x todo_write, 16/16 tool_calls — statistically
indistinguishable from native+cache (15/1). Full square: cache alone = NO effect (native cells identical);
tree alone = healthy; ONLY tree x cache collapses (4/16 delegate, 6/16 NO_TOOL). The cold-prefill route
defect is 100% the interaction term. Fix path = task #10 ladder (first divergent op feeding the tree
decoder's first steps under align-prefill). Full reduce: research/fr13_workflows/fr13_route_probe_2x2_reduce.txt

## 24. v3 gate: BEST TREE+CACHE RUN EVER — 32 turns, real patch, no wall; refold half-live

m_tree_cache_fixv3 (d3f9b325: conv baked + BLOCK_REFOLD=1 REFOLD_TO_SNAPSHOT=1, R1 budget 75k, fence 0.78):
- dur=1187s, 32 turns (2x fixv1, 5-16x the give-up era), **patch_bytes=377** (FIRST tree+cache patch ever),
  verdict=failed (eval). ZERO compression events — R1 confirmed end-to-end (wall eliminated).
- Give-up phenotype: fully dead across drift AND wall classes. Remaining gap = patch QUALITY
  (native+cache also failed w/ 398B patch; only no-cache resolved; all n=1 -> R1-regime re-baselines +
  §20 A/B are the discriminators).
- Refold pipeline (unthrottled counters, OBS_LAST_SUMMARY authoritative — atexit JSON raced teardown,
  known limit): seeds refold_seed_statep=6910, contiguity refold_abs_seeded=1440 vs fail-closed
  lineage_block=96, realign_skip=31584, **refold_published=96 (FIRST ever >0)**, pub_miss=48;
  BUT redirect_engaged=672 / **redirect_used=0** -> the consume hop never matched a published state.
  v4 = consume-hop debug: publish-key (req,layer.prefix)->{pos,tensor} vs consumer lookup key/timing;
  grep shows the consumer's literal map reference may have drifted from the producer's (:8702-8732).
- conv again perfect: 68352/68352, miss=0 (n=2 for the baked fix).

## 25. Cold-ladder verdict: PREFILL EXONERATED BIT-EXACT — route drift localizes to DECODE accumulation (or graph replay)

Paired eager teacher-forced boots (cache vs CONFIG_ONLY, byte-identical turn-1 prompt, max_tokens=1):
- ALL 48 GDN layers: final_state/conv/core max_abs = 0.0, argmax flips 0/5184 rows. Route (token-1)
  logits BYTE-IDENTICAL (top-5 equal, margin 7.125 both, max_abs=0.0).
- => the cold-prefill COMPUTATION is bit-exact across the cache config bundle (align mode, mamba-block,
  fp32 dtype, block-size) in EAGER. The §22 route collapse is NOT a prefill numerics defect.
- RECONCILIATION (two live axes): (1) POSITION — the ladder measured token-1 ('Let', think-start); the
  route fork is at <answer> after ~hundreds of TREE-DECODE steps: drift accumulates in decode under the
  cache config (think text identical across arms per probe => the logit delta at the fork position is
  the target); (2) GRAPH MODE — ladder ran eager (capture requirement); probe/production run
  FULL_AND_PIECEWISE (same label both arms, but cache config changes the captured programs).
- NEXT INSTRUMENT (task #10 iter-2): teacher-force prompt + identical think text to the <answer> fork,
  capture logits THERE, 2x2 (cache/config_only x eager/graphs). Discriminates decode-accumulation vs
  graph-replay vs (if 0.0 everywhere) sampler-path.
- H3 CLOSED positively for free: --enforce-eager threading fixed + full_attn dumps captured both arms
  (h3_threading_confirmed=true). The June-28 position-wiring suspicion is now measured-and-closed.
- OPS incident banked: the ladder reduce OOM class (unified-memory reducer) -> session protections +
  capped self-demoting scopes + streaming reducer (memory file updated).

## 26. v4 gate + honest downgrade: single-gate verdicts retired — RATES are the instrument now

- m_tree_cache_fixv4 (f2f2b7a5, refold flags ON): 735s, 6 turns, 0B, give-up class (monolithic route,
  tool-less terminal). Counters: conv 5568/0 clean; snapshot_events=0 = RUN-LENGTH artifact (no 1024
  boundary crossed in 6 turns) => refold consume liveness UNTESTED by this draw, v4 edit unfalsified
  (scope re-audited clean; default path flag-gated).
- DOWNGRADE (red-team of our own narrative): the conv dose-response (2-6 -> 8 -> 16 -> 32 turns) was
  OVERFIT to n=1 draws. §22 already measured healthy-trajectory rate ~4/16 under tree+cache with conv
  fixed. v3's 32-turn/patch draw and v4's 6-turn draw are both consistent with that distribution.
  DURABLE claims kept: conv wrong-row eliminated (counters, n=3 runs), context wall eliminated (R1),
  prefill bit-exact (§25). OPEN: the decode/graph term (§25) still degrades the trajectory distribution.
- NEW INSTRUMENT OF RECORD: the RATE MATRIX = R1-regime re-baseline x §20 A/B combined:
  arms {tree_nocache, native_cache, tree_cache_convonly, tree_cache_refold} x server seeds {0,1,2,3}
  (SEED env varies the boot seed => genuine trajectory draws; seed pinned => per-config determinism).
  Metrics: give-up/patch/resolve rates, turns, route choice, refold counters (long draws exercise
  boundaries), per-arm memory. Speed tax measured separately logging-off (task #4).

## 27. Refold status HONEST: NEVER FIRED — usefulness UNMEASURABLE until liveness proven

redirect_used=0 on ALL refold runs to date: v1 published=0 (unengaged), v3 published=96 but used=0
(the -1 skew, v4's fix), v4 snapshot_events=0 (6-turn draw too short to cross a 1024 boundary).
=> the v4 consume-hop fix is UNTESTED and refold's agent-behavior value is UNKNOWN (can't help via a
path that never executes).

TWO-STAGE REFOLD GATE — both BEFORE the speed gate (task #4), per user:
 STAGE A (LIVENESS, mechanical): prove redirect_used>0 on a draw long enough to cross >=1 boundary
   (snapshot_events>0). Needs a non-give-up draw; the rate matrix's refold arms across 4 seeds are the
   vehicle (>=1 long draw expected). If refold NEVER fires across all draws => it is IRRELEVANT to agent
   behavior in practice (give-ups terminate before turns-2+ restores matter) => valid conclusion: deploy
   conv-only, refold stays a documented lever. This is the instrument-vacuity guard: a "refold arm" with
   used=0 IS conv-only in disguise; the A/B would be vacuous.
 STAGE B (USEFULNESS, behavioral): ONLY on liveness-confirmed draws, compare refold-ON vs conv-only
   give-up/patch/resolve RATE + route distribution (the §20 A/B, rate-based per §26 — NOT single draws).
   Bake refold only if it materially improves the rate AND its speed tax is <=1% (fr13_measure).

ORDERING: eager route probe (running, route-drift localization) -> rate matrix (Stage A liveness +
Stage B usefulness, conv-only vs refold x seeds) -> speed gate on the winning deployed config.
Refold is a candidate lever, NOT a shipped fix, until Stage A+B pass.

## 28. REFRAME (user): the route drift IS a cache LOSSLESSNESS VIOLATION on the tree DECODE path

Refold EXONERATED: route probe cache arm has FR13_APC_BLOCK_REFOLD=0 REFOLD_TO_SNAPSHOT=0 (only
EXACT_SEED=1). The drift exists with refold fully off => not refold.

User's losslessness logic (correct): probe resets prefix cache before EVERY sample => cold prefill,
ZERO hits => cache-ON must == cache-OFF on turn-1 IF the cache is lossless. It doesn't (cache collapses
delegation 16/16 -> ~4/16; eager cache confirming ~same => real numerics not graph). Eliminations:
prefill bit-exact (§25), no hit (reset), refold off => the ONLY residue is DECODE (tokens 2..~30).
CARRIER = enabling the cache CONFIG (--enable-prefix-caching + block-mamba layout) is NOT decode-lossless
on the TREE path: the co-resident node-bank GDN rows are read/written through block-pool indexing
(cache-ON) vs contiguous (cache-OFF), diverging fp-accumulation per step until the route token flips at
temp 0.6. Tree-only because native=single spine (one row, layout-invariant); tree=co-resident rows
(layout-sensitive) — the campaign's core co-residency, now at DECODE not restore.

FIX TARGET (crisp): make tree+cache-ON decode BIT-IDENTICAL to tree+cache-OFF on identical input.
NEXT INSTRUMENT (task #10 iter-3): DECODE-STEP ladder = extend §25's cold ladder INTO decode —
teacher-force the identical token sequence through cache-ON vs cache-OFF tree boots (eager), capture
per-DECODE-STEP GDN state (SSM+conv, all rows) + logits, report first divergent decode step + op.
Then classify WIRING (block-index/layout) vs KERNEL (accumulation order) and fix to 0.0.

## 29. EAGER 2x2 VERDICT: DECODE NUMERICS confirmed, graph-replay artifact REFUTED

| mode | cat8_cache healthy-route | cat8_nocache healthy-route |
|---|---|---|
| GRAPH (§22, FULL_AND_PIECEWISE) | 4/16 (25%) | 16/16 |
| EAGER (--enforce-eager in live argv) | 3/16 (todo_write 3, read_file 10, NO_TOOL 3) | 12/12 todo_write (100%) |

Collapse magnitude IDENTICAL across modes (cache ~19-25% healthy, nocache ~100% both) => the route-token
logit shift is present WITHOUT cuda-graph capture => REAL decode numerics, NOT a graph artifact. All temp
0.6, paired seeds. Confirms §28: cache-config decode losslessness violation on the tree path.
NEXT: decode-step ladder (below) localizes the first divergent decode step/op; fix to 0.0; re-probe.

## 30. INSTRUMENT CORRECTION (user): seeded temp-0.6 paired streams, NOT teacher-forcing; + refold is structurally untestable by a reset-probe

Two user corrections to the §28 decode-ladder plan:
1. NO teacher-forcing (off-distribution risk). Use the campaign's SAME-SEED PAIRED STREAMS: cache-ON vs
   cache-OFF, both tree, temp 0.6, FIXED SEED. Lossless <=> identical logits+seed => identical samples.
   First divergent TOKEN localizes where cache-config numerics first flip a real sampled token
   (on-distribution, deterministic, paired). Per-step GDN state capture: first STATE divergence (precedes
   the token flip) pins the carrier op.
2. Refold is structurally UNTESTABLE by the route probe: probe RESETS cache each sample => cold => NO hit
   => refold (restore-side) cannot fire, cannot affect the result. So refold CANNOT solve the turn-1 route
   drift — but the reset-probe also cannot MEASURE refold. Deployment give-up = TWO possibly-separate
   carriers: (a) turn-1 COLD decode drift (route probe; refold irrelevant) + (b) turn-2+ RESTORE
   losslessness (refold's actual domain, never exercised by a reset-probe).

NEW INSTRUMENT (task #10, replaces teacher-force ladder): 2-TURN SEEDED PROBE. A 2-turn conversation at
temp 0.6 + fixed seed; turn-2 re-sends turn-1 and HITS the cached prefix. Arms: cache-OFF (lossless ref)
vs cache-ON conv-only vs cache-ON refold. Per-decode-step dump {logits, sampled token, GDN state all
rows}. Reduce: first-divergent-token vs cache-OFF ref, split by turn-1 (cold => localizes route-drift
carrier, refold-invariant) vs turn-2 (hit => does refold push divergence later / reduce it = refold's
value). ONE probe answers both the route-drift localization AND the refold-usefulness question at token
level (cheaper + deterministic than the rate matrix). Byte-identical-stream gate = campaign standard.

## 31. PIVOT (user): real task, not synthetic probe

Localization stays on REAL input (route_probe_payload IS the real astropy-13453 turn-1 request — not
synthetic); DROP the reconstructed synthetic turn-2. Split:
- TURN-1 decode-carrier localization: real turn-1 request, seeded temp-0.6, cache-ON-conv vs cache-OFF
  (CONFIG_ONLY = lossless ref) + a cache-OFF FLOOR-BRACKET 2nd boot (bounds the tokens-11-71 autotune
  fork so it can't masquerade as signal), per-decode-step GDN STATE capture (state diverges BEFORE the
  sampled route token flips + beats the ~1-ULP floor; cold ladder showed cross-boot state hits 0.0 when
  truly identical). First divergent-STATE step/layer/component = the carrier op. WINDOWED two-pass
  (OOM-safe per §25 incident).
- REFOLD value + BEHAVIORAL verdict: the REAL agentic multi-turn task (rate matrix), NOT a synthetic
  2-turn — refold's turn-2+ restore domain tested on the live SWE task where hits genuinely fire.
Instrument built by workflow wf_8d07a324-3fb (seeded2turn_* + decode_gdn_capture patch, default-OFF
byte-identical); adopting its turn-1 half + floor bracket, deferring turn-2 to the real rate matrix.

## 32. FLOOR MEASURED: cross-boot byte-divergence confounds turn-1 localization; two clean signals survive

Floor bracket (cat8_nocache bootA vs bootB, SAME seed, lossless-vs-itself, turn-1): 0/16 seeds
byte-identical — streams fork at char ~25 (MID-PREAMBLE, before the route token ~char 130). Confirms
feedback_no_cross_boot_byte_gate (autotune forks decode at ~step 6). => BYTE first-divergence localization
of the turn-1 route drift is DEAD (floor forks before the route decision). Two signals survive:
1. ROUTE DISTRIBUTION over seeds (§22/§29): cache 3-4/16 vs nocache 16/16 — robust to the floor (huge
   effect vs per-seed noise). Behavioral truth, but not the OP.
2. FLOOR-REFERENCED EARLY-STEP STATE (pass-2): prefill is bit-exact cross-boot (§25, step-0 identical all
   boots); decode divergence accumulates from step 1. Capture GDN state at steps 0-6 (pre-fork) for
   cache-ON + nocache-A + nocache-B; the carrier is the earliest step where cache-vs-nocache state
   divergence EXCEEDS the nocache-A-vs-B floor. Set pass-2 STEP_LO=0 STEP_HI=6.
3. SAME-BOOT MISS-vs-HIT (cache arm turn-1 vs turn-2, ZERO cross-boot confound): lossless restore =>
   turn-2(hit) byte-identical to turn-1(miss); a fork = pure RESTORE-losslessness failure (refold's
   domain). This is the clean refold gate, floor-free.
Pass-1 (streams+hits+obs) completing; pass-2 = floor-referenced state at steps 0-6.

## 33. Seeded-probe refold arm VACUOUS (killed); refold value -> real rate matrix only
cat8_cache_refold (eager seeded, flags ON, turn2_hit=5/5, es_seed_applied=481): redirect_used=0
refold_published=0 snapshot_events=0 — refold does NOT engage (short single-turn completions cross no
1024 DECODE boundary => no decode snapshots to redirect; conv-only in disguise, §27 trap). Killed w/
refutation. Refold value deferred to the REAL agentic rate matrix (long trajectories cross boundaries;
v3 gate had snapshot_events=336). Seeded probe keeps its cache-conv route/state + same-boot miss/hit signals.

## 34. DECISIVE: two floor-controlled cache route carriers; fix target = ROUTE-DISTRIBUTION parity (not byte-exact)

Seeded probe, 3 arms (payload identity confirmed: turn1_send==turn2_send byte-identical, resend mode):
- CONTROL (engine determinism): nocache resend (same prompt+seed twice, no hit) = bytes DIFFER 16/16
  (same-boot nondeterminism exists) BUT route-flip 0/16. nocache_b identical: 0/16 flip. => the route
  decision is ROCK-STABLE on the lossless reference despite byte nondeterminism. => "same-seed byte-
  identical streams" is NOT achievable even same-boot; ROUTE is the robust gate.
- CARRIER 1 (cold-config decode, turn-1): cache-conv 5/16 healthy vs nocache 16/16 AND nocache_b 16/16
  (floor arm agrees => NOT autotune, systematic cache-config effect). The turn-1 route collapse is real;
  conv fix did NOT fix it (still 5/16).
- CARRIER 2 (restore path, turn-2 hit): cache flips route 8/16 miss->hit while nocache flips 0/16
  (floor-free, same-boot). The EXACT_SEED restore flips the agent route half the time.

TWO SEPARATE cache losslessness violations, both ROUTE-level, both floor-controlled. NO FIX yet.
FIX TARGET REFRAMED: NOT byte-exact decode (engine isn't byte-reproducible even lossless) but
ROUTE-DISTRIBUTION PARITY: cache-ON turn-1 must match nocache 16/16 (carrier 1) and cache hit must not
flip route vs miss (carrier 2). Localization next: pass-2 state capture for carrier 1 (cold decode op);
carrier 2 = the EXACT_SEED restore path (which state the hit restores differently). Both are behavioral-
lossless targets, achievable without chasing unattainable byte-exactness.

## 35. Carrier-1 op-localization CONFOUNDED — cache==nocache BIT-EXACT through step 7; the effect is distributional, past the autotune fork

Pass-2 logit capture (decode steps 0-7, cache-conv vs nocache vs nocacheB, per-step spine logits):
- Steps 0-7: cache-ON logits BIT-IDENTICAL to nocache (torch.equal=True, max_abs=0.0) AND to nocacheB.
  Distinct steps confirmed (argmax varies 79566/79320/3074/...). The cache config does NOT perturb early
  decode — bit-exact (extends §25 prefill-exact into decode).
- The byte fork (§32, char~25 = token ~8) is at step 8, JUST past the window; there ALL arms
  (nocache A/B AND cache) fork together (autotune floor). The route DECISION is ~step 30, downstream of
  the step-8 fork.
- CONSEQUENCE: carrier-1's OP is NOT deterministically localizable. The carrier acts at the route token
  (~step 30), but the autotune floor forks the streams at step 8 FIRST => no clean same-input cross-arm
  window at the route token. Through step 7 (clean window) there is NOTHING to localize (bit-exact).
- So carrier-1 (route dist 5/16 vs 16/16, systematic per §34) is a DISTRIBUTIONAL effect emerging past
  the fork, NOT a single early-decode op. The FR13_DECODE_GDN_CAPTURE state dump also never fired
  (env-threaded but insertion point off the executed path; H3-class) — but the logits already answer it.
- STATE-CAPTURE FIX PATH IS A DEAD END for carrier-1. Two viable paths:
  (a) TEACHER-FORCE the model's OWN preamble (the byte-identical natural continuation, NOT off-distribution
      forcing) to reach the route token with identical input across arms => clean cache-vs-nocache logit
      compare AT the route token. This is the only deterministic localization left.
  (b) Treat carrier-1 as a behavioral RATE effect (charter = no give-up) and measure the deployment
      give-up RATE on the real agentic task (rate matrix) with conv-fix + R1 already in. If acceptable, ship.

## 36. REFOLD v4 CONFIRMED NON-FUNCTIONAL (its proper domain); route framing refined; give-up is a RATE

trf_s2 (refold ON, 39 turns = long trajectory, boundaries crossed): redirect_engaged=576 refold_published=192
snapshot_events=288 es_seed=4978 -- the decode-side machinery ENGAGED for the first time. BUT redirect_used=0
(all 576 FELL BACK). => v4's per-boundary-history + apos-indexed consume-hop STILL does not match/consume;
refold has NEVER consumed a state across v1-v4 (4 attempts). => ALL refold arms are VACUOUS (conv-only in
disguise, §27/§33 trap); the §20 refold A/B CANNOT be run until the consume-hop works. trf_s2's 39-turn
engagement was a good DRAW (refold inert), NOT a refold effect. DECISION: shelve refold (4 failed consume
attempts, deep rabbit hole, zero payoff); campaign focus = carrier-1 + give-up RATE.

ROUTE FRAMING REFINED (trf_s2 overturns "read_file => give-up"): read_file route gives up (trf_s1 4t) OR
engages long (trf_s2 39t+patch, under R1 budget). So the give-up is NOT deterministic on the turn-1 route;
it's a broader trajectory-variance RATE. Tree+cache real-task tally: tnc 1 resolved(56t) | tcv/trf cache
arms so far 1 engaged(39t,patch,failed-eval) + 2 give-up(4-6t) = 0 RESOLVED, ~1/3 non-give-up, all n small.
Non-give-up config = still no such thing; it's a draw rate.

## (ops) 2026-07-05 infra hang: codex-on-alienware stall
tcv_s2 hung 37min (codex container on alienware Up but producing nothing; vLLM fine, offload LINK up,
trace=0/0-tool-calls). Killed rc=137, recorded NA (excluded from rate — infra, not give-up). WATCHDOG
HEURISTIC for the loop: a run with codex_trace.jsonl size 0 (or unchanged) for >15min = HUNG -> kill +
clean alienware codex + re-run. Replacement queued with AGENT_WALL_S=2400 to bound future hangs.

## 37. CARRIER-1 IS THE MASTER SWITCH: cache flips round-1 route agent->read_file (9/9 real runs), read_file never resolves

Round-1 tool across ALL tree+cache real runs (give-up AND non-give-up): read_file 9/9. The non-give-up ones
(fixv1 17t, fixv3 32t, trf_s2 39t, trf_s3 45t) ALSO took read_file. The ONLY run on the healthy 'agent'
(delegation) route = tnc_s1 (NO-CACHE), which RESOLVED. => "non-give-up = not-read_file" REFUTED.
REAL PATTERN: no-cache -> agent -> RESOLVES; tree+cache -> read_file -> 0 RESOLVES (0/9), give-up ~1/3.
The cache UNIVERSALLY flips the round-1 route agent->read_file in the real harness (9/9, stronger than the
seeded probe's ~1/3 healthy — real-harness turn-1 prompt more biased). Give-up-vs-engage WITHIN read_file is
later trajectory variance, decoupled from round 1. => carrier-1 (round-1 route flip) is the MASTER SWITCH:
it both starves resolution (monolithic route never resolves) and causes ~1/3 give-ups. Fixing carrier-1
(restore the agent/delegation route under cache) plausibly fixes BOTH the give-up AND the 0-resolve.
Highest-leverage target; refold/conv/speed all downstream of it.

## 38. DECOUPLE fix = TWO TIERS; incremental (SGLang-style) is the deployment target and is realization-consistent by construction

The fix for carrier-1 (§37 master switch): cache full-attn KV, put GDN on a recompute path (not block-pool)
so decode realization matches no-cache (§35 bit-exact) -> route returns to 'agent'. TWO tiers:
- T1 FULL-COLD (safe, slow): recompute the WHOLE GDN prefix contiguously each turn (= the exact no-cache
  path). Provably no route flip (§35). PROVE-THE-CONCEPT tier: does the round-1 route return to agent?
- T2 INCREMENTAL (SGLang-style, the DEPLOYMENT target, user 2026-07-05): store the GDN RECURRENT state at
  turn/radix-node boundaries AS THE CANONICAL-KERNEL REALIZATION, and on the next turn recompute FORWARD
  over only the NEW tokens THROUGH THE CANONICAL (recurrent) KERNEL. Cost O(new tokens) not O(prefix).
KEY INSIGHT (resolves the safe-vs-fast tension): T2 is realization-CONSISTENT BY CONSTRUCTION — store
recurrent + forward recurrent = same kernel throughout, NO chunked-vs-recurrent mismatch. Our block-pool
failure was storing a CHUNKED realization then restoring into RECURRENT decode (mismatch); SGLang/T2 stores
recurrent + continues recurrent (consistent). So T2 can be BOTH fast AND route-flip-free — unlike the
block-pool approach. GATE for T2: store-recurrent+forward-recurrent == full-cold within route-distribution
(re-run the route probe under T2). SEQUENCE: T1 proves the route fix -> T2 makes it deployable.
Prerequisite = the decouple feasibility (workflow wemm3xfin: can vLLM cache KV while GDN recomputes).

## 39. DECOUPLE ABANDONS EXACT_SEED (not on top of it); EXACT_SEED was never the culprit
§15: es0 (EXACT_SEED OFF) STILL gave up => the route flip is vLLM's BASE prefix-caching putting GDN state
in the block-pool, NOT EXACT_SEED. EXACT_SEED/conv/refold were add-ons patching the block-pool GDN RESTORE
(a path decouple removes). Under decouple: KEEP full-attn KV cache (standard vLLM, TTFT win, exact);
ABANDON block-pool GDN state (-> recompute T1 / recurrent-node T2); EXACT_SEED + conv-fix + refold all
become MOOT (no block-pool restore to fix). Net: shed the whole EXACT_SEED->conv->refold stack, replace
with SGLang-validated recompute. CRUX (workflow wemm3xfin): the model INTERLEAVES attn+GDN layers, so a
decoupled forward must use CACHED KV for attn layers but RECOMPUTE-from-tokens for GDN layers IN ONE PASS
-- is per-layer-type cache behavior expressible in vLLM's hybrid allocator, or does it force both into the
same block-pool? Config/patch vs deep-patcher hinges on that.

## (ops) 2026-07-05 pgrep-wait DEADLOCK — the chained queue stalled
The 3 chained waiters (tnc_extra/nat_extra/tcv_s2_redo) each waited via `pgrep -f 'trf_first.sh|...'`.
BUG: pgrep -f matches a process's full COMMAND LINE, and each waiter's cmdline CONTAINS the sibling script
names (in its own wait pattern) -> every waiter matched a sibling -> all 3 deadlocked while the real driver
was DONE and the GPU sat idle ~15min. FIX: killed the waiters, replaced with ONE sequential driver
(rate_finish.sh: tnc_s2/s3, nat_s1/2/3, tcv2redo) — single script, sequential run() calls, no inter-script
pgrep waits. LESSON: never gate a queued job on `pgrep -f <sibling-script-name>` (self/sibling-match);
use a container/marker-file check or a single sequential driver.

## 40. DECOUPLE DEAD but PREMISE CORRECTED + carrier LOCALIZED to a SLOT seam (workflow wemm3xfin, source-verified)

DECOUPLE = STRUCTURALLY IMPOSSIBLE (live vLLM /tmp/lumo_tree_patch_probe/vllm): (1) ONE num_computed_tokens
per request drives the whole 64-layer stack (kv_cache_manager.py:176-216) — can't feed attn the hit-suffix
while GDN gets the full prefix in one forward; (2) config coupling forces mamba_cache_mode none->align when
prefix-caching on (config.py:436-480; Qwen3-Next forbids 'all' qwen3_next.py:717); (3) coordinator MIN
collapses the combined hit across attn+mamba groups (kv_cache_coordinator.py:453-544) so uncached GDN kills
the attn TTFT too; (4) even a shadow-recompute: GDN state depends on lower full-attn OUTPUTS (shared
residual, interleaved layers qwen3_next.py:512-535) -> must recompute O(L^2) attn -> no TTFT; (5) align
RESTORES GDN by cheap gather/scatter COPY, not recompute — "uncache to recompute" is strictly worse. §38/§39
T1/T2 decouple/recompute = RETRACTED.

PREMISE CORRECTED (my §37-39 over-generalized): GDN block-pool caching is BENIGN IN ISOLATION — native+cache
is 16/16 healthy (§22/§23) with GDN cached in align. Only TREE(cat8) x cache collapses (4/16). => the
carrier is NOT "GDN caching" but the TREE x ALIGN interaction.

CARRIER LOCALIZED (source, not numeric — this is why §35 op-localization failed: it's a SLOT/WIRING address,
not a kernel op): align gathers the GDN checkpoint rows at (seq_len-1)//block_size — a ROTATING block-pool
row (utils.py:879-891); the tree drafter binds its node bank into spec_state_indices_tensor[:, :num_spec+1]
(gdn_attn.py:266-268); 'none'/no-cache keeps a FIXED leading slot. So on turn-1 COLD decode the tree's
co-resident GDN node-bank slots COLLIDE with align's rotating gather position — a layout mismatch, zero hit
involved. (Consistent: §29 collapse persists EAGER 3/16 = real not graph-baked; §25/§35 bit-exact through
step 7 then step-8+ distributional shift = the slot layout diverges as decode proceeds.)

THE FIX (zero TTFT cost, feasible): PIN the tree's GDN DECODE to a fixed contiguous 'none'-style slot for
spec/tree requests (override the align branch of mamba_get_block_table_tensor, utils.py:873-891, to a stable
node-bank slot instead of re-gathering at (seq_len-1)//block_size), leaving align's checkpoint gather (KV +
GDN restore) UNTOUCHED. Removes exactly the turn-1 layout difference; keeps both caches; no recompute.

DECISIVE EXPERIMENT: route probe Arm C = cat8+cache + slot-pin patch, N=16 paired seeds. Arm C >=~14/16
delegation (~no-cache 16/16) => layout IS the carrier, SHIP (zero TTFT); Arm C ~4/16 => not layout. Cheapest
pre-GPU gate: in-process same-boot torch.equal GDN decode steps 0-15 Arm C vs no-cache (bit-exact past step 8
CONFIRMS; step-8 fork inconclusive per §29 autotune floor). Full: research/fr13_workflows/FR13_KV_GDN_DECOUPLE_DESIGN.md

## 41. Rate matrix PAUSED (pre-fix baseline); reprefill/refold/recompute to be SHELVED after slot-pin

PAUSED (user 2026-07-05): the rate matrix measures the UNFIXED give-up rate — we re-run on the FIXED
(slot-pin §40) build anyway, and native+cache-healthy is already known (§22, 16/16). PARTIAL PRE-FIX
BASELINE (real task astropy-13453): tnc(no-cache) 1 RESOLVED (56t,agent) | tcv(conv) 2/2 give-up (6t,9t) |
trf(refold-inert) 1 give-up(4t) + 2 engage-no-resolve(39t/45t, ~380B failed-eval). Tree+cache: 0 resolves,
~3/5 non-give-up, all draw-variance (refold inert). Confirms §37 (cache flips to read_file route, never
resolves) as the pre-fix state; the slot-pin Arm C is now the decisive verdict, not this matrix.

SHELVE (user): remove the confirmed-DEAD machinery (git preserves): REFOLD (FR13_APC_BLOCK_REFOLD,
FR13_APC_REFOLD_TO_SNAPSHOT, v1-v4 fold/consume-hop/per-boundary-history — never consumed, §36) + RECOMPUTE
(FR13_SCAN_ALIGN=recompute, FR13_RECOMPUTE_NODE_PARALLEL — retired for give-up, §10) + the §-audit dead
flags (FR13_APC_FIXED_BUFFER no-consumer, FR13_APC_REQUIRE_SHADOW never-passable gate). KEEP: conv
(CONV_LEAF_COMPLETE, working), EXACT_SEED/align (kept under §40 slot-pin), FR13_DECODE_GDN_CAPTURE (the
slot-pin in-process gate instrument), the new slot-pin flag. SEQUENCE: do the removal AFTER the slot-pin
fix applies (single patcher-edit-stream at a time — the slot-pin worktree is mid-flight off main; avoid
same-19k-file diff conflict).

## 42. Slot-pin = NO-OP (proven algebraically); real fix = DEDICATED node-bank (SGLang extra_buffer)
FR13_TREE_GDN_SLOT_PIN (committed e7cbf9f4, default-OFF, correct+verified) is a NUMERICAL NO-OP in the
deployed config: (a) verify proved first-seen pinned window == stock align scratch by algebra (caches
raw[start+1:start+num_spec+1] == spec[1:num_spec+1], first write identity); (b) block_size=1024 + prefix
~15k + fork@~step8 => start=(seqlen-1)//1024=14 CONSTANT thru the fork => align already uses block-14
scratch every step => re-deriving it changes nothing. Arm C == Arm B (still gives up). Cheap gate couldn't
even measure it: FR13_DECODE_GDN_CAPTURE state dump STILL doesn't fire (only logit.* — 2nd H3-class
instrument failure, insertion point off the executed decode path); algebra is decisive, no measurement
needed. §40 "rotating gather" localization directionally right (align IS block-indexed) but rotation does
NOT fire within the fork window (no 1024-boundary crossed at ~15k+8). The align-vs-none diff at the fork is
NOT the index — it's the BUFFER/ALLOCATOR: align puts the tree GDN state in the shared block-pool (stride/
co-residency); 'none' uses a dedicated contiguous buffer. REAL FIX = give tree spec-decode a DEDICATED GDN
state buffer (SGLang extra_buffer / per-draft slots) so decode layout matches 'none'. Allocator/stride
patch, scoping next.

## 43. HONEST RECKONING: node-bank likely ALSO a no-op; §40 layout-localization is a FALSE LEAD; carrier is diffuse/unmeasured
Chain of evidence forces this: §35 proved cache(align) vs nocache(none) GDN DECODE state is BIT-EXACT
through step 7 (torch.equal, 48 layers). cache=align uses the block-pool buffer; nocache=none uses a
dedicated contiguous buffer. So align-buffer and none-buffer produce IDENTICAL GDN state values. Therefore
NO GDN-state layout change — neither the INDEX (slot-pin, §42 no-op) NOR the BUFFER (dedicated node-bank)
— can alter the bit-exact values => a node-bank is ALSO predicted a no-op. => §40/§42's "align GDN slot/
buffer layout" localization is a FALSE LEAD: the GDN state is bit-exact, so it is NOT the carrier.
WHERE THE CARRIER ACTUALLY IS: everything MEASURED through step 7 is bit-exact (GDN state §35, token-1
logits §25). The route flips at step ~30, past the step-8 autotune fork. The one path NEVER measured =
the 16 FULL-ATTENTION layers' KV under align (block storage) vs none (contiguous) — H3 capture NEVER fired
(instrument broken 2x). So the honest next step is NOT another GDN-layout fix (they keep hitting no-ops)
but MEASURE THE UNMEASURED: fix the H3/full-attn capture instrument, compare full-attn KV/output cache-vs-
nocache at turn-1. If it differs => that's the carrier. If it too is bit-exact => the carrier is a genuinely
DIFFUSE distributional effect past the autotune fork (op-unlocalizable, §35) and the fix options collapse to
(A) amplification-reduction (keep diffuse drift below the route-flip margin) or (B) accept the give-up rate.
This is a research-before-deadend gate: measure the full-attn path BEFORE any wall call.

## 44. §43 full-attn measured — bit-exact at seed=1 but seed=1 is a NON-FLIP coincidence seed; decisive flip-seed capture (seed=2) launching
Full-attn capture instrument (H3) FIXED and PROVEN after 3 failure modes: (1) graph-vs-eager — captures
live inside the @support_torch_compile model forward, so they only run in EAGER (never CUDA-graph replay);
requires --enforce-eager (route_probe/famz already do). (2) FR12 layer-pin + prefill-throttle (workflow
wekb306lm fix: per-(prefix,step) windowed, all 16 self_attn layers, ENGAGED breadcrumb). (3) run-script
path bug: famz_run LOG_DIR="$PWD/$cdir" double-prefixed an already-ABSOLUTE $cdir -> container wrote /logs
to an off-tree mount, FAIL-LOUD-5 found 0 (FALSE vacuous). Fixed -> 680 non-trivial .pt/arm, all 16 layers,
steps 2-66, widths {1,9}. Served model confirmed /models/qwen3.6-27b-fp8 (Qwen3_5 VL GDN-hybrid, 64 layers,
full-attn @[3,7,..,63], 48 GDN) — the "80B" label was stale.

RESULT (3 arms, --enforce-eager FB-pinned, SEED=1, cold turn-1): full-attn q/k/attn_out/o_proj cache(align)
vs nocache(none) = BYTE-IDENTICAL (torch.equal, max_abs=0.0, 12/12 output triples, all 16 layers, steps
2-66). FLOOR (nocacheA vs nocacheB) ALSO byte-identical => the engine is byte-REPRODUCIBLE cross-boot in
this eager FB-pinned regime (floor=0), UNLIKE §32's stream-fork. Route todo_write all 3 arms.

RED-TEAM (why this is NOT §43's answer): SEED=1 is one of cache's 3/16 NON-FLIP coincidence seeds (per-seed
table below). At seed=1 cache AND nocache both route todo_write => identical trajectory => byte-identical
state TRIVIALLY (same tokens in => same state out). It CANNOT localize where cache!=nocache because at
seed=1 they don't differ. Per §40 the tree co-resident node-bank divergence is DRAFT-PATTERN- (hence seed-)
dependent: seed=1's drafts don't trigger it; the ~13/16 flip seeds do. GDN corroborator crashed (None.detach,
optional) — but §35 already has GDN bit-exact. Determinism-as-fix REFUTED: route_probe.sh uses the SAME
FB-pin flags as famz, and §29's flip (3/16 vs 12/12) already survives them.

PER-SEED ROUTE TABLE (eager route_probe, FB-pinned): nocache = todo_write 16/16 (rock-stable healthy).
cache = todo_write@{1,12,16} (non-flip), read_file@{2,3,4,5,6,7,10,11,13,14} (FLIP), NO_TOOL@{8,9,15}.

DECISIVE NEXT (launching): capture full-attn at a FLIP seed (SEED=2: cache=read_file, nocache=todo_write),
wide window (FA_LIMIT=200, MAX_TOKENS=512). §35 established the preamble (think) is byte-identical across
arms with the fork AT the route token, so through the identical preamble the FIRST step cache-state diverges
from nocache-state (on identical input, floor=0) = the CLEAN carrier onset §35 couldn't reach (it stopped at
step 7). Sharp onset => localized carrier (draft-pattern-triggered layout divergence, actionable). Gradual
ramp / no divergence until a late route-token fork => DIFFUSE confirmed => §43 (A) amplification-reduction or
(B) accept-rate — escalate the fork with an airtight case. This is the research-before-deadend gate.

## 45. §43 CLOSED — DIFFUSE confirmed, op-localization impossible by construction, ALL fix levers exhausted (WALL)
Two final results close the localization arc:
- FLIP-SEED CAPTURE UNWORKABLE: re-booted SEED=2 (prior route_probe = read_file/FLIP) => todo_write (NON-flip),
  completion_tokens 202 vs seed-1's 182 (genuinely different trajectory). The flip is STOCHASTIC cross-boot
  (confirms §34: cache route unstable, nocache rock-stable), NOT reproducible per-seed. => there is NO clean
  flip trajectory to capture => cache-vs-nocache op-localization is impossible BY CONSTRUCTION (the two arms
  are different boots; the autotune floor forks them before the route token; the flip doesn't pin to a seed).
- DETERMINISM LEVER EXHAUSTED (the last untested (A) amplification-reduction candidate):
  (a) TARGETED M-invariance (LUMO_FB_KERNEL_ROWS=1, authorized #42960 pad-block) is ALREADY ACTIVE in
      §29/route_probe/famz — and the route STILL flips 3/16. So the projection-GEMM M-variance is NOT the cause.
  (b) FULL VLLM_BATCH_INVARIANT is COUNTERPRODUCTIVE on GB10 (launcher:142-143): takes the REDUCED override
      branch, perturbs fp8/scan, cat9+BI=34. Not a clean determinism test and known-bad. (FR13_BI_TREE_ATTN
      Method-A requires full BI => coupled to the bad path.)

NET (all measured, all bit-exact; all levers dead): logits bit-exact (§25), GDN state bit-exact (§35),
full-attn KV bit-exact (§44/famz). Fix levers: slot-pin no-op (§42), node-bank predicted no-op (§43),
decouple structurally impossible (§40), targeted M-invariance active-but-ineffective, full BI counterproductive.
=> the §37 MASTER-SWITCH route flip (tree+cache -> read_file, 0/9 resolves, ~1/3 give-up) has NO localizable
numerical carrier and NO remaining clean fix lever. It is a DIFFUSE stochastic distributional effect of the
tree x cache-config interaction, riding the cross-boot autotune floor that cache is sensitive to and nocache
is robust to (WHY cache is boundary-sensitive is the residual mystery, but it is not op-localizable).

WALL (escalated to user): the charter (tree(cat8)+cache resolving like no-cache) is not reachable via
localization/op-fix/determinism. tree+no-cache resolves (§22, tnc 56t); native+cache is 16/16 healthy
(§22/§23). ONLY the tree x cache COMBINATION collapses. Decision fork for the user: (A) quantify via the
rate matrix (§26) then accept/ship tree+cache-conv-only if the rate is acceptable (§37 indicates 0 resolves,
likely not); (B) ship a proven single-benefit config (tree+no-cache decode-speed OR native-MTP+cache TTFT),
abandoning the combination; (C) a fundamentally different attack on the tree x cache interaction (unscoped).
(Acceptance-pattern angle REFUTED: bit-exact logits => identical tree accept/reject decisions.)

## 46. WALL REJECTED (user 2026-07-05): new attack — the §45 conclusion has four cracks; carrier localization resumes
User directive: fix tree+cache to BEHAVE LIKE tree+no-cache and localize the carrier; cleanup/refold-decision/
speed-matrix only AFTER behavior parity. §45's "stochastic/diffuse/unmeasurable" is rejected as a wall. Re-read
of the evidence finds four cracks, none tested:
1. NEAR-TIE DEDUCTION (untested): §22's own data says the think preamble is byte-identical across all 16 seeds
   WITHIN each arm, fork exactly at the route token. Identical within-boot preamble => identical route-token
   LOGITS across seeds within that boot => the per-seed route split inside the cache boot (§44 table: todo@3,
   read_file@10, NO_TOOL@3 seeds) can ONLY be the seeded sampler drawing at a NEAR-TIE top1-top2 margin, while
   nocache 16/16 stability = decisive margin. That is a SYSTEMATIC, within-boot-measurable margin shift — not a
   stochastic effect. §45's "flip is stochastic cross-boot" is exactly the phenotype a near-tie predicts (autotune
   wobble moves a near-tie across the sampling threshold boot-to-boot); it is NOT evidence of diffuseness.
2. UNEXPLOITED INSTRUMENT: per-token API logprobs/top_logprobs at the route token — zero server-side
   perturbation, on-distribution (no teacher-forcing, §30-compliant), within-boot (immune to the cross-boot
   autotune floor §32), never used in this campaign. §12 promoted a "margin probe" as decisive months of
   sections ago; it was never run.
3. FLOOR CONTRADICTION UNRECONCILED: §32 (nocache cross-boot forks at char ~25) vs §44 (nocacheA-vs-B
   byte-identical, floor=0) — same claimed regime. Whatever differs may yield a byte-reproducible regime =
   direct byte-localization at the route token. §45's "impossible by construction" rests on the §32 floor.
4. RNG-OFFSET ALTERNATIVE UN-AUDITED (playbook class 2): if the tree accept/commit/sample path consumes RNG
   as a function of accept counts (or draws from a global generator), routes can flip with ZERO logit shift.
   §45 dismissed acceptance-pattern via "bit-exact logits" measured only on non-flip trajectories.
IN FLIGHT: CPU workflow wf_20eeac71-6c4 (mine probe streams: verify preamble-identity deduction + between-arm
lcp; reconcile floors + per-arm flag audit incl. HRS; source-audit RNG path + logprobs safety + cache cold-path
deltas; adversarial verify; margin-probe design). NEXT GPU: route-token margin probe (cat8_cache vs cat8_nocache,
N=16 paired seeds + repeats, logprobs top-20) — decision table: systematic-shift => within-boot config bisect by
margin; RNG-offset => patcher generator fix; boot-variance => multi-boot margin distribution. Then carrier-2
(§34 miss-vs-hit 8/16) same-boot margin gate with HRS/ES arms. Pending tasks (shelve refold §41, speed matrix
§20/task#4) gated behind behavior parity per user order.

## 47. §45 OVERTURNED: the "identical preamble" premise was a REDUCER BUG; carrier = REQUEST-ORDER state
## contamination at decode step 1, within-boot measurable (workflow wf_20eeac71-6c4 + first-hand verify)

THREE MINING RESULTS (each re-verified first-hand against raw artifacts before banking):

1. PREMISE REFUTED (reducer bug, class 9+12). The §22/§45 claim "think channel byte-identical across all 16
   seeds in every arm; fork at the route token" is FALSE for cat8_cache. The 2x2 reducer compared
   message.reasoning_content — which is NULL on this server for ALL samples (1 distinct empty string) — instead
   of the populated message.reasoning. Verified: reasoning_content null 16/16; real reasoning = 8 DISTINCT
   strings/16 in cat8_cache (fork at TOKEN 1, char 0), 1 distinct/16 in cat8_nocache. Every downstream §45
   deduction inherited this vacuous measurement.
2. THE SMOKING GUN (seeded2turn logprobs — the probe already had logprobs=True and nobody read them): within
   ONE cat8_cache boot, the token-1 chosen-token logprob on the byte-identical cold request VARIES BY REQUEST:
   send order 1..16 = -0.001, -0.049x4, -0.075, -0.129, -0.910x2, -0.001, -0.477, -1.374, -1.153, -0.317,
   -0.004, -0.008. cat8_nocache: -0.001 FLAT 16/16. Logprobs are raw log_softmax of the LOGITS (pre-temperature,
   RNG-independent; sampler.py:82-83,291-292) => step-1 LOGITS differ across requests on identical cold input
   => PER-REQUEST-HISTORY STATE CONTAMINATION (bug-class 1), systematic, within-boot, NOT stochastic/diffuse.
3. ORDER≡SEED CONFOUND: every prior probe sent seed=i as the i-th request — seed and send-position were NEVER
   separated. Fixed-point signature present: cache seeds 2-7 = byte-identical 71-token read_file completions
   (deterministic given predecessor, class-1's discriminator); seed 1 (the boot's FIRST request) is byte-
   identical to nocache (277-char reasoning, route agent, LOSSLESS). Every first-request-of-boot ever measured
   was clean: route_probe seed1, seeded2turn seed1 (-0.001), famz run (seed1, bit-exact 12/12 triples), famz
   run_s2 (seed2, non-flip), and the §25/§35/§44 instruments (single-request boots) — 5+ boots, ZERO
   counterexamples. => §45's "flip is stochastic cross-boot / not reproducible per-seed" = position/history-
   keying misread as stochasticity (famz s2's seed-2 request was its boot's REQUEST #1 = clean; route_probe's
   seed-2 was REQUEST #2 = post-contamination). All prior bit-exact evidence (§25/§35/§44) measured REQUEST #1
   and is VACUOUS for this carrier.
4. GARBLE CLASS: 6/16 cold cache samples are token-salad runaways (finish=length/stop, digit/multilingual
   salad) — gross corruption on the cold path, consistent with the apc-runaway lossless finding
   (project_fr13_apc_runaway_and_graph_fix); read_file 6/16 = coherent-but-wrong reinterpretation; only 4/16
   healthy. This is CORRUPTION dynamics, not a benign route preference.
5. FLOORS RECONCILED (the §32-vs-§44 contradiction): the "cross-boot autotune floor" is dominated by a
   SUPPRESSIBLE async/timing term. Per-step capture (.cpu() sync each decode step) makes boots byte-
   reproducible: famz (sync window 0..40) floor=0 through step ~40; seeded2turn pass-2 (logit capture LIMIT=8 =
   sync 0..7) bit-exact steps 0-7 with the fork EXACTLY at step 8; route_probe_eager (no capture) forks. Same
   seed, same config, same payload (sha 6b2e9fb7 all instruments) — only the sync-window width differs, and the
   fork tracks its edge. §34's same-boot resend divergence (bytes differ, fixed kernels) confirms the async
   term. => a REPRODUCIBLE LOCALIZATION REGIME exists (eager + per-step sync): request-1-vs-request-N state
   diff on identical input = the op-localization §35/§44 failed at. Caveat: sync perturbs the near-tie sampling
   (famz seed2 todo_write vs unsynced read_file) — use for localization, not behavior gates.
6. RNG PATH AUDITED (class 2): per-request torch.Generator (gpu_model_runner.py:1142) everywhere incl. tree
   committer (FR13_TREE_PER_REQ_GEN hard-ON, patcher:10917-19; device committer re-seeds a fresh dev_gen from
   1 randint/step, fr13_device_multidraft_kernel.py:342-352) — draws per step FIXED, accept-independent.
   Secondary carrier noted (per-request gen advances 1/DECODE-STEP, so cumulative-accept differences shift the
   generator offset at a given OUTPUT position) — but RNG cannot explain the logit-level (logprob) deltas in
   (2). LOGIT instrument: chat logprobs are reliable for token 1 (normal sampler path, read-only, raw);
   spec-committed tokens' top_logprobs are UNRELIABLE (tree topology mismatch, gold_margin_probe.py:363-370).
7. ERRATUM: §44's per-seed route table (todo@{1,12,16}...) is the EAGER probe's table, not the graph probe's
   (graph cat8_cache: agent@{1,10,12,13}, read_file@{2-7}, NO_TOOL@{8,9,11,14,15,16}). Also eager healthy
   route = todo_write vs graph healthy route = agent (both stable 16/16 nocache; both collapse under cache).

NEXT (rp2, GPU, launched): scripts/probes/fr13_rp2_order_probe.sh — the ORDER-vs-SEED discriminator. Fixed
seed=5 x10 sequential cold (reset each) + x6 no-reset (fixed-seed miss-vs-hit) + seeds 1-8 sweep, on cat8_cache;
fixed-seed + sweep control on cat8_nocache. Readout = token-1 chosen logprob + top-20 (zero-perturbation) +
route + per-request prefix-cache metric brackets. PREDICTIONS: P1 sample-1 clean ('Let'~-0.001/agent), later
positions drift => ORDER CONFIRMED (seed exonerated) => localize under the sync regime (request-1 vs request-N
GDN/conv state on identical input, first divergent tensor = the stale-state reader). P1 flat => order refuted
=> P3 spread would implicate a genuine seed->forward leak. Control flat. Probe assets committed under
scripts/probes/ (harness+payload were previously scratchpad-only).

## 48. CARRIER LOCALIZED (rp2 + seam-hunt convergence): recycled mamba block-pool rows are NEVER ZEROED —
## request N's cold prefill/decode reads request N-1's GDN/conv residue; vLLM zeroes only FULL-ATTN blocks

rp2 (scripts/probes/fr13_rp2_order_probe.sh, output/fr13_rp2_order/, 24+10 samples, engagement asserts all
green, per-sample hit-bracket verified):
- P1 (seed=5 FIXED x10, reset before each, hitD=0 all): request #1 CLEAN (agent, 'Let'@-0.0006); #2-#7
  byte-identical read_file fixed point ('The'@-0.0111, ct=71); #8-#10 drift on (NO_TOOL/-0.0045, -0.3527,
  tool_search/-1.3121). Spread 1.31 nats, 4 distinct routes AT ONE SEED => ORDER_DEPENDENT=True. SEED IS
  EXONERATED — the campaign's entire per-seed framing was position/history in disguise.
- P3 (seeds 1-8 on the by-then contaminated pool): mostly garble/NO_TOOL incl. 3-9-token salads; seeds 2 and 7
  = FULL clean recoveries (agent@-0.0006) right after 3-9-token predecessors — outcome tracks the residue left
  by the PREDECESSOR (short predecessor => little residue), not the seed. (The reducer's naive
  SEED_EFFECT_EXTRA=True flag is an artifact of comparing spreads across different pool histories; the s2/s7
  cleans at previously-"bad" seeds directly refute a seed term.)
- P2 (seed=5 x6 NO-reset => full-prefix HIT, hitD=23552 each): stable glob route, tok1 'Let'@-0.038..-0.180.
  The hit RESTORES the state written by the (contaminated) P1_10 prefill => hits FREEZE the writer's
  contamination: consistent-but-wrong. Carrier-2 (§34 miss-vs-hit flips) is this same residue seen through the
  restore path — predicted to collapse under the same fix.
- CONTROL cat8_nocache: spread 0.0000 — tok1 EXACTLY 'Let'@-0.0035 on all 10 (6x fixed-seed + 4-seed sweep),
  all agent. CONTROL_CLEAN=True.

MECHANISM (seam-hunt wf_5fda85e5-3c4: two independent audits converged, then adversarial cross-exam corrected
the read-site attribution; all file:line source-verified):
- vLLM DOES zero newly-allocated blocks — but ONLY full-attention groups: new_block_ids is recorded by
  FullAttentionSpec managers (single_type_kv_cache_manager.py:213-214,241-242) and consumed by
  scheduler.py:913-917 -> gpu_model_runner.py:1087-1088 (_zero_block_ids); needs_kv_cache_zeroing=
  has_mamba_layers (kv_cache_interface.py:608) yet MambaManager.allocate_new_blocks (:934-1010) NEVER records
  its blocks. get_new_blocks pops freed blocks with NO memset (block_pool.py:322-352); reset_prefix_cache
  rebuilds the hash map but never touches contents (block_pool.py:443-476). Boot zeroes everything ONCE
  (gpu_model_runner.py:6498) => request #1 clean, always.
- S1 (owns the TOKEN-1 drift): align chunked-prefill chunk-k>0 running-state carry reads
  ssm_state[non_spec_state_indices] with has_initial_state=True (gdn_linear_attn.py:982-1006 zeroes only
  ~has_initial_state rows) through the align rotating gather (utils.py:874-892) + the num_spec>0-only block
  reshuffle (single_type_kv_cache_manager.py:966-1001) that native never executes => a recycled residual row
  can feed the prefill accumulation => token-1 logits shift. (Exact chunk/row = E4 if ever needed.)
- S2 (owns the tokens-2+ garble): tree spec node-bank/conv prior-window residual reads at num_accepted>1
  (patcher:3069-3076; gdn_linear_attn.py:864-877,957-977) — the long-suspected conv-prior-window carrier
  (project_fr13_conv_priorwindow_root), now explained as pool residue. Spec-decode-gated => cannot touch
  token-1 (cross-exam correction of both audits).
- 2x2 CLOSED: nocache 'none' = fixed request-local rows fully rewritten (utils.py:874 passthrough) => clean;
  native+cache: num_spec=0 => no spec rows, empty reshuffle, col-0 freshly written => clean; full-attn KV IS
  zeroed on alloc => §44's full-attn bit-exactness was a REAL exoneration. §25/§35/§43/§44 all measured
  REQUEST #1 => vacuous for this carrier (why the campaign kept measuring bit-exact).
- REFUTED with verified grounds: _fr13_es_ckpt (cleared on reset + hit-only + P2-healthy-while-P1-drifts),
  all *_BY_REQ maps (random_uuid ids, no collision, freed per request), slot-pin (default-OFF no-op). The old
  "reset_prefix_cache was the artifact" memory is EXPLAINED: reset doesn't corrupt — it frees-not-zeroes,
  exposing residue to the next cold request.

FIX IN FLIGHT (E5, workflow wf_39bf3af0-1a8): flag-gated FR13_APC_ZERO_MAMBA_ON_ALLOC — route MambaManager-
allocated blocks through the SAME zeroing path full-attn already uses (+ engagement needle; default-OFF
byte-identical; adversarial verify incl. hit-path-must-not-zero footgun). Then rp3 = rp2 battery with flag ON:
predict P1 flat agent@~-0.0006 x10, garble gone, P2 hit arm inherits a CLEAN writer. E2 (es0 order arm) held
in reserve — E5's outcome supersedes it if flat.

## 49. E5 fix BUILT + adversarially verified SHIP; launcher -e whitelist trap closed; rp3 causal gate launching

E5 (wf_39bf3af0-1a8, build + independent adversarial verify, both with executed evidence):
- FR13_APC_ZERO_MAMBA_ON_ALLOC: producer hooks in MambaManager.allocate_new_blocks (both branches) record
  ONLY freshly-popped block ids (get_new_blocks results; never null/reused/hit blocks) into new_block_ids;
  they ride the EXISTING drain (take_new_block_ids iterates ALL single-type managers, kv_cache_manager.py:543-548
  -> scheduler.py:913-917 -> _zero_block_ids in _update_states, BEFORE the forward). Consumer adds the
  mamba-specific row zeroer the stock KVBlockZeroer deliberately lacks (worker/utils.py:115-124 "Mamba layers
  are skipped"): zeros conv/ssm dim-0 rows (= block ids; reshape target_shape=(num_blocks,*shape)) for the
  drained ids. Zero == the boot-zeroed request-#1 value == the lossless-proven semantic.
- VERIFY = SHIP: hit-path safe at the allocator level (hit blocks arrive via allocate_new_computed_blocks
  touch path, never hooked; align reused-spec-block branch records nothing; null block never enqueued);
  full-patcher 3-config run: OFF byte-identical (sha equal HEAD-vs-edited), ON compiles + AST-verified;
  anchors 4/4 count==1 with fail-loud collision guards; engagement needles (boot needle per side + throttled
  counters, first fire at event 1); uniprocess (UniProcExecutor TP=1) => no env-bridge risk. Non-blocking
  notes: combined-list cross-zeroing is dead-row-harmless; external-computed-tokens path unhooked (KV
  connector only, not deployed); cached tensor-list refs stable (ES copies contents, never reallocates).
- LAUNCHER TRAP CLOSED (user directive): docker-run env was an explicit -e whitelist — any NEW FR13_* flag
  silently absent in-container (class 9). Fixed with a prefix catch-all (FR<N>_*/LUMO_*/VLLM_* auto-forwarded,
  placed BEFORE the explicit list so explicit ${VAR:-default} entries win on duplicates => zero semantic change
  for existing flags). bash -n + isolated dry-test pass.
- rp3 = rp2 battery, cat8_cache arm, flag ON (output/fr13_rp3_zerofix). PREDICTIONS (banked pre-run, verify
  V5): thesis-right => P1 all agent@~-0.0006 spread~0, no garble; P2 hit arm inherits a CLEAN writer (carrier-2
  gone); P3 clean at ALL seeds; needles: producer record events>0, consumer "registered ~96 mamba state
  tensors" (48 GDN x conv+ssm) + zero calls>0. Partial-fix (S2 residue via an unhooked alloc site) => token-1
  flat but continuation garble persists. Wrong-seam => P1 still spreads WITH full engagement (N>0). Vacuous =>
  rp2 reproduced AND needles absent/N==0. Engagement asserts gate any conclusion (class 9).

## 50. rp3 VERDICT: E5 kills the accumulating carrier (P1 spread 1.31 -> 0.0019 nats, garble ELIMINATED);
## residual = ONE deterministic cold-path fixed point, request-1-vs-rest keyed; hit path now HEALTHY+stable

rp3 = rp2 battery, cat8_cache, FR13_APC_ZERO_MAMBA_ON_ALLOC=1 (output/fr13_rp3_zerofix/). Engagement PROVEN:
producer boot needle + record counters (batches of 9 = num_spec+1 spec blocks, +1/decode-boundary), consumer
"registered 96 mamba state tensors" (=48 GDN x conv+ssm, exact prediction) + zero counters (524 rows by call
101). Results vs rp2 (same battery, fix OFF):
| phase | rp2 (fix OFF) | rp3 (fix ON) |
|---|---|---|
| P1 seed5 x10 cold | spread 1.31 nats, 4 routes, garble x3, progressive drift | spread 0.0019, routes {agent(#1), read_file(#2-10) x9 IDENTICAL}, NO garble |
| P2 seed5 x6 hits | glob x6 (contaminated-stable), tok1 wobble .038-.180 | agent x6 (HEALTHY route), tok1 -0.0149 BYTE-STABLE (spread 0.0) |
| P3 seeds 1-8 cold | mostly garble/NO_TOOL + 2 clean recoveries | read_file@-0.0111 x8 IDENTICAL (spread 0.0, seed-independent) |
- CONFIRMED: the §48 pool-residue carrier is REAL and E5 removes it — accumulation, garble/salad, history
  sensitivity, and seed sensitivity are ALL gone. The hit path (carrier-2 axis) is now on the HEALTHY agent
  route with byte-stable behavior (formal carrier-2 gate = seeded2turn re-run, pending).
- RESIDUAL (the one remaining defect): every cold request AFTER the first lands on ONE deterministic fixed
  point (read_file, tok1 'The'@-0.0111, ct=68) — identical across positions 2-10 AND across seeds 1-8.
  Request-1-vs-rest keying + E5-resistance narrows it to state OUTSIDE the fresh-alloc pool rows: prime
  suspects = the NULL/padding block row (padded spec-tree nodes; never from get_new_blocks => never zeroed
  by E5; request 1's decode pollutes it once via unmasked padded-node writes, every later request reads it
  identically at cold prefill/first-decode => a FIXED POINT exactly as observed) or a persistent ring/buffer
  with a cache-gated consumer. Residual-hunt workflow wf_3dd23e9b-cbc (null-block write audit + ring audit +
  adversarial cross-exam) in flight; E5b candidate = extend the zeroer to the null row per admit (or mask
  padded writes at the writer) -> predict P1 x10 ALL clean.
- Interpretation note: the fixed point is self-sustaining (request N's read_file trajectory rewrites ~the same
  residue request N+1 reads) — which is why pre-fix rp2 showed EVOLVING states (varying trajectories wrote
  varying residue into recycled rows) while post-fix rp3 shows a binary clean/fixed-point.

## 51. Residual carrier cross-exam (wf_3dd23e9b-cbc): D1's null-exoneration REFUTED; top suspect = NULL-ROW
## write/read by the UNGUARDED chunked-prefill SSM carry; D-E4CAP trace = the decisive one-boot discriminator

- KEY CORRECTION (D3, source-proven): the null guards D1 cited (fused_sigmoid_gating.py:114/163,
  fused_recurrent.py:114/163, causal_conv1d.py:137-140) protect the SPEC/DECODE kernels and conv ONLY. The
  PREFILL SSM carry — initial_state = ssm_state[non_spec_state_indices] (gdn_linear_attn.py:984, zeroed only
  for ~has_initial_state rows at :986) and the carry WRITE ssm_state[...] = last_recurrent_state (:1004) —
  is plain UNGUARDED torch indexing. And "cold prefill reads zero state" is FALSE here: ~24.7k prompt /
  max_num_batched=1024 => ~24 chunks, has_initial_state=True at every chunk boundary. The recurrent-carry
  read/write path IS live on every cold request.
- E5-INVARIANCE constraint (first-hand, this session): the residual fixed point 'The'@-0.0111 is IDENTICAL
  pre-E5 (rp2 positions 2-7) and post-E5 (rp3 positions 2-10) => the residual carrier is NOT fresh-alloc pool
  rows; E5 only removed the ACCUMULATING second mechanism on top of it. Two mechanisms confirmed:
  (i) accumulator = recycled fresh rows (E5-fixed), (ii) write-once request-1 residue (open).
- SURVIVORS (all k1-k8 constraints): S1 NULL ROW (block 0) — request #1's unguarded carry WRITE deposits
  state into null at some chunk; null is never re-zeroed (never in get_new_blocks => outside E5; reset only
  rebuilds hashes) => every later cold request's carry READ at that chunk consumes it deterministically. Hit
  path restores the checkpoint and skips the accumulation => P2 healthy (measured). nocache = none-mode
  passthrough, no null padding => flat (measured). S2 = same carry but a stale NON-fresh pool id escaping
  request-2's own alloc set (align start/reshuffle off-by-one). S3 = stale mamba bookkeeping
  (mamba_state_idx/last_state_block_idx surviving reset) — weak (sequential-finish clears it).
- REFUTED: FR13 persistent rings (all baked-on-both-paths, decode-only; nocache flat 0.0 kills them);
  BY_REQ maps (uuid keys); D1's high-confidence null-not-carrier verdict (rested on decode-kernel guards +
  the false cold-reads-zero premise).
- NEXT: D-E4CAP (one boot, read-only, default-OFF FR13_APC_PREFILL_CARRY_TRACE): per prefill chunk log
  {chunk_idx, has_initial_state, non_spec_state_indices ids (flag id==0), as-read initial_state fingerprint
  per layer} to a side jsonl; run >=3 identical cold requests (reset between); diff request-1-vs-2 per chunk.
  Verdicts: first-divergence at id 0 => S1 (fix = mask the :1004 write for index==0, mirroring the fused-
  kernel guard, + optionally zero-null knob); at a non-fresh non-zero id => S2 (fix = correct the carry
  index); identical fingerprints + no null => recurrent exonerated => hunt non-recurrent (full-attn null row /
  positional / sampler buffers). Then fix + rp4 P1x10 all-clean gate.

## 52. E4CAP TRACE VERDICT: the residual carrier = the CHUNK-BOUNDARY STATE COPY silently carrying ZEROS for
## every request after the first — request 1's carry works; request 2+ prefill runs stateless per chunk

E4CAP boot (output/fr13_e4cap_trace/, tracer+E5 ON, P1 x10; engagement: boot needles both procs, first-record
print, 20000 records/48 layers): layer-14 (and by construction all layers) per-chunk as-read/as-written carry:
- REQUEST 1: healthy. chunk k writes state (fp ~581-1046) to its gathered block; chunk k+1 READS THE SAME VALUE
  at the NEXT block id => the inter-chunk copy (mamba_utils preprocess_mamba: prev_state_idx->curr_state_idx,
  collect_mamba_copy_meta + do_mamba_copy_block) is moving the carry forward correctly (e.g. write 818.771@id19
  -> read 818.771@id20).
- REQUEST 2..N: chunk-0 compute LOSSLESS (writes the identical 818.771 — byte-identical input, zero init), but
  EVERY continuation chunk reads 0.0 with has_initial_state=True: the copy contributes ZEROS at every boundary
  (writes stay healthy: 917.78@144, ... => the chain computes chunk-local state from a zero carry each time).
  25 chunks/request, all zero-carry. P1_2..P1_10 traces identical => the deterministic fixed point.
- REDUCE-TRAP CAUGHT (class 12): fr13_e4cap_reduce's naive verdict "S1_NULL_ROW" compared the WARMUP segment
  (which legitimately self-uses null-table [0]) vs request 1. Real pair (req1 vs req2) shows the zero-carry.
- POOL RECYCLING EXONERATED for this phenotype: block ids ascend virgin all boot (19->143->267->391..., stride
  124/request; the pool never recycled in 10 requests) yet request 2+ still flips => the residual carrier was
  NEVER pool-content-keyed. UNIFICATION: pre-E5, a zero... a MISSING/WRONG-SOURCE copy read whatever the
  destination/source rows held = recycled RESIDUE (varying => §48's accumulating corruption); post-E5/virgin
  rows it reads ZEROS (deterministic fixed point). ONE mechanism spans both eras; E5 remains correct hygiene
  (fresh rows must be zeroed) and is what made the defect deterministic enough to trace.
- WHY req-1-vs-rest: the copy is driven by preprocess_mamba (mamba_utils.py:150-220, mamba_state_idx keyed by
  req uuid — clean) but the PATCHER rewrites the copy internals: accept-token-bias override
  (_fr10_tree_accept_token_bias, patcher:13543 — returns linear bias 0 for clean prefills, no crash observed
  => not the carrier) AND the replaced collect_mamba_copy_meta with the EXACT_SEED/SNAP_FIX src_ptr REDIRECT
  (GAP-2 comment patcher:13671-97: swap src on published-checkpoint pos match, else recurrent-leaf fallback).
  PRIME SUSPECT: request-2+ prefill boundary copies take a redirect/fallback whose src resolves to an
  E5-zeroed spec column (or otherwise-zero row), while request 1 (no published/first-seen map state) copies
  natively. Focused source workflow launched to pin the exact branch + minimal fix.

## 53. RESIDUAL CARRIER PINNED + FIX SHIPPED (wf_63d39450-55e): stale batch-position accepted-tree globals
## poison the prefill carry's copy SOURCE; freshness-gated fix FR13_APC_COPY_SRC_FIX (default ON)

PIN (F1, all links source-verified; F2 independently re-confirmed A/B/C):
- The chunk-boundary carry copy is STOCK; its accept_token_bias is POISONED for request 2+. Chain:
  _LUMO_FA_LAST_ACCEPTED_TREE_LENS/_NODE_PATHS[i] are BATCH-POSITION-keyed module globals written ONLY at a
  tree spec-decode COMMIT (patcher:10325-33/11213-19) and NEVER cleared on finish/admit. On request 2+'s
  chunk-0 POSTPROCESS, _fr10_tree_record_request_accept (patcher:13531-40) stamps that STALE tree (elements =
  node_id+1 >= 1, patcher:10234-36) under the NEW req_id. The next chunk-boundary PREPROCESS translates linear
  bias 0 -> tree_bias=path[0]>=1 (patcher:13543-13626; no crash: a path EXISTS) and get_temporal_copy_spec
  (mamba_utils.py:421) computes src = block_ids[prev_state_idx + bias] = a FUTURE, never-written, E5-zeroed
  block => whole-row ZERO carry, every boundary. Request 1 dodges (globals empty at boot => bias 0 => stock
  src = the just-written running state). Self-sustaining across requests. Class 1 (the exact hazard the
  tree-VERIFY path already guards via FR13_TREE_REQKEY; the mamba-copy path was left on raw globals). Pre-E5
  the shifted src read recycled RESIDUE (=§48's accumulator); post-E5 zeros (deterministic) — one defect,
  both eras. E5 stays (correct hygiene; NOT the fix).
- num_accepted_tokens_cpu admit-reset EXONERATED stock (gpu_input_batch add_request resets to 1 => linear 0).
FIX (F2, SHIP; adversarial verify PASS on all axes incl. the decode-boundary bar): freshness-gate the stamp —
thread num_draft_tokens=len(scheduled_spec_decode_tokens[req_id]) into _fr10_tree_record_request_accept; when
FR13_APC_COPY_SRC_FIX=1 (runtime default ON) AND num_draft_tokens<=0 (no spec decode this step = cold prefill
= no fresh tree) SUPPRESS the stale stamp (pop) instead of stamping. Decode steps (ndt>0 = fresh commit) are
byte-identical — translation preserved exactly where needed ("needed IFF the carried step accepted a
multi-token tree IFF ndt>0"). Kill-switch =0 restores old behavior for A/B. Needle: one-shot
"[FR13_APC_COPY_SRC_FIX] engaged..." + counter _FR13_COPY_SRC_FIX_N (engagement gate >0 on any multi-request
cold boot). Hit/restore path untouched (P2 stays healthy); request-1 unchanged (ON==OFF proven); native
short-circuits; no tensor ops; eager-only region. Self-test: all 6 mamba_utils patch fns regen + anchors
matched + py_compile; call-site name verified in scope (patched:814/830).
rp5 GATE (launching): full battery fix-ON+E5-ON. PREDICT P1 x10 ALL agent (tok1 spread ~0 vs request-1),
fixed point GONE; P2 agent byte-stable; P3 all clean; _FR13_COPY_SRC_FIX_N>0. Then carrier-2 seeded2turn gate,
native regression arm, live SWE gate -> bake E5+COPY_SRC_FIX default-ON (user directive) -> cleanup -> speed.

## 54. rp5 GATE PASS 24/24: carrier-1 CLOSED at probe level — tree+cache cold route == tree+no-cache;
## fixed-seed miss-vs-hit flip GONE; ORDER_DEPENDENT=False

rp5 (output/fr13_rp5_copyfix/, FR13_APC_COPY_SRC_FIX=1 default + FR13_APC_ZERO_MAMBA_ON_ALLOC=1, engagement:
"[FR13_APC_COPY_SRC_FIX] engaged ... stale_len=1 stale0=1" fired at request 2 exactly as pinned + in-container
patched source verified):
| phase | result |
|---|---|
| P1 seed5 x10 cold | agent x10, ct=264 IDENTICAL, tok1 'Let'@-0.0033 IDENTICAL — spread 0.0 (rp3: 1 agent + 9x read_file fixed point; rp2: 4 routes, spread 1.31) |
| P2 seed5 x6 hits | agent x6 byte-stable @-0.0037 (was glob x6 pre-fix) — miss-vs-hit ROUTE FLIP GONE at fixed seed |
| P3 seeds 1-8 cold | agent x8, tok1 identical, healthy ct variation 254-361 (pre-fix: garble/NO_TOOL majority) |
Reducer verdicts: ORDER_DEPENDENT=False, SEED_EFFECT_EXTRA=False, all spreads 0.0000 — the cat8+cache arm now
carries the EXACT signature of the cat8+nocache control (agent x all, flat tok1, entropy only in downstream
lengths). User bar "choose agent instead of read_file" MET at probe level.
REMAINING GATES before bake (user directive: bake E5+COPY_SRC_FIX default-ON after verification): (1) formal
carrier-2 seeded2turn re-run, 16 seeds miss-vs-hit, predict 0/16 flips (was 8/16 §34); (2) native+cache
regression arm (predict unchanged 15-16/16 healthy); (3) LIVE SWE-Verified nudge-free gate (task #5) —
tree+cache must engage/resolve like tree+no-cache on astropy-13453 class tasks. Then: bake, DELETE dead
patches (user list), speed matrix.

## 55. Carrier-2 formal gate (eager seeded2turn, fixes ON): corruption GONE, hit-side DETERMINISTIC; residual =
## near-tie route split miss-vs-hit (7/16, one-directional) — deployment regime already clean; live gate decides

output/fr13_s2t_gate/ (arm B cat8_cache only, N=16, resend mode, EAGER 0.78; engagement: both fix needles,
turn2 hits 16/16, turn1 hits 0, es_seed_applied=1440, snapshot_events=0 — completions ~130t never cross the
decode boundary):
- turn-2 (hit): delegate agent 16/16, stable ~131c contents — the RESTORED trajectory is deterministic-healthy.
  (§34 pre-fix: hit flips scattered 8/16 on corrupted turn-1s.)
- turn-1 (cold): SPLITS agent 9 / todo_write 7 — a genuine near-tie at temp 0.6 in this eager boot (both
  routes healthy; eager's historic healthy route was todo_write, graph's is agent).
- Miss-vs-hit route flips 7/16, ALL todo->agent (no agent-turn1 ever flips). Byte-level: 0/16 identical,
  first-div median token 6 — SAME signature as §34's nocache resend control (bytes differ 16/16, same-boot
  async floor) => byte-forks are the KNOWN floor, NOT restore evidence. The honest deficit vs the nocache
  control = route-flip rate 7/16 vs 0/16, structure = near-tie cold margin vs decisive hit margin.
- CLASSIFICATION: no corruption anywhere (transformed from §34); residual = REALIZATION-EPSILON-scale margin
  difference between the live chunk-carry realization (cold) and the restore realization (hit) surfacing at a
  near-tie — the §16 epsilon family, NOT our (fixed) carriers. DEPLOYMENT REGIME (graph): rp5 cold agent x24
  spread 0.0 == fresh same-build nocache graph control (agent x16), miss==hit stable — parity HOLDS where it
  ships. The stale eager-nocache reference (todo x12, old build) makes eager-side attribution unresolvable
  without another control boot — not worth GPU ahead of the binding gate.
- DECISION (per feedback_live_swe_verified_only): the LIVE SWE-Verified gate in the deployment regime is the
  binding verdict for task #5/bake. HRS deletion stays CONTINGENT on that gate (if live engagement/resolve
  lags nocache, HRS / EXACT_SEED-decode-half is the ready lever for the epsilon; if live passes, the eager
  epsilon is a documented floor note).

## 56. LIVE SWE GATE: tree+cache (both fixes) RESOLVES astropy-13453 nudge-free — the campaign's missing cell
## FILLED; tnc reference arm wall-censored (NA, re-running unwalled per user rule)

output/fr13_live_gate/ (git 3d94cea6 serving build; qwen-code offloaded to alienware, SWE_EMPTY_PATCH_RETRIES=0,
temp 0.6, real astropy__astropy-13453; engagement needles verified live in the serve container: E5 producer+
consumer + "[FR13_APC_COPY_SRC_FIX] engaged ... stale0=1"):
- **tcfix (cat8 TREE + EXACT_SEED cache + E5 + COPY_SRC_FIX): rc=0 dur=1194s turns=34 first=agent
  patch_bytes=551 verdict=RESOLVED.** Round-1 route = agent (the §37 master switch, healthy for the first
  time under cache: pre-fix 9/9 read_file); 34 engaged turns; real patch; eval resolved; ZERO nudges. This is
  the FIRST tree+cache resolve in campaign history (pre-fix: 0 resolves across all runs, §37/§41) — the cell
  the campaign existed to fix. Hit-heavy by construction (every turn 2+ hits the growing prefix) => the §55
  realization epsilon produced no behavioral damage across ~33 hit-serving turns.
- tnc reference: NA / WALL-CENSORED. I mistakenly carried AGENT_WALL_S=2400 from tcv_s2_redo (user rule: NO
  agent wall on gates — total-time caps right-censor the signal; banked to memory). The wall's docker-kill
  destroyed the buffered trace (traces flush at clean exit — turns=0/first=NONE is a kill artifact, NOT agent
  behavior: the server processed healthy multi-turn traffic for ~40min, POSTs + KV oscillation observed live)
  and aborted before patch extraction. tnc's historic behavior on this build lineage: resolved (56t).
  RE-RUN launching with AGENT_WALL_S unset (timeout_s<=0 => no harness wall, runner-native).
- Gate reading so far: the cell under test PASSED its half decisively (delegate + engage + patch + resolve,
  matching the tnc behavior class). The unwalled tnc re-run completes the like-for-like pair for the record;
  bake (E5+COPY_SRC_FIX default-ON, user directive) proceeds after it lands.

## 57. s1-vs-s2 trace forensics (wf_9363ecce-d74): s2's near-miss patch = healthy draw variance, BUT s2 carries
## a GARBLE signature in the long hit-heavy trajectory — a residual-corruption SIGNAL pointing at the unfixed
## decode-boundary restore family (HRS domain); native n=3 control queued

- PATCH AXIS (draw variance, benign): both runs' REASONING found the identical correct fix (the upstream
  astropy PR pair: 'self.data.cols = cols' + 'self.data._set_col_formats()'). s1 emitted both lines in ONE
  clean edit -> 551B -> RESOLVED. s2 stated the cols line >=4 times but DROPPED it from the emitted edit ->
  377B single-line patch -> failed (the applier iterates self.data.cols which is never set on the HTML path
  => no-op). s2's 377B artifact matches the historic ~377-398B near-miss family INCLUDING native+cache's 398B
  => end-artifact difference alone = solution-luck variance.
- GARBLE AXIS (the signal): s2 contains 3 unambiguous generation-degradation episodes at the sanctioned
  instrument (codex_trace agent messages): edit#62 file_path arg = 22.7KB hallucinated academic text ending in
  a '2.1.2.1...' repetition loop (tool-rejected); edit#70 = 29.9KB fake climate paper + Debian apt text
  (tool-rejected); FINAL thinking block (11KB) opens coherently then derails into the same fake paper and
  terminates in Chinese /etc/profile.d instructions -> empty final answer. Plus the Explore subagent collapsed
  (93-char non-answer vs s1's 3,952-char report). s1: ZERO garbled blocks. Cost asymmetry 2.2x (2602s/49 calls
  vs 1194s/34; incl. 22 doomed dependency-install shell calls — the eval workspace lacks packaging/Python.h/
  pip, so agent-side verification is impossible for EITHER arm; harness gap noted, common across arms).
- CLASSIFICATION (medium confidence, honest caveats: n=1 pair, garble was tool-rejected and did NOT corrupt
  the submitted patch, no same-build control yet): serving-side QUALITY-DEGRADATION SIGNAL in long
  trajectories. MECHANISM CANDIDATE: long turns cross 1024-token DECODE boundaries -> decode-side snapshot
  writes (recurrent/branch realizations; EXACT_SEED's decode half was never wired, §16) -> later HITS restore
  them. The rp probes could never see this (completions <=448t cross no decode boundary; §55's own obs showed
  snapshot_events=0). This is exactly the E' family = HRS's designated domain (user 2026-07-05: HRS preferred,
  speed-first, SGLang-proven).
- ATTRIBUTION PLAN (in motion): tcfix_s4 (running) + native+cache n=3 (nativemtp5_exseed, queued on GPU
  handoff) = the more-seeds + native-control the verdict demands. Garble reproducing in tree arms but not
  native => decode-boundary tree-state restore confirmed as carrier-3 -> HRS=1 arm gate (+ fr13_measure
  hit-recompute tax). Garble equal on native or non-reproducing => downgrade to temp-0.6 sampling-tail.

## 58. Eval-workspace root cause + fix design (wf_d717de65-cfc): agent and grader live in DISJOINT envs;
## fix = run the agent INSIDE the SWE-bench per-instance image (SWE_AGENT_ENV=instance_image, default OFF)

ROOT CAUSE (file:line-verified + image-probed): the agent works on a SOURCE-ONLY git worktree (run_swe_bench
_q36_a.py:358-374) rsynced to alienware and mounted into qwen-code-runner:v1 (node:22-bookworm; probe: NO pip,
NO Python.h, NO conda, import astropy/numpy FAIL) — while the EVAL runs inside official SWE-bench per-instance
images (swe_eval_x86_worker.py:107-136; /opt/miniconda3/envs/testbed with editable astropy). Nothing exposes a
usable env to the agent => self-verification structurally impossible (the AGENTS.md "do NOT pip/build" copy is
a band-aid). SECONDARY: the rsynced .git is a worktree POINTER to a GB10 path => git is broken inside the
agent container (patch extraction immune — runs on GB10, :463-469).
FIX (designed, implementation AFTER the native n=3 attribution series; ON only for the 16-task matrix):
(1) SWE_AGENT_ENV=instance_image mode — run qwen-code INSIDE the instance image editing /testbed (agent gets
the testbed conda env + a real git repo; SWE-agent/OpenHands convention); (2) one-time relocatable node+qwen
bundle on alienware (official nodejs tarball for glibc-compat, qwen-code version PINNED to runner:v1 for
cross-arm fairness) bind-mounted :ro into any instance image; (3) flag-coupled prompt copy update ("you have a
working testbed env: reproduce + pytest before finishing"); (4) REJECTED: baking deps into qwen-code-runner
(non-durable across 500+ instances, drifts from eval). Validation ladder: env smoke (import astropy + qwen
--version in-image) -> one-instance probe (trace shows import astropy rc=0 + pytest collecting) -> patch
parity (git diff inside /testbed) -> then matrix. Task #9.

## 59. Offload stream-stall diagnosis (wf_1b8212ee-707): WAN ruled out; wedge = GB10 emit/serialize (or proxy
## read) with generation CONTINUING; fix stack designed (instrument-first, watchdog, model-invisible heartbeat);
## auto-retry REJECTED as nudge-unsafe; possible unification with the garble family via a parser wedge

- PATH: qwen-code (alienware docker, --network=host) -> inference_proxy :8023 (SSE relay) -> tailscale ->
  GB10 vLLM :9950. The 120s detector = qwen-code's BUILT-IN stream-idle default (unconfigured; codex's 600s
  applies only to /v1/responses). Proxy has NO heartbeat + 1200s upstream timeout => client 120s always trips
  first.
- FORENSICS: WAN RULED OUT (offload_link_state.log: link up every 10s across BOTH incidents). Engine
  generation CONTINUED through the silence (Running:1, 11-14 tok/s, ~1080 tokens produced for the stalled req
  that never reached the client). Wedge = engine emit/serialize vs proxy iter_content — UNRESOLVABLE from
  artifacts because the proxy capture is gated to /v1/responses (chat path = ZERO rows; class-9 vacuous
  instrument). Stall correlates with the LARGEST request of the run (167KB, pos=35840 restore), not with time.
- UNIFICATION HYPOTHESIS (flagged for the new instrument): generated-but-undelivered is ALSO the signature of
  a GARBLED/runaway generation wedging the qwen3xml tool-parser (unterminated tool-call buffers => no deltas
  emitted) — i.e. the s3 stall may be carrier-3's garble surfacing through the parser on the deepest-context
  request. The chat-path capture (below) records upstream bytes => next occurrence is attributable.
- FIX STACK (designed; implement with the workspace fix AFTER the native n=3 series; all env-gated):
  (1) INSTRUMENT-FIRST: chat-path per-chunk timestamped capture + terminal-reason logging in
  _write_chunked_stream (closes the blind spot). (2) RUN-LEVEL STALL WATCHDOG keyed on TRACE GROWTH (not
  wallclock, not SSE): no growth for LUMO_SWE_STALL_KILL_S => kill + classify infra_stall_suspect. Replaces
  walls; covers the pre-first-byte mode (tcv_s2's 37-min hang). (3) EMPTY-DELTA SSE HEARTBEAT, chat-path only:
  ': ping' comments are PROVEN stripped by the qwen SDK; an empty chat.completion.chunk resets the idle timer
  and is dropped pre-model (model-invisible; 0 injected on clean runs => byte-identical). (4) Belt: raise
  QWEN_STREAM_IDLE_TIMEOUT_MS to 240s (documented knob). (5) NO AUTO-RETRY (nudge analysis): engine-generating
  -while-client-silent is the SAME signature for a transport wedge AND a model runaway (the known cache-ON
  class) — auto-replay would selectively erase model failures = a nudge. Stalls => NA + classification only.

## 60. HRS preload (wf_4479679c-4f7) + uniform garble scan: HRS is DEAD CODE under today's lattice while its
## designated domain is served by the exact path it was built to replace; garble <=> decode-boundary crossing
## (post-fix biconditional); s4 garble correction; per-row hybrid re-arm designed

- UNIFORM SCAN (fr13_garble_scan.py, committed 08e4de92) CORRECTS my earlier weak scan (class-12: >8KB-only
  missed sub-8KB episodes): tcfix_s4 has 2 GARBLE episodes (pipe-repetition + 479-CJK run_shell arg; Angular/
  webpack GitHub-issue hallucination) + empty final. POST-FIX TABLE: s1 CLEAN/resolved (0 boundary-crossing
  turns, max 487 gen tok) | s2 GARBLE x3/failed (3 crossings: 7415/7018/2280 tok turns) | s3 CLEAN/NA-stall |
  s4 GARBLE x2/failed (1 crossing: 1496) | nat_s1 CLEAN/failed-nearmiss (0 crossings, max 467).
  **Post-fix biconditional: garble <=> >=1 turn crossing a 1024-token decode boundary (5/5 runs).** Native is
  UNDERPOWERED as a control (its turns never exceed ~500 tok => never arms the mechanism) => the forced-
  boundary probe is the binding discriminator, arms {tree+cache, tree+NOCACHE, native+cache} x max_tokens~1600.
  Historic-scan nuance: PRE-fix-era traces show garble WITHOUT crossings (m_tree_cache_base_r2, tcv_s1,
  m_cat8on_obs) = the OLD carriers (pool residue / copy-src poison), consistent with both being real and fixed.
  Causality caveat (H2): for a FIRST episode, crossing->garble vs garble->runaway-length->crossing not yet
  separated; restore-propagation for later episodes well-supported (corrupted crossed blocks re-hit every turn).
- HRS STATE (H1, file:line): fully implemented (patcher :5978-6174, recurrent roll-forward from restored
  boundary state, CAP=64 default) but DEAD CODE twice over: launcher HRS:=0 AND a WHOLE-BATCH veto at :6002
  (`not _fr13_es_on` — EXACT_SEED=1 unconditionally disables HRS even if =1). Live tree+cache arms force
  ES=1 => HRS never engages. WHAT RUNS INSTEAD at decode-crossed hit rows: EXACT_SEED's chunk-resume from the
  ES_REDIRECT_FALLBACK RECURRENT-lineage state (decode-side chunked producer unwired, §14/§16) = the
  restart-fold realization-mismatch family — THE §57 GARBLE CANDIDATE. HRS's designated domain is currently
  handled by the exact path HRS was written to replace.
- RE-ARM DESIGN (H1): per-row hybrid, flag FR13_APC_HRS_ON_DECODE_CROSSED (default 0 = byte-identical):
  relax the :6002 veto per-row; ONLY decode-crossed hit rows (discriminator: per-req snapshot_events>0 /
  chunked-ptr absence) take HRS roll-forward with CAP covering the remainder from the last CLEAN aligned
  boundary (<=1023) => lineage self-consistent, restart-fold mismatch eliminated; prefill-aligned rows keep
  EXACT_SEED's proven bit-exact chunked restore untouched. HONEST: within-floor fix (recurrent epsilon =
  the class native itself ships), NOT bit-exact; the bit-exact alternative (wire the chunked producer,
  §16 task#7) stays de-prioritized per user. SGLang citation woven per user directive (MambaRadixCache:
  states stored ONLY at stage boundaries, all-or-nothing prefix reuse, recompute-forward-from-checkpoint).
  Speed: tax lands on hit-time only (decode untouched, HBM-bound); ~1 crossed-hit/turn worst case; quantify
  via fr13_measure hit-time before bake.
- BENCHMARK NOTE (user): official SWE-bench Verified runs agents INSIDE the per-instance image (full runtime
  deps) — our agent-in-bare-worktree deviates from the published convention; §58's instance_image mode is the
  return to standard. Resolve rates before that fix are not externally comparable.

## 61. Native control series COMPLETE: resolve PARITY (1/3 vs 1/3); native never arms the boundary mechanism
## (0 crossings in 3 runs) => forced-boundary probe = the attribution instrument (launching)

| arm | run | turns | patch | verdict | garble (uniform scan) | boundary-crossing turns |
|---|---|---|---|---|---|---|
| tree+cache | s1 | 34 | 551B | RESOLVED | clean | 0 |
| tree+cache | s2 | 49 | 377B | failed | x3 | 3 (7415/7018/2280 tok) |
| tree+cache | s4 | 28 | 477B | failed | x2 | 1 (1496 tok) |
| native+cache | s1 | 34 | 377B | failed | clean | 0 (max 467) |
| native+cache | s2 | 67 | 429B | RESOLVED | clean | 0 (max 319) |
| native+cache | s3 | 42 | 377B | failed | clean | 0 (max 297) |
- RESOLVE AXIS: parity 1/3 vs 1/3 (identical near-miss patch classes both arms; blind-emission luck under the
  broken agent env §58 — rates jump for ALL arms once the official-benchmark env lands). Round-1 route =
  agent 6/6. Give-up class: extinct (0/6).
- GARBLE AXIS: biconditional garble<=>crossing intact 6/6; native runs NEVER produce >1024-token turns on this
  task (natural style) => native cleanliness is real but UNARMED — cannot discharge attribution.
- NEXT (GPU, launching): forced-boundary probe — real payload, min_tokens forcing ~1600-token generations
  (mechanism-attribution probe, NOT a behavior gate; live-SWE remains the gate class), 3 turns/seed
  (t1 cold long-gen crosses boundary => decode-side snapshot writes [engagement: snapshot_events>0, the cell
  no probe has ever exercised]; t2/t3 resend+extend => hits restore the crossed blocks), N=6 seeds/arm, arms
  {cat8_cache(fixes ON), cat8_nocache(CONFIG_ONLY — no boundary machinery: splits long-gen-itself vs
  align-boundary machinery), native_exseed(+cache, forced long: tree-specificity)}. Readout = garble scanner
  per turn + snapshot/hit OBS brackets. PREDICTIONS: restart-fold-mismatch story => tree+cache garbles
  (t2/t3 amplified), tree+nocache CLEAN (no cache machinery), native+cache clean-or-epsilon; long-gen-itself
  story => tree+nocache garbles too; sampling-tail => all arms equal.

## 62. Forced-boundary probe: ENGAGEMENT-PROVEN NULL — the restore-mismatch mechanism ran hard (snapshot_events
## =864, live ES_REDIRECT_FALLBACKs) and 18/18 forced-long turns stayed CLEAN; pivot to LIVE-RUN garble hunt
## (user directive)

- fb probe cat8_cache arm (output/fr13_fb_probe/, fixes ON, N=6 seeds x3 turns, min_tokens=1200 forcing >=1
  decode-boundary crossing per turn): ALL 18 turns clean (0 CJK, 0 repetition, 0 offtask, healthy finishes).
  NOT vacuous: eng log shows snapshot_events climbing to 864 (decode-boundary snapshot writes fired throughout)
  + ES_REDIRECT_FALLBACK lines live at the tail (the §60 recurrent-lineage fallback consumed on hits). The
  §60 restart-fold-mismatch mechanism EXECUTED at scale and produced no garble in forced-prose continuation.
  (My probe's docker-log needle grep was wrong — counters live in fr13_apc_exact_seed_eng.log; corrected.)
  Remaining arms killed as no-longer-decision-relevant (they only mattered had the cache arm garbled).
- RE-RANKING: the pure decode-crossed restore epsilon is exonerated as the garble TRIGGER (it remains the
  §55 route-margin epsilon; HRS stays the designated lever for THAT, not for garble). Live-garble ingredients
  the probe lacked: (a) TOOL-ARGUMENT/constrained-decoding contexts (every live episode sat inside a tool-arg
  string or terminal thinking — the qwen3xml parser regime; ties to §59's parser-wedge stall unification);
  (b) natural runaway onset + restore-propagation lock-in (H2's causality caveat).
- PIVOT (user 2026-07-06): live runs, not prompt probes. Plan: (1) more live tcfix seeds NOW under the exact
  s2/s4 conditions (legacy env, unwalled, scanner post-hoc) to grow the garble sample; (2) once the §59
  chat-path capture lands+deploys, subsequent live runs record upstream bytes => the next garble episode is
  fully attributable (engine-emitted garbage vs transport artifact) and the exact garbling request is
  replayable; (3) exact-context replay of a captured garbling turn = the precision instrument if needed.

## 63. First instrumented live run: the WEDGE IS MEASURED (transient 38-91s self-recovering emit pauses,
## bridged by the heartbeat) — but my trace-growth watchdog FALSE-KILLED the healthy run (signal wrong for
## qwen's flush-at-exit trace); watchdog disabled pending metrics-based redesign

- tcfix_i5 (instrumented, §59 stack ON): 25 requests captured with per-chunk timing. NORMAL cadence ~0.5s
  inter-chunk. Request #6: 46 chunks -> 91.3s MID-STREAM SILENCE -> resumed -> completed (upstream_done);
  6 heartbeats bridged it. Request #23: same at 38.2s. => the §59 stall class = TRANSIENT SELF-RECOVERING
  ENGINE-SIDE EMIT PAUSES, request-correlated, with a heavy tail (s3's fatal was the tail crossing the old
  120s client timer). Heartbeat + idle-raise WORK: pre-fix, these gaps would have burned turn(s).
- FALSE KILL (mine): runner watchdog keyed on trace-file growth — but qwen-code's trace FLUSHES AT EXIT ONLY
  (0 bytes all run, re-confirmed) => "no growth for 600s" fired on a HEALTHY 25-request run at 605.9s and
  docker-killed the client mid-request (#24 broken_pipe, empty trace, turns=0). J2's no-false-kill test used
  a growing fake trace = class-8 offline!=live. Artifacts preserved: output/fr13_live_gate/tcfix_i5_watchdogkill.
- ACTIONS: batch relaunched with LUMO_SWE_STALL_KILL_S=0 (watchdog OFF; heartbeat+240s idle remain the stall
  protection — they demonstrably bridge the wedge). WATCHDOG REDESIGN (follow-up, before re-enable): key on
  GB10 /metrics vllm generation_tokens_total growth (moves every decode step of any live request; local poll,
  no ssh; detects both pre-first-byte and permanent-wedge modes without trusting client write patterns).
- WEDGE HUNT NEXT: the capture's chunk-timestamp arrays localize the pause onset within the request; correlate
  wedged requests (#6/#23 class) with engine-side events (boundary snapshots? scheduler states?) once 2-3 more
  instrumented specimens land.

## 64. INSTRUMENTED BATCH COMPLETE (i5-relaunch/i6/i7): §59 stack VALIDATED IN PRODUCTION — i6 RESOLVED
## through a 103s wedge; 0 content-garble this round (no long-turn draw); scanner false-positive fixed
| run | turns | patch | verdict | garble (fixed scanner) | max emit-gap | heartbeats | boundary crossings |
|---|---|---|---|---|---|---|---|
| tcfix_i5 (relaunch) | 21 | 0 | failed | CLEAN (empty-final only) | 8.7s | 0 | 0 |
| tcfix_i6 | 23 | 408 | **RESOLVED** | CLEAN | 103.4s | 6 | 0 |
| tcfix_i7 | 27 | 398 | failed | CLEAN | 26.1s | 1 | 0 |
- §59 STACK VALIDATED: i6 RESOLVED while surviving a 103.4s mid-stream emit wedge (6 heartbeats bridged it) —
  pre-fix that gap crosses the 120s client timer => fatal patch-less give-up (the s3 class). The fix converts
  a fatal stall into a resolved task. Task #8 = validated-in-production (capture+heartbeat+idle-raise). The
  watchdog stays OFF (§63 false-kill; metrics-based redesign pending, NOT needed — heartbeat handles the
  transient class; watchdog is only for a PERMANENT wedge, which hasn't recurred).
- WEDGE CHARACTERIZED (n=3 instrumented runs, per-chunk capture): TRANSIENT SELF-RECOVERING engine-side emit
  pauses, request-correlated, heavy-tailed (0.5s normal; tail 26/38/91/103s). All self-recovered
  (upstream_done). NOT correlated with garble (i6 wedged 103s + CLEAN + resolved).
- GARBLE: 0 content-garble across the instrumented batch — BUT no run drew a >1024-token turn (max 332-397t),
  so the decode-boundary mechanism was never armed => consistent with the surviving "real garble only on
  crossing turns" picture (2/7 earlier runs), NOT evidence against it. SCANNER FIX (0242224e): empty_final_
  answer alone is BENIGN (subtype=success terminal), no longer sets GARBLE; i5-relaunch corrected CLEAN;
  s2/s4 stay GARBLE. So the real garble tally is 2/9 valid tree+cache runs, both crossing turns; every
  crossing-free run (tree AND native) content-clean.
- RESOLVE TALLY: tree+cache 2/6 (s1 551B, i6 408B), native+cache 1/3 — parity within small-n; give-up class
  extinct 0/9, round-1 agent/todo (healthy) 9/9. The behavior charter (tree+cache behaves like tree+no-cache)
  HOLDS on live SWE across the fixed stack.
- NEXT: catch an ATTRIBUTABLE garble = replay a known s2/s4 garbling request (captured chatreq_*.json) against
  a populated cache (deterministic, now fully instrumented) OR more seeds until a long-turn draw. Then the
  bake decision (E5+COPY_SRC_FIX now backed by resolves through real load + wedge survival).
