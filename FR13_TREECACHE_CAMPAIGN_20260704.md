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
