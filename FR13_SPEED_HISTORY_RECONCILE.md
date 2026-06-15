# FR13 SPEED HISTORY RECONCILE — lm-head / FIX-1/2/3 lineage

Scope: the lm-head / FIX-1/2/3 lineage = the bulk of the original cat9-vs-native B=1
speed gap. Reconciles every stale speed number against the full git history + current
HEAD code. Basis throughout = `decode_seconds/spec_drafts` raw counter, greedy, B=1,
BI=0 pinned both arms, metrics OFF, per-request (class 12). MEASURED vs INFERRED
labeled per claim. Native E5 ref = 0.2182 s/fwd.

## s/fwd progression (cat9, exact, MEASURED unless noted)

| stage | commit | cat9 s/fwd | ratio vs native | chain5 s/fwd | ratio | source |
|---|---|---|---|---|---|---|
| pre-FIX-1 (drafter double lm-head) | 008631cd / dd45c3c1 | 0.3044 (chain5; cat9 not separately captured) | 1.40x (chain5 1.4025x; metrics 1.39x) | 0.3034 | 1.39x | FR13_B1_SPEED_ATTRIBUTION_BIND, profile sqlite re-mine |
| post-FIX-1 (single-logits) | d407e545 / 93a4043a | 0.2373 | 1.088x | 0.2294 | 1.051x | FR13_B1_FIX1_GATE_BIND (gate pinned-probe OFF 0.3118 -> ON 0.2373; SWE 11k OFF 0.3216 -> ON 0.2458) |
| post-FIX-2 (eager-pack) | 6c2f46d6 / 7fe500b5 | 0.2347-0.2349 | 1.076x | 0.2254-0.2265 | **1.033x (best tree)** | FR13_B1_FIX2_GATE_BIND gate(e) |
| post-FIX-3 (conv-fusion) | 1f5f37f0 / ef4d7514 | **0.2247-0.2249** | **1.030x** | **0.2223-0.2247** | **1.019-1.030x** | FR13_B1_FIX3_GATE_BIND gate(e) |
| + in_proj_ba baked (current) | 4d0452df | 0.2248 (vs 0.2249 OFF) | SPEED-NEUTRAL | — | — | 4d0452df commit msg (pad hidden behind bandwidth-bound weight read) |

## What each fix REMOVED (MEASURED)

- **FIX-1 single-logits (the ~96% lever).** Root: the caterpillar drafter computed the
  full-vocab bf16 lm-head logits TWICE per drafter step — once for top-2 packing, once
  inside live `eagle.py:385-389`'s `_greedy_sample` (which RECOMPUTEs `compute_logits`).
  MEASURED tax: +5.45 cuBLAS `internal::gemvx` bf16 calls/draft x 15.05 ms/call =
  **+81.94 ms/draft of the +92.25 ms/draft GPU-busy delta** (chain 356.28 vs native
  264.03 ms/draft). Each call re-reads the lm-head weight: vocab 248,320 x 5120 bf16 =
  **2.543 GB** at ~169 GB/s = 62% of GB10's 273 GB/s. Fix = reuse the already-computed
  logits tensor (`draft_token_ids = _fr10_step_logits.argmax(dim=-1)`, drafter-only,
  ~6-10 lines). MEASURED engagement: gemvx 11.1 -> 5.9/draft. chain5 1.41x -> 1.05x.
  Currently BAKED ON in HEAD (`fr10_phase4_patch_vllm_tree_gdn.py:11070` "single_logits
  baked ON"; argmax reuse :11476). Lossless proven IN-PROCESS (FR13_FIX1_SELFCHECK dual-
  path: 6235 drafter-step compares both topologies, greedy+t0.6, ZERO mismatches).

- **FIX-2 eager-pack (FR13_EAGER_PACK).** Removed the eager-op storm on the engine
  thread: packed committer DtoH (102 -> 1), batched all-layer GDN replay launch (96 ->
  2), pinned HtoD staging, runner metadata cache. The committer's 6x blocking
  `.cpu().tolist()` (`:4088-4098`) was MEASURED as the dominant WAIT LOCATION (chain5
  main thread blocked 91.7-91.9% inside synchronous DtoH `cudaMemcpyAsync`; engagement
  DtoH 109.6 -> 8.0/draft). cat9 1.088x -> 1.076x; chain5 1.051x -> **1.033x**. Required
  a byte-A/B kernel fix (anchor+offset, Triton 3.6.0 AxisInfo divisibility) + a builder
  group-union/re-init fix (Qwen3-Next = 3 separate 16-layer GDN kv-cache groups, not 1).
  Default ON. NOTE the dd45c3c1 "committer sync-DtoH = dominant tax" hypothesis was
  later DEMOTED: it is the WAIT LOCATION where the slow GPU pipeline is absorbed, true
  serialization residual ~4.7 ms/event (FR13_B1_SPEED_ATTRIBUTION_BIND).

- **FIX-3 conv-fusion (FR13_TREE_CONV_FUSED).** The tree causal-conv was a Python
  torch-op emulation inside the captured forward: ~95 (chain5)/~124 (cat9) device nodes
  per GDN layer x 48 layers = the 3.01x captured-node delta vs native's single
  `causal_conv1d_update`. Fused to vectorized torch + static gather tables, bit-exact by
  construction (byte A/B 283/283). cat9 1.076x -> **1.030x** (~7.7 ms saved); chain5 ->
  **1.019-1.030x** (best 1.0188x). The <=1.0x bar was NOT reached — chain5 ON still
  ~4.1-6.5 ms/fwd above native. Default ON. NOTE: replay route is ALWAYS ON and FIX-2/3
  REQUIRE it (`_tree_gdn_replay_kernel` default-ON; WY kernel is a SEPARATE parked
  kernel, archive only — 9aa28ce5).

## What is LEFT after FIX-1/2/3 (MEASURED + INFERRED, sharpened)

- **Per-forward tax now ~1.05x = ~+10 ms/fwd** (fdf5ffa7; @64-tok 1.045x, @11k 1.056-
  1.063x). This is MEASURED s/fwd, a stable substrate property (rep1==rep2). It is NOT
  the +96 ms pre-FIX-1 drafter regime.

- **The residual is NOT the lm-head — the lm-head is already a single batched GEMM.**
  VERIFY lm-head: `gpu_model_runner.py:4090-4091` gathers all M rows (native 6 / cat9 10
  / cat10 11) into ONE [M,5120] tensor -> single `compute_logits` -> F.linear over
  [M,5120]x[5120,248320], bf16 weight 2.5428 GB read ONCE. Per added verify row =
  **+0.0019 ms** (M-invariant, MEASURED; 539 rows per +1 ms). So "batch the verify head"
  = NO-OP; the +81.9 ms was the DRAFTER head, already collapsed by FIX-1 (fdf5ffa7
  refutes the L1 lever).

- **Residual carrier = GDN state-row traffic + the N_PAD~16 register-spill wall**
  (INFERRED from FR13_SPEED_TAX_BASELINE: +42-46 ms/fwd per node ~= 7x the row-traffic
  floor). The lm-head scales to arbitrary tree WIDTH for free; the binding cost of
  adding a NODE is GDN per-node state traffic. The scaling lever = wide tree + a WY/
  replay-class GDN kernel (36 -> 6 row-touches/layer, spill-free) — at the GDN layer,
  not the lm-head. [NOTE: per MEMORY, WY is PARKED, do-not-revive per user 2026-06-15;
  recorded here as the historical lever framing, not a recommendation.]

## CURRENT STATE (cat9 vs native, B=1)

- **cat9 ~1.030x native s/fwd at accept-parity = VERY close, NOT 2.3x.** The ~2.336x-
  slower number (FR13_WHY_SLOWER_VERDICT) is STALE — it PRE-DATES FIX-1/2/3 and reflects
  the drafter double lm-head regime. Corrected current ratio = **1.030x** (cat9, post-
  FIX-3) / **1.019-1.030x** (chain5).
- Accept/event: cat9 ~3.18 (FIX-A ac1d3039: 2.03 -> 3.1789, +1.15, first crossing ABOVE
  native 3.1613) >= native ~3.16. Lossless confirmed at scale (big-denom cat9 13.55% ~=
  native 13.99%). [Caveat per fdf5ffa7 holds=FALSE: the token-weighted AGGREGATE accept
  draw is UNRESOLVED (-0.43 to ~0, trajectory-confounded); needs a paired teacher-forced
  capture. Accept-PARITY framing here is the median-basis + FIX-A crossing, not a
  settled aggregate.]
- in_proj_ba is BAKED (4d0452df) SPEED-NEUTRAL (0.2248 vs 0.2249).

## The two levers to cross to sub-native (INFERRED projections, NOT measured)

1. **OPT-1 GPU-resident committer (a0e8cc3d).** The committer sync-DtoH readback is the
   dominant remaining main-thread tax (the WAIT LOCATION FIX-2 only partially packed);
   moving it GPU-resident kills the sync + restores run-ahead. INFERRED est 35-60 ms/fwd
   was the pre-FIX-2 census number; post-FIX-2/3 the run-ahead residual is smaller —
   needs a clean GPU measurement. Lossless (committer logic unchanged, just where it
   runs).
2. **OPT-A GB10-tuned fp8 GEMV (087fbd51).** NO GB10 fp8-GEMV config exists in the stock
   kernel; a GB10-tuned config is lossless-by-construction (same math, better tiling),
   INFERRED ~1.45-1.55x s/fwd (140-150 ms vs 218) — a fundamental bandwidth win (NOT the
   98.6 ms peak floor; GB10 lacks TMA/WGMMA). [This is the joint lm-head class — native
   pays the same 85.15 ms/draft lm-head, 39% of its forward; sub-native lever for BOTH
   arms.]

Both must be TREE-ONLY / align-to-native, never modify native's executed machinery
(6c2f46d6 constraint; spine===native superset premise). All per-forward decompositions
above are INFERRED unless the row says MEASURED; the real sub-native number needs the
clean GPU measurement under the prelaunch host-mem protocol.

## STALE numbers corrected

- "~2.336x slower" (FR13_WHY_SLOWER_VERDICT) -> **1.030x** (cat9, current). PRE-FIX.
- "+96 ms/fwd tax" -> **~+10 ms/fwd (~1.05x)**. The +96/+81.9 was the drafter double
  lm-head, removed by FIX-1.
- "batch the verify lm-head (L1)" -> NO-OP; already one batched GEMM (+0.0019 ms/row).
- "committer sync-DtoH = the dominant tax cause" -> it is the WAIT LOCATION, not the
  cause; true serialization residual ~4.7 ms/event; partially packed by FIX-2.
- accept cat9 2.03 -> **~3.18** (FIX-A), >= native ~3.16.

Citations: 008631cd (FR13_B1_SPEED_ATTRIBUTION_BIND), dd45c3c1, 93a4043a
(FR13_B1_FIX1_CONFIRM_BIND + FR13_B1_FIX1_GATE_BIND), 7fe500b5
(FR13_B1_FIX2_GATE_BIND), edc39213, f42aab8c, 1f5f37f0/ef4d7514
(FR13_B1_FIX3_GATE_BIND), fdf5ffa7 (FR13_SPEED_TAX_SCALING_BIND), ac1d3039 (FIX-A),
4d0452df (in_proj_ba), 9aa28ce5 (replay vs WY kernel policy); HEAD code
fr10_phase4_patch_vllm_tree_gdn.py:11069-11476 (single-logits baked ON);
gpu_model_runner.py:4090-4091 (verify lm-head single batched GEMM).
