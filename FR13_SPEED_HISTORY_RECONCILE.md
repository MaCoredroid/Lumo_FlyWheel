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

---

# APPENDIX (per-forward-tax lineage) — residual-tax SLICE decomposition + which lever removes which slice + the N_PAD spill

This appendix sharpens the *current residual tax* (the ~6.5 ms cat9 over native) into named
slices and maps each lever to a slice. Complements the lm-head lineage above (which proves the
+82 ms lm-head adder is GONE). Same basis: decode_seconds, greedy, metrics OFF, native E5 0.2182.

## Current deployed s/fwd (MEASURED, FR13_B1_FIX3_GATE_BIND.md L251-257)
- native E5 = 0.2182 s/fwd. cat9 ON clean = **0.2247-0.2249 = 1.030x = +6.5 ms**. chain5 ON_b =
  0.2223-0.2226 = **1.019x = +4.4 ms**. in_proj_ba-baked cat9 = 0.2248 (SPEED-NEUTRAL, 4d0452df).
- The fdf5ffa7 "~1.05x = +10 ms" is the @11k-ctx figure (1.056-1.063x); the deployed 64-tok
  greedy gate figure is the tighter 1.030x / +6.5 ms. Both MEASURED; use 1.030x as the current
  deployed cat9 number, +10 ms as the long-context end.

## The ~6.5 ms residual, decomposed (only FIX-3's 7.7 ms + FIX-2's 3-4 ms are DIRECTLY MEASURED;
the slice estimates are INFERRED from output/fr13_b1_speed_census/, NOT a per-kernel nsys trace —
nsys cuda_gpu_kern_sum export is EMPTY so there is no per-kernel attribution)
| slice | est ms (INFERRED) | what it is | removed by |
|---|---|---|---|
| committer DtoH+sync | ~2.5-4 | the FIX-2-packed *single* blocking `current_stream().synchronize()`+`.tolist()` on the MAIN launch thread; census: chain5 blocks 91.9% in memcpyAsync vs native 0.8% -> loses async run-ahead | **OPT-1** |
| eager Python glue | ~1.5-2.5 | committer path-LCP tree-walk + replay-publish + runner tree-metadata, dispatched eager on the critical thread, gated behind that sync | **OPT-1** (folds in) |
| residual graph-node (#4-#7 conv residue + per-layer staging) | ~1-2 | post-FIX-3 node residue; wall-bounded <=10 ms by the no-tree-GDN 1.347x discriminator -> NOT dominant | OPT-C / FIX-3 follow-ups |
| residual eager launches | ~0.5-1 | post-FIX-1/2 leftover torch-op dispatch | folds into node/glue |
The DOMINANT residual slice is the committer DtoH+sync + the eager glue it gates (~4-6 ms
combined). That is OPT-1's target. Node-count is NOT dominant (FIX-3 took the big slice).

## N_PAD=16 register-spill wall (the SCALING tax, MEASURED ptxas wp5hsu63v, FR13_BV_NATIVE_MATCH_BIND)
The verify scan holds ALL N_PAD nodes' recurrent state in registers (`h_cache=(N_PAD,BV,DIM_K)`,
DIM_K=128, cap 255 regs/thread). cat9 = N_PAD = 1<<(9-1).bit_length() = 16.
| geometry | tree | measured n_regs | spill B/thread | verdict |
|---|---|---:|---:|---|
| BV=16, warps=8 (DEPLOYED) | cat9 N_PAD=16 | 254 | **0** | FITS (at the 255 cap) |
| BV=32, warps=4 (native geom) | cat9 N_PAD=16 | 255 clamped | **636** | SPILLS hard (runs, no CUDA-701) |
| BV=32, warps=4 | N_PAD<=4 | 235 | 0 | FITS (small trees only) |
KEY: the spill is a **SPEED cost, not a correctness wall** — native geometry runs at cat9, just
spills 636 B/thread to local DRAM. The deployed BV=16/warps=8 is the only 0-spill cat9 geometry
but it does NOT match native's reduction tree (the open diffuse-GDN seam). **recompute-state-
from-spine** (3fd0717c) is the spill-free fix: ~64-90 regs (O(1) in tree size), lossless by
construction (relocates only the state SOURCE, never touches the BV/warps reduction tile), lifts
the N_PAD<=16 cap. CAVEAT (red-team holds=FALSE): the existing replay kernel runs at warps=8 and
is live-broken at gate-4 -> recompute's losslessness is a GPU OBLIGATION, not a CPU-proven fact.

## Lever -> slice map (which residual each of the two levers removes)
- **OPT-1 GPU-resident committer (a0e8cc3d, UNBUILT design — FR13_GPU_COMMITTER flag not yet in
  HEAD): removes the committer DtoH+sync slice (~2.5-4 ms) + the eager glue behind it (~1.5-2.5
  ms) = ~4-6 ms.** Mechanism: move the integer accept/path-LCP/bonus decision to a Triton
  committer kernel + CUDA-12.4 graph conditional-node so the data-dependent branch captures,
  restoring native-style run-ahead. INFERRED: cat9 -> ~220.7-218 ms = parity-to-just-below native;
  chain5 crosses below native first (less reclaim needed). Lossless by construction (pure integer,
  location-only move). The real win is the **accept-edge TPS** (cat9 ~3.18 vs native ~3.07
  tok/fwd) at s/fwd parity, NOT "way faster on s/fwd."
- **OPT-A GB10-tuned fp8 GEMV config (087fbd51, UNBUILT — no GB10 JSON authored): does NOT touch
  the residual tree-tax slices; attacks NATIVE's OWN ~45%-of-peak slack** (native = 2.21x the 98.6
  ms floor). NO GB10/sm_121 config exists (fp8_utils.py:803 -> default BLOCK_SIZE_M=64,
  num_stages=2); a GB10-tuned config (BLOCK_SIZE_M=tree-rows, num_stages=3-4) is lossless by
  construction (BLOCK_SIZE_K=128 pinned -> fp32 K-accum unchanged). INFERRED ~140-150 ms/fwd
  (60-70% peak) = ~1.45-1.55x faster s/fwd vs native 218 ms — a FUNDAMENTAL bandwidth win, NOT the
  98.6 ms floor (GB10 lacks TMA/WGMMA). Standalone win is small (low-single-digit-% e2e per
  GemLite); the fraction-of-peak needs it bundled with full CUDA-graph capture (OPT-C) +
  cross-layer prefetch (OPT-D, risk-flagged on sm_121: vLLM #37431 Mamba-Triton async illegal-
  instruction crash). This is the joint lm-head/GEMV class the lineage above flags (native pays
  the same lm-head GEMV) — sub-native lever for BOTH arms.

NET (per-forward-tax lineage): current deployed cat9 = **1.030x native s/fwd (+6.5 ms), accept-
parity** — NOT 2.3x. Residual = committer DtoH+sync + eager glue (~4-6 ms, dominant) + minor
node/launch residue, plus the N_PAD=16 spill capping the scan to deployed-geometry-only. OPT-1
removes the dominant residual (-> s/fwd parity, accept-edge TPS win); OPT-A attacks native's own
bandwidth slack (-> fundamental ~1.45-1.55x). Both UNBUILT; all per-forward ms are INFERRED
(census/literature anchored, nsys per-kernel export empty); the clean GPU measurement is the arbiter.

Appendix citations: a0e8cc3d (FR13_BEAT_NATIVE_SPEED_DESIGN_BIND, OPT-1), dd45c3c1 (committer
sync tax hypothesis), 087fbd51 (FR13_FUNDAMENTAL_SPEED_FLOOR_BIND, OPT-A), df631112 (2-lever
framing), 0ecd94aa (~97% fixed-overhead-not-bandwidth, 3x graph nodes), 4b409630/07f7ce6a/3fd0717c
(N_PAD=16 ptxas spill 636 B/thread + recompute-from-spine), b12d8a40 (BV=16/warps=8 vs native
BV=32/warps=4 geometry seam); HEAD source fr10_gdn_tree_kernel.py:19-126 (N_PAD cap + GEOM_OVERRIDE),
fp8_utils.py:803 (no GB10 fp8 config), output/fr13_b1_speed_census/ (91.9% main-thread block).

---

# APPENDIX 2 (measurement-methodology lineage) — the EXACT clean B=1 protocol to reuse

How to MEASURE the cat9-vs-native B=1 number correctly so the next clean GPU run is a verdict, not
another artifact. Distinct from the lm-head/tax lineage above (what the number IS); this is HOW to
get it. Sources cited inline.

## The basis (the ONLY s/fwd that is a verdict)
`s/forward = vllm:request_decode_time_seconds_sum / vllm:spec_decode_num_drafts_total` (delta over
the window), scraped from `/metrics` RAW counters. NOT TPS, NOT accept (both BANNED as a basis,
reference_fr10_speed_measurement_pitfalls + feedback_dont_handroll_speed). NOT HTTP wall (tree
early-stops make wall unfair). Per-event (drafts denominator) so it is concurrency-immune.
Source: FR13_B1_CURRENT_GATE_BIND.md L34-35,64-65; FR13_B1_FIX3_GATE_BIND.md gate (e) L244.
⚠ `fr10_quick_decode_tps_probe.py`'s per-request `decode_sum_s` is HTTP `req_elapsed` (probe
L181,213) — for the s/fwd verdict use the `/metrics` `metric_delta` fields it also records
(`decode_seconds`, `spec_drafts`), NOT the per-request wall.

## FR10 pitfalls to avoid (reference_fr10_speed_measurement_pitfalls)
(1) AGGREGATE decode_seconds SUMS the concurrent per-request decode wall -> deflates ~concurrency
(faked the FR10 "5x"); fix = per-request OR the spec_drafts denominator. (2) `VLLM_BATCH_INVARIANT=1`
forces slow deterministic GEMMs/attn ("SMs busy, low TPS") — it is the DETERMINISM tool, NOT a speed
config; measure BI=0. These two together were the FR10 double artifact.

## Flags / regime (pin on EVERY arm; FR13_B1_FIX3_GATE_BIND.md L127-139, FR13_B1_CURRENT_GATE L26-58)
PORT=9950, GPU_UTIL=0.82, MAX_NUM_SEQS=1 (B=1), BATCH_INVARIANT=0, FR13_BI_TREE_ATTN=0,
FR10_METRICS=0 (all FR10/12/13 diag writers gated on ==1 -> compiled-out/dead), FR13_REPLAY_ROUTE=1;
FIX-1/2/3/A + in_proj_ba pinned at committed defaults (fr13_launch_locked.sh) — only the
flag-under-test varies. Heavy captures + final-logit diagnostics UNSET. The only varying flag per
A/B; everything else byte-identical (class 11).

## Determinism / BI caveat (93a4043a, MEASURED)
BI=1 is NOT cross-boot deterministic at B=1 on GB10 (zero-diff boots fork at tokens 11-71 = boot
autotune/kernel-selection channel, outside batch-invariance). So: speed reps WITHIN-boot (rep1==rep2
IS byte-identical, class-8); cross-boot needs a same-flag zero-diff floor pair. Pin BI identically on
both arms (it is OFF for speed anyway). Corollary (feedback_no_cross_boot_byte_gate): never gate on
reproducing a banked free-running stream byte-for-byte across boots.

## Prompts / sampling
`output/fr13_acceptance_ladder/prompts_swe4.json` (4 pinned SWE prompts), seed 1313, max_tokens 128,
warmup 1×16, samples_per_prompt=1, client batch_size=1. GREEDY (temp 0.0 top_p 1.0) is the speed
basis — t0.6 carries wall jitter between identical-stream reps and is NOT a speed basis
(FR13_B1_FIX3_GATE_BIND.md L269). Assert prompt-token identity across the paired arms
(fr13_e2e_measure.py `assert_prompt_identity`; reference_capture_once_native_pin_prompt).

## Arms + prelaunch
- Tree: `scripts/fr13_launch_locked.sh` (TREE_ATTN, tree_mtp, num_spec=9, 9-node caterpillar; FIX-1/2/3/A
  + in_proj_ba baked ON). Native: `scripts/fr10_launch_speed_server.sh` num_spec=5 FLASH_ATTN naive_mtp
  (locked launcher L5 names it as the baseline).
- Prelaunch host-mem protocol = `recover_host_memory()` at boot (forked launcher L241; native L162;
  reference_modelserver_host_memory_recovery — sync+drop_caches+swap-cycle via LUMO_SUDO_PASSWORD).
  Between arms: reset_prefix_cache + torch.cuda.empty_cache + docker rm -f + verify `docker ps` empty
  + `free -g` (gpu-mem collection memory). GPU serialized (max 1 GPU workflow).
- Class-9 engagement asserts BEFORE trusting any number: FULL CUDA-graph capture proven ("Graph
  capturing finished"), tok/draft==9 (cat9) / ==5 (native), has_tree_parent_indices, tree_sample_accept
  (probe `--require-tree-engagement`). Fail-loud on disengagement (feedback_fail_loud_assert_engagement).

## Fairness (46e89f22, holds=True)
Diagnostics-OFF residue = 0.5-2.5% of the speed gap, ≥97% intrinsic; gold TPS comparison FAIR; accept
spread NOT instrument-caused. So the measured delta is real. Do NOT present any per-forward
TPS/accept decomposition as a MEASURED fact — label INFERRED (the per-kernel nsys export is empty);
the clean s/fwd is the arbiter.

## Reusable scripts
`scripts/fr10_quick_decode_tps_probe.py` (probe: metric_delta + accept/event + engagement assert);
`scripts/fr13_e2e_measure.py` (orchestrator: capture -> prompt-identity assert -> reducers -> 1 JSON);
`scripts/fr13_launch_locked.sh` / `scripts/fr10_launch_speed_server.sh` (arms);
`output/fr13_b1_fix3_gate/{run_fix3_arm.sh,run_fix3_campaign.sh,reduce_fix3_gate.py}` (per-arm
runner/reducer pattern).

## Exit bars (d7ea6ccd, user 2026-06-12)
(a) strong-lossless (per-change gate: same-seed byte-identical streams greedy+t0.6, accept/event
unchanged, regular-decode pristine); (b) s/fwd AT-OR-UNDER native AND actively TRY strictly-sub-native
(do not settle for parity); (c) cat9 accept/event STRICTLY > native E5 (>, not ≥) before any B=4.
Depth-match speed comparisons (feedback_depth_matched_accept_compare): cat9 (d5) -> native MTP-5 / E5
0.21816; a d3 arm -> E3 (E3 currently UNMEASURED). Final verdict = B=4 CUDA-captured SWE-Verified-4.

Appendix-2 citations: FR13_B1_CURRENT_GATE_BIND.md L26-90, FR13_B1_FIX3_GATE_BIND.md L127-308,
FR13_B1_BACKEND_ABLATION_BIND.md L80, 93a4043a, 46e89f22, d7ea6ccd; scripts/
fr10_quick_decode_tps_probe.py L34-44,181,213,232-275, scripts/fr13_e2e_measure.py,
scripts/fr13_launch_locked.sh, scripts/fr13_launch_forked_fa2_tree_server.sh L241,
scripts/fr10_launch_speed_server.sh L162; reference_fr10_speed_measurement_pitfalls;
output/fr10_native_mtp5_same8_20260604T210257Z (B=4 E5, NOT the B=1 ref).

---

# APPENDIX 3 (THE-TWO-LEVERS lineage) — OPT-1 + OPT-A: implemented first-drafts, GPU-verify pending

This appendix is the **the-two-levers** lineage: the path to STRICTLY sub-native B=1. It reads the
actual implementation commits (not just the design binds Appendix-1/per-forward cited) and CORRECTS a
staleness in this file's earlier sections: OPT-1 and OPT-A are NOT "UNBUILT / flag not yet in HEAD" —
both were implemented as first-drafts on 2026-06-14 (a few hours after their design binds) and are on
main HEAD. The design binds (a0e8cc3d OPT-1, 087fbd51 OPT-A) are dated 06:19/06:43; the IMPLEMENTATION
first-drafts (10ebccac OPT-1, e90de7ef OPT-A) landed 07:31/07:20 same day. Both are flag-gated
default-OFF with CPU byte-A/B gates passing; NEITHER is GPU-verified.

## Lever 1 — OPT-1 GPU-resident committer (FR13_GPU_COMMITTER)
- **Mechanism (MEASURED root).** The dominant remaining MAIN-thread tax is the committer's packed
  DtoH + `cuda.synchronize()` readback: census chain5 blocks the main launching thread in memcpyAsync
  **91.9%** of the verify window vs native **0.8%** (native waits on the async OUTPUT thread, keeping
  run-ahead). The accept/path-LCP/bonus decision (`fr10_phase4_patch_vllm_tree_gdn.py` ~:5780-5879) is
  pure-Python on host lists -> one DtoH+sync/forward on the critical thread. **Note FIX-2 already
  collapsed 6 separate committer DtoH -> ONE packed DtoH + ONE blocking sync/forward** (7fe500b5); OPT-1
  removes that surviving single sync by moving the pure-INTEGER decision (drafts[node]==parent_targets[node]
  compares, parent walk, LCP scan, earliest-leaf strict-`>` tie-break, 3-way bonus select) to a Triton
  integer kernel and side-streaming the readback (CUDA event), restoring run-ahead.
- **Projected reach (INFERRED — design arithmetic).** Reclaims ~4-6 ms of the ~6.6 ms cat9 tax. native
  218.2 / cat9 224.7 ms (1.030x). Conservative 2.5-4 ms -> ~220.7-222.2 ms (1.011-1.018x, still ABOVE
  native); optimistic 6 ms + OPT-2/3/4 -> ~217-218 ms (AT/just-below native). **chain5 (+4.4 ms) crosses
  below native FIRST.** The decisive win is the accept-edge TPS (cat9 ~3.18 vs native ~3.07 tok/fwd)
  making cat9 faster e2e even at s/fwd PARITY — "way faster on s/fwd alone" is NOT realistic from the
  structural pass.
- **Implemented / tested / banked?** IMPLEMENTED FIRST-DRAFT, NOT GPU-verified. Commit **10ebccac** (on
  main HEAD). Ships `scripts/fr13_gpu_committer_kernel.py` (bit-exact CPU oracle + Triton kernel
  `_fr13_committer_kernel` + `fr13_gpu_committer[_full]` dispatch), hook
  `_patch_rejection_sampler_gpu_committer` (registered after the LCP committer patcher on
  REJECTION_SAMPLER_PATH; sentinel `# FR13_GPU_COMMITTER`, version-guarded), and
  `scripts/fr13_gpu_committer_byte_ab_gate.py`. **CPU byte-A/B: 52/52 trees byte-identical, exit 0**
  (TRITON arm skipped on CPU host). Five GPU obligations remain (G1-G5 in FR13_GPU_COMMITTER_BIND.md):
  **G1** Triton byte-A/B on GPU; **G2 = the actual speed win — KILL THE SYNC** (first draft STILL does a
  host `.cpu().tolist()` readback; must move to the FR13_EAGER_PACK side-stream + event, census target
  91.9% -> ~0.8%); **G3** CUDA-12.4 graph conditional-node / torch.cond in-capture accept branch; **G4**
  on-device variable node_count packing; **G5** e2e class-9/10 byte A/B + s/fwd vs E5.
- **Lossless argument (STRONG, verify holds=True for a0e8cc3d).** Pure-integer, location-only host->Triton
  move; no float / no reduction / no reorder (verified at :5780-5879). Flag-off -> legacy loop iterates
  `counts` unchanged -> default path byte-identical. CAVEAT (decisive red-team, OPT-1 raw): the "keep
  CURRENT numerics bit-exact" premise rests on a current state that FAILS its own greedy gold gate (the
  deployed committer serves a non-argmax token at margin = the 22-flip lossless deficit); OPT-1 preserves
  whatever the committer decides, it does not fix that deficit.
- **Tree-only / constraint (CLEAN, 6c5aeaae).** Hook only PREPENDS a flag-guarded branch; touches the
  TREE committer only; native's spine path untouched. Compliant.
- **Blocker.** G2 (kill the sync) is the whole win and is unbuilt; needs a GPU boot. Effort: large.
  Sequenced AFTER the lossless 22->3 fix + the in_proj bake (df631112 endgame roadmap step 4).

## Lever 2 — OPT-A GB10/sm_121-tuned fp8 GEMV (FR13_GB10_FP8_GEMV_CFG)
- **Mechanism (root verified against LIVE vLLM source, pinned 3dbe092e via scripts/vllm_src.sh).** vLLM
  ships NO GB10/Spark fp8 JSON. CONFIRMED in the live image `fp8_utils.py`: `get_w8a8_block_fp8_configs`
  (line 640) looks up `configs/N=..,device_name=..,dtype=fp8_w8a8.json`; no GB10 file -> returns None ->
  `w8a8_triton_block_scaled_mm` (line 678) takes the **else/Default branch (lines 722-731): BLOCK_SIZE_M=64,
  GROUP_SIZE_M=32, num_warps=4, num_stages=2**. At decode M=6-10 with BLOCK_SIZE_M=64 only ~9-16% of the
  M-tile rows are real and num_stages=2 barely hides the LPDDR5X weight DMA. OPT-A injects (flag + GB10 +
  M<=32 + block_size[1]==128) **BLOCK_SIZE_M=16, GROUP_SIZE_M=1, num_warps=8, num_stages=4**; BLOCK_SIZE_N/K
  stay pinned (K=128).
- **Projected reach (INFERRED — design; OPT-A parent verify holds=FALSE on the BUCKET-SPLIT only).** ~140-150
  ms/fwd (60-70% of peak) = **~1.45-1.55x faster s/fwd vs native 218 ms**; optimistic ~126 ms (78%). NOT the
  98.6 ms weight-bandwidth floor (26.9 GB / 273 GB/s; GB10 lacks TMA/WGMMA/cp.async.bulk for a megakernel).
  native = 2.21x floor = ~45% of peak (123 GB/s, matches Hazy ~50%). A FUNDAMENTAL bandwidth win, separate
  from (not double-counting) OPT-1's parity.
- **Implemented / tested / banked?** IMPLEMENTED FIRST-DRAFT, NOT GPU-verified. Commit **e90de7ef** (on main
  HEAD), hook `_patch_fp8_utils_gb10_gemv_cfg` registered in main(). Ships the patcher + CPU byte-A/B gate
  `scripts/fr13_gb10_fp8_gemv_cfg_byte_ab_gate.py` (G0 patched compiles, G1 idempotent, G2 stock dict
  verbatim, G3 N/K pinned). NO separate bind doc (design = 087fbd51 + commit msg). NEVER GPU-verified.
- **Lossless argument (STRONG, verify-confirmed against fp8_utils.py).** BLOCK_SIZE_K pinned at 128 => the
  fp32 K-accumulation loop (line 611 `accumulator=tl.zeros(...,float32)`; lines 612-621
  `for k in range(0,cdiv(K,BLOCK_SIZE_K)): accumulator += tl.dot(a,b)*a_s*b_s`) runs the IDENTICAL tile
  count in the IDENTICAL order; the output cast (line 624+) happens after full accumulation.
  BLOCK_SIZE_M / GROUP_SIZE_M / num_warps / num_stages are tiling / L2-swizzle / warp-count / pipeline-depth
  only -> bit-identical output bytes => lossless by construction. **HARD EXCLUSION (verify red-flag):
  Split-K / RevSplit-K variants split the K-reduction across CTAs (cross-CTA combine = different fp32
  order) = NOT bit-exact; only JSON-meta tuning of the existing single-K-CTA kernel is lossless.**
- **Tree-only / constraint (TENSION — flag for the user).** OPT-A patches `w8a8_triton_block_scaled_mm`, a
  SHARED kernel native's spine ALSO executes, and the `M<=32` guard ALSO catches native MTP-5 decode (also
  small-M). This is "deviate-shared"-ADJACENT under 6c5aeaae. Defense: the change is OUTPUT-bit-identical
  (lossless by construction), so it is "align-to-native in output" (spine bytes unchanged), only scheduling
  changes. But it IS a config change to a kernel native runs — strictly a SHARED-PATH speedup, not tree-only.
  Reconcile path: if native is run through the SAME patched image its GEMMs speed up too (preserves the
  relative bar), OR gate the override to the tree-verify dispatch only. The bind itself frames it as
  "touches NATIVE's own un-tuned GEMM path -> beats native fundamentally" (the fundamental-win premise),
  which sits against the literal tree-only constraint. **NEEDS a user ruling** on whether shared-but-bit-
  exact config tuning is in-scope.
- **Blocker.** GPU-verify (byte A/B on one fp8 GEMM same M=6-10 OFF-vs-ON + per-token argmax probe +
  s/fwd) + the constraint-scope decision. Sequenced LAST in the fundamental-floor build order
  (OPT-A -> OPT-C -> OPT-D risky -> OPT-B), AFTER lossless 22->3 + OPT-1.

## The two levers as the endgame (df631112 roadmap step 4)
"Verify the 2 previous speed fixes (lossless AND fast)": OPT-1 (10ebccac) + OPT-A (e90de7ef), both
default-OFF, both never GPU-verified. GPU-verify each: lossless (same-boot det + per-token argmax
unchanged + regular-decode pristine) AND fast (B=1 s/fwd vs native E5 FLASH MTP-5); goal sub-native B=1.
Then step 5 = B=4 final gate (CUDA-graph + SWE-Verified 4). Both queued BEHIND the lossless 22->3 fix +
the in_proj bake.

Appendix-3 citations: 10ebccac (OPT-1 impl + FR13_GPU_COMMITTER_BIND.md G1-G5; CPU gate 52/52),
e90de7ef (OPT-A impl + fr13_gb10_fp8_gemv_cfg_byte_ab_gate.py G0-G3), a0e8cc3d
(FR13_BEAT_NATIVE_SPEED_DESIGN_BIND, OPT-1 design, verify holds=True; 91.9% main-thread block;
218.2/224.7 ms arithmetic), 087fbd51 (FR13_FUNDAMENTAL_SPEED_FLOOR_BIND, OPT-A design, verify
holds=FALSE on bucket-split, lossless-by-construction confirmed; 45%-of-peak / 98.6 ms floor),
dd45c3c1 (committer sync-DtoH dominant-tax recon, 2 streams), df631112 (FR13_ENDGAME_ROADMAP.md step
4 = the 2 levers), 6c5aeaae (tree-only / align-to-native constraint, FR13_TRAIL.md); LIVE vLLM source
fp8_utils.py:611-624,640,678,717-731 (K-accum loop + no-GB10-config default branch) via
scripts/vllm_src.sh (pinned 3dbe092ec5b2); HEAD code
fr10_phase4_patch_vllm_tree_gdn.py:14168-14248 (_patch_fp8_utils_gb10_gemv_cfg),
:14249+ (_patch_rejection_sampler_gpu_committer registered in main()).

# APPENDIX 4 (ACCEPT-SIDE lineage) — the TPS NUMERATOR: cat9 vs native accept/event + the L3 lever

Scope = accept/event (tokens-committed-per-step), the TPS numerator (TPS = (accept+1)/s_fwd). The
s/fwd appendices above are the DENOMINATOR; this is the numerator. Basis = spec_accepted_tokens /
spec_drafts (draft-only, bonus excluded, class-12), pinned same-N prompts, greedy, B=1.

## Current cat9 vs native accept/event (depth-matched: cat9 d5 -> native E5/MTP-5)
| basis | cat9 | native | delta | MEASURED? | source |
|---|---|---|---|---|---|
| **controlled pinned-probe, greedy, post-FIX-A** | **3.1789** | **3.1613 (E5)** | **+0.0176** | **MEASURED** | ac1d3039 FIX-A bind |
| native depth-5 structural accept floor | — | **3.076** | — | MEASURED | FR13_PEREVENT_SUPERSET_GATE_RESULT (E5 [0,1,1,1]) |
| gold-gate aggregate (token-weighted, long ctx) | 3.149 | 3.577 | -0.427 | MEASURED but CONFOUNDED | fdf5ffa7 (wf_d8a86320) |
| gold-gate per-request MEDIAN | 3.5872 | 3.5955 | ~0 | MEASURED but CONFOUNDED | fdf5ffa7 |

The two non-confounded MEASURED anchors agree: cat9 has a TINY real accept EDGE (+0.0176) over
native E5. The -0.43 "behind native" is TRAJECTORY-CONFOUNDED (cat9/native fork early into different
greedy streams; cat9's long low-accept requests deflate the token-weighted aggregate on a different
token path; cross-boot tree spread 0.648 >> the ~0.2 margin = a DRAW basis, not a verdict).

## Does cat9 have a real accept EDGE? YES, small + real (NOT the -0.43)
- Controlled: +0.0176/event over native E5 (MEASURED). Structural proof: per-event SUPERSET gate
  (e720b0be, verify HOLDS) = cat9 net **+15 lossless leaf-saves** over E5 (21 lossless - 6 lossy - 0
  spine-regressions), **0 spine-regressions by committer construction** (strict >best_lcp spine-favored
  tie-break, confirmed over 250 dump recs). cat9 IS a per-event lossless superset of E5 at parity.
- Edge is SMALL because it is a decontaminated branch bonus: FIX-A (ac1d3039, H1 ROWBUG = +1-shifted
  path publication on ~51% of partial-accept events; FR13_TREE_SAMPLE_ROW, baked ON HEAD patcher
  L10869 + launcher :-1) lifted accept **2.0274 -> 3.1789 (+1.15)**; root-reject 39.7% -> 13.8%;
  next-spine-LCP 0.97 -> 3.04. Pre-FIX-A the +0.21 GROSS branch bonus was net-zeroed by ~0.75-0.85
  spine contamination; post-FIX-A realized NET bonus is +0.02..+0.05 (branches still 90.8% co-located
  with parent rejections, capping below gross).
- Spine accept = native at parity (N5 spine oracle, MEASURED, acceptance_ladder): native-on-tree-path
  d0..d4 tree {.662,.650,.688,.764,.694} vs native {.694,.573,.675,.739,.701}, net tree +12 (verify
  not under-accepting). The d0 marginal deficit (tree .667 vs native .894) = genuine DRAFTER quality
  on the tree stream, NOT a verify bug.

## The ~4.3% verify-flips do NOT eat accept (lossless-violations, not accept-drains)
22-23 clear-margin flips land on ALREADY-FULLY-ACCEPTED spine rows (p2 pos21 node7 SPINE
num_accepted=5/5; p3 pos73 node7 reject_correction num_accepted=4) - corrupt WHICH token emits (byte
losslessness), NOT how many accept. Per-event decomp of 23: 6 lossy leaf-saves (net-paid-for by +15)
+ 17 spine-realization/bonus drift (tree-verify FA2+tree-scan vs native FLASH = the 7.39x own-rate
lossless gap, wgb0yegin). Fixing verify bit-exactness moves accept ~0 (+0..+0.05). Edge NOT blocked.

## L3 confidence-gated root-sibling — the accept lever's reach (cat10 lineage)
- Unconditional cat10 (always-on (1,) sibling): NET accept LOSS -0.27 (3.198 -> 2.932), but
  ARTIFACT-DOMINATED (cd30f5ad, verify holds=FALSE corrected readers): trajectory/EOS confound +
  SIBLING-STOP DENOMINATOR ARTIFACT (sibling win is accepted_len=1, deflates d1|d0; de-confound
  recovers ~0.84). VERDICT=no_help (9aa28ce5), archived not merged (986c6e77), NOT on HEAD; lossless
  FLAT (22==22); cost +1 lm-head row ~2.9 ms/fwd.
- d0-rescue is REAL: P(target==root-rank-2 | rank-1 missed) = **0.273 (27%)** MEASURED (2-horse-race);
  unconditional sibling gets d0-reject 0.129 -> 0.094 = **+0.035 d0 accept (~+21/boot) MEASURED**, but
  pays the row 100% of events + dilutes deeper -> net -0.27.
- **L3 LEVER = CONFIDENCE-GATED form** (non-refuted, future, NOT on HEAD): emit (1,) ONLY when root
  top-2 margin < tau (62% of rejects step-0). Gate is FREE (top2 already materialized, one scalar
  compare, zero extra forward). Keeps +0.035 d0 rescue WITHOUT the d1-d4 (mostly-artifact) dilution.
  **L3 reach = +0.035 (d0-rescue floor, MEASURED) to +0.08..+0.15 (first-order doc bound, INFERRED)**
  -> cat9 accept ~3.21..3.33. Decisive test needs a per-node sibling-vs-spine counter (ABSENT from
  saved data) so the upper reach is INFERRED. Non-refuted in wgb0yegin (eabb07f9).

## CORRECTED stale accept numbers
- "cat9 -0.43 BEHIND native" (fdf5ffa7 headline) -> confounded gold-gate; controlled = +0.0176 AHEAD.
- "accept gap UNRESOLVED draw" (fdf5ffa7) -> RESOLVED for cat9 by per-event gate (e720b0be): net +15,
  0 spine-regressions, cat9 IS a lossless superset of E5.
- "cat9 accept 2.03" (pre-FIX-A) -> 3.1789 (FIX-A baked ON).
- "cat10 root sibling helps accept" -> unconditional -0.27 artifact; only confidence-gated lives.

NET (accept-side): cat9 numerator is ALREADY >= native E5 at parity (+0.0176 controlled, structurally
net +15 lossless superset, 0 spine-regressions). The TPS verdict is gated by the DENOMINATOR
(s/fwd ~1.05x), not the numerator - the edge is too small to clear the per-forward tax alone
(break-even needs accept ~3.43 at 1.05x). L3 confidence-gated root sibling is the cheapest numerator
lever (+0.035..+0.15, attacking the 62%-step-0-reject + d0 drafter deficit) but must combine with the
speed levers (Appendix-3) to cross to strictly sub-native TPS. MEASURED vs INFERRED labeled inline.

Appendix-4 citations: ac1d3039 (FIX-A bind FR13_CHASE_FIXA_BIND.md), e720b0be /
FR13_PEREVENT_SUPERSET_GATE_RESULT.md (77e2a0e8 reducer scripts/fr13_perevent_superset_gate.py),
fdf5ffa7 / FR13_SPEED_TAX_SCALING_BIND.md, 4b6769ee (b1_superset_speed_research_wf_c618b0c9.raw.json),
cd30f5ad + 9aa28ce5 + 986c6e77 + 31e227cf (cat10 / FR13_CAT10_INVESTIGATE_BIND.md), eabb07f9 /
FR13_MATH_HISTORY_RECONCILE.md + math_history_reconcile_innovate_wgb0yegin.raw.json, wgb0yegin;
research/fr13_workflows/{acceptance_ladder_wc11426x6.raw.json (N5 spine oracle),
branch_upside_wlhtzqvib.raw.json, chase_fixA_wf_164f7b0d.raw.json}; scripts/
fr10_phase4_patch_vllm_tree_gdn.py L10869 (FIX-A baked ON),
scripts/fr13_launch_forked_fa2_tree_server.sh L114 (FR13_TREE_SAMPLE_ROW :-1).
