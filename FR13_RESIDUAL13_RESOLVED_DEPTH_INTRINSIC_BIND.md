# FR13 — +13 residual RESOLVED as depth-intrinsic (kernel evidence decisive; empirical A/B blocked 3x on env-infra, not science)

Date 2026-06-14. The L0-GDN sub-op A/B (the empirical +13 discriminator) failed THREE times on INFRASTRUCTURE,
never on the science: (1) original = CUDA device-side assert on the reduced-row arm geometry (fixed by Front B
8cdda4c4); (2) re-run task wc0gyx2za = FR13_GDN_SUBOP_MAB env not in the EngineCore worker (vLLM ray spawn
copies only VLLM_-prefixed allowlist); (3) re-run task wl043ivfu = SAME (env-fix attempt did not propagate;
the hard worker-env gate /proc/176/environ caught it FAIL-LOUD, no wasted teacher-force - class-9). The
env-propagation to the EngineCore worker is the blocker, NOT the residual science.

## DECISION (research-before-deadend satisfied): accept the kernel evidence
The +13 residual (cat9 ~22 minus pure-spine 5, after the in_proj_ba ~8 fix lands cat9 at ~18) is DEPTH-
INTRINSIC + a small FA2-downstream correlate, with NO paddable L0-GDN op. Code-PROVEN (residual-13
FR13_RESIDUAL13_DECISIVE_TEST_BIND, fresh kernel evidence):
- conv = row-occupancy M-INVARIANT (our fused tree conv, no GEMM, per-row).
- GDN scan = BIT-EXACT to native at BOTH BV geometries (FR13_BV_GEOMETRY_NOT_THE_SEAM, RAW 0.0 D16=D32 at
  N_PAD 1 and 16).
- fp8 in_proj_qkvz + o_proj = M-INVARIANT (GB10 sm_121 -> w8a8_triton_block_scaled_mm BLOCK_SIZE_M=64
  constexpr, no split-K).
- gate (RMSNormGated) = M-INVARIANT (ROWS_PER_BLOCK=1 both M, per-row rms).
- The ONLY bf16 GEMM on the spine data path = in_proj_ba (the banked ~8 fix). No MoE (Qwen3Next dense).
=> NO remaining paddable/M-keyed L0-GDN op. Batch-invariance is EXHAUSTED at in_proj_ba.

## The cat9 22 -> native decomposition is now COMPLETE
- native floor 3 (irreducible).
- +2 spine: RESOLVED (FR13_PLUS2_DECASCADE - deep-spine 5 raw = 2 INDEPENDENT events <= native 3 = cascade/
  measuring artifact; small real diffuse residual = FA2 2-ULP MMA floor + diffuse GDN recurrence, NOT
  paddable, parked; accept/event ~native = sub-deployment-impact).
- +17 leaf co-residency: in_proj_ba ~8 (FIXED, LUMO_FB pad, lossless+accept~native, about to BAKE); residual
  ~9 = depth-intrinsic chunk-vs-recurrent + FA2-downstream (NOT paddable, the same diffuse/cascade floor).
The binding arbiter = accept/event (cat9 ~3.0 ~ native), NOT the raw flip count (inflated by cascades +
diffuse near-ties). The in_proj_ba bake is the one clean lossless lever; the rest is the diffuse floor that
accept/event already shows is ~native.

## Optional future (NOT blocking the bake): if a true empirical conv/scan M10-vs-M5 is wanted, the env-fix
is to make FR13_GDN_SUBOP_MAB reach the EngineCore worker (set at patcher import-time the way the 14
propagated FR13 vars got through, OR a VLLM_-prefixed alias) - infra only; the kernel evidence already gives
the answer (~0 = depth-intrinsic). PROCEED to: bake in_proj_ba -> B=1 -> verify OPT-1/OPT-A speed -> B=4
(FR13_ENDGAME_ROADMAP). Pairs with [[feedback_kill_wrong_gpu_task_immediately]] (don't keep grinding a fragile
infra-blocked instrument when the science is settled), [[feedback_fail_loud_assert_engagement]] (the worker-env
gate worked), [[reference_scalar_metric_per_token_blindspot]] (accept/event is the arbiter).
