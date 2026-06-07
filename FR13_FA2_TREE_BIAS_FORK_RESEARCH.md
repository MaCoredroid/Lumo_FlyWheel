# FR-13 research verdict — byte-exact tree attention requires forking FA2 (Triton can't)

Background research agent (opus, online + live-source, 2026-06-07), source-cited. This is the basis for `FR13_FA2_TREE_BIAS_FORK.md`.

## Verdict: byte-exact Triton TREE_ATTN → CUTLASS FA2 is IMPOSSIBLE
The obstacle is NOT the alignable software choices — it is the **tensor-core MMA accumulation + warp-reduction order**, fixed by each kernel's fragment→thread→register layout, and **not controllable from Triton**. fp32 addition is non-associative, so different summation orders → different bf16 roundings. The ~0.00097 (1 ULP) op-level gap is a **floor**, not a bug; it compounds over 16 full-attn layers → ~18.78. The determinism literature (Thinking Machines "Defeating Nondeterminism", LMSYS/SGLang, FA2's own tests assert only "≤2× a reference") only makes each kernel batch-invariant *against itself*, never Triton==CUTLASS.

Alignable but NOT sufficient (each shrinks, none reaches 0.0): exp base (Triton base-e `tl.exp` vs FA2 base-2 `exp2f` with `M_LOG2E`), softmax scale placement (fused FFMA vs separate mul), KV-iteration order, online-rescale formula, final normalize, tile sizes. IRREDUCIBLE: QK fp32 accum reduction order, P@V fp32 accum reduction order, tiling/reduction grouping. (The P→bf16 cast is ALREADY identical — fp32-PV is a red herring that moves *away* from FA2.)

## Source map (file:line)
- Triton TREE_ATTN: `triton_unified_attention.py` (QK `S += score_scale*tl.dot(Q,K)` :509-511; tree mask `S += load_qq_bias_tile` :525-528; PV `acc += tl.dot(P.to(V.dtype),V)` :545; norm `acc/L` :605); softmax base-e in `triton_attention_helpers.py:371-382`.
- CUTLASS FA2: `vllm-flash-attn-src/csrc/flash_attn/src/` — QK `gemm(acc_s,...)` `flash_fwd_kernel.h:319`; softmax `exp2f(x*scale - max_scaled)` `softmax.h:86-92,118`; rescale `softmax.h:157`; warp reduce `Allreduce<4>` quad `__shfl_xor_sync` `utils.h:112-130`; P cast `convert_type<Element>(acc_s)` `flash_fwd_kernel.h:347`; PV `gemm_rs(acc_o,...)` :367; op reg `flash_api_torch_lib.cpp:109-114` (NO attn_bias arg — only `alibi_slopes` + `softcap`).
- FA2 runs on GB10: forced for `device_capability.major>=10` under `VLLM_BATCH_INVARIANT=1` (`fa_utils.py:117-123`); E5/native launches `--attention-backend FLASH_ATTN`. SM120 crash reports are FA3/FA4 datacenter builds, not the FA2 path we run.

## Paths evaluated
- **(a) Spine→FA2 [free, byte-exact spine]:** route the causal spine through `flash_attn_varlen_func` (native's kernel) → byte-exact by construction; kills the spine compounding (the E5 gate scores a spine). The whole tree currently goes through Triton `unified_attention(qq_bias=...)` (`tree_attn.py:425-444`).
- **(b) Whole tree through stock FA2 + bias — UNAVAILABLE:** FA2 varlen has no additive-bias/custom-mask arg (only alibi + softcap); the newer CuTe-DSL `score_mod`/`mask_mod` is the SM90/SM100 datacenter path, not our FA2 build (Dao-AILab #1805, #1179). The team's `flashpath` 0.427 already showed stock FA2 ignores the tree mask.
- **(c) Fork FA2 CUTLASS + tree-bias [the chosen path]:** add a `tree_bias` input, `acc_s += bias` post-QK pre-softmax — elementwise add, does NOT change MMA/reduction order → byte-exact spine AND branch (the only byte-exact-per-layer route). Real CUDA/CUTLASS work; the build for GB10 sm_120/cu130 is the risk.

## Decision (user, 2026-06-07): path (c) — fork FA2 now, byte-exact both, cuda-fast.
Do NOT spend effort aligning the Triton kernel (provably can't reach 0.0). Speed: ~neutral (attention is a small slice of the bandwidth-bound ~27GB/forward; FA2 ≥ Triton; both cuda-graph-capture). Speed win = accept/event (superset), which byte-exactness unlocks.
