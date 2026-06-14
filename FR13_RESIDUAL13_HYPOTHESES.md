# FR13 — the residual +13 leaf-width co-residency channels (re-examined with fresh eyes + the empirical anchor)

Date 2026-06-14. READ-ONLY locate (no GPU; concurrent `fr13-forked-fa2-tree` GateOFF control running — pathspec
commit only). Live source read from `/tmp/vllm_live_019/vllm/` (the deployed `vllm/vllm-openai@sha256:3dbe092e`
0.19 tree WITH the LUMO_FB pad block at `gdn_linear_attn.py:553-601/674-723`). Reframes
FR13_WIDTH_CARRIER_INPROJ_BA_BIND (in_proj_ba = only ~4/17), FR13_FA2_CARRIER_OVERTURNED, FR13_FA2_MDEPENDENT.

## EMPIRICAL ANCHOR (ground truth)
chain5 (5-spine, no branches) = **5 flips** ≈ native E5 = 3. cat9 (5-spine + 4 leaves) = **22**. The branches add
**+17 via co-residency** (2fe2c567: 11/11 ch2 flips ON the spine = SPINE_PERTURBATION). LUMO_FB pad of
in_proj_ba+out_proj (M-invariant, lossless [T,T,T,T], accept/event 3.017) drops **22→18 = only −4 (~18%)**.
**RESIDUAL +13 = 18 vs chain5's 5.** The prior w6rbv6qot locate rated in_proj_ba ★★★★★ as THE carrier — WRONG;
it is a CONTRIBUTOR. Expect MULTIPLE channels summing to ~13.

## What the live-source re-read DEFINITIVELY CONFIRMS as M-invariant (do not re-chase)
Read kernel-by-kernel, not behavior. These hold under fresh eyes + the +13 anchor:

- **fp8 `in_proj_qkvz` (q/k/v/z) — M-INVARIANT, code-proven.** GB10 (sm_121, `to_int()//10==12`) is NOT
  `is_device_capability_family(100)` (that needs major==10) and `support_deep_gemm()=is_cap(90) or family(100)`
  ⇒ **DeepGEMM/Flashinfer/cutlass-block paths are all OFF on GB10** (`cuda.py:515-517`, `deep_gemm.py:87-92`).
  Route = `self.w8a8_blockscale_op = _run_triton` → `w8a8_triton_block_scaled_mm` (`fp8_utils.py:1193,1069`).
  `get_w8a8_block_fp8_configs`→None (no Spark JSON) ⇒ **default cfg `BLOCK_SIZE_M=64, BLOCK_SIZE_K=128`** for
  BOTH M=5 and M=10. In the jit kernel (`:1127-1138`) the K-loop `accumulator += tl.dot(a,b)*a_s*b_s` runs a
  FIXED tile-count `cdiv(K,128)` in fixed order **per output row**, no split-K, MMA fragment shape keyed on the
  `BLOCK_SIZE_M=64` constexpr (NOT on runtime M). Input quant = `per_token_group_quant_fp8` (group=128, row-local,
  no cross-row reduce). Online-search confirm: TM batch-invariant fp8 path also pins `BLOCK_SIZE_M=64` — GB10's
  default already equals the BI config. **Genuinely M-invariant; prior refutation HOLDS, now with kernel evidence.**
- **`out_proj` — fp8 (NOT in `modules_to_not_convert`), same triton path ⇒ M-INVARIANT.** The LUMO_FB out_proj
  pad is therefore a no-op (consistent with the −4 being entirely the ba seam). out_proj is post-scan AND M-invariant.
- **gate `RMSNormGated` — M-INVARIANT at M≤2·sm_count.** `forward_cuda`→FLA `rmsnorm_fn` (`layernorm_guard.py`):
  reduction is `tl.sum(...,axis=1)` over the HIDDEN dim per row (`:123-133`), num_warps keyed on `BLOCK_N`
  (=group_size constexpr) NOT M, and `rows_per_block = min(next_pow2(cdiv(M,2·sm_count)),4)` (`:171-175`) =
  **1 for both M=5 and M=10** on GB10's large sm_count ⇒ identical constexpr/codegen, 1 row per program. No
  cross-row reduction. **M-invariant.** Prior refutation HOLDS.
- **conv (our fused tree conv + ex2-silu) — ROW-OCCUPANCY M-INVARIANT.** `fr13_tree_conv_fused.py`: per-row
  `index_select` window gather + ONE bf16 elementwise tap-mul over `[tree_n,width,dim]` + width-reduction
  (`range(start,width)`); **grep confirms ZERO `tl.dot/bmm/matmul/conv1d`** in the path. ex2-silu = flat
  element-parallel (program_id over flattened elems, BLOCK=256). Adding leaf rows extends independent rows;
  spine tap-acc unchanged. M-invariant. Prior refutation HOLDS.
- **layer RMSNorms (input/post_attn/q_norm/k_norm) — M-INVARIANT.** Oink SM100 fast path is OFF on sm_121;
  route = `ops.fused_add_rms_norm` CUDA kernel = one block per token, reduce over hidden dim per row. Per-row.
- **MoE — N/A.** `/models/qwen3.6-27b-fp8/config.json` (arch `Qwen3_5ForConditionalGeneration`, `qwen3_5_text`)
  has **NO `num_experts`** ⇒ `config.num_experts==0` ⇒ DENSE `Qwen3NextMLP` (fp8 gate_up/down, M-invariant),
  NOT `SharedFusedMoE`. The fused-MoE grouped-GEMM token-co-residency channel (`moe_align_block_size` sorted-block
  lane shift) — a strong-looking M-keyed candidate — **does NOT exist for this checkpoint.** (Checked before ranking.)
- **full-attn q/k/v/o proj — fp8 (not in not_convert) ⇒ M-invariant;** `attn_output_gate = sigmoid·mul` elementwise.
- **cross-branch ancestry mask (TREE_ATTN scan side) — exact 0.0** (FR13_BV_GEOMETRY; strict -inf select folds no leaf).

## RANKED RESIDUAL +13 CHANNELS (code evidence + A/B signature + targeted fix; summing ~13)

### R1 — FA2-fork query-tile M-dependence × 16 full-attn layers  (≈ +6, the largest residual block)
**Code evidence.** `scripts/fr13_patch_fa2_tree_bias.py`: the fork adds the dense ancestry bias to `acc_s` AFTER
QK, BEFORE softmax (`apply_tree_bias`, lines 138-156). The leaf-KV columns get `-INFINITY` for spine rows
(`:63-64`) ⇒ algebraically the spine attn_out is independent of leaf KV (exp(-inf)=0 never updates online-softmax
max/sum). **So leaf KV co-residency does NOT perturb the spine via the KV axis** — the FA2 fork is NOT a KV-block
batch-variance (and this build has NO split-KV: `flash_attn_interface.py:298-299` raises on num_splits>1). The
M-dependence is the **QUERY-tile**: the carrier (`FR13_FA2_MDEPENDENT`, in-process MAB) measured the deep-spine row
attn_out **RAW≠0 in 14/16 full-attn layers, every value an EXACT bf16 power-of-2, MONOTONE in spine depth** =
kBlockM=64 MMA-fragment count + `Is_even_MN=false` predication + tree_bias lane offsets `q/k_offset=max_seqlen_q−rows`
all shift when M=tree_n changes. This is the canonical FA2 batch-non-invariance (TM blog: "the way the split is done
depends on the number of new tokens; pre-update the layout so every reduction sees the same sequence"). 16 layers ×
~1 bf16-ULP, compounding into the deep stack = a real spine-residual perturbation.

**Why NOT cleanly refuted (the prior OVERTURNED verdict is SOFT).** QPAD (pad query to N_PAD_Q=64) GATE-1 drove the
named carrier L31 3.9e-3→**0.0 and 14/16 layers→0.0** (the M-dependence IS real and IS the FA2 tile). GATE-2 e2e =
24 flips (did not drop) BUT was **class-12 trajectory-confounded** (served_lens [76,103,128,128] vs [128,128,128,126]
forked the stream) — its own verdict flagged "complete non-response cannot be a confound" yet QPAD only fixed 14/16
(L23/L35 1-2 ULP residual from M5-pad59 vs M9-pad54 suffix-KV layouts slipping), so it was NOT a complete fix and
GATE-2 was NOT a clean instrument. **Verdict: FA2 query-tile is a CONFIRMED M-dependent spine perturbation; its e2e
contribution is UNRESOLVED, not zero.** Distinct mechanism from the GEMMs (a fork kernel, not a cuBLAS/triton GEMM).

**A/B signature.** In-process MAB already banked: `scripts/fr13_fa2_mab_replay.py` + `FR13_FA2_MAB` hook —
deep-spine row attn_out RAW max_abs per full-attn layer, M=tree_n vs M=spine-slice, on captured K/V. RESIDUAL13
discriminator: re-run with QPAD ON **AND fix the L23/L35 suffix-KV-layout slip** (pad the suffix-KV region to a
fixed length too, not just the query) → all 16 layers → 0.0; THEN an e2e gate with **served-len-pinned prompts**
(force identical commit prefixes so GATE-2 is not trajectory-confounded) → does the +13 drop?

**Targeted fix (real, not reward-hack).** Extend FR13_FA2_QPAD: pad BOTH the query tile to fixed N_PAD_Q=64 AND the
suffix-KV span to a fixed length so kBlockM tiling, `Is_even_MN`, and the tree_bias q/k offsets are M-invariant;
padded rows -inf-masked (contribute 0, sliced off). Value-preserving, tree-verify-only (not global BI), our kernel
computes. This is the "pre-update the KV layout so every reduction sees the same sequence" prescription applied to
the fork. NOTE this is a DIFFERENT mechanism than LUMO_FB (a fork-kernel tiling pad, not a cuBLAS-shape pad).

### R2 — bf16 `in_proj_ba` residual M-dependence NOT fully pinned by the current pad  (≈ +2, on top of the −4 it already gave)
**Code evidence.** `in_proj_a`/`in_proj_b` ARE in `modules_to_not_convert` ⇒ **bf16** `UnquantizedLinearMethod`
→ cuBLAS/cuBLASLt `mm`. cuBLASLt's heuristic kernel + split-K selection is keyed on (M,N,K); at skinny M, M=tree_n
vs the pad-M can pick different tensor-core tiles → ~1-ULP a/b shift → feeds the gate `b_g=-exp(A_log)·softplus(a+dt)`,
`b_beta=sigmoid(b)` into the spine recurrent state at EVERY accepted node, amplified ~32× by gate 1/rms over ~48 GDN
layers. The LUMO_FB pad (`:557-601`) issues ONE `in_proj_ba(_lumo_fb_padded)` at fixed `pad_rows·row_len` (pad_rows=16)
— this PINS the cuBLASLt shape to a constant **across cat9 events** (the −4 it banked). But: (a) the row_len varies
with tree shape; (b) more importantly the pad fixes the shape but cuBLASLt may STILL pick a split-K kernel at the
padded M that has a different reduction order than chain5's M=5 — the pad makes cat9 self-consistent, not chain5-equal.

**A/B signature.** Extended in_proj_ba RE-RUN A/B (the doc's "covers only the ba seam"): from a captured PRE-projection
hidden span, re-invoke `in_proj_ba` at M=pad16·row_len vs M=5 (chain5 geometry) vs M=10, compare the deep-spine a/b
row RAW (int-view, not atol). If pad16 ≠ M=5 → the pad pins cat9 but does not match chain5 → residual ba contribution.

**Targeted fix.** Set `LUMO_FB_PROJ_PAD_ROWS` so the padded M EQUALS the chain5 spine M-regime (or force the cuBLASLt
algo via a deterministic `cublasLtMatmulAlgoGetHeuristic` pin / `torch.backends.cuda.matmul` workspace-fixed path so
M=5 and M=16·row_len select the SAME kernel). Directive-authorized batch-invariance (#42960). Real per-row math
preserved (zero pads contribute nothing).

### R3 — chunk-vs-recurrent GDN scan STATE-FEED, DEPTH-keyed (not co-residency) — the chain5 5-vs-3 baseline AMPLIFIED by cat9's deeper accept  (≈ +4, but NOT M-invariance-fixable)
**Code evidence / mechanism.** FR13_CONV_NOT_CARRIER + FR13_NODE5_LADDER: at num_accepted=4 the live arm builds
node-5's state via a **rank-1 tree-scan over the accepted chain** seeded from b_h0; the clean oracle builds the same
logical state via a 1687-token **chunked-prefill scan** — two realizations of one recurrent state = the documented
chunk-vs-recurrent ~1-ULP gap, born at L0, amplified ~32× by gate 1/rms, crystallizing at L60/L61. The scan KERNEL is
M-invariant (BV-16 D16=D32=0.0); the gap is the **state-feed realization across accept DEPTH**, NOT a branch-occupancy
op. This is ALREADY present in chain5 (5 vs native 3 = +2) and SCALES with accept depth — cat9 accepts deeper (3.198)
than chain5 (2.664), so the SAME depth-intrinsic gap produces MORE flips at cat9. **Part of the +13 is this depth
amplification, which is co-resident-correlated (deeper accept happens because leaves exist) but is NOT an M-keyed op:
no M-invariance pad/fix touches it.**

**A/B signature.** The concurrent/queued GPU sub-op A/B (w68z6gxgy class): deep-spine scan_out first-nonzero
**M=10 vs M=5 on the SAME captured input**. If scan_out M10-vs-M5 ≈ 0.0 (expected, kernel M-invariant) → this block
is DEPTH-intrinsic, confirming it is NOT a co-residency op to align → route = scan state-feed bit-exact (fp32 state
accumulation / op-order / l2norm-raw-g align) OR tree-reshape, NOT an M-pad.

**Targeted fix.** NOT an M-invariance fix. Either (a) align the rank-1 tree-scan state-feed to the chunked-prefill
realization within native's self-floor (WY territory — PARKED, not revived without the user; non-WY sub-levers = fp32
state accumulation, op-order/l2norm/raw-g alignment), or (b) tree-reshape (shallower committed spine → less
depth-accumulation) per [[project_fr13_tree_reshape_unifying_lever]].

### R4 — bf16 conv-tap carrying R2's a/b is N/A; conv consumes q/k/v not a/b  (≈ 0, control)
conv1d consumes `mixed_qkv` (fp8, M-invariant), NOT a/b — so it cannot carry R2's perturbation and adds no NEW
M-dependence (R4 from the prior doc, re-confirmed). Predict ~0 in any A/B; keep as the clean control arm.

### R5 — residual fp8/RMSNorm sub-ULP — ruled to ~0 by the code reads above. Listed for completeness; predict null.

## SUMMARY TABLE (the +13, ranked)
| # | channel | mechanism | ~contrib | M-invariance-fixable? | A/B signature |
|---|---|---|---|---|---|
| R1 | FA2-fork query-tile | kBlockM/Is_even_MN/tree_bias-offset, ×16 full-attn layers | ~6 | YES (extended QPAD: query+suffix-KV pad) | fr13_fa2_mab_replay RAW→0 all 16 + served-len-pinned e2e |
| R2 | bf16 in_proj_ba residual | cuBLASLt split-K M-keyed, pad pins cat9 not chain5 | ~2 | YES (pad-rows match chain5-M / algo pin) | extended in_proj_ba re-run A/B (pad16 vs M5 vs M10, int-view) |
| R3 | GDN scan state-feed | chunk-vs-recurrent, DEPTH-keyed (amplified by deeper accept) | ~4 | NO (depth, not M) — state-feed align or reshape | scan_out M10-vs-M5 on same input ≈0 ⇒ depth-intrinsic |
| R4 | conv-tap | consumes fp8 q/k/v not a/b | ~0 | n/a | control, predict 0 |
| R5 | fp8/RMSNorm sub-ULP | code-proven M-invariant | ~0 | n/a | null |

## HONEST CAVEAT (the load-bearing tension)
The static code read says **every spine op except in_proj_ba (bf16) and the FA2 fork query-tile is M-invariant**.
The empirical +17/+13 is firm. So the residual is most consistently explained as **R1 (FA2 query-tile, the only other
proven-M-dependent op feeding the spine, 16 layers) + R2 (in_proj_ba not fully pinned) as the M-KEYED co-residency
piece (~8), and R3 (depth-keyed chunk-vs-recurrent scan, amplified by cat9's deeper accept) as the rest (~4-5)**.
R3 is co-residency-CORRELATED (leaves cause deeper accept) but is NOT an M-keyed op — so an M-invariance program
caps out around R1+R2 (~8 of 13), and the last ~4-5 needs the scan state-feed/reshape lever, NOT a pad. The decisive
disambiguation = the deep-spine **scan_out M10-vs-M5 first-nonzero on identical input**: ≈0 ⇒ R3 is depth-intrinsic
(no co-residency op left to align → R1+R2 are the whole M-keyed budget); ≠0 ⇒ a surprise M-keyed scan-input op re-opens.
**Do NOT overstate a single new carrier** — the +13 is R1+R2+R3, not one op.

Pairs with [[reference_diffuse_gdn_accumulation_explained]], [[feedback_math_correct_vs_bitexact]],
[[reference_gdn_verify_sequential_dispatch]], [[feedback_no_reroute_reward_hacking]],
[[feedback_check_artifact_before_concluding]] (cat9-vs-chain5 accept is cross-trajectory; flip-count each-vs-own-oracle
is the comparable metric), [[project_fr13_tree_reshape_unifying_lever]], [[feedback_read_vllm_source_first]].

Sources (online research): Thinking Machines "Defeating Nondeterminism in LLM Inference"
(https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) — Split-KV/query-count batch-variance +
fixed-split-size + pre-update-KV-layout fix; vLLM fp8 W8A8 docs
(https://docs.vllm.ai/en/stable/features/quantization/fp8/) — Triton block-FP8 BLOCK_SIZE_M=64 BI config;
FlashAttention-2 (https://arxiv.org/pdf/2307.08691) — kBlockM query-tile / online-softmax tiling.
