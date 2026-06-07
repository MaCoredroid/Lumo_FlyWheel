# FR-13 — Fork vLLM FlashAttention-2 (CUTLASS) to carry a tree additive-bias → byte-exact tree verify

**Branch:** main · ONE GPU · Driver: codex_fr13 · Decision: user chose the CUTLASS fork (byte-exact spine AND branch) over the cost-gated spine→FA2. Read `FR13_FA2_TREE_BIAS_FORK_RESEARCH` (the background-agent verdict, source-cited) first.

## Why (the verdict that forced this)
Byte-exact Triton TREE_ATTN → CUTLASS FA2 is **provably impossible**: the tensor-core MMA accumulation + warp-reduction order is fixed by each kernel's fragment layout and is **not Triton-controllable**; fp32 non-associativity makes the ~0.00195/layer attn_out gap an **irreducible floor** that compounds over the 16 full-attn layers to ~18.78 → the tree verify is lossy vs native E5 (which uses FA2). Native MTP-5 (E5) uses **FA2** (`_vllm_fa2_C.varlen_fwd`, CUTLASS); the tree uses **TREE_ATTN** (Triton, with the `qq_bias` tree mask). Two implementations ⇒ never bit-identical.

## The fix: run the tree through FA2 itself, plus a tree additive-bias
FA2's varlen API has **no** additive-bias/custom-mask input (only `alibi_slopes` + `softcap`), so stock FA2 can't carry the tree ancestry mask (the `flashpath` experiment gave 0.427 = FA2 computing plain causal, wrong for branches). So **fork the vllm-flash-attn FA2 csrc to add one input**: a per-(query,key) additive bias tile added to `acc_s` **after the QK gemm, before softmax** — mirroring exactly how `qq_bias` is added in the Triton kernel. Because the bias is an **elementwise add that does NOT change the MMA/reduction order**, the result stays byte-exact to native FA2:
- **Spine rows** get a causal-equivalent bias (0 on ancestors, −inf elsewhere) ⇒ FA2 computes exactly native-FA2-causal ⇒ **byte-exact spine** (kills the 16-layer compounding).
- **Branch rows** get their ancestry bias ⇒ FA2 in its exact order ⇒ **byte-exact to the branch-path oracle** (also FA2).

## Precondition (codex confirm first — the multi-layer capture)
Before building: confirm from the multi-layer capture that **q/k/v (post-norm/proj/RoPE) is byte-exact 0.0 on spine AND branch** at every full-attn layer, so the 0.00195 attn_out is *purely* the kernel (not input drift). If q/k/v drifts, fix that wiring first — the fork won't help an input-drift problem.

## Build steps
1. **Locate the FA2 source** to fork: the vllm-flash-attn csrc (`csrc/flash_attn/src/{flash_fwd_kernel.h, softmax.h, utils.h}` + the op registration `csrc/flash_attn/flash_api_torch_lib.cpp`). Local checkout: `output/auto_research/...cutlass_source_workspace/vllm-source/.deps/vllm-flash-attn-src/`. Pinned commit per cmake `bce29425`.
2. **Add the bias input:** extend `varlen_fwd` (and the kvcache/append path used for spec-decode DECODE) signature with `Tensor? tree_bias` ([num_q, num_kv] or block-tiled), thread it to the kernel, and inside the kernel add `acc_s += tree_bias_tile` **after** the QK `gemm` (`flash_fwd_kernel.h:319`) and **before** `softmax` (`softmax.h` exp2). Keep everything else (exp2/scale-fold/rescale/PV gemm/normalize) untouched so the MMA path is identical to native.
3. **Build the forked extension for GB10** (sm_120 / cu130). THE MAIN RISK — flag immediately if the CUDA/CUTLASS build for sm_120/cu130-nightly fails; do not silently fall back. Build inside the vLLM container (the GPU is only in the NVIDIA container).
4. **Route the tree attention** through the forked kernel: in `tree_attn.py` (forward), replace the `unified_attention(..., qq_bias=...)` Triton call with the forked `flash_attn_varlen_func(..., tree_bias=<the same ancestry-bias tile>)`. Keep it flag-gated (`FR13_FA2_TREE_BIAS=1`) so default path is unchanged.
5. **Must cuda-graph-capture at B=4** (native FA2 captures; the forked one + a static bias buffer should too). Confirm FULL capture, no PIECEWISE downgrade.

## Verify (TOP-DOWN, byte-exact, spine AND branch)
- Per-layer ladder, every full-attn layer (3,7,..,63): `attn_out` and the residual = **byte-exact 0.0** vs native FA2 — spine vs native chain, each branch vs its native-on-path oracle. No compounding (kill the 18.78).
- Re-run the input→layer0→…→logits ladder: all stages 0.0.
- Then e2e: **spine-only accept** (should now reach ~E5 ~2.7) + **bag_TV vs E5** (→ floor) + **branch bonus** (the superset). LARGE sample for the lossless TV.
- ALSO (cheap, do early): re-measure the **post-input-fix accept** on the current TREE_ATTN server (does 0.83 already recover?) so we know the trajectory before the build pays off.

## Constraints
- ONE GPU; relaunch crashed captures WITHOUT --rm; empty_cache; kill leftover containers. Read LIVE source before patching. Commit+push every real step to main. NO copy/dense/re-stream; the forked FA2 computing the tree IS our kernel (not a splice — it runs in the real path, byte-exactness verified against the native-FA2 oracle, not by calling native on the spine). Ask before any close/pass-fail.
- Speed note: on bandwidth-bound GB10 the attention is a small fraction of the ~27GB/forward weight stream; FA2 ≥ Triton; the fork is ~neutral on per-forward time. The speed win is accept/event (superset), which byte-exactness unlocks.

## Kernel requirements (user, non-negotiable)
- The forked FA2 kernel must take a **GENERAL tree** (arbitrary topology, expressed via the additive ancestry bias: `bias[q,k]=0` if k is an ancestor of q [prefix + self + path-to-root], `-inf` otherwise) and **verify the WHOLE tree (spine + ALL branches) in ONE kernel call** — NOT a per-row spine/branch backend split, NOT per-row rerouting. One general kernel, one pass, any tree.
- **NO reward-hacking:** do NOT copy native output, do NOT reroute any row to a native call, NO splice. The forked kernel genuinely computes the tree attention. Byte-exactness is verified against the **native-FA2-on-path oracle with the splice OFF** (our forked kernel computing), per the standing reward-hacking rule. A green number that comes from calling native on the spine is rejected.

## Standing rule (user): top-down divergence after EVERY change, all layers, no regression
After EVERY new kernel build OR wiring change, run the FULL top-down divergence ladder — input → every one of the 64 layers (attn_out + residual) → final_norm → final logits — tree vs native, SPINE AND BRANCH (each branch vs its native-on-path oracle). TWO checks every single time:
1. the targeted stage moved toward 0.0 as intended;
2. **NO previously-0.0 stage REGRESSED to nonzero** — input stage = 0.0, GDN layers = 0.0, and every already-fixed full-attn layer STAYS 0.0.
Cover ALL layers, not just the one you changed — a kernel/wiring change can silently regress an unrelated stage. Run this BEFORE any e2e accept/lossless measurement. If anything regressed, fix it before proceeding. The top-down drift graph is examined EVERY run, not occasionally.

## Standing rule (user): TWO strict drift gates per commit, verifier-only, bound to the commit
All our changes (forked FA2 + any wiring) MUST be **VERIFIER-ONLY** — they must NOT change regular (non-spec) decoding. Every commit (kernel/wiring change) is **bound to TWO STRICT top-down ladder results**, recorded in a COMMITTED tracked file `FR13_LADDER_LOG.md` (commit hash + config=strict + the per-layer table), committed WITH the change — so `git log` + that file = a per-commit audit trail ("at this commit we ran, we got this").
1. **VERIFY-PATH gate** — the tree verify (with tree-bias) byte-exact to the native-FA2-on-path oracle: input → every full_attn layer attn_out → final_norm → final logits = **0.0**, spine AND branch.
2. **REGULAR-DECODE gate (the verifier-only proof)** — a plain decode (no tree/spec, no bias) with the forked FA2 byte-exact to the **ORIGINAL pristine/stock FA2** (or unpatched model): top-down ladder = **0.0 at every layer**. This proves our changes do NOT touch regular decode.
CRITICAL: a `FR10_ALLOW_LINEAR_FALLBACK` run is a DIAGNOSTIC ONLY (GDN may be linear) — its numbers are NEVER a valid ladder result and must NEVER be bound to a commit. Both gates must be STRICT + 0.0 before a commit is recorded as passing.
