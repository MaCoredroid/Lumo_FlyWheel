# FR13 — the +13 residual co-residency is NOT code-rankable; decisive test = the FIXED L0-GDN sub-op A/B (in_proj_ba SOLID, FA2 downstream, fp8/gate/conv genuinely M-invariant)

Date 2026-06-14. CPU residual-13 locate `wf_ce6e5f7f-c98` (task ws8oezioa), **verify holds=FALSE** (caught the
Locate's overstatement). Raw: `research/fr13_workflows/residual13_locate_ws8oezioa.raw.json`. Empirical anchor:
in_proj_ba+out_proj LUMO_FB pad dropped cat9 22->18 (~4 of +17); residual +13 = 18 vs pure-spine 5.

## What the re-examination SOLIDLY established (fresh kernel evidence, refutations that HOLD)
- **fp8 in_proj_qkvz = M-INVARIANT** (was right): GB10 sm_121 routes to w8a8_triton_block_scaled_mm with
  BLOCK_SIZE_M=64 (constexpr, not runtime M), no split-K, MMA fragment keyed on the constexpr; M=5 and M=10
  give bit-identical spine rows. (The in_proj_ba bf16-cuBLASLt lesson does NOT transfer — qkvz is an explicit
  M-independent triton kernel.) TM batch-invariant fp8 ALSO pins BLOCK_SIZE_M=64 = GB10's default.
- **gate (RMSNormGated) = M-INVARIANT**: ROWS_PER_BLOCK=calc_rows_per_block(M)=1 for both M=5/10 (cdiv(small,
  ~96 sm) =1), per-row rms over hidden, no cross-row reduce.
- **conv = ROW-OCCUPANCY M-INVARIANT**: our fused tree conv = per-row gather + bf16 tap-mul + width-reduce,
  ZERO tl.dot/bmm/matmul; consumes fp8 q/k/v not a/b.
- **Only bf16 GEMM on the spine = in_proj_ba** (R2, SOLID ~2-4). out_proj is fp8 (the LUMO_FB out_proj pad was
  a no-op => the banked -4 is ENTIRELY in_proj_ba). No MoE (Qwen3Next dense, num_experts==0). RMSNorms M-invariant.

## What the Locate OVERSTATED (Verify holds=False caught it)
The Locate ranked R1=FA2 query-tile at ~6 "the largest residual." REFUTED by 2 dual-verified binds it
discounted: (1) FR13_FA2_CARRIER_OVERTURNED (QPAD drove the FA2 carrier L31 + 14/16 layers -> 0.0, yet e2e
22->24 did NOT drop); (2) NODE7 ladder (first-nonzero is L0 GDN; full-attn L3+ is DOWNSTREAM — a fix at L3+
cannot remove an L0-born 2-ULP divergence). So **FA2 is a downstream CORRELATE; its +13 contribution is most
likely small, not ~6.** (R3 GDN scan state-feed = the DEPTH-intrinsic chunk-vs-recurrent = the +2 spine +
depth-scaling; NOT an M-keyed op, NOT batch-invariance-fixable — the BV/warps fix was already refuted,
FR13_PLUS2_BV_FIX_REFUTED.) The +13 is therefore NOT confidently rankable from static code.

## DECISIVE TEST (the single discriminator) = the FIXED L0-GDN sub-op A/B
Run the FIXED FR13_GDN_SUBOP_MAB (Front B re-fixed the crash, commit 8cdda4c4 / d30755c8: reduced-row arm =
valid committed prior bank ssi0[0,0] as 1-row init-state, all indices bounds-guarded; 11/11 CPU tests,
default byte-identical). gpuRunPlan: boot cat9 + FR13_GDN_SUBOP_MAB=1 FR13_GDN_SUBOP_MAB_LAYER=...layers.0.linear_attn
FR13_GDN_SUBOP_MAB_EXPECT_TREE_N=10. Capture the deep-spine conv1d_out + scan_out FIRST-NONZERO at M=10 vs M=5
on the IDENTICAL captured carrier input (decoherence-free). DISCRIMINATOR:
- conv1d_out or scan_out M10-vs-M5 != 0 => a paddable M-keyed L0-GDN carrier => align/pad it (more of the +13
  fixable, like in_proj_ba);
- both ~0 => the +13 residual is POSITIONAL/DEPTH/FA2-downstream (R3 depth-intrinsic + small FA2 correlate),
  NOT an M-keyed L0-GDN op => batch-invariance is exhausted at in_proj_ba; remaining is the chunk-vs-recurrent
  floor (accept ~native anyway) + the served-len-pinned FA2 question.
(Note the A/B is BLIND to in_proj_ba itself — it slices the M=10 ba output — so it tests conv/scan, the right
residual question.) Pairs with [[project_fr13_22flip_carrier_l0gdn]], [[feedback_check_artifact_before_concluding]],
[[reference_gb10_gdn_backend_fla]], [[feedback_no_reroute_reward_hacking]].
