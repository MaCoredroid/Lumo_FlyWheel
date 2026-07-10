# FR13 — garble fix plan (drift-fix design, 2026-07-10)

From design workflow wf_c7d5ed49 (map existing infra + design fix+gate), then verified
against the study docs. Supersedes the naive "rms-clamp first" idea.

## The reframe (KEY, verified 2026-07-10)
The live-pinned garble (`from_geodetic`->`from_geodentic`, cat8 task 13398) is a
MANIFESTATION of the campaign's central hard problem, NOT a separable quick fix. Verified
against FR13_AMPLIFICATION_LEVERS.md (header: "SUPERSEDED AS THE PRIMARY LEVER") +
FR13_E5_VS_CAT9_SPINE_DRIFT.md (study wsvy4vn5k):
- The ~1.166x/layer amplification (gate 1/rms + deep full-attn) is SHARED with native E5
  (GDN 1.075x/layer, full-attn 1.236x/layer). Reducing it (fp32 / rms-clamp / residual
  re-anchor) lowers BOTH arms equally -> does NOT differentially close the cat9-vs-E5 gap
  -> WILL NOT fix the garble. => DO NOT build the amplification levers as the garble fix.
- The real carrier = CO-RESIDENCY M-DEPENDENCE at the L0 GDN birth-amplitude. Existence
  proof: `chain5` (cat8's exact kernels, spine alone at M=5, no co-resident branches)
  de-cascades to 2 <= native 3. So our tree kernels CAN spine at the floor when M is low.
- The baked pad-block (LUMO_FB_KERNEL_ROWS=1) made in_proj_ba M-invariant, yet garble
  persists -> a DIFFERENT sub-op still carries M-dependence: conv1d_update (receptive-field
  entanglement) OR fused_sigmoid_gating recurrent scan (state-feed). UNLOCALIZED.

## Sanctioned levers (source-side, NOT amplification)
1. SPINE M-INVARIANCE: localize the surviving M-dependent sub-op, then make it M-invariant
   (extend the pad-block idea to conv/scan). Localize via FR13_GDN_SUBOP_MAB.
2. TREE-RESHAPE: narrower/shallower tree -> lower co-residency M -> less drift (chain5 proof;
   project_fr13_tree_reshape_unifying_lever). Tradeoff: fewer draft tokens = lower accept = speed.

## Gate (from the design synthesis, still valid)
PRIMARY (cross-boot-immune): teacher-forced oracle-flip WRONG-ACCEPT gate. For each committed
near-neighbor token, teacher-force cat8 itself POST /v1/completions {prompt: prompt_ids+
served[:i], max_tokens:1, temperature:1.0, logprobs:20} (real prefill dist, reset_prefix_cache
first — dodges the -12.422 spec placeholder). WRONG-ACCEPT iff oracle argmax==correct AND
margin(correct-committed)>1.0 nat. Count over G1 low-margin prompts (fr13_garble_gate PROMPTS,
which DO elicit; geodetic is high-margin, 0 elicit). Glue over fr13_oracle_stream_teacher_force.py.
CROSS-CHECK (per-layer ladder): fr13_node5_ladder_drive.py + fr13_ladder_table.py (tree-verify
vs no-spec-clean final-logit max_abs + argmax-flip, within-boot).

## Next step = LOCALIZE (cheap, sanctioned, gap #3 of the mapper)
Boot cat8 dedicated (scripts/fr13_cat8_dedicated_server.sh EXTRA_FLAGS="FR13_GDN_SUBOP_MAB=1";
patcher auto-writes sidecar /logs/fr13_gdn_subop_mab.flag). Run scripts/fr13_gdn_subop_mab_drive.py
(2 passes, asserts within_boot_det) -> reduce scripts/fr13_gdn_subop_table.py. Read M10-vs-M5 raw
max_abs per sub-op (conv1d_update, fused_sigmoid_gating). FIRST sub-op with M10!=M5 on the deep
spine row = the carrier birthplace -> that determines the M-invariance fix.
KNOWN FRAGILITY (mapper gap #2): the drive needs the right pinned capture event
(output/fr13_commit_argmax/tree_capture.json + prompts_swe4.json); class-9 vacuous risk if it
lands on the wrong event -> assert engagement (EXPECT_TREE_N=10, within_boot_det) before trusting.

## Do NOT
- Build rms-clamp/fp32/re-anchor as the garble fix (superseded, shared with E5).
- Edit the sampler (reward-hack).
- Trust garble RATE (boot-noise) or wa_capture top_logprobs=1 (-12.422 placeholder).

## Localization attempt (2026-07-10) — MAB tooling BLOCKED

Ran the sanctioned FR13_GDN_SUBOP_MAB localization on a dedicated cat9 server. Two blockers:
1. GRAPH mode (default FULL_AND_PIECEWISE) -> hook is EAGER-ONLY (patcher:1924 returns during
   capture) -> VACUOUS (0 records, no engage-fail markers). Fix: ENFORCE_EAGER=1.
2. EAGER mode -> hook ENGAGED (FR13_SUBOP_STAGE=engaged tree_n=10 deep_row=8 num_spec=1) but the
   reduced-M (M5/M1) sub-op re-run hit `Triton Error [CUDA]: device-side assert triggered` ->
   arm-fail -> crashed the EngineCore (server died, GPU recovered clean). The re-run code
   (patcher:2072-2160) already carries a HISTORY of device-assert fixes ("FR13_SUBOP_MAB
   device-assert fix", "FR13_SUBOP_AB_CRASHED_PIVOT_CHAIN3 FIX") and STILL asserts on this
   cat9-eager boot -> the tool is unreliable for this config; a further fix needs
   CUDA_LAUNCH_BLOCKING=1 + Triton index analysis = a rabbit hole into the diagnostic internals,
   not the target.

STATUS: the carrier (conv1d receptive-field vs recurrent-scan state-feed) remains UNLOCALIZED by
the MAB. Mapper-2's HYPOTHESIS (well-supported by FR13_E5_VS_CAT9_SPINE_DRIFT.md: drift born at L0
GDN compute; reference_gdn_verify_sequential_dispatch: verify uses fused_sigmoid_gating) = the
recurrent-scan state-feed at num_accepted>1 (chunk-vs-recurrent realization).

## THE FORK (needs a call; speed matters, user asked)
The garble fix = the campaign's CORE M-invariance problem, two sanctioned paths:
  A. SPINE M-INVARIANCE (speed-preserving, sanctioned PRIMARY): make the surviving M-dependent
     sub-op M-invariant (extend the pad-block idea to the scan/conv). Blocked on localization
     (MAB crashes) OR proceed on the recurrent-scan hypothesis (risk: could be conv). Deep build.
  B. TREE-RESHAPE (sidesteps localization): narrower/shallower tree -> lower co-residency M ->
     less drift. chain5 (M=5 spine-alone) de-cascades to 2<=native3 = existence proof. SPEED COST:
     fewer draft tokens = lower accept rate. Testable via live-SWE (slow ~90min/task) + garble-watch.
Alternate diagnostic to try instead of the crashing MAB: FR13_REPLAY_DURABLE_AB (mapper-1 "PRIME
lead") compares our replay GDN recurrent state vs the native sequential kernel over the accepted
chain (directly tests the scan-state-feed hypothesis) — also eager-only, may also be fragile.
RECOMMENDATION: get a user call on the speed tradeoff (A preserves speed but is deep + tooling-blocked;
B is simpler but costs speed). All fix-validation is slow live-SWE (fast instruments compromised).

## DECISION (user 2026-07-10) — PATH A, with a hard no-HBM-tax rule
Direction LOCKED:
1. FIX THE TOOLING — repair the FR13_GDN_SUBOP_MAB device-assert (and/or find a cheaper
   localization) so we can actually localize the surviving M-dependent sub-op.
2. LOCALIZE cheaply — which ops need M-invariance (conv1d_update vs fused_sigmoid_gating scan
   vs anything else surviving the pad-block), with a MEASURED SPEED COST for each candidate fix.
3. HARD RULE — NO HBM TAX. Any M-invariance fix must be COMPUTE-ONLY (no extra HBM traffic /
   no added copies), because GB10 is memory-bandwidth-bound (273 GB/s) — compute is ~free,
   bandwidth is the ceiling. The pad-block is the template (pads a GEMM = more compute, zero
   extra HBM). Reject any fix that adds HBM copies/reads even if numerically correct.
4. APPLY — build the compute-only M-invariance fix for each localized op (default-OFF flag),
   measure speed, verify the garble drops (teacher-forced oracle-flip gate + live-SWE 13398).
This is the sanctioned speed-preserving path (spine M-invariance), NOT tree-reshape (speed cost)
and NOT amplification levers (superseded).

## LOCALIZATION RESOLVED (2026-07-10) — carrier = conv1d prior-window (skip the broken MAB)
The per-layer ladder in FR13_E5_VS_CAT9_SPINE_DRIFT.md ALREADY localized it (don't need the crashing MAB):
enter L0 byte-exact -> pre_conv(in_proj)=0.0 (pad-block fixed in_proj_ba) -> FIRST-nonzero sub-op =
`conv1d_out` = 0.000977 (1 bf16-ULP) at num_accepted>1. Named prime carrier = the conv1d PRIOR-WINDOW /
state-bank column geometry (project_fr13_conv_priorwindow_root: "conv1d_out wrong bank-row at num_accepted>1;
OPEN"). Mechanism = the deep SPINE row's depthwise-conv receptive window is contaminated by CO-RESIDENT
BRANCH rows instead of its true spine ancestry (M-dependent). Fix target = fr10_phase4_patch_vllm_tree_gdn.py
~:797-818 (conv source-index wiring). Existing PARTIAL infra aimed here: FR13_CONV_COMMITTED_PATH (baked ON,
snapshots committed-path conv rows) + FR13_TREE_CONV_FUSED (source-by-width, gather_committed_path_conv_prior)
— but garble PERSISTS -> the source-by-width wiring is INCOMPLETE (some conv window index still co-resident-keyed
at num_accepted>1). This is M-DEPENDENT + the fix is a RE-INDEX/WIRING correction = COMPUTE-FREE, NO HBM ->
passes the hard no-HBM-tax rule cleanly.

REVISED PLAN (autonomous): (a) MAP the conv prior-window path — what FR13_CONV_COMMITTED_PATH/TREE_CONV_FUSED
do, WHICH index is still co-resident-keyed at num_accepted>1, why garble persists. (b) DESIGN the compute-only
re-index fix (key the deep spine row's conv window to the spine/path0 ancestry, not the flat co-resident
layout). default-OFF flag, byte-identical off. (c) BUILD + measure speed (expect ~0 tax) + verify garble drops
(teacher-forced oracle-flip gate + live-SWE 13398 fr13_garble_watch). NO HBM copies.

## FIX DESIGN (2026-07-10, wf_9b24f818) — REALIZATION REPLICA, not a re-index
Re-index premise REFUTED: the deep-spine conv window is ALREADY spine-keyed (fr10_tree_conv_source_indices
= parent-chain ancestry, :3371; flat/co-resident source is diagnostic-only). The residual conv1d_out=9.77e-4
(1 bf16-ULP) from byte-exact pre_conv=0.0 is a KERNEL REALIZATION diff: OUR tree-conv (fused_tree_conv_taps_acc
bf16-tap MAC @fr13_tree_conv_fused.py:236-251 + triton_ex2_silu_bf16 @fr13_ex2_silu.py:17-23) produces a
different bf16 than native causal_conv1d_update on IDENTICAL input. Only num_accepted>=3 (deep-accept) carries
enough ULP to flip an argmax.

FIX = FR13_CONV_EX2_REPLICA (default OFF, sidecar for spawn worker): (a) align tap-MAC accumulation ORDER to
native's column order (native oldest..newest; reverse the width<=4 loop if it differs); (b) make
triton_ex2_silu_bf16 a bit-exact ex2.approx replica of native's silu (mul -x,log2(e) -> ex2.approx.f32 -> 1+ ->
div -> cvt.rn.bf16). KEEP bf16 taps (fp32-taps REGRESSES L0 to 0.0625). PURE COMPUTE, ZERO HBM (same kernel
launch, loop reorder) -> passes no-HBM rule. FR12_NATIVE_SPINE oracle (copies spine rows) = validation-only, BANNED as ship.

VALIDATION (cheap-first): (1) engagement assert (worker self-report, replica fired>0). (2) OFFLINE per-layer
ladder BOOT-FREE: capture spine-only conv inputs (pre_conv+conv_state+weights+bias) for clean layers L0/4/8/12/24/36/44,
iterate replica vs native causal_conv1d_update on IDENTICAL inputs until conv1d_out==0.0 RAW/int-view (NOT atol)
every clean layer. (3) teacher-forced oracle-flip gate vs FR12_NATIVE_SPINE. (4) live-SWE 13398 fr13_garble_watch
same-boot A/B. (5) speed derived_tps_gpu ON vs OFF ~0.
KEY RISK (#3): the L0->L63 GROWTH is dominated by the DIFFUSE GDN recurrent SCAN state-feed (reference_gdn_verify_sequential_dispatch),
which the conv fix does NOT touch. If perfect conv alignment does NOT collapse the flips -> route to tree-reshape,
NOT another per-op patch. So conv-replica is NECESSARY-maybe-not-SUFFICIENT. Design detail: FR13_CONV_FIX_DESIGN.md.

## Native kernel located + build plan (2026-07-10)
Native `_causal_conv1d_update_kernel` (Triton, readable) at
/tmp/lumo_vllm_main_audit/vllm-v0.22.0/.../mamba/ops/causal_conv1d.py (host copy; LIVE served =
v0.19.2rc1.dev134 — match THAT via the offline harness since bit-exact is codegen/SASS-sensitive, class-10).
Structure: loads width taps explicitly col0(oldest)..col3(newest) (width-4), then a fused fp32 MAC + ex2.approx
silu + bf16 store in ONE Triton program. Our replica target = fused_tree_conv_taps_acc (fr13_tree_conv_fused.py:236-251,
current order (((bias+p0)+p1)+p2)+p3, bf16 products cast f32) + triton_ex2_silu_bf16 (fr13_ex2_silu.py).
§6 OPTIMISM: conv1d_out is FIRST-NONZERO, GDN scan is M-invariant-as-op (downstream), so a bit-exact conv replica
should CASCADE the whole L0-GDN to 0 (feedback_fr12_subkernel_zero_gate) -> plausibly SUFFICIENT.

NEXT (autonomous build):
1. OFFLINE BIT-MATCH HARNESS (the cheap loop, NO model boot): a script run INSIDE the live vLLM image
   (docker run lumo-flywheel-vllm...v0.19, no server) that imports native causal_conv1d_update + our
   fused_tree_conv_taps_acc+triton_ex2_silu_bf16, feeds IDENTICAL random bf16 window/weights/bias/conv_state,
   compares int-view (RAW 0.0, NOT atol). Iterate our MAC order + silu until bit-exact vs native. Random inputs
   expose the order/rounding diff (no captured real inputs needed for MAC-order matching).
2. Wire FR13_CONV_EX2_REPLICA (default OFF, sidecar for spawn worker, byte-identical off) selecting the matched
   replica MAC/silu in fr13_tree_conv_fused.py + fr13_ex2_silu.py.
3. GPU GATES: engagement assert; per-layer ladder conv1d_out->0 within-boot; teacher-forced oracle-flip vs
   FR12_NATIVE_SPINE; live-SWE 13398 fr13_garble_watch same-boot A/B; speed derived_tps_gpu ~0.
HARD RULE reaffirmed: compute-only, NO HBM copy (loop reorder / same kernel launch). Do NOT promote taps to fp32
(regresses L0 to 0.0625). If conv1d_out->0 but e2e flips persist -> scan/tree-reshape, NOT another per-op patch.
