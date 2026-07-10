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
