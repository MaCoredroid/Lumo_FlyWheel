export const meta = {
  name: 'fr13-conv-crossevent-investigate',
  description: 'PIVOT (replay SSM durable-state NOT the back-loaded carrier - flat/non-growing + L0 4.17 = harness ring-gather artifact). The NEXT lead, named by prior evidence: the CONV cross-event path. FR13_DRIFT_LOCALIZE_BIND already pinned "conv prior-window READ as the carrier" (conv1d_out 18.375 at num_accepted>1) with h0/SSM BYTE-EXACT; sglang #25587 corroborates conv-state corruption post-partial-accept (~100 tok = back-loading). Conv fixes (c0b53f5d) landed AFTER that capture. CPU read-only: (1) is conv STILL a live cross-event carrier in the CURRENT locked build? (2) design the conv-state cross-event A/B reusing the PROVEN replay-durable-AB harness. Adversarial verify.',
  phases: [
    { title: 'Investigate' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (a GPU/CPU',
'workflow may run concurrently; do NOT edit code; write ONLY FR13_CONV_CROSSEVENT_INVESTIGATE.md). Pathspec',
'commits only.',
'',
'WHY THIS (the pivot): the replay-durable-state A/B (w2vaqcsmx) just showed the SSM recurrent durable-state is',
'NOT the back-loaded carrier of the 21 cat9 flips: per-event L0 max_abs is FLAT/slightly-decreasing (slope',
'-0.011/event), NOT growing; and the L0 4.17 is itself a harness ring-gather artifact (a real 4.17 in the',
'committed durable state would garbage the next event, but serving is coherent at 21 small flips). The honest',
'SSM divergence is small (~0.05-0.2). This MATCHES the earlier finding (project_fr13_conv_priorwindow_root /',
'FR13_DRIFT_LOCALIZE_BIND): "the conv prior-window READ is the carrier" - conv1d_out diverges 18.375 at',
'num_accepted>1 while h0_state_in (SSM recurrent) was BYTE-EXACT (recurrent-drift REFUTED). External',
'corroboration sglang #25587: conv-state corruption after partial accept, diverges after ~100 tokens (=',
'BACK-LOADING, the 21-flip fingerprint norm-mean 0.696). So the CONV cross-event path - the sliding-window',
'conv state handed between verify events via the conv-fused replay (FR13_TREE_CONV_FUSED=1, baked ALWAYS-ON) -',
'is the better-supported carrier than the SSM recurrent state.',
'',
'KEY UNCERTAINTY: conv prior-window FIXES landed AFTER the 18.375 capture (e.g. c0b53f5d 06-10, the wiring fix',
'at fr10_phase4_patch_vllm_tree_gdn.py ~797-818, gather_committed_path_conv_prior / FR13_CONV_COMMITTED_PATH).',
'So the question is whether conv STILL diverges from native in the CURRENT locked build, or whether the fixes',
'closed it and the 21-flip carrier is elsewhere. Do NOT assume conv is still broken - re-verify from the',
'current code + the post-fix evidence.',
'',
'YOUR JOB (read-only investigation + A/B design):',
'1. CURRENT-BUILD CONV STATUS: read the conv-fused tree path (src/lumo_flywheel_serving/fr13_tree_conv_fused.py,',
'   the gather_committed_path_conv_prior / prepare_committed_path_conv_rows / replay_conv_state_linear_remap',
'   helpers, + the patcher wiring ~L779-820/L4824-4835) AND native causal_conv1d_update conv-state handling.',
'   Trace the CONV-STATE cross-event handoff: at commit, what conv sliding-window state is written for the next',
'   event, and how does our conv-fused replay compute it vs native causal_conv1d_update? Read FR13_DRIFT_',
'   LOCALIZE_BIND + FR13_CONVFIX_AB_BIND + the c0b53f5d fix + FR13_S1S2S3_DISCRIMINATE_BIND: what was the',
'   conv1d_out 18.375 root (wrong bank-row/cols at num_accepted>1), what did the fix change, and is there',
'   RESIDUAL conv divergence in the current build (post-fix evidence, or unverified)?',
'2. BACK-LOADING FIT: does the conv-state cross-event hypothesis fit the 21-flip back-loading (norm-mean 0.696)',
'   + the near-disjoint-from-native boundary set + sglang #25587 (~100-token divergence) BETTER than the',
'   (refuted) SSM-recurrent? Is the conv-state the state that ACCUMULATES across events (unlike the SSM durable',
'   state which the A/B reset each event)?',
'3. DESIGN THE CONV-STATE CROSS-EVENT A/B (reuse the PROVEN replay-durable-AB harness, w2vaqcsmx /',
'   _fr13_replay_durable_ab at patcher ~L6475-6602): an observe-only per-event A/B comparing OUR conv-state',
'   handoff vs native causal_conv1d_update conv-state over the SAME accepted chain from the SAME cloned conv',
'   prior-window. CRITICAL design constraints learned the hard way: (a) KERNEL-VALID geometry - the conv/scan',
'   SUBOP A/B device-asserted 5x because the reduced-row M5/M1 tree-slice geometry over-ran causal_conv1d_',
'   update bank cols (state_len=width-1+(seqlen-1) > committed width-1 window). The replay-durable-AB DODGED it',
'   with a LINEAR B=1 varlen chain + NO ssm_state_indices/num_accepted_tokens (non-spec path). Design the',
'   conv A/B the SAME way: native causal_conv1d in a non-spec/prefill-style call over the linear accepted',
'   chain so state_len fits - PREVENT the assert, do not catch (device assert = unrecoverable context poison).',
'   (b) observe-only cloned state, served bank untouched. (c) RECORD relative error + state-norm (the replay',
'   A/B lacked these = could not disambiguate gross-vs-tiny; fix that here). (d) sidecar env + loud stage',
'   markers + eager-only. (e) the harness ring-gather column convention MUST match the conv replay kernel',
'   exactly (the replay A/B 4.17 artifact came from a ring-gather mismatch).',
'4. CRITICAL DISCRIMINATOR: the A/B must distinguish (i) conv-state GROWS-across-events (the back-loaded',
'   carrier) from (ii) flat per-event (not the carrier, like SSM), AND (iii) a harness artifact (record',
'   relative error + align the gather convention). Specify the GROWS-across-events test directly (does the',
'   conv-state our-vs-native gap accumulate when our-conv-state is fed forward as the next h0).',
'',
'Be SKEPTICAL: the conv prior-window may already be FIXED (then conv is NOT the carrier and the pivot goes',
'elsewhere - say so). Reward-hacks BANNED (observe-only; align OUR conv kernel if it diverges, do NOT splice in',
'native). Quote FR13_BUG_CLASS_PLAYBOOK rows (#12 cross-event, #10 codegen-identity, #9 silent/vacuous).',
'Write FR13_CONV_CROSSEVENT_INVESTIGATE.md, commit pathspec.'
].join('\n');

phase('Investigate');
const I_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['convStateHandoff','currentBuildStatus','backLoadingFit','abDesign','assertPrevention','growsAcrossEventsTest','committed','notes'],
  properties: {
    convStateHandoff: { type: 'string', description: 'how OUR conv-fused replay computes the next-event conv sliding-window state vs native causal_conv1d_update, with file:line' },
    currentBuildStatus: { type: 'string', description: 'is conv STILL a live cross-event divergence in the CURRENT locked build (post c0b53f5d fix), or closed? grounded in the fix + post-fix evidence' },
    backLoadingFit: { type: 'string', description: 'does conv-state fit the 21-flip back-loading (0.696) + sglang #25587 ~100-tok better than the refuted SSM-recurrent? is conv-state the accumulating state?' },
    abDesign: { type: 'string', description: 'the conv-state cross-event A/B design reusing the replay-durable-AB harness (observe-only, cloned, linear-chain, relative-error+state-norm, sidecar, eager)' },
    assertPrevention: { type: 'string', description: 'how the native causal_conv1d call is kernel-valid (linear non-spec geometry) to PREVENT the reduced-row assert that killed the conv/scan SUBOP A/B 5x' },
    growsAcrossEventsTest: { type: 'string', description: 'the explicit GROWS-across-events test (conv-state gap accumulates when fed forward) to distinguish carrier vs flat vs artifact' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const inv = await agent(
  CTX + '\n\nTASK (Investigate, no GPU, read-only). Do steps 1-4. Write FR13_CONV_CROSSEVENT_INVESTIGATE.md, '
  + 'commit pathspec. Return the schema. If conv is already FIXED in the current build (not a live carrier), '
  + 'say so plainly with the evidence and name where the pivot should go instead.',
  { label: 'investigate-conv-crossevent', phase: 'Investigate', schema: I_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','convStillLive','abDesignSound','assertPreventionGrounded','backLoadingFitHonest','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    convStillLive: { type: 'string', description: 'grounded: is conv a live cross-event carrier in the current build, or already fixed? (do not overstate - conv-priorwindow was a prior overstated-then-fixed lead)' },
    abDesignSound: { type: 'string', description: 'does the A/B design avoid the 5x assert (kernel-valid linear geometry), record relative-error, and match the gather convention (the replay 4.17 artifact lesson)?' },
    assertPreventionGrounded: { type: 'string', description: 'is the assert-prevention grounded in the native causal_conv1d geometry heuristics, not asserted?' },
    backLoadingFitHonest: { type: 'string', description: 'is the conv-back-loading-fit honest or over-claimed (the replay was also "triply reinforced" and still failed)?' },
    recommendation: { type: 'string', description: 'single recommendation: run the conv-state A/B (gated how) or conv is fixed → pivot elsewhere (where). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(inv) + '. Default holds=false if "conv is still the '
  + 'carrier" is asserted rather than grounded in the current post-fix code (conv-priorwindow was overstated '
  + 'before), or if the A/B design would hit the reduced-row assert / lacks relative-error / mismatches the '
  + 'gather convention. The replay lead was "triply reinforced" and still failed - hold conv to the same bar. '
  + 'No close/pass-fail; no reward-hack.',
  { label: 'verify-conv-crossevent', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { inv, v };
