export const meta = {
  name: 'fr13-scan-align-rerun',
  description: 'THE DECISIVE GPU re-run (instruments now FIXED + hardened): execute FR13_SCAN_ALIGN_RERUN_PLAN.md. (1) Fixed STATE int-view gate (e428db3a: durable GDN state h, explicit native_norm>0+ours_norm>0 neg-control) → the GENUINE OFF/body/recompute scan-STATE-vs-native-packed gap (the carrier measurement never yet obtained). (2) Boot locked cat9 spec server, capture OFF + recompute streams (verify recompute ENGAGES, #9 fail-loud). (3) The binding flips: rescore OFF + recompute + native-E5 vs the SAME oracle frame (Mechanism B = fr13_recurrent_decode_oracle deployment-correct recurrent; Mechanism A = --mode tree_mtp chunked for continuity-with-21). Prove EVERY instrument non-vacuous BEFORE trusting any number. Discriminator: recompute STATE int-view 0.0/floor + flips_after<flips_before toward native 3 + lossless gate = the win. Adversarial verify.',
  phases: [
    { title: 'Verify' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene:',
'source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps empty. Teardown trap: docker',
'rm -f the container + recover_host_memory between every boot. boot ENFORCE_EAGER=1.',
'',
'GROUNDING RULE (user): read vLLM source via `scripts/vllm_src.sh <relpath>` (0.19.2rc1.dev134), NEVER a',
'/tmp cache. COMPARE TARGET (user): lossless = US(cat9) vs native-E5 each-vs-its-own-no-spec-oracle; native-E5',
'~3 = THE BAR; US 21 = the gap to close. int-view NEVER atol. oracle = no-spec NOT prefill.',
'',
'THE PLAN: follow FR13_SCAN_ALIGN_RERUN_PLAN.md (committed) EXACTLY. The two binding instruments were just',
'FIXED + HARDENED (d406fe2b + e428db3a, 24 CPU tests): (I1) the int-view STATE gate scripts/fr13_gdn_scan_warp_gate.py',
'now compares the durable GDN STATE h (real ssm_state slot>=1, not the state_idx==0 zeros short-circuit) and',
'negative_control_powered requires int-view-False AND native(ref)_norm>0 AND ours_norm>0 (cannot re-vacuum off',
'a zeros ref). (I2) the no-spec oracle: per-request non_mtp CRASHES (EagleProposer.positions, DECODE path);',
'use the WORKING mechanisms - A = --mode tree_mtp teacher-force (max_tokens=1 re-prefill, the ORIGINAL 21',
'mechanism, CHUNKED ~9x frame) AND B = scripts/fr13_recurrent_decode_oracle.py rescore (in-process, NO spec',
'config = EagleProposer never built, forced single-step recurrent decode = deployment-correct frame).',
'',
'NON-VACUITY IS MANDATORY (the recurring trap this session): prove EACH instrument non-vacuous BEFORE trusting',
'its number - the STATE gate negative_control_powered MUST be True (the hardened check); the oracle engagement',
'(_forward_core_decode_non_spec monkeypatch counter for B; spec_metrics_delta~0 for A) MUST be observed; the',
'recompute arm MUST be confirmed engaged at the EngineCore (/proc/<pid>/environ or the sitecustomize bridge).',
'A number from a vacuous/disengaged instrument is WORSE than no number - fail loud (bug-class #9).',
'',
'YOUR JOB:',
'PHASE 1 (Verify, GPU - multiple boots allowed per the plan, hygiene between each):',
'  STEP 1 - STATE int-view gate: run `python3 scripts/fr13_gdn_scan_warp_gate.py --payload <gdn scan capture>',
'  --out <json>` (capture a fresh paired GDN-scan payload from a quick cat9 eager forward if none banked; PIN',
'  the prompt). ASSERT negative_control_powered==True (hardened). READ off_arm_spine_state_vs_native (OUR',
'  deployed-OFF scan STATE vs native-packed STATE = the GENUINE carrier gap, expect non-zero), and the',
'  body/recompute arms STATE-vs-native (does recompute reach int-view 0.0 or the bf16 floor vs native-packed',
'  STATE?). This is the kernel-level question: does the alignment make our scan STATE bit-exact to native.',
'  STEP 2 - streams: boot locked cat9 spec server (fr13_launch_locked.sh) ENFORCE_EAGER=1, assert tok/draft==9',
'  + within-boot det [T,T,T,T]; capture the OFF served streams; then boot with FR13_SCAN_ALIGN=1',
'  FR13_SCAN_ALIGN_MODE=recompute (CONFIRM it reaches the EngineCore - sitecustomize/sidecar, fail-loud if',
'  not), capture the recompute served streams. Same prompts_swe4, seed 1313, greedy.',
'  STEP 3 - binding flips (re-score-both-arms discipline, FR13_ORACLE_FRAME): for the SAME oracle frame,',
'  rescore OFF, recompute, AND native-E5 streams → per-token argmax clear-margin flips vs the oracle. Use',
'  Mechanism B (fr13_recurrent_decode_oracle rescore, deployment-correct recurrent) as PRIMARY + Mechanism A',
'  (tree_mtp chunked) for continuity-with-the-original-21. Report flips_before(OFF), flips_after(recompute),',
'  flips_native(E5) for each frame. Teardown + recover after every boot; never leak.',
'PHASE 2 (Verdict). DISCRIMINATOR: (a) recompute STATE int-view 0.0/within-floor vs native-packed STATE AND',
'(b) flips_after(recompute) < flips_before(OFF), dropping toward flips_native(E5)~3, AND (c) lossless gate',
'(det [T,T,T,T], regular-decode pristine, accept/event not collapsed) = THE LOSSLESS WIN (report to user, bake',
'= user call → then speed → B=1). If recompute STATE int-view 0.0 but flips do NOT drop = scan-STATE was not',
'the e2e carrier (re-open, NOT cleared - the deployment frame matters). If flips drop partway = quantify',
'op-order vs co-residency. If the OFF scan-STATE-vs-native gap is ~0 (no real divergence) = the scan was never',
'the carrier (re-open). Reward-hacks BANNED (native = A/B oracle only; the recompute/seams are OUR kernel,',
'committed zero-diff). Quote FR13_BUG_CLASS_PLAYBOOK rows (#9 silent/vacuous, #10 codegen-identity, #12).'
].join('\n');

phase('Verify');
const VR_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['negControlPowered_hardened','off_state_vs_native','body_state_vs_native','recompute_state_vs_native','recompute_engaged','flips_before_off','flips_after_recompute','flips_native_e2e','oracleFrame','oracleEngaged','within_boot_det','accept_per_event','ok','notes'],
  properties: {
    negControlPowered_hardened: { type: ['boolean','null'], description: 'STATE gate negative_control_powered==True with the hardened native_norm>0+ours_norm>0 check?' },
    off_state_vs_native: { type: ['string','null'], description: 'OFF deployed scan STATE vs native-packed STATE: int-view eq? max_abs/rel_err (the GENUINE carrier gap, expect non-zero)' },
    body_state_vs_native: { type: ['string','null'] },
    recompute_state_vs_native: { type: ['string','null'], description: 'recompute STATE vs native-packed STATE: int-view 0.0 / within-floor?' },
    recompute_engaged: { type: ['boolean','null'], description: 'recompute confirmed reaching the EngineCore (not silent-OFF)?' },
    flips_before_off: { type: ['integer','string','null'], description: 'OFF flips vs the oracle (~21 expected for continuity)' },
    flips_after_recompute: { type: ['integer','string','null'], description: 'recompute flips vs the SAME oracle - THE BINDING NUMBER (target → native ~3)' },
    flips_native_e2e: { type: ['integer','string','null'], description: 'native-E5 flips vs the SAME oracle (~3, the BAR)' },
    oracleFrame: { type: 'string', description: 'which oracle frame(s) measured: B recurrent (primary) and/or A chunked (continuity)' },
    oracleEngaged: { type: ['boolean','null'], description: 'oracle proven non-vacuous (B: _forward_core_decode_non_spec counter; A: spec_metrics~0)?' },
    within_boot_det: { type: 'string' },
    accept_per_event: { type: ['number','null'] },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const vr = await agent(
  CTX + '\n\nTASK (Verify, GPU). Execute PHASE 1 steps 1-3 per FR13_SCAN_ALIGN_RERUN_PLAN.md. Prove EVERY '
  + 'instrument non-vacuous (hardened neg-control True, oracle engaged, recompute engaged) BEFORE any number. '
  + 'Report the genuine STATE gaps + flips_before/after/native per oracle frame. Teardown + recover. Return the schema.',
  { label: 'rerun-verify', phase: 'Verify', schema: VR_SCHEMA, model: 'opus' }
);

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','instrumentsNonVacuous','scanIsCarrier','recomputeStateBitExact','flipsDropped','isLosslessWin','flips_summary','nextAction','rewardHackCheck','issues'],
  properties: {
    holds: { type: 'boolean' },
    instrumentsNonVacuous: { type: 'string', description: 'were ALL instruments proven non-vacuous (hardened neg-control + oracle engaged + recompute engaged)? if not, holds=false' },
    scanIsCarrier: { type: ['boolean','null'], description: 'is the OFF scan-STATE genuinely != native-packed STATE (the scan diverges = carrier real)?' },
    recomputeStateBitExact: { type: ['boolean','null'], description: 'does recompute reach STATE int-view 0.0/floor vs native-packed?' },
    flipsDropped: { type: ['boolean','string','null'], description: 'did flips_after drop below flips_before toward native ~3?' },
    isLosslessWin: { type: ['boolean','null'], description: 'STATE bit-exact + flips→~3 + lossless gate = the win?' },
    flips_summary: { type: 'string', description: 'flips_before / flips_after / flips_native per oracle frame' },
    nextAction: { type: 'string', description: 'if win: report to user, bake=user call → speed → B=1. if STATE-0-but-flips-flat: scan not e2e carrier, re-open. if OFF-gap~0: scan never carrier, re-open. No close decision here.' },
    rewardHackCheck: { type: 'string', description: 'native oracle-only, committed kernel zero-diff, no splice' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(vr) + '. Default holds=false if ANY instrument was '
  + 'vacuous/disengaged (hardened neg-control not True / oracle not engaged / recompute not engaged / number '
  + 'from streamed top_logprobs not the gold-margin oracle). The session burned 3 vacuous instruments - hold '
  + 'this to PROVEN non-vacuity. Conclude: is the scan the carrier (OFF gap real), does recompute fix it '
  + '(STATE 0.0 + flips→3), or re-open. If a genuine lossless WIN, report for the user (bake = their call). '
  + 'No close decision; no reward-hack.',
  { label: 'rerun-verdict', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { vr, v };
