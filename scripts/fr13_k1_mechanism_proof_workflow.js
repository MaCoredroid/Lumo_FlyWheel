export const meta = {
  name: 'fr13-k1-mechanism-proof',
  description: 'K1 BAKE PROOF (user wants to bake K1; gated on this; run AFTER the N_PAD test settles). K1 theory is sound (carries the GDN state through native\'s exact bf16 store-reload = the deployment-decode realization) but the e2e 18->12 is a SINGLE cross-boot comparison within the ±9 autotune floor, and K1\'s OWN state-toward-native effect was never isolated. PROVE the mechanism SAME-BOOT (immune to cross-boot noise) with the proven int-view state gate scripts/fr13_gdn_scan_warp_gate.py: K1-ON carried GDN state vs native-packed-decode state (int-view, NEVER atol) MUST be < the OFF gap 0.0289 (= K1 moves the state TOWARD native) + accept/event neutral. PASS => bake K1 default-ON in the locked path; FAIL (K1 state NOT closer to native) => do NOT bake. Single GPU diagnostic. Adversarial verify.',
  phases: [
    { title: 'Gate' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next). Repo /home/mark/shared/lumoFlyWheel. GPU',
'SERIALIZED - run ONLY when free (after the N_PAD test wtuyrq24t). Pre-boot hygiene: source .venv;',
'recover_host_memory(); MemAvailable>=100GiB + docker ps empty. Teardown + recover after. boot ENFORCE_EAGER=1.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned 3dbe092e = 0.19.2rc1.dev134). int-view',
'NEVER atol (#10). PROVE non-vacuous before any number (#9): neg-control native_norm>0 AND ours_norm>0; flag',
'actually live in the worker.',
'',
'WHY (FR13_K1_BAKE_DECISION.md): K1 (bf16 b_h store-boundary, _gdn_node_step, FR13_SCAN_ALIGN=1 MODE=body) is',
'THEORY-sound (adopts native packed-decode\'s per-token bf16 store-reload of the carried state) but its e2e',
'18->12 de-cascaded is a SINGLE CROSS-BOOT comparison within the ±9 autotune floor (feedback_no_cross_boot_byte',
'_gate), and its OWN state-toward-native shift was never isolated (we have OFF-state-vs-native=0.0289 and full-',
'recompute=0.0, but NOT K1\'s partial). The int-view STATE gate is SAME-BOOT and immune to the cross-boot floor.',
'',
'YOUR JOB:',
'PHASE 1 (Gate, GPU): hygiene + the proven int-view state gate scripts/fr13_gdn_scan_warp_gate.py (hardened',
'  neg-control native_norm>0, the e428db3a fix). On the SAME fixed input, compare the carried GDN scan STATE h',
'  to the native-packed-decode reference state (scripts/fr13_native_packed_decode_ref.py, the d406fe2b ≥2-slot',
'  fixed ref), int-view:',
'  (A) K1-OFF (FR13_SCAN_ALIGN unset): expect int-view False, max_abs ~0.0289 (the banked OFF gap = neg-control',
'      powered + the baseline).',
'  (B) K1-ON (FR13_SCAN_ALIGN=1 MODE=body): max_abs vs native-packed. The TEST: is it < 0.0289 (K1 moved the',
'      state TOWARD native)?',
'  Prove non-vacuous: native_norm>0 AND ours_norm>0 on BOTH arms (cannot vacuum off a zeros ref); the SCAN_ALIGN',
'  constexpr actually live (served stream differs OFF vs ON, #10). ALSO capture accept/event with K1 ON from a',
'  served run (vs OFF ~3.15 / native 3.076) to confirm neutral. Teardown + recover.',
'PHASE 2 (Verdict). DECISION (bring to user, do NOT bake yourself): (BAKE) K1-ON state max_abs < OFF 0.0289',
'  (state moved toward native, monotone) AND accept/event neutral (~OFF within the cross-boot band) => K1 is',
'  the proven correct realization that reduces the kernel drift => recommend baking default-ON. (DO NOT BAKE)',
'  K1-ON state max_abs >= 0.0289 (NOT closer to native, or larger) OR accept craters => K1 is not doing what',
'  theory says => do not bake, report. Reward-hacks BANNED: K1 = OUR kernel adopting native\'s rounding',
'  (authorized); native = A/B reference ONLY (no served-path splice); NOT recompute (geometry change), NOT WY',
'  (parked), NOT bonus (rejected). Quote FR13_BUG_CLASS_PLAYBOOK (#9 vacuous, #10 int-view/codegen, #12 cross-',
'  boot).',
].join('\n');

phase('Gate');
const G_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['flagLive','negControlPowered','k1off_state_vs_native','k1on_state_vs_native','k1_moved_toward_native','k1_accept_per_event','ok','notes'],
  properties: {
    flagLive: { type: ['boolean','null'], description: 'FR13_SCAN_ALIGN=1 live in worker + constexpr actually taken (served diverges OFF vs ON)' },
    negControlPowered: { type: ['boolean','null'], description: 'native_norm>0 AND ours_norm>0 on both arms (int-view not vacuous off a zeros ref)' },
    k1off_state_vs_native: { type: ['string','null'], description: 'K1-OFF carried-state vs native-packed int-view max_abs (expect ~0.0289, the baseline/neg-control)' },
    k1on_state_vs_native: { type: ['string','null'], description: 'K1-ON carried-state vs native-packed int-view max_abs (the TEST)' },
    k1_moved_toward_native: { type: ['boolean','null'], description: 'is K1-ON max_abs < OFF 0.0289 (state moved TOWARD native)?' },
    k1_accept_per_event: { type: ['number','null'], description: 'accept/event with K1 ON (vs OFF ~3.15, native 3.076) - neutral?' },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const g = await agent(
  CTX + '\n\nTASK (Gate, GPU). Run the int-view state gate K1-OFF vs K1-ON each vs native-packed, PROVE non-'
  + 'vacuous (neg-control both-norms>0, flag live, int-view NOT atol). Report whether K1-ON state moved toward '
  + 'native (< 0.0289) + accept/event. Teardown + recover. Return the schema.',
  { label: 'k1-mechanism-gate', phase: 'Gate', schema: G_SCHEMA, model: 'opus' }
);

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','nonVacuous','mechanismProven','acceptNeutral','bakeRecommendation','rewardHackCheck','issues'],
  properties: {
    holds: { type: 'boolean' },
    nonVacuous: { type: 'string', description: 'neg-control both-norms>0 + flag-live + int-view-not-atol all proven?' },
    mechanismProven: { type: 'string', description: 'did K1-ON state max_abs drop below the OFF 0.0289 (moved toward native), int-view?' },
    acceptNeutral: { type: 'string', description: 'accept/event with K1 ON neutral vs OFF/native?' },
    bakeRecommendation: { type: ['boolean','string','null'], description: 'BAKE (state toward native + accept neutral) or DO NOT BAKE (not closer / accept craters)? For the user.' },
    rewardHackCheck: { type: 'string', description: 'K1 = our kernel adopting native rounding (authorized); native = ref only; not recompute/WY/bonus' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(g) + '. Default holds=false if neg-control not powered '
  + '(both norms must be >0 — a zeros native-ref makes the int-view vacuous, the exact #9 trap that broke this '
  + 'gate before), flag not live, int-view used atol, or accept from streamed logprobs. Conclude honestly: did '
  + 'K1 move the state toward native (< 0.0289) at neutral accept => bake-recommended, or not => do-not-bake. '
  + 'No bake decision here (user call); no reward-hack.',
  { label: 'verify-k1-mechanism', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { g, v };
