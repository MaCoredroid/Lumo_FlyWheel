export const meta = {
  name: 'fr13-temp06-drift-gate',
  description: 'USER (2026-06-15): the lossless math still hinges on the 13% drift on the large sample, but that was measured at TEMP 0 (greedy ARGMAX: served argmax != recurrent-decode argmax). Deployment is TEMP 0.6 (sampling), where lossless is DISTRIBUTIONAL not argmax. Design a PROPER temp-0.6 drift gate (the deployment-relevant lossless measure) comparing cat9-spec vs its OWN no-spec recurrent decode, with native-spec-vs-native-no-spec as the FLOOR, THEN decide if the lossless levers (L1 margin-aware commit / L4 boundary fp32) are worth doing. CONFIRM the user: argmax=temp0 is CORRECT; the big-denom 13.55%/13.99% is a temp-0 argmax point-measure that can PASS while temp-0.6 sampling drifts (a sub-argmax distribution shift changes the sampled token even when the top token matches). Design the proper instrument: per-position distributional TV/KL at temp 0.6 between the spec verify-forward dist (q) and the no-spec recurrent-decode dist (p), AND/OR the realized-sample bag-TV with a multi-sample native floor (reuse the banked bag-TV machinery FR13_DRIFT_TRACKER_DESIGN / BAG_TV_FLOOR=0.0593 / fr13_drift_tracker.py, the multi-sample p95-of-C(N,2) native floor) - pick the soundest. Recurrent-oracle frame (NEVER chunked/streamed/serial/backend-name). First-pass estimate on banked data IF the verify dist or multi-seed temp-0.6 streams exist (note: the big-denom oracle recorded the recurrent-decode top-20 log_softmax = p, but the SERVED stream was temp-0 greedy + the spec VERIFY dist q may NOT be banked -> say what is computable now vs needs a fresh temp-0.6 GPU capture). Decision rule: cat9 temp-0.6 drift WITHIN native temp-0.6 floor -> lossless met at deployment temp, NO levers; cat9 ABOVE floor -> levers warranted. CPU read-only (speed campaign holds the GPU - design + banked-data estimate, the fresh temp-0.6 capture queues after). Output FR13_TEMP06_DRIFT_GATE.md.',
  phases: [
    { title: 'Design' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10. Repo /home/mark/shared/lumoFlyWheel. CPU read-only (a speed-campaign GPU run is live -',
'do NOT boot; design + estimate on BANKED data, the fresh temp-0.6 GPU capture runs AFTER). Read our code + git',
'history + the banked binds + the big-denom rescore data + vLLM source via scripts/vllm_src.sh (pinned 3dbe092e,',
'NEVER /tmp). Write ONLY FR13_TEMP06_DRIFT_GATE.md (+ a CPU estimate script if computable). Pathspec commit.',
'',
'THE FRAMING (confirm + sharpen): TEMP 0 (greedy) -> served token = ARGMAX of the spec verify forward; "flip" =',
'served argmax != recurrent-decode argmax = a POINT measure (fires only when the TOP token flips). The big-denom',
'(cat9 13.548% [12.85,14.28] ~= native 13.985% [13.27,14.73], CIs overlap, recurrent oracle, spec-vs-non-spec',
'CONFIRMED) is exactly this temp-0 argmax measure. TEMP 0.6 (deployment) -> served token = SAMPLED via rejection',
'sampling; lossless is DISTRIBUTIONAL (the spec output distribution == the no-spec target distribution, the',
'rejection-sampling losslessness property). A position can be argmax-IDENTICAL yet have a drifted sub-argmax',
'distribution that changes which token is SAMPLED at 0.6 -> the temp-0 argmax gate can PASS while temp-0.6',
'sampling drifts. So the deployment-binding lossless gate is the temp-0.6 distributional one, and it decides the',
'lossless levers.',
'',
'COMPARE TARGET (feedback_fr13_lossless_compare_target, unchanged): US (cat9-spec) vs native-E5-spec, EACH vs',
'its OWN no-spec RECURRENT decode oracle (fr13_recurrent_decode_oracle, FR12_NO_SPECULATIVE_CONFIG=1, FLASH_ATTN,',
'recurrent single-token roll), NEVER chunked-prefill / streamed-logprobs / serial-torch / backend-name. native-',
'spec-vs-native-no-spec = the FLOOR; cat9 is lossless-at-0.6 iff its temp-0.6 drift is WITHIN that floor. int-',
'view never atol for any byte check; TV/KL for the distributional part.',
'',
'BANKED temp-0.6 machinery (build ON, do not reinvent): FR13_DRIFT_TRACKER_DESIGN (multi-sample floor = N=6-8',
'native runs, floor = p95 of C(N,2) native-vs-native draws, NOT single-draw; forced-decode logit-KL/TV isolates',
'verify-drift from path non-det; reusable fr13_drift_tracker.py = K-seed wrapper -> scalar D = excess drift over',
'floor), BAG_TV_FLOOR=0.0593 (the bag-TV the lossless gate historically used at temp 0.6), the per-token argmax',
'probe (fr13_gold_margin_probe.py - records the verify-forward margin/topk = q) + the recurrent oracle (records',
'the decode top-20 log_softmax = p), reference_scalar_metric_per_token_blindspot (scalar bag-TV alone hid a real',
'defect -> pair with a per-position instrument). The big-denom rescore (output/fr13_bigdenom_rescore/rescore_',
'{cat9,native}.json) recorded the RECURRENT-decode top-20 log_softmax (p) per served position, but the served',
'stream was TEMP-0 GREEDY and the spec VERIFY dist (q) may not be in it - establish what is computable NOW.',
'',
'YOUR JOB:',
'1. CONFIRM/correct the framing (argmax=temp0; the 13% is a temp-0 point-measure; temp-0.6 needs distributional).',
'2. DESIGN the PROPER temp-0.6 drift gate - pick the soundest of: (a) per-position distributional TV/KL at temp',
'   0.6 between the spec verify-forward dist q=softmax(verify_logits/0.6) and the no-spec recurrent-decode dist',
'   p=softmax(decode_logits/0.6) (NOTE: from a recorded top-20 log_softmax you CAN recover softmax(.../0.6) since',
'   the per-position additive constant cancels - so temp-0.6 TV is computable IF BOTH q and p top-20 are banked);',
'   (b) realized-sample BAG-TV with the multi-sample native floor (multi-seed temp-0.6 served streams, bag-TV vs',
'   the p95-of-C(N,2) native floor) - needs streams not logits. State which is more deployment-faithful + why,',
'   and the native-spec FLOOR for it. Pair the scalar with a per-position view (blindspot).',
'3. FIRST-PASS ESTIMATE on banked data IF computable (e.g. if a banked capture has BOTH q and p top-20: recompute',
'   the temp-0.6 TV cat9 vs native on the temp-0 trajectory as an early read, clearly labelled an ESTIMATE on the',
'   temp-0 trajectory not the temp-0.6 trajectory). RUN it (CPU). If q is not banked, say so + skip to the plan.',
'4. The RIGOROUS GPU measurement plan: a fresh TEMP-0.6 paired capture recording BOTH the spec verify dist q AND',
'   the recurrent decode dist p per position (or the multi-seed bag-TV streams), cat9 + native, recurrent-oracle',
'   frame, the multi-sample native floor - queues AFTER the speed campaign frees the GPU.',
'5. DECISION RULE for the levers: cat9 temp-0.6 drift WITHIN the native temp-0.6 floor -> lossless met at',
'   deployment temp, do NOT do the levers; cat9 ABOVE the floor by margin M -> the levers (L1 margin-aware',
'   commit-at-near-tie / L4 last-stage boundary fp32) are warranted, with the predicted M-reduction.',
'',
'DELIVERABLE: FR13_TEMP06_DRIFT_GATE.md = the framing confirmation, the proper temp-0.6 drift gate design, the',
'first-pass banked estimate (or why not computable), the rigorous GPU capture plan, the lever decision rule.',
'Distinguish MEASURED/COMPUTED from INFERRED. Quote FR13_BUG_CLASS_PLAYBOOK (#9 vacuous, #11 oracle identity).',
'NOT chunked/streamed/serial as the oracle; WY parked; no reward-hack. research-before-deadend.',
].join('\n');

phase('Design');
const D_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['framingConfirmed','temp06GateDesign','bankedEstimate','gpuCapturePlan','leverDecisionRule','committed','notes'],
  properties: {
    framingConfirmed: { type: 'string', description: 'confirm argmax=temp0 + the 13% is a temp-0 point-measure that can pass while temp-0.6 sampling drifts (the distributional gap)' },
    temp06GateDesign: { type: 'string', description: 'the PROPER temp-0.6 drift gate: distributional TV/KL (q=softmax(verify/0.6) vs p=softmax(decode/0.6)) and/or realized bag-TV with the multi-sample native floor - which is soundest + the native-spec FLOOR + a per-position view paired with the scalar' },
    bankedEstimate: { type: 'string', description: 'first-pass temp-0.6 drift estimate on banked data (if BOTH q and p top-20 banked: the computed cat9-vs-native temp-0.6 TV on the temp-0 trajectory, labelled an estimate, with run numbers) OR why not computable (q not banked) -> needs the GPU capture' },
    gpuCapturePlan: { type: 'string', description: 'the rigorous fresh temp-0.6 paired capture (both q and p per position, or multi-seed bag-TV streams), cat9 + native, recurrent-oracle frame, multi-sample floor - queued after the speed campaign' },
    leverDecisionRule: { type: 'string', description: 'cat9 within native temp-0.6 floor -> NO levers; above by margin M -> L1/L4 warranted with predicted M-reduction' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const d = await agent(
  CTX + '\n\nTASK (Design, CPU, no GPU). Confirm the framing, design the proper temp-0.6 drift gate, RUN the '
  + 'first-pass banked estimate if computable, give the GPU capture plan + the lever decision rule. Write FR13_'
  + 'TEMP06_DRIFT_GATE.md, commit pathspec. Return the schema.',
  { label: 'temp06-drift-gate', phase: 'Design', schema: D_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','framingSound','gateProper','estimateReal','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    framingSound: { type: 'string', description: 'is the argmax=temp0 / temp-0.6-is-distributional framing correct (argmax-identical can still distributionally drift at 0.6), not a hand-wave?' },
    gateProper: { type: 'string', description: 'is the temp-0.6 gate a PROPER distributional/sample measure (not argmax) with the native-spec FLOOR as the bar + the recurrent-oracle frame + a per-position view (not a blind scalar)?' },
    estimateReal: { type: 'string', description: 'if a banked estimate is claimed, was it actually RUN (spot-check the temp-0.6 softmax(log_softmax/0.6) recovery + the TV numbers) - or honestly stated not-computable (q not banked)?' },
    recommendation: { type: 'string', description: 'single: does the temp-0.6 gate (estimate or planned) indicate the levers are needed or not, and the GPU capture to confirm. No close/pass-fail beyond the gate.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(d) + '. Default holds=false if the framing is wrong (e.g. '
  + 'claims temp-0 argmax == temp-0.6 lossless), the gate is still argmax/a blind scalar rather than a proper '
  + 'distributional/bag-TV measure with the NATIVE-spec floor + recurrent-oracle frame, a claimed banked estimate '
  + 'was not actually run or mis-recovers the temp-0.6 softmax (softmax(log_softmax/0.6) constant-cancel must be '
  + 'correct + restricted to top-20 mass with the truncation noted), or the lever decision is asserted without the '
  + 'floor comparison. No reward-hack (recurrent-only oracle, WY parked). No close/pass-fail.',
  { label: 'verify-temp06-drift-gate', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { d, v };
