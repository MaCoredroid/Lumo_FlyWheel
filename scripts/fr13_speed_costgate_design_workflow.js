export const meta = {
  name: 'fr13-speed-costgate-design',
  description: 'USER (2026-06-15): lossless-vs-native CONFIRMED at B=1 deployable scale (big-denom cat9 13.55% ~= native 13.99%, CIs overlap; spec-vs-nonspec confirmed) -> PIVOT to SPEED. Bar = cat9 B=1 decode-TPS STRICTLY > native E5, lossless held as a per-change gate. This is the DISCIPLINED first step (cost-gate BEFORE building, feedback_speed_is_the_goal_cost_gate): read the prior speed verdicts (the tree was ~2.3x slower = 1.43x more forwards x 1.63x more time/forward; WY PARKED; accept-only gate-4 FAILED) + the current locked pipeline + the non-refuted accept-side lever (L3 confidence-gated root sibling, 62% of rejects are step-0), then (1) design the EXACT clean B=1 measurement protocol (decode_seconds basis NOT TPS-div-accept which is BANNED, metrics OFF, per-request not aggregate, the prelaunch host-mem protocol), (2) honest COST-GATE: is there a plausibly-cheap path to cat9 > native at B=1, or is the per-forward GDN HBM tax structural + the accept ceiling ~3.18 = NO cheap path => STOP+report, (3) ranked lever menu, (4) STOP-or-GO. NO hand-rolled per-forward TPS decompositions as measured facts (feedback_dont_handroll_speed - caught twice). CPU read-only (GPU free, do NOT boot - measurement runs after, under the prelaunch protocol). Output FR13_SPEED_COSTGATE.md.',
  phases: [
    { title: 'CostGateDesign' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s LPDDR5X, B=1 decode HBM-bandwidth-bound). Repo',
'/home/mark/shared/lumoFlyWheel. CPU read-only (GPU free but do NOT boot - the GPU measurement runs AFTER this',
'design, under the prelaunch protocol: recover_host_memory + free -g ~100GiB + docker ps empty). Read our code +',
'git history + the banked speed verdicts + vLLM source via scripts/vllm_src.sh. Write ONLY FR13_SPEED_COSTGATE.md.',
'Pathspec commit.',
'',
'CONTEXT: lossless-vs-native is CONFIRMED at B=1 deployable scale (big-denom: cat9 clear-margin flip rate 13.55%',
'[12.85,14.28] ~= native 13.99% [13.27,14.73], CIs OVERLAP cat9 LOWER, ~8.7k positions, recurrent oracle, spec-',
'vs-nonspec confirmed code-read). cat9 accept/event ~3.18 ~= native E5 3.076 (depth-matched, at parity). So',
'lossless is met; the GOAL is now SPEED: cat9 B=1 decode-TPS STRICTLY > native E5, with lossless held as a',
'per-change gate (same-seed byte-identical streams greedy + accept/event unchanged + regular-decode pristine).',
'',
'PRIOR SPEED WORK (build ON, do NOT re-derive; correct any stale number):',
'- FR13_WHY_SLOWER_VERDICT (wacoxe6i2, verify holds): tree ~2.336x SLOWER = 1.432x more forwards x 1.632x more',
'  time-per-forward (basis = decode_seconds raw counter, NOT wall, NOT TPS/accept which is BANNED). Per-forward',
'  tax dominant adder = GDN tree-scan per-node ~9x state r+w HBM amplification (per-node S ~3.0 MiB/node/layer).',
'  REMOVABLE only by a kernel rewrite (WY one-pass / accept-only state commit). FA2-fork/TREE_ATTN ~time-neutral',
'  (attention hidden behind the bandwidth floor). NOTE: that measurement PRE-DATES the in_proj_ba bake + the',
'  locked pipeline + the lossless fixes (accept/event was contaminated ~2.0 then, now ~3.18) - so the gap may',
'  have CHANGED; treat 2.3x as a stale prior, re-measure.',
'- WY one-pass kernel = PARKED (user firm call, do NOT revive) - the per-node state HBM tax it would remove is',
'  real but WY-as-built is lossy (accept 1.199) + only fixes the secondary GDN-scan carrier.',
'- ACCEPT-ONLY state commit = gate-4 LIVE FAIL (FR13_ACCEPT_ONLY_GATE4_FAIL_BIND, branch fr13-accept-only-wip):',
'  offline gates 1-3 passed but live B=4 captured accept/event 2.024->1.521 (deferred-publish vs next-step h0',
'  read ordering). Retry needs a live publish-before-next-h0-read probe. The HBM-tax-removal endpoint is the',
'  same as WY but loses byte-exactness, not speed.',
'- THE NON-REFUTED accept-side lever (wgb0yegin): L3 CONFIDENCE-GATED ROOT SIBLING - emit the (1,) root sibling',
'  ONLY when the root top-2 margin g < tau (62% of rejects are step-0); free scalar compare; raises accept/event',
'  (fewer forwards) WITHOUT the per-forward tax; fixes cat10\'s mispriced-unconditional-row. The deployed K=9',
'  caterpillar CANNOT beat the deployed K=9 LINEAR native at equal verify budget (FR13 endgame note) - but the',
'  accept-side lever + the per-forward-tax question is the whole ballgame for B=1 TPS.',
'',
'THE SPEED ARITHMETIC (the cost-gate must do this HONESTLY, no hand-rolled measured-fact claims): B=1 decode-TPS',
'~ (tokens committed per step) / (wall-time per step); tokens/step = accept/event; time/step = forwards/step x',
'time/forward. cat9 vs native: cat9 has accept/event ~3.18 vs 3.076 (slightly MORE committed/step) but ~1.63x',
'time/forward (the per-node GDN HBM tax) -> net cat9 ~0.6x native TPS unless the per-forward tax is cut OR accept',
'rises a lot. The cost-gate question: is there a PLAUSIBLY-CHEAP path to cat9 > native (e.g. L3 lifts accept',
'enough? a cheap per-forward-tax reduction that is NOT WY/accept-only? something else?), or is it structurally',
'blocked (tax structural + accept ceiling ~3.18) => STOP per feedback_speed_is_the_goal_cost_gate (do NOT build',
'expensive-but-correct; STOP if no plausibly-cheap correct path).',
'',
'YOUR JOB:',
'1. MEASUREMENT PROTOCOL: the EXACT clean B=1 cat9-vs-native decode-TPS + forwards/step + accept/event measurement',
'   (decode_seconds raw counter basis, FR10_METRICS off + FR12/13 diagnostics compiled out, per-request not',
'   aggregate, same 4 pinned prompts temp 0.0 seed 1313, the prelaunch host-mem protocol, BI pinned identical).',
'   Cite the existing scripts (the E5 native ref output/fr10_native_mtp5_same8_*, the speed harness). NO hand-',
'   rolled TPS/accept decomposition presented as a measured fact - only the raw decode_seconds counter.',
'2. COST-GATE (honest, the binding output): given the stale ~2.3x + WY parked + accept-only failed + the L3',
'   accept-side lever, is there a PLAUSIBLY-CHEAP path to cat9 B=1 TPS > native E5? Decompose the arithmetic with',
'   the CURRENT accept/event ~3.18 (label INFERRED vs to-be-MEASURED). Name the cheap candidates + their plausible',
'   reach; name the structural blockers. STOP-or-GO with the reasoning.',
'3. LEVER MENU: ranked by cost/feasibility (L3 conf-gated root sibling first; any cheap per-forward-tax reduction',
'   that is NOT WY/accept-only; drafter-quality wins). Mark each lossless-safe vs needs-a-lossless-gate.',
'4. If GO: the exact minimal GPU experiment sequence (measure the current gap FIRST, then the cheapest lever).',
'   If STOP: the honest statement (cat9 cannot cheaply beat native at B=1; lossless+at-parity-accept is the',
'   deliverable) + what a non-cheap path would cost.',
'',
'DELIVERABLE: FR13_SPEED_COSTGATE.md = the measurement protocol, the honest cost-gate (STOP-or-GO with the',
'arithmetic), the ranked lever menu, and the minimal GPU sequence (or the STOP statement). Distinguish MEASURED/',
'CODE-READ from INFERRED. NOT WY (parked), NOT accept-only-as-built (failed), NOT copy/dense/multispine/bonus.',
'Quote FR13_BUG_CLASS_PLAYBOOK + feedback_dont_handroll_speed (no inferred TPS decomposition as fact) + feedback_',
'speed_is_the_goal_cost_gate (STOP if no cheap path). research-before-deadend (measure before concluding STOP).',
].join('\n');

phase('CostGateDesign');
const D_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['measurementProtocol','costGateVerdict','leverMenu','minimalGpuSequence','committed','notes'],
  properties: {
    measurementProtocol: { type: 'string', description: 'the EXACT clean B=1 cat9-vs-native decode-TPS + forwards/step + accept/event protocol (decode_seconds basis, metrics off, per-request, pinned prompts, prelaunch host-mem, BI pinned), citing existing scripts/refs' },
    costGateVerdict: { type: 'string', description: 'STOP or GO + the honest arithmetic (accept/event ~3.18 vs 3.076, time/forward ~1.63x, forwards/step) with INFERRED-vs-MEASURE labels; the plausibly-cheap candidates + reach; the structural blockers' },
    leverMenu: { type: 'string', description: 'ranked levers by cost/feasibility (L3 conf-gated root sibling first; cheap non-WY/non-accept-only tax reductions; drafter wins), each lossless-safe vs needs-gate' },
    minimalGpuSequence: { type: 'string', description: 'if GO: the minimal GPU experiment sequence (measure current gap FIRST, then cheapest lever); if STOP: the honest statement + what a non-cheap path costs' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const d = await agent(
  CTX + '\n\nTASK (CostGateDesign, CPU read-only, no GPU). Read the prior speed verdicts + current pipeline + the '
  + 'L3 lever; produce the measurement protocol, the honest STOP-or-GO cost-gate, the ranked lever menu, the '
  + 'minimal GPU sequence. Write FR13_SPEED_COSTGATE.md, commit pathspec. Return the schema.',
  { label: 'speed-costgate-design', phase: 'CostGateDesign', schema: D_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','protocolSound','costGateHonest','leversReal','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    protocolSound: { type: 'string', description: 'is the measurement protocol sound (decode_seconds basis NOT TPS/accept, metrics off, per-request, prelaunch) per reference_fr10_speed_measurement_pitfalls?' },
    costGateHonest: { type: 'string', description: 'is the cost-gate honest - no hand-rolled TPS/accept decomposition presented as MEASURED (feedback_dont_handroll_speed), INFERRED labeled, STOP/GO grounded not premature?' },
    leversReal: { type: 'string', description: 'are the levers real + non-refuted (NOT WY/accept-only-as-built/copy/dense/bonus), correctly marked lossless-safe vs needs-gate?' },
    recommendation: { type: 'string', description: 'single: GO (measure the gap + try the cheapest lever) or STOP (no cheap path, lossless+parity is the deliverable). No close/pass-fail beyond the cost-gate.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(d) + '. Default holds=false if the protocol uses the BANNED '
  + 'TPS/accept basis or omits the prelaunch/metrics-off/per-request rigor; if the cost-gate presents a hand-'
  + 'rolled per-forward TPS decomposition as a MEASURED fact (it must be labeled INFERRED, the real number needs '
  + 'the GPU measurement) or concludes STOP/GO prematurely without the arithmetic; or if a lever re-proposes WY/'
  + 'accept-only-as-built/copy/dense/bonus. research-before-deadend (measure before STOP). No close/pass-fail.',
  { label: 'verify-speed-costgate', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { d, v };
