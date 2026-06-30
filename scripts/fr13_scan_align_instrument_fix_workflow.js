export const meta = {
  name: 'fr13-scan-align-instrument-fix',
  description: 'FIX the two BROKEN binding instruments from the inconclusive scan-align verify (w77rygxwf): (1) the int-view gate is VACUOUS - fr13_native_packed_decode_ref returns all-zeros OUTPUT (q=0 → b_o=0); fix it to compare the durable STATE h (norm 25.9, what the scan STATE-FEED actually is), with a genuinely-powered neg-control; (2) the same-boot no-spec oracle CRASHES (EagleProposer.positions) → investigate how the ORIGINAL 21 flips were measured (the working oracle mechanism, likely a SEPARATE non-spec server boot) and design the re-run. CPU read+fix+plan; the GPU re-run follows. Adversarial verify.',
  phases: [
    { title: 'Fix' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY (no GPU boot this workflow).',
'Read vLLM source DIRECTLY from the pinned image via `scripts/vllm_src.sh <relpath>` (0.19.2rc1.dev134),',
'NEVER a /tmp cache (grounding rule). Write fixes + FR13_SCAN_ALIGN_RERUN_PLAN.md; pathspec commits.',
'',
'CONTEXT (FR13_SCAN_ALIGN_VERIFY_INCONCLUSIVE_BIND.md, w77rygxwf holds=false): the committed scan-alignment',
'(5e56b7aa: FR13_SCAN_ALIGN l2norm/beta seams + _tree_gdn_recompute_kernel recompute-from-spine) ENGAGES',
'(recompute = different deterministic stream, det clean, accept 3.29) and default-OFF is byte-identical - but',
'whether it drives the lossless flips 21→3 is UNMEASURED because BOTH binding instruments broke (playbook #9):',
'',
'INSTRUMENT 1 - int-view gate VACUOUS: scripts/fr13_native_packed_decode_ref.py::native_packed_decode_per_path',
'returns ALL-ZEROS `out` (native_out norm=0.0). Root: the ref passes q=0 (q is irrelevant to the durable',
'state), and the packed-decode kernel output is `b_o = tl.sum(b_h * b_q)` → 0; meanwhile it updates the STATE',
'(b_h, norm 25.9) in-place. So the gate compared the WRONG tensor (the zeros output o) - every arm mismatches',
'by |serving_out| vs zeros, int-view trivially False, +0.5 neg-control vacuously powered.',
'INSTRUMENT 2 - e2e no-spec oracle CRASHES: per-request non_mtp AND naive_mtp both crash the EngineCore',
'(`EagleProposer has no attribute positions` in propose_tree) on the spec-configured server, so the same-boot',
'no-spec oracle is impossible. The ORIGINAL 21 flips (the bake B=1 HOLD) WERE measured somehow - find that',
'mechanism and reuse it.',
'',
'YOUR JOB:',
'PHASE 1 (Fix + Investigate, no GPU):',
'1. FIX INSTRUMENT 1 (the int-view gate / native-packed ref) to compare the durable GDN STATE h, NOT the',
'   zeros output. Via vllm_src.sh read the native packed-decode kernel (model_executor/layers/fla/ops/',
'   fused_recurrent.py: the packed_decode kernel + wrapper) and gdn_linear_attn decode path: the durable STATE',
'   is what the kernel writes in-place (initial_state/ssm_state, the [HV,V,K] recurrent matrix), NOT the `out`.',
'   Our scan exposes the per-node state via STORE_NODE_STATES (fr10_gdn_tree_kernel.py). Rebuild',
'   native_packed_decode_per_path so the A/B compares OUR scan STATE vs native-packed STATE (both non-zero),',
'   int-view (NEVER atol) + rel_err, and a POWERED neg-control that actually flips the comparison (perturb an',
'   INPUT that affects the STATE, verify the arm mismatches). Confirm the deployed-OFF arm now gives a REAL',
'   non-vacuous result vs native-packed STATE (likely nonzero = the genuine scan-vs-native gap, the carrier',
'   measurement we never got). Keep it observe-only (native = oracle, no served-path splice). CPU wiring test.',
'2. INVESTIGATE INSTRUMENT 2 (the working no-spec oracle): how was the ORIGINAL 21 flips measured? Read',
'   fr13_gold_margin_probe.py (the binding instrument), the bake B=1 HOLD artifacts/launcher (FR13_BAKE_B1_',
'   HOLD_BIND, fr13_launch_locked.sh, any oracle/non-spec launcher), and how the no-spec oracle stream is',
'   produced (separate non-spec server boot? a teacher-force endpoint? a banked oracle?). The oracle = no-spec',
'   greedy decode of the SAME model (NOT prefill, NOT the spec drafter). Document the EXACT reproducible',
'   mechanism the re-run must use to get flips_before(OFF) and flips_after(recompute) US-vs-its-own-no-spec-',
'   oracle on THIS-boot streams. If it needs a 2nd (non-spec) server boot, specify the launch.',
'3. WRITE FR13_SCAN_ALIGN_RERUN_PLAN.md: the fixed-gate design + the exact oracle mechanism + the re-run',
'   sequence (int-view STATE gate for OFF/body/recompute; then OFF + recompute spec streams + the no-spec',
'   oracle → gold_margin flips). Commit pathspec (the gate/ref fix + the plan + the test). FAIL ready=false',
'   with the blocker if the native STATE cannot be cleanly extracted for the A/B or the oracle mechanism is',
'   unrecoverable.',
'',
'Reward-hacks BANNED (native = A/B oracle only; do not splice native into the served path). Quote',
'FR13_BUG_CLASS_PLAYBOOK rows (#9 silent/vacuous, #10 codegen-identity). Be SKEPTICAL - this exists because',
'the last verify was vacuous; the fixed gate must be PROVABLY non-vacuous (the neg-control must really flip).'
].join('\n');

phase('Fix');
const F_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['gateStateFix','offArmRealResult','negControlNowPowered','oracleMechanism','rerunPlan','cpuTest','committed','ready','notes'],
  properties: {
    gateStateFix: { type: 'string', description: 'how native_packed_decode_per_path now compares the durable STATE h (not zeros output o); how native STATE is extracted; int-view+rel_err' },
    offArmRealResult: { type: 'string', description: 'the deployed-OFF arm scan-STATE vs native-packed-STATE - is it now a REAL non-vacuous number (the genuine scan-vs-native gap)?' },
    negControlNowPowered: { type: ['boolean','null'], description: 'does the neg-control genuinely flip the STATE comparison now (provably non-vacuous)?' },
    oracleMechanism: { type: 'string', description: 'the EXACT reproducible mechanism the original 21 used to get the no-spec oracle (separate non-spec boot? endpoint? banked) - with the launch/command' },
    rerunPlan: { type: 'string', description: 'FR13_SCAN_ALIGN_RERUN_PLAN.md: the int-view STATE gate + oracle + the re-run sequence for OFF + recompute flips' },
    cpuTest: { type: 'string' },
    committed: { type: 'string' },
    ready: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const f = await agent(
  CTX + '\n\nTASK (Fix + Investigate, no GPU). Do PHASE 1. Commit pathspec. Return the schema. ready=false with '
  + 'the blocker if the native STATE cannot be cleanly extracted or the oracle mechanism is unrecoverable.',
  { label: 'fix-scan-align-instruments', phase: 'Fix', schema: F_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','gateNonVacuous','stateNotOutput','oracleGrounded','rerunReady','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    gateNonVacuous: { type: 'string', description: 'is the fixed gate PROVABLY non-vacuous (neg-control really flips; not comparing vs zeros)?' },
    stateNotOutput: { type: 'string', description: 'does it compare the durable STATE h (grounded in the real native kernel), not the q=0 output?' },
    oracleGrounded: { type: 'string', description: 'is the no-spec oracle mechanism the real one the original 21 used (grounded, reproducible)?' },
    rerunReady: { type: ['boolean','null'], description: 'is the re-run plan executable (instruments fixed, oracle specified)?' },
    recommendation: { type: 'string', description: 'single recommendation for the GPU re-run. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(f) + '. Default holds=false if the fixed gate is still '
  + 'comparing the output not the STATE, the neg-control is not provably powered, or the oracle mechanism is '
  + 'not grounded in how the original 21 was actually measured (spot-check via the bake artifacts). The last '
  + 'verify was vacuous - this fix must be PROVABLY non-vacuous. No close/pass-fail; no reward-hack.',
  { label: 'verify-instrument-fix', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { f, v };
