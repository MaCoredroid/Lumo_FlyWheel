export const meta = {
  name: 'fr13-baproj-batchinvariant-fix',
  description: 'Wire + test the LUMO_FB_BATCH_INVARIANT_BA_PROJ targeted fix: boot cat9 with the bf16 in_proj_ba GEMM padded to a fixed M (M-invariant a/b) and measure the flip count. cat9 22 to ~5 confirms H1 (the +17 leaf co-residency = bf16 in_proj_ba M-variance) AND fixes it lossless-by-construction; stays high => flag insufficient/H1 wrong. Conditional flag-OFF control if ambiguous. Adversarial verify.',
  phases: [
    { title: 'Build' },
    { title: 'GateON' },
    { title: 'GateOFF' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene every',
'boot: source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps empty. Teardown trap each',
'boot: docker rm -f the container + recover_host_memory.',
'',
'LOCALIZED CARRIER (FR13_WIDTH_CARRIER_INPROJ_BA_BIND.md, verify holds=True): the +17 leaf width co-residency',
'(cat9 22 vs pure-spine 5; the leaves perturbing the SPINE) = the bf16 in_proj_ba GEMM is M-keyed. in_proj_a',
'and in_proj_b are in the checkpoint modules_to_not_convert => bf16 cuBLAS/cuBLASLt (NOT fp8); at M=tree_n=10',
'vs spine M=5, cuBLASLt picks a different kernel/Split-K => ~1 bf16-ULP shift in a/b (= the scan raw_a/raw_b',
'-> gate b_g/b_beta) on the spine row => the proven-bit-exact scan consumes M-variant inputs => spine flips.',
'ALL OTHER candidates refuted (fp8 GEMMs M-invariant; conv row-M-invariant; ancestry-mask no leaf-bleed',
'FR13_BV_GEOMETRY 0.0 at N_PAD 1 and 16; chunk/state-feed null).',
'',
'THE FIX (real, NOT a reward-hack; rewardHackCheck passed; batch-invariance is directive-authorized #42960):',
'LUMO_FB_BATCH_INVARIANT_BA_PROJ (live container vllm/model_executor/layers/mamba/gdn_linear_attn.py:553-601,',
'gated by LUMO_FB_KERNEL_ROWS=1 + a fixed pad-rows env) pads the in_proj_ba GEMM to a FIXED, tree_n-independent',
'M (constant row group), computes, SCATTERS the real rows back, discards pads => pins cuBLASLt to ONE shape =>',
'M-invariant a/b. Mathematically the SAME per-row computation (GEMM row = W @ hidden[row], independent per row;',
'zero pad-rows contribute nothing) => lossless-by-construction. It is OFF in the locked launcher',
'(scripts/fr13_launch_locked.sh exports BATCH_INVARIANT=0; the forked launcher',
'scripts/fr13_launch_forked_fa2_tree_server.sh does NOT pass LUMO_FB_BATCH_INVARIANT_BA_PROJ / LUMO_FB_KERNEL_ROWS).',
'RECONCILES the prior cat9+BI=34 (counterproductive): FULL vLLM BATCH_INVARIANT takes the GB10 REDUCED override',
'branch (perturbs fp8/scan) => worse; the TARGETED ba-proj pad ONLY fixes the carrier.',
'',
'THE TEST (decisive localization + fix in one): boot cat9 with the ba-proj batch-invariant flag ON and measure',
'the flip count (fr13_oracle_stream_teacher_force.py, each-vs-own no-spec oracle, thr 1.0, same prompts_swe4 =',
'the comparable lossless metric). Bars: cat9 chunked baseline = 22 (banked q3_tree_classify); pure-spine floor',
'= 5; native = 3. cat9+BA_PROJ_BI -> ~5 (== pure-spine) => H1 CONFIRMED + the +17 leaf co-residency FIXED',
'(leaves now give accept WITHOUT perturbing the spine). Stays ~22 => flag insufficient or H1 wrong. ALSO report',
'accept/event (the leaves must still give the accept edge) + within_boot_det [T,T,T,T] (class-8) + engagement',
'tok/draft==9 (class-9). Reward-hacks BANNED. Quote FR13_BUG_CLASS_PLAYBOOK.md rows.'
].join('\n');

phase('Build');
const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['flagSemantics','envWired','padValue','defaultPathProof','filesWritten','notes'],
  properties: {
    flagSemantics: { type: 'string', description: 'the exact env vars + gating read from the live gdn_linear_attn.py:553-601 (LUMO_FB_BATCH_INVARIANT_BA_PROJ / LUMO_FB_KERNEL_ROWS / pad-rows) and what they do' },
    envWired: { type: 'string', description: 'how the forked launcher now passes the ba-proj flag (additive -e passthrough, default-OFF preserved)' },
    padValue: { type: 'string', description: 'the fixed pad-rows chosen (>= max tree_n=10, e.g. 16) + why' },
    defaultPathProof: { type: 'string', description: 'flag-OFF byte-identical to the locked launcher (additive, default unset)' },
    filesWritten: { type: 'string' },
    notes: { type: 'string' },
  },
};
const build = await agent(
  CTX + '\n\nTASK (Build, no GPU). Read the live gdn_linear_attn.py LUMO_FB_BATCH_INVARIANT_BA_PROJ flag '
  + 'semantics + wire the forked launcher (scripts/fr13_launch_forked_fa2_tree_server.sh) to pass it (additive, '
  + 'default-OFF; pick a fixed pad-rows >= 16). Prove the flag-OFF default path is byte-identical to the locked '
  + 'launcher. Commit pathspec. Return the schema. If the flag needs more than an env passthrough (e.g. a code '
  + 'path not in the live image), SAY SO with the blocker.',
  { label: 'build-baproj-wire', phase: 'Build', schema: BUILD_SCHEMA, model: 'opus' }
);

phase('GateON');
const GATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['name','flag','total_clear_margin_flips','per_prompt','accept_per_event','within_boot_det','tok_per_draft','engaged','ok','notes'],
  properties: {
    name:{type:'string'}, flag:{type:'string'},
    total_clear_margin_flips:{type:['integer','null']}, per_prompt:{type:'string'},
    accept_per_event:{type:['number','null']}, within_boot_det:{type:'string'},
    tok_per_draft:{type:['number','null']}, engaged:{type:'boolean'}, ok:{type:'boolean'}, notes:{type:'string'},
  },
};
const gateOn = await agent(
  CTX + '\n\nTASK (GateON, GPU - the ONLY boot now). Build: ' + JSON.stringify(build) + '. Hygiene + boot cat9 '
  + 'TREE=[(0,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0),(0,1),(0,0,1),(0,0,0,1),(0,0,0,0,1)] with '
  + 'LUMO_FB_BATCH_INVARIANT_BA_PROJ ON (+ LUMO_FB_KERNEL_ROWS + the pad). Detached+poll if slow. Engagement gate '
  + 'tok/draft==9 + within_boot_det [T,T,T,T] (FAIL LOUD else). Flip count vs the cat9 oracle + accept/event. '
  + 'THE KEY NUMBER = total_clear_margin_flips (22 baseline / 5 pure-spine / 3 native). Teardown + recover; never '
  + 'leak. Report flag="BA_PROJ_BI_ON".',
  { label: 'gate:baproj-on', phase: 'GateON', schema: GATE_SCHEMA, model: 'opus' }
);

phase('GateOFF');
let gateOff = null;
const onFlips = gateOn && typeof gateOn.total_clear_margin_flips === 'number' ? gateOn.total_clear_margin_flips : null;
if (gateOn && gateOn.ok && onFlips !== null && onFlips > 8) {
  log('BA_PROJ_BI_ON = ' + onFlips + ' flips (>8, not a dramatic drop) -> boot flag-OFF control to rule out cross-boot confound');
  gateOff = await agent(
    CTX + '\n\nTASK (GateOFF, GPU control). BA_PROJ_BI_ON gave ' + onFlips + ' flips (ambiguous). Boot the SAME cat9 '
    + 'with LUMO_FB_BATCH_INVARIANT_BA_PROJ OFF (the locked default) and measure the flip count under identical '
    + 'conditions, to rule out a cross-boot autotune confound (the banked 22 is a different boot). Same gates. '
    + 'Report flag="BA_PROJ_BI_OFF". Teardown + recover.',
    { label: 'gate:baproj-off', phase: 'GateOFF', schema: GATE_SCHEMA, model: 'opus' }
  );
} else {
  log('BA_PROJ_BI_ON = ' + onFlips + ' flips (<=8 dramatic drop, or gate failed) -> banked cat9=22 is sufficient control, SKIP flag-OFF');
}

phase('Verdict');
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','h1Confirmed','flipsDropped','losslessByConstruction','acceptPreserved','isFix','nextAction','issues'],
  properties: {
    holds: { type: 'boolean' },
    h1Confirmed: { type: ['boolean','null'], description: 'did cat9+BA_PROJ_BI drop to ~pure-spine (~5)? => H1 (bf16 in_proj_ba) is the +17 carrier' },
    flipsDropped: { type: 'string', description: 'cat9+BA_PROJ_BI flips vs 22 baseline (and the flag-OFF control if run) - the +17 removed?' },
    losslessByConstruction: { type: 'string', description: 'is the fix lossless (det [T,T,T,T] + per-row math identical) - does it move the spine TOWARD the no-spec decode (fewer flips) not away?' },
    acceptPreserved: { type: 'string', description: 'did accept/event hold (leaves still give the accept edge)?' },
    isFix: { type: ['boolean','null'], description: 'is LUMO_FB_BATCH_INVARIANT_BA_PROJ a deployable fix for the +17 (lossless + accept-preserving + flip-reducing)?' },
    nextAction: { type: 'string', description: 'single next action (e.g. bake the flag with verification, or the extended A/B if ambiguous). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const verdict = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: Build ' + JSON.stringify(build) + ' GateON ' + JSON.stringify(gateOn)
  + ' GateOFF ' + JSON.stringify(gateOff) + '. Default holds=false if gates failed. Conclude: H1 confirmed '
  + '(cat9+BA_PROJ_BI ~= pure-spine 5)? Is it lossless-by-construction (fewer flips = closer to the no-spec '
  + 'decode, det clean) and accept-preserving? Is it a deployable fix for the +17? Red-team a cross-boot/'
  + 'trajectory confound on any non-dramatic drop. No close/pass-fail; no reward-hack.',
  { label: 'verdict', phase: 'Verdict', schema: VERDICT_SCHEMA, agentType: 'general-purpose' }
);

return { build, gateOn, gateOff, verdict };
