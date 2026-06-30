export const meta = {
  name: 'fr13-baproj-implement-test',
  description: 'IMPLEMENT the in_proj_ba pad-to-fixed-M (M-invariant a/b) for the DEPLOYED image via the patcher (default-OFF gated by LUMO_FB_KERNEL_ROWS), CPU-verify lossless + default-path byte-identical, then boot cat9 + flag and measure the flip count. cat9 22 -> ~5 confirms H1 + fixes the +17 leaf co-residency. Conditional flag-OFF control; adversarial verify.',
  phases: [
    { title: 'Implement' },
    { title: 'GateON' },
    { title: 'GateOFF' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene:',
'source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps empty. Teardown trap: docker',
'rm -f the container + recover_host_memory.',
'',
'GOAL: fix the +17 leaf width co-residency = the bf16 in_proj_ba GEMM being M-keyed (FR13_WIDTH_CARRIER_INPROJ_BA_BIND.md,',
'verify holds=True; in_proj_a/b in checkpoint modules_to_not_convert => bf16 cuBLAS; at M=tree_n=10 vs spine M=5',
'cuBLASLt picks a different Split-K => ~1 bf16-ULP shift in a/b on the spine => spine flips). Make in_proj_ba',
'M-invariant by padding to a FIXED, tree_n-independent M (constant row group), compute ONE GEMM, scatter real',
'rows back, discard pads => cuBLASLt pins to ONE shape => M-invariant a/b. Lossless-by-construction (a GEMM row',
'= W @ hidden[row], independent per row; zero pads contribute nothing). Batch-invariance is directive-authorized',
'(#42960). Reward-hacks BANNED (no copy/dense/splice/reroute of the spine a/b from a clean run).',
'',
'THE BLOCKER TO SOLVE (task wu7mj0vls): the LUMO_FB pad-block code exists ONLY in the stale snapshot',
'/tmp/vllm_live_019/gdn_linear_attn.py (1420 lines, in_proj_ba block at 553-601, out_proj at 674-723) which is',
'a DIFFERENT/newer vLLM than the DEPLOYED image (vllm/vllm-openai cu130-nightly, gdn_linear_attn.py = 1211',
'lines, NO pad block: image line ~545 ba,_ = self.in_proj_ba(hidden_states)). The launcher is ALREADY wired',
'(scripts/fr13_launch_forked_fa2_tree_server.sh commit e376aec3: LUMO_FB_KERNEL_ROWS default empty/OFF,',
'LUMO_FB_PROJ_PAD_ROWS default 16, both -e passed). So the env is wired but INERT until the code path exists',
'in the image. YOU must INSERT the pad block into the deployed image via the in-container editor',
'scripts/fr10_phase4_patch_vllm_tree_gdn.py (the GDN_LINEAR edit, read_text/write_text on gdn_linear_attn.py).',
'',
'IMPLEMENT REQUIREMENTS:',
'1. Extract gdn_linear_attn.py from the DEPLOYED image (docker create + docker cp, no GPU) and find the exact',
'   anchors: ba,_ = self.in_proj_ba(hidden_states) and the out_proj line. Read the stale-snapshot block',
'   (/tmp/vllm_live_019 :553-601, :674-723) for the algorithm but PORT it to the IMAGE API (do NOT lift',
'   verbatim - the snapshot has divergent imports/params).',
'2. VERIFY the image runtime exposes what the block needs: get_forward_context().attn_metadata with',
'   spec_query_start_loc + num_spec_decodes (+ num_prefills/num_decodes) on the image TreeAttentionMetadata.',
'   If an attr is missing, ADAPT (derive spans from spec_state_indices / the tree layout) or SAY SO as a blocker.',
'3. GATE: the whole pad path activates ONLY when os.environ.get("LUMO_FB_KERNEL_ROWS")=="1" AND',
'   num_spec_decodes>1 (verify path). Pad-rows = max(num_spec_decodes, int(LUMO_FB_PROJ_PAD_ROWS,16)). Wrap in',
'   try/except: stock fallback. Apply the SAME to in_proj_ba AND out_proj (out_proj is post-scan but pad it too',
'   for consistency; out_proj is fp8 so likely a no-op, harmless).',
'4. DEFAULT-OFF BYTE-IDENTICAL: with LUMO_FB_KERNEL_ROWS unset/empty the emitted gdn_linear_attn.py served',
'   path must be BYTE-IDENTICAL to stock (the locked cat9 [6,6,4,6] fingerprint unaffected). Prove it: the',
'   patcher addition is purely additive + flag-guarded; the gate-OFF reduced form == stock. CPU tests.',
'5. LOSSLESS-BY-CONSTRUCTION CPU CHECK: on a synthetic hidden_states, the padded in_proj_ba real-row output ==',
'   the unpadded output bit-for-bit (per-row GEMM; zero pads dropped). If NOT bit-identical, the pad layout is',
'   wrong - fix it (this is the lossless gate, not a tolerance).',
'Commit pathspec (only the patcher; do NOT sweep the untracked fr13_*_workflow.js).',
'',
'Bars for the GPU test: cat9 chunked baseline=22 (banked q3_tree_classify), pure-spine floor=5, native=3.',
'Instrument fr13_oracle_stream_teacher_force.py thr 1.0 prompts_swe4 (each-vs-own oracle). Gates: engagement',
'tok/draft==9 (class-9), within_boot_det [T,T,T,T] (class-8), accept/event must keep the leaf edge (class-12).',
'Quote FR13_BUG_CLASS_PLAYBOOK.md rows.'
].join('\n');

phase('Implement');
const IMPL_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['imageAnchors','apiPorted','gate','losslessCPUCheck','defaultPathProof','filesWritten','ready','notes'],
  properties: {
    imageAnchors: { type: 'string', description: 'the exact deployed-image gdn_linear_attn.py anchor lines for in_proj_ba + out_proj' },
    apiPorted: { type: 'string', description: 'how the block was ported to the image API (spec_query_start_loc/num_spec_decodes present? or adapted how?)' },
    gate: { type: 'string', description: 'the LUMO_FB_KERNEL_ROWS==1 + num_spec_decodes>1 gating + pad-rows sizing + try/except fallback' },
    losslessCPUCheck: { type: 'string', description: 'the synthetic CPU check that padded real-row output == unpadded bit-for-bit (PASS/FAIL + max_abs)' },
    defaultPathProof: { type: 'string', description: 'gate-OFF emitted gdn_linear_attn.py byte-identical to stock (the locked path unaffected)' },
    filesWritten: { type: 'string' },
    ready: { type: 'boolean', description: 'true only if lossless CPU check PASS + default-path byte-identical + API ported (no blocker)' },
    notes: { type: 'string' },
  },
};
const impl = await agent(
  CTX + '\n\nTASK (Implement, no GPU). Do requirements 1-5. Commit pathspec. Return the schema. If a required '
  + 'image API attr is missing and cannot be cleanly derived, set ready=false with the blocker (do NOT fake it).',
  { label: 'implement-baproj-pad', phase: 'Implement', schema: IMPL_SCHEMA, model: 'opus' }
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
let gateOn = null;
if (impl && impl.ready) {
  gateOn = await agent(
    CTX + '\n\nTASK (GateON, GPU - the ONLY boot). Implement: ' + JSON.stringify(impl) + '. Hygiene + boot cat9 '
    + 'TREE=[(0,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0),(0,1),(0,0,1),(0,0,0,1),(0,0,0,0,1)] with '
    + 'LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16. Detached+poll. Engagement tok/draft==9 + within_boot_det '
    + '[T,T,T,T] (FAIL LOUD else). Flip count vs the cat9 oracle + accept/event. KEY NUMBER = '
    + 'total_clear_margin_flips (22 baseline / 5 pure-spine / 3 native). Teardown + recover; never leak. '
    + 'flag="BA_PROJ_BI_ON".',
    { label: 'gate:baproj-on', phase: 'GateON', schema: GATE_SCHEMA, model: 'opus' }
  );
} else {
  log('Implement not ready (blocker) -> SKIP GPU boot (would be vacuous). Verdict will report the blocker.');
}

phase('GateOFF');
let gateOff = null;
const onFlips = gateOn && typeof gateOn.total_clear_margin_flips === 'number' ? gateOn.total_clear_margin_flips : null;
if (gateOn && gateOn.ok && onFlips !== null && onFlips > 8) {
  log('BA_PROJ_BI_ON = ' + onFlips + ' flips (>8, not dramatic) -> flag-OFF control to rule out cross-boot confound');
  gateOff = await agent(
    CTX + '\n\nTASK (GateOFF, GPU control). BA_PROJ_BI_ON gave ' + onFlips + ' (ambiguous). Boot the SAME cat9 with '
    + 'LUMO_FB_KERNEL_ROWS unset (default OFF) and measure the flip count under identical conditions (the banked 22 '
    + 'is a different boot). Same gates. flag="BA_PROJ_BI_OFF". Teardown + recover.',
    { label: 'gate:baproj-off', phase: 'GateOFF', schema: GATE_SCHEMA, model: 'opus' }
  );
} else {
  log('GateON flips=' + onFlips + ' (<=8 dramatic / failed / skipped) -> banked cat9=22 sufficient control, SKIP flag-OFF');
}

phase('Verdict');
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','h1Confirmed','flipsDropped','losslessByConstruction','acceptPreserved','isFix','nextAction','issues'],
  properties: {
    holds: { type: 'boolean' },
    h1Confirmed: { type: ['boolean','null'] },
    flipsDropped: { type: 'string', description: 'cat9+BA_PROJ_BI flips vs 22 baseline (and flag-OFF control if run)' },
    losslessByConstruction: { type: 'string', description: 'CPU lossless check + det [T,T,T,T]; does it move TOWARD the no-spec decode (fewer flips)?' },
    acceptPreserved: { type: 'string' },
    isFix: { type: ['boolean','null'], description: 'is the ba-proj pad a deployable fix for the +17 (lossless + accept-preserving + flip-reducing)?' },
    nextAction: { type: 'string', description: 'single next action (bake with verification, or extended A/B if blocked/ambiguous). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const verdict = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: Implement ' + JSON.stringify(impl) + ' GateON ' + JSON.stringify(gateOn)
  + ' GateOFF ' + JSON.stringify(gateOff) + '. Default holds=false if Implement blocked or gates failed. '
  + 'Conclude H1 confirmed (cat9+BA_PROJ_BI ~= 5)? lossless-by-construction + accept-preserving? deployable fix? '
  + 'Red-team cross-boot/trajectory confound on any non-dramatic drop. No close/pass-fail; no reward-hack.',
  { label: 'verdict', phase: 'Verdict', schema: VERDICT_SCHEMA, agentType: 'general-purpose' }
);

return { impl, gateOn, gateOff, verdict };
